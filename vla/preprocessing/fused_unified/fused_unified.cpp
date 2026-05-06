// fused_unified.cpp — im2col + GEMM conv2d patch-embedding on AMD NPU.
//
// Key overhead reduction vs preprocessing_fused_bf16.py:
//   Python calls 320 separate Allo NPU invocations (~0.05 ms each = 16 ms overhead).
//   This C++ implementation opens exactly 2 hw_contexts for the entire inference:
//     - Context 1: im2col  — 128 dispatches (32 patch rows × 4 quarter-cols)
//     - Context 2: GEMM    — 192 dispatches (32 M-tiles × 6 N-tiles)
//
// AMD NPU constraint: only ONE hw_context may be active at a time.
// Instruction BOs (XCL_BO_FLAGS_CACHEABLE) must be allocated inside a live context.
// Data BOs (XRT_BO_FLAGS_HOST_ONLY) are allocated once in the constructor.

#include "fused_unified.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using clk = std::chrono::high_resolution_clock;

static double elapsed_us(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// FuKernelSpec::preload
// ---------------------------------------------------------------------------
void FuKernelSpec::preload(xrt::device       &dev,
                            const std::string &xclbin_path,
                            const std::string &instr_path)
{
    xclbin_obj = xrt::xclbin(xclbin_path);
    dev.register_xclbin(xclbin_obj);

    auto xks = xclbin_obj.get_kernels();
    auto xk  = *std::find_if(xks.begin(), xks.end(),
        [](xrt::xclbin::kernel &k){ return k.get_name().rfind("MLIR_AIE", 0) == 0; });
    kernel_name = xk.get_name();

    std::ifstream f(instr_path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open instruction file: " + instr_path);
    f.seekg(0, std::ios::end);
    std::streamsize nb = f.tellg();
    f.seekg(0, std::ios::beg);
    if (nb % 4 != 0)
        throw std::runtime_error("Instruction file size not a multiple of 4: " + instr_path);
    instr.resize(nb / 4);
    f.read(reinterpret_cast<char *>(instr.data()), nb);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
FusedPreprocessor::FusedPreprocessor(
    unsigned int        device_index,
    const std::string & im2col_xclbin,
    const std::string & im2col_instr,
    const std::string & gemm_xclbin,
    const std::string & gemm_instr,
    int verbosity)
    : verbosity_(verbosity)
    , device_(device_index)
    , patches_(FU_PATCHES_ELEMS)
    , image_strips_((size_t)FU_PH * FU_IMG2D * FU_PIX_LEN)
{
    // Step 1: preload xclbins (no hw_context needed)
    im2col_spec_.preload(device_, im2col_xclbin, im2col_instr);
    gemm_spec_.preload(device_, gemm_xclbin, gemm_instr);
    if (verbosity_ >= 1)
        std::cout << "[FusedPreprocessor] Kernel specs loaded.\n";

    // Step 2: discover data group IDs via a temporary context
    int g_im2col_in, g_im2col_out, g_gemm_a, g_gemm_b, g_gemm_c, g_trace;
    {
        xrt::hw_context tmp(device_, im2col_spec_.xclbin_obj.get_uuid());
        xrt::kernel     k(tmp, im2col_spec_.kernel_name);
        g_im2col_in  = k.group_id(3);   // first  data slot
        g_im2col_out = k.group_id(4);   // second data slot
        g_trace      = k.group_id(5);   // trace slot (2-arg kernel: no slot 5 data)
        if (verbosity_ >= 1)
            std::cout << "[FusedPreprocessor] im2col group IDs: in=" << g_im2col_in
                      << " out=" << g_im2col_out << "\n";
    }
    {
        xrt::hw_context tmp(device_, gemm_spec_.xclbin_obj.get_uuid());
        xrt::kernel     k(tmp, gemm_spec_.kernel_name);
        // Allo GEMM slot order (from generated test.cpp): A=slot3, C_out=slot4, B=slot5
        g_gemm_a = k.group_id(3);
        g_gemm_c = k.group_id(4);   // output C is slot 4
        g_gemm_b = k.group_id(5);   // input B is slot 5
        if (verbosity_ >= 1)
            std::cout << "[FusedPreprocessor] GEMM group IDs: A=" << g_gemm_a
                      << " C_out=" << g_gemm_c << " B=" << g_gemm_b << "\n";
    }

    // Step 3: allocate persistent HOST_ONLY data BOs
    // GEMM BO sizes from generated test.cpp: A=24576, C_out=4096, B=98304
    bo_im2col_in_  = xrt::bo(device_,
        FU_IM2COL_IN_ELEMS * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g_im2col_in);
    bo_im2col_out_ = xrt::bo(device_,
        FU_IM2COL_OUT_ELEMS * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g_im2col_out);
    bo_gemm_a_     = xrt::bo(device_,
        FU_GEMM_A_ELEMS * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g_gemm_a);
    bo_gemm_c_     = xrt::bo(device_,
        FU_GEMM_C_ELEMS * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g_gemm_c);
    bo_gemm_b_     = xrt::bo(device_,
        FU_GEMM_B_ELEMS * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g_gemm_b);

    if (verbosity_ >= 1)
        std::cout << "[FusedPreprocessor] Data BOs allocated.\n";
}

// ---------------------------------------------------------------------------
// Helper: allocate an instruction BO inside an already-open hw_context.
// Must be called while hw_context is alive (XCL_BO_FLAGS_CACHEABLE requires it).
// ---------------------------------------------------------------------------
static xrt::bo make_instr_bo(xrt::device &dev, xrt::kernel &k,
                              const std::vector<uint32_t> &instr)
{
    size_t nb = instr.size() * sizeof(uint32_t);
    xrt::bo bo(dev, nb, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    memcpy(bo.map<void *>(), instr.data(), nb);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

// ---------------------------------------------------------------------------
// exec_im2col — one im2col dispatch.
// ph: patch row [0,PH), qt: quarter column [0,4)
// Reads from image_strips_[ph, :, qt*PIX_QT:(qt+1)*PIX_QT].
// Writes to patches_[(ph*PW + qt*PW_QT)*K_FULL ... +PW_QT*K_FULL].
// ---------------------------------------------------------------------------
void FusedPreprocessor::exec_im2col(
    xrt::kernel &k, xrt::bo &bo_instr, int instr_size,
    int ph, int qt)
{
    // image_strips_ layout: [PH=32, IMG2D=48, PIX_LEN=512] row-major
    // quarter-column slice: image_strips_[ph, :, qt*PIX_QT : (qt+1)*PIX_QT]
    // The full row is contiguous (stride=PIX_LEN), the slice is NOT contiguous.
    // We must gather the IMG2D rows into a contiguous [IMG2D, PIX_QT] buffer.
    auto *in_map = bo_im2col_in_.map<std::bfloat16_t *>();
    const std::bfloat16_t *strip_base =
        image_strips_.data() + (size_t)ph * FU_IMG2D * FU_PIX_LEN;
    int col_start = qt * FU_PIX_QT;
    for (int row = 0; row < FU_IMG2D; ++row)
        memcpy(in_map + row * FU_PIX_QT,
               strip_base + row * FU_PIX_LEN + col_start,
               FU_PIX_QT * sizeof(std::bfloat16_t));

    bo_im2col_in_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = k(3u, bo_instr, instr_size, bo_im2col_in_, bo_im2col_out_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("im2col kernel did not complete");

    bo_im2col_out_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Write patches_qt [PW_QT, K_FULL] into patches_[pat_start : pat_start+PW_QT, :]
    // patches_ is [SEQ, K_FULL] = contiguous, so each patch row is K_FULL elements.
    int pat_start = ph * FU_PW + qt * FU_PW_QT;
    memcpy(patches_.data() + (size_t)pat_start * FU_K_FULL,
           bo_im2col_out_.map<std::bfloat16_t *>(),
           FU_IM2COL_OUT_ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// exec_gemm — one GEMM dispatch.
// m: M-tile index [0, SEQ/M_TILE), n: N-tile index [0, EMBD_DIM/N_TILE)
// A = patches_[m*M_TILE:(m+1)*M_TILE, :] — contiguous [M_TILE, K_FULL]
// B = kernel_B_tiles[n * K_FULL * N_TILE] — contiguous [K_FULL, N_TILE]
// C = output[m*M_TILE:(m+1)*M_TILE, n*N_TILE:(n+1)*N_TILE] — non-contiguous slice
// ---------------------------------------------------------------------------
void FusedPreprocessor::exec_gemm(
    xrt::kernel &k, xrt::bo &bo_instr, int instr_size,
    const std::bfloat16_t *kernel_B_tiles,
    std::bfloat16_t *output,
    int m, int n)
{
    // A: patches_[m*M_TILE:(m+1)*M_TILE, :] — contiguous in patches_
    memcpy(bo_gemm_a_.map<std::bfloat16_t *>(),
           patches_.data() + (size_t)m * FU_M_TILE * FU_K_FULL,
           FU_GEMM_A_ELEMS * sizeof(std::bfloat16_t));
    bo_gemm_a_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // B: kernel_B_tiles[n] — [K_FULL, N_TILE] contiguous
    memcpy(bo_gemm_b_.map<std::bfloat16_t *>(),
           kernel_B_tiles + (size_t)n * FU_GEMM_B_ELEMS,
           FU_GEMM_B_ELEMS * sizeof(std::bfloat16_t));
    bo_gemm_b_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Slot order matches Allo-generated test.cpp: (A=slot3, C_out=slot4, B=slot5)
    auto run = k(3u, bo_instr, instr_size, bo_gemm_a_, bo_gemm_c_, bo_gemm_b_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("GEMM kernel did not complete");

    bo_gemm_c_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Scatter C_tile [M_TILE, N_TILE] into output[m*M_TILE:(m+1)*M_TILE, n*N_TILE:(n+1)*N_TILE]
    // output is row-major [SEQ, EMBD_DIM], stride=(EMBD_DIM, 1) — tile rows are non-contiguous.
    const auto *c_map = bo_gemm_c_.map<const std::bfloat16_t *>();
    int row_off = m * FU_M_TILE;
    int col_off = n * FU_N_TILE;
    for (int r = 0; r < FU_M_TILE; ++r)
        memcpy(output + ((size_t)(row_off + r) * FU_EMBD_DIM + col_off),
               c_map + r * FU_N_TILE,
               FU_N_TILE * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// run() — full pipeline. 2 hw_context opens total.
// ---------------------------------------------------------------------------
void FusedPreprocessor::run(
    const std::bfloat16_t * image,
    const std::bfloat16_t * kernel_B_tiles,
    std::bfloat16_t       * output)
{
    im2col_us_ = 0.0;
    gemm_us_   = 0.0;

    // ------------------------------------------------------------------
    // CPU: rearrange image [C, PIX_LEN, PIX_LEN] → image_strips_ [PH, IMG2D, PIX_LEN]
    // image_strips_[ph, c*KH + ky, col] = image[c, ph*KH + ky, col]
    // Equivalent to Python:
    //   image.reshape(C, PH, KH, PIX_LEN).transpose(1,0,2,3).reshape(PH, IMG2D, PIX_LEN)
    // ------------------------------------------------------------------
    for (int ph = 0; ph < FU_PH; ++ph) {
        for (int c = 0; c < FU_CHANNELS; ++c) {
            for (int ky = 0; ky < FU_KH; ++ky) {
                int row_in_strip = c * FU_KH + ky;
                int src_row      = ph * FU_KH + ky;
                memcpy(image_strips_.data() + (size_t)(ph * FU_IMG2D + row_in_strip) * FU_PIX_LEN,
                       image + ((size_t)c * FU_PIX_LEN + src_row) * FU_PIX_LEN,
                       FU_PIX_LEN * sizeof(std::bfloat16_t));
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 1 — im2col: ONE hw_context, 128 dispatches
    //   (PH=32 patch rows × 4 quarter-columns each)
    // ------------------------------------------------------------------
    {
        xrt::hw_context ctx(device_, im2col_spec_.xclbin_obj.get_uuid());
        xrt::kernel     im2col_k(ctx, im2col_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, im2col_k, im2col_spec_.instr);
        int             n_instr  = (int)im2col_spec_.instr.size();

        auto t0 = clk::now();
        for (int ph = 0; ph < FU_PH; ++ph)
            for (int qt = 0; qt < 4; ++qt)
                exec_im2col(im2col_k, bo_instr, n_instr, ph, qt);
        im2col_us_ = elapsed_us(t0, clk::now());

        if (verbosity_ >= 1)
            std::cout << "[FusedPreprocessor] im2col done: " << im2col_us_ << " us\n";
    } // ctx + instr BO destroyed

    // ------------------------------------------------------------------
    // Phase 2 — GEMM: ONE hw_context, 192 dispatches
    //   (n_m=32 M-tiles × n_n=6 N-tiles)
    // ------------------------------------------------------------------
    {
        xrt::hw_context ctx(device_, gemm_spec_.xclbin_obj.get_uuid());
        xrt::kernel     gemm_k(ctx, gemm_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, gemm_k, gemm_spec_.instr);
        int             n_instr  = (int)gemm_spec_.instr.size();

        int n_m = FU_SEQ      / FU_M_TILE;   // 32
        int n_n = FU_EMBD_DIM / FU_N_TILE;   // 6

        auto t0 = clk::now();
        for (int m = 0; m < n_m; ++m)
            for (int n = 0; n < n_n; ++n)
                exec_gemm(gemm_k, bo_instr, n_instr, kernel_B_tiles, output, m, n);
        gemm_us_ = elapsed_us(t0, clk::now());

        if (verbosity_ >= 1)
            std::cout << "[FusedPreprocessor] GEMM done: " << gemm_us_ << " us\n";
    } // ctx + instr BO destroyed
}
