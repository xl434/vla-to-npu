// unified.cpp — fused conv2d patch embedding via im2col GEMM on AMD NPU.
//
// Mirrors preprocessing_fused_bf16.py.
//
// im2col reformulates the conv2d as GEMM:
//   patches[1024, 768] = im2col(image[3, 512, 512])
//   kernel_2d_T[768, 768] = reshape+transpose(kernel[768, 3, 16, 16])
//   output[1024, 768] = patches @ kernel_2d_T
//
// The GEMM is tiled (16 M-tiles × 12 N-tiles = 192 calls). Each call uses
// a Pk=4 K-chain on the AIE array, so the full K=768 dot product — which
// is exactly the conv2d + channel-add — is computed in one streaming pass.
//
// One hw_context covers all 192 calls: only 1 context open + 1 instr BO
// allocated per run(), vs 14,592 context opens in the original design.

#include "unified.h"

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
// FusedKernelSpec::preload
// ---------------------------------------------------------------------------
void FusedKernelSpec::preload(xrt::device       &dev,
                               const std::string &xclbin_path,
                               const std::string &instr_path)
{
    xclbin_obj = xrt::xclbin(xclbin_path);
    dev.register_xclbin(xclbin_obj);

    auto xks = xclbin_obj.get_kernels();
    auto xk  = *std::find_if(xks.begin(), xks.end(),
        [](xrt::xclbin::kernel &k){
            return k.get_name().rfind("MLIR_AIE", 0) == 0; });
    kernel_name = xk.get_name();

    std::ifstream f(instr_path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open instruction file: " + instr_path);
    f.seekg(0, std::ios::end);
    std::streamsize nb = f.tellg();
    f.seekg(0, std::ios::beg);
    if (nb % 4 != 0)
        throw std::runtime_error("Instruction file not multiple of 4 bytes: " + instr_path);
    instr.resize(nb / 4);
    f.read(reinterpret_cast<char *>(instr.data()), nb);
}

// ---------------------------------------------------------------------------
// Helper: allocate + fill instruction BO inside an already-open context.
// XCL_BO_FLAGS_CACHEABLE BOs MUST be allocated inside an active hw_context.
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
// Constructor
// ---------------------------------------------------------------------------
FusedPreprocessor::FusedPreprocessor(
    unsigned int        device_index,
    const std::string & xclbin_fused,
    const std::string & instr_fused,
    int trace_size,
    int verbosity)
    : verbosity_(verbosity)
    , trace_size_(trace_size)
    , device_(device_index)
    , patches_(FUSED_OUTPUT_ELEMS)      // [1024, 768]
    , kernel_2d_T_(FUSED_EMBD_DIM * FUSED_K_FULL)  // [768, 768]
{
    fused_spec_.preload(device_, xclbin_fused, instr_fused);
    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Kernel spec loaded.\n";

    // Discover data group IDs using a temporary context
    int g3, g4, g5, g7;
    {
        xrt::hw_context tmp_ctx(device_, fused_spec_.xclbin_obj.get_uuid());
        xrt::kernel     tmp_k(tmp_ctx, fused_spec_.kernel_name);
        g3 = tmp_k.group_id(3);
        g4 = tmp_k.group_id(4);
        g5 = tmp_k.group_id(5);
        g7 = tmp_k.group_id(7);
        if (verbosity_ >= 1)
            std::cout << "[FusedPreprocessor] Data group IDs: slot3=" << g3
                      << " slot4=" << g4 << " slot5=" << g5
                      << " slot7=" << g7 << "\n";
    }

    int tmp_trace = (trace_size_ > 0) ? trace_size_ : 1;

    // A[M_TILE, K_FULL] = [64, 768]
    bo_A_ = xrt::bo(device_, FUSED_A_TILE_ELEMS * sizeof(std::bfloat16_t),
                    XRT_BO_FLAGS_HOST_ONLY, g3);
    // C[M_TILE, N_TILE] = [64, 64]
    bo_C_ = xrt::bo(device_, FUSED_C_TILE_ELEMS * sizeof(std::bfloat16_t),
                    XRT_BO_FLAGS_HOST_ONLY, g4);
    // B[K_FULL, N_TILE] = [768, 64]
    bo_B_ = xrt::bo(device_, FUSED_B_TILE_ELEMS * sizeof(std::bfloat16_t),
                    XRT_BO_FLAGS_HOST_ONLY, g5);
    bo_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4,
                        XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_trace_.map<char *>(), 0, trace_size_);

    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Data BOs allocated.\n";
}

// ---------------------------------------------------------------------------
// im2col
//
// Converts image[C, H, W] (row-major, contiguous) into patches[PH*PW, C*KH*KW].
//
// patches[ph*PW + pw, c*KH*KW + ky*KW + kx] = image[c, ph*KH + ky, pw*KW + kx]
//
// This matches the Python:
//   image.reshape(C, PH, KH, PW, KW).transpose(1,3,0,2,4).reshape(PH*PW, K_FULL)
// ---------------------------------------------------------------------------
void FusedPreprocessor::im2col(const std::bfloat16_t *image)
{
    constexpr int PH = FUSED_PIX_LEN / FUSED_KERNEL_H;  // 32
    constexpr int PW = FUSED_PIX_LEN / FUSED_KERNEL_W;  // 32
    constexpr int KH = FUSED_KERNEL_H;
    constexpr int KW = FUSED_KERNEL_W;
    constexpr int C  = FUSED_CHANNELS;
    constexpr int W  = FUSED_PIX_LEN;

    for (int ph = 0; ph < PH; ++ph) {
        for (int pw = 0; pw < PW; ++pw) {
            int patch_row = ph * PW + pw;
            for (int c = 0; c < C; ++c) {
                for (int ky = 0; ky < KH; ++ky) {
                    for (int kx = 0; kx < KW; ++kx) {
                        int k_col = c * KH * KW + ky * KW + kx;
                        int img_row = ph * KH + ky;
                        int img_col = pw * KW + kx;
                        patches_[(size_t)patch_row * FUSED_K_FULL + k_col] =
                            image[(size_t)c * W * W + (size_t)img_row * W + img_col];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// exec_fused — single [64,768] × [768,64] → [64,64] GEMM call
// ---------------------------------------------------------------------------
void FusedPreprocessor::exec_fused(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *a_tile,
    const std::bfloat16_t *b_tile,
    std::bfloat16_t       *c_tile)
{
    memcpy(bo_A_.map<std::bfloat16_t *>(), a_tile,
           FUSED_A_TILE_ELEMS * sizeof(std::bfloat16_t));
    memcpy(bo_B_.map<std::bfloat16_t *>(), b_tile,
           FUSED_B_TILE_ELEMS * sizeof(std::bfloat16_t));
    bo_A_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_B_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size, bo_A_, bo_C_, bo_B_, bo_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("fused GEMM kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_C_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(c_tile, bo_C_.map<std::bfloat16_t *>(),
           FUSED_C_TILE_ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// run() — mirrors conv2d_fused() in Python
// ---------------------------------------------------------------------------
void FusedPreprocessor::run(
    const std::bfloat16_t *image,
    const std::bfloat16_t *kernel,
    std::bfloat16_t       *output)
{
    npu_us_ = 0.0;

    // -----------------------------------------------------------------------
    // Step 1 — im2col on CPU: image[3,512,512] → patches_[1024,768]
    // -----------------------------------------------------------------------
    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Step 1: im2col.\n";
    im2col(image);

    // -----------------------------------------------------------------------
    // Step 2 — reshape + transpose kernel: [768,3,16,16] → kernel_2d_T_[768,768]
    //
    // kernel_2d[n, k] = kernel[n, k//(KH*KW), (k%(KH*KW))//KW, k%KW]
    // We want B = kernel_2d.T so that patches @ B = output, i.e.
    // kernel_2d_T[k, n] = kernel_2d[n, k].
    // Since kernel is contiguous [EMBD_DIM, K_FULL] after reshape, the
    // transpose is just a copy with swapped indices.
    // -----------------------------------------------------------------------
    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Step 2: kernel reshape.\n";
    {
        // kernel is already laid out as [EMBD_DIM, K_FULL] = [768, 768] in memory
        // (C-major, innermost = kx, then ky, then c, then n).
        // kernel_2d_T[k, n] = kernel_flat[n * K_FULL + k]
        const std::bfloat16_t *kflat = kernel;
        for (int n = 0; n < FUSED_EMBD_DIM; ++n)
            for (int k = 0; k < FUSED_K_FULL; ++k)
                kernel_2d_T_[(size_t)k * FUSED_EMBD_DIM + n] =
                    kflat[(size_t)n * FUSED_K_FULL + k];
    }

    // -----------------------------------------------------------------------
    // Step 3 — 192 fused GEMM calls (16 M-tiles × 12 N-tiles)
    //
    // One hw_context covers all 192 calls: same xclbin, one instr BO.
    // Each call: patches[m*64:(m+1)*64, :] × kernel_2d_T[:, n*64:(n+1)*64]
    //   → output[m*64:(m+1)*64, n*64:(n+1)*64]
    // -----------------------------------------------------------------------
    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Step 3: 192 fused GEMM calls.\n";

    constexpr int n_m = FUSED_SEQ      / FUSED_M_TILE;  // 16
    constexpr int n_n = FUSED_EMBD_DIM / FUSED_N_TILE;  // 12

    {
        xrt::hw_context ctx(device_, fused_spec_.xclbin_obj.get_uuid());
        xrt::kernel     fused_k(ctx, fused_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, fused_k, fused_spec_.instr);
        int             n_instr  = (int)fused_spec_.instr.size();

        for (int m = 0; m < n_m; ++m) {
            const std::bfloat16_t *a_tile =
                &patches_[(size_t)m * FUSED_M_TILE * FUSED_K_FULL];

            for (int n = 0; n < n_n; ++n) {
                // B tile: kernel_2d_T[K_FULL, N_TILE] — column slice
                // kernel_2d_T is [768, 768]; col-slice starting at n*N_TILE
                // but stored row-major so we must gather into a contiguous buffer
                std::vector<std::bfloat16_t> b_tile(FUSED_B_TILE_ELEMS);
                for (int k = 0; k < FUSED_K_FULL; ++k)
                    memcpy(&b_tile[(size_t)k * FUSED_N_TILE],
                           &kernel_2d_T_[(size_t)k * FUSED_EMBD_DIM + n * FUSED_N_TILE],
                           FUSED_N_TILE * sizeof(std::bfloat16_t));

                std::bfloat16_t *c_tile =
                    &output[(size_t)m * FUSED_M_TILE * FUSED_EMBD_DIM
                            + (size_t)n * FUSED_N_TILE];

                // c_tile points into strided output; need contiguous buf
                std::vector<std::bfloat16_t> c_buf(FUSED_C_TILE_ELEMS);
                exec_fused(fused_k, bo_instr, n_instr,
                           a_tile, b_tile.data(), c_buf.data());

                // scatter c_buf back into strided output
                for (int r = 0; r < FUSED_M_TILE; ++r)
                    memcpy(&output[(size_t)(m * FUSED_M_TILE + r) * FUSED_EMBD_DIM
                                   + n * FUSED_N_TILE],
                           &c_buf[(size_t)r * FUSED_N_TILE],
                           FUSED_N_TILE * sizeof(std::bfloat16_t));
            }
        }
    } // context + instr BO destroyed

    if (verbosity_ >= 1) std::cout << "[FusedPreprocessor] Done.\n";
}
