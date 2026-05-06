// unified.cpp — connector pipeline for AMD NPU.
//
// Accumulation strategy:
//   NPU GEMM outputs bfloat16. Each bf16 GEMM tile is upcast to float32 on
//   CPU and accumulated into out[]. Matches connector_bf16.py exactly.
//
// Context schedule:
//   Phase 1 — pixel shuffle:
//     { copy ctx }  allocate copy instr BO, run 256 copy calls
//   Phase 2 — tiled GEMM + CPU float32 accumulate:
//     { gemm ctx }  allocate gemm instr BO once, run all 480 GEMM calls
//                   (m=2 × n=15 × k=16 = 480)
//   Total context opens: 2.

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
// KernelSpec::preload
// ---------------------------------------------------------------------------
void KernelSpec::preload(xrt::device       &dev,
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
// Constructor
// ---------------------------------------------------------------------------
Connector::Connector(
    unsigned int        device_index,
    const std::string & xclbin_copy,
    const std::string & instr_copy,
    const std::string & xclbin_gemm,
    const std::string & instr_gemm,
    int trace_size,
    int verbosity)
    : verbosity_(verbosity)
    , trace_size_(trace_size)
    , device_(device_index)
    , A_shuf_(CONN_NEW_SEQ * CONN_NEW_EMBD)
    , a_tile_(CONN_GEMM_A_ELEMS)
    , b_tile_(CONN_GEMM_B_ELEMS)
    , c_tile_(CONN_GEMM_C_ELEMS)
{
    copy_spec_.preload(device_, xclbin_copy, instr_copy);
    gemm_spec_.preload(device_, xclbin_gemm, instr_gemm);
    if (verbosity_ >= 1) std::cout << "[Connector] Kernel specs loaded.\n";

    // Discover data group IDs
    int g3, g4, g5, g7;
    {
        xrt::hw_context tmp_ctx(device_, copy_spec_.xclbin_obj.get_uuid());
        xrt::kernel     tmp_k(tmp_ctx, copy_spec_.kernel_name);
        g3 = tmp_k.group_id(3);
        g4 = tmp_k.group_id(4);
        g5 = tmp_k.group_id(5);
        g7 = tmp_k.group_id(7);
        if (verbosity_ >= 1)
            std::cout << "[Connector] Data group IDs: slot3=" << g3
                      << " slot4=" << g4 << " slot5=" << g5
                      << " slot7=" << g7 << "\n";
    }

    int tmp_trace = (trace_size_ > 0) ? trace_size_ : 1;

    bo_copy_in0_   = xrt::bo(device_, CONN_COPY_IN_ELEMS  * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_copy_out1_  = xrt::bo(device_, CONN_COPY_OUT_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_copy_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_copy_trace_.map<char *>(), 0, trace_size_);

    bo_gemm_in0_   = xrt::bo(device_, CONN_GEMM_A_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_gemm_out1_  = xrt::bo(device_, CONN_GEMM_C_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_gemm_in2_   = xrt::bo(device_, CONN_GEMM_B_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g5);
    bo_gemm_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_gemm_trace_.map<char *>(), 0, trace_size_);

    if (verbosity_ >= 1) std::cout << "[Connector] Data BOs allocated.\n";
}

// ---------------------------------------------------------------------------
// Helper: allocate + fill instruction BO inside an already-open context.
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
// exec_copy
// ---------------------------------------------------------------------------
void Connector::exec_copy(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *src,
    std::bfloat16_t       *dst)
{
    memcpy(bo_copy_in0_.map<std::bfloat16_t *>(), src,
           CONN_COPY_IN_ELEMS * sizeof(std::bfloat16_t));
    bo_copy_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_copy_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size,
                 bo_copy_in0_, bo_copy_out1_, bo_copy_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("copy kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_copy_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(dst, bo_copy_out1_.map<std::bfloat16_t *>(),
           CONN_COPY_OUT_ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// exec_gemm
// ---------------------------------------------------------------------------
void Connector::exec_gemm(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *a_tile,
    const std::bfloat16_t *b_tile,
    std::bfloat16_t       *c_out)
{
    memcpy(bo_gemm_in0_.map<std::bfloat16_t *>(), a_tile,
           CONN_GEMM_A_ELEMS * sizeof(std::bfloat16_t));
    memcpy(bo_gemm_in2_.map<std::bfloat16_t *>(), b_tile,
           CONN_GEMM_B_ELEMS * sizeof(std::bfloat16_t));
    bo_gemm_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_gemm_in2_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_gemm_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size,
                 bo_gemm_in0_, bo_gemm_out1_, bo_gemm_in2_, bo_gemm_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("gemm kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_gemm_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(c_out, bo_gemm_out1_.map<std::bfloat16_t *>(),
           CONN_GEMM_C_ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// run() — mirrors fused_op() in connector_bf16.py
//
// out[] is float32. Each GEMM tile (bf16) is upcast to float32 before
// accumulation, preventing compounding rounding error across 480 GEMM calls.
// ---------------------------------------------------------------------------
void Connector::run(
    const std::bfloat16_t *A,
    const std::bfloat16_t *W,
    float                 *out)
{
    npu_us_ = 0.0;
    std::fill(out, out + CONN_OUT_ELEMS, 0.0f);

    // ==================================================================
    // Phase 1 — Pixel shuffle: A[SEQ, EMBD] → A_shuf_[NEW_SEQ, NEW_EMBD]
    // ==================================================================
    if (verbosity_ >= 1) std::cout << "[Connector] Phase 1: pixel shuffle.\n";
    {
        xrt::hw_context ctx(device_, copy_spec_.xclbin_obj.get_uuid());
        xrt::kernel     copy_k(ctx, copy_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, copy_k, copy_spec_.instr);
        int             n_instr  = (int)copy_spec_.instr.size();

        for (int i = 0; i < CONN_NEW_SEQ; ++i) {
            int offset = (i / 8) * 128 + (i % 8) * 4;
            for (int j = 0; j < 4; ++j) {
                int src_row = offset + j * 32;
                const std::bfloat16_t *src = &A[(size_t)src_row * CONN_EMBD];
                std::bfloat16_t *dst =
                    &A_shuf_[(size_t)i * CONN_NEW_EMBD + j * CONN_EMBD * 4];
                exec_copy(copy_k, bo_instr, n_instr, src, dst);
            }
        }
    } // copy ctx + instr BO destroyed
    if (verbosity_ >= 1) std::cout << "[Connector] Phase 1 done.\n";

    // ==================================================================
    // Phase 2 — Tiled GEMM + float32 accumulation on CPU
    //
    // Triple loop: m (2) × n (15) × k (16) = 480 GEMM calls.
    // One gemm hw_context is opened for ALL 480 calls.
    //
    // A_tile [M_TILE, K_TILE]: gathered contiguously from A_shuf_ column slice.
    // B_tile [K_TILE, N_TILE]: gathered contiguously from W 2D slice.
    // C_tile [M_TILE, N_TILE]: scattered row-by-row into out[m*M:(m+1)*M, n*N:(n+1)*N].
    // ==================================================================
    if (verbosity_ >= 1) std::cout << "[Connector] Phase 2: tiled GEMM + float32 accumulate.\n";

    constexpr int n_m = CONN_NEW_SEQ  / CONN_M_TILE;  // 2
    constexpr int n_k = CONN_NEW_EMBD / CONN_K_TILE;  // 16
    constexpr int n_n = CONN_TEXT     / CONN_N_TILE;  // 15

    {
        xrt::hw_context ctx(device_, gemm_spec_.xclbin_obj.get_uuid());
        xrt::kernel     gemm_k(ctx, gemm_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, gemm_k, gemm_spec_.instr);
        int             n_instr  = (int)gemm_spec_.instr.size();

        for (int m = 0; m < n_m; ++m) {
            for (int n = 0; n < n_n; ++n) {
                for (int k = 0; k < n_k; ++k) {
                    // Gather A_tile [M_TILE, K_TILE] — contiguous from column slice of A_shuf_
                    for (int r = 0; r < CONN_M_TILE; ++r)
                        memcpy(&a_tile_[r * CONN_K_TILE],
                               &A_shuf_[(size_t)(m * CONN_M_TILE + r) * CONN_NEW_EMBD
                                        + k * CONN_K_TILE],
                               CONN_K_TILE * sizeof(std::bfloat16_t));

                    // Gather B_tile [K_TILE, N_TILE] — contiguous from row slice of W
                    for (int r = 0; r < CONN_K_TILE; ++r)
                        memcpy(&b_tile_[r * CONN_N_TILE],
                               &W[(size_t)(k * CONN_K_TILE + r) * CONN_TEXT
                                  + n * CONN_N_TILE],
                               CONN_N_TILE * sizeof(std::bfloat16_t));

                    exec_gemm(gemm_k, bo_instr, n_instr,
                              a_tile_.data(), b_tile_.data(), c_tile_.data());

                    // Scatter-accumulate C_tile [M_TILE, N_TILE] → out (float32)
                    for (int r = 0; r < CONN_M_TILE; ++r)
                        for (int c = 0; c < CONN_N_TILE; ++c)
                            out[(size_t)(m * CONN_M_TILE + r) * CONN_TEXT
                                + n * CONN_N_TILE + c] +=
                                (float)c_tile_[r * CONN_N_TILE + c];
                }
            }
        }
    } // gemm ctx + instr BO destroyed

    if (verbosity_ >= 1) std::cout << "[Connector] Phase 2 done.\n";
}
