#pragma once

#include <cstdint>
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Problem dimensions — must match connector_bf16.py
static constexpr int CONN_SEQ      = 1024;
static constexpr int CONN_EMBD     = 768;
static constexpr int CONN_NEW_SEQ  = 64;
static constexpr int CONN_NEW_EMBD = 12288;
static constexpr int CONN_TEXT     = 960;
static constexpr int CONN_M_TILE   = 32;    // NEW_SEQ / 2
static constexpr int CONN_K_TILE   = 768;   // NEW_EMBD / 16
static constexpr int CONN_N_TILE   = 64;    // TEXT / 15

static constexpr size_t CONN_A_ELEMS   = (size_t)CONN_SEQ * CONN_EMBD;
static constexpr size_t CONN_W_ELEMS   = (size_t)CONN_NEW_EMBD * CONN_TEXT;
static constexpr size_t CONN_OUT_ELEMS = (size_t)CONN_NEW_SEQ * CONN_TEXT;

// OPT-B pixel shuffle constants (replaces old CONN_COPY_* constants)
// Phase 1 — pixel shuffle: 32 calls (4 col-chunks × 8 row-batches) instead of 256.
// Input:  [SHUFFLE_CORES * SHUFFLE_BATCH, EMBD] = [32, 768] = 24576 elements
// Output: [SHUFFLE_BATCH, SHUFFLE_CORES * EMBD] = [8, 3072] = 24576 elements
static constexpr int CONN_SHUFFLE_CORES = 4;
static constexpr int CONN_SHUFFLE_BATCH = 8;
static constexpr int CONN_N_COL_CHUNKS  = 4;
static constexpr int CONN_N_ROW_BATCHES = CONN_NEW_SEQ / CONN_SHUFFLE_BATCH;  // 8
static constexpr size_t CONN_PS_IN_ELEMS  = (size_t)CONN_SHUFFLE_CORES * CONN_SHUFFLE_BATCH * CONN_EMBD;  // 24576
static constexpr size_t CONN_PS_OUT_ELEMS = (size_t)CONN_SHUFFLE_BATCH * CONN_SHUFFLE_CORES * CONN_EMBD;  // 24576

static constexpr size_t CONN_GEMM_A_ELEMS   = (size_t)CONN_M_TILE * CONN_K_TILE;  // 24576
static constexpr size_t CONN_GEMM_B_ELEMS   = (size_t)CONN_K_TILE * CONN_N_TILE;  // 49152
static constexpr size_t CONN_GEMM_C_ELEMS   = (size_t)CONN_M_TILE * CONN_N_TILE;  // 2048

// ---------------------------------------------------------------------------
// KernelSpec — xclbin + instructions without an active hw_context.
// ---------------------------------------------------------------------------
struct KernelSpec {
    xrt::xclbin           xclbin_obj;
    std::string           kernel_name;
    std::vector<uint32_t> instr;

    void preload(xrt::device       &dev,
                 const std::string &xclbin_path,
                 const std::string &instr_path);
};

// ---------------------------------------------------------------------------
// Connector
//
// Implements the pixel-shuffle + tiled-GEMM pipeline from connector_bf16.py.
//
// Accumulation strategy:
//   The NPU GEMM kernel outputs bfloat16. Each bf16 GEMM tile is upcast to
//   float32 on the CPU and accumulated into a float32 output buffer.
//   This matches the Python connector_bf16.py approach exactly.
//
// Hardware constraint (AMD NPU): only ONE hw_context may be active at a time.
// run() manages contexts with scoped blocks:
//   Phase 1: { pixel_shuffle ctx }  32 pixel_shuffle calls (OPT-B: 4 col-chunks × 8 row-batches)
//   Phase 2: { gemm ctx }          480 GEMM calls (m×n×k = 2×15×16)
//             CPU float32 accumulate into out[] per m/n tile
// Total context opens: 2.
// ---------------------------------------------------------------------------
class Connector {
public:
    Connector(
        unsigned int        device_index,
        const std::string & xclbin_pixel_shuffle,
        const std::string & instr_pixel_shuffle,
        const std::string & xclbin_gemm,
        const std::string & instr_gemm,
        int trace_size = 0,
        int verbosity  = 0
    );

    ~Connector() = default;

    // Run the full pipeline (mirrors fused_op() in Python).
    //   A   : [SEQ=1024,    EMBD=768]     bfloat16
    //   W   : [NEW_EMBD=12288, TEXT=960]  bfloat16
    //   out : [NEW_SEQ=64,  TEXT=960]     float32 (upcast for precision)
    void run(
        const std::bfloat16_t * A,
        const std::bfloat16_t * W,
        float                 * out
    );

    // Returns accumulated NPU kernel execution time from the last run() call.
    double npu_time_us() const { return npu_us_; }

private:
    int verbosity_;
    int trace_size_;
    double npu_us_ = 0.0;
    xrt::device device_;

    KernelSpec ps_spec_, gemm_spec_;

    // Persistent data BOs (XRT_BO_FLAGS_HOST_ONLY — no context needed).
    // Instruction BOs (XCL_BO_FLAGS_CACHEABLE) require an active hw_context
    // and are allocated transiently inside each context block in run().
    xrt::bo bo_ps_in0_, bo_ps_out1_, bo_ps_trace_;
    xrt::bo bo_gemm_in0_, bo_gemm_out1_, bo_gemm_in2_, bo_gemm_trace_;

    // Precomputed row-index tables for pixel shuffle pre-gather on CPU.
    // PIXEL_ROW_IDX[j][i] = (i/8)*128 + (i%8)*4 + j*32 + (i%4_within_group)
    // Computed as: for j in 0..4, for i in 0..NEW_SEQ=64, for r in 0..4
    //   PIXEL_ROW_IDX[j][ i*4 + r ] = (i/8)*128 + (i%8)*4 + j*32 + r
    //   shape: [4][256]
    int PIXEL_ROW_IDX[4][256];

    // PS_ROW_IDX[j][b][g][r] = source row in A for pixel_shuffle call (j, b), core g, row r
    // j=col_chunk in 0..4, b=row_batch in 0..8, g=group in 0..4, r=row in 0..8
    int PS_ROW_IDX[4][8][4][8];

    // Scratch host buffers
    std::vector<std::bfloat16_t> A_shuf_;     // [NEW_SEQ, NEW_EMBD]
    std::vector<std::bfloat16_t> ps_in_buf_;  // [SHUFFLE_CORES*SHUFFLE_BATCH, EMBD] = [32, 768]
    std::vector<std::bfloat16_t> ps_out_buf_; // [SHUFFLE_BATCH, SHUFFLE_CORES*EMBD] = [8, 3072]
    std::vector<std::bfloat16_t> a_tile_;     // [M_TILE, K_TILE]  contiguous gather
    std::vector<std::bfloat16_t> b_tile_;     // [K_TILE, N_TILE]  contiguous gather
    std::vector<std::bfloat16_t> c_tile_;     // [M_TILE, N_TILE]  GEMM bf16 output

    void exec_pixel_shuffle(xrt::kernel           &k,
                            xrt::bo               &bo_instr,
                            int                    instr_size,
                            const std::bfloat16_t *src,
                            std::bfloat16_t       *dst);
    void exec_gemm(xrt::kernel       &k,
                   xrt::bo           &bo_instr,
                   int                instr_size,
                   const std::bfloat16_t *a_tile,
                   const std::bfloat16_t *b_tile,
                   std::bfloat16_t       *c_out);
};
