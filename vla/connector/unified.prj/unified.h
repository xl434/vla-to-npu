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
static constexpr int CONN_K        = 64;
static constexpr int CONN_N        = 64;

static constexpr size_t CONN_A_ELEMS   = (size_t)CONN_SEQ * CONN_EMBD;
static constexpr size_t CONN_W_ELEMS   = (size_t)CONN_NEW_EMBD * CONN_TEXT;
static constexpr size_t CONN_OUT_ELEMS = (size_t)CONN_NEW_SEQ * CONN_TEXT;

static constexpr size_t CONN_COPY_IN_ELEMS  = 4 * CONN_EMBD;
static constexpr size_t CONN_COPY_OUT_ELEMS = 1 * CONN_EMBD * 4;
static constexpr size_t CONN_GEMM_A_ELEMS   = (size_t)CONN_NEW_SEQ * CONN_K;
static constexpr size_t CONN_GEMM_B_ELEMS   = (size_t)CONN_K * CONN_TEXT;
static constexpr size_t CONN_GEMM_C_ELEMS   = (size_t)CONN_NEW_SEQ * CONN_TEXT;
static constexpr size_t CONN_ADD_ELEMS      = (size_t)CONN_NEW_SEQ * CONN_N;

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
//   The NPU GEMM kernel outputs bfloat16. Accumulating 192 bfloat16 tiles in
//   bfloat16 causes ~23% mean error due to compounding rounding. Instead,
//   each bfloat16 GEMM tile is upcast to float32 on the CPU and accumulated
//   into a float32 buffer. The final result is stored as float32.
//
// Hardware constraint (AMD NPU): only ONE hw_context may be active at a time.
// run() manages contexts with scoped blocks:
//   Phase 1: { copy ctx }  256 copy calls
//   Phase 2: for each of 192 K-tiles:
//              { gemm ctx }  1 gemm call
//              CPU float32 accumulate into out_f32_
// Total context opens: 1 + 192 = 193.
// ---------------------------------------------------------------------------
class Connector {
public:
    Connector(
        unsigned int        device_index,
        const std::string & xclbin_copy,
        const std::string & instr_copy,
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

    KernelSpec copy_spec_, gemm_spec_;

    // Persistent data BOs (XRT_BO_FLAGS_HOST_ONLY — no context needed).
    // Instruction BOs (XCL_BO_FLAGS_CACHEABLE) require an active hw_context
    // and are allocated transiently inside each context block in run().
    xrt::bo bo_copy_in0_, bo_copy_out1_, bo_copy_trace_;
    xrt::bo bo_gemm_in0_, bo_gemm_out1_, bo_gemm_in2_, bo_gemm_trace_;

    // Scratch host buffers
    std::vector<std::bfloat16_t> A_shuf_;  // [NEW_SEQ, NEW_EMBD]
    std::vector<std::bfloat16_t> C_tmp_;   // [NEW_SEQ, TEXT] — GEMM bf16 output
    std::vector<std::bfloat16_t> a_tile_;  // [NEW_SEQ, K]

    void exec_copy(xrt::kernel       &k,
                   xrt::bo           &bo_instr,
                   int                instr_size,
                   const std::bfloat16_t *src,
                   std::bfloat16_t       *dst);
    void exec_gemm(xrt::kernel       &k,
                   xrt::bo           &bo_instr,
                   int                instr_size,
                   const std::bfloat16_t *a_tile,
                   const std::bfloat16_t *b_tile,
                   std::bfloat16_t       *c_out);
};
