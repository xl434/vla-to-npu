#pragma once

#include <cstdint>
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Problem dimensions — must match preprocessing_fused_bf16.py
static constexpr int FUSED_SEQ       = 1024;   // (512/16)^2 patches
static constexpr int FUSED_EMBD_DIM  = 768;    // conv output channels
static constexpr int FUSED_CHANNELS  = 3;
static constexpr int FUSED_KERNEL_H  = 16;
static constexpr int FUSED_KERNEL_W  = 16;
static constexpr int FUSED_PIX_LEN   = 512;
static constexpr int FUSED_K_FULL    = FUSED_CHANNELS * FUSED_KERNEL_H * FUSED_KERNEL_W; // 768
static constexpr int FUSED_M_TILE    = 32;
static constexpr int FUSED_N_TILE    = 32;

static constexpr size_t FUSED_IMAGE_ELEMS  =
    (size_t)FUSED_CHANNELS * FUSED_PIX_LEN * FUSED_PIX_LEN;
static constexpr size_t FUSED_KERNEL_ELEMS =
    (size_t)FUSED_EMBD_DIM * FUSED_K_FULL;   // 768*768 after reshape
static constexpr size_t FUSED_OUTPUT_ELEMS =
    (size_t)FUSED_SEQ * FUSED_EMBD_DIM;      // 1024*768

// Per-call tile element counts
static constexpr size_t FUSED_A_TILE_ELEMS = (size_t)FUSED_M_TILE * FUSED_K_FULL; // 64*768
static constexpr size_t FUSED_B_TILE_ELEMS = (size_t)FUSED_K_FULL * FUSED_N_TILE; // 768*64
static constexpr size_t FUSED_C_TILE_ELEMS = (size_t)FUSED_M_TILE * FUSED_N_TILE; // 64*64

// ---------------------------------------------------------------------------
// KernelSpec — registers xclbin + loads instructions WITHOUT opening a
// hw_context.
// ---------------------------------------------------------------------------
struct FusedKernelSpec {
    xrt::xclbin           xclbin_obj;
    std::string           kernel_name;
    std::vector<uint32_t> instr;

    void preload(xrt::device       &dev,
                 const std::string &xclbin_path,
                 const std::string &instr_path);
};

// ---------------------------------------------------------------------------
// FusedPreprocessor
//
// Implements the fused patch-embedding pipeline from preprocessing_fused_bf16.py.
//
// Pipeline:
//   1. im2col on CPU: image[C,512,512] → patches[1024,768]
//   2. kernel reshape: [768,3,16,16] → kernel_2d_T[768,768] (transposed)
//   3. 192 fused GEMM calls (16 M-tiles × 12 N-tiles), each:
//      patches[m*64:(m+1)*64, :] × kernel_2d_T[:, n*64:(n+1)*64] → output tile [64,64]
//      via Pk=4 K-chain on AIE (fuses conv + channel add in one pass)
//
// Hardware constraint: only one hw_context active at a time; one context
// covers all 192 calls (same xclbin), reducing overhead vs old 14,592 calls.
// ---------------------------------------------------------------------------
class FusedPreprocessor {
public:
    FusedPreprocessor(
        unsigned int        device_index,
        const std::string & xclbin_fused,
        const std::string & instr_fused,
        int trace_size = 0,
        int verbosity  = 0
    );

    ~FusedPreprocessor() = default;

    // Run the full pipeline (mirrors conv2d_fused() in Python).
    //   image  : [CHANNELS=3, PIX_LEN=512, PIX_LEN=512]   bfloat16
    //   kernel : [EMBD_DIM=768, CHANNELS=3, KH=16, KW=16] bfloat16 (kernel_2d_T computed internally)
    //   output : [SEQ=1024, EMBD_DIM=768]                  bfloat16
    void run(
        const std::bfloat16_t * image,
        const std::bfloat16_t * kernel,
        std::bfloat16_t       * output
    );

    double npu_time_us() const { return npu_us_; }

private:
    int verbosity_;
    int trace_size_;
    double npu_us_ = 0.0;
    xrt::device device_;

    FusedKernelSpec fused_spec_;

    // Persistent data BOs (XRT_BO_FLAGS_HOST_ONLY — no context needed)
    xrt::bo bo_A_, bo_B_, bo_C_, bo_trace_;

    // Scratch host buffers
    std::vector<std::bfloat16_t> patches_;      // [1024, 768] — im2col output
    std::vector<std::bfloat16_t> kernel_2d_T_;  // [768, 768]  — reshaped + transposed kernel

    // im2col: image[C,H,W] → patches[H/16*W/16, C*16*16]
    void im2col(const std::bfloat16_t *image);

    // Single fused GEMM tile call
    void exec_fused(xrt::kernel       &k,
                    xrt::bo           &bo_instr,
                    int                instr_size,
                    const std::bfloat16_t *a_tile,
                    const std::bfloat16_t *b_tile,
                    std::bfloat16_t       *c_tile);
};
