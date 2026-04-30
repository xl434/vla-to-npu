#pragma once

#include <cstdint>
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Problem dimensions — must match preprocessing_bf16.py
static constexpr int PREPROC_INPUT_DIM   = 256;
static constexpr int PREPROC_KERNEL_DIM  = 16;
static constexpr int PREPROC_OUTPUT_DIM  = 16;
static constexpr int PREPROC_CHANNELS    = 3;
static constexpr int PREPROC_PIX_LEN     = 512;
static constexpr int PREPROC_SEQ         = 1024;
static constexpr int PREPROC_EMBD_DIM    = 768;
static constexpr int PREPROC_PATCHES_SIDE = PREPROC_PIX_LEN / PREPROC_KERNEL_DIM; // 32
static constexpr int PREPROC_TILE_SIDE    = PREPROC_PIX_LEN / PREPROC_INPUT_DIM;  // 2

static constexpr size_t PREPROC_IMAGE_ELEMS  =
    (size_t)PREPROC_CHANNELS * PREPROC_PIX_LEN * PREPROC_PIX_LEN;
static constexpr size_t PREPROC_KERNEL_ELEMS =
    (size_t)PREPROC_EMBD_DIM * PREPROC_CHANNELS
    * PREPROC_KERNEL_DIM * PREPROC_KERNEL_DIM;
static constexpr size_t PREPROC_OUTPUT_ELEMS =
    (size_t)PREPROC_SEQ * PREPROC_EMBD_DIM;

// ---------------------------------------------------------------------------
// KernelSpec — registers xclbin + loads instructions WITHOUT opening a
// hw_context. The AMD NPU only supports one active hw_context at a time, so
// contexts are created and destroyed inside run() as needed.
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
// Preprocessor
//
// Implements the full conv2d patch-embedding pipeline from
// preprocessing_bf16.py using three AIE kernels (conv, add_32_32, copy).
//
// Hardware constraint (AMD NPU Phoenix/Strix): only ONE hw_context may be
// active at a time. run() opens one context per kernel-type batch and
// destroys it before opening the next.
// ---------------------------------------------------------------------------
class Preprocessor {
public:
    Preprocessor(
        unsigned int        device_index,
        const std::string & xclbin_conv,
        const std::string & instr_conv,
        const std::string & xclbin_add,
        const std::string & instr_add,
        const std::string & xclbin_copy,
        const std::string & instr_copy,
        int trace_size = 0,
        int verbosity  = 0
    );

    ~Preprocessor() = default;

    // Run the full pipeline (mirrors conv2d() in Python).
    //   image  : [CHANNELS, PIX_LEN, PIX_LEN]
    //   kernel : [EMBD_DIM, CHANNELS, KERNEL_DIM, KERNEL_DIM]
    //   output : [SEQ, EMBD_DIM]
    void run(
        const std::bfloat16_t * image,
        const std::bfloat16_t * kernel,
        std::bfloat16_t       * output
    );

    // Returns accumulated NPU kernel execution time from the last run() call.
    double npu_time_us() const { return npu_us_; }

private:
    int verbosity_;
    int trace_size_;
    double npu_us_ = 0.0;
    xrt::device device_;

    KernelSpec conv_spec_, add_spec_, copy_spec_;

    // Persistent data BOs — XRT_BO_FLAGS_HOST_ONLY BOs can be allocated without
    // an active hw_context and are reused across all kernel invocations.
    // NOTE: XCL_BO_FLAGS_CACHEABLE instruction BOs REQUIRE an active hw_context
    // and are therefore allocated transiently inside each context block in run().
    xrt::bo bo_conv_in0_, bo_conv_out1_, bo_conv_in2_, bo_conv_trace_;
    xrt::bo bo_add_in0_,  bo_add_out1_,  bo_add_in2_,  bo_add_trace_;
    xrt::bo bo_copy_in0_, bo_copy_out1_, bo_copy_trace_;

    // Scratch host buffers
    std::vector<std::bfloat16_t> embd_;         // [PS*PS*EMBD_DIM] channel-last
    std::vector<std::bfloat16_t> tmp_per_ch_;   // [CHANNELS*PS*PS] — tmp[j] at offset j*PS*PS
    std::vector<std::bfloat16_t> a_tile_;       // [INPUT_DIM*INPUT_DIM]
    std::vector<std::bfloat16_t> conv_tile_out_;// [OUTPUT_DIM*OUTPUT_DIM]
    std::vector<std::bfloat16_t> embd_slice_;   // [PS*PS]
    std::vector<std::bfloat16_t> add_result_;   // [PS*PS]
    std::vector<std::bfloat16_t> copy_src_;     // [256]
    std::vector<std::bfloat16_t> copy_dst_;     // [256]

    // Kernel dispatch helpers — caller passes an already-open xrt::kernel AND
    // the matching instruction BO (allocated inside the same hw_context).
    void exec_conv(xrt::kernel       &k,
                   xrt::bo           &bo_instr,
                   int                instr_size,
                   const std::bfloat16_t *src_img,
                   const std::bfloat16_t *src_ker,
                   std::bfloat16_t       *dst);
    void exec_add(xrt::kernel        &k,
                  xrt::bo            &bo_instr,
                  int                 instr_size,
                  const std::bfloat16_t *src_x,
                  const std::bfloat16_t *src_y,
                  std::bfloat16_t       *dst);
    void exec_copy(xrt::kernel       &k,
                   xrt::bo           &bo_instr,
                   int                instr_size,
                   const std::bfloat16_t *src,
                   std::bfloat16_t       *dst);
};
