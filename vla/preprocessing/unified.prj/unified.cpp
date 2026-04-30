// unified.cpp — conv2d patch-embedding pipeline for AMD NPU.
//
// Key constraint (from unified_kernel_guide.md §2):
//   The AMD NPU only supports ONE active xrt::hw_context at a time.
//   Creating a second context while one is alive fails with EINVAL.
//
// Key constraint (discovered via debugging):
//   XCL_BO_FLAGS_CACHEABLE instruction BOs MUST be allocated while an
//   hw_context is open. XRT_BO_FLAGS_HOST_ONLY data BOs can be allocated
//   at any time (no context required).
//
// Solution:
//   - Data BOs: allocated once in the constructor, reused forever.
//   - Instruction BOs: allocated transiently inside each context block.
//
// Context schedule per embed dim i:
//   { conv ctx }  allocate conv instr BO, run 12 convs, destroy ctx+instr BO
//   { add  ctx }  allocate add  instr BO, run  3 adds,  destroy ctx+instr BO
//   { copy ctx }  allocate copy instr BO, run  4 copies, destroy ctx+instr BO
// Total context opens: 3 × EMBD_DIM = 2304.

#include "unified.h"

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
Preprocessor::Preprocessor(
    unsigned int        device_index,
    const std::string & xclbin_conv,
    const std::string & instr_conv,
    const std::string & xclbin_add,
    const std::string & instr_add,
    const std::string & xclbin_copy,
    const std::string & instr_copy,
    int trace_size,
    int verbosity)
    : verbosity_(verbosity)
    , trace_size_(trace_size)
    , device_(device_index)
    , embd_((size_t)PREPROC_PATCHES_SIDE * PREPROC_PATCHES_SIDE * PREPROC_EMBD_DIM)
    , tmp_per_ch_((size_t)PREPROC_CHANNELS * PREPROC_PATCHES_SIDE * PREPROC_PATCHES_SIDE)
    , a_tile_((size_t)PREPROC_INPUT_DIM * PREPROC_INPUT_DIM)
    , conv_tile_out_((size_t)PREPROC_OUTPUT_DIM * PREPROC_OUTPUT_DIM)
    , embd_slice_((size_t)PREPROC_PATCHES_SIDE * PREPROC_PATCHES_SIDE)
    , add_result_((size_t)PREPROC_PATCHES_SIDE * PREPROC_PATCHES_SIDE)
    , copy_src_(256)
    , copy_dst_(256)
{
    // ---- Step 1: preload all three kernel specs (no hw_context opened) ----
    conv_spec_.preload(device_, xclbin_conv, instr_conv);
    add_spec_.preload(device_, xclbin_add,   instr_add);
    copy_spec_.preload(device_, xclbin_copy, instr_copy);
    if (verbosity_ >= 1)
        std::cout << "[Preprocessor] Kernel specs loaded.\n";

    // ---- Step 2: discover data group IDs using ONE temporary context ----
    // XCL_BO_FLAGS_CACHEABLE instr BOs need a live context — discovered per
    // context open in run() via kernel.group_id(1).
    // XRT_BO_FLAGS_HOST_ONLY data BOs only need a group ID number, which we
    // capture once here and reuse.
    int g3, g4, g5, g7;
    {
        xrt::hw_context tmp_ctx(device_, conv_spec_.xclbin_obj.get_uuid());
        xrt::kernel     tmp_k(tmp_ctx, conv_spec_.kernel_name);
        g3 = tmp_k.group_id(3);
        g4 = tmp_k.group_id(4);
        g5 = tmp_k.group_id(5);
        g7 = tmp_k.group_id(7);
        if (verbosity_ >= 1)
            std::cout << "[Preprocessor] Data group IDs: slot3=" << g3
                      << " slot4=" << g4 << " slot5=" << g5
                      << " slot7=" << g7 << "\n";
    } // tmp_ctx and tmp_k destroyed

    int tmp_trace = (trace_size_ > 0) ? trace_size_ : 1;

    // ---- Step 3: allocate data BOs (HOST_ONLY — no active context needed) ----
    bo_conv_in0_   = xrt::bo(device_,
        (size_t)PREPROC_INPUT_DIM  * PREPROC_INPUT_DIM  * sizeof(std::bfloat16_t),
        XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_conv_out1_  = xrt::bo(device_,
        (size_t)PREPROC_OUTPUT_DIM * PREPROC_OUTPUT_DIM * sizeof(std::bfloat16_t),
        XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_conv_in2_   = xrt::bo(device_,
        (size_t)PREPROC_KERNEL_DIM * PREPROC_KERNEL_DIM * sizeof(std::bfloat16_t),
        XRT_BO_FLAGS_HOST_ONLY, g5);
    bo_conv_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_conv_trace_.map<char *>(), 0, trace_size_);

    bo_add_in0_   = xrt::bo(device_, 1024 * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_add_out1_  = xrt::bo(device_, 1024 * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_add_in2_   = xrt::bo(device_, 1024 * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g5);
    bo_add_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_add_trace_.map<char *>(), 0, trace_size_);

    bo_copy_in0_   = xrt::bo(device_, 256 * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_copy_out1_  = xrt::bo(device_, 256 * sizeof(std::bfloat16_t), XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_copy_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_copy_trace_.map<char *>(), 0, trace_size_);

    if (verbosity_ >= 1) std::cout << "[Preprocessor] Data BOs allocated.\n";
}

// ---------------------------------------------------------------------------
// Helper: allocate + fill an instruction BO inside an already-open context.
// MUST be called while hw_context is alive (XCL_BO_FLAGS_CACHEABLE requires it).
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
// exec_conv — caller owns the open xrt::kernel AND instruction BO
// ---------------------------------------------------------------------------
void Preprocessor::exec_conv(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *src_img,
    const std::bfloat16_t *src_ker,
    std::bfloat16_t       *dst)
{
    constexpr size_t IN0 = (size_t)PREPROC_INPUT_DIM  * PREPROC_INPUT_DIM;
    constexpr size_t IN2 = (size_t)PREPROC_KERNEL_DIM * PREPROC_KERNEL_DIM;
    constexpr size_t OUT = (size_t)PREPROC_OUTPUT_DIM  * PREPROC_OUTPUT_DIM;

    memcpy(bo_conv_in0_.map<std::bfloat16_t *>(), src_img, IN0 * sizeof(std::bfloat16_t));
    memcpy(bo_conv_in2_.map<std::bfloat16_t *>(), src_ker, IN2 * sizeof(std::bfloat16_t));
    bo_conv_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv_in2_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_conv_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size,
                 bo_conv_in0_, bo_conv_out1_, bo_conv_in2_, bo_conv_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("conv kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_conv_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(dst, bo_conv_out1_.map<std::bfloat16_t *>(), OUT * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// exec_add — caller owns the open xrt::kernel AND instruction BO
// ---------------------------------------------------------------------------
void Preprocessor::exec_add(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *src_x,
    const std::bfloat16_t *src_y,
    std::bfloat16_t       *dst)
{
    constexpr size_t ELEMS = 1024; // 32*32
    memcpy(bo_add_in0_.map<std::bfloat16_t *>(), src_x, ELEMS * sizeof(std::bfloat16_t));
    memcpy(bo_add_in2_.map<std::bfloat16_t *>(), src_y, ELEMS * sizeof(std::bfloat16_t));
    bo_add_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_add_in2_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_add_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size,
                 bo_add_in0_, bo_add_out1_, bo_add_in2_, bo_add_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("add kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_add_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(dst, bo_add_out1_.map<std::bfloat16_t *>(), ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// exec_copy — caller owns the open xrt::kernel AND instruction BO
// ---------------------------------------------------------------------------
void Preprocessor::exec_copy(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *src,
    std::bfloat16_t       *dst)
{
    constexpr size_t ELEMS = 256;
    memcpy(bo_copy_in0_.map<std::bfloat16_t *>(), src, ELEMS * sizeof(std::bfloat16_t));
    bo_copy_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_copy_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    auto run = k(3u, bo_instr, instr_size,
                 bo_copy_in0_, bo_copy_out1_, bo_copy_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("copy kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_copy_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(dst, bo_copy_out1_.map<std::bfloat16_t *>(), ELEMS * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// run() — mirrors conv2d() in preprocessing_bf16.py.
//
// Context schedule per embed dim i:
//   { conv ctx }  allocate conv instr BO, run 12 invocations, ctx destroyed
//   { add  ctx }  allocate add  instr BO, run  3 invocations, ctx destroyed
//   { copy ctx }  allocate copy instr BO, run  4 invocations, ctx destroyed
// Total context opens: 3 × EMBD_DIM = 2304.
// ---------------------------------------------------------------------------
void Preprocessor::run(
    const std::bfloat16_t *image,
    const std::bfloat16_t *kernel,
    std::bfloat16_t       *output)
{
    constexpr int PS  = PREPROC_PATCHES_SIDE; // 32
    constexpr int KD  = PREPROC_KERNEL_DIM;   // 16
    constexpr int OD  = PREPROC_OUTPUT_DIM;   // 16
    constexpr int ID  = PREPROC_INPUT_DIM;    // 256
    constexpr int PL  = PREPROC_PIX_LEN;      // 512
    constexpr int C   = PREPROC_CHANNELS;     // 3
    constexpr int OC  = PREPROC_EMBD_DIM;     // 768
    constexpr int TS  = PREPROC_TILE_SIDE;    // 2

    npu_us_ = 0.0;
    std::fill(embd_.begin(),  embd_.end(),  std::bfloat16_t(0.0f));
    std::fill(output, output + PREPROC_OUTPUT_ELEMS, std::bfloat16_t(0.0f));

    for (int i = 0; i < OC; ++i) {

        // ==============================================================
        // Phase A — convolutions for embed dim i
        // Open one conv context; allocate instr BO inside it; run 12 calls.
        // ==============================================================
        {
            xrt::hw_context ctx(device_, conv_spec_.xclbin_obj.get_uuid());
            xrt::kernel     conv_k(ctx, conv_spec_.kernel_name);
            xrt::bo         bo_instr = make_instr_bo(device_, conv_k, conv_spec_.instr);
            int             n_instr  = (int)conv_spec_.instr.size();

            for (int j = 0; j < C; ++j) {
                std::bfloat16_t *tmp_j = &tmp_per_ch_[(size_t)j * PS * PS];
                std::fill(tmp_j, tmp_j + PS * PS, std::bfloat16_t(0.0f));

                for (int k = 0; k < TS; ++k) {
                    for (int l = 0; l < TS; ++l) {
                        for (int row = 0; row < ID; ++row) {
                            size_t src_off = (size_t)j * PL * PL
                                           + (size_t)(k * ID + row) * PL
                                           + l * ID;
                            memcpy(&a_tile_[row * ID], &image[src_off],
                                   ID * sizeof(std::bfloat16_t));
                        }
                        size_t b_off = ((size_t)i * C + j) * KD * KD;
                        exec_conv(conv_k, bo_instr, n_instr,
                                  a_tile_.data(), &kernel[b_off],
                                  conv_tile_out_.data());

                        for (int row = 0; row < OD; ++row) {
                            size_t dst_off = (size_t)(k * OD + row) * PS + l * OD;
                            memcpy(&tmp_j[dst_off],
                                   &conv_tile_out_[row * OD],
                                   OD * sizeof(std::bfloat16_t));
                        }
                    }
                }
            }
        } // conv ctx + instr BO destroyed

        // ==============================================================
        // Phase B — add: embd[:,:,i] += tmp[j] for each channel j
        // ==============================================================
        {
            xrt::hw_context ctx(device_, add_spec_.xclbin_obj.get_uuid());
            xrt::kernel     add_k(ctx, add_spec_.kernel_name);
            xrt::bo         bo_instr = make_instr_bo(device_, add_k, add_spec_.instr);
            int             n_instr  = (int)add_spec_.instr.size();

            for (int j = 0; j < C; ++j) {
                const std::bfloat16_t *tmp_j = &tmp_per_ch_[(size_t)j * PS * PS];

                for (int r = 0; r < PS * PS; ++r)
                    embd_slice_[r] = embd_[(size_t)r * OC + i];

                exec_add(add_k, bo_instr, n_instr,
                         embd_slice_.data(), tmp_j, add_result_.data());

                for (int r = 0; r < PS * PS; ++r)
                    embd_[(size_t)r * OC + i] = add_result_[r];
            }
        } // add ctx + instr BO destroyed

        // ==============================================================
        // Phase C — copy: embd[kk*8:(kk+1)*8, :, i] → output patches
        // ==============================================================
        {
            xrt::hw_context ctx(device_, copy_spec_.xclbin_obj.get_uuid());
            xrt::kernel     copy_k(ctx, copy_spec_.kernel_name);
            xrt::bo         bo_instr = make_instr_bo(device_, copy_k, copy_spec_.instr);
            int             n_instr  = (int)copy_spec_.instr.size();

            for (int kk = 0; kk < 4; ++kk) {
                for (int r = 0; r < 8; ++r)
                    for (int c = 0; c < PS; ++c)
                        copy_src_[r * PS + c] =
                            embd_[((size_t)(kk * 8 + r) * PS + c) * OC + i];

                exec_copy(copy_k, bo_instr, n_instr,
                          copy_src_.data(), copy_dst_.data());

                for (int p = 0; p < 256; ++p)
                    output[(size_t)(kk * 256 + p) * OC + i] = copy_dst_[p];
            }
        } // copy ctx + instr BO destroyed

    } // embed dim
}
