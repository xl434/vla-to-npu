#pragma once

#include <cstddef>
#include <cstdint>
#include <stdfloat>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// ---------------------------------------------------------------------------
// Problem dimensions — must match preprocessing_fused_bf16.py
// ---------------------------------------------------------------------------
static constexpr int FU_SEQ      = 1024;
static constexpr int FU_EMBD_DIM = 768;
static constexpr int FU_CHANNELS = 3;
static constexpr int FU_KH       = 16;
static constexpr int FU_KW       = 16;
static constexpr int FU_PIX_LEN  = 512;
static constexpr int FU_PH       = FU_PIX_LEN / FU_KH;   // 32 — patch rows
static constexpr int FU_PW       = FU_PIX_LEN / FU_KW;   // 32 — patches per row
static constexpr int FU_K_FULL   = FU_CHANNELS * FU_KH * FU_KW; // 768

// im2col tile dimensions (quarter-row granularity, single core)
static constexpr int FU_IMG2D  = FU_CHANNELS * FU_KH;    // 48
static constexpr int FU_PIX_QT = FU_PIX_LEN / 4;         // 128
static constexpr int FU_PW_QT  = FU_PW / 4;              // 8

// GEMM tile dimensions (Pn=4, N_TILE=128)
static constexpr int FU_M_TILE  = FU_PW;                 // 32
static constexpr int FU_N_TILE  = 128;                   // 32 * Pn(=4)
static constexpr int FU_N_TILES = FU_EMBD_DIM / FU_N_TILE; // 6

// Buffer element counts
static constexpr size_t FU_IM2COL_IN_ELEMS  = (size_t)FU_IMG2D  * FU_PIX_QT; // 6144
static constexpr size_t FU_IM2COL_OUT_ELEMS = (size_t)FU_PW_QT  * FU_K_FULL; // 6144
static constexpr size_t FU_GEMM_A_ELEMS     = (size_t)FU_M_TILE * FU_K_FULL; // 24576
static constexpr size_t FU_GEMM_B_ELEMS     = (size_t)FU_K_FULL * FU_N_TILE; // 98304
static constexpr size_t FU_GEMM_C_ELEMS     = (size_t)FU_M_TILE * FU_N_TILE; // 4096

static constexpr size_t FU_PATCHES_ELEMS    = (size_t)FU_SEQ * FU_K_FULL;    // 786432
static constexpr size_t FU_OUTPUT_ELEMS     = (size_t)FU_SEQ * FU_EMBD_DIM;  // 786432

// ---------------------------------------------------------------------------
// KernelSpec — registers xclbin + loads instructions without opening a
// hw_context (AMD NPU supports only ONE active hw_context at a time).
// ---------------------------------------------------------------------------
struct FuKernelSpec {
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
// Implements the im2col + GEMM fused conv2d patch-embedding pipeline from
// preprocessing_fused_bf16.py using two AIE kernel types.
//
// Call overhead reduction:
//   Python: 128 im2col calls + 192 GEMM calls = 320 hw_context interactions
//   C++:    1 im2col context (128 dispatches) + 1 GEMM context (192 dispatches)
//           = 2 hw_context opens total
//
// Data flow:
//   image[3,512,512] → image_strips[32,48,512] (CPU rearrange)
//   image_strips → im2col NPU × 128 → patches[1024,768] (DDR)
//   patches + kernel_B_tiles → GEMM NPU × 192 → output[1024,768]
// ---------------------------------------------------------------------------
class FusedPreprocessor {
public:
    // xclbin/instr paths are the compiled outputs from df.build():
    //   im2col_xclbin  : preprocessing/im2col.prj/build/final.xclbin
    //   im2col_instr   : preprocessing/im2col.prj/build/insts.bin
    //   gemm_xclbin    : preprocessing/fused_conv_add.prj/build/final.xclbin
    //   gemm_instr     : preprocessing/fused_conv_add.prj/build/insts.bin
    FusedPreprocessor(
        unsigned int        device_index,
        const std::string & im2col_xclbin,
        const std::string & im2col_instr,
        const std::string & gemm_xclbin,
        const std::string & gemm_instr,
        int verbosity = 0
    );

    ~FusedPreprocessor() = default;

    // Run full pipeline.
    //   image          : [FU_CHANNELS, FU_PIX_LEN, FU_PIX_LEN]  bfloat16, row-major
    //   kernel_B_tiles : [FU_N_TILES, FU_K_FULL, FU_N_TILE]     bfloat16 (pre-tiled B^T)
    //   output         : [FU_SEQ, FU_EMBD_DIM]                  bfloat16, row-major
    void run(
        const std::bfloat16_t * image,
        const std::bfloat16_t * kernel_B_tiles,
        std::bfloat16_t       * output
    );

    double im2col_time_us() const { return im2col_us_; }
    double gemm_time_us()   const { return gemm_us_;   }

private:
    int verbosity_;
    double im2col_us_ = 0.0, gemm_us_ = 0.0;

    xrt::device   device_;
    FuKernelSpec  im2col_spec_, gemm_spec_;

    // Persistent HOST_ONLY data BOs — reused across all dispatches within run()
    xrt::bo bo_im2col_in_;   // image_qt  [IMG2D, PIX_QT]
    xrt::bo bo_im2col_out_;  // patches_qt [PW_QT, K_FULL]
    xrt::bo bo_gemm_a_;      // A tile     [M_TILE, K_FULL]
    xrt::bo bo_gemm_b_;      // B tile     [K_FULL, N_TILE]
    xrt::bo bo_gemm_c_;      // C tile     [M_TILE, N_TILE]

    // Intermediate patches buffer [SEQ, K_FULL] — host-side DDR
    std::vector<std::bfloat16_t> patches_;

    // CPU image rearrangement: [3,512,512] → [32,48,512]
    std::vector<std::bfloat16_t> image_strips_;

    // Kernel dispatch helpers (called inside an already-open hw_context)
    void exec_im2col(xrt::kernel &k, xrt::bo &bo_instr, int instr_size,
                     int ph, int qt);
    void exec_gemm(xrt::kernel &k, xrt::bo &bo_instr, int instr_size,
                   const std::bfloat16_t *kernel_B_tiles,
                   std::bfloat16_t *output,
                   int m, int n);
};
