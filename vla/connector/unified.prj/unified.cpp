// unified.cpp — connector pipeline for AMD NPU (OPT-B: pixel shuffle).
//
// OPT-B optimization: Replace 256 exec_copy calls with 32 exec_pixel_shuffle calls.
//   - Pre-gather on CPU: reorder A rows so pixel_shuffle NPU kernel gets its 8 rows
//   - Input to NPU: [32, 768] (4 groups × 8 rows each)
//   - Output from NPU: [8, 3072] (8 output rows × 3072 cols)
//   - 4 col-chunks × 8 row-batches = 32 total pixel_shuffle calls (was 256)
//
// Accumulation strategy:
//   NPU GEMM outputs bfloat16. Each bf16 GEMM tile is upcast to float32 on
//   CPU and accumulated into out[]. Matches connector_bf16.py exactly.
//
// Context schedule:
//   Phase 1 — pixel shuffle (OPT-B):
//     { ps ctx }   allocate ps instr BO, run 32 pixel_shuffle calls
//   Phase 2 — tiled GEMM + CPU float32 accumulate:
//     { gemm ctx } allocate gemm instr BO once, run all 480 GEMM calls
//                  (m=2 × n=15 × k=16 = 480)
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
    const std::string & xclbin_pixel_shuffle,
    const std::string & instr_pixel_shuffle,
    const std::string & xclbin_gemm,
    const std::string & instr_gemm,
    int trace_size,
    int verbosity)
    : verbosity_(verbosity)
    , trace_size_(trace_size)
    , device_(device_index)
    , A_shuf_(CONN_NEW_SEQ * CONN_NEW_EMBD)
    , ps_in_buf_(CONN_PS_IN_ELEMS)
    , ps_out_buf_(CONN_PS_OUT_ELEMS)
    , a_tile_(CONN_GEMM_A_ELEMS)
    , b_tile_(CONN_GEMM_B_ELEMS)
    , c_tile_(CONN_GEMM_C_ELEMS)
{
    ps_spec_.preload(device_, xclbin_pixel_shuffle, instr_pixel_shuffle);
    gemm_spec_.preload(device_, xclbin_gemm, instr_gemm);
    if (verbosity_ >= 1) std::cout << "[Connector] Kernel specs loaded.\n";

    // Discover data group IDs
    int g3, g4, g5, g7;
    {
        xrt::hw_context tmp_ctx(device_, ps_spec_.xclbin_obj.get_uuid());
        xrt::kernel     tmp_k(tmp_ctx, ps_spec_.kernel_name);
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

    // Pixel shuffle BOs: input [32,768]=24576 elems, output [8,3072]=24576 elems
    bo_ps_in0_   = xrt::bo(device_, CONN_PS_IN_ELEMS  * sizeof(std::bfloat16_t),
                           XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_ps_out1_  = xrt::bo(device_, CONN_PS_OUT_ELEMS * sizeof(std::bfloat16_t),
                           XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_ps_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_ps_trace_.map<char *>(), 0, trace_size_);

    bo_gemm_in0_   = xrt::bo(device_, CONN_GEMM_A_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g3);
    bo_gemm_out1_  = xrt::bo(device_, CONN_GEMM_C_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g4);
    bo_gemm_in2_   = xrt::bo(device_, CONN_GEMM_B_ELEMS * sizeof(std::bfloat16_t),
                             XRT_BO_FLAGS_HOST_ONLY, g5);
    bo_gemm_trace_ = xrt::bo(device_, (size_t)tmp_trace * 4, XRT_BO_FLAGS_HOST_ONLY, g7);
    if (trace_size_ > 0) memset(bo_gemm_trace_.map<char *>(), 0, trace_size_);

    if (verbosity_ >= 1) std::cout << "[Connector] Data BOs allocated.\n";

    // ----------------------------------------------------------------
    // Precompute PIXEL_ROW_IDX[j][i*4+r] and PS_ROW_IDX[j][b][g][r]
    //
    // From connector_bf16.py:
    //   _PIXEL_ROW_IDX = np.array(
    //       [[(i // 8) * 128 + (i % 8) * 4 + j * 32 + r
    //         for i in range(NEW_SEQ) for r in range(4)]
    //        for j in range(4)],
    //       dtype=np.intp,
    //   )  # shape [4, 256]
    //
    //   for _j in range(4):
    //     for _b in range(8):
    //       for _g in range(4):
    //         _PS_ROW_IDX[_j, _b, _g, :] = _PIXEL_ROW_IDX[_j][_g::4][_b * 8:(_b + 1) * 8]
    // ----------------------------------------------------------------
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < CONN_NEW_SEQ; ++i) {
            for (int r = 0; r < 4; ++r) {
                PIXEL_ROW_IDX[j][i * 4 + r] =
                    (i / 8) * 128 + (i % 8) * 4 + j * 32 + r;
            }
        }
    }

    // PS_ROW_IDX[j][b][g][r] = PIXEL_ROW_IDX[j][g::4][b*8:(b+1)*8][r]
    // i.e., from the flat 256-element PIXEL_ROW_IDX[j], pick every 4th element
    // starting at g, then slice [b*8 : (b+1)*8].
    // PIXEL_ROW_IDX[j] has 256 elements (64 i-values × 4 r-values).
    // Indexing _g::4 from start 0 gives positions: g, g+4, g+8, ..., g+252 (64 elements).
    // Then _b*8 : (_b+1)*8 picks 8 of those.
    for (int j = 0; j < 4; ++j) {
        for (int b = 0; b < 8; ++b) {
            for (int g = 0; g < 4; ++g) {
                for (int r = 0; r < 8; ++r) {
                    // Position in _g::4 slice: b*8 + r → flat index g + (b*8 + r)*4
                    int flat_idx = g + (b * 8 + r) * 4;
                    PS_ROW_IDX[j][b][g][r] = PIXEL_ROW_IDX[j][flat_idx];
                }
            }
        }
    }

    if (verbosity_ >= 1) std::cout << "[Connector] Row index tables precomputed.\n";
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
// exec_pixel_shuffle — pixel_shuffle kernel (2 slots: slot3=in, slot4=out)
//   src: [32, 768] bf16 pre-gathered input (4 groups × 8 rows × 768 cols)
//   dst: [8, 3072] bf16 output (8 output rows × 4 groups × 768 cols)
// ---------------------------------------------------------------------------
void Connector::exec_pixel_shuffle(
    xrt::kernel           &k,
    xrt::bo               &bo_instr,
    int                    instr_size,
    const std::bfloat16_t *src,
    std::bfloat16_t       *dst)
{
    memcpy(bo_ps_in0_.map<std::bfloat16_t *>(), src,
           CONN_PS_IN_ELEMS * sizeof(std::bfloat16_t));
    bo_ps_in0_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size_ > 0) bo_ps_trace_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t0  = clk::now();
    // pixel_shuffle is a 2-slot kernel (slot3=in, slot4=out), no slot5
    auto run = k(3u, bo_instr, instr_size,
                 bo_ps_in0_, bo_ps_out1_, bo_ps_trace_);
    if (run.wait() != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("pixel_shuffle kernel did not complete");
    npu_us_ += elapsed_us(t0, clk::now());

    bo_ps_out1_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(dst, bo_ps_out1_.map<std::bfloat16_t *>(),
           CONN_PS_OUT_ELEMS * sizeof(std::bfloat16_t));
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
// OPT-B Phase 1: pixel shuffle uses 32 calls instead of 256.
// Phase 2: tiled GEMM + float32 accumulation (unchanged).
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
    // Phase 1 — Pixel shuffle (OPT-B): A[SEQ=1024, EMBD=768] → A_shuf_[NEW_SEQ=64, NEW_EMBD=12288]
    //
    // Row index formula (from connector_bf16.py _PS_ROW_IDX):
    //   For col_chunk j in [0,4), row_batch b in [0,8), group g in [0,4), row r in [0,8):
    //     src_row = PS_ROW_IDX[j][b][g][r]
    //
    // Pre-gather: ps_in_buf_[g*8+r, :] = A[src_row, :]
    // NPU output: ps_out_buf_[r, g*EMBD:(g+1)*EMBD] = pixel_shuffle result
    //
    // Scatter into A_shuf_:
    //   A_shuf_[b*8+r, j*NEW_EMBD/4 + g*EMBD : j*NEW_EMBD/4 + (g+1)*EMBD] = ps_out_buf_[r, g*EMBD:(g+1)*EMBD]
    // ==================================================================
    if (verbosity_ >= 1) std::cout << "[Connector] Phase 1: OPT-B pixel shuffle (32 calls).\n";

    // A_shuf_: NEW_EMBD = SHUFFLE_CORES * EMBD = 4 * 768 = 3072 per "super-column"
    // Full A_shuf_ layout: [NEW_SEQ=64, NEW_EMBD=12288]
    // j-th chunk of columns: j * (NEW_EMBD/CONN_N_COL_CHUNKS) = j * 3072
    const int CHUNK_COLS = CONN_NEW_EMBD / CONN_N_COL_CHUNKS;  // 3072

    {
        xrt::hw_context ctx(device_, ps_spec_.xclbin_obj.get_uuid());
        xrt::kernel     ps_k(ctx, ps_spec_.kernel_name);
        xrt::bo         bo_instr = make_instr_bo(device_, ps_k, ps_spec_.instr);
        int             n_instr  = (int)ps_spec_.instr.size();

        for (int j = 0; j < CONN_N_COL_CHUNKS; ++j) {
            for (int b = 0; b < CONN_N_ROW_BATCHES; ++b) {
                // --- CPU pre-gather into ps_in_buf_ [32, 768] ---
                // For group g in [0,4), row r in [0,8):
                //   ps_in_buf_[g*8+r, :] = A[PS_ROW_IDX[j][b][g][r], :]
                std::bfloat16_t *in_ptr = ps_in_buf_.data();
                for (int g = 0; g < CONN_SHUFFLE_CORES; ++g) {
                    for (int r = 0; r < CONN_SHUFFLE_BATCH; ++r) {
                        int src_row = PS_ROW_IDX[j][b][g][r];
                        const std::bfloat16_t *a_row = &A[(size_t)src_row * CONN_EMBD];
                        std::bfloat16_t *dst_row = &in_ptr[(g * CONN_SHUFFLE_BATCH + r) * CONN_EMBD];
                        memcpy(dst_row, a_row, CONN_EMBD * sizeof(std::bfloat16_t));
                    }
                }

                // --- NPU pixel shuffle ---
                exec_pixel_shuffle(ps_k, bo_instr, n_instr,
                                   ps_in_buf_.data(), ps_out_buf_.data());

                // --- Scatter output into A_shuf_ ---
                // ps_out_buf_: [SHUFFLE_BATCH=8, SHUFFLE_CORES*EMBD=3072]
                // Core g contributes columns [g*EMBD, (g+1)*EMBD) of each output row.
                // A_shuf_ target: rows [b*8, (b+1)*8], cols [j*3072 + g*768, j*3072 + (g+1)*768)
                for (int r = 0; r < CONN_SHUFFLE_BATCH; ++r) {
                    int out_row = b * CONN_SHUFFLE_BATCH + r;
                    const std::bfloat16_t *out_row_ptr =
                        &ps_out_buf_[r * CONN_SHUFFLE_CORES * CONN_EMBD];
                    std::bfloat16_t *a_shuf_row =
                        &A_shuf_[(size_t)out_row * CONN_NEW_EMBD + j * CHUNK_COLS];
                    memcpy(a_shuf_row, out_row_ptr,
                           (size_t)CONN_SHUFFLE_CORES * CONN_EMBD * sizeof(std::bfloat16_t));
                }
            }
        }
    } // ps ctx + instr BO destroyed

    if (verbosity_ >= 1) std::cout << "[Connector] Phase 1 done.\n";

    // ==================================================================
    // Phase 2 — Tiled GEMM + float32 accumulation on CPU
    //
    // Triple loop: m (2) × n (15) × k (16) = 480 GEMM calls.
    // One gemm hw_context is opened for ALL 480 calls.
    //
    // A_tile [M_TILE, K_TILE]: gathered contiguously from A_shuf_ column slice.
    // B_tile [K_TILE, N_TILE]: gathered contiguously from W 2D slice.
    // C_tile [M_TILE, N_TILE]: scattered row-by-row into out[].
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
                    // Gather A_tile [M_TILE, K_TILE]
                    for (int r = 0; r < CONN_M_TILE; ++r)
                        memcpy(&a_tile_[r * CONN_K_TILE],
                               &A_shuf_[(size_t)(m * CONN_M_TILE + r) * CONN_NEW_EMBD
                                        + k * CONN_K_TILE],
                               CONN_K_TILE * sizeof(std::bfloat16_t));

                    // Gather B_tile [K_TILE, N_TILE]
                    for (int r = 0; r < CONN_K_TILE; ++r)
                        memcpy(&b_tile_[r * CONN_N_TILE],
                               &W[(size_t)(k * CONN_K_TILE + r) * CONN_TEXT
                                  + n * CONN_N_TILE],
                               CONN_N_TILE * sizeof(std::bfloat16_t));

                    exec_gemm(gemm_k, bo_instr, n_instr,
                              a_tile_.data(), b_tile_.data(), c_tile_.data());

                    // Scatter-accumulate C_tile → out (float32)
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
