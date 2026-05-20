//=============================================================================
// Text encoder self-attention pipeline (no RoPE)
//
// Pipeline (mirrors text_encoder_bf16.py text_encoder_forward, attn only):
//   rms_norm × 8   (16-row tiles of the full [128,960] input)
//   gemm_q   × 1   (normed[128,960] @ Wq[960,960] → query[128,960])
//   gemm_kv  × 2   (normed @ Wk → key[128,320], normed @ Wv → value[128,320])
//   per head h in 0..14:
//     CPU: extract Q_head[128,64] and transpose K_head_T[64,128]
//     gemm_attn_score × 1   (Q_head @ K_head_T → score[128,128])
//     masked_softmax  × 16  (8-row tiles → weight[128,128])
//     gemm_attn_value × 1   (weight[128,128] @ V_head[128,64] → ctx[128,64])
//     CPU: assemble ctx_head into ctx_full[128,960]
//   gemm_out × 1   (ctx_full[128,960] @ Wo[960,960] → output[128,960])
//
// Run from inside attn.prj/:
//   ./build/attn --input x.data --W_norm norm.data \
//                --Wq Wq.data --Wk Wk.data --Wv Wv.data --Wo Wo.data \
//                --output out.data
//=============================================================================

#include <boost/program_options.hpp>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;

// ============================================================
// Model dimensions (text_encoder_bf16)
// ============================================================
static constexpr int SEQ          = 128;
static constexpr int EMBD         = 960;
static constexpr int Q_H          = 15;
static constexpr int KV_H         = 5;
static constexpr int HEAD_DIM     = 64;
static constexpr int NORM_TILE    = 16;   // rms_norm processes 16 rows at a time
static constexpr int SOFTMAX_TILE = 8;    // masked_softmax processes 8 rows at a time

// ============================================================
// Buffer sizes in bytes (all bf16 = 2 bytes unless noted)
// ============================================================
static constexpr size_t SZ_NORM_TILE  = NORM_TILE   * EMBD * 2;         //  30720: [16,960]  bf16
static constexpr size_t SZ_NORM_W     = EMBD * 2;                       //   1920: [960]     bf16
static constexpr size_t SZ_FULL       = SEQ  * EMBD * 2;                // 245760: [128,960] bf16
static constexpr size_t SZ_KV_OUT     = SEQ  * KV_H * HEAD_DIM * 2;    //  81920: [128,320] bf16
static constexpr size_t SZ_WQ         = EMBD * Q_H * HEAD_DIM * 2;     //1843200: [960,960] bf16
static constexpr size_t SZ_WKV        = EMBD * KV_H * HEAD_DIM * 2;    // 614400: [960,320] bf16
static constexpr size_t SZ_Q_HEAD     = SEQ  * HEAD_DIM * 2;            //  16384: [128,64]  bf16
static constexpr size_t SZ_K_HEAD_T   = HEAD_DIM * SEQ * 2;             //  16384: [64,128]  bf16
static constexpr size_t SZ_SCORE      = SEQ  * SEQ * 2;                 //  32768: [128,128] bf16
static constexpr size_t SZ_SCORE_TILE = SOFTMAX_TILE * SEQ * 2;         //   2048: [8,128]   bf16
static constexpr size_t SZ_ROW_START  = sizeof(int32_t);                //      4: scalar int32
static constexpr size_t SZ_V_HEAD     = SEQ  * HEAD_DIM * 2;            //  16384: [128,64]  bf16
static constexpr size_t SZ_CTX_HEAD   = SEQ  * HEAD_DIM * 2;            //  16384: [128,64]  bf16

// ============================================================
// KernelSpec: xclbin + instructions, registered but no context
// NPU only supports one hw_context at a time, so contexts are
// created on-demand (see ActiveKernel below).
// ============================================================
struct KernelSpec {
    xrt::xclbin          xclbin_obj;
    xrt::uuid            uuid;
    std::string          kernel_name;
    std::vector<uint32_t> instr;

    void preload(xrt::device &dev, const std::string &xclbin_path,
                 const std::string &insts_path) {
        xclbin_obj  = xrt::xclbin(xclbin_path);
        uuid        = xclbin_obj.get_uuid();
        dev.register_xclbin(xclbin_obj);

        auto xks = xclbin_obj.get_kernels();
        auto xk  = *std::find_if(xks.begin(), xks.end(),
            [](xrt::xclbin::kernel &k){ return k.get_name().rfind("MLIR_AIE",0)==0; });
        kernel_name = xk.get_name();

        std::ifstream f(insts_path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open: " << insts_path << "\n"; exit(1); }
        f.seekg(0, std::ios::end);
        size_t nb = f.tellg(); f.seekg(0);
        instr.resize(nb / 4);
        f.read(reinterpret_cast<char*>(instr.data()), nb);
    }
};

// ============================================================
// ActiveKernel: RAII wrapper — creates hw_context on construction,
// destroys it on destruction.  Only one may exist at a time.
// ============================================================
struct ActiveKernel {
    xrt::hw_context ctx;
    xrt::kernel     k;
    xrt::bo         bo_instr;
    int             instr_size;

    ActiveKernel(xrt::device &dev, KernelSpec &spec) {
        ctx = xrt::hw_context(dev, spec.uuid);
        k   = xrt::kernel(ctx, spec.kernel_name);

        size_t nb = spec.instr.size() * sizeof(uint32_t);
        bo_instr = xrt::bo(dev, nb, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
        memcpy(bo_instr.map<void*>(), spec.instr.data(), nb);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        instr_size = (int)spec.instr.size();
    }

    void run3(xrt::bo &b3, xrt::bo &b4, xrt::bo &b5) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.set_arg(5, b5);
        r.start();
        r.wait();
    }
};

// ============================================================
// Helpers
// ============================================================
static xrt::bo make_bo(xrt::device &dev, int group_id, size_t bytes) {
    return xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, group_id);
}

static void load_file_to_bo(xrt::bo &bo, const std::string &path, size_t expected_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; exit(1); }
    f.seekg(0, std::ios::end);
    if ((size_t)f.tellg() != expected_bytes) {
        std::cerr << path << ": expected " << expected_bytes << " bytes, got " << f.tellg() << "\n";
        exit(1);
    }
    f.seekg(0);
    f.read(bo.map<char*>(), expected_bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// Extract a contiguous head slice from a packed [SEQ, n_heads*HEAD_DIM] bf16 buffer
static void extract_head(uint16_t *dst, const uint16_t *src, int head, int n_heads) {
    for (int s = 0; s < SEQ; s++)
        memcpy(dst + s * HEAD_DIM,
               src + s * n_heads * HEAD_DIM + head * HEAD_DIM,
               HEAD_DIM * sizeof(uint16_t));
}

// Transpose [SEQ, HEAD_DIM] → [HEAD_DIM, SEQ] bf16
static void transpose_head(uint16_t *dst, const uint16_t *src) {
    for (int s = 0; s < SEQ; s++)
        for (int d = 0; d < HEAD_DIM; d++)
            dst[d * SEQ + s] = src[s * HEAD_DIM + d];
}

// ============================================================
// Main
// ============================================================
int main(int argc, const char *argv[]) {
    po::options_description opts("Text encoder attention pipeline");
    opts.add_options()
        ("help,h",    "help")
        ("input",     po::value<std::string>()->required(), "input activations [128,960] bf16")
        ("W_norm",    po::value<std::string>()->required(), "rms_norm weight [960] bf16")
        ("Wq",        po::value<std::string>()->required(), "Q projection weight [960,960] bf16")
        ("Wk",        po::value<std::string>()->required(), "K projection weight [960,320] bf16")
        ("Wv",        po::value<std::string>()->required(), "V projection weight [960,320] bf16")
        ("Wo",        po::value<std::string>()->required(), "output projection weight [960,960] bf16")
        ("output",    po::value<std::string>()->default_value("output.data"), "output [128,960] bf16")
        ("iters,n",   po::value<int>()->default_value(1),  "number of forward passes to run")
        ("verbosity,v", po::value<int>()->default_value(0), "verbosity");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, opts), vm);
        if (vm.count("help")) { std::cout << opts; return 0; }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n" << opts; return 1;
    }

    int verbosity = vm["verbosity"].as<int>();
    int n_iters   = vm["iters"].as<int>();

    // --------------------------------------------------------
    // Paths to each kernel's compiled artefacts
    // (relative to attn.prj/ working directory)
    // --------------------------------------------------------
    const std::string BASE = "..";
    auto xclbin_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/build/final.xclbin";
    };
    auto insts_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/insts.txt";
    };

    // --------------------------------------------------------
    // Open device and pre-load all kernel specs
    // (register xclbins; no hw_context yet)
    // --------------------------------------------------------
    if (verbosity >= 1) std::cout << "Preloading kernel specs...\n";
    auto device = xrt::device(0);

    KernelSpec spec_rms_norm, spec_gemm_q, spec_gemm_kv;
    KernelSpec spec_attn_score, spec_masked_softmax, spec_attn_value;
    KernelSpec spec_gemm_out;

    spec_rms_norm.preload     (device, xclbin_path("rms_norm"),        insts_path("rms_norm"));
    spec_gemm_q.preload       (device, xclbin_path("gemm_q"),          insts_path("gemm_q"));
    spec_gemm_kv.preload      (device, xclbin_path("gemm_kv"),         insts_path("gemm_kv"));
    spec_attn_score.preload   (device, xclbin_path("gemm_attn_score"), insts_path("gemm_attn_score"));
    spec_masked_softmax.preload(device, xclbin_path("masked_softmax"), insts_path("masked_softmax"));
    spec_attn_value.preload   (device, xclbin_path("gemm_attn_value"), insts_path("gemm_attn_value"));
    spec_gemm_out.preload     (device, xclbin_path("gemm_out"),        insts_path("gemm_out"));

    if (verbosity >= 1) std::cout << "All specs loaded.\n";

    // --------------------------------------------------------
    // Discover memory group IDs by briefly activating one kernel.
    // For AMD NPU all data slots (3,4,5) share the same group.
    // --------------------------------------------------------
    int g3, g4, g5;
    {
        ActiveKernel tmp(device, spec_rms_norm);
        g3 = tmp.k.group_id(3);
        g4 = tmp.k.group_id(4);
        g5 = tmp.k.group_id(5);
    }
    if (verbosity >= 1)
        std::cout << "Data group IDs: slot3=" << g3 << " slot4=" << g4 << " slot5=" << g5 << "\n";

    // --------------------------------------------------------
    // Allocate all BOs (no active context required after discovery)
    // --------------------------------------------------------

    // --- rms_norm: (slot3=tile_in, slot4=tile_out, slot5=weight) ---
    auto bo_rms_tile_in  = make_bo(device, g3, SZ_NORM_TILE);
    auto bo_rms_tile_out = make_bo(device, g4, SZ_NORM_TILE);
    auto bo_W_norm       = make_bo(device, g5, SZ_NORM_W);

    // --- gemm_q: (slot3=normed, slot4=query, slot5=Wq) ---
    auto bo_normed = make_bo(device, g3, SZ_FULL);
    auto bo_query  = make_bo(device, g4, SZ_FULL);
    auto bo_Wq     = make_bo(device, g5, SZ_WQ);

    // --- gemm_kv: (slot3=normed reused, slot4=kv_out, slot5=weight) ---
    auto bo_key  = make_bo(device, g4, SZ_KV_OUT);
    auto bo_val  = make_bo(device, g4, SZ_KV_OUT);
    auto bo_Wk   = make_bo(device, g5, SZ_WKV);
    auto bo_Wv   = make_bo(device, g5, SZ_WKV);

    // --- gemm_attn_score: (slot3=Q_head, slot4=score, slot5=K_head_T) ---
    auto bo_Q_head   = make_bo(device, g3, SZ_Q_HEAD);
    auto bo_score    = make_bo(device, g4, SZ_SCORE);
    auto bo_K_head_T = make_bo(device, g5, SZ_K_HEAD_T);

    // --- masked_softmax: (slot3=score_tile, slot4=row_start, slot5=weight_tile) ---
    auto bo_score_tile  = make_bo(device, g3, SZ_SCORE_TILE);
    auto bo_row_start   = make_bo(device, g4, SZ_ROW_START);
    auto bo_weight_tile = make_bo(device, g5, SZ_SCORE_TILE);

    // --- gemm_attn_value: (slot3=attn_weight, slot4=ctx_head, slot5=V_head) ---
    auto bo_attn_weight = make_bo(device, g3, SZ_SCORE);
    auto bo_ctx_head    = make_bo(device, g4, SZ_CTX_HEAD);
    auto bo_V_head      = make_bo(device, g5, SZ_V_HEAD);

    // --- gemm_out: (slot3=ctx_full, slot4=output, slot5=Wo) ---
    auto bo_ctx_full = make_bo(device, g3, SZ_FULL);
    auto bo_output   = make_bo(device, g4, SZ_FULL);
    auto bo_Wo       = make_bo(device, g5, SZ_WQ);

    // --- input BO ---
    auto bo_input = make_bo(device, g3, SZ_FULL);

    // --------------------------------------------------------
    // Load weights (constant across forward passes)
    // --------------------------------------------------------
    if (verbosity >= 1) std::cout << "Loading weights...\n";
    load_file_to_bo(bo_W_norm, vm["W_norm"].as<std::string>(), SZ_NORM_W);
    load_file_to_bo(bo_Wq,     vm["Wq"].as<std::string>(),     SZ_WQ);
    load_file_to_bo(bo_Wk,     vm["Wk"].as<std::string>(),     SZ_WKV);
    load_file_to_bo(bo_Wv,     vm["Wv"].as<std::string>(),     SZ_WKV);
    load_file_to_bo(bo_Wo,     vm["Wo"].as<std::string>(),     SZ_WQ);

    // CPU-side staging buffers for assembling intermediate results
    std::vector<uint16_t> normed_host(SEQ * EMBD, 0);
    std::vector<uint16_t> key_host   (SEQ * KV_H * HEAD_DIM, 0);
    std::vector<uint16_t> val_host   (SEQ * KV_H * HEAD_DIM, 0);
    std::vector<uint16_t> attn_weight_host(SEQ * SEQ, 0);
    std::vector<uint16_t> ctx_full_host(SEQ * Q_H * HEAD_DIM, 0);

    // --------------------------------------------------------
    // Forward pass — each kernel type activates/deactivates
    // its own hw_context in sequence (only one active at a time)
    // --------------------------------------------------------
    auto run_attention = [&]() {

        // ---- Step 1: Load input ----
        load_file_to_bo(bo_input, vm["input"].as<std::string>(), SZ_FULL);
        uint16_t *input_map = bo_input.map<uint16_t*>();

        // ---- Step 2: rms_norm × 8 (16-row tiles) ----
        {
            ActiveKernel ak(device, spec_rms_norm);
            uint16_t *rms_in_map  = bo_rms_tile_in.map<uint16_t*>();
            uint16_t *rms_out_map = bo_rms_tile_out.map<uint16_t*>();
            for (int t = 0; t < SEQ / NORM_TILE; t++) {
                memcpy(rms_in_map, input_map + t * NORM_TILE * EMBD, SZ_NORM_TILE);
                bo_rms_tile_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_rms_tile_in, bo_rms_tile_out, bo_W_norm);
                bo_rms_tile_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(normed_host.data() + t * NORM_TILE * EMBD, rms_out_map, SZ_NORM_TILE);
            }
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_FULL);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Step 3: gemm_q → query[128,960] ----
        {
            ActiveKernel ak(device, spec_gemm_q);
            ak.run3(bo_normed, bo_query, bo_Wq);
        }
        bo_query.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // ---- Step 4: gemm_kv → key[128,320] ----
        {
            ActiveKernel ak(device, spec_gemm_kv);
            ak.run3(bo_normed, bo_key, bo_Wk);
            bo_key.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(key_host.data(), bo_key.map<uint16_t*>(), SZ_KV_OUT);

            // ---- Step 5: gemm_kv → value[128,320] ----
            ak.run3(bo_normed, bo_val, bo_Wv);
        }
        bo_val.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(val_host.data(), bo_val.map<uint16_t*>(), SZ_KV_OUT);

        uint16_t *query_map = bo_query.map<uint16_t*>();

        // ---- Step 6: per-head attention (×15) ----
        for (int h = 0; h < Q_H; h++) {
            int kv_idx = h * KV_H / Q_H;

            // CPU: extract Q_head[128,64] from query[128,960]
            extract_head(bo_Q_head.map<uint16_t*>(), query_map, h, Q_H);
            bo_Q_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // CPU: extract K_head[128,64] from key[128,320] and transpose → [64,128]
            std::vector<uint16_t> K_head_tmp(SEQ * HEAD_DIM);
            extract_head(K_head_tmp.data(), key_host.data(), kv_idx, KV_H);
            transpose_head(bo_K_head_T.map<uint16_t*>(), K_head_tmp.data());
            bo_K_head_T.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_attn_score → score[128,128]
            {
                ActiveKernel ak(device, spec_attn_score);
                ak.run3(bo_Q_head, bo_score, bo_K_head_T);
            }
            bo_score.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            uint16_t *score_map = bo_score.map<uint16_t*>();

            // NPU: masked_softmax × 16 (8-row tiles) → attn_weight[128,128]
            {
                ActiveKernel ak(device, spec_masked_softmax);
                uint16_t *score_tile_map  = bo_score_tile.map<uint16_t*>();
                uint16_t *weight_tile_map = bo_weight_tile.map<uint16_t*>();
                for (int t = 0; t < SEQ / SOFTMAX_TILE; t++) {
                    memcpy(score_tile_map, score_map + t * SOFTMAX_TILE * SEQ, SZ_SCORE_TILE);
                    bo_score_tile.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    *bo_row_start.map<int32_t*>() = t * SOFTMAX_TILE;
                    bo_row_start.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    // masked_softmax slot order: out is at slot5 (weight_tile)
                    ak.run3(bo_score_tile, bo_row_start, bo_weight_tile);
                    bo_weight_tile.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    memcpy(attn_weight_host.data() + t * SOFTMAX_TILE * SEQ,
                           weight_tile_map, SZ_SCORE_TILE);
                }
            }
            memcpy(bo_attn_weight.map<uint16_t*>(), attn_weight_host.data(), SZ_SCORE);
            bo_attn_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // CPU: extract V_head[128,64] from value[128,320]
            extract_head(bo_V_head.map<uint16_t*>(), val_host.data(), kv_idx, KV_H);
            bo_V_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_attn_value → ctx_head[128,64]
            {
                ActiveKernel ak(device, spec_attn_value);
                ak.run3(bo_attn_weight, bo_ctx_head, bo_V_head);
            }
            bo_ctx_head.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // CPU: assemble ctx_head into ctx_full[128, Q_H*HEAD_DIM]
            uint16_t *ctx_head_map = bo_ctx_head.map<uint16_t*>();
            for (int s = 0; s < SEQ; s++)
                memcpy(ctx_full_host.data() + s * Q_H * HEAD_DIM + h * HEAD_DIM,
                       ctx_head_map + s * HEAD_DIM,
                       HEAD_DIM * sizeof(uint16_t));
        }

        // ---- Step 7: gemm_out → output[128,960] ----
        memcpy(bo_ctx_full.map<uint16_t*>(), ctx_full_host.data(), SZ_FULL);
        bo_ctx_full.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {
            ActiveKernel ak(device, spec_gemm_out);
            ak.run3(bo_ctx_full, bo_output, bo_Wo);
        }

    }; // end run_attention

    // --------------------------------------------------------
    // Warmup + timed runs
    // --------------------------------------------------------
    if (verbosity >= 1) std::cout << "Warmup run...\n";
    run_attention();  // warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) run_attention();
    auto t1 = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
    std::cout << "Attention forward pass: " << total_ms / n_iters << " ms/iter"
              << " (avg over " << n_iters << " iters)\n";

    // --------------------------------------------------------
    // Write output
    // --------------------------------------------------------
    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::ofstream out(vm["output"].as<std::string>(), std::ios::binary);
    out.write(bo_output.map<char*>(), SZ_FULL);
    if (verbosity >= 1) std::cout << "Output written to " << vm["output"].as<std::string>() << "\n";

    return 0;
}
