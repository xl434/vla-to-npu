// vision_encoder.cpp — Vision block transformer forward pass for AMD NPU.
//
// Build:
//   cd vision_block/unified.prj && mkdir -p build && cd build && cmake .. && make -j4
// Run (from vision_block/unified.prj/):
//   build/vision_encoder \
//     --input x.data --W_norm_1 w1.data --b_norm_1 b1.data \
//     --Wq wq.data --Wk wk.data --Wv wv.data --Wo wo.data \
//     --W_up wup.data --W_norm_2 w2.data --b_norm_2 b2.data \
//     --W_down wdown.data --output out.data -v 1
// Validate:
//   python3 -c "
//   import numpy as np
//   from ml_dtypes import bfloat16 as bf16
//   ref = np.fromfile('ref_out.data', dtype='uint16').view(bf16).astype(np.float32)
//   ours = np.fromfile('out.data', dtype='uint16').view(bf16).astype(np.float32)
//   print('max_err:', np.max(np.abs(ref - ours)))
//   print('allclose:', np.allclose(ref, ours, atol=1e-1, rtol=1e-1))
//   "
//
// Pipeline mirrors vision_block() in vision_block_bf16.py.
// Kernels: layer_norm_bf16, gemm_embd_embd_bf16, gemm_score_bf16, gemm_hid_embd_bf16,
//          gemm_head_seq_bf16, softmax_f32, gelu_bf16

#include <boost/program_options.hpp>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;

// ============================================================
// Model dimensions (vision_block_bf16)
// ============================================================
static constexpr int SEQ      = 1024;
static constexpr int EMBD     = 768;
static constexpr int N_HEAD   = 12;
static constexpr int HEAD_DIM = EMBD / N_HEAD;  // 64

static constexpr int FFN_HID          = EMBD * 4;   // 3072
static constexpr int FFN_DOWN_K_CHUNK = EMBD;        // 768 (reuse gemm_embd_embd)
static constexpr int FFN_DOWN_K_CHUNKS = FFN_HID / FFN_DOWN_K_CHUNK;  // 4

static constexpr int NORM_SEQ_TILE = 32;   // layer_norm processes 32 rows per call
static constexpr int GELU_SEQ_TILE = 32;   // gelu processes 32 rows per call

// Softmax physical layout: score[SEQ,SEQ] as [2*SEQ, SEQ/2] physical
static constexpr int SOFTMAX_PHYS_COLS     = SEQ / 2;       // 512
static constexpr int SOFTMAX_PHYS_ROWS_PER = 2;             // 2 physical rows per logical row
static constexpr int SOFTMAX_BATCH_PHYS    = 64;            // 64 physical rows per batch
static constexpr int SOFTMAX_NUM_BATCHES   = SEQ / (SOFTMAX_BATCH_PHYS / SOFTMAX_PHYS_ROWS_PER);  // 32

// Attention scale: 1/sqrt(HEAD_DIM=64) = 0.125
static constexpr float ATTN_SCALE = 0.125f;

// ============================================================
// Buffer sizes
// ============================================================
static constexpr size_t SZ_FULL      = (size_t)SEQ  * EMBD * 2;           // [1024,768] bf16
static constexpr size_t SZ_NORM_W    = EMBD * 2;                           // [768] bf16
static constexpr size_t SZ_NORM_TILE = (size_t)(NORM_SEQ_TILE + 1) * EMBD * 2; // packed: [33,768] bf16 (input+weight)
static constexpr size_t SZ_Q_HEAD   = (size_t)SEQ  * HEAD_DIM * 2;        // [1024,64] bf16
static constexpr size_t SZ_K_HEAD_T = (size_t)HEAD_DIM * SEQ * 2;         // [64,1024] bf16
static constexpr size_t SZ_SCORE    = (size_t)SEQ  * SEQ * 2;             // [1024,1024] bf16
static constexpr size_t SZ_SCORE_F32_TILE = (size_t)SOFTMAX_BATCH_PHYS * SOFTMAX_PHYS_COLS * 4;  // [64,512] f32
static constexpr size_t SZ_FFN_UP   = (size_t)SEQ  * FFN_HID * 2;        // [1024,3072] bf16
static constexpr size_t SZ_FFN_TILE = (size_t)GELU_SEQ_TILE * FFN_HID * 2;  // [32,3072] bf16
static constexpr size_t SZ_WQ       = (size_t)EMBD * EMBD * 2;            // [768,768] bf16
static constexpr size_t SZ_W_UP     = (size_t)EMBD * FFN_HID * 2;        // [768,3072] bf16
static constexpr size_t SZ_W_DOWN_CHUNK = (size_t)FFN_DOWN_K_CHUNK * EMBD * 2;  // [768,768] bf16

// ============================================================
// KernelSpec: xclbin + instructions, registered but no context
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
// ActiveKernel: RAII — creates hw_context, destroys on scope exit.
// Only one may exist at a time (AMD NPU constraint).
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

    // 2-slot kernel (softmax, gelu): slot3=in, slot4=out
    void run2(xrt::bo &b3, xrt::bo &b4) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.start();
        r.wait();
    }

    // 3-slot kernel (gemm, layer_norm): slot3=in0, slot4=out, slot5=in2
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

// bf16 raw → float32
static inline float bf16_to_f32(uint16_t raw) {
    uint32_t bits = (uint32_t)raw << 16;
    float val; memcpy(&val, &bits, 4);
    return val;
}

// float32 → bf16 raw
static inline uint16_t f32_to_bf16(float val) {
    uint32_t bits; memcpy(&bits, &val, 4);
    return (uint16_t)(bits >> 16);
}

// Extract contiguous head slice [SEQ, HEAD_DIM] from packed [SEQ, N_HEAD*HEAD_DIM]
static void extract_head(uint16_t *dst, const uint16_t *src, int head, int n_heads) {
    for (int s = 0; s < SEQ; s++)
        memcpy(dst + s * HEAD_DIM,
               src + s * n_heads * HEAD_DIM + head * HEAD_DIM,
               HEAD_DIM * sizeof(uint16_t));
}

// Transpose [SEQ, HEAD_DIM] → [HEAD_DIM, SEQ]
static void transpose_head(uint16_t *dst, const uint16_t *src) {
    for (int s = 0; s < SEQ; s++)
        for (int d = 0; d < HEAD_DIM; d++)
            dst[d * SEQ + s] = src[s * HEAD_DIM + d];
}

// Scale Q head in-place: bf16 → f32 → * scale → bf16
static void scale_bf16_inplace(uint16_t *buf, int n, float scale) {
    for (int i = 0; i < n; i++) {
        float v = bf16_to_f32(buf[i]) * scale;
        buf[i]  = f32_to_bf16(v);
    }
}

// Unpack bf16 score tile [SEQ,SEQ] → float32 physical tile [64,512] for softmax batch b
// Physical layout: each logical row r_log maps to 2 physical rows (left/right 512 cols)
static void unpack_score_tile(const uint16_t *score_bf16, float *score_f32_tile, int batch) {
    for (int rp = 0; rp < SOFTMAX_BATCH_PHYS; rp++) {
        int rl   = rp / 2;    // logical row within batch
        int half = rp % 2;    // 0 = left 512 cols, 1 = right 512 cols
        int global_rl = batch * (SOFTMAX_BATCH_PHYS / 2) + rl;
        for (int c = 0; c < SOFTMAX_PHYS_COLS; c++) {
            int global_c = half * SOFTMAX_PHYS_COLS + c;
            score_f32_tile[rp * SOFTMAX_PHYS_COLS + c] =
                bf16_to_f32(score_bf16[global_rl * SEQ + global_c]);
        }
    }
}

// Repack float32 physical tile [64,512] → bf16 weight [SEQ,SEQ] for batch b
static void repack_weight_tile(const float *weight_f32_tile, uint16_t *attn_weight_bf16, int batch) {
    for (int rp = 0; rp < SOFTMAX_BATCH_PHYS; rp++) {
        int rl   = rp / 2;
        int half = rp % 2;
        int global_rl = batch * (SOFTMAX_BATCH_PHYS / 2) + rl;
        for (int c = 0; c < SOFTMAX_PHYS_COLS; c++) {
            int global_c = half * SOFTMAX_PHYS_COLS + c;
            attn_weight_bf16[global_rl * SEQ + global_c] =
                f32_to_bf16(weight_f32_tile[rp * SOFTMAX_PHYS_COLS + c]);
        }
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, const char *argv[]) {
    po::options_description opts("Vision encoder forward pass");
    opts.add_options()
        ("help,h",     "help")
        ("input",      po::value<std::string>()->required(),            "[1024,768] bf16 input")
        ("W_norm_1",   po::value<std::string>()->required(),            "[768] bf16 layer norm weight 1")
        ("b_norm_1",   po::value<std::string>()->required(),            "[768] bf16 layer norm bias 1")
        ("Wq",         po::value<std::string>()->required(),            "[768,768] bf16")
        ("Wk",         po::value<std::string>()->required(),            "[768,768] bf16")
        ("Wv",         po::value<std::string>()->required(),            "[768,768] bf16")
        ("Wo",         po::value<std::string>()->required(),            "[768,768] bf16")
        ("W_up",       po::value<std::string>()->required(),            "[768,3072] bf16 FFN up weight")
        ("W_norm_2",   po::value<std::string>()->required(),            "[768] bf16 layer norm weight 2")
        ("b_norm_2",   po::value<std::string>()->required(),            "[768] bf16 layer norm bias 2")
        ("W_down",     po::value<std::string>()->required(),            "[3072,768] bf16 FFN down weight")
        ("output",     po::value<std::string>()->default_value("output.data"), "[1024,768] bf16 output")
        ("iters,n",    po::value<int>()->default_value(1),              "number of forward passes")
        ("verbosity,v",po::value<int>()->default_value(0),              "verbosity");

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

    // Paths to kernels (relative to vision_block/unified.prj/)
    const std::string BASE = "..";
    auto xclbin_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/build/final.xclbin";
    };
    auto insts_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/insts.txt";
    };

    if (verbosity >= 1) std::cout << "Preloading kernel specs...\n";
    auto device = xrt::device(0);

    KernelSpec spec_layer_norm, spec_gemm_embd, spec_gemm_hid, spec_gemm_head_seq;
    KernelSpec spec_gemm_score, spec_softmax, spec_gelu;

    spec_layer_norm.preload (device, xclbin_path("layer_norm_bf16"),    insts_path("layer_norm_bf16"));
    spec_gemm_embd.preload  (device, xclbin_path("gemm_embd_embd_bf16"),insts_path("gemm_embd_embd_bf16"));
    spec_gemm_hid.preload   (device, xclbin_path("gemm_hid_embd_bf16"), insts_path("gemm_hid_embd_bf16"));
    spec_gemm_head_seq.preload(device,xclbin_path("gemm_head_seq_bf16"),insts_path("gemm_head_seq_bf16"));
    spec_gemm_score.preload (device, xclbin_path("gemm_score_bf16"),    insts_path("gemm_score_bf16"));
    spec_softmax.preload    (device, xclbin_path("softmax_f32"),        insts_path("softmax_f32"));
    spec_gelu.preload       (device, xclbin_path("gelu_bf16"),          insts_path("gelu_bf16"));

    if (verbosity >= 1) std::cout << "All specs loaded.\n";

    // Discover group IDs
    int g3, g4, g5;
    {
        ActiveKernel tmp(device, spec_layer_norm);
        g3 = tmp.k.group_id(3);
        g4 = tmp.k.group_id(4);
        g5 = tmp.k.group_id(5);
    }
    if (verbosity >= 1)
        std::cout << "Data group IDs: slot3=" << g3 << " slot4=" << g4 << " slot5=" << g5 << "\n";

    // ---- Allocate all BOs ----

    // layer_norm: slot3=packed[input+weight], slot4=normed_tile, slot5=bias
    // The layer_norm kernel packs input [32,768] + weight [768] in slot3 = [33,768]
    auto bo_norm_in    = make_bo(device, g3, SZ_NORM_TILE);  // packed [33,768]
    auto bo_norm_out   = make_bo(device, g4, (size_t)NORM_SEQ_TILE * EMBD * 2);  // [32,768]
    auto bo_W_norm_1   = make_bo(device, g5, SZ_NORM_W);
    auto bo_b_norm_1   = make_bo(device, g5, SZ_NORM_W);
    auto bo_W_norm_2   = make_bo(device, g5, SZ_NORM_W);
    auto bo_b_norm_2   = make_bo(device, g5, SZ_NORM_W);

    // gemm_embd_embd: slot3=A, slot4=C, slot5=B
    auto bo_normed = make_bo(device, g3, SZ_FULL);   // [1024,768] normed input
    auto bo_query  = make_bo(device, g4, SZ_FULL);   // [1024,768]
    auto bo_key    = make_bo(device, g4, SZ_FULL);   // [1024,768]
    auto bo_value  = make_bo(device, g4, SZ_FULL);   // [1024,768]
    auto bo_Wq     = make_bo(device, g5, SZ_WQ);
    auto bo_Wk     = make_bo(device, g5, SZ_WQ);
    auto bo_Wv     = make_bo(device, g5, SZ_WQ);
    auto bo_Wo     = make_bo(device, g5, SZ_WQ);

    // gemm_score: slot3=Q_head, slot4=score, slot5=K_head_T
    auto bo_Q_head   = make_bo(device, g3, SZ_Q_HEAD);
    auto bo_score    = make_bo(device, g4, SZ_SCORE);
    auto bo_K_head_T = make_bo(device, g5, SZ_K_HEAD_T);

    // softmax: slot3=score_f32_tile[64,512], slot4=weight_f32_tile[64,512]
    auto bo_score_f32_tile  = make_bo(device, g3, SZ_SCORE_F32_TILE);
    auto bo_weight_f32_tile = make_bo(device, g4, SZ_SCORE_F32_TILE);

    // gemm_head_seq (attn value): slot3=attn_weight[1024,1024], slot4=ctx_head[1024,64], slot5=V_head[1024,64]
    auto bo_attn_weight = make_bo(device, g3, SZ_SCORE);    // [1024,1024] bf16
    auto bo_ctx_head    = make_bo(device, g4, SZ_Q_HEAD);   // [1024,64] bf16
    auto bo_V_head      = make_bo(device, g5, SZ_Q_HEAD);   // [1024,64] bf16

    // gemm_embd_embd again for Wo output
    auto bo_attn_value = make_bo(device, g3, SZ_FULL);      // [1024,768] assembled
    auto bo_x_out      = make_bo(device, g4, SZ_FULL);      // [1024,768] output of gemm

    // gemm_hid_embd: slot3=[1024,768], slot4=[1024,3072], slot5=[768,3072]
    auto bo_ffn_up     = make_bo(device, g4, SZ_FFN_UP);    // [1024,3072]
    auto bo_W_up       = make_bo(device, g5, SZ_W_UP);

    // gelu: slot3=[32,3072], slot4=[32,3072]
    auto bo_gelu_in    = make_bo(device, g3, SZ_FFN_TILE);
    auto bo_gelu_out   = make_bo(device, g4, SZ_FFN_TILE);

    // gemm_embd_embd for FFN down (reused 4 times)
    auto bo_ffn_in_chunk  = make_bo(device, g3, SZ_W_DOWN_CHUNK);  // [768,768] (act chunk reinterpreted as [SEQ_CHUNK*768])
    // Actually FFN down: A=[1024,768], B=[768,768], C=[1024,768]
    // So bo_ffn_in_chunk is [1024,768] = SZ_FULL, and bo_W_down_chunk is [768,768] = SZ_WQ
    // Resize bo_ffn_in_chunk:
    auto bo_ffn_down_A    = make_bo(device, g3, SZ_FULL);          // act chunk [1024,768]
    auto bo_ffn_down_C    = make_bo(device, g4, SZ_FULL);          // partial result [1024,768]
    auto bo_W_down_chunk  = make_bo(device, g5, SZ_WQ);            // [768,768] bf16

    // Input BO
    auto bo_input = make_bo(device, g3, SZ_FULL);

    // ---- Load weights (constant across forward passes) ----
    if (verbosity >= 1) std::cout << "Loading weights...\n";
    load_file_to_bo(bo_W_norm_1,  vm["W_norm_1"].as<std::string>(), SZ_NORM_W);
    load_file_to_bo(bo_b_norm_1,  vm["b_norm_1"].as<std::string>(), SZ_NORM_W);
    load_file_to_bo(bo_Wq,        vm["Wq"].as<std::string>(),       SZ_WQ);
    load_file_to_bo(bo_Wk,        vm["Wk"].as<std::string>(),       SZ_WQ);
    load_file_to_bo(bo_Wv,        vm["Wv"].as<std::string>(),       SZ_WQ);
    load_file_to_bo(bo_Wo,        vm["Wo"].as<std::string>(),       SZ_WQ);
    load_file_to_bo(bo_W_up,      vm["W_up"].as<std::string>(),     SZ_W_UP);
    load_file_to_bo(bo_W_norm_2,  vm["W_norm_2"].as<std::string>(), SZ_NORM_W);
    load_file_to_bo(bo_b_norm_2,  vm["b_norm_2"].as<std::string>(), SZ_NORM_W);

    // W_down is loaded in chunks below (per-call)

    // Host staging buffers
    std::vector<uint16_t> residual_host(SEQ * EMBD, 0);
    std::vector<uint16_t> normed_host(SEQ * EMBD, 0);
    std::vector<uint16_t> key_host(SEQ * EMBD, 0);
    std::vector<uint16_t> value_host(SEQ * EMBD, 0);
    std::vector<uint16_t> attn_weight_host(SEQ * SEQ, 0);
    std::vector<uint16_t> attn_value_host(SEQ * EMBD, 0);
    std::vector<uint16_t> ffn_up_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> act_host(SEQ * FFN_HID, 0);
    std::vector<float>    ffn_out_f32(SEQ * EMBD, 0.0f);

    auto run_forward = [&]() {
        // ---- Step 1: Load input → residual ----
        load_file_to_bo(bo_input, vm["input"].as<std::string>(), SZ_FULL);
        memcpy(residual_host.data(), bo_input.map<uint16_t*>(), SZ_FULL);

        // ---- Step 2: LayerNorm 1 — 32 tile calls ----
        // layer_norm slot3=packed[input[32,768]||weight[768]], slot4=normed_tile, slot5=bias
        {
            ActiveKernel ak(device, spec_layer_norm);
            uint16_t *norm_in_map  = bo_norm_in.map<uint16_t*>();
            uint16_t *norm_out_map = bo_norm_out.map<uint16_t*>();
            uint16_t *w1_map       = bo_W_norm_1.map<uint16_t*>();

            for (int t = 0; t < SEQ / NORM_SEQ_TILE; t++) {
                // Pack: input tile [32,768] followed by weight [768] in slot3
                memcpy(norm_in_map,
                       residual_host.data() + t * NORM_SEQ_TILE * EMBD,
                       NORM_SEQ_TILE * EMBD * sizeof(uint16_t));
                memcpy(norm_in_map + NORM_SEQ_TILE * EMBD,
                       w1_map, EMBD * sizeof(uint16_t));
                bo_norm_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_norm_in, bo_norm_out, bo_b_norm_1);
                bo_norm_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(normed_host.data() + t * NORM_SEQ_TILE * EMBD,
                       norm_out_map, NORM_SEQ_TILE * EMBD * sizeof(uint16_t));
            }
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_FULL);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Step 3-5: GEMM Wq, Wk, Wv ----
        {
            ActiveKernel ak(device, spec_gemm_embd);
            ak.run3(bo_normed, bo_query, bo_Wq);
            bo_query.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            ak.run3(bo_normed, bo_key, bo_Wk);
            bo_key.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(key_host.data(), bo_key.map<uint16_t*>(), SZ_FULL);

            ak.run3(bo_normed, bo_value, bo_Wv);
            bo_value.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(value_host.data(), bo_value.map<uint16_t*>(), SZ_FULL);
        }

        // ---- Step 6: Scale query by ATTN_SCALE (CPU) ----
        uint16_t *query_map = bo_query.map<uint16_t*>();
        scale_bf16_inplace(query_map, SEQ * EMBD, ATTN_SCALE);

        // ---- Step 7: Per-head attention (12 heads) ----
        for (int h = 0; h < N_HEAD; h++) {

            // CPU: extract Q_head[1024,64] from query_scaled
            extract_head(bo_Q_head.map<uint16_t*>(), query_map, h, N_HEAD);
            bo_Q_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // CPU: extract K_head[1024,64] and transpose → K_head_T[64,1024]
            std::vector<uint16_t> K_head_tmp(SEQ * HEAD_DIM);
            extract_head(K_head_tmp.data(), key_host.data(), h, N_HEAD);
            transpose_head(bo_K_head_T.map<uint16_t*>(), K_head_tmp.data());
            bo_K_head_T.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_score → score[1024,1024]
            {
                ActiveKernel ak(device, spec_gemm_score);
                ak.run3(bo_Q_head, bo_score, bo_K_head_T);
            }
            bo_score.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            uint16_t *score_map = bo_score.map<uint16_t*>();

            // Softmax (f32): 32 batches of [64 phys_rows, 512 phys_cols]
            {
                ActiveKernel ak(device, spec_softmax);
                float *sf_in_map  = bo_score_f32_tile.map<float*>();
                float *sf_out_map = bo_weight_f32_tile.map<float*>();

                for (int b = 0; b < SOFTMAX_NUM_BATCHES; b++) {
                    unpack_score_tile(score_map, sf_in_map, b);
                    bo_score_f32_tile.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    ak.run2(bo_score_f32_tile, bo_weight_f32_tile);
                    bo_weight_f32_tile.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    repack_weight_tile(sf_out_map, attn_weight_host.data(), b);
                }
            }
            memcpy(bo_attn_weight.map<uint16_t*>(), attn_weight_host.data(), SZ_SCORE);
            bo_attn_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // CPU: extract V_head[1024,64]
            extract_head(bo_V_head.map<uint16_t*>(), value_host.data(), h, N_HEAD);
            bo_V_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_head_seq → ctx_head[1024,64]
            {
                ActiveKernel ak(device, spec_gemm_head_seq);
                ak.run3(bo_attn_weight, bo_ctx_head, bo_V_head);
            }
            bo_ctx_head.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // CPU: assemble ctx_head into attn_value_host[1024, 12*64]
            uint16_t *ctx_map = bo_ctx_head.map<uint16_t*>();
            for (int s = 0; s < SEQ; s++)
                memcpy(attn_value_host.data() + s * EMBD + h * HEAD_DIM,
                       ctx_map + s * HEAD_DIM,
                       HEAD_DIM * sizeof(uint16_t));
        }

        // ---- Step 8: GEMM Wo → x_out[1024,768] ----
        memcpy(bo_attn_value.map<uint16_t*>(), attn_value_host.data(), SZ_FULL);
        bo_attn_value.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {
            ActiveKernel ak(device, spec_gemm_embd);
            ak.run3(bo_attn_value, bo_x_out, bo_Wo);
        }
        bo_x_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // ---- Step 9: residual += x_out (CPU bf16 add) ----
        {
            uint16_t *x_out_map = bo_x_out.map<uint16_t*>();
            for (int i = 0; i < SEQ * EMBD; i++) {
                float r = bf16_to_f32(residual_host[i]) + bf16_to_f32(x_out_map[i]);
                residual_host[i] = f32_to_bf16(r);
            }
        }

        // ---- Step 10: LayerNorm 2 — 32 tile calls ----
        {
            ActiveKernel ak(device, spec_layer_norm);
            uint16_t *norm_in_map  = bo_norm_in.map<uint16_t*>();
            uint16_t *norm_out_map = bo_norm_out.map<uint16_t*>();
            uint16_t *w2_map       = bo_W_norm_2.map<uint16_t*>();

            for (int t = 0; t < SEQ / NORM_SEQ_TILE; t++) {
                memcpy(norm_in_map,
                       residual_host.data() + t * NORM_SEQ_TILE * EMBD,
                       NORM_SEQ_TILE * EMBD * sizeof(uint16_t));
                memcpy(norm_in_map + NORM_SEQ_TILE * EMBD,
                       w2_map, EMBD * sizeof(uint16_t));
                bo_norm_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_norm_in, bo_norm_out, bo_b_norm_2);
                bo_norm_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(normed_host.data() + t * NORM_SEQ_TILE * EMBD,
                       norm_out_map, NORM_SEQ_TILE * EMBD * sizeof(uint16_t));
            }
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_FULL);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Step 11: GEMM W_up → ffn_up_x[1024,3072] ----
        {
            ActiveKernel ak(device, spec_gemm_hid);
            ak.run3(bo_normed, bo_ffn_up, bo_W_up);
        }
        bo_ffn_up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(ffn_up_host.data(), bo_ffn_up.map<uint16_t*>(), SZ_FFN_UP);

        // ---- Step 12: GELU — 32 tile calls ----
        {
            ActiveKernel ak(device, spec_gelu);
            uint16_t *gin_map  = bo_gelu_in.map<uint16_t*>();
            uint16_t *gout_map = bo_gelu_out.map<uint16_t*>();
            for (int t = 0; t < SEQ / GELU_SEQ_TILE; t++) {
                memcpy(gin_map, ffn_up_host.data() + t * GELU_SEQ_TILE * FFN_HID,
                       GELU_SEQ_TILE * FFN_HID * sizeof(uint16_t));
                bo_gelu_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run2(bo_gelu_in, bo_gelu_out);
                bo_gelu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(act_host.data() + t * GELU_SEQ_TILE * FFN_HID,
                       gout_map, GELU_SEQ_TILE * FFN_HID * sizeof(uint16_t));
            }
        }

        // ---- Step 13: FFN Down (4 chunks), float32 accumulation ----
        std::fill(ffn_out_f32.begin(), ffn_out_f32.end(), 0.0f);
        {
            // Read W_down file in chunks
            std::ifstream wdown_f(vm["W_down"].as<std::string>(), std::ios::binary);
            if (!wdown_f) { std::cerr << "Cannot open W_down\n"; exit(1); }

            ActiveKernel ak(device, spec_gemm_embd);
            for (int chunk = 0; chunk < FFN_DOWN_K_CHUNKS; chunk++) {
                // Copy act chunk [1024, 768] → bo_ffn_down_A
                uint16_t *down_in_map = bo_ffn_down_A.map<uint16_t*>();
                for (int r = 0; r < SEQ; r++)
                    memcpy(down_in_map + r * FFN_DOWN_K_CHUNK,
                           act_host.data() + r * FFN_HID + chunk * FFN_DOWN_K_CHUNK,
                           FFN_DOWN_K_CHUNK * sizeof(uint16_t));
                bo_ffn_down_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Load W_down chunk [768, 768]
                wdown_f.seekg((size_t)chunk * FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                wdown_f.read(bo_W_down_chunk.map<char*>(),
                             FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                bo_W_down_chunk.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                ak.run3(bo_ffn_down_A, bo_ffn_down_C, bo_W_down_chunk);
                bo_ffn_down_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // Accumulate in float32
                uint16_t *c_map = bo_ffn_down_C.map<uint16_t*>();
                for (int i = 0; i < SEQ * EMBD; i++)
                    ffn_out_f32[i] += bf16_to_f32(c_map[i]);
            }
        }

        // ---- Step 14: residual += ffn_out_f32 (convert back to bf16) ----
        for (int i = 0; i < SEQ * EMBD; i++) {
            float r = bf16_to_f32(residual_host[i]) + ffn_out_f32[i];
            residual_host[i] = f32_to_bf16(r);
        }
    };

    // ---- Warmup + timed runs ----
    if (verbosity >= 1) std::cout << "Warmup run...\n";
    run_forward();  // warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) run_forward();
    auto t1 = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
    std::cout << "Vision encoder forward: " << total_ms / n_iters << " ms/iter"
              << " (avg over " << n_iters << " iters)\n";

    // ---- Write output ----
    std::ofstream out_f(vm["output"].as<std::string>(), std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(residual_host.data()),
                residual_host.size() * sizeof(uint16_t));
    if (verbosity >= 1)
        std::cout << "Output written to " << vm["output"].as<std::string>() << "\n";

    return 0;
}
