// text_encoder.cpp — Text encoder transformer block forward pass for AMD NPU.
//
// Build:
//   cd text_encoder_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4
// Run (from text_encoder_bf16/unified.prj/):
//   build/text_encoder \
//     --input x.data --W_norm_1 wn1.data --W_norm_2 wn2.data \
//     --Wq wq.data --Wk wk.data --Wv wv.data --Wo wo.data \
//     --W_gate wgate.data --W_up wup.data --W_down wdown.data \
//     --output out.data --key_out key.data --val_out val.data -v 1
// Validate:
//   python3 -c "
//   import numpy as np; from ml_dtypes import bfloat16 as bf16
//   ref = np.fromfile('ref_out.data', dtype='uint16').view(bf16).astype(np.float32)
//   ours = np.fromfile('out.data', dtype='uint16').view(bf16).astype(np.float32)
//   print('max_err:', np.max(np.abs(ref - ours)))
//   "
//
// Pipeline mirrors text_encoder_forward() in text_encoder_bf16.py.
// Extends attn.prj/attn.cpp with: FFN (gate+up+silu+down), RoPE (fused 5-head NPU),
// and cross-attention K/V export (CPU RMSNorm + CPU matmul).
//
// Key dimensions: SEQ=128, EMBD=960, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2560

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

#include <filesystem>

namespace po = boost::program_options;

// ============================================================
// Model dimensions
// ============================================================
static constexpr int SEQ      = 128;
static constexpr int EMBD     = 960;
static constexpr int Q_H      = 15;
static constexpr int KV_H     = 5;
static constexpr int HEAD_DIM = 64;
static constexpr int KV_DIM   = KV_H * HEAD_DIM;   // 320
static constexpr int FFN_HID  = 2560;

static constexpr int FFN_DOWN_K_CHUNK  = 320;
static constexpr int FFN_DOWN_K_CHUNKS = FFN_HID / FFN_DOWN_K_CHUNK;  // 8

static constexpr int NORM_SEQ_TILE  = 32;   // rms_norm: 32 rows per call
static constexpr int SOFTMAX_TILE   = 8;    // masked_softmax: 8 rows per call (unused below)
static constexpr int SILU_SEQ_TILE  = 16;   // silu: 16 rows per call

// RoPE: fused_5h kernel processes 5 heads × 32 rows = 160 rows at once
static constexpr int ROPE_FUSED_TILE = 32;
static constexpr int ROPE_CHUNK      = 5;   // heads per dispatch

// Attention scale
static constexpr float ATTN_SCALE = 0.125f;  // 1/sqrt(64)

// Max wavelength for RoPE
static constexpr float MAX_WAVELENGTH = 10000.0f;

// ============================================================
// Buffer sizes
// ============================================================
static constexpr size_t SZ_FULL      = (size_t)SEQ  * EMBD * 2;         // [128,960] bf16
static constexpr size_t SZ_NORM_TILE = (size_t)NORM_SEQ_TILE * EMBD * 2;// [32,960] bf16
static constexpr size_t SZ_NORM_W    = EMBD * 2;                         // [960] bf16
static constexpr size_t SZ_KV_OUT    = (size_t)SEQ  * KV_DIM * 2;       // [128,320] bf16
static constexpr size_t SZ_WQ        = (size_t)EMBD * Q_H * HEAD_DIM * 2;  // [960,960] bf16
static constexpr size_t SZ_WKV       = (size_t)EMBD * KV_DIM * 2;       // [960,320] bf16
static constexpr size_t SZ_Q_HEAD    = (size_t)SEQ  * HEAD_DIM * 2;     // [128,64] bf16
static constexpr size_t SZ_K_HEAD_T  = (size_t)HEAD_DIM * SEQ * 2;      // [64,128] bf16
static constexpr size_t SZ_SCORE     = (size_t)SEQ  * SEQ * 2;          // [128,128] bf16
static constexpr size_t SZ_V_HEAD    = (size_t)SEQ  * HEAD_DIM * 2;     // [128,64] bf16
static constexpr size_t SZ_CTX_HEAD  = (size_t)SEQ  * HEAD_DIM * 2;     // [128,64] bf16
static constexpr size_t SZ_FFN_UP    = (size_t)SEQ  * FFN_HID * 2;      // [128,2560] bf16
static constexpr size_t SZ_SILU_TILE = (size_t)SILU_SEQ_TILE * FFN_HID * 2;  // [16,2560] bf16
static constexpr size_t SZ_FFN_DOWN_A = (size_t)SEQ * FFN_DOWN_K_CHUNK * 2;  // [128,320] bf16
static constexpr size_t SZ_FFN_DOWN_C = (size_t)SEQ * EMBD * 2;         // [128,960] bf16
static constexpr size_t SZ_W_DOWN_CHUNK = (size_t)FFN_DOWN_K_CHUNK * EMBD * 2;  // [320,960] bf16

// RoPE BOs: float32
static constexpr size_t SZ_ROPE_X   = (size_t)ROPE_CHUNK * ROPE_FUSED_TILE * HEAD_DIM * 4;  // [160,64] f32
static constexpr size_t SZ_ROPE_SC  = (size_t)ROPE_CHUNK * ROPE_FUSED_TILE * HEAD_DIM * 4;  // [160,64] f32

// ============================================================
// KernelSpec
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
// ActiveKernel
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

    // 2-slot kernel (silu, softmax): slot3=in, slot4=out
    void run2(xrt::bo &b3, xrt::bo &b4) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.start(); r.wait();
    }

    // 3-slot kernel: slot3=in0, slot4=out, slot5=in2
    void run3(xrt::bo &b3, xrt::bo &b4, xrt::bo &b5) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.set_arg(5, b5);
        r.start(); r.wait();
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

static inline float bf16_to_f32(uint16_t raw) {
    uint32_t bits = (uint32_t)raw << 16;
    float val; memcpy(&val, &bits, 4);
    return val;
}

static inline uint16_t f32_to_bf16(float val) {
    uint32_t bits; memcpy(&bits, &val, 4);
    return (uint16_t)(bits >> 16);
}

static void extract_head(uint16_t *dst, const uint16_t *src, int head, int n_heads) {
    for (int s = 0; s < SEQ; s++)
        memcpy(dst + s * HEAD_DIM,
               src + s * n_heads * HEAD_DIM + head * HEAD_DIM,
               HEAD_DIM * sizeof(uint16_t));
}

static void transpose_head(uint16_t *dst, const uint16_t *src, int seq_len) {
    for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < HEAD_DIM; d++)
            dst[d * seq_len + s] = src[s * HEAD_DIM + d];
}

// Precompute sin/cos table: sin_cos[r][i] where i < HEAD_DIM/2 = sin, i >= HALF = cos
static void precompute_sin_cos(float *sin_cos, int n_rows, int head_dim,
                                float max_wavelength, int pos_offset) {
    int half = head_dim / 2;
    for (int r = 0; r < n_rows; r++) {
        float pos = (float)(pos_offset + r);
        for (int i = 0; i < half; i++) {
            float inv_ts = std::pow(max_wavelength, -(2.0f * (float)i / (float)head_dim));
            float rad = pos * inv_ts;
            sin_cos[r * head_dim + i]        = std::sin(rad);
            sin_cos[r * head_dim + half + i] = std::cos(rad);
        }
    }
}

// Apply causal mask to score_head [SEQ, SEQ] bf16 (set upper triangular to -inf)
static void apply_causal_mask(uint16_t *score, int seq_len) {
    static const uint16_t NEG_INF_BF16 = 0xFF80u;  // -inf in bf16
    for (int r = 0; r < seq_len; r++)
        for (int c = r + 1; c < seq_len; c++)
            score[r * seq_len + c] = NEG_INF_BF16;
}

// ============================================================
// Main
// ============================================================
int main(int argc, const char *argv[]) {
    po::options_description opts("Text encoder forward pass");
    opts.add_options()
        ("help,h",       "help")
        ("input",        po::value<std::string>()->required(),              "[128,960] bf16 input")
        ("W_norm_1",     po::value<std::string>()->default_value(""),      "[960] bf16 rms_norm weight 1")
        ("W_norm_2",     po::value<std::string>()->default_value(""),      "[960] bf16 rms_norm weight 2")
        ("Wq",           po::value<std::string>()->default_value(""),      "[960,960] bf16")
        ("Wk",           po::value<std::string>()->default_value(""),      "[960,320] bf16")
        ("Wv",           po::value<std::string>()->default_value(""),      "[960,320] bf16")
        ("Wo",           po::value<std::string>()->default_value(""),      "[960,960] bf16")
        ("W_gate",       po::value<std::string>()->default_value(""),      "[960,2560] bf16")
        ("W_up",         po::value<std::string>()->default_value(""),      "[960,2560] bf16")
        ("W_down",       po::value<std::string>()->default_value(""),      "[2560,960] bf16")
        ("output",       po::value<std::string>()->default_value("output.data"), "[128,960] bf16")
        ("key_out",      po::value<std::string>()->default_value(""),       "[128,320] bf16 (optional)")
        ("val_out",      po::value<std::string>()->default_value(""),       "[128,320] bf16 (optional)")
        ("num-layers,L", po::value<int>()->default_value(1),               "number of transformer layers")
        ("layers-dir",   po::value<std::string>()->default_value(""),      "dir with layer_0/,layer_1/,... weight subdirs")
        ("kv-dir",       po::value<std::string>()->default_value(""),      "dir to write per-layer key/val for cross-attention")
        ("iters,n",      po::value<int>()->default_value(1),               "number of forward passes")
        ("verbosity,v",  po::value<int>()->default_value(0),               "verbosity");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, opts), vm);
        if (vm.count("help")) { std::cout << opts; return 0; }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n" << opts; return 1;
    }

    int verbosity  = vm["verbosity"].as<int>();
    int n_iters    = vm["iters"].as<int>();
    int num_layers = vm["num-layers"].as<int>();
    std::string layers_dir = vm["layers-dir"].as<std::string>();
    std::string kv_dir     = vm["kv-dir"].as<std::string>();

    auto weight_path = [&](const std::string &short_name, const std::string &cli_arg,
                            int layer) -> std::string {
        if (!layers_dir.empty())
            return layers_dir + "/layer_" + std::to_string(layer) + "/" + short_name;
        return vm[cli_arg].as<std::string>();
    };

    const std::string BASE = "..";
    auto xclbin_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/build/final.xclbin";
    };
    auto insts_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/insts.txt";
    };

    if (verbosity >= 1) std::cout << "Preloading kernel specs...\n";
    auto device = xrt::device(0);

    KernelSpec spec_rms_norm, spec_gemm_q, spec_gemm_kv;
    KernelSpec spec_attn_score, spec_softmax, spec_attn_value;
    KernelSpec spec_gemm_out, spec_gemm_ffn_up, spec_gemm_ffn_down;
    KernelSpec spec_silu, spec_rope_5h;

    spec_rms_norm.preload   (device, xclbin_path("rms_norm"),        insts_path("rms_norm"));
    spec_gemm_q.preload     (device, xclbin_path("gemm_q"),          insts_path("gemm_q"));
    spec_gemm_kv.preload    (device, xclbin_path("gemm_kv"),         insts_path("gemm_kv"));
    spec_attn_score.preload (device, xclbin_path("gemm_attn_score"), insts_path("gemm_attn_score"));
    spec_softmax.preload    (device, xclbin_path("softmax"),         insts_path("softmax"));
    spec_attn_value.preload (device, xclbin_path("gemm_attn_value"), insts_path("gemm_attn_value"));
    spec_gemm_out.preload   (device, xclbin_path("gemm_out"),        insts_path("gemm_out"));
    spec_gemm_ffn_up.preload(device, xclbin_path("gemm_ffn_up"),     insts_path("gemm_ffn_up"));
    spec_gemm_ffn_down.preload(device,xclbin_path("gemm_ffn_down"),  insts_path("gemm_ffn_down"));
    spec_silu.preload       (device, xclbin_path("silu"),            insts_path("silu"));
    spec_rope_5h.preload    (device, "../rope/fused_5h.prj/build/final.xclbin",
                                     "../rope/fused_5h.prj/insts.txt");

    if (verbosity >= 1) std::cout << "All specs loaded.\n";

    // Discover group IDs
    int g3, g4, g5;
    {
        ActiveKernel tmp(device, spec_rms_norm);
        g3 = tmp.k.group_id(3);
        g4 = tmp.k.group_id(4);
        g5 = tmp.k.group_id(5);
    }
    if (verbosity >= 1)
        std::cout << "Data group IDs: slot3=" << g3 << " slot4=" << g4 << " slot5=" << g5 << "\n";

    // ---- Allocate all BOs ----

    // rms_norm: slot3=tile_in[32,960], slot4=tile_out[32,960], slot5=weight[960]
    auto bo_rms_in  = make_bo(device, g3, SZ_NORM_TILE);
    auto bo_rms_out = make_bo(device, g4, SZ_NORM_TILE);
    auto bo_W_norm_1 = make_bo(device, g5, SZ_NORM_W);
    auto bo_W_norm_2 = make_bo(device, g5, SZ_NORM_W);

    // Full matrices
    auto bo_normed  = make_bo(device, g3, SZ_FULL);
    auto bo_query   = make_bo(device, g4, SZ_FULL);
    auto bo_key     = make_bo(device, g4, SZ_KV_OUT);
    auto bo_val     = make_bo(device, g4, SZ_KV_OUT);
    auto bo_Wq      = make_bo(device, g5, SZ_WQ);
    auto bo_Wk      = make_bo(device, g5, SZ_WKV);
    auto bo_Wv      = make_bo(device, g5, SZ_WKV);
    auto bo_Wo      = make_bo(device, g5, SZ_WQ);

    // Attention per-head
    auto bo_Q_head   = make_bo(device, g3, SZ_Q_HEAD);
    auto bo_score    = make_bo(device, g4, SZ_SCORE);
    auto bo_K_head_T = make_bo(device, g5, SZ_K_HEAD_T);
    auto bo_attn_weight = make_bo(device, g3, SZ_SCORE);
    auto bo_ctx_head    = make_bo(device, g4, SZ_CTX_HEAD);
    auto bo_V_head      = make_bo(device, g5, SZ_V_HEAD);

    // Softmax: processes full [128,128] bf16 in one call (mapping=[16])
    // slot3=score_head[128,128] bf16, slot4=weight_head[128,128] bf16
    // (same BO as bo_score/bo_attn_weight, reused below)

    // GEMM out
    auto bo_ctx_full = make_bo(device, g3, SZ_FULL);
    auto bo_x_out    = make_bo(device, g4, SZ_FULL);

    // FFN
    auto bo_gate_proj  = make_bo(device, g4, SZ_FFN_UP);   // [128,2560]
    auto bo_up_proj    = make_bo(device, g4, SZ_FFN_UP);   // [128,2560]
    auto bo_W_gate     = make_bo(device, g5, (size_t)EMBD * FFN_HID * 2);
    auto bo_W_up       = make_bo(device, g5, (size_t)EMBD * FFN_HID * 2);
    auto bo_silu_in    = make_bo(device, g3, SZ_SILU_TILE);
    auto bo_silu_out   = make_bo(device, g4, SZ_SILU_TILE);
    auto bo_ffn_down_A = make_bo(device, g3, SZ_FFN_DOWN_A);
    auto bo_ffn_down_C = make_bo(device, g4, SZ_FFN_DOWN_C);
    auto bo_W_down_chunk = make_bo(device, g5, SZ_W_DOWN_CHUNK);

    // RoPE: float32 [5*32=160, 64]
    auto bo_rope_x   = make_bo(device, g3, SZ_ROPE_X);
    auto bo_rope_sc  = make_bo(device, g4, SZ_ROPE_SC);
    auto bo_rope_out = make_bo(device, g5, SZ_ROPE_X);

    // Input
    auto bo_input = make_bo(device, g3, SZ_FULL);

    // Host staging
    std::vector<uint16_t> residual_host(SEQ * EMBD, 0);
    std::vector<uint16_t> normed_host(SEQ * EMBD, 0);
    std::vector<uint16_t> key_host(SEQ * KV_DIM, 0);
    std::vector<uint16_t> val_host(SEQ * KV_DIM, 0);
    std::vector<uint16_t> attn_weight_host(SEQ * SEQ, 0);
    std::vector<uint16_t> ctx_full_host(SEQ * Q_H * HEAD_DIM, 0);
    std::vector<uint16_t> gate_proj_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> up_proj_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> act_host(SEQ * FFN_HID, 0);
    std::vector<float>    ffn_out_f32(SEQ * EMBD, 0.0f);
    // RoPE buffers (float32)
    std::vector<float> q_f32(SEQ * Q_H * HEAD_DIM, 0.0f);
    std::vector<float> k_f32(SEQ * KV_DIM, 0.0f);

    // Helper: apply RoPE via fused_5h NPU kernel
    // packed_f32: [seq_len, heads*HEAD_DIM] float32 (modified in-place)
    auto rope_apply = [&](float *packed_f32, int seq_len, int heads, ActiveKernel &ak) {
        float *x_map  = bo_rope_x.map<float*>();
        float *sc_map = bo_rope_sc.map<float*>();
        float *out_map = bo_rope_out.map<float*>();

        for (int t0 = 0; t0 < seq_len; t0 += ROPE_FUSED_TILE) {
            // Precompute sin_cos for this tile
            std::vector<float> sin_cos(ROPE_FUSED_TILE * HEAD_DIM);
            precompute_sin_cos(sin_cos.data(), ROPE_FUSED_TILE, HEAD_DIM, MAX_WAVELENGTH, t0);
            // Tile sin_cos for ROPE_CHUNK copies
            for (int i = 0; i < ROPE_CHUNK; i++)
                memcpy(sc_map + i * ROPE_FUSED_TILE * HEAD_DIM,
                       sin_cos.data(), ROPE_FUSED_TILE * HEAD_DIM * sizeof(float));
            bo_rope_sc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (int h0 = 0; h0 < heads; h0 += ROPE_CHUNK) {
                // Pack ROPE_CHUNK heads × ROPE_FUSED_TILE rows
                memset(x_map, 0, SZ_ROPE_X);
                for (int i = 0; i < ROPE_CHUNK; i++) {
                    int h = h0 + i;
                    for (int r = 0; r < ROPE_FUSED_TILE && (t0 + r) < seq_len; r++) {
                        memcpy(x_map + (i * ROPE_FUSED_TILE + r) * HEAD_DIM,
                               packed_f32 + (t0 + r) * heads * HEAD_DIM + h * HEAD_DIM,
                               HEAD_DIM * sizeof(float));
                    }
                }
                bo_rope_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_rope_x, bo_rope_sc, bo_rope_out);
                bo_rope_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // Unpack
                for (int i = 0; i < ROPE_CHUNK; i++) {
                    int h = h0 + i;
                    for (int r = 0; r < ROPE_FUSED_TILE && (t0 + r) < seq_len; r++) {
                        memcpy(packed_f32 + (t0 + r) * heads * HEAD_DIM + h * HEAD_DIM,
                               out_map + (i * ROPE_FUSED_TILE + r) * HEAD_DIM,
                               HEAD_DIM * sizeof(float));
                    }
                }
            }
        }
    };

    auto run_forward = [&](int layer = 0, bool first_layer = true) {
        // ---- Load weights for this layer ----
        load_file_to_bo(bo_W_norm_1, weight_path("wn1.data",   "W_norm_1", layer), SZ_NORM_W);
        load_file_to_bo(bo_W_norm_2, weight_path("wn2.data",   "W_norm_2", layer), SZ_NORM_W);
        load_file_to_bo(bo_Wq,       weight_path("wq.data",    "Wq",       layer), SZ_WQ);
        load_file_to_bo(bo_Wk,       weight_path("wk.data",    "Wk",       layer), SZ_WKV);
        load_file_to_bo(bo_Wv,       weight_path("wv.data",    "Wv",       layer), SZ_WKV);
        load_file_to_bo(bo_Wo,       weight_path("wo.data",    "Wo",       layer), SZ_WQ);
        load_file_to_bo(bo_W_gate,   weight_path("wgate.data", "W_gate",   layer), (size_t)EMBD * FFN_HID * 2);
        load_file_to_bo(bo_W_up,     weight_path("wup.data",   "W_up",     layer), (size_t)EMBD * FFN_HID * 2);

        // ---- Step 1: Load input → residual ----
        if (first_layer) {
            load_file_to_bo(bo_input, vm["input"].as<std::string>(), SZ_FULL);
            memcpy(residual_host.data(), bo_input.map<uint16_t*>(), SZ_FULL);
        }

        // ---- Step 2: RMSNorm 1 — 4 tile calls ----
        {
            ActiveKernel ak(device, spec_rms_norm);
            uint16_t *in_map  = bo_rms_in.map<uint16_t*>();
            uint16_t *out_map = bo_rms_out.map<uint16_t*>();
            for (int t = 0; t < SEQ / NORM_SEQ_TILE; t++) {
                memcpy(in_map, residual_host.data() + t * NORM_SEQ_TILE * EMBD, SZ_NORM_TILE);
                bo_rms_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_rms_in, bo_rms_out, bo_W_norm_1);
                bo_rms_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(normed_host.data() + t * NORM_SEQ_TILE * EMBD, out_map, SZ_NORM_TILE);
            }
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_FULL);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Steps 3-5: GEMM Wq, Wk, Wv ----
        {
            ActiveKernel ak(device, spec_gemm_q);
            ak.run3(bo_normed, bo_query, bo_Wq);
        }
        bo_query.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        {
            ActiveKernel ak(device, spec_gemm_kv);
            ak.run3(bo_normed, bo_key, bo_Wk);
            bo_key.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(key_host.data(), bo_key.map<uint16_t*>(), SZ_KV_OUT);

            ak.run3(bo_normed, bo_val, bo_Wv);
        }
        bo_val.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(val_host.data(), bo_val.map<uint16_t*>(), SZ_KV_OUT);

        // ---- Step 5b: RoPE on Q (15 heads) and K (5 heads) ----
        // Convert Q and K to float32, apply RoPE, convert back to bf16
        {
            uint16_t *q_map = bo_query.map<uint16_t*>();
            for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++)
                q_f32[i] = bf16_to_f32(q_map[i]);
            for (int i = 0; i < SEQ * KV_DIM; i++)
                k_f32[i] = bf16_to_f32(key_host[i]);

            {
                ActiveKernel ak(device, spec_rope_5h);
                rope_apply(q_f32.data(), SEQ, Q_H, ak);
                rope_apply(k_f32.data(), SEQ, KV_H, ak);
            }

            // Convert back to bf16
            for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++)
                q_map[i] = f32_to_bf16(q_f32[i]);
            for (int i = 0; i < SEQ * KV_DIM; i++)
                key_host[i] = f32_to_bf16(k_f32[i]);
        }

        // ---- Step 6: Scale query ----
        {
            uint16_t *q_map = bo_query.map<uint16_t*>();
            for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++) {
                float v = bf16_to_f32(q_map[i]) * ATTN_SCALE;
                q_map[i] = f32_to_bf16(v);
            }
        }

        // ---- Step 7: Per-head attention (15 heads) ----
        for (int h = 0; h < Q_H; h++) {
            int kv_idx = h * KV_H / Q_H;
            uint16_t *q_map = bo_query.map<uint16_t*>();

            // Extract Q_head
            extract_head(bo_Q_head.map<uint16_t*>(), q_map, h, Q_H);
            bo_Q_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Extract K_head and transpose
            std::vector<uint16_t> K_head_tmp(SEQ * HEAD_DIM);
            extract_head(K_head_tmp.data(), key_host.data(), kv_idx, KV_H);
            transpose_head(bo_K_head_T.map<uint16_t*>(), K_head_tmp.data(), SEQ);
            bo_K_head_T.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_attn_score
            {
                ActiveKernel ak(device, spec_attn_score);
                ak.run3(bo_Q_head, bo_score, bo_K_head_T);
            }
            bo_score.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // CPU: apply causal mask
            apply_causal_mask(bo_score.map<uint16_t*>(), SEQ);
            bo_score.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: softmax (1 call, all 16 tiles parallel, mapping=[16])
            {
                ActiveKernel ak(device, spec_softmax);
                ak.run2(bo_score, bo_attn_weight);
            }
            bo_attn_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // Extract V_head
            extract_head(bo_V_head.map<uint16_t*>(), val_host.data(), kv_idx, KV_H);
            bo_V_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // NPU: gemm_attn_value
            {
                ActiveKernel ak(device, spec_attn_value);
                ak.run3(bo_attn_weight, bo_ctx_head, bo_V_head);
            }
            bo_ctx_head.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // Assemble into ctx_full_host
            uint16_t *ctx_map = bo_ctx_head.map<uint16_t*>();
            for (int s = 0; s < SEQ; s++)
                memcpy(ctx_full_host.data() + s * Q_H * HEAD_DIM + h * HEAD_DIM,
                       ctx_map + s * HEAD_DIM, HEAD_DIM * sizeof(uint16_t));
        }

        // ---- Step 8: GEMM Wo ----
        memcpy(bo_ctx_full.map<uint16_t*>(), ctx_full_host.data(), SZ_FULL);
        bo_ctx_full.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {
            ActiveKernel ak(device, spec_gemm_out);
            ak.run3(bo_ctx_full, bo_x_out, bo_Wo);
        }
        bo_x_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // ---- Step 9: residual += x_out ----
        {
            uint16_t *x_map = bo_x_out.map<uint16_t*>();
            for (int i = 0; i < SEQ * EMBD; i++) {
                float r = bf16_to_f32(residual_host[i]) + bf16_to_f32(x_map[i]);
                residual_host[i] = f32_to_bf16(r);
            }
        }

        // ---- Step 10: Save post_attn for cross-K/V export ----
        std::vector<uint16_t> post_attn_host(residual_host);

        // ---- Step 11: RMSNorm 2 ----
        {
            ActiveKernel ak(device, spec_rms_norm);
            uint16_t *in_map  = bo_rms_in.map<uint16_t*>();
            uint16_t *out_map = bo_rms_out.map<uint16_t*>();
            for (int t = 0; t < SEQ / NORM_SEQ_TILE; t++) {
                memcpy(in_map, residual_host.data() + t * NORM_SEQ_TILE * EMBD, SZ_NORM_TILE);
                bo_rms_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run3(bo_rms_in, bo_rms_out, bo_W_norm_2);
                bo_rms_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(normed_host.data() + t * NORM_SEQ_TILE * EMBD, out_map, SZ_NORM_TILE);
            }
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_FULL);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Steps 12-13: Gate and Up projections ----
        {
            ActiveKernel ak(device, spec_gemm_ffn_up);
            ak.run3(bo_normed, bo_gate_proj, bo_W_gate);
            bo_gate_proj.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(gate_proj_host.data(), bo_gate_proj.map<uint16_t*>(), SZ_FFN_UP);

            ak.run3(bo_normed, bo_up_proj, bo_W_up);
        }
        bo_up_proj.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(up_proj_host.data(), bo_up_proj.map<uint16_t*>(), SZ_FFN_UP);

        // ---- Step 14: SiLU — 8 tile calls ----
        {
            ActiveKernel ak(device, spec_silu);
            uint16_t *sin_map = bo_silu_in.map<uint16_t*>();
            uint16_t *sout_map = bo_silu_out.map<uint16_t*>();
            for (int t = 0; t < SEQ / SILU_SEQ_TILE; t++) {
                memcpy(sin_map, gate_proj_host.data() + t * SILU_SEQ_TILE * FFN_HID,
                       SZ_SILU_TILE);
                bo_silu_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run2(bo_silu_in, bo_silu_out);
                bo_silu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(act_host.data() + t * SILU_SEQ_TILE * FFN_HID, sout_map, SZ_SILU_TILE);
            }
        }

        // CPU SiLU fallback for unsafe values (|x| > 2.5)
        {
            for (int i = 0; i < SEQ * FFN_HID; i++) {
                float g = bf16_to_f32(gate_proj_host[i]);
                if (std::abs(g) > 2.5f) {
                    float clipped = g > 20.0f ? 20.0f : (g < -20.0f ? -20.0f : g);
                    float cpu_val = g / (1.0f + std::exp(-clipped));
                    act_host[i] = f32_to_bf16(cpu_val);
                }
                // Also patch Inf/NaN
                float a = bf16_to_f32(act_host[i]);
                if (std::isinf(a) || std::isnan(a)) {
                    float clipped = g > 20.0f ? 20.0f : (g < -20.0f ? -20.0f : g);
                    act_host[i] = f32_to_bf16(g / (1.0f + std::exp(-clipped)));
                }
            }
        }

        // Step 14b: Hadamard act *= up_proj
        for (int i = 0; i < SEQ * FFN_HID; i++) {
            float v = bf16_to_f32(act_host[i]) * bf16_to_f32(up_proj_host[i]);
            act_host[i] = f32_to_bf16(v);
        }

        // ---- Step 15: FFN Down (8 chunks of K=320) ----
        std::fill(ffn_out_f32.begin(), ffn_out_f32.end(), 0.0f);
        {
            std::ifstream wdown_f(weight_path("wdown.data", "W_down", layer), std::ios::binary);
            if (!wdown_f) { std::cerr << "Cannot open W_down\n"; exit(1); }

            ActiveKernel ak(device, spec_gemm_ffn_down);
            for (int chunk = 0; chunk < FFN_DOWN_K_CHUNKS; chunk++) {
                // Copy act chunk [128, 320]
                uint16_t *down_in_map = bo_ffn_down_A.map<uint16_t*>();
                for (int r = 0; r < SEQ; r++)
                    memcpy(down_in_map + r * FFN_DOWN_K_CHUNK,
                           act_host.data() + r * FFN_HID + chunk * FFN_DOWN_K_CHUNK,
                           FFN_DOWN_K_CHUNK * sizeof(uint16_t));
                bo_ffn_down_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Load W_down chunk [320, 960]
                wdown_f.seekg((size_t)chunk * FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                wdown_f.read(bo_W_down_chunk.map<char*>(),
                             FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                bo_W_down_chunk.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                ak.run3(bo_ffn_down_A, bo_ffn_down_C, bo_W_down_chunk);
                bo_ffn_down_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                uint16_t *c_map = bo_ffn_down_C.map<uint16_t*>();
                for (int i = 0; i < SEQ * EMBD; i++)
                    ffn_out_f32[i] += bf16_to_f32(c_map[i]);
            }
        }

        // ---- Step 16: residual += ffn_out ----
        for (int i = 0; i < SEQ * EMBD; i++) {
            float r = bf16_to_f32(residual_host[i]) + ffn_out_f32[i];
            residual_host[i] = f32_to_bf16(r);
        }

        // ---- Step 17: Cross-K/V export (CPU only) ----
        // RMSNorm of post_attn on CPU, then CPU matmul
        // (avoids disturbing AIE pipeline state)
        std::string key_out_path, val_out_path;
        if (!kv_dir.empty()) {
            std::string layer_kv = kv_dir + "/layer_" + std::to_string(layer);
            std::filesystem::create_directories(layer_kv);
            key_out_path = layer_kv + "/key.data";
            val_out_path = layer_kv + "/val.data";
        } else {
            key_out_path = vm["key_out"].as<std::string>();
            val_out_path = vm["val_out"].as<std::string>();
        }
        if (!key_out_path.empty() || !val_out_path.empty()) {
            // Load W_norm_1 and Wk, Wv from files for CPU cross-norm
            // (already in BOs, map them out)
            uint16_t *w1_map = bo_W_norm_1.map<uint16_t*>();
            bo_W_norm_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            uint16_t *wk_map = bo_Wk.map<uint16_t*>();
            bo_Wk.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            uint16_t *wv_map = bo_Wv.map<uint16_t*>();
            bo_Wv.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // CPU RMSNorm: cross_norm[s,e] = (post_attn[s,e] / rms[s]) * W_norm_1[e]
            std::vector<float> cross_norm_f32(SEQ * EMBD);
            for (int s = 0; s < SEQ; s++) {
                float sum_sq = 0.0f;
                for (int e = 0; e < EMBD; e++) {
                    float v = bf16_to_f32(post_attn_host[s * EMBD + e]);
                    sum_sq += v * v;
                }
                float rms_val = std::sqrt(sum_sq / EMBD + 1e-6f);
                for (int e = 0; e < EMBD; e++) {
                    float v = bf16_to_f32(post_attn_host[s * EMBD + e]);
                    cross_norm_f32[s * EMBD + e] = (v / rms_val) * bf16_to_f32(w1_map[e]);
                }
            }

            if (!key_out_path.empty()) {
                std::vector<uint16_t> key_out_host(SEQ * KV_DIM, 0);
                for (int s = 0; s < SEQ; s++)
                    for (int d = 0; d < KV_DIM; d++) {
                        float acc = 0.0f;
                        for (int e = 0; e < EMBD; e++)
                            acc += cross_norm_f32[s * EMBD + e] * bf16_to_f32(wk_map[e * KV_DIM + d]);
                        key_out_host[s * KV_DIM + d] = f32_to_bf16(acc);
                    }
                std::ofstream kf(key_out_path, std::ios::binary);
                kf.write(reinterpret_cast<const char*>(key_out_host.data()),
                         key_out_host.size() * sizeof(uint16_t));
            }

            if (!val_out_path.empty()) {
                std::vector<uint16_t> val_out_host(SEQ * KV_DIM, 0);
                for (int s = 0; s < SEQ; s++)
                    for (int d = 0; d < KV_DIM; d++) {
                        float acc = 0.0f;
                        for (int e = 0; e < EMBD; e++)
                            acc += cross_norm_f32[s * EMBD + e] * bf16_to_f32(wv_map[e * KV_DIM + d]);
                        val_out_host[s * KV_DIM + d] = f32_to_bf16(acc);
                    }
                std::ofstream vf(val_out_path, std::ios::binary);
                vf.write(reinterpret_cast<const char*>(val_out_host.data()),
                         val_out_host.size() * sizeof(uint16_t));
            }
        }
    };

    if (num_layers == 1 && layers_dir.empty()) {
        if (verbosity >= 1) std::cout << "Warmup run...\n";
        run_forward(0, true);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iters; i++) run_forward(0, true);
        auto t1 = std::chrono::high_resolution_clock::now();

        float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
        std::cout << "Text encoder forward: " << total_ms / n_iters << " ms/iter"
                  << " (avg over " << n_iters << " iters)\n";
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int l = 0; l < num_layers; l++)
            run_forward(l, l == 0);
        auto t1 = std::chrono::high_resolution_clock::now();

        float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
        std::cout << "Text encoder forward: " << total_ms << " ms total ("
                  << num_layers << " layers, " << total_ms / num_layers << " ms/layer)\n";
    }

    // ---- Write output ----
    std::ofstream out_f(vm["output"].as<std::string>(), std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(residual_host.data()),
                residual_host.size() * sizeof(uint16_t));
    if (verbosity >= 1)
        std::cout << "Output written to " << vm["output"].as<std::string>() << "\n";

    return 0;
}
