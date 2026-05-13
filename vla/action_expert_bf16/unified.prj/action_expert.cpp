// action_expert.cpp — Action expert transformer block forward pass for AMD NPU.
//
// Build:
//   cd action_expert_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4
// Run self-attention mode (from action_expert_bf16/unified.prj/):
//   build/action_expert --mode self \
//     --input x.data --W_norm_1 wn1.data --W_norm_2 wn2.data \
//     --Wq wq.data --Wk wk.data --Wv wv.data --Wo wo.data \
//     --W_gate wgate.data --W_up wup.data --W_down wdown.data \
//     --output out.data -v 1
// Run cross-attention mode:
//   build/action_expert --mode cross \
//     --input x.data --W_norm_1 wn1.data --W_norm_2 wn2.data \
//     --Wq wq.data --Wk wk.data --Wv wv.data --Wo wo.data \
//     --W_gate wgate.data --W_up wup.data --W_down wdown.data \
//     --text_k tk.data --text_v tv.data \
//     --output out.data -v 1
//
// Pipeline mirrors action_expert_bf16.py (self and cross attention modes).
// Key dimensions: SEQ=32, TEXT_SEQ=128, EMBD=768, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2048

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
// Model dimensions
// ============================================================
static constexpr int SEQ      = 32;
static constexpr int TEXT_SEQ = 128;
static constexpr int EMBD     = 768;
static constexpr int Q_H      = 15;
static constexpr int KV_H     = 5;
static constexpr int HEAD_DIM = 64;
static constexpr int KV_DIM   = KV_H * HEAD_DIM;    // 320
static constexpr int FFN_HID  = 2048;

static constexpr int FFN_DOWN_K_CHUNK  = 256;
static constexpr int FFN_DOWN_K_CHUNKS = FFN_HID / FFN_DOWN_K_CHUNK;  // 8

// RMSNorm: SEQ=32 = NORM_SEQ_TILE → just 1 call
static constexpr int NORM_SEQ_TILE = 32;
static constexpr int SILU_SEQ_TILE = 16;

// RoPE: fused_5h kernel
static constexpr int ROPE_FUSED_TILE = 32;
static constexpr int ROPE_CHUNK      = 5;

// Attention scale
static constexpr float ATTN_SCALE    = 0.125f;  // 1/sqrt(64)
static constexpr float MAX_WAVELENGTH = 10000.0f;

// ============================================================
// Buffer sizes
// ============================================================
static constexpr size_t SZ_FULL_EMBD  = (size_t)SEQ  * EMBD * 2;       // [32,768] bf16
static constexpr size_t SZ_NORM_W     = EMBD * 2;                        // [768] bf16
static constexpr size_t SZ_NORM_TILE  = (size_t)NORM_SEQ_TILE * EMBD * 2;// [32,768] bf16
static constexpr size_t SZ_Q_OUT      = (size_t)SEQ  * Q_H * HEAD_DIM * 2;  // [32,960] bf16
static constexpr size_t SZ_KV_SELF    = (size_t)SEQ  * KV_DIM * 2;      // [32,320] bf16
static constexpr size_t SZ_KV_CROSS_IN  = (size_t)TEXT_SEQ * KV_DIM * 2;// [128,320] bf16
static constexpr size_t SZ_KV_CROSS_OUT = (size_t)TEXT_SEQ * KV_DIM * 2;// [128,320] bf16
static constexpr size_t SZ_WQ         = (size_t)EMBD * Q_H * HEAD_DIM * 2;  // [768,960] bf16
static constexpr size_t SZ_WKV_SELF   = (size_t)EMBD * KV_DIM * 2;     // [768,320] bf16
static constexpr size_t SZ_WKV_CROSS  = (size_t)KV_DIM * KV_DIM * 2;   // [320,320] bf16
static constexpr size_t SZ_WO         = (size_t)Q_H * HEAD_DIM * EMBD * 2;  // [960,768] bf16

// Attention: self [32,32], cross [32,128]
static constexpr size_t SZ_SCORE_SELF  = (size_t)SEQ  * SEQ * 2;        // [32,32] bf16
static constexpr size_t SZ_SCORE_CROSS = (size_t)SEQ  * TEXT_SEQ * 2;   // [32,128] bf16
static constexpr size_t SZ_Q_HEAD     = (size_t)SEQ  * HEAD_DIM * 2;    // [32,64] bf16
static constexpr size_t SZ_K_HEAD_T_SELF  = (size_t)HEAD_DIM * SEQ * 2; // [64,32] bf16
static constexpr size_t SZ_K_HEAD_T_CROSS = (size_t)HEAD_DIM * TEXT_SEQ * 2;  // [64,128] bf16
static constexpr size_t SZ_V_HEAD_SELF  = (size_t)SEQ  * HEAD_DIM * 2;  // [32,64] bf16
static constexpr size_t SZ_V_HEAD_CROSS = (size_t)TEXT_SEQ * HEAD_DIM * 2;  // [128,64] bf16
static constexpr size_t SZ_CTX_HEAD   = (size_t)SEQ  * HEAD_DIM * 2;   // [32,64] bf16
static constexpr size_t SZ_CTX_FULL   = (size_t)SEQ  * Q_H * HEAD_DIM * 2;  // [32,960] bf16

// FFN
static constexpr size_t SZ_FFN_UP    = (size_t)SEQ  * FFN_HID * 2;     // [32,2048] bf16
static constexpr size_t SZ_SILU_TILE = (size_t)SILU_SEQ_TILE * FFN_HID * 2;  // [16,2048] bf16
static constexpr size_t SZ_FFN_DOWN_A = (size_t)SEQ * FFN_DOWN_K_CHUNK * 2;  // [32,256] bf16
static constexpr size_t SZ_FFN_DOWN_C = SZ_FULL_EMBD;                    // [32,768] bf16
static constexpr size_t SZ_W_DOWN_CHUNK = (size_t)FFN_DOWN_K_CHUNK * EMBD * 2;  // [256,768] bf16
static constexpr size_t SZ_W_FFN_UP  = (size_t)EMBD * FFN_HID * 2;     // [768,2048] bf16

// RoPE
static constexpr size_t SZ_ROPE_X  = (size_t)ROPE_CHUNK * ROPE_FUSED_TILE * HEAD_DIM * 4;  // [160,64] f32
static constexpr size_t SZ_ROPE_SC = (size_t)ROPE_CHUNK * ROPE_FUSED_TILE * HEAD_DIM * 4;

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

    void run2(xrt::bo &b3, xrt::bo &b4) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr); r.set_arg(2, instr_size);
        r.set_arg(3, b3); r.set_arg(4, b4);
        r.start(); r.wait();
    }

    void run3(xrt::bo &b3, xrt::bo &b4, xrt::bo &b5) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr); r.set_arg(2, instr_size);
        r.set_arg(3, b3); r.set_arg(4, b4); r.set_arg(5, b5);
        r.start(); r.wait();
    }
};

// ============================================================
// Helpers
// ============================================================
static xrt::bo make_bo(xrt::device &dev, int g, size_t bytes) {
    return xrt::bo(dev, bytes, XRT_BO_FLAGS_HOST_ONLY, g);
}

static void load_file(const std::string &path, void *buf, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; exit(1); }
    f.seekg(0, std::ios::end);
    if ((size_t)f.tellg() != bytes) {
        std::cerr << path << ": expected " << bytes << " bytes, got " << f.tellg() << "\n";
        exit(1);
    }
    f.seekg(0); f.read(reinterpret_cast<char*>(buf), bytes);
}

static void load_to_bo(xrt::bo &bo, const std::string &path, size_t bytes) {
    load_file(path, bo.map<void*>(), bytes);
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

// Extract [seq, HEAD_DIM] head slice from [seq, n_heads*HEAD_DIM]
static void extract_head(uint16_t *dst, const uint16_t *src, int head, int n_heads, int seq_len) {
    for (int s = 0; s < seq_len; s++)
        memcpy(dst + s * HEAD_DIM,
               src + s * n_heads * HEAD_DIM + head * HEAD_DIM,
               HEAD_DIM * sizeof(uint16_t));
}

// Transpose [seq, HEAD_DIM] → [HEAD_DIM, seq]
static void transpose_head(uint16_t *dst, const uint16_t *src, int seq_len) {
    for (int s = 0; s < seq_len; s++)
        for (int d = 0; d < HEAD_DIM; d++)
            dst[d * seq_len + s] = src[s * HEAD_DIM + d];
}

// Precompute sin/cos [n_rows, head_dim]: first half=sin, second half=cos
static void precompute_sin_cos(float *sc, int n_rows, int head_dim, float max_wl, int pos_off) {
    int half = head_dim / 2;
    for (int r = 0; r < n_rows; r++) {
        float pos = (float)(pos_off + r);
        for (int i = 0; i < half; i++) {
            float inv_ts = std::pow(max_wl, -(2.0f * (float)i / (float)head_dim));
            float rad = pos * inv_ts;
            sc[r * head_dim + i]        = std::sin(rad);
            sc[r * head_dim + half + i] = std::cos(rad);
        }
    }
}

// CPU causal masked softmax for [SEQ, SEQ] bf16 → returns bf16 weights
static std::vector<uint16_t> masked_softmax_cpu(const uint16_t *score, int seq_len) {
    std::vector<uint16_t> weights(seq_len * seq_len);
    static const float NEG_INF = -1e38f;
    for (int r = 0; r < seq_len; r++) {
        // Apply causal mask and find max
        std::vector<float> row(seq_len);
        for (int c = 0; c < seq_len; c++)
            row[c] = (c > r) ? NEG_INF : bf16_to_f32(score[r * seq_len + c]);
        float row_max = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (int c = 0; c < seq_len; c++) {
            row[c] = std::exp(row[c] - row_max);
            sum += row[c];
        }
        for (int c = 0; c < seq_len; c++)
            weights[r * seq_len + c] = f32_to_bf16(row[c] / sum);
    }
    return weights;
}

// ============================================================
// Main
// ============================================================
int main(int argc, const char *argv[]) {
    po::options_description opts("Action expert forward pass");
    opts.add_options()
        ("help,h",       "help")
        ("mode",         po::value<std::string>()->required(),          "self or cross")
        ("input",        po::value<std::string>()->required(),          "[32,768] bf16 input")
        ("W_norm_1",     po::value<std::string>()->required(),          "[768] bf16")
        ("W_norm_2",     po::value<std::string>()->required(),          "[768] bf16")
        ("Wq",           po::value<std::string>()->required(),          "[768,960] bf16")
        ("Wk",           po::value<std::string>()->required(),          "[768,320] (self) or [320,320] (cross) bf16")
        ("Wv",           po::value<std::string>()->required(),          "[768,320] (self) or [320,320] (cross) bf16")
        ("Wo",           po::value<std::string>()->required(),          "[960,768] bf16")
        ("W_gate",       po::value<std::string>()->required(),          "[768,2048] bf16")
        ("W_up",         po::value<std::string>()->required(),          "[768,2048] bf16")
        ("W_down",       po::value<std::string>()->required(),          "[2048,768] bf16")
        ("text_k",       po::value<std::string>()->default_value(""),  "[128,320] bf16 (cross only)")
        ("text_v",       po::value<std::string>()->default_value(""),  "[128,320] bf16 (cross only)")
        ("output",       po::value<std::string>()->default_value("output.data"), "[32,768] bf16")
        ("iters,n",      po::value<int>()->default_value(1),           "forward passes")
        ("verbosity,v",  po::value<int>()->default_value(0),           "verbosity");

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
    std::string mode = vm["mode"].as<std::string>();
    bool is_cross = (mode == "cross");

    const std::string BASE = "..";
    auto xclbin_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/build/final.xclbin";
    };
    auto insts_path = [&](const std::string &name) {
        return BASE + "/" + name + ".prj/insts.txt";
    };

    if (verbosity >= 1) std::cout << "Mode: " << mode << "\nPreloading kernel specs...\n";
    auto device = xrt::device(0);

    KernelSpec spec_rms_norm, spec_gemm_q, spec_gemm_out;
    KernelSpec spec_gemm_ffn_up, spec_gemm_ffn_down, spec_silu, spec_rope_5h;
    KernelSpec spec_gemm_kv_self, spec_attn_self_score, spec_attn_self_value;
    KernelSpec spec_gemm_kv_cross, spec_attn_cross_score, spec_attn_cross_value;
    KernelSpec spec_softmax_cross;

    // Always loaded
    spec_rms_norm.preload  (device, xclbin_path("rms_norm"),        insts_path("rms_norm"));
    spec_gemm_q.preload    (device, xclbin_path("gemm_q"),          insts_path("gemm_q"));
    spec_gemm_out.preload  (device, xclbin_path("gemm_out"),        insts_path("gemm_out"));
    spec_gemm_ffn_up.preload(device,xclbin_path("gemm_ffn_up"),     insts_path("gemm_ffn_up"));
    spec_gemm_ffn_down.preload(device,xclbin_path("gemm_ffn_down"), insts_path("gemm_ffn_down"));
    spec_silu.preload      (device, xclbin_path("silu"),            insts_path("silu"));
    spec_rope_5h.preload   (device, "../rope/fused_5h.prj/build/final.xclbin",
                                    "../rope/fused_5h.prj/insts.txt");

    if (!is_cross) {
        spec_gemm_kv_self.preload    (device, xclbin_path("gemm_kv_self"),          insts_path("gemm_kv_self"));
        spec_attn_self_score.preload (device, xclbin_path("gemm_attn_self_score"),   insts_path("gemm_attn_self_score"));
        spec_attn_self_value.preload (device, xclbin_path("gemm_attn_self_value"),   insts_path("gemm_attn_self_value"));
    } else {
        spec_gemm_kv_cross.preload    (device, xclbin_path("gemm_kv_cross"),         insts_path("gemm_kv_cross"));
        spec_attn_cross_score.preload (device, xclbin_path("gemm_attn_cross_score"), insts_path("gemm_attn_cross_score"));
        spec_attn_cross_value.preload (device, xclbin_path("gemm_attn_cross_value"), insts_path("gemm_attn_cross_value"));
        spec_softmax_cross.preload    (device, xclbin_path("softmax_cross"),          insts_path("softmax_cross"));
    }

    if (verbosity >= 1) std::cout << "All specs loaded.\n";

    int g3, g4, g5;
    {
        ActiveKernel tmp(device, spec_rms_norm);
        g3 = tmp.k.group_id(3);
        g4 = tmp.k.group_id(4);
        g5 = tmp.k.group_id(5);
    }
    if (verbosity >= 1)
        std::cout << "Data group IDs: slot3=" << g3 << " slot4=" << g4 << " slot5=" << g5 << "\n";

    // ---- Allocate BOs ----
    auto bo_rms_in     = make_bo(device, g3, SZ_NORM_TILE);
    auto bo_rms_out    = make_bo(device, g4, SZ_NORM_TILE);
    auto bo_W_norm_1   = make_bo(device, g5, SZ_NORM_W);
    auto bo_W_norm_2   = make_bo(device, g5, SZ_NORM_W);

    auto bo_normed     = make_bo(device, g3, SZ_NORM_TILE);   // [32,768] (same as rms_in)
    auto bo_query      = make_bo(device, g4, SZ_Q_OUT);       // [32,960]
    auto bo_Wq         = make_bo(device, g5, SZ_WQ);

    // Self-attn specific
    auto bo_key_self   = make_bo(device, g4, SZ_KV_SELF);     // [32,320]
    auto bo_val_self   = make_bo(device, g4, SZ_KV_SELF);
    auto bo_Wk_self    = make_bo(device, g5, SZ_WKV_SELF);
    auto bo_Wv_self    = make_bo(device, g5, SZ_WKV_SELF);
    auto bo_Q_head     = make_bo(device, g3, SZ_Q_HEAD);
    auto bo_score_self = make_bo(device, g4, SZ_SCORE_SELF);
    auto bo_K_head_T_self = make_bo(device, g5, SZ_K_HEAD_T_SELF);
    auto bo_ctx_head   = make_bo(device, g4, SZ_CTX_HEAD);
    auto bo_V_head_self = make_bo(device, g5, SZ_V_HEAD_SELF);

    // Cross-attn specific
    auto bo_text_k      = make_bo(device, g3, SZ_KV_CROSS_IN);
    auto bo_text_v      = make_bo(device, g3, SZ_KV_CROSS_IN);
    auto bo_key_cross   = make_bo(device, g4, SZ_KV_CROSS_OUT);
    auto bo_val_cross   = make_bo(device, g4, SZ_KV_CROSS_OUT);
    auto bo_Wk_cross    = make_bo(device, g5, SZ_WKV_CROSS);
    auto bo_Wv_cross    = make_bo(device, g5, SZ_WKV_CROSS);
    auto bo_score_cross = make_bo(device, g4, SZ_SCORE_CROSS);
    auto bo_K_head_T_cross = make_bo(device, g5, SZ_K_HEAD_T_CROSS);
    auto bo_weight_cross = make_bo(device, g4, SZ_SCORE_CROSS);
    auto bo_V_head_cross = make_bo(device, g5, SZ_V_HEAD_CROSS);

    // Output projection
    auto bo_ctx_full   = make_bo(device, g3, SZ_CTX_FULL);    // [32,960]
    auto bo_x_out      = make_bo(device, g4, SZ_FULL_EMBD);   // [32,768]
    auto bo_Wo         = make_bo(device, g5, SZ_WO);

    // FFN
    auto bo_gate_proj  = make_bo(device, g4, SZ_FFN_UP);
    auto bo_up_proj    = make_bo(device, g4, SZ_FFN_UP);
    auto bo_W_gate     = make_bo(device, g5, SZ_W_FFN_UP);
    auto bo_W_up       = make_bo(device, g5, SZ_W_FFN_UP);
    auto bo_silu_in    = make_bo(device, g3, SZ_SILU_TILE);
    auto bo_silu_out   = make_bo(device, g4, SZ_SILU_TILE);
    auto bo_ffn_down_A = make_bo(device, g3, SZ_FFN_DOWN_A);
    auto bo_ffn_down_C = make_bo(device, g4, SZ_FFN_DOWN_C);
    auto bo_W_down_chunk = make_bo(device, g5, SZ_W_DOWN_CHUNK);

    // RoPE
    auto bo_rope_x   = make_bo(device, g3, SZ_ROPE_X);
    auto bo_rope_sc  = make_bo(device, g4, SZ_ROPE_SC);
    auto bo_rope_out = make_bo(device, g5, SZ_ROPE_X);

    // Input
    auto bo_input = make_bo(device, g3, SZ_FULL_EMBD);

    // ---- Load weights ----
    if (verbosity >= 1) std::cout << "Loading weights...\n";
    load_to_bo(bo_W_norm_1, vm["W_norm_1"].as<std::string>(), SZ_NORM_W);
    load_to_bo(bo_W_norm_2, vm["W_norm_2"].as<std::string>(), SZ_NORM_W);
    load_to_bo(bo_Wq,       vm["Wq"].as<std::string>(),       SZ_WQ);
    load_to_bo(bo_Wo,       vm["Wo"].as<std::string>(),       SZ_WO);
    load_to_bo(bo_W_gate,   vm["W_gate"].as<std::string>(),   SZ_W_FFN_UP);
    load_to_bo(bo_W_up,     vm["W_up"].as<std::string>(),     SZ_W_FFN_UP);

    if (!is_cross) {
        load_to_bo(bo_Wk_self, vm["Wk"].as<std::string>(), SZ_WKV_SELF);
        load_to_bo(bo_Wv_self, vm["Wv"].as<std::string>(), SZ_WKV_SELF);
    } else {
        load_to_bo(bo_Wk_cross, vm["Wk"].as<std::string>(), SZ_WKV_CROSS);
        load_to_bo(bo_Wv_cross, vm["Wv"].as<std::string>(), SZ_WKV_CROSS);
        if (!vm["text_k"].as<std::string>().empty())
            load_to_bo(bo_text_k, vm["text_k"].as<std::string>(), SZ_KV_CROSS_IN);
        if (!vm["text_v"].as<std::string>().empty())
            load_to_bo(bo_text_v, vm["text_v"].as<std::string>(), SZ_KV_CROSS_IN);
    }

    // Host buffers
    std::vector<uint16_t> residual_host(SEQ * EMBD, 0);
    std::vector<uint16_t> normed_host(SEQ * EMBD, 0);
    std::vector<uint16_t> key_host(is_cross ? TEXT_SEQ * KV_DIM : SEQ * KV_DIM, 0);
    std::vector<uint16_t> val_host(is_cross ? TEXT_SEQ * KV_DIM : SEQ * KV_DIM, 0);
    std::vector<uint16_t> ctx_full_host(SEQ * Q_H * HEAD_DIM, 0);
    std::vector<uint16_t> gate_proj_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> up_proj_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> act_host(SEQ * FFN_HID, 0);
    std::vector<float>    ffn_out_f32(SEQ * EMBD, 0.0f);
    // RoPE float32 buffers
    std::vector<float> q_f32(SEQ * Q_H * HEAD_DIM, 0.0f);
    std::vector<float> k_f32(SEQ * KV_DIM, 0.0f);

    // RoPE helper lambda
    auto rope_apply = [&](float *packed_f32, int seq_len, int heads, ActiveKernel &ak) {
        float *x_map   = bo_rope_x.map<float*>();
        float *sc_map  = bo_rope_sc.map<float*>();
        float *out_map = bo_rope_out.map<float*>();
        for (int t0 = 0; t0 < seq_len; t0 += ROPE_FUSED_TILE) {
            std::vector<float> sin_cos(ROPE_FUSED_TILE * HEAD_DIM);
            precompute_sin_cos(sin_cos.data(), ROPE_FUSED_TILE, HEAD_DIM, MAX_WAVELENGTH, t0);
            for (int i = 0; i < ROPE_CHUNK; i++)
                memcpy(sc_map + i * ROPE_FUSED_TILE * HEAD_DIM,
                       sin_cos.data(), ROPE_FUSED_TILE * HEAD_DIM * sizeof(float));
            bo_rope_sc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (int h0 = 0; h0 < heads; h0 += ROPE_CHUNK) {
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

    auto run_forward = [&]() {
        // ---- Step 1: Load input → residual ----
        load_to_bo(bo_input, vm["input"].as<std::string>(), SZ_FULL_EMBD);
        memcpy(residual_host.data(), bo_input.map<uint16_t*>(), SZ_FULL_EMBD);

        // ---- Step 2: RMSNorm 1 (1 call: SEQ=32=NORM_SEQ_TILE) ----
        {
            ActiveKernel ak(device, spec_rms_norm);
            memcpy(bo_rms_in.map<uint16_t*>(), residual_host.data(), SZ_NORM_TILE);
            bo_rms_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ak.run3(bo_rms_in, bo_rms_out, bo_W_norm_1);
            bo_rms_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(normed_host.data(), bo_rms_out.map<uint16_t*>(), SZ_NORM_TILE);
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_NORM_TILE);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Step 3: GEMM Wq → query[32,960] ----
        {
            ActiveKernel ak(device, spec_gemm_q);
            ak.run3(bo_normed, bo_query, bo_Wq);
        }
        bo_query.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if (!is_cross) {
            // ---- SELF-ATTENTION ----

            // Steps 4-5: GEMM Wk, Wv (self)
            {
                ActiveKernel ak(device, spec_gemm_kv_self);
                ak.run3(bo_normed, bo_key_self, bo_Wk_self);
                bo_key_self.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(key_host.data(), bo_key_self.map<uint16_t*>(), SZ_KV_SELF);

                ak.run3(bo_normed, bo_val_self, bo_Wv_self);
            }
            bo_val_self.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(val_host.data(), bo_val_self.map<uint16_t*>(), SZ_KV_SELF);

            // RoPE: Q (15 heads) and K (5 heads)
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

                for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++)
                    q_map[i] = f32_to_bf16(q_f32[i]);
                for (int i = 0; i < SEQ * KV_DIM; i++)
                    key_host[i] = f32_to_bf16(k_f32[i]);
            }

            // Scale query
            {
                uint16_t *q_map = bo_query.map<uint16_t*>();
                for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++) {
                    float v = bf16_to_f32(q_map[i]) * ATTN_SCALE;
                    q_map[i] = f32_to_bf16(v);
                }
            }

            // Per-head self-attention (15 heads)
            for (int h = 0; h < Q_H; h++) {
                int kv_idx = h * KV_H / Q_H;
                uint16_t *q_map = bo_query.map<uint16_t*>();

                extract_head(bo_Q_head.map<uint16_t*>(), q_map, h, Q_H, SEQ);
                bo_Q_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                std::vector<uint16_t> K_tmp(SEQ * HEAD_DIM);
                extract_head(K_tmp.data(), key_host.data(), kv_idx, KV_H, SEQ);
                transpose_head(bo_K_head_T_self.map<uint16_t*>(), K_tmp.data(), SEQ);
                bo_K_head_T_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                {
                    ActiveKernel ak(device, spec_attn_self_score);
                    ak.run3(bo_Q_head, bo_score_self, bo_K_head_T_self);
                }
                bo_score_self.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // CPU causal masked softmax
                auto weights = masked_softmax_cpu(bo_score_self.map<uint16_t*>(), SEQ);

                extract_head(bo_V_head_self.map<uint16_t*>(), val_host.data(), kv_idx, KV_H, SEQ);
                bo_V_head_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Copy weights to a BO for gemm_attn_self_value
                // (reuse bo_score_self as attn_weight BO since same shape)
                memcpy(bo_score_self.map<uint16_t*>(), weights.data(), SZ_SCORE_SELF);
                bo_score_self.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                {
                    ActiveKernel ak(device, spec_attn_self_value);
                    ak.run3(bo_score_self, bo_ctx_head, bo_V_head_self);
                }
                bo_ctx_head.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                uint16_t *ctx_map = bo_ctx_head.map<uint16_t*>();
                for (int s = 0; s < SEQ; s++)
                    memcpy(ctx_full_host.data() + s * Q_H * HEAD_DIM + h * HEAD_DIM,
                           ctx_map + s * HEAD_DIM, HEAD_DIM * sizeof(uint16_t));
            }

        } else {
            // ---- CROSS-ATTENTION ----

            // K/V from text: gemm_kv_cross [128,320] @ [320,320] → [128,320]
            {
                ActiveKernel ak(device, spec_gemm_kv_cross);
                ak.run3(bo_text_k, bo_key_cross, bo_Wk_cross);
                bo_key_cross.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(key_host.data(), bo_key_cross.map<uint16_t*>(), SZ_KV_CROSS_OUT);

                ak.run3(bo_text_v, bo_val_cross, bo_Wv_cross);
            }
            bo_val_cross.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(val_host.data(), bo_val_cross.map<uint16_t*>(), SZ_KV_CROSS_OUT);

            // RoPE on Q only for cross-attention
            {
                uint16_t *q_map = bo_query.map<uint16_t*>();
                for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++)
                    q_f32[i] = bf16_to_f32(q_map[i]);

                {
                    ActiveKernel ak(device, spec_rope_5h);
                    rope_apply(q_f32.data(), SEQ, Q_H, ak);
                }

                for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++)
                    q_map[i] = f32_to_bf16(q_f32[i]);
            }

            // Scale query
            {
                uint16_t *q_map = bo_query.map<uint16_t*>();
                for (int i = 0; i < SEQ * Q_H * HEAD_DIM; i++) {
                    float v = bf16_to_f32(q_map[i]) * ATTN_SCALE;
                    q_map[i] = f32_to_bf16(v);
                }
            }

            // Per-head cross-attention (15 heads, Q[32,64] × K_T[64,128] → [32,128])
            for (int h = 0; h < Q_H; h++) {
                int kv_idx = h * KV_H / Q_H;
                uint16_t *q_map = bo_query.map<uint16_t*>();

                extract_head(bo_Q_head.map<uint16_t*>(), q_map, h, Q_H, SEQ);
                bo_Q_head.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // K_head_T: [TEXT_SEQ=128, HEAD_DIM] → [HEAD_DIM, TEXT_SEQ]
                std::vector<uint16_t> K_tmp(TEXT_SEQ * HEAD_DIM);
                extract_head(K_tmp.data(), key_host.data(), kv_idx, KV_H, TEXT_SEQ);
                transpose_head(bo_K_head_T_cross.map<uint16_t*>(), K_tmp.data(), TEXT_SEQ);
                bo_K_head_T_cross.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                {
                    ActiveKernel ak(device, spec_attn_cross_score);
                    ak.run3(bo_Q_head, bo_score_cross, bo_K_head_T_cross);
                }
                bo_score_cross.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // NPU unmasked softmax [32,128]
                bo_score_cross.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                {
                    ActiveKernel ak(device, spec_softmax_cross);
                    ak.run2(bo_score_cross, bo_weight_cross);
                }
                bo_weight_cross.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // V_head: [TEXT_SEQ, HEAD_DIM] = [128, 64]
                extract_head(bo_V_head_cross.map<uint16_t*>(), val_host.data(), kv_idx, KV_H, TEXT_SEQ);
                bo_V_head_cross.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                {
                    ActiveKernel ak(device, spec_attn_cross_value);
                    ak.run3(bo_weight_cross, bo_ctx_head, bo_V_head_cross);
                }
                bo_ctx_head.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                uint16_t *ctx_map = bo_ctx_head.map<uint16_t*>();
                for (int s = 0; s < SEQ; s++)
                    memcpy(ctx_full_host.data() + s * Q_H * HEAD_DIM + h * HEAD_DIM,
                           ctx_map + s * HEAD_DIM, HEAD_DIM * sizeof(uint16_t));
            }
        }

        // ---- Step 8: GEMM Wo → x_out[32,768] ----
        memcpy(bo_ctx_full.map<uint16_t*>(), ctx_full_host.data(), SZ_CTX_FULL);
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

        // ---- Step 10: RMSNorm 2 (1 call) ----
        {
            ActiveKernel ak(device, spec_rms_norm);
            memcpy(bo_rms_in.map<uint16_t*>(), residual_host.data(), SZ_NORM_TILE);
            bo_rms_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            ak.run3(bo_rms_in, bo_rms_out, bo_W_norm_2);
            bo_rms_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(normed_host.data(), bo_rms_out.map<uint16_t*>(), SZ_NORM_TILE);
        }
        memcpy(bo_normed.map<uint16_t*>(), normed_host.data(), SZ_NORM_TILE);
        bo_normed.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ---- Step 11: Gate + Up projections ----
        {
            ActiveKernel ak(device, spec_gemm_ffn_up);
            ak.run3(bo_normed, bo_gate_proj, bo_W_gate);
            bo_gate_proj.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(gate_proj_host.data(), bo_gate_proj.map<uint16_t*>(), SZ_FFN_UP);

            ak.run3(bo_normed, bo_up_proj, bo_W_up);
        }
        bo_up_proj.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(up_proj_host.data(), bo_up_proj.map<uint16_t*>(), SZ_FFN_UP);

        // ---- Step 12: SiLU — 2 tile calls (SEQ=32, tile=16) ----
        {
            ActiveKernel ak(device, spec_silu);
            uint16_t *sin_map  = bo_silu_in.map<uint16_t*>();
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
        // CPU SiLU fallback
        for (int i = 0; i < SEQ * FFN_HID; i++) {
            float g = bf16_to_f32(gate_proj_host[i]);
            if (std::abs(g) > 2.5f) {
                float clipped = g > 20.0f ? 20.0f : (g < -20.0f ? -20.0f : g);
                act_host[i] = f32_to_bf16(g / (1.0f + std::exp(-clipped)));
            }
            float a = bf16_to_f32(act_host[i]);
            if (std::isinf(a) || std::isnan(a)) {
                float clipped = g > 20.0f ? 20.0f : (g < -20.0f ? -20.0f : g);
                act_host[i] = f32_to_bf16(g / (1.0f + std::exp(-clipped)));
            }
        }
        // Hadamard act *= up_proj
        for (int i = 0; i < SEQ * FFN_HID; i++) {
            float v = bf16_to_f32(act_host[i]) * bf16_to_f32(up_proj_host[i]);
            act_host[i] = f32_to_bf16(v);
        }

        // ---- Step 13: FFN Down (8 chunks of K=256) ----
        std::fill(ffn_out_f32.begin(), ffn_out_f32.end(), 0.0f);
        {
            std::ifstream wdown_f(vm["W_down"].as<std::string>(), std::ios::binary);
            if (!wdown_f) { std::cerr << "Cannot open W_down\n"; exit(1); }

            ActiveKernel ak(device, spec_gemm_ffn_down);
            for (int chunk = 0; chunk < FFN_DOWN_K_CHUNKS; chunk++) {
                uint16_t *down_in_map = bo_ffn_down_A.map<uint16_t*>();
                for (int r = 0; r < SEQ; r++)
                    memcpy(down_in_map + r * FFN_DOWN_K_CHUNK,
                           act_host.data() + r * FFN_HID + chunk * FFN_DOWN_K_CHUNK,
                           FFN_DOWN_K_CHUNK * sizeof(uint16_t));
                bo_ffn_down_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

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

        // ---- Step 14: residual += ffn_out ----
        for (int i = 0; i < SEQ * EMBD; i++) {
            float r = bf16_to_f32(residual_host[i]) + ffn_out_f32[i];
            residual_host[i] = f32_to_bf16(r);
        }
    };

    if (verbosity >= 1) std::cout << "Warmup run...\n";
    run_forward();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) run_forward();
    auto t1 = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
    std::cout << "Action expert (" << mode << ") forward: " << total_ms / n_iters
              << " ms/iter (avg over " << n_iters << " iters)\n";

    std::ofstream out_f(vm["output"].as<std::string>(), std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(residual_host.data()),
                residual_host.size() * sizeof(uint16_t));
    if (verbosity >= 1)
        std::cout << "Output written to " << vm["output"].as<std::string>() << "\n";

    return 0;
}
