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
static constexpr int GELU_SEQ_TILE = 16;   // gelu processes 16 rows per call (vectorized Padé, 7/7 degree)

// ============================================================
// Buffer sizes
// ============================================================
static constexpr size_t SZ_FULL      = (size_t)SEQ  * EMBD * 2;           // [1024,768] bf16
static constexpr size_t SZ_NORM_W    = EMBD * 2;                           // [768] bf16
static constexpr size_t SZ_NORM_TILE = (size_t)(NORM_SEQ_TILE + 1) * EMBD * 2; // packed: [33,768] bf16 (input+weight)
static constexpr size_t SZ_Q_HEAD   = (size_t)SEQ  * HEAD_DIM * 2;        // [1024,64] bf16
static constexpr size_t SZ_K_HEAD_T = (size_t)HEAD_DIM * SEQ * 2;         // [64,1024] bf16
// FA (Q_tile=SEQ=1024): slot3=Q[1024,64]+V[1024,64] packed, slot4=O[1024,64], slot5=K_T[64,1024]
// One call per head (12 total); scale=0.125 applied internally.
static constexpr size_t SZ_FA_QV   = 2 * (size_t)SEQ * HEAD_DIM * 2; // Q+V packed = 262144 bytes
static constexpr size_t SZ_FA_O    = (size_t)SEQ * HEAD_DIM * 2;      // [1024,64] bf16 = 131072 bytes
// slot 5: K_T[64,1024] = SZ_K_HEAD_T = 131072 bytes
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

    // flash_attn: slot3=QV, slot4=O, slot5=K_T, arg6=trace (always required by FA kernel)
    void run_fa(xrt::bo &b3, xrt::bo &b4, xrt::bo &b5, xrt::bo &b6) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.set_arg(5, b5);
        r.set_arg(6, b6);
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

// ============================================================
// Main
// ============================================================
int main(int argc, const char *argv[]) {
    po::options_description opts("Vision encoder forward pass");
    opts.add_options()
        ("help,h",       "help")
        ("input",        po::value<std::string>()->required(),            "[1024,768] bf16 input")
        ("W_norm_1",     po::value<std::string>()->default_value(""),     "[768] bf16 layer norm weight 1")
        ("b_norm_1",     po::value<std::string>()->default_value(""),     "[768] bf16 layer norm bias 1")
        ("Wq",           po::value<std::string>()->default_value(""),     "[768,768] bf16")
        ("Wk",           po::value<std::string>()->default_value(""),     "[768,768] bf16")
        ("Wv",           po::value<std::string>()->default_value(""),     "[768,768] bf16")
        ("Wo",           po::value<std::string>()->default_value(""),     "[768,768] bf16")
        ("W_up",         po::value<std::string>()->default_value(""),     "[768,3072] bf16 FFN up weight")
        ("W_norm_2",     po::value<std::string>()->default_value(""),     "[768] bf16 layer norm weight 2")
        ("b_norm_2",     po::value<std::string>()->default_value(""),     "[768] bf16 layer norm bias 2")
        ("W_down",       po::value<std::string>()->default_value(""),     "[3072,768] bf16 FFN down weight")
        ("output",       po::value<std::string>()->default_value("output.data"), "[1024,768] bf16 output")
        ("num-layers,L", po::value<int>()->default_value(1),              "number of transformer layers")
        ("layers-dir",   po::value<std::string>()->default_value(""),     "dir with layer_0/,layer_1/,... weight subdirs")
        ("iters,n",      po::value<int>()->default_value(1),              "number of forward passes (single-layer bench)")
        ("verbosity,v",  po::value<int>()->default_value(0),              "verbosity");

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

    // Resolve weight file: single-layer mode uses CLI args; multi-layer uses layers-dir.
    auto weight_path = [&](const std::string &short_name, const std::string &cli_arg,
                            int layer) -> std::string {
        if (!layers_dir.empty())
            return layers_dir + "/layer_" + std::to_string(layer) + "/" + short_name;
        return vm[cli_arg].as<std::string>();
    };

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

    KernelSpec spec_layer_norm, spec_gemm_embd, spec_gemm_hid, spec_gelu, spec_flash_attn;

    spec_layer_norm.preload (device, xclbin_path("layer_norm_bf16"),    insts_path("layer_norm_bf16"));
    spec_gemm_embd.preload  (device, xclbin_path("gemm_embd_embd_bf16"),insts_path("gemm_embd_embd_bf16"));
    spec_gemm_hid.preload   (device, xclbin_path("gemm_hid_embd_bf16"), insts_path("gemm_hid_embd_bf16"));
    spec_gelu.preload       (device, xclbin_path("gelu_bf16_ffn"),      insts_path("gelu_bf16_ffn"));
    spec_flash_attn.preload (device, xclbin_path("flash_attn_vit"),     insts_path("flash_attn_vit"));

    if (verbosity >= 1) std::cout << "All specs loaded.\n";

    // Discover group IDs (each xclbin may assign different memory groups)
    int g3, g4, g5;
    int fa_g3, fa_g4, fa_g5, g7;
    int gelu_g3, gelu_g4;
    {
        ActiveKernel tmp(device, spec_layer_norm);
        g3 = tmp.k.group_id(3);
        g4 = tmp.k.group_id(4);
        g5 = tmp.k.group_id(5);
    }
    {
        ActiveKernel tmp(device, spec_flash_attn);
        fa_g3 = tmp.k.group_id(3);
        fa_g4 = tmp.k.group_id(4);
        fa_g5 = tmp.k.group_id(5);
        g7    = tmp.k.group_id(7);
    }
    {
        ActiveKernel tmp(device, spec_gelu);
        gelu_g3 = tmp.k.group_id(3);
        gelu_g4 = tmp.k.group_id(4);
    }

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

    // flash_attn: slot3=QV packed, slot4=O, slot5=K_T, arg6=trace (mandatory for FA kernel)
    auto bo_fa_QV    = make_bo(device, fa_g3, SZ_FA_QV);   // Q+V packed = 139264 bytes
    auto bo_fa_O     = make_bo(device, fa_g4, SZ_FA_O);    // [64,64] bf16
    auto bo_fa_K_T   = make_bo(device, fa_g5, SZ_K_HEAD_T);// [64,1024] bf16
    auto bo_fa_trace = make_bo(device, g7,    4);           // minimal trace (always required)

    // gemm_embd_embd again for Wo output
    auto bo_attn_value = make_bo(device, g3, SZ_FULL);      // [1024,768] assembled
    auto bo_x_out      = make_bo(device, g4, SZ_FULL);      // [1024,768] output of gemm

    // gemm_hid_embd: slot3=[1024,768], slot4=[1024,3072], slot5=[768,3072]
    auto bo_ffn_up     = make_bo(device, g4, SZ_FFN_UP);    // [1024,3072]
    auto bo_W_up       = make_bo(device, g5, SZ_W_UP);

    // gelu: slot3=[32,3072], slot4=[32,3072]
    auto bo_gelu_in    = make_bo(device, gelu_g3, SZ_FFN_TILE);
    auto bo_gelu_out   = make_bo(device, gelu_g4, SZ_FFN_TILE);

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
    // Weights loaded per-layer inside run_forward(); W_down loaded in chunks per-call.

    // Host staging buffers
    std::vector<uint16_t> residual_host(SEQ * EMBD, 0);
    std::vector<uint16_t> normed_host(SEQ * EMBD, 0);
    std::vector<uint16_t> key_host(SEQ * EMBD, 0);
    std::vector<uint16_t> value_host(SEQ * EMBD, 0);
    std::vector<uint16_t> attn_value_host(SEQ * EMBD, 0);
    std::vector<uint16_t> ffn_up_host(SEQ * FFN_HID, 0);
    std::vector<uint16_t> act_host(SEQ * FFN_HID, 0);
    std::vector<float>    ffn_out_f32(SEQ * EMBD, 0.0f);

    // Per-head host buffers for flash attention pre-pass.
    // K_T and V are extracted once per head, reused across all 16 Q_tiles of that head.
    std::vector<std::vector<uint16_t>> K_head_Ts_host(N_HEAD, std::vector<uint16_t>(HEAD_DIM * SEQ));
    std::vector<std::vector<uint16_t>> V_heads_host(N_HEAD, std::vector<uint16_t>(SEQ * HEAD_DIM));

    using Clock = std::chrono::high_resolution_clock;
    auto ms_since = [](Clock::time_point t0) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - t0).count() / 1000.0f;
    };
    auto span_ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0f;
    };

    auto run_forward = [&](int layer = 0, bool first_layer = true) {
        auto p0 = Clock::now();

        // Load per-layer weights (always, since different layers have different weights)
        load_file_to_bo(bo_W_norm_1, weight_path("w1.data",    "W_norm_1", layer), SZ_NORM_W);
        load_file_to_bo(bo_b_norm_1, weight_path("b1.data",    "b_norm_1", layer), SZ_NORM_W);
        load_file_to_bo(bo_Wq,       weight_path("wq.data",    "Wq",       layer), SZ_WQ);
        load_file_to_bo(bo_Wk,       weight_path("wk.data",    "Wk",       layer), SZ_WQ);
        load_file_to_bo(bo_Wv,       weight_path("wv.data",    "Wv",       layer), SZ_WQ);
        load_file_to_bo(bo_Wo,       weight_path("wo.data",    "Wo",       layer), SZ_WQ);
        load_file_to_bo(bo_W_up,     weight_path("wup.data",   "W_up",     layer), SZ_W_UP);
        load_file_to_bo(bo_W_norm_2, weight_path("w2.data",    "W_norm_2", layer), SZ_NORM_W);
        load_file_to_bo(bo_b_norm_2, weight_path("b2.data",    "b_norm_2", layer), SZ_NORM_W);
        auto p_wl = Clock::now();

        // ---- Step 1: Load input → residual (first layer only) ----
        if (first_layer) {
            load_file_to_bo(bo_input, vm["input"].as<std::string>(), SZ_FULL);
            memcpy(residual_host.data(), bo_input.map<uint16_t*>(), SZ_FULL);
        }
        auto p_input = Clock::now();

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
        auto p_n1 = Clock::now();

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
        auto p_qkv = Clock::now();

        // ---- Step 6-7: Flash attention — 1 hw_context, 12×16=192 dispatches ----
        // FA handles the 0.125 scale internally; no pre-scaling of Q needed.
        // CPU pre-pass: extract K_T and V for each head (reused across all 16 Q-tiles).
        uint16_t *query_map = bo_query.map<uint16_t*>();
        {
            std::vector<uint16_t> K_tmp(SEQ * HEAD_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                extract_head(K_tmp.data(), key_host.data(), h, N_HEAD);
                transpose_head(K_head_Ts_host[h].data(), K_tmp.data());
                extract_head(V_heads_host[h].data(), value_host.data(), h, N_HEAD);
            }
        }

        auto p_fa_pre = Clock::now();

        // FA: 1 hw_context, 12 dispatches (one per head, full SEQ per call)
        {
            ActiveKernel ak(device, spec_flash_attn);
            uint16_t *fa_qv_map = bo_fa_QV.map<uint16_t*>();
            uint16_t *fa_o_map  = bo_fa_O.map<uint16_t*>();
            for (int h = 0; h < N_HEAD; h++) {
                // Pack K_T for this head into slot 5
                memcpy(bo_fa_K_T.map<uint16_t*>(), K_head_Ts_host[h].data(), SZ_K_HEAD_T);
                bo_fa_K_T.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                // Pack Q[SEQ,HEAD_DIM] at offset 0 in slot 3
                extract_head(fa_qv_map, query_map, h, N_HEAD);
                // Pack V[SEQ,HEAD_DIM] at offset SEQ*HEAD_DIM in slot 3
                memcpy(fa_qv_map + SEQ * HEAD_DIM,
                       V_heads_host[h].data(), SZ_Q_HEAD);
                bo_fa_QV.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run_fa(bo_fa_QV, bo_fa_O, bo_fa_K_T, bo_fa_trace);
                bo_fa_O.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                // Scatter O[SEQ,HEAD_DIM] → attn_value_host[SEQ, EMBD]
                for (int s = 0; s < SEQ; s++)
                    memcpy(attn_value_host.data() + s * EMBD + h * HEAD_DIM,
                           fa_o_map + s * HEAD_DIM,
                           HEAD_DIM * sizeof(uint16_t));
            }
        }

        auto p_fa = Clock::now();

        // ---- Step 8: GEMM Wo → x_out[1024,768] ----
        memcpy(bo_attn_value.map<uint16_t*>(), attn_value_host.data(), SZ_FULL);
        bo_attn_value.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        {
            ActiveKernel ak(device, spec_gemm_embd);
            ak.run3(bo_attn_value, bo_x_out, bo_Wo);
        }
        bo_x_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto p_wo = Clock::now();

        // ---- Step 9: residual += x_out (CPU bf16 add) ----
        {
            uint16_t *x_out_map = bo_x_out.map<uint16_t*>();
            for (int i = 0; i < SEQ * EMBD; i++) {
                float r = bf16_to_f32(residual_host[i]) + bf16_to_f32(x_out_map[i]);
                residual_host[i] = f32_to_bf16(r);
            }
        }
        auto p_res1 = Clock::now();

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
        auto p_n2 = Clock::now();

        // ---- Step 11: GEMM W_up → ffn_up_x[1024,3072] ----
        {
            ActiveKernel ak(device, spec_gemm_hid);
            ak.run3(bo_normed, bo_ffn_up, bo_W_up);
        }
        bo_ffn_up.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(ffn_up_host.data(), bo_ffn_up.map<uint16_t*>(), SZ_FFN_UP);
        auto p_up = Clock::now();

        // ---- Step 12: GELU — 32 tile calls ----
        // Optimization: Allocate persistent BO for full [1024, 3072] GELU output
        // to avoid host round-trip and per-tile sync overhead
        auto bo_gelu_full = make_bo(device, gelu_g4, (size_t)SEQ * FFN_HID * 2);
        uint16_t *gelu_full_host = (uint16_t*)malloc((size_t)SEQ * FFN_HID * 2);
        {
            ActiveKernel ak(device, spec_gelu);
            uint16_t *gin_map  = bo_gelu_in.map<uint16_t*>();
            for (int t = 0; t < SEQ / GELU_SEQ_TILE; t++) {
                memcpy(gin_map, ffn_up_host.data() + t * GELU_SEQ_TILE * FFN_HID,
                       GELU_SEQ_TILE * FFN_HID * sizeof(uint16_t));
                bo_gelu_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                ak.run2(bo_gelu_in, bo_gelu_out);
                bo_gelu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                memcpy(gelu_full_host + t * GELU_SEQ_TILE * FFN_HID,
                       bo_gelu_out.map<uint16_t*>(), GELU_SEQ_TILE * FFN_HID * sizeof(uint16_t));
            }
        }
        // Now copy full result to device BO once
        memcpy(bo_gelu_full.map<uint16_t*>(), gelu_full_host, (size_t)SEQ * FFN_HID * 2);
        bo_gelu_full.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Keep gelu_full_host for GEMM_down chunk access (avoid extra sync)
        auto p_gelu = Clock::now();

        // ---- Step 13: FFN Down (4 chunks), float32 accumulation ----
        std::fill(ffn_out_f32.begin(), ffn_out_f32.end(), 0.0f);
        float fdn_disk_ms = 0.0f, fdn_gemm_ms = 0.0f;
        {
            std::ifstream wdown_f(weight_path("wdown.data", "W_down", layer), std::ios::binary);
            if (!wdown_f) { std::cerr << "Cannot open W_down\n"; exit(1); }

            ActiveKernel ak(device, spec_gemm_embd);
            for (int chunk = 0; chunk < FFN_DOWN_K_CHUNKS; chunk++) {
                auto ck0 = Clock::now();

                // Copy act chunk [1024, 768] → bo_ffn_down_A
                // (read from gelu_full_host which is already in memory)
                uint16_t *down_in_map = bo_ffn_down_A.map<uint16_t*>();
                for (int r = 0; r < SEQ; r++)
                    memcpy(down_in_map + r * FFN_DOWN_K_CHUNK,
                           gelu_full_host + r * FFN_HID + chunk * FFN_DOWN_K_CHUNK,
                           FFN_DOWN_K_CHUNK * sizeof(uint16_t));
                bo_ffn_down_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Load W_down chunk [768, 768] from disk
                wdown_f.seekg((size_t)chunk * FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                wdown_f.read(bo_W_down_chunk.map<char*>(),
                             FFN_DOWN_K_CHUNK * EMBD * sizeof(uint16_t));
                bo_W_down_chunk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto ck1 = Clock::now();

                ak.run3(bo_ffn_down_A, bo_ffn_down_C, bo_W_down_chunk);
                bo_ffn_down_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                // Accumulate in float32
                uint16_t *c_map = bo_ffn_down_C.map<uint16_t*>();
                for (int i = 0; i < SEQ * EMBD; i++)
                    ffn_out_f32[i] += bf16_to_f32(c_map[i]);
                auto ck2 = Clock::now();

                fdn_disk_ms += span_ms(ck0, ck1);
                fdn_gemm_ms += span_ms(ck1, ck2);
            }
        }
        auto p_fdn_end = Clock::now();

        // ---- Step 14: residual += ffn_out_f32 (convert back to bf16) ----
        for (int i = 0; i < SEQ * EMBD; i++) {
            float r = bf16_to_f32(residual_host[i]) + ffn_out_f32[i];
            residual_host[i] = f32_to_bf16(r);
        }
        // Cleanup GELU output buffer
        free(gelu_full_host);
        auto p_res2 = Clock::now();

        if (verbosity >= 2) {
            std::printf("[profile layer %d]\n", layer);
            std::printf("  weight_load  : %6.1f ms\n",          span_ms(p0,       p_wl));
            std::printf("  input_load   : %6.1f ms  (first layer only)\n", span_ms(p_wl, p_input));
            std::printf("  norm1        : %6.1f ms  (32 tiles)\n",         span_ms(p_input,  p_n1));
            std::printf("  gemm_qkv     : %6.1f ms  (3x[1024,768]^2)\n",  span_ms(p_n1,     p_qkv));
            std::printf("  fa_prepass   : %6.1f ms  (cpu: extract K_T,V x %d heads)\n", span_ms(p_qkv, p_fa_pre), N_HEAD);
            std::printf("  fa_dispatch  : %6.1f ms  (%d FA calls)\n",     span_ms(p_fa_pre, p_fa),   N_HEAD);
            std::printf("  gemm_wo      : %6.1f ms\n",                     span_ms(p_fa,     p_wo));
            std::printf("  residual1    : %6.1f ms\n",                     span_ms(p_wo,     p_res1));
            std::printf("  norm2        : %6.1f ms  (32 tiles)\n",         span_ms(p_res1,   p_n2));
            std::printf("  gemm_up      : %6.1f ms  ([1024,768]x[768,3072])\n", span_ms(p_n2, p_up));
            std::printf("  gelu         : %6.1f ms  (32 tiles)\n",         span_ms(p_up,     p_gelu));
            std::printf("  fdn_load     : %6.1f ms  (4 W_down chunks disk+act copy)\n", fdn_disk_ms);
            std::printf("  fdn_compute  : %6.1f ms  (4 gemm chunks + f32 accum)\n",     fdn_gemm_ms);
            std::printf("  residual2    : %6.1f ms\n",                     span_ms(p_fdn_end, p_res2));
            std::printf("  TOTAL        : %6.1f ms\n",                     span_ms(p0,        p_res2));
        }
    };

    if (num_layers == 1 && layers_dir.empty()) {
        // ---- Single-layer benchmark mode (backward compatible) ----
        if (verbosity >= 1) std::cout << "Warmup run...\n";
        run_forward(0, true);  // warmup

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iters; i++) run_forward(0, true);
        auto t1 = std::chrono::high_resolution_clock::now();

        float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
        std::cout << "Vision encoder forward: " << total_ms / n_iters << " ms/iter"
                  << " (avg over " << n_iters << " iters)\n";
    } else {
        // ---- Multi-layer mode: run all layers in one process ----
        if (verbosity >= 1)
            std::cout << "Running " << num_layers << " layers from " << layers_dir << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int l = 0; l < num_layers; l++)
            run_forward(l, l == 0);
        auto t1 = std::chrono::high_resolution_clock::now();

        float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
        std::cout << "Vision encoder forward: " << total_ms << " ms total"
                  << " (" << num_layers << " layers, "
                  << total_ms / num_layers << " ms/layer)\n";
    }

    // ---- Write output ----
    std::ofstream out_f(vm["output"].as<std::string>(), std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(residual_host.data()),
                residual_host.size() * sizeof(uint16_t));
    if (verbosity >= 1)
        std::cout << "Output written to " << vm["output"].as<std::string>() << "\n";

    return 0;
}
