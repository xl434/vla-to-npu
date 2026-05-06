// test.cpp — end-to-end test for the fused im2col + GEMM patch-embedding pipeline.
//
// Builds a FusedPreprocessor, runs it on random bfloat16 data, and validates
// against a float32 CPU reference (direct patch extraction + matmul).
//
// Usage:
//   ./test --xclbin_im2col <path> --instr_im2col <path>
//          --xclbin_gemm   <path> --instr_gemm   <path>
//          [--input_image <path>]        (binary bf16 [3,512,512])
//          [--input_kernel_B <path>]     (binary bf16 [6,768,128] pre-tiled B^T)
//          [--input_kernel_raw <path>]   (binary bf16 [768,3,16,16] — auto-tiled)
//          [--output_file <path>]        (default: output.data)
//          [--verify 1]                  (default: 1)
//          [--test_iter N]               (default: 1)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdfloat>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "fused_unified.h"

namespace po = boost::program_options;
using clk = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
static po::variables_map parse_args(int argc, const char *argv[]) {
    po::options_description opts("Allowed options");
    opts.add_options()
        ("help,h",            "produce help message")
        ("xclbin_im2col",     po::value<std::string>()->required(), "xclbin for im2col kernel")
        ("instr_im2col",      po::value<std::string>()->required(), "instr binary for im2col kernel")
        ("xclbin_gemm",       po::value<std::string>()->required(), "xclbin for GEMM kernel")
        ("instr_gemm",        po::value<std::string>()->required(), "instr binary for GEMM kernel")
        ("input_image",       po::value<std::string>()->default_value(""), "bf16 image [3,512,512]")
        ("input_kernel_B",    po::value<std::string>()->default_value(""), "pre-tiled B^T [6,768,128]")
        ("input_kernel_raw",  po::value<std::string>()->default_value(""), "raw kernel [768,3,16,16]")
        ("output_file",       po::value<std::string>()->default_value("output.data"), "output [1024,768]")
        ("verify",            po::value<bool>()->default_value(true),  "validate against CPU reference")
        ("verbosity,v",       po::value<int>()->default_value(0),      "verbosity")
        ("test_iter,t",       po::value<int>()->default_value(1),      "profiling iterations");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, opts), vm);
        if (vm.count("help")) { std::cout << opts << "\n"; std::exit(0); }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n\nUsage:\n" << opts << "\n";
        std::exit(1);
    }
    return vm;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------
static bool load_bf16(const std::string &path,
                      std::vector<std::bfloat16_t> &buf,
                      size_t expected_elems)
{
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t nb = f.tellg();
    f.seekg(0, std::ios::beg);
    if (nb != expected_elems * 2)
        throw std::runtime_error("File size mismatch: " + path);
    buf.resize(expected_elems);
    f.read(reinterpret_cast<char *>(buf.data()), nb);
    return true;
}

static void save_bf16(const std::string &path, const std::vector<std::bfloat16_t> &buf) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot create: " + path);
    f.write(reinterpret_cast<const char *>(buf.data()),
            buf.size() * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// pre_tile_kernel: raw kernel [EMBD_DIM, K_FULL] → kernel_B_tiles [N_TILES, K_FULL, N_TILE]
// (mirrors pre_tile_kernel in preprocessing_fused_bf16.py)
// ---------------------------------------------------------------------------
static std::vector<std::bfloat16_t>
pre_tile_kernel(const std::vector<std::bfloat16_t> &raw_kernel)
{
    // raw_kernel: [EMBD_DIM=768, CHANNELS=3, KH=16, KW=16] = [768, 768]
    // Treated as [EMBD_DIM, K_FULL] = [768, 768]
    constexpr int EMBD = FU_EMBD_DIM;   // 768
    constexpr int K    = FU_K_FULL;     // 768
    constexpr int NT   = FU_N_TILE;     // 128
    constexpr int NN   = FU_N_TILES;    // 6

    std::vector<std::bfloat16_t> tiles((size_t)NN * K * NT);
    for (int n = 0; n < NN; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < NT; ++j) {
                // tiles[n, k, j] = raw_kernel[n*NT + j, k]   (transpose of [N,K] slice)
                tiles[(size_t)n * K * NT + (size_t)k * NT + j] =
                    raw_kernel[(size_t)(n * NT + j) * K + k];
            }
        }
    }
    return tiles;
}

// ---------------------------------------------------------------------------
// CPU reference: direct patch extraction + matmul (float32 accumulation)
// image: [C, H, W]  kernel_raw: [EMBD, C, KH, KW]
// out:   [SEQ, EMBD]
// ---------------------------------------------------------------------------
static void cpu_reference(
    const std::vector<std::bfloat16_t> &image,
    const std::vector<std::bfloat16_t> &kernel_raw,
    std::vector<float>                 &out)
{
    constexpr int C  = FU_CHANNELS, KH = FU_KH, KW = FU_KW;
    constexpr int H  = FU_PIX_LEN,  W  = FU_PIX_LEN;
    constexpr int PH = FU_PH, PW = FU_PW;
    constexpr int K  = FU_K_FULL, N = FU_EMBD_DIM;

    out.assign((size_t)FU_SEQ * N, 0.0f);
    for (int ph = 0; ph < PH; ++ph) {
        for (int pw = 0; pw < PW; ++pw) {
            int patch_idx = ph * PW + pw;
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int c = 0; c < C; ++c)
                    for (int ky = 0; ky < KH; ++ky)
                        for (int kx = 0; kx < KW; ++kx) {
                            float img_val = static_cast<float>(
                                image[((size_t)c * H + (ph * KH + ky)) * W + (pw * KW + kx)]);
                            float ker_val = static_cast<float>(
                                kernel_raw[((size_t)n * C + c) * KH * KW + ky * KW + kx]);
                            acc += img_val * ker_val;
                        }
                out[(size_t)patch_idx * N + n] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    auto vm = parse_args(argc, argv);

    const std::string xclbin_im2col = vm["xclbin_im2col"].as<std::string>();
    const std::string instr_im2col  = vm["instr_im2col"].as<std::string>();
    const std::string xclbin_gemm   = vm["xclbin_gemm"].as<std::string>();
    const std::string instr_gemm    = vm["instr_gemm"].as<std::string>();
    const bool        verify        = vm["verify"].as<bool>();
    const int         verbosity     = vm["verbosity"].as<int>();
    const int         test_iter     = vm["test_iter"].as<int>();

    // ------------------------------------------------------------------
    // Input data
    // ------------------------------------------------------------------
    constexpr size_t IMAGE_ELEMS  = (size_t)FU_CHANNELS * FU_PIX_LEN * FU_PIX_LEN;
    constexpr size_t KERNEL_ELEMS = (size_t)FU_EMBD_DIM * FU_K_FULL;   // [768,768]
    constexpr size_t KTILE_ELEMS  = (size_t)FU_N_TILES * FU_K_FULL * FU_N_TILE;

    std::vector<std::bfloat16_t> image(IMAGE_ELEMS);
    std::vector<std::bfloat16_t> kernel_raw(KERNEL_ELEMS);
    std::vector<std::bfloat16_t> kernel_B_tiles(KTILE_ELEMS);
    std::vector<std::bfloat16_t> output(FU_OUTPUT_ELEMS, std::bfloat16_t(0.0f));

    // Generate or load image
    if (!load_bf16(vm["input_image"].as<std::string>(), image, IMAGE_ELEMS)) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto &v : image) v = std::bfloat16_t(dist(rng));
        std::cout << "Generated random image.\n";
    }

    // Load or generate raw kernel; tile it
    bool have_raw = false;
    if (!load_bf16(vm["input_kernel_B"].as<std::string>(), kernel_B_tiles, KTILE_ELEMS)) {
        if (!load_bf16(vm["input_kernel_raw"].as<std::string>(), kernel_raw, KERNEL_ELEMS)) {
            std::mt19937 rng(7);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto &v : kernel_raw) v = std::bfloat16_t(dist(rng));
            std::cout << "Generated random kernel.\n";
        }
        kernel_B_tiles = pre_tile_kernel(kernel_raw);
        have_raw = true;
    }

    // ------------------------------------------------------------------
    // Build FusedPreprocessor
    // ------------------------------------------------------------------
    FusedPreprocessor proc(0,
        xclbin_im2col, instr_im2col,
        xclbin_gemm,   instr_gemm,
        verbosity);

    // ------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------
    std::cout << "Warming up...\n";
    proc.run(image.data(), kernel_B_tiles.data(), output.data());

    // ------------------------------------------------------------------
    // Timed run(s)
    // ------------------------------------------------------------------
    double total_wall_us = 0.0;
    for (int it = 0; it < test_iter; ++it) {
        std::fill(output.begin(), output.end(), std::bfloat16_t(0.0f));
        auto t0 = clk::now();
        proc.run(image.data(), kernel_B_tiles.data(), output.data());
        total_wall_us += std::chrono::duration_cast<std::chrono::microseconds>(
            clk::now() - t0).count();
    }
    double avg_wall_ms  = total_wall_us / test_iter / 1000.0;
    double avg_im2col_ms = proc.im2col_time_us() / 1000.0;
    double avg_gemm_ms   = proc.gemm_time_us()   / 1000.0;

    std::cout << "--- Timing (last run) ---\n";
    std::cout << "  im2col (NPU, 128 dispatches): " << avg_im2col_ms  << " ms\n";
    std::cout << "  GEMM   (NPU, 192 dispatches): " << avg_gemm_ms    << " ms\n";
    std::cout << "  wall total (avg "              << test_iter << " iter): "
              << avg_wall_ms << " ms\n";

    // ------------------------------------------------------------------
    // Save output
    // ------------------------------------------------------------------
    save_bf16(vm["output_file"].as<std::string>(), output);
    std::cout << "Saved output to " << vm["output_file"].as<std::string>() << "\n";

    // ------------------------------------------------------------------
    // Verify
    // ------------------------------------------------------------------
    if (verify && have_raw) {
        std::cout << "Running CPU reference...\n";
        std::vector<float> ref;
        cpu_reference(image, kernel_raw, ref);

        double max_err = 0.0, sum_err = 0.0;
        int mismatches = 0;
        for (size_t i = 0; i < FU_OUTPUT_ELEMS; ++i) {
            double npu_val = static_cast<float>(output[i]);
            double err     = std::abs(npu_val - ref[i]);
            sum_err += err;
            if (err > max_err) max_err = err;
            if (err > 1.0 + 0.1 * std::abs(ref[i])) ++mismatches;
        }
        double mean_err = sum_err / FU_OUTPUT_ELEMS;
        std::cout << "Max  abs error: " << max_err  << "\n";
        std::cout << "Mean abs error: " << mean_err << "\n";
        std::cout << "Mismatches (>10% or >1.0): " << mismatches
                  << " / " << FU_OUTPUT_ELEMS << "\n";
        if (mismatches == 0)
            std::cout << "PASSED\n";
        else
            std::cout << "FAILED\n";
    } else if (verify && !have_raw) {
        std::cout << "[verify] Skipped — no raw kernel available (provide --input_kernel_raw).\n";
    }

    return 0;
}
