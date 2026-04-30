// test.cpp — End-to-end test for the connector block pipeline.
//
// Builds a Connector, runs it on random bfloat16 data, and validates the
// output against a float32 CPU reference implementation.
//
// CPU reference (matches Python):
//   A_[i, j*3072:(j+1)*3072] = flatten(A[offset+j*32 : offset+j*32+4, :])
//   C = A_ @ W           (float32 matmul, [64,12288] × [12288,960] → [64,960])
//
// Usage:
//   ./test --xclbin_copy <path> --instr_copy <path>
//          --xclbin_gemm <path> --instr_gemm <path>
//          [--input_A <path>]     (binary bf16 [1024,768], generated randomly if omitted)
//          [--input_W <path>]     (binary bf16 [12288,960], generated randomly if omitted)
//          [--output_file <path>] (default: output_C.data, float32)
//          [--verify 1]           (default: 1)
//          [--verbosity N]        (default: 0)
//          [--profile 1 --test_iter N]

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdfloat>
#include <string>
#include <vector>

#include "test_utils.h"
#include "unified.h"

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
static po::variables_map parse_args(int argc, const char *argv[]) {
    po::options_description opts("Allowed options");
    opts.add_options()
        ("help,h",         "produce help message")
        ("xclbin_copy",    po::value<std::string>()->required(), "xclbin for copy (pixel-shuffle) kernel")
        ("instr_copy",     po::value<std::string>()->required(), "instruction binary for copy kernel")
        ("xclbin_gemm",    po::value<std::string>()->required(), "xclbin for gemm kernel")
        ("instr_gemm",     po::value<std::string>()->required(), "instruction binary for gemm kernel")
        ("input_A",        po::value<std::string>()->default_value(""),
            "binary bf16 activation [SEQ=1024, EMBD=768] — generated randomly if omitted")
        ("input_W",        po::value<std::string>()->default_value(""),
            "binary bf16 weight [NEW_EMBD=12288, TEXT=960] — generated randomly if omitted")
        ("output_file",    po::value<std::string>()->default_value("output_C.data"),
            "output file [NEW_SEQ=64, TEXT=960] float32")
        ("verify",         po::value<bool>()->default_value(true),  "validate against CPU reference")
        ("verbosity,v",    po::value<int>()->default_value(0),      "verbosity level")
        ("warmup",         po::value<int>()->default_value(0),      "warmup iterations")
        ("trace_sz",       po::value<int>()->default_value(0),      "AIE trace buffer size")
        ("trace_file",     po::value<std::string>()->default_value("trace.txt"), "trace output file")
        ("profile,p",      po::value<bool>()->default_value(false), "enable profiling")
        ("test_iter,t",    po::value<int>()->default_value(1),      "profiling iterations");

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
                      size_t expected_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes != expected_elems * sizeof(std::bfloat16_t)) {
        std::cerr << path << ": expected " << expected_elems * 2
                  << " bytes, got " << bytes << "\n";
        return false;
    }
    buf.resize(expected_elems);
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return true;
}

static void save_f32(const std::string &path, const float *data, size_t elems) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "Cannot write: " << path << "\n"; return; }
    f.write(reinterpret_cast<const char *>(data), elems * sizeof(float));
}

// ---------------------------------------------------------------------------
// CPU reference (float32 throughout)
// ---------------------------------------------------------------------------
static std::vector<float> cpu_reference(
    const std::bfloat16_t *A,
    const std::bfloat16_t *W)
{
    // pixel shuffle
    std::vector<float> A_shuf(CONN_NEW_SEQ * CONN_NEW_EMBD, 0.0f);
    for (int i = 0; i < CONN_NEW_SEQ; ++i) {
        int offset = (i / 8) * 128 + (i % 8) * 4;
        for (int j = 0; j < 4; ++j) {
            int src_row = offset + j * 32;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < CONN_EMBD; ++c)
                    A_shuf[(size_t)i * CONN_NEW_EMBD + j * CONN_EMBD * 4 + r * CONN_EMBD + c] =
                        (float)A[(size_t)(src_row + r) * CONN_EMBD + c];
        }
    }

    // float32 matmul
    std::vector<float> C(CONN_OUT_ELEMS, 0.0f);
    for (int i = 0; i < CONN_NEW_SEQ; ++i)
        for (int m = 0; m < CONN_NEW_EMBD; ++m) {
            float a_val = A_shuf[(size_t)i * CONN_NEW_EMBD + m];
            for (int k = 0; k < CONN_TEXT; ++k)
                C[(size_t)i * CONN_TEXT + k] +=
                    a_val * (float)W[(size_t)m * CONN_TEXT + k];
        }
    return C;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    auto vm = parse_args(argc, argv);

    int  verbosity = vm["verbosity"].as<int>();
    int  trace_sz  = vm["trace_sz"].as<int>();
    bool do_verify = vm["verify"].as<bool>();
    bool do_prof   = vm["profile"].as<bool>();
    int  n_iter    = vm["test_iter"].as<int>();
    int  n_warmup  = vm["warmup"].as<int>();

    // load or generate inputs
    std::vector<std::bfloat16_t> A_data(CONN_A_ELEMS);
    std::vector<std::bfloat16_t> W_data(CONN_W_ELEMS);

    auto a_path = vm["input_A"].as<std::string>();
    auto w_path = vm["input_W"].as<std::string>();

    if (!a_path.empty()) {
        if (!load_bf16(a_path, A_data, CONN_A_ELEMS)) return 1;
        if (verbosity >= 1) std::cout << "Loaded A from " << a_path << "\n";
    } else {
        srand(42);
        for (auto &v : A_data)
            v = std::bfloat16_t((float)rand() / (float)RAND_MAX);
        if (verbosity >= 1) std::cout << "Generated random A (" << CONN_A_ELEMS << " bf16)\n";
    }

    if (!w_path.empty()) {
        if (!load_bf16(w_path, W_data, CONN_W_ELEMS)) return 1;
        if (verbosity >= 1) std::cout << "Loaded W from " << w_path << "\n";
    } else {
        srand(43);
        for (auto &v : W_data)
            v = std::bfloat16_t((float)rand() / (float)RAND_MAX);
        if (verbosity >= 1) std::cout << "Generated random W (" << CONN_W_ELEMS << " bf16)\n";
    }

    // float32 output (NPU accumulates in f32 on CPU)
    std::vector<float> output(CONN_OUT_ELEMS, 0.0f);

    // build Connector (no xclbin_add needed)
    std::cout << "Initialising Connector...\n";
    Connector conn(
        /*device_index=*/ 0,
        vm["xclbin_copy"].as<std::string>(), vm["instr_copy"].as<std::string>(),
        vm["xclbin_gemm"].as<std::string>(), vm["instr_gemm"].as<std::string>(),
        trace_sz, verbosity);
    std::cout << "Connector ready.\n";

    auto run_once = [&]() {
        conn.run(A_data.data(), W_data.data(), output.data());
    };

    if (!do_prof) {
        std::cout << "Running single-shot pipeline...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "Wall clock time:       " << wall_us / 1e3 << " ms\n";
        std::cout << "NPU kernel time:       " << conn.npu_time_us() / 1e3 << " ms\n";
        std::cout << "Overhead (ctx+IO+CPU): " << (wall_us - conn.npu_time_us()) / 1e3 << " ms\n";
    } else {
        if (verbosity >= 1) std::cout << "Warmup (" << n_warmup << " iters)...\n";
        for (int i = 0; i < n_warmup; ++i) run_once();

        if (verbosity >= 1) std::cout << "Profiling (" << n_iter << " iters)...\n";
        double total_wall = 0.0, total_npu = 0.0, wall_min = 1e12;
        for (int i = 0; i < n_iter; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_once();
            auto t1 = std::chrono::high_resolution_clock::now();
            double wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            total_wall += wall_us;
            total_npu  += conn.npu_time_us();
            wall_min    = std::min(wall_min, wall_us);
        }
        std::cout << "Avg wall clock time:   " << total_wall / n_iter / 1e3 << " ms\n";
        std::cout << "Avg NPU kernel time:   " << total_npu  / n_iter / 1e3 << " ms\n";
        std::cout << "Min wall clock time:   " << wall_min / 1e3 << " ms\n";
    }

    save_f32(vm["output_file"].as<std::string>(), output.data(), CONN_OUT_ELEMS);
    if (verbosity >= 1)
        std::cout << "Output written to " << vm["output_file"].as<std::string>() << "\n";

    if (!do_verify) {
        std::cout << "Skipping verification (--verify 0).\n";
        return 0;
    }

    std::cout << "Computing CPU float32 reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> ref = cpu_reference(A_data.data(), W_data.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    float ref_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
    std::cout << "CPU reference time: " << ref_ms << " ms\n";

    double max_err = 0.0, sum_err = 0.0, max_ref = 0.0;
    size_t n_fail = 0;
    constexpr float rtol = 0.1f;

    for (size_t i = 0; i < CONN_OUT_ELEMS; ++i) {
        double out_f   = (double)output[i];
        double ref_f   = (double)ref[i];
        double abs_err = std::abs(out_f - ref_f);
        double thresh  = rtol * std::abs(ref_f);
        if (abs_err > thresh + 1e-5) ++n_fail;
        max_err  = std::max(max_err, abs_err);
        sum_err += abs_err;
        max_ref  = std::max(max_ref, std::abs(ref_f));
    }
    double mean_err = sum_err / CONN_OUT_ELEMS;

    std::cout << "Output shape:      [" << CONN_NEW_SEQ << ", " << CONN_TEXT << "]\n";
    std::cout << "Max reference val: " << max_ref  << "\n";
    std::cout << "Max absolute err:  " << max_err  << "\n";
    std::cout << "Mean absolute err: " << mean_err << "\n";
    std::cout << "Elements outside " << (int)(rtol * 100) << "% rtol: "
              << n_fail << " / " << CONN_OUT_ELEMS << "\n";

    if (n_fail == 0) {
        std::cout << "PASS: all outputs within tolerance.\n";
        return 0;
    } else {
        std::cerr << "FAIL: " << n_fail << " elements outside tolerance.\n";
        return 1;
    }
}
