// test.cpp — End-to-end test for the conv2d patch-embedding pipeline.
//
// Builds a Preprocessor, runs it on random bfloat16 data, and validates
// the output against a float32 CPU reference implementation of the same
// patch-embedding conv2d.
//
// Usage:
//   ./test --xclbin_conv <path> --instr_conv <path>
//          --xclbin_add  <path> --instr_add  <path>
//          --xclbin_copy <path> --instr_copy <path>
//          [--input_image <path>]   (binary bf16, default: generated randomly)
//          [--input_kernel <path>]  (binary bf16, default: generated randomly)
//          [--output_file <path>]   (default: output.data)
//          [--verify 1]             (default: 1, set 0 to skip)
//          [--profile 1]            (default: 0)
//          [--test_iter N]          (profiling iterations, default: 1)

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
// Argument parsing (does NOT call add_default_options — that helper forces
// --xclbin / --kernel / --instr as required, which we don't use here)
// ---------------------------------------------------------------------------
static po::variables_map parse_args(int argc, const char *argv[]) {
    po::options_description opts("Allowed options");
    opts.add_options()
        ("help,h",         "produce help message")
        ("xclbin_conv",    po::value<std::string>()->required(), "xclbin for conv kernel")
        ("instr_conv",     po::value<std::string>()->required(), "instruction binary for conv kernel")
        ("xclbin_add",     po::value<std::string>()->required(), "xclbin for add kernel")
        ("instr_add",      po::value<std::string>()->required(), "instruction binary for add kernel")
        ("xclbin_copy",    po::value<std::string>()->required(), "xclbin for copy kernel")
        ("instr_copy",     po::value<std::string>()->required(), "instruction binary for copy kernel")
        ("input_image",    po::value<std::string>()->default_value(""),
            "binary bf16 image file [3,512,512] — generated randomly if omitted")
        ("input_kernel",   po::value<std::string>()->default_value(""),
            "binary bf16 kernel file [768,3,16,16] — generated randomly if omitted")
        ("output_file",    po::value<std::string>()->default_value("output.data"),
            "output file [1024,768] bf16")
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
    if (!f.is_open()) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
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

static void save_bf16(const std::string &path,
                      const std::bfloat16_t *data, size_t elems) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot write: " << path << "\n";
        return;
    }
    f.write(reinterpret_cast<const char *>(data), elems * sizeof(std::bfloat16_t));
}

// ---------------------------------------------------------------------------
// CPU reference: float32 conv2d patch embedding
//   output[pr*PS+pc, i] = Σ_{j,ky,kx} image[j,pr*KD+ky,pc*KD+kx]*kernel[i,j,ky,kx]
// ---------------------------------------------------------------------------
static std::vector<float> cpu_reference(
    const std::bfloat16_t *image,
    const std::bfloat16_t *kernel)
{
    constexpr int PS  = PREPROC_PATCHES_SIDE;  // 32
    constexpr int KD  = PREPROC_KERNEL_DIM;    // 16
    constexpr int C   = PREPROC_CHANNELS;      // 3
    constexpr int OC  = PREPROC_EMBD_DIM;      // 768
    constexpr int PL  = PREPROC_PIX_LEN;       // 512

    std::vector<float> ref(PREPROC_OUTPUT_ELEMS, 0.0f);

    for (int i = 0; i < OC; ++i) {
        for (int pr = 0; pr < PS; ++pr) {
            for (int pc = 0; pc < PS; ++pc) {
                float acc = 0.0f;
                for (int j = 0; j < C; ++j) {
                    for (int ky = 0; ky < KD; ++ky) {
                        for (int kx = 0; kx < KD; ++kx) {
                            float img_v = (float)image[
                                (size_t)j * PL * PL
                                + (size_t)(pr * KD + ky) * PL
                                + pc * KD + kx];
                            float ker_v = (float)kernel[
                                ((size_t)i * C + j) * KD * KD
                                + ky * KD + kx];
                            acc += img_v * ker_v;
                        }
                    }
                }
                ref[(size_t)(pr * PS + pc) * OC + i] = acc;
            }
        }
    }
    return ref;
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

    // ---- load or generate input data ----
    std::vector<std::bfloat16_t> image(PREPROC_IMAGE_ELEMS);
    std::vector<std::bfloat16_t> kernel(PREPROC_KERNEL_ELEMS);

    auto img_path = vm["input_image"].as<std::string>();
    auto ker_path = vm["input_kernel"].as<std::string>();

    if (!img_path.empty()) {
        if (!load_bf16(img_path, image, PREPROC_IMAGE_ELEMS)) return 1;
        if (verbosity >= 1) std::cout << "Loaded image from " << img_path << "\n";
    } else {
        srand(42);
        for (auto &v : image)
            v = std::bfloat16_t((float)rand() / (float)RAND_MAX);
        if (verbosity >= 1) std::cout << "Generated random image (" << PREPROC_IMAGE_ELEMS << " bf16)\n";
    }

    if (!ker_path.empty()) {
        if (!load_bf16(ker_path, kernel, PREPROC_KERNEL_ELEMS)) return 1;
        if (verbosity >= 1) std::cout << "Loaded kernel from " << ker_path << "\n";
    } else {
        srand(43);
        for (auto &v : kernel)
            v = std::bfloat16_t((float)rand() / (float)RAND_MAX);
        if (verbosity >= 1) std::cout << "Generated random kernel (" << PREPROC_KERNEL_ELEMS << " bf16)\n";
    }

    std::vector<std::bfloat16_t> output(PREPROC_OUTPUT_ELEMS, std::bfloat16_t(0.0f));

    // ---- build Preprocessor ----
    std::cout << "Initialising Preprocessor...\n";
    Preprocessor prep(
        /*device_index=*/ 0,
        vm["xclbin_conv"].as<std::string>(), vm["instr_conv"].as<std::string>(),
        vm["xclbin_add"].as<std::string>(),  vm["instr_add"].as<std::string>(),
        vm["xclbin_copy"].as<std::string>(), vm["instr_copy"].as<std::string>(),
        trace_sz, verbosity);
    std::cout << "Preprocessor ready.\n";

    // ---- run pipeline ----
    auto run_once = [&]() {
        prep.run(image.data(), kernel.data(), output.data());
    };

    if (!do_prof) {
        std::cout << "Running single-shot pipeline...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "Wall clock time:       " << wall_us / 1e3 << " ms\n";
        std::cout << "NPU kernel time:       " << prep.npu_time_us() / 1e3 << " ms\n";
        std::cout << "Overhead (ctx+IO+CPU): " << (wall_us - prep.npu_time_us()) / 1e3 << " ms\n";
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
            total_npu  += prep.npu_time_us();
            wall_min    = std::min(wall_min, wall_us);
        }
        std::cout << "Avg wall clock time:   " << total_wall / n_iter / 1e3 << " ms\n";
        std::cout << "Avg NPU kernel time:   " << total_npu  / n_iter / 1e3 << " ms\n";
        std::cout << "Min wall clock time:   " << wall_min / 1e3 << " ms\n";
    }

    // ---- write output ----
    auto out_path = vm["output_file"].as<std::string>();
    save_bf16(out_path, output.data(), PREPROC_OUTPUT_ELEMS);
    if (verbosity >= 1)
        std::cout << "Output written to " << out_path << "\n";

    // ---- optional CPU reference validation ----
    if (!do_verify) {
        std::cout << "Skipping verification (--verify 0).\n";
        return 0;
    }

    std::cout << "Computing CPU float32 reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> ref = cpu_reference(image.data(), kernel.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    float ref_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
    std::cout << "CPU reference time: " << ref_ms << " ms\n";

    // Convert NPU output to float for comparison
    std::vector<float> out_f(PREPROC_OUTPUT_ELEMS);
    for (size_t i = 0; i < PREPROC_OUTPUT_ELEMS; ++i)
        out_f[i] = (float)output[i];

    // Compute error statistics
    double max_err  = 0.0;
    double sum_err  = 0.0;
    double max_ref  = 0.0;
    size_t n_fail   = 0;
    constexpr float rtol = 0.1f; // 10% relative tolerance (matches Python test)

    for (size_t i = 0; i < PREPROC_OUTPUT_ELEMS; ++i) {
        double abs_err = std::abs((double)out_f[i] - (double)ref[i]);
        double thresh  = rtol * std::abs((double)ref[i]);
        if (abs_err > thresh + 1e-5f) ++n_fail;
        max_err  = std::max(max_err, abs_err);
        sum_err += abs_err;
        max_ref  = std::max(max_ref, std::abs((double)ref[i]));
    }
    double mean_err = sum_err / PREPROC_OUTPUT_ELEMS;

    std::cout << "Output shape:      [" << PREPROC_SEQ << ", " << PREPROC_EMBD_DIM << "]\n";
    std::cout << "Max reference val: " << max_ref  << "\n";
    std::cout << "Max absolute err:  " << max_err  << "\n";
    std::cout << "Mean absolute err: " << mean_err << "\n";
    std::cout << "Elements outside " << (int)(rtol*100) << "% rtol: "
              << n_fail << " / " << PREPROC_OUTPUT_ELEMS << "\n";

    if (n_fail == 0) {
        std::cout << "PASS: all outputs within tolerance.\n";
        return 0;
    } else {
        std::cerr << "FAIL: " << n_fail << " elements outside tolerance.\n";
        return 1;
    }
}
