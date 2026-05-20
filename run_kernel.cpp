//=============================================================================
// Unified NPU kernel runner — replaces per-kernel auto-generated test.cpp
//
// Usage:
//   ./run_kernel -x final.xclbin -i insts.txt -k MLIR_AIE \
//     --buf in:BYTES:input0.data \
//     --buf out:BYTES:output.data \
//     [--buf in:BYTES:input1.data ...]
//
// --buf args are assigned to kernel slots 3, 4, 5, ... in the order given.
// Inputs are loaded from their file before launch.
// Outputs are written to their file after launch.
//=============================================================================

#include <boost/program_options.hpp>
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;

struct BufSpec {
    bool is_output;
    size_t n_bytes;
    std::string filename;
};

// Parse "in:BYTES:filename" or "out:BYTES:filename"
static BufSpec parse_buf(const std::string &s) {
    auto p1 = s.find(':');
    auto p2 = s.find(':', p1 + 1);
    if (p1 == std::string::npos || p2 == std::string::npos) {
        std::cerr << "Error: --buf must be in:BYTES:FILE or out:BYTES:FILE, got: " << s << "\n";
        exit(1);
    }
    BufSpec spec;
    std::string role = s.substr(0, p1);
    if (role == "in")       spec.is_output = false;
    else if (role == "out") spec.is_output = true;
    else {
        std::cerr << "Error: --buf role must be 'in' or 'out', got: " << role << "\n";
        exit(1);
    }
    spec.n_bytes  = std::stoull(s.substr(p1 + 1, p2 - p1 - 1));
    spec.filename = s.substr(p2 + 1);
    return spec;
}

static std::vector<uint32_t> load_instr_binary(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Error: cannot open instr file: " << path << "\n"; exit(1); }
    f.seekg(0, std::ios::end);
    size_t n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> v(n_bytes / sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(v.data()), n_bytes);
    return v;
}

int main(int argc, const char *argv[]) {
    po::options_description opts("Allowed options");
    std::vector<std::string> buf_strs;
    opts.add_options()
        ("help,h",    "produce help message")
        ("xclbin,x",  po::value<std::string>()->required(),               "path to final.xclbin")
        ("instr,i",   po::value<std::string>()->required(),               "path to insts.txt")
        ("kernel,k",  po::value<std::string>()->default_value("MLIR_AIE"),"kernel name in xclbin")
        ("buf",       po::value<std::vector<std::string>>(&buf_strs)->multitoken()->composing(),
                      "buffer spec: in:BYTES:FILE or out:BYTES:FILE (repeatable, in slot order)")
        ("verbosity,v", po::value<int>()->default_value(0),               "verbosity level")
        ("profile,p", po::value<bool>()->default_value(false),            "profiling mode (100 iters)")
        ("warmup",    po::value<int>()->default_value(0),                 "warmup iterations")
        ("iters",     po::value<int>()->default_value(100),               "profiling iterations")
        ("trace_sz",  po::value<int>()->default_value(0),                 "trace buffer size");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, opts), vm);
        if (vm.count("help")) { std::cout << opts << "\n"; return 0; }
        po::notify(vm);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n" << opts << "\n";
        return 1;
    }

    if (buf_strs.empty()) {
        std::cerr << "Error: at least one --buf is required.\n";
        return 1;
    }

    int verbosity   = vm["verbosity"].as<int>();
    bool do_profile = vm["profile"].as<bool>();
    int n_warmup    = vm["warmup"].as<int>();
    int n_iters     = vm["iters"].as<int>();
    int trace_size  = vm["trace_sz"].as<int>();

    std::vector<BufSpec> specs;
    for (auto &s : buf_strs) specs.push_back(parse_buf(s));

    // -------------------------------------------------------------------------
    // Load instruction sequence
    // -------------------------------------------------------------------------
    auto instr_v = load_instr_binary(vm["instr"].as<std::string>());
    if (verbosity >= 1)
        std::cout << "Instruction count: " << instr_v.size() << "\n";

    // -------------------------------------------------------------------------
    // Open device, load xclbin, get kernel handle
    // -------------------------------------------------------------------------
    auto device  = xrt::device(0);
    auto xclbin  = xrt::xclbin(vm["xclbin"].as<std::string>());
    std::string kname = vm["kernel"].as<std::string>();

    auto xkernels = xclbin.get_kernels();
    auto xkernel  = *std::find_if(xkernels.begin(), xkernels.end(),
        [&](xrt::xclbin::kernel &k) { return k.get_name().rfind(kname, 0) == 0; });
    auto kernelName = xkernel.get_name();

    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, kernelName);

    if (verbosity >= 1)
        std::cout << "Kernel: " << kernelName << "\n";

    // -------------------------------------------------------------------------
    // Allocate buffer objects
    //   slot 1: instructions (cacheable)
    //   slot 3, 4, 5, ...: data buffers in spec order
    //   slot 7: trace
    // -------------------------------------------------------------------------
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                             XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));

    int tmp_trace = (trace_size > 0) ? trace_size : 4;
    auto bo_trace = xrt::bo(device, tmp_trace, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));
    if (trace_size > 0) memset(bo_trace.map<char *>(), 0, trace_size);

    std::vector<xrt::bo> bos;
    for (int i = 0; i < (int)specs.size(); i++) {
        auto &spec = specs[i];
        auto bo = xrt::bo(device, spec.n_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3 + i));
        if (!spec.is_output) {
            // Load input from file
            std::ifstream f(spec.filename, std::ios::binary);
            if (!f) { std::cerr << "Error: cannot open " << spec.filename << "\n"; return 1; }
            f.seekg(0, std::ios::end);
            if ((size_t)f.tellg() != spec.n_bytes) {
                std::cerr << "Error: " << spec.filename << " is " << f.tellg()
                          << " bytes, expected " << spec.n_bytes << "\n";
                return 1;
            }
            f.seekg(0);
            f.read(bo.map<char *>(), spec.n_bytes);
            if (verbosity >= 1)
                std::cout << "Loaded " << spec.n_bytes << " bytes from " << spec.filename << "\n";
        }
        bos.push_back(std::move(bo));
    }

    // -------------------------------------------------------------------------
    // Sync to device
    // -------------------------------------------------------------------------
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    for (int i = 0; i < (int)specs.size(); i++)
        if (!specs[i].is_output)
            bos[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (trace_size > 0)
        bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // -------------------------------------------------------------------------
    // Launch kernel using set_arg (supports variable number of data buffers)
    // -------------------------------------------------------------------------
    unsigned int opcode = 3;
    auto launch = [&]() -> xrt::run {
        xrt::run run(kernel);
        run.set_arg(0, opcode);
        run.set_arg(1, bo_instr);
        run.set_arg(2, (int)instr_v.size());
        for (int i = 0; i < (int)bos.size(); i++)
            run.set_arg(3 + i, bos[i]);
        run.set_arg(3 + (int)bos.size(), bo_trace);
        run.start();
        return run;
    };

    if (!do_profile) {
        auto start = std::chrono::high_resolution_clock::now();
        auto run = launch();
        auto r = run.wait();
        auto end = std::chrono::high_resolution_clock::now();
        if (r != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "Kernel did not complete. Status: " << r << "\n";
            return 1;
        }
        float us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "NPU execution time: " << us << "us\n";
    } else {
        for (int i = 0; i < n_warmup; i++) launch().wait();
        float total = 0, tmin = 1e9;
        for (int i = 0; i < n_iters; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            launch().wait();
            float us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            total += us;
            tmin = std::min(tmin, us);
        }
        std::cout << "Avg NPU execution time: " << total / n_iters << "us\n";
        std::cout << "Min NPU execution time: " << tmin << "us\n";
    }

    // -------------------------------------------------------------------------
    // Sync outputs back and write to file
    // -------------------------------------------------------------------------
    for (int i = 0; i < (int)specs.size(); i++) {
        if (!specs[i].is_output) continue;
        bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::ofstream f(specs[i].filename, std::ios::binary);
        if (!f) { std::cerr << "Error: cannot write " << specs[i].filename << "\n"; return 1; }
        f.write(bos[i].map<char *>(), specs[i].n_bytes);
        if (verbosity >= 1)
            std::cout << "Wrote " << specs[i].n_bytes << " bytes to " << specs[i].filename << "\n";
    }

    if (trace_size > 0) {
        bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::ofstream tf("trace.txt");
        char *buf = bo_trace.map<char *>();
        for (int i = 0; i < trace_size; i++)
            tf << std::hex << (int)(unsigned char)buf[i] << " ";
    }

    return 0;
}
