# Composing Multiple NPU Kernels into a Unified C++ Pipeline

This guide explains how to chain multiple Allo-compiled NPU kernels into a single
C++ host program — as done in `vla/text_encoder_bf16/attn.prj/attn.cpp`.

---

## 1. Background: What Each Allo Build Produces

When you call `allo.build()` or `df.build()` on a kernel, Allo runs the full
compilation pipeline and drops these files into the kernel's `.prj/` directory:

| File | What it is |
|------|-----------|
| `build/final.xclbin` | Packaged NPU firmware (tile ELFs + routing) |
| `insts.txt` | Binary DMA control program (one per kernel invocation) |
| `input*.data` / `output*.data` | Reference I/O for the *single-kernel* test |
| `test.cpp` | Auto-generated single-kernel host program |

A unified pipeline reuses these compiled artifacts across multiple kernels without
relying on the auto-generated `test.cpp`.

---

## 2. Key Hardware Constraint: One `hw_context` at a Time

The AMD NPU (Phoenix/Strix) only supports **one active `xrt::hw_context` at a time**.
Each `hw_context` loads one xclbin (firmware) onto the NPU.

**Wrong approach** — all contexts alive simultaneously:
```cpp
Kernel rms_norm, gemm_q, gemm_kv, ...;   // each holds xrt::hw_context
rms_norm.load(...);   // creates context 1
gemm_q.load(...);     // creates context 2 — CRASHES (EINVAL)
```

**Correct approach** — create and destroy contexts one at a time:
```cpp
{ ActiveKernel ak(dev, spec_rms_norm); ak.run3(...); }  // context created, then destroyed
{ ActiveKernel ak(dev, spec_gemm_q);  ak.run3(...); }  // now safe to create context
```

---

## 3. Folder / Project Setup

### 3.1 Directory layout

```
vla/text_encoder_bf16/
├── rms_norm.prj/
│   ├── build/final.xclbin      ← compiled by Allo
│   ├── insts.txt
│   └── input1.data             ← reference weight / input files
├── gemm_q.prj/
│   ├── build/final.xclbin
│   ├── insts.txt
│   └── input*.data
├── ...
└── attn.prj/                   ← YOUR unified pipeline lives here
    ├── attn.cpp
    ├── CMakeLists.txt
    └── build/                  ← created by cmake
        └── attn               ← the binary
```

### 3.2 Check that every kernel has been compiled

Before writing the unified pipeline, verify each kernel's xclbin exists:

```bash
ls <kernel>.prj/build/final.xclbin
```

If `build/` is empty (Allo built the MLIR but never packaged it), run `aiecc.py`
manually:

```bash
cd <kernel>.prj
aiecc.py --alloc-scheme=basic-sequential \
         --aie-generate-xclbin --no-compile-host \
         --xclbin-name=build/final.xclbin \
         --no-xchesscc --no-xbridge \
         --peano ${PEANO_INSTALL_DIR} \
         --aie-generate-npu-insts --npu-insts-name=insts.txt \
         top.mlir
```

This re-packages the already-compiled ELFs into `build/final.xclbin`.

---

## 4. Writing the Unified C++ File

### 4.1 Two-struct pattern

Split kernel state into two structs:

```cpp
// Stores xclbin + instruction data — registered with device, NO hw_context.
struct KernelSpec {
    xrt::xclbin          xclbin_obj;
    xrt::uuid            uuid;
    std::string          kernel_name;
    std::vector<uint32_t> instr;

    void preload(xrt::device &dev,
                 const std::string &xclbin_path,
                 const std::string &insts_path) {
        xclbin_obj = xrt::xclbin(xclbin_path);
        uuid       = xclbin_obj.get_uuid();
        dev.register_xclbin(xclbin_obj);  // safe to call multiple times

        // Cache kernel name
        auto xks = xclbin_obj.get_kernels();
        auto xk  = *std::find_if(xks.begin(), xks.end(),
            [](xrt::xclbin::kernel &k){
                return k.get_name().rfind("MLIR_AIE", 0) == 0; });
        kernel_name = xk.get_name();

        // Load instruction bytes
        std::ifstream f(insts_path, std::ios::binary);
        f.seekg(0, std::ios::end);
        size_t nb = f.tellg(); f.seekg(0);
        instr.resize(nb / 4);
        f.read(reinterpret_cast<char*>(instr.data()), nb);
    }
};

// RAII wrapper — creates hw_context on construction, destroys on destruction.
// Only one ActiveKernel may exist at a time.
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

    // Launch with 3 data buffer args (slots 3, 4, 5) and wait for completion
    void run3(xrt::bo &b3, xrt::bo &b4, xrt::bo &b5) {
        xrt::run r(k);
        r.set_arg(0, (unsigned int)3);  // opcode
        r.set_arg(1, bo_instr);
        r.set_arg(2, instr_size);
        r.set_arg(3, b3);
        r.set_arg(4, b4);
        r.set_arg(5, b5);
        r.start();
        r.wait();
    }
};
```

### 4.2 Discover data group IDs and allocate BOs

`xrt::bo` needs a `group_id` (memory bank ID). For AMD NPU all data slots (3, 4, 5)
share the same group. Discover it once from a temporary kernel context:

```cpp
// Open device and preload all specs
auto device = xrt::device(0);
KernelSpec spec_a, spec_b, spec_c;
spec_a.preload(device, "a.prj/build/final.xclbin", "a.prj/insts.txt");
spec_b.preload(device, "b.prj/build/final.xclbin", "b.prj/insts.txt");
spec_c.preload(device, "c.prj/build/final.xclbin", "c.prj/insts.txt");

// Discover data group IDs (same for all kernels on AMD NPU)
int g3, g4, g5;
{
    ActiveKernel tmp(device, spec_a);   // temporary context
    g3 = tmp.k.group_id(3);
    g4 = tmp.k.group_id(4);
    g5 = tmp.k.group_id(5);
}  // context destroyed here

// Allocate all BOs now (no active context required for allocation)
auto bo_input  = xrt::bo(device, INPUT_BYTES,  XRT_BO_FLAGS_HOST_ONLY, g3);
auto bo_output = xrt::bo(device, OUTPUT_BYTES, XRT_BO_FLAGS_HOST_ONLY, g4);
auto bo_weight = xrt::bo(device, WEIGHT_BYTES, XRT_BO_FLAGS_HOST_ONLY, g5);
```

> **Rule:** Use slot 3 → g3, slot 4 → g4, slot 5 → g5.  
> On AMD NPU, g3 == g4 == g5, but use the matching slot's group_id to be safe.

### 4.3 Determine slot order for each kernel

Each kernel's slot order (what goes in slots 3, 4, 5) is defined by Allo.
Check the auto-generated `test.cpp` to read off the slot assignments:

```bash
grep "group_id\|bo_in\|bo_out" <kernel>.prj/test.cpp | head -10
```

Typical patterns:
- **Most kernels**: `(slot3=input, slot4=output, slot5=weight)`
- **`masked_softmax`**: `(slot3=score_tile, slot4=row_start, slot5=weight_tile)` — output is at slot 5!

Always verify against the kernel's own `test.cpp`.

### 4.4 Load weights once, run inference loop

```cpp
// Load weights into BOs (sync to device once)
{
    std::ifstream f("weight.data", std::ios::binary);
    f.read(bo_weight.map<char*>(), WEIGHT_BYTES);
    bo_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// Inference loop — create/destroy contexts sequentially
for (int iter = 0; iter < n_iters; iter++) {

    // Load input
    { std::ifstream f("input.data", std::ios::binary);
      f.read(bo_input.map<char*>(), INPUT_BYTES);
      bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

    // Run kernel A
    { ActiveKernel ak(device, spec_a); ak.run3(bo_input, bo_mid, bo_weight_a); }

    // Run kernel B (reads mid, writes output)
    bo_mid.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // ... CPU post-processing on bo_mid if needed ...
    bo_mid.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    { ActiveKernel ak(device, spec_b); ak.run3(bo_mid, bo_output, bo_weight_b); }

    // Sync output back
    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}
```

### 4.5 CPU ↔ NPU data handoff

Between NPU kernel calls, CPU operations on BO data need explicit syncs:

```cpp
// After NPU writes (reading result on CPU):
bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
auto *ptr = bo_out.map<uint16_t*>();
// ... use ptr ...

// Before NPU reads (writing from CPU):
auto *ptr = bo_in.map<uint16_t*>();
// ... fill ptr ...
bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
```

---

## 5. CMakeLists.txt

Create `<pipeline>.prj/CMakeLists.txt` — no dependency on Allo's `test_utils.cpp`:

```cmake
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)

project(my_pipeline)

set(BOOST_ROOT /usr/include/boost CACHE STRING "Path to Boost install")
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include dir")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT lib dir")

find_package(Boost REQUIRED)

add_executable(my_pipeline my_pipeline.cpp)

target_compile_definitions(my_pipeline PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories(my_pipeline PUBLIC
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_directories(my_pipeline PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

target_link_libraries(my_pipeline PUBLIC
    xrt_coreutil
    boost_program_options
    boost_filesystem
    uuid               # required by xrt::uuid
)
```

> **Note:** `uuid` is required when using `xrt::uuid` (which `KernelSpec` stores).
> Without it, you'll get `undefined reference to symbol 'uuid_clear@@UUID_1.0'`.

---

## 6. Building

```bash
cd <pipeline>.prj
mkdir -p build && cd build
cmake ..
make -j4
```

The binary appears at `build/<pipeline>`.

---

## 7. Running

Run from **inside** the `.prj/` directory so that relative paths like
`../rms_norm.prj/build/final.xclbin` resolve correctly:

```bash
cd vla/text_encoder_bf16/attn.prj

build/attn \
  --input  ../gemm_q.prj/input0.data \
  --W_norm ../rms_norm.prj/input1.data \
  --Wq     ../gemm_q.prj/input1.data \
  --Wk     ../gemm_kv.prj/input1.data \
  --Wv     ../gemm_kv.prj/input1.data \
  --Wo     ../gemm_out.prj/input1.data \
  --output output.data -v 1
```

Expected output:
```
Preloading kernel specs...
All specs loaded.
Data group IDs: slot3=65536 slot4=65536 slot5=65536
Loading weights...
Warmup run...
Attention forward pass: 520.404 ms/iter (avg over 1 iters)
Output written to output.data
```

---

## 8. Testing / Validation

### 8.1 Functional check against per-kernel reference outputs

Each kernel's `.prj/` has `output*.data` — the reference output from the
single-kernel test run. Use these to verify each stage of your pipeline.

Example: verify rms_norm output matches reference:

```python
import numpy as np

ref  = np.fromfile("rms_norm.prj/output2.data",  dtype="<u2").view(np.float16)
ours = np.fromfile("attn.prj/rms_norm_out.data", dtype="<u2").view(np.float16)

print("Max abs diff:", np.max(np.abs(ref.astype(float) - ours.astype(float))))
print("Close?", np.allclose(ref.astype(float), ours.astype(float), atol=1e-2))
```

For bf16, an absolute tolerance of `1e-2` is typical.

### 8.2 Dump intermediate buffers for debugging

Add optional file dumps after each kernel stage:

```cpp
if (verbosity >= 2) {
    bo_rms_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::ofstream f("debug_rms_out.data", std::ios::binary);
    f.write(bo_rms_out.map<char*>(), SZ_NORM_TILE);
}
```

Then compare with the reference `output2.data` from the corresponding `.prj/`.

### 8.3 Performance profiling

```bash
build/attn ... --iters 100 -v 0
# Output: Attention forward pass: X ms/iter (avg over 100 iters)
```

For per-kernel timing, wrap each `ActiveKernel` block with `chrono`:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
{ ActiveKernel ak(device, spec_rms_norm); ak.run3(...); }
auto t1 = std::chrono::high_resolution_clock::now();
float us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
std::cout << "rms_norm: " << us << " us\n";
```

---

## 9. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=22)` | Multiple `hw_context` objects alive simultaneously | Use `ActiveKernel` RAII pattern; only one context at a time |
| `No such file 'xxx.prj/build/final.xclbin'` | Allo compiled MLIR but never ran `aiecc.py` packaging | Run `aiecc.py` manually (see §3.2) |
| `undefined reference to symbol 'uuid_clear@@UUID_1.0'` | Missing `libuuid` link | Add `uuid` to `target_link_libraries` in CMakeLists.txt |
| `expected N bytes, got M` | Input file is one tile (per-kernel reference), not full tensor | Use the correct file; check sizes in the data file listing |
| `Cannot find source file: test_utils.cpp` | Using Allo-generated CMakeLists.txt | Replace with the self-contained CMakeLists.txt from §5 |
| Output is NaN or all zeros | Slot order wrong for a kernel | Check the kernel's `test.cpp` for the correct slot assignment |
