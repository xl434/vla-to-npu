# NPU Kernel Programming Guide

This guide explains how to write kernels for AMD Phoenix NPU using the Allo compiler framework, and optionally how to optimize them with unified binaries for faster end-to-end performance.

## Overview: Two Development Paths

### Path 1: Allo Programming (Recommended for Development)

Write and test Allo code directly:
- Edit Python Allo code: `vla/my_component_bf16.py`
- Run Allo to compile individual kernels
- Test each kernel against PyTorch reference
- Use `vla.py` to run end-to-end with individual kernels
- Result: ~23 seconds end-to-end
- **Benefits:** Easy to implement, test, and debug

### Path 2: Allo + Claude-Generated Unified.prj (Optimized Implementation)

After Allo code works, optimize with unified binaries:
- Keep Allo code unchanged
- Claude generates `unified.prj` (C++ combining all kernels)
- Use `vla_standalone.py` to run end-to-end with unified binary
- Result: ~6.16 seconds end-to-end (3.7× faster)
- **Benefits:** Fast end-to-end, still maintains correctness

**Workflow:**
```
Step 1: Allo Programming (verify correctness)
        Write Allo code → Compile → Test with vla.py

Step 2: Unified.prj Optimization (improve speed)
        Ask Claude for unified.prj → Compile → Test with vla_standalone.py
```

**Key insight:** Both paths run on NPU. Start with Allo programming for simplicity, then add unified.prj for speed.

---

## Level 1: Single-Tile Kernel (C++ with AIE API)

### Example: SiLU Activation

**Kernel source:** `cc/bf16_vla/silu_128_bf16.cc`

This implements SiLU (x * sigmoid(x)) for a 4×128 tile of bfloat16 data using the [AMD AIE API](https://download.amd.com/docnav/aiengine/xilinx2022_2/aiengine_api/aie_api/doc/index.html) with vectorized Padé polynomial for sigmoid.

**See actual code:** `cc/bf16_vla/silu_128_bf16.cc`

**Test:** `kernel_testing/silu/test_silu_bf16_new.py` (validates against PyTorch)

---

## Level 2: Allo Programming (Dataflow Model on NPU)

Allo uses a **dataflow programming model** to express how kernels communicate via streams. See the [Allo dataflow documentation](https://cornell-zhang.github.io/allo/dive/dataflow.html) for details on regions, kernels, and streams.

### Example: RMS Norm with Allo

**Allo code:** `vla/text_encoder_bf16.py` and `vla/llama_block_rope_bf16.py`

Uses kernel source: `cc/bf16_vla/rms_norm_960_bf16.cc`

RMS Norm with Allo:
1. Define input/output streams
2. Map RMS computation to multiple AIE cores
3. Allo handles point-to-point communication for reductions
4. Allo generates .xclbin and insts.txt

**Generated outputs:**
- `vla/text_encoder_bf16/rms_norm.prj/build/final.xclbin` — Compiled kernel (runs on NPU)
- `rms_norm.prj/insts.txt` — Instructions for NPU

**Test:** `kernel_testing/layer_norm/test_layer_bf16_new.py` (validates against PyTorch)

**Important:** This kernel runs entirely on NPU. No Claude optimization needed for correctness.

---

## Level 3: Unified.prj (Optional: Claude-Generated Optimization)

### The Problem: Per-Kernel Overhead

When running multiple Allo kernels with `vla_standalone.py`:

```python
# vla.py (calls Allo kernels individually)
x = vision_output
for layer in range(12):
    x = rms_norm(x)        # ← XRT launch #1
    x = attention(x)       # ← XRT launch #2
    x = mlp(x)             # ← XRT launch #3
# ... 12 layers × 3 = 36 XRT launches total
```

**Overhead per kernel:**
- XRT context switch
- Allocate NPU memory buffers
- DMA transfer input data
- Launch AIE executable
- DMA transfer output data
- Deallocate buffers

**Result:** 36 kernels × overhead = ~17 seconds total

### Solution: Unified.prj (Single Executable)

Claude generates a single C++ executable that:
1. Calls all kernels in sequence
2. Keeps intermediate data in NPU memory (no DMA between kernels)
3. Single XRT launch for entire component

**Example locations:**
- `vla/text_encoder_bf16/unified.prj/test.cpp` — Text encoder (12 layers)
- `vla/vision_block/unified.prj/test.cpp` — Vision encoder (12 layers)
- `vla/action_expert_bf16/unified.prj/test.cpp` — Action expert (16 layers)

**Benefits:**
- ✅ Single XRT launch per component (vs 36 for 12 layers × 3 ops/layer)
- ✅ Zero DMA between kernels (data stays in NPU memory)
- ✅ Predictable, composable performance
- ✅ Faster end-to-end (6.16s vs 23s)

---

## How to Generate Unified.prj

### Prerequisites

Before asking Claude to generate unified.prj, you must:

1. **Write kernel sources** in `cc/bf16_vla/`
2. **Write Allo code** in `vla/` that references these kernels
3. **Run Allo to compile** — Generate `.prj/build/final.xclbin` files
4. **Test Allo kernels** — Verify they run correctly on NPU

Only then is unified.prj ready to be generated.

### Step-by-Step Workflow

**Step 1: Write kernel source**

```
cc/bf16_vla/my_kernel_bf16.cc
```

**Step 2: Write Allo code**

```
vla/my_component_bf16.py  (uses my_kernel_bf16.cc)
```

**Step 3: Compile with Allo**

```bash
python vla/my_component_bf16.py
# Generates: vla/my_component/my_kernel.prj/build/final.xclbin
```

**Step 4: Test the Allo kernel**

```bash
python kernel_testing/my_kernel/test_my_kernel.py
# Should PASS — kernel runs correctly on NPU
```

**Step 5: Ask Claude to generate unified.prj**

Only after Allo kernels are tested and working, use this prompt:

---

### Claude Prompt Template

```
Generate a unified C++ executable for [COMPONENT_NAME].

REQUIRED CHANGES (replace with your values):
- [COMPONENT_PATH] = vla/text_encoder_bf16 (example)
- [KERNEL_LIST] = list of .prj directories to include
- [LAYER_COUNT] = number of transformer layers (e.g., 12)
- [COMPUTATION_PATTERN] = pattern of kernel calls per layer

KERNEL DETAILS:
List the .prj directories that exist and are compiled:
- [COMPONENT_PATH]/rms_norm.prj/build/final.xclbin
- [COMPONENT_PATH]/attention.prj/build/final.xclbin
- [COMPONENT_PATH]/mlp.prj/build/final.xclbin
(examples for text encoder; your kernels may differ)

EXPECTED COMPUTATION FLOW:
For each layer:
1. Call [kernel_1] (e.g., rms_norm)
2. Call [kernel_2] (e.g., attention)
3. Call [kernel_3] (e.g., mlp)
Repeat for [LAYER_COUNT] layers.

INPUT/OUTPUT SPECS:
- Input: [input_shape] (e.g., [1, 48, 960] for text encoder)
- Output: [output_shape] (e.g., [1, 48, 960])
- Intermediate data stays in NPU memory between kernels

TASK:
Generate [COMPONENT_PATH]/unified.prj/test.cpp that:
1. Includes headers for all kernels from [KERNEL_LIST]
2. Implements main() that calls kernels in [COMPUTATION_PATTERN]
3. Manages data flow: input → layer 1 → layer 2 → ... → [LAYER_COUNT] → output
4. Uses in-place computation where possible
5. Compiles without errors
6. When compiled and run, produces same outputs as individual Allo kernels

Verify correctness by comparing unified.prj output vs individual Allo kernel outputs.
```

### Example Usage

If generating unified.prj for text encoder:

```
Generate a unified C++ executable for text_encoder_bf16.

REQUIRED CHANGES:
- [COMPONENT_PATH] = vla/text_encoder_bf16
- [KERNEL_LIST] = rms_norm, attention, mlp (3 kernels per layer)
- [LAYER_COUNT] = 12
- [COMPUTATION_PATTERN] = RMSNorm → Attention → MLP, repeat 12 times

KERNEL DETAILS:
These .prj directories exist and are compiled:
- vla/text_encoder_bf16/rms_norm.prj/build/final.xclbin
- vla/text_encoder_bf16/attn.prj/build/final.xclbin
- vla/text_encoder_bf16/mlp.prj/build/final.xclbin

EXPECTED COMPUTATION FLOW:
For each of 12 layers:
1. Call rms_norm
2. Call attention
3. Call mlp

INPUT/OUTPUT SPECS:
- Input: [1, 48, 960] (sequence_len=48, embed_dim=960)
- Output: [1, 48, 960]

TASK:
Generate vla/text_encoder_bf16/unified.prj/test.cpp...
```

**Step 6: Compile unified.prj**

```bash
cd [COMPONENT_PATH]/unified.prj
mkdir -p build && cd build
cmake .. && make -j4
```

**Step 7: Test unified.prj**

```bash
# Run vla_standalone.py
cd vla
python3 vla_standalone.py
# Should complete in ~6.16 seconds (vs ~23 seconds with individual Allo kernels)
```

---

## Allo Limitations & Why We Use Claude

### Limitation 1: Per-Kernel Overhead

**Allo generates:** Individual `.xclbin` for each kernel

**Result:** Each kernel call = XRT launch = overhead (36 launches for 12 layers × 3 ops/layer)

### Limitation 2: No Host Code Generation

Allo generates AIE kernel code but NOT:
- C++ main() function
- Buffer management across multiple kernels
- Kernel composition logic
- Data flow orchestration for multiple kernels

**Why:** Allo focuses on individual kernel compilation, not full-system integration.

**Why Claude helps:** Claude writes the C++ "glue" code (unified.prj) that ties kernels together efficiently.

### Limitation 3: Multi-Kernel Data Management

Allo optimizes individual kernels but doesn't automatically:
- Keep intermediate data in NPU memory between kernels
- Minimize DMA transfers across kernel boundaries
- Coordinate memory allocation for pipelined execution

**Claude solution:** unified.prj manages these automatically by calling kernels in sequence without intermediate DMA.

---

## Summary: The Complete Workflow

### For Best Performance (Recommended)

```
1. Write kernel source:
   cc/bf16_vla/my_kernel_bf16.cc

2. Write Allo code:
   vla/my_component_bf16.py

3. Run Allo to compile:
   Allo compiler generates:
   → my_component/my_kernel.prj/build/final.xclbin
   
4. Test Allo kernel:
   python kernel_testing/my_kernel/test_my_kernel.py
   (Should PASS)

5. Ask Claude for unified.prj:
   Claude generates:
   → vla/my_component/unified.prj/test.cpp

6. Compile unified.prj:
   CMake + make
   → vla/my_component/unified.prj/build/my_component_executable

7. Run vla_standalone.py:
   vla_standalone.py calls:
   → vla/my_component/unified.prj/build/my_component_executable
   
   Result: ~6.16 seconds end-to-end ✅
```

### Allo Programming (Without Unified.prj)

```
1-4. Same as above (write kernel, Allo code, test)

5. Run vla.py to test end-to-end:
   Example: vla/vla.py calls individual Allo kernels directly
   Result: ~23 seconds end-to-end
   
   Use for:
   - Verifying Allo code correctness
   - Debugging individual kernel behavior
   - Before generating unified.prj
```

---

## Key Takeaways

| Concept | Details |
|---------|---------|
| **Allo Programming** | Write & test Allo code directly (easier, slower ~23s) |
| **Unified.prj** | Claude-generated C++ combining kernels (faster ~6.16s) |
| **Start with** | Allo programming (verify correctness) |
| **Then optimize with** | Unified.prj (improve speed) |
| **Prerequisites** | Allo kernels must be compiled and tested first |
| **Example implementations** | vla.py (Allo), vla_standalone.py (unified.prj) |

---

## Resources

- **Allo documentation**: https://github.com/heterogeneous-computing-lab/allo
- **Allo dataflow model**: See `/tribeca/.claude/skills/npu-kernel-gen/references/allo_docs/dataflow.rst`
- **AIE API reference**: https://github.com/Xilinx/AI-Engine-Intrinsics
- **Kernel examples**: `cc/bf16_vla/`
- **Allo Python examples**: `vla/`
- **Unified.prj examples**: `vla/*/unified.prj/`
- **Test examples**: `kernel_testing/*/test_*.py`
