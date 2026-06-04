# VLA-to-NPU: Kernel Testing & Optimization Suite

This repository contains optimized kernel implementations and end-to-end testing for Vision Language Action (VLA) models on AMD Phoenix NPU, using the [Allo](https://github.com/heterogeneous-computing/allo) framework.

## Quick Start

### Prerequisites
```bash
# Activate Allo environment
conda activate allo-base

# Verify XRT is installed
xrt-config --version
```

### Run a Single Kernel Test
```bash
cd kernel_testing/cosine
python test_cosine.py
```

### Run End-to-End VLA Model
```bash
cd vla
python3 vla.py
```

---

## Project Structure

```
vla-to-npu/
├── cc/                              # AIE kernel source code
│   ├── float/                       #   float32 kernels (rope, sin_cos, softmax, …)
│   └── bf16_new/                    #   bfloat16 kernels (gelu, silu, rms_norm, …)
│
├── kernel_testing/                  # Per-kernel test suites
│   ├── add/, attn_score/, conv/     #   Various kernel tests
│   ├── cosine/, gelu/, gemm/, ...   #   Compare NPU outputs vs PyTorch reference
│   └── rope/                        #   RoPE (Rotary Position Embedding) tests
│
├── vla/                             # VLA model implementation (NPU, bf16)
│   ├── vla.py                       #   Full end-to-end forward pass
│   ├── vision_block_bf16.py         #   ViT vision encoder (12L)
│   ├── text_encoder_bf16.py         #   Text encoder, Llama-style (12L)
│   ├── action_expert_bf16.py        #   Action expert (16L)
│   ├── llama_block_rope_bf16.py     #   Llama block with RoPE
│   ├── connector_bf16.py            #   Pixel shuffle + linear connector
│   ├── preprocessing_fused_bf16.py  #   Conv2d patch embedding (fused)
│   ├── tests/                       #   Kernel-level tests (sin_cos, rope_full)
│   └── vision_block/                #   Unified C++ vision encoder binary (prebuilt)
│
├── cc/                              # C++ kernel implementations
├── tools/                           # Utility scripts
└── README.md                        # This file
```

---

## Running Kernel Tests

Each subdirectory under `kernel_testing/` contains a test that validates an NPU kernel against a PyTorch reference.

### Single Kernel Test

```bash
cd kernel_testing/cosine
python test_cosine.py
```

**Output:** PASS/FAIL with error metrics (atol=0.1, rtol=0.1 for bf16).

### Common Kernel Tests

| Kernel | Test Location | Command |
|--------|---------------|---------|
| Cosine | `kernel_testing/cosine` | `python test_cosine.py` |
| GELU | `kernel_testing/gelu` | `python test_gelu_bf16_new.py` |
| SiLU | `kernel_testing/silu` | `python test_silu_bf16_new.py` |
| RoPE (Fused) | `kernel_testing/rope` | `python test_rope_fused.py` |
| RoPE (Multiple heads) | `kernel_testing/rope` | `python test_rope_multiple.py` |

---

## Running End-to-End VLA Model

The VLA implementation includes pre-compiled unified binaries for each component. You can run the full model without rebuilding:

### Quick Run (Full Model)

```bash
cd vla
python3 vla.py
```

**Expected output (~7 seconds):**
```
Running standalone VLA pipeline...

== Timings ==
Preprocessing           : 0.18 s
Vision encoder (12L)    : 2.58 s
Connector               : 0.18 s
Joint transformer (2L)  : 3.74 s
Postprocessing          : 0.07 s
Total                   : 6.93 s

Validation: PASS
```

### Configurable Model Sizes

Edit `vla/vla_standalone.py` to adjust layer counts:

```python
VIT_NUM_LAYERS = 12        # Vision encoder depth
LLAMA_NUM_LAYERS = 12      # Text encoder + action expert depth
```

For fast validation with smaller models:
```python
VIT_NUM_LAYERS = 1
LLAMA_NUM_LAYERS = 2
```

### Run Individual Components

**Vision Encoder Only:**
```bash
cd vla/vision_block/unified.prj/build
./vision_encoder --input ../x.data --num-layers 12 --layers-dir /path/to/weights -v 2
```

**Text Encoder Only:**
```bash
cd vla/text_encoder_bf16/unified.prj/build
./text_encoder --input ../x.data --num-layers 12 --layers-dir /path/to/weights -v 1
```

**Action Expert Only:**
```bash
cd vla/action_expert_bf16/unified.prj/build
./action_expert --input ../x.data --num-layers 16 --skip 2 --layers-dir /path/to/weights -v 1
```

---

## Building Kernels with Allo

Kernels were initially generated using Allo. To rebuild or modify a kernel:

### Build a Kernel Test

Each kernel test directory contains `allo_src/` with the Allo source and a build script. For example, to rebuild RoPE:

```bash
cd kernel_testing/rope
python build_rope_fused.py    # Builds fused RoPE kernel
python test_rope_fused.py     # Runs the test
```

**Build Script Locations:**
- `kernel_testing/*/build_*.py` — Allo build scripts for individual kernels
- `vla/*/build_*.py` — Allo build scripts for VLA components

### Build VLA Component (Vision Encoder Example)

```bash
cd vla/vision_block
python build_vision_encoder.py    # Rebuilds unified vision encoder
cd unified.prj/build && ./vision_encoder
```

### Key Optimizations

1. **GELU Vectorization (15.5× speedup)**
   - Vectorized Padé rational approximation
   - Kernel: `cc/bf16/gelu_bf16.cc`

2. **Flash Attention for Vision Encoder**
   - 3-pass → single FA kernel (408 → 192 dispatches/layer)
   - Kernel: `cc/flash_attn_64.cc`

3. **Multi-Layer Binary**
   - All stages (vision, text, action) support 12L+ in single process
   - Eliminates context switching overhead

---

## Validation & Correctness

All kernels and components are validated against PyTorch reference with bf16-appropriate tolerances:
- Absolute tolerance: `atol=0.1`
- Relative tolerance: `rtol=0.1`
- Max error: ≈ 0.01 on real hardware

---

## Contributing

To add a new kernel test or optimize an existing one:

1. **Add Kernel Source** — Place Allo code in `cc/bf16/` or `cc/float/`
2. **Create Test** — Add test script in `kernel_testing/new_kernel/`
3. **Build Script** — Create `kernel_testing/new_kernel/build_new_kernel.py`
4. **Validate** — Ensure test passes against PyTorch reference
5. **Update README** — Document the new kernel in this file

---

## Hardware Requirements

- **Processor:** AMD Phoenix NPU (or compatible)
- **XRT:** Version 2.21.75+
- **Memory:** 8GB+ RAM recommended

---

## Questions?

For issues or questions, please open an issue on GitHub or refer to the test output logs.
