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

The VLA implementation uses `vla_standalone.py` — an optimized inference pipeline that calls pre-compiled unified binaries directly, with zero build-time overhead.

### Run Full Model (12L ViT + 12L Text + 16L Action)

**Edit `vla/vla_standalone.py` to set:**

```python
VIT_NUM_LAYERS = 12        # Vision encoder depth
LLAMA_NUM_LAYERS = 12      # Text encoder + action expert depth
SKIP = 2                   # Action expert skip factor
```

**Then run:**

```bash
cd vla
python3 vla_standalone.py
```

**Expected output (~7 seconds):**
```
Running standalone VLA pipeline (no df.build())...

== Timings (standalone, no rebuild) ==
Preprocessing           : 0.18 s
Vision encoder (12L)    : 2.58 s
Connector               : 0.18 s
Joint transformer (12L) : 3.74 s
Postprocessing          : 0.07 s
Total                   : 6.93 s
```

### Quick Test with Smaller Model

For fast validation, use smaller layer counts:

```python
VIT_NUM_LAYERS = 3         # Quick test (instead of 12)
LLAMA_NUM_LAYERS = 2       # Quick test (instead of 12)
```

This configuration runs in ~1-2 seconds for rapid iteration.

### Run Individual Components (Binary Mode)

If you have prebuilt unified binaries, you can benchmark individual stages:

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

### Test Individual Components (Python)

To test individual VLA components against PyTorch reference:

```bash
cd vla
python3 vision_block_bf16.py    # Test vision encoder
python3 text_encoder_bf16.py    # Test text encoder
python3 action_expert_bf16.py   # Test action expert
python3 llama_block_rope_bf16.py # Test Llama block with RoPE
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

If you want to rebuild a component from source:

```bash
cd vla/vision_block/unified.prj
mkdir -p build && cd build
cmake .. && make -j4
./vision_encoder --help    # Test the built binary
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
