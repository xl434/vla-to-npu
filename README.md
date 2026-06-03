# vla_npu_kernel

## Project Structure

```
vla-to-npu/
├── cc/                        # C++ AIE kernel implementations
│   ├── float/                 #   float32 kernels (rope, sin_cos, softmax, …)
│   ├── bf16/                  #   bfloat16 kernels (silu_160, silu_256, …)
│   ├── bf16_new/              #   updated bf16 kernels (gelu, silu, rms_norm, …)
│   └── bf16_old/              #   legacy bf16 kernels (kept for reference)
├── kernel_testing/            # Per-kernel test scripts
│   ├── add/                   #   elementwise add
│   ├── attn_score/            #   attention score (transpose matmul + scale)
│   ├── conv/                  #   conv2d
│   ├── cosine/                #   cosine kernel
│   ├── gelu/                  #   GELU activation
│   ├── gemm/                  #   GEMM variants
│   ├── gemm_silu/             #   fused GEMM + SiLU
│   ├── layer_norm/            #   layer norm
│   ├── masked_softmax/        #   masked softmax
│   ├── rms_norm/              #   RMS norm
│   ├── silu/                  #   SiLU activation
│   ├── sine/                  #   sine kernel
│   ├── softmax_bf16/          #   bf16 softmax
│   ├── softmax_float/         #   float32 softmax
│   └── run_tests.sh           #   runs all VLA component + e2e tests
├── vla/                       # VLA model implementation (NPU, bf16)
│   ├── vla.py                 #   full end-to-end VLA forward pass
│   ├── action_expert_bf16.py  #   action expert transformer (cross + self attn)
│   ├── text_encoder_bf16.py   #   text encoder (Llama-style)
│   ├── llama_block_rope_bf16.py  # Llama block with RoPE (bf16)
│   ├── vision_block_bf16.py   #   ViT vision block
│   ├── connector_bf16.py      #   pixel shuffle + linear connector
│   ├── preprocessing_fused_bf16.py  # conv2d patch embed (im2col + GEMM, NPU)
│   ├── test_rope_fused.py     #   test: fused RoPE kernel
│   ├── test_rope_multiple.py  #   test: multi-head RoPE tiling
│   ├── tests/                 #   additional kernel-level tests (sin_cos, rope_full)
│   ├── vla_float/             #   float32 reference implementations
│   └── old_version/           #   legacy preprocessing (bf16, pipelined)
├── analysis/                  # Profiling and performance analysis scripts
├── log/                       # Design notes and performance logs
└── tools/                     # Utility scripts
```

## Running Kernel Tests

Each subdirectory under `kernel_testing/` contains test scripts that validate the kernel against a PyTorch reference.

**Run a single kernel test:**
```bash
cd kernel_testing/cosine
python test_cosine.py
```

**Run all VLA component and e2e tests:**
```bash
bash kernel_testing/run_tests.sh
# Results written to test_results.log
```

Tests compare NPU kernel outputs against PyTorch reference with bf16-appropriate tolerances (`atol=0.1, rtol=0.1`) and report PASS/FAIL.

---

## Running End-to-End VLA Model (Optimized)

The VLA model has been optimized to run on AMD Phoenix NPU with 3.3× speedup (23s → 6.9s for full model).

**Prerequisites:**
```bash
# Ensure you have the Allo environment set up
conda activate allo-base

# Verify XRT installation (should be system XRT 2.21.75)
xrt-config --version
```

### Quick Start: Full Model Benchmark

```bash
cd /home/xl434/vla-to-npu/vla

# Run full VLA with 12-layer ViT, 12-layer text encoder, 16-layer action expert
python3 vla_standalone.py
```

This runs:
- 1 forward pass of full VLA model (12L ViT + 12L text + 16L action)
- Reports end-to-end timing and intermediate component timings
- Validates correctness against PyTorch reference

**Expected output (~7 seconds):**
```
Running standalone VLA pipeline (no df.build())...

== Timings (standalone, no rebuild) ==
Preprocessing           : 0.18 s
Vision encoder (12L)    : 2.58 s
Connector               : 0.18 s
Joint transformer (2L)  : 3.74 s
Postprocessing          : 0.07 s
Total                   : 6.93 s
```

### Configurable Model Sizes

Edit `vla/vla_standalone.py` to change model depth:

```python
VIT_NUM_LAYERS = 12        # Change for testing smaller models (e.g., 1, 3, 6)
LLAMA_NUM_LAYERS = 12      # Text encoder + action expert depth
```

For quick validation, use smaller configs:
```bash
# Fast validation (1 ViT layer, 2 text/action)
# Edit: VIT_NUM_LAYERS=1, LLAMA_NUM_LAYERS=2
python3 vla_standalone.py   # ~1.7 seconds
```

### Key Optimizations Integrated

1. **GELU Vectorization (15.5× speedup)**
   - Vectorized Padé rational approximation for tanh(x)
   - Kernel: `cc/bf16/gelu_bf16.cc`
   - Builds: `vla/vision_block/gelu_bf16_ffn.prj`

2. **Flash Attention for Vision Encoder**
   - 3-pass attention → single FA kernel
   - 408 dispatches → 192 dispatches per layer
   - Kernel: `cc/flash_attn_64.cc`
   - Builds: `vla/vision_block/flash_attn_vit.prj`

3. **Multi-Layer Binary**
   - All stages (vision, text, action) support 12L in single process
   - Eliminates 11 XRT context switches per stage
   - Run with: `--num-layers N --layers-dir DIR`

### Custom Per-Component Benchmarks

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

### Profiling Details

Run with verbosity level 2 to see per-layer breakdown:

```bash
cd vla && python3 vla_standalone.py  # Sets VIT_NUM_LAYERS=1 by default for profiling

# Expected per-layer profile (1L ViT):
# [profile layer 0]
#   weight_load  :    1.2 ms
#   norm1        :   14.8 ms
#   gemm_qkv     :    6.1 ms
#   fa_dispatch  :   81.8 ms
#   gelu         :   30.5 ms
#   fdn_compute  :    5.8 ms
#   TOTAL        :  223.0 ms
```

### Validation & Correctness

All optimizations have been validated against PyTorch reference with:
- Bitwise matching against 3-pass attention (Flash Attention)
- Max error ≈ 0.01 (well within bf16 tolerance)
- Correctness verified on real hardware

See `log/5-26_kernel_fusion_plan.md` for detailed optimization report.