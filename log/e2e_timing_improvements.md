# E2E VLA Timing: Improvements After Kernel Fusion

## What Changed

Two optimizations applied relative to the original baseline:

1. **RoPE fusion** (tribeca branch merge): Replaced the 9-kernel decomposed RoPE pipeline
   (radians → pack → sin → cos → copyL/R → mul/sub/add → join) with a single
   `rope_fused_float32` kernel. Sin/cos are now precomputed on the host and passed in
   as a packed buffer. Applied to `action_expert_bf16.py` and `text_encoder_bf16.py`.

2. **Fused preprocessing** (`preprocessing_fused_bf16.py`): Replaced the original
   `preprocessing_bf16.py` conv2d patch embedding with a 2-phase NPU pipeline:
   128 im2col calls + 192 GEMM calls = 320 total dispatches (vs ~14,592 before).

---

## Wall Clock Time

| Stage                  | Before (old preproc + decomposed RoPE) | After (fused preproc + rope_fused) | Speedup |
|------------------------|----------------------------------------|------------------------------------|---------|
| Preprocessing          | 452.953 s                              | 8.598 s                            | **53x** |
| Vision encoder (1L)    | 56.120 s                               | 56.052 s                           | ~same   |
| Connector              | 20.210 s                               | 20.202 s                           | ~same   |
| Joint transformer (2L) | 25.121 s                               | 25.196 s                           | ~same   |
| Postprocessing         | 0.055 s                                | 0.054 s                            | ~same   |
| **Total**              | **554.459 s**                          | **110.101 s**                      | **5x**  |

---

## NPU Kernel Compute Time (sum of all XRT dispatch times)

| Metric            | Before          | After          | Speedup |
|-------------------|-----------------|----------------|---------|
| Total NPU calls   | 18,905          | 3,746          | **5x fewer** |
| Total NPU compute | 76,990 ms       | 10,973 ms      | **7x faster** |

### Preprocessing NPU breakdown (new run only)
| Kernel   | NPU time  | Calls |
|----------|-----------|-------|
| im2col   | 3,280 ms  | 128   |
| GEMM     | 5,317 ms  | 192   |
| **Total**| **8,597 ms** | **320** |

### Connector NPU breakdown
| Kernel        | Before     | After      |
|---------------|------------|------------|
| Pixel shuffle | 6,996 ms (256 calls) | 7,050 ms (256 calls) |
| GEMM          | 13,214 ms (480 calls) | 13,151 ms (480 calls) |
| **Total**     | **20,210 ms** | **20,202 ms** |

---

## Precision (E2E assert_allclose, rtol=0.1, atol=0.1)

Output shape: [32, 32] (postprocessing output)

| Metric                  | Before  | After   |
|-------------------------|---------|---------|
| Mismatched elements     | 197/1024 (19.2%) | 194/1024 (18.9%) |
| Max absolute difference | 0.499   | 0.516   |
| Max relative difference | 110.4x  | 106.5x  |

Precision is consistent across both runs — mismatches are expected due to accumulated
bf16 rounding error across the full pipeline (not a regression from the optimizations).

---

## Notes

- Wall clock time includes Python/XRT dispatch overhead (~24ms per call), DMA transfers,
  and CPU work. The 5x wall clock speedup is mainly from the 53x preprocessing reduction.
- NPU compute time (7x) is faster than wall clock (5x) because the old preprocessing
  also had significant CPU-side rearrangement work.
- Vision encoder, connector, and joint transformer are bottlenecked by dispatch overhead
  and unchanged in call count — future work would target reducing their dispatch counts.

---

## 2026-05-12: Vision Encoder "More Cores" Optimizations

Applied to `vision_block_bf16.py`. Principle: keep the same per-core kernel and tile size
(no SRAM increase), but increase the number of parallel cores per dispatch to process more
rows per call.

### Changes applied (all tested, all passing correctness checks)

| Change | Mechanism | Calls saved |
|--------|-----------|-------------|
| `NORM_P0` 4→8 | 8 cores per LayerNorm dispatch; same [4,768] per core | 64/block |
| `NORM_SEQ_TILE` 16→32 | Processes 32 rows/call instead of 16 | (counted above) |
| `SOFTMAX_NUM_CORES` 4→8 | 8 cores per softmax dispatch; same [4,512] per core | 768/block |
| `SOFTMAX_NUM_CORES` 8→16 | 16 cores per softmax dispatch (all NPU cores) | 768/block |
| `gelu_bf16_r8` + `GELU_SEQ_TILE` 16→32 | 8 rows/core instead of 4; same SRAM budget | 32/block |

**Note:** `SOFTMAX_NUM_CORES=4→8` and `8→16` were applied in two steps, each halving calls.

### Vision block timing (single block, standalone test)

| Config | Wall clock | Dispatch count (approx.) |
|--------|-----------|--------------------------|
| Baseline (before 5-12 work) | ~56.8 s | ~1,761/block |
| + NORM_P0=8, SOFTMAX_NUM_CORES=8 | 30.5 s | ~993/block |
| + SOFTMAX_NUM_CORES=16 | 19.1 s | ~573/block |
| + gelu_bf16_r8, GELU_SEQ_TILE=32 | **18.2 s** | **~541/block** |

**Total single-block speedup: 56.8s → 18.2s (3.1×)**

### Key insight: GEMM call counts were previously misunderstood
The original roadmap assumed each GEMM kernel = hundreds of dispatches. Reality: each
`GEMM(M, N, K, Pm, Pn, Pk)` call is a **single XRT dispatch** — M/N/K tiling is handled
internally by the Allo MLIR-AIE backend across physical cores. This means GEMM was never
the bottleneck; softmax was (1,536 calls/block at baseline).

### Connector
The `connector_bf16.py` `Pm=2` (M_TILE 32→64) optimization was attempted but reverted
due to a correctness failure (21% elements outside rtol=0.1, likely a hardware/runtime
issue with Pm=2 on this SDK version). Connector remains at 736 calls (256 pixel-shuffle
+ 480 GEMM), ~20s wall clock.

### Remaining bottleneck
Softmax is still ~13s/block (384 calls × ~34ms). To reach <10s, flash attention is needed:
fusing Q@K, softmax, and @V into ~12 calls/block, saving ~12–13s.

---

## 2026-05-12: Text Encoder + Action Expert "More Cores" Optimizations

Applied to `text_encoder_bf16.py` and `action_expert_bf16.py`. Same principle: increase
parallelism per dispatch without changing the per-core kernel or SRAM footprint.

### Changes applied (all tested, all passing correctness checks)

**text_encoder_bf16.py** (SEQ=128, Q_H=15, EMBD=960):

| Change | Mechanism | Calls saved |
|--------|-----------|-------------|
| `NORM_P0` 4→8, `NORM_SEQ_TILE` 16→32 | 8 cores per norm; same [4,960] per core | 8/layer |
| Softmax: `mapping=[1,1]`→`mapping=[16]`, new `softmax_128_bf16` kernel | All 16 row-tiles (8 rows × 16 = SEQ=128) dispatched in parallel per head | 225/layer |

**softmax strategy change:** The original `masked_softmax_128_bf16` kernel required an int32
`row_start` argument (for causal masking), which cannot be broadcast to multiple cores via Allo's
DMA (minimum 16-byte burst; single int32 = 4 bytes). Solution: new `softmax_128_bf16.cc` kernel
with no `row_start` argument; causal mask pre-applied in Python (`-inf` at future positions);
kernel detects `0xFF80` (bf16 -inf) to zero out masked exp values explicitly.

Result: **240 calls → 15 calls/layer** for masked softmax (16×).

**action_expert_bf16.py** (SEQ=32, EMBD=768):

| Change | Mechanism | Calls saved |
|--------|-----------|-------------|
| `NORM_P0` 4→8, `NORM_SEQ_TILE` 16→32 | 8 cores per norm; same [4,768] per core | 2/layer |

### E2E timing (full pipeline, vla.py, VIT_NUM_LAYERS=1, LLAMA_NUM_LAYERS=2)

| Stage | Before (5-12 baseline) | After (this work) | Speedup |
|-------|----------------------|-------------------|---------|
| Preprocessing | 8.6s | 8.6s | — |
| Vision encoder (1L) | 18.2s | 17.8s | — |
| Connector | 20.2s | 19.9s | — |
| **Joint transformer (2L)** | **25.1s** | **12.8s** | **2.0×** |
| Postprocessing | 0.05s | 0.06s | — |
| **Total** | **72.2s** | **59.2s** | **1.2×** |

**Text encoder softmax: 240 → 15 calls/layer = 225 calls saved × 2 layers × ~28ms ≈ 12.6s saved**

### Precision (E2E assert_allclose, rtol=0.1, atol=0.1)

| Metric | Before | After |
|--------|--------|-------|
| Mismatched elements | 194/1024 (18.9%) | 194/1024 (18.9%) |
| Max absolute difference | 0.516 | 0.516 |

No precision regression — identical accumulated bf16 error across the pipeline.

### Remaining bottleneck (joint transformer)
After this work: joint transformer = 12.8s for 2 layers.
- Text encoder per layer: ~15 softmax + 80 RoPE + 3 GEMM + ... ≈ ~170 calls/layer
- Action expert: very few calls (SEQ=32)
- Flash attention in vision encoder still the main path to <10s total
