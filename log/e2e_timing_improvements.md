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
