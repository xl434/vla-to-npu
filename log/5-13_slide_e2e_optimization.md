# VLA-to-NPU: End-to-End Optimization Results

---

## The Core Problem

Each NPU kernel dispatch costs **~28 ms** of overhead (Python → XRT context → DMA),
regardless of actual compute time (~4–12 ms). Dispatch overhead was **70–85% of total runtime**.

**Two levers to reduce latency:**
1. **Fewer dispatches** — tile/fuse kernels so more work happens per call
2. **Eliminate Python overhead** — replace Python→XRT with C++ executables (overhead: ~28ms → ~4ms)

---

## Optimization Timeline

| Step | What Changed | Total E2E Time | Speedup vs Previous |
|------|-------------|---------------|---------------------|
| **Baseline** | Original vla.py (~18,905 NPU calls) | **554 s** | — |
| **RoPE fusion** | 9-kernel decomposed RoPE → 1 fused kernel (sin/cos precomputed on CPU) | — | — |
| **Preprocessing fusion** | Old per-pixel conv (14,592 calls) → im2col + GEMM (320 calls) | **110 s** | **5×** |
| **More cores: vision softmax** | `SOFTMAX_NUM_CORES` 4→16; 1,536 → 384 calls/block | — | — |
| **More cores: vision norm/GELU** | `NORM_P0` 4→8, `GELU_SEQ_TILE` 16→32 | ~18 s (vision only) | 3.1× vision |
| **More cores: text encoder softmax** | `mapping=[1,1]`→`mapping=[16]`; 240→15 softmax calls/layer | **59 s** | **1.9×** |
| **OPT-D: C++ unified executables** | Python→XRT overhead eliminated for all transformer stages | — | — |
| **OPT-D: fused_unified preprocessing** | 320 subprocess spawns → 2 XRT hw_context opens | **5.45 s** | **10.8×** |

---

## Stage-by-Stage Breakdown

| Stage | Baseline | After fusion (Python) | After C++ unified | Key change |
|-------|----------|-----------------------|-------------------|------------|
| Preprocessing | 453 s | 8.6 s | **0.16 s** | 14,592 → 320 → 2 hw_context opens |
| Vision encoder (1L) | 56 s | 17.8 s | **3.5 s** | Softmax 1,536→384 calls; C++ unified |
| Connector | 20 s | 19.9 s | **0.18 s** | C++ unified (pixel shuffle + GEMM) |
| Joint transformer (2L) | 25 s | 12.8 s | **1.59 s** | Text softmax 240→15 calls/layer; C++ |
| Postprocessing | 0.05 s | 0.05 s | **0.05 s** | Unchanged |
| **Total** | **554 s** | **59 s** | **5.45 s** | |
| **Startup** | ~7 min | ~7 min | **~1 s** | No xclbin rebuild |

---

## What "More Cores" Means

Same per-core kernel, more cores per dispatch → fewer total dispatches.

| Kernel | Before | After | Mechanism |
|--------|--------|-------|-----------|
| Vision LayerNorm | 128 calls/block | 64 | `NORM_P0` 4→8 (same [4,768] tile per core) |
| Vision Softmax | 1,536 calls/block | 384 | `NUM_CORES` 4→16 (same [4,512] tile per core) |
| Vision GELU | 64 calls/block | 32 | `GELU_SEQ_TILE` 16→32 rows per dispatch |
| Text Softmax | 240 calls/layer | 15 | All 16 row-tiles per head dispatched in parallel |

Text encoder softmax required a new kernel (`softmax_128_bf16.cc`) — the original kernel
had a `row_start` int32 arg that Allo's DMA couldn't broadcast to >1 core (4-byte minimum burst).
New kernel: causal mask pre-applied in Python (-inf), kernel detects `0xFF80` pattern.

---

## What C++ Unified Executables Do

Python NPU call: **open XRT context → transfer → dispatch → close** (~28 ms/call, Python overhead dominates)

C++ executable: **open XRT once → loop 128+ dispatches → close** (~4–12 ms/call, only kernel compute)

```
Python (vla_standalone.py)
  ├─ preprocessing/fused_unified   → 2 XRT opens  (128 im2col + 192 GEMM dispatches)
  ├─ vision_block/unified.prj      → 1 XRT open   (all vision kernels per layer)
  ├─ connector/unified.prj         → 1 XRT open   (pixel shuffle + GEMM)
  ├─ text_encoder_bf16/unified.prj → 1 XRT open   (all text encoder kernels per layer)
  └─ action_expert_bf16/unified.prj→ 1 XRT open   (self or cross attention per layer)
```

---

## NPU Dispatch Count Over Time

| Checkpoint | Total dispatches | Wall time |
|------------|-----------------|-----------|
| Original baseline | ~18,905 | 554 s |
| After fusion (Python overhead still present) | ~3,746 | 110 s |
| After more-cores (Python overhead still present) | ~1,942 | 59 s |
| After C++ unified (no Python overhead) | ~1,942 | **5.45 s** |

> The final step (Python → C++) reduced time by 10.8× **without changing dispatch count** —
> purely by eliminating ~20 ms Python overhead per call.

---

## Remaining Bottleneck

Vision encoder is 64% of total (3.5s / 5.45s). Breakdown:
- Softmax still 384 calls/block × ~9ms (C++ overhead) ≈ ~3.5s
- **Flash attention** would fuse score + softmax + value → ~12 calls/block → estimated **~0.5s**

| Target | Est. Time After | Effort |
|--------|----------------|--------|
| Current (OPT-D) | 5.45 s | — |
| + Flash attention in vision encoder | ~2.5 s | Hard (new AIE kernel) |
