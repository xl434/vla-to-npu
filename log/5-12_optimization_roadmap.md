# NPU Optimization Roadmap

## Background

Each NPU kernel dispatch costs ~28ms of overhead (XRT subprocess spawn + DMA),
regardless of compute time. Actual kernel compute is ~4–12ms per call. This means
**~70-85% of wall-clock time is dispatch overhead**, not compute.

The only way to speed things up is to **reduce the number of dispatch calls**.

### Baseline (after preprocessing fusion + rope fusion, before 5-12 work)

| Stage | Wall clock | ~NPU calls |
|-------|-----------|------------|
| Preprocessing | 8.6s | 320 |
| Vision encoder (1 block) | 56s | ~1,761 |
| Connector | 20s | 736 |
| Joint transformer (2L) | 25s | ~300 |
| **Total** | **~110s** | **~3,117** |

### Vision encoder actual call breakdown (per block, SEQ=1024)

> **CORRECTION from original roadmap:** Each `GEMM(M,N,K,Pm,Pn,Pk)` is a SINGLE
> XRT dispatch — internal M/N/K tiling is handled by the Allo backend, not Python loops.
> The original roadmap incorrectly assumed each GEMM = hundreds of calls.

| Operation | Calls | Notes |
|-----------|-------|-------|
| LayerNorm × 2 | 128 | SEQ / 16-row tile × 2 norms |
| Q, K, V projections | 3 | 1 dispatch each (full [1024,768]×[768,768]) |
| Attention score (12 heads) | 12 | 1 dispatch per head |
| **Softmax (12 heads)** | **1,536** | **12 heads × (SEQ/SEQ_per_batch=128 batches) — ACTUAL bottleneck** |
| Attention value (12 heads) | 12 | 1 dispatch per head |
| Output projection | 1 | 1 dispatch |
| FFN up | 1 | 1 dispatch |
| GELU | 64 | SEQ / 16-row tile |
| FFN down (4 K-chunks) | 4 | 4 Python K-tiles × 1 dispatch each |
| **Total** | **~1,761** | Softmax is 87% of calls |

### After 5-12 "more cores" optimizations (measured)

| Operation | Calls (before) | Calls (after) | Mechanism |
|-----------|---------------|---------------|-----------|
| LayerNorm × 2 | 128 | **64** | NORM_P0 4→8 (same [4,768] per core) |
| Softmax | 1,536 | **384** | SOFTMAX_NUM_CORES 4→16 (same [4,512] per core) |
| GELU | 64 | **32** | gelu_bf16_r8, GELU_SEQ_TILE 16→32 (same 8-row per core) |
| Everything else | 33 | 33 | unchanged |
| **Total** | **~1,761** | **~513** | **3.4× fewer calls** |

**Measured timing: 56.8s → 18.2s (3.1× speedup), all correctness checks passing.**

---

---

## Optimization Opportunities

### ✅ DONE: "More cores" parallelism — vision encoder (norm, softmax, GELU)

Applied 2026-05-12. Principle: keep the same per-core kernel unchanged (no SRAM increase),
double parallelism per dispatch. Saves 1,248 calls/block × ~28ms = ~35s/block.

See `e2e_timing_improvements.md` for full details.

---

### ✅ DONE: "More cores" parallelism — text encoder + action expert (softmax, norm)

Applied 2026-05-12. Same principle applied to text encoder and action expert.

**Text encoder masked softmax (240 → 15 calls/layer):**
- Original: `mapping=[1,1]`, 240 calls/layer (15 heads × 16 row-tiles per head)
- New: `mapping=[16]`, 15 calls/layer (all 16 row-tiles in parallel; 1 call per head)
- New kernel `softmax_128_bf16.cc`: no `row_start` int32 argument (avoids Allo DMA 4-byte minimum
  burst size issue); causal mask pre-applied in Python; -inf detected via `0xFF80` bit pattern
- Files: `cc/bf16/softmax_128_bf16.cc`, `vla/text_encoder_bf16.py`
- **Saves: 225 calls × 2 layers × ~28ms ≈ 12.6s per joint transformer forward pass**

**Text/action encoder norm (saves ~8 + 2 calls/layer):**
- `NORM_P0`: 4→8, `NORM_SEQ_TILE`: 16→32 in both files (same [4,dim] per core)

**Measured E2E speedup: joint transformer 25.1s → 12.8s (2.0×), total 110s → 59s**

See `e2e_timing_improvements.md` for full details.

---

### OPT-1: Increase Vision Encoder GEMM M-tile (64→128 rows)

**Target:** Vision encoder Q/K/V/Out/FFN projections  
**Mechanism:** Current `LINEAR_TILE=64`. Doubling to 128 halves M-iterations:
- Q/K/V/Out: 256 → 128 calls each → saves 512 calls
- FFN up: 768 → 384 calls → saves 384 calls
- FFN down: 1024 → 512 calls → saves 512 calls
- **Total saved: ~1,400 calls × 24ms ≈ 34s**

**Constraint:** Each AIE core has 64KB SRAM. With M=128, K=64, N=32:
- A tile: 128×192 bf16 = 49,152 B (double-buffered = 98,304 B) — **too large!**
- Need to reduce K-tile or use Pk pipeline instead. Feasible with Pk=4, K_per_core=192:
  - A: 128×192 × 2 = 49,152 B, B: 192×32 × 2 = 24,576 B, C: 128×32 × 4 = 32,768 B → 106 KB — still too large
- Safer: M=128, Pn=4, Pk=2 → A: 128×384×2=196KB — still too large
- **Realistic option:** M=64, Pn=8 (more N parallelism). Needs `Pn=8` which uses all 16 cores for N instead of K. Reduces N-iterations 2×.

**Difficulty:** Medium. Need to tune Pm/Pn/Pk to fit SRAM with larger tiles.  
**Estimated speedup:** 15–25s reduction in vision encoder time.  
**Status:** Not started.

---

### OPT-2: Fuse FFN K-chunks (Vision Encoder FFN Down)

**Target:** Vision encoder FFN down (3072→768, currently 4 K-chunks)  
**Mechanism:** `FFN_DOWN_K_CHUNKS = 4` splits K=3072 into 4 sequential GEMM blocks,
each doing K=768. Since these blocks need to accumulate (partial sums), they currently
require 4× the dispatch calls. If we can increase K_TILE to cover all of K=3072 in one
pass using Pk=4, the K-chunking in Python disappears.

Current: `GEMM(SEQ, EMBD, FFN_HID)` with `FFN_HID//EMBD=4` K-chunks called from Python loop.  
Target: Single GEMM call with Pk=4 handling all K internally via stream accumulation.

**Constraint:** This is already how the connector GEMM works (`K_TILE=768, Pk=4`). The
vision encoder can use the same pattern if K_TILE=768 fits memory (it does — same as connector).

**Difficulty:** Easy-Medium. Change `FFN_DOWN_K_CHUNKS` Python loop to single GEMM with
larger K_TILE. May need a new GEMM kernel config.  
**Estimated speedup:** ~512 fewer calls × 24ms ≈ 12s.  
**Status:** Not started.

---

### OPT-3: Fuse Norm + GEMM (RMSNorm/LayerNorm before each projection)

**Target:** Vision encoder and text encoder norms (128 calls/layer in vision encoder)  
**Mechanism:** Norm and the following GEMM always operate on the same tensor sequentially.
Fusing them into one kernel eliminates the norm dispatch calls entirely. The norm kernel
reads a 16-row tile, normalizes, and feeds it directly into the GEMM core.

**Constraint:** Requires a custom fused norm+GEMM AIE kernel. The norm needs to reduce
across the full EMBD=768 dimension before GEMM can start — doable within one AIE call
since norm output is used immediately. Memory footprint increases slightly.

**Difficulty:** Hard. Requires new C++ kernel. Not a simple parameter change.  
**Estimated speedup:** 128 calls × 24ms ≈ 3s per layer for vision encoder.  
**Status:** Not started.

---

### OPT-4: Flash Attention (Fused Score + Softmax + Value)

**Target:** Vision encoder attention (currently 192+4+192 = 388 calls/layer)  
**Mechanism:** Standard attention dispatches score, softmax, and value separately. Flash
attention fuses all three: for each Q-tile, it streams over K/V tiles and accumulates
the output with an online softmax correction — no materialized attention matrix.

For vision encoder (SEQ=1024, 12 heads, HEAD_DIM=64):
- Current: 192 (score) + 4 (softmax) + 192 (value) = 388 calls
- Flash: 12 heads × (SEQ/tile) ≈ 12 × 16 = 192 calls (one per Q-tile per head)
- **Saves ~196 calls × 24ms ≈ 5s**

**SRAM budget per AIE core (64KB):**
```
Q-tile  [16, 64] bf16  = 2,048 B
K-tile  [16, 64] bf16  = 2,048 B
V-tile  [16, 64] bf16  = 2,048 B
Output  [16, 64] f32   = 4,096 B
Softmax accum  [16] f32 = 64 B
Total: ~10 KB << 64 KB  ✓
```

SRAM fits easily. The challenge is the online softmax algorithm (Dao et al. 2022)
in AIE intrinsics — requires tracking running max and sum per row across K-tiles.

**Difficulty:** Hard. New AIE kernel implementing the flash attention algorithm.
Probably 2–3 weeks of kernel development + testing.  
**Estimated speedup:** 5s for vision encoder attention.  
**Status:** Not started.

---

### OPT-5: Connector Pixel Shuffle Fusion

**Target:** Connector pixel shuffle (currently 256 separate copy calls)  
**Mechanism:** The pixel shuffle reshapes [1024, 768] → [64, 12288] by rearranging
patch embeddings. Currently uses 256 copy_mod calls (one per output row quarter).
A single custom kernel could do all 256 rearrangements in one dispatch.

**Constraint:** The rearrangement pattern is fixed (stride-4 gather), so a single
vectorized kernel is straightforward. Data is 1024×768×2 = 1.5MB — too large for
one AIE call, but 4 calls (256KB each) is much better than 256 calls.

**Difficulty:** Easy-Medium. Write a `pixel_shuffle_4x.cc` kernel that processes
256 output rows per call instead of 1. Change Python loop from 256 to 4 iterations.  
**Estimated speedup:** 252 fewer calls × 24ms ≈ 6s.  
**Status:** Not started.

---

### OPT-6: Fused QKV Projection

**Target:** Vision encoder Q, K, V projections (3 × 256 = 768 calls)  
**Mechanism:** Q, K, V are all [SEQ, EMBD] @ [EMBD, EMBD] with the same input X.
Fusing into one kernel that writes three output tiles cuts 768 → 256 calls.

**Constraint:** Allo's `ExternalModule` currently supports only one output index group.
A 4-input (X, Wq, Wk, Wv), 3-output kernel would require either:
(a) A custom multi-output Allo kernel — may hit the same memref.copy bug seen with sin_cos
(b) Packing Wq+Wk+Wv into one concatenated weight matrix [EMBD, 3×EMBD] and using
    one GEMM call, then splitting the output — cleaner option.

**Difficulty:** Medium-Hard. The weight concatenation approach is clean but requires
changes to model loading and the GEMM kernel config.  
**Estimated speedup:** 512 fewer calls × 24ms ≈ 12s.  
**Status:** Not started.

---

## Priority Order & Implementation Plan

### Phase 1 — Low-hanging fruit (try first, ~2 weeks)

| # | Opt | Est. speedup | Effort |
|---|-----|-------------|--------|
| 1 | **OPT-2: FFN K-chunk fusion** | ~12s | 1–2 days |
| 2 | **OPT-5: Pixel shuffle fusion** | ~6s | 2–3 days |
| 3 | **OPT-1: Larger GEMM tiles (Pn=8)** | 15–25s | 3–5 days |

**Steps for OPT-2:**
1. Check if connector GEMM config works for vision encoder FFN down (`K_TILE=768, Pk=4`)
2. Replace Python K-chunk loop in `vision_block_bf16.py` with a single GEMM call
3. Test with `kernel_testing/` or vision_block standalone

**Steps for OPT-5:**
1. Write `cc/bf16_new/pixel_shuffle_256.cc` — processes 64 output rows per call (4 calls total)
2. Register in `connector_bf16.py`, replace 256-call loop with 4-call loop
3. Test correctness with connector standalone test

**Steps for OPT-1:**
1. Profile memory usage for various (M, Pn, Pk) configs in vision encoder
2. Try `LINEAR_TILE=128, Pn=8, Pk=2` in `vision_block_bf16.py`
3. Verify output correctness, measure call count reduction

---

### Phase 2 — Medium effort (~3–4 weeks)

| # | Opt | Est. speedup | Effort |
|---|-----|-------------|--------|
| 4 | **OPT-6: Fused QKV** | ~12s | 1 week |
| 5 | **OPT-3: Norm+GEMM fusion** | ~3s | 1 week |

**Steps for OPT-6:**
1. Concatenate Wq, Wk, Wv into `W_qkv [EMBD, 3*EMBD]` at model load time
2. Single GEMM call: `[SEQ, EMBD] @ [EMBD, 3*EMBD]` → `[SEQ, 3*EMBD]`
3. Slice output on CPU: `Q=out[:,:EMBD]`, `K=out[:,EMBD:2*EMBD]`, `V=out[:,2*EMBD:]`
4. Verify this doesn't re-introduce dispatch overhead in slicing

**Steps for OPT-3:**
1. Write fused `rms_norm_gemm_bf16.cc` kernel: reads input, normalizes, outputs GEMM-ready tile
2. Add to Allo as a new region
3. Replace `norm_fn()` + first GEMM call in text/action encoder forward

---

### Phase 3 — High effort (~4–6 weeks)

| # | Opt | Est. speedup | Effort |
|---|-----|-------------|--------|
| 6 | **OPT-4: Flash attention** | ~5s | 3–4 weeks |

**Steps for OPT-4:**
1. Implement `flash_attn_bf16.cc` on AIE: online softmax with Q/K/V tile streaming
2. Validate numerics against PyTorch `scaled_dot_product_attention`
3. Integrate into `vision_block_bf16.py`, replace score+softmax+value triple

---

## Expected Total Speedup (if remaining phases completed)

### Updated baseline (after all 2026-05-12 "more cores" work)

| Stage | Time |
|-------|------|
| Preprocessing | 8.6s |
| Vision encoder (1L) | 17.8s |
| Connector | 19.9s |
| Joint transformer (2L) | 12.8s |
| Postprocessing | 0.06s |
| **Total** | **~59.1s** |

Vision encoder softmax still the dominant cost: 384 calls × ~34ms ≈ 13s/block.
Joint transformer bottleneck: ~6.4s/layer (RoPE + GEMM + softmax + FFN).

### Vision encoder — remaining optimization targets

| After | Vision block | Reduction from 17.8s |
|-------|-------------|----------------------|
| Current baseline | 17.8s | — |
| OPT-4: Flash attention (~12 calls replace 384+24) | ~5–7s | ~60–70% |
| OPT-2: FFN K-chunk fusion (4→1 call) | ~17.3s | ~0.5s |
| Flash attn + OPT-2 | ~5–6s | ~65–70% |

Note: Flash attention is the only remaining change that can reach <10s for vision encoder.
All other vision encoder optimizations (OPT-1/2/3/5/6) save at most 1–3s each.
