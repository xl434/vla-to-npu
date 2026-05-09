# Kernel Call Analysis — SmolVLA NPU Pipeline

> **Updates (2026-05-07):**
> 1. **RoPE fusion** — `rope_apply_packed` was switched from a 9-kernel decomposition
>    (radians, pack, sin_cos, copyL/R, mul, add, sub, join) to a single fused kernel
>    `rope_fused_mod`. Sin/cos are precomputed on the host. Joint-transformer RoPE
>    calls dropped **1183 → 195** (−988, ~83% reduction).
> 2. **SiLU tile** — `SILU_SEQ_TILE` was raised 4 → 16 (tile=32 overflowed action_expert
>    at 32 KB ping-pong = 64 KB tile budget). SiLU calls dropped **80 → 20** (−60).
> Measured joint-transformer wall-clock after both: **27.132 s → 25.352 s** (the SiLU step alone).

## Root Cause of Slowness: Subprocess Dispatch Overhead

Every NPU kernel call spawns a subprocess with **~24 ms overhead** (XRT setup + file I/O +
process start), measured in `analysis/measure_overhead_breakdown.py`. The actual NPU compute
is a small fraction of that time. So the number of kernel calls directly determines latency.

---

## RoPE: Decomposed → Fused (`rope_fused_mod`)

### Before — `rope_apply_packed` decomposed (9 kernels per head per tile + 3 shared)

```
Shared per tile:   radians_mod + pack_mod + sin_cos_mod           = 3 calls
Per head per tile: copyL(x) + copyR(x) + copyL(sin) + copyL(cos)
                   + mul×4 + sub + add + join                     = 10 calls
```

Tile size used `ROPE_TILE = 64`, so SEQ=128 → 2 tiles, SEQ=32 → 1 tile.

### After — `rope_fused_mod` (single fused kernel: split + mul/sub/add + join)

```
Per head per tile: rope_fused_mod(x, sin_cos, out)                = 1 call
Sin/cos precomputed on host (numpy) — 0 NPU calls
```

Tile size is `ROPE_FUSED_TILE = 32`, so SEQ=128 → 4 tiles, SEQ=32 → 1 tile.
The smaller tile is forced by the 64 KB tile-memory budget (3×[64][64] ping-pong overflows).

### Side-by-side call counts

| Block | Before (decomposed, tile=64) | After (fused, tile=32) | Saved |
|-------|------------------------------|------------------------|-------|
| Text encoder Q (per layer) | 2×(3+15·10) = **306** | 4·15 = **60** | −246 |
| Text encoder K (per layer) | 2×(3+5·10) = **106** | 4·5 = **20** | −86 |
| AE self Q | 1×(3+15·10) = **153** | 1·15 = **15** | −138 |
| AE self K | 1×(3+5·10) = **53** | 1·5 = **5** | −48 |
| AE cross Q | 1×(3+15·10) = **153** | 1·15 = **15** | −138 |
| **Joint transformer total (2 layers)** | **1183** | **195** | **−988 (~83%)** |

---

## Kernel Calls per Forward Function (current state: fused RoPE)

The "Before" / "After" columns below isolate the **SiLU tile change** (4 → 16).
RoPE numbers reflect the post-fusion state in both columns.

### `text_encoder_forward` (SEQ=128, Q_H=15, KV_H=5)

| Operation | Detail | Before | After |
|-----------|--------|--------|-------|
| RMSNorm × 2 | 128/16 per norm × 2 | 16 | 16 |
| gemm_q | 1 | 1 | 1 |
| gemm_kv × 2 (K and V) | 2 | 2 | 2 |
| RoPE Q | 4 tiles × 15 heads (fused) | 60 | 60 |
| RoPE K | 4 tiles × 5 heads (fused) | 20 | 20 |
| gemm_attn_score loop | 1 per head, Q_H=15 | 15 | 15 |
| masked_softmax loop | 15 heads × 128/8 row-tiles | 240 | 240 |
| gemm_attn_value loop | 1 per head, Q_H=15 | 15 | 15 |
| gemm_out | 1 | 1 | 1 |
| gemm_gate + gemm_up | 2 | 2 | 2 |
| **SiLU** | row-tiled (4 → 16) | **32** | **8** |
| gemm_ffn_down | 8 K-chunks (2560/320) | 8 | 8 |
| **TOTAL** |  | **412** | **388** |

### `action_expert_self_forward` (SEQ=32, Q_H=15, KV_H=5)

| Operation | Detail | Before | After |
|-----------|--------|--------|-------|
| RMSNorm × 2 | 32/16 per norm × 2 | 4 | 4 |
| gemm_q + gemm_kv_self × 2 | 3 | 3 | 3 |
| RoPE Q | 1 tile × 15 heads (fused) | 15 | 15 |
| RoPE K | 1 tile × 5 heads (fused) | 5 | 5 |
| gemm_attn_self_score loop | 1 per head, Q_H=15 | 15 | 15 |
| masked_softmax | CPU only | 0 | 0 |
| gemm_attn_self_value loop | 1 per head, Q_H=15 | 15 | 15 |
| gemm_out + gemm_gate + gemm_up | 3 | 3 | 3 |
| **SiLU** | row-tiled (4 → 16) | **8** | **2** |
| gemm_ffn_down | 8 K-chunks (2048/256) | 8 | 8 |
| **TOTAL** |  | **76** | **70** |

### `action_expert_cross_forward` (SEQ=32, TEXT_SEQ=128, Q_H=15)

| Operation | Detail | Before | After |
|-----------|--------|--------|-------|
| RMSNorm × 2 | 32/16 per norm × 2 | 4 | 4 |
| gemm_q + gemm_kv_cross × 2 | 3 | 3 | 3 |
| RoPE Q only | 1 tile × 15 heads (fused) | 15 | 15 |
| gemm_attn_cross_score loop | 1 per head, Q_H=15 | 15 | 15 |
| softmax_cross (NPU, unmasked) | 1 per head, Q_H=15 | 15 | 15 |
| gemm_attn_cross_value loop | 1 per head, Q_H=15 | 15 | 15 |
| gemm_out + gemm_gate + gemm_up | 3 | 3 | 3 |
| **SiLU** | row-tiled (4 → 16) | **8** | **2** |
| gemm_ffn_down | 8 K-chunks (2048/256) | 8 | 8 |
| **TOTAL** |  | **86** | **80** |

---

## Joint Transformer Totals — Before vs After (LLAMA_NUM_LAYERS=2, SKIP=2)

Layer 0 uses self-attention; layer 1 uses cross-attention.
Both states use the fused RoPE; only SiLU differs.

**Before SiLU change (fused RoPE + SiLU tile=4):**

| Layer | text_encoder | action_expert | Layer total |
|-------|-------------|---------------|-------------|
| 0 (self-attn)  | 412 | 76 | **488** |
| 1 (cross-attn) | 412 | 86 | **498**  |
| **TOTAL**       | **824** | **162** | **986** |

**After SiLU change (fused RoPE + SiLU tile=16):**

| Layer | text_encoder | action_expert | Layer total |
|-------|-------------|---------------|-------------|
| 0 (self-attn)  | 388 | 70 | **458** |
| 1 (cross-attn) | 388 | 80 | **468**  |
| **TOTAL**       | **776** | **150** | **926** |

**Comparison (joint transformer):**

| Metric | Before SiLU | After SiLU | Δ |
|--------|-------------|------------|---|
| Joint-transformer kernel calls | 986 | 926 | **−60** (−6.1%) |
| SiLU calls (subset) | 80 | 20 | **−60** (−75%) |
| Projected @ 24 ms/call | 23.7 s | 22.2 s | −1.5 s |
| Measured wall-clock | 27.132 s | **25.352 s** | **−1.78 s (−6.6%)** |

**Pre-fusion baseline for context** (decomposed RoPE + SiLU tile=4): **1974 calls.**
The RoPE fusion alone removed 988 calls (1974 → 986), the SiLU change another 60 (986 → 926).

---

## What's Making It Slow — current state

Masked softmax now dominates more than RoPE does, because the RoPE fusion already
took out the bulk of those calls.

| Source | Calls | % of 926 |
|--------|-------|----------|
| **Masked softmax (text encoder)** | **480** | **51.8%** |
| RoPE (all, fused) | **195** | **21.1%** |
| Other GEMMs + RMSNorm | **79** | 8.5% |
| Attn score GEMMs (head loop) | **60** | 6.5% |
| Attn value GEMMs (head loop) | **60** | 6.5% |
| FFN down (K-chunked) | **32** | 3.5% |
| SiLU (row-tiled, tile=16) | **20** | 2.2% |
| **TOTAL** | **926** | 100% |

RoPE breakdown across the 2-layer pipeline (post-fusion):

| Block | Calls |
|-------|-------|
| Text encoder Q (2 layers) | 2×60 = 120 |
| Text encoder K (2 layers) | 2×20 = 40  |
| AE self Q                 | 15  |
| AE self K                 | 5   |
| AE cross Q                | 15  |
| **Total RoPE**            | **195** |

---

## Fusion Opportunities

### 1. RoPE Fusion — APPLIED (`rope_apply_packed` → `rope_fused_mod`)

The decomposed pipeline (radians + pack + sin_cos + 10 ops per head) was replaced by
a single `rope_fused_float32` kernel that does split+mul+sub+add+join in one shot,
with sin/cos precomputed on host. Saved **988 calls** in the joint transformer.

A further fusion to `rope_full.cc` (computes sin/cos on-chip too) is still possible:
that would reuse the existing `cc/float/rope_full.cc` and remove the host-side
`_precompute_sin_cos`, but the call-count is already at 1 NPU call per head per tile,
so further savings come from reducing tile count (i.e., increasing the tile size,
which currently overflows tile memory at 64×64 — see RoPE 2x experiment).

---

### 2. Masked Softmax: Batch Heads — 240 → 16 calls per encoder layer (HIGH)

Now the largest bucket. Current loop:

```python
for h in range(Q_H):                          # 15 heads
    for tile_idx in range(SEQ // 8):           # 16 row-tiles of 8 rows each
        masked_softmax_mod(...)                # 1 call = 240 total per layer
```

A kernel that processes all 15 heads per row-tile would reduce to **16 calls per layer**,
saving 224 per layer, **448 total** (≈ 10.7 s at 24 ms/call).

---

### 3. Attention Head Loops: Per-Head GEMM → Batched (MEDIUM)

Score and value GEMMs loop over Q_H=15 individually:

```python
for k_idx in range(Q_H):   # 15 calls each for score and value
    gemm_attn_score_mod(Q_head, K_head_T, score)
    gemm_attn_value_mod(head_weight, head_value, head_out)
```

A batched GEMM stacking all heads would reduce 30 → 2 per attention block
(save 28 × 4 attn blocks = **112 calls**, ≈ 2.7 s).

---

### 4. SiLU: Merge Row Tiles — APPLIED (tile 4 → 16)

Memory check at tile=16 fits comfortably in the 64 KB AIE2 tile (peak 32 KB ping-pong
on action_expert). Tile=32 was attempted and overflowed action_expert at exactly the
64 KB budget, so 16 is the current ceiling without redistributing the feature dim
across more cores.

| Site | Before (tile=4) | After (tile=16) | Saved |
|------|-----------------|-----------------|-------|
| text_encoder MLP / layer | 32 | 8 | 24 |
| AE self/cross MLP / layer | 8 | 2 | 6 |
| **Joint transformer total (2 layers)** | **80** | **20** | **60** |

---

### 5. RMSNorm: Larger Tiles (LOW)

Text encoder runs 128/16 = 8 calls per norm. Increasing `NORM_SEQ_TILE` to 128 drops it
to 1 call per norm, saving 7 per norm × 4 norms in the 2-layer pipeline = **28 calls**
(≈ 0.7 s).

---

## Projected Impact of All Fusions

Original baseline (pre-fusion, decomposed RoPE + SiLU tile=4): **1974 calls.**
Two changes already applied; remaining work below.

| Fix | Status | Calls saved | Running total |
|-----|--------|-------------|---------------|
| Original baseline | — | — | 1974 |
| **RoPE fusion (decomposed → `rope_fused_mod`)** | **DONE** | **988** | **986** |
| **SiLU tile merge (4 → 16)** | **DONE** | **60** | **926** |
| Masked softmax head batching | TODO | 448 | 478 |
| Batched attention GEMMs | TODO | 112 | 366 |
| RMSNorm tile increase | TODO | 28 | 338 |

After all remaining fusions: **~338 calls** (≈ 8.1 s at 24 ms/call vs. ~47 s baseline).
Wall-clock speedup will be smaller because launches already overlap (current 926 calls
run in ~25 s, not 22 s).
