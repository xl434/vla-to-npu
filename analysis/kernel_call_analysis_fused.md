# RoPE Fusion — Before vs After

This document quantifies the impact of replacing the 9-kernel `rope_apply_packed`
decomposition with a single-kernel `rope_fused_float32` (defined in
`cc/float/rope_fused.cc`) for the joint-transformer path that `vla.py` runs.

The `rope_apply_packed` entry point in `text_encoder_bf16.py` and
`action_expert_bf16.py` now dispatches the fused kernel; sin/cos are precomputed
on host and packed into one `[32][64]` buffer per tile.

Subprocess dispatch overhead per NPU kernel call: **~24 ms**
(measured in `analysis/measure_overhead_breakdown.py`).

---

## RoPE call counts per head per tile

| Variant | Per-tile shared | Per head per tile | Tile rows |
|---|---|---|---|
| Packed (before) | 3 (radians + pack + sin_cos) | 10 (copyL×3 + copyR + mul×4 + sub + add + join) | 64 |
| Fused (after) | 0 (sin/cos on host) | **1** (`rope_fused_float32`) | 32 |

Tile size halves with the fused kernel (32 vs 64), so SEQ=128 produces 4 tiles
instead of 2 — but the 1-call-per-head economy still wins by a wide margin.

---

## Per-block RoPE calls

| Block | SEQ | Tiles (before / after) | Heads | Before | After | Δ |
|---|---|---|---|---|---|---|
| Text encoder Q | 128 | 2 / 4 | Q_H=15 | 2×(3+15×10) = **306** | 4×15 = **60** | −246 |
| Text encoder K | 128 | 2 / 4 | KV_H=5 | 2×(3+5×10) = **106**  | 4×5 = **20**  | −86  |
| AE self Q     | 32  | 1 / 1 | Q_H=15 | 1×(3+15×10) = **153** | 1×15 = **15** | −138 |
| AE self K     | 32  | 1 / 1 | KV_H=5 | 1×(3+5×10) = **53**   | 1×5 = **5**   | −48  |
| AE cross Q    | 32  | 1 / 1 | Q_H=15 | 1×(3+15×10) = **153** | 1×15 = **15** | −138 |

---

## Joint-transformer totals (2 layers: layer 0 self, layer 1 cross)

| Block (×layers it runs) | RoPE before | RoPE after | Δ calls |
|---|---|---|---|
| Text encoder Q+K × 2 | 824 | 160 | **−664** |
| AE self Q+K × 1      | 206 | 20  | **−186** |
| AE cross Q × 1       | 153 | 15  | **−138** |
| **Total RoPE**       | **1183** | **195** | **−988** |

Holding all other operations constant, the joint-transformer grand total drops
from **1974 → 986** calls.

| Metric | Before | After | Speedup |
|---|---|---|---|
| RoPE calls       | 1183 | 195 | **6.07×** |
| Total calls      | 1974 | 986 | **2.00×** |
| RoPE wall time   | ~28.4 s | ~4.7 s | **−23.7 s** |
| Total wall time  | ~47.4 s | ~23.7 s | **−23.7 s** |

(Wall-time figures use the 24 ms/call dispatch overhead; actual on-device
compute is a small fraction of this.)

---

## Per-forward-function breakdown

| Function | Calls before | Calls after | RoPE share before → after |
|---|---|---|---|
| `text_encoder_forward` (SEQ=128) | 744 | 412 | 412/744 → 80/412 |
| `action_expert_self_forward` (SEQ=32) | 262 | 76  | 206/262 → 20/76 |
| `action_expert_cross_forward` (SEQ=32) | 224 | 86  | 153/224 → 15/86 |

After the swap, RoPE drops from ~60% of all joint-transformer calls to ~20%.
Masked softmax (480 calls) is now the single largest contributor.

---

## What changed in code

`vla/text_encoder_bf16.py` and `vla/action_expert_bf16.py`:

- New `ROPE_FUSED_TILE = 32` constant + `ROPE_FUSED_IMPL` path to `rope_fused.cc`.
- New `rope_fused_ext` ExternalModule (`top="rope_fused_float32"`).
- New `rope_fused_region` and `rope_fused_mod` build target.
- `rope_apply_packed` rewritten to: (a) precompute `sin_cos[32][64]` on host
  using `_precompute_sin_cos`, then (b) call `rope_fused_mod(x_tile, sin_cos, out)`
  once per head per tile.
- The legacy decomposed kernel modules (`radians_mod`, `pack_mod`, `sin_cos_mod`,
  `copyL_mod`, `copyR_mod`, `join_mod`, `mul32_mod`, `add32_mod`, `sub32_mod`)
  are still built so external scripts (e.g. `analysis/measure_repeated.py`,
  which imports `copyL_mod`) keep working — they're simply no longer on the
  inference path.

---

## Caveat

These figures count NPU kernel dispatches and multiply by the measured 24 ms
subprocess-launch cost. They do not account for the small per-tile host-side
sin/cos computation (microseconds) added by the fused path, nor for
on-device kernel-execution time differences between the packed and fused
implementations (also a small fraction of the dispatch overhead).
