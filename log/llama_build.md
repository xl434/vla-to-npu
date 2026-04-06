Good thinking. Nearby multiples of 64:

  ┌──────┬─────┬────────────────────────┬───────────────────┐
  │ Size │ /64 │    Bundling options    │ Distance from 720 │
  ├──────┼─────┼────────────────────────┼───────────────────┤
  │ 640  │ 10  │ 10%2=0, 10%5=0         │ -80 (11% smaller) │
  ├──────┼─────┼────────────────────────┼───────────────────┤
  │ 768  │ 12  │ 12%3=0, 12%4=0, 12%6=0 │ +48 (6.7% larger) │
  ├──────┼─────┼────────────────────────┼───────────────────┤
  │ 704  │ 11  │ 11 is prime — bad      │ -16               │
  └──────┴─────┴────────────────────────┴───────────────────┘

  768 is the best choice — it's already proven in the existing llama_block_rope.py, has the most flexible bundling (divisible by 2,3,4,6),
  and is only 6.7% larger than 720.

  Final dimensions for implementation

  ┌───────────────────┬──────────────────┬─────────────────────────────────────────┐
  │                   │   Text Encoder   │              Action Expert              │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ SEQ               │ 128              │ 32                                      │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ EMBD              │ 960              │ 768                                     │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ Q_H               │ 15               │ 15                                      │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ KV_H              │ 5                │ 5                                       │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ HEAD_DIM          │ 64               │ 64                                      │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ Attn dim (Q_H×HD) │ 960              │ 960                                     │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ FFN_HID           │ 2560             │ 2048                                    │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ q_proj            │ 960→960          │ 768→960                                 │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ k_proj (self)     │ 960→320          │ 768→320                                 │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ k_proj (cross)    │ —                │ 320→320                                 │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ o_proj            │ 960→960          │ 960→768                                 │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ gate/up           │ 960→2560         │ 768→2048                                │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ down              │ 2560→960         │ 2048→768                                │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ RoPE              │ yes              │ yes (both self & cross)                 │
  ├───────────────────┼──────────────────┼─────────────────────────────────────────┤
  │ Attn pattern      │ causal self-attn │ alternating self/cross (every 2 layers) │
  └───────────────────┴──────────────────┴─────────────────────────────────────────┘

  All dimensions are multiples of 64.

---

# SmolVLA Single-Layer Implementation Plan (bf16)

## Overview

Build a single text-encoder layer and a single action-expert layer on NPU,
referencing the pattern in `llama_block_rope_bf16.py`. Focus on bf16 since the
GEMM module only supports bf16. Four deliverable files:

- `text_encoder_bf16.py` — 1 text encoder layer
- `text_encoder.py` — 1 text encoder layer (float32, for reference/debug)
- `action_expert_bf16.py` — 1 action expert layer
- `action_expert.py` — 1 action expert layer (float32, for reference/debug)

## Existing Kernel Inventory

| Kernel file | Signature | Reusable? |
|---|---|---|
| `cc/bf16_old/rms_norm_bf16.cc` | `rms_norm_bf16(bf16 [4][768], bf16 [768], bf16 [4][768])` | NO — width=768 hardcoded |
| `cc/bf16_new/rms_norm_bf16.cc` | `rms_norm(bf16 [4][768], bf16 [768], bf16 [4][768])` | NO — width=768 hardcoded |
| `cc/bf16_old/silu_bf16.cc` | `silu_bf16(bf16 [4][768], bf16 [4][768])` | NO — width=768 hardcoded |
| `cc/bf16_new/silu_bf16.cc` | `silu_bf16(bf16 [4][768], bf16 [4][768])` | NO — width=768 hardcoded |
| `cc/float/masked_softmax.cc` | `masked_softmax_float32(f32 [32][64], int [1], f32 [32][64])` | NO — 64-col rows only |
| `cc/bf16_old/v2_softmax_bf16.cc` | `softmax_bf16_32_128(bf16 [32][128], bf16 [32][128])` | YES — for 128-col softmax |
| `cc/bf16_old/v2_softmax_bf16.cc` | `softmax_bf16_64_64(bf16 [64][64], bf16 [64][64])` | Possibly — for 64-col softmax |
| `cc/float/transpose_matmul_with_scale.cc` | `transpose_matmul_with_scale(f32 [32][64], f32 [32][64], f32 [32][32])` | Float32 only, HEAD_DIM=64 |
| `cc/bf16_old/transpose_matmul_with_scale_bf16.cc` | `transpose_matmul_with_scale_bf16(bf16 [32][64], bf16 [32][64], bf16 [32][32])` | YES — HEAD_DIM=64 attn score |
| `cc/float/rope_vec_ops.cc` | Various (radians, pack, copy, mul, etc.) | YES — RoPE pipeline |
| `cc/float/sine.cc` / `cosine.cc` | `sin_float32(f32 [32][64], f32 [32][64])` | YES — RoPE sin/cos |
| GEMM module (`allo.library.aie.modules.gemm`) | `GEMM(M, N, K, Pm, Pn, Pk, TyI, TyO, col_num, row_num)` | YES — bf16 matmul |

## Phase 1: New Kernels to Write & Test

Each kernel gets a `.cc` file and a standalone test script.

### Kernel 1: `rms_norm_960_bf16.cc`
- **Signature**: `rms_norm_960_bf16(bf16 [4][960], bf16 [960], bf16 [4][960])`
- **Why**: Text encoder EMBD=960. Current kernel hardcodes width=768.
- **Impl**: Copy template from `cc/bf16_new/rms_norm_bf16.cc`, change `HIDDEN=768` → `HIDDEN=960`.
- **Test**: `test_rms_norm_960_bf16.py` — compare vs `torch.nn.RMSNorm(960)`.
- **Location**: `cc/bf16/rms_norm_960_bf16.cc`

### Kernel 2: `rms_norm_768_bf16.cc`
- **Signature**: `rms_norm_768_bf16(bf16 [4][768], bf16 [768], bf16 [4][768])`
- **Why**: Action expert EMBD=768. Already exists but may need renaming for clarity.
- **Impl**: Reuse existing `cc/bf16_old/rms_norm_bf16.cc` or `cc/bf16_new/rms_norm_bf16.cc` as-is.
- **Test**: Already tested (existing llama_block_rope_bf16.py uses it). No new test needed.

### Kernel 3: `silu_960_bf16.cc`
- **Signature**: `silu_960_bf16(bf16 [4][960], bf16 [4][960])` — NOT needed, see below.
- **Actually**: SiLU is applied to FFN gate output, which is `[SEQ, FFN_HID]`.
  - Text encoder: `[128, 2560]` — need `silu_bf16` that handles width=2560.
  - Action expert: `[32, 2048]` — need `silu_bf16` that handles width=2048.
- **But**: The SiLU kernel is tiled via the allo region/mapping. The external kernel
  processes a tile of `[SILU_SEQ_TILE, FFN_HID // (SILU_P0 * SILU_P1)]`.
  With SILU_P0=4, SILU_P1=4 (16 cores), each core processes `[4, FFN_HID/16]`:
  - Text: `[4, 2560/16] = [4, 160]` — 160 is fine (multiple of 16 for vectorization).
  - Expert: `[4, 2048/16] = [4, 128]` — 128 is fine.
- **The C++ kernel uses a template parameterized by tensor shape**. But the extern "C" wrapper
  is hardcoded to `[4][768]`. Need new wrappers:

### Kernel 3a: `silu_2560_bf16.cc`
- **Signature**: `silu_2560_bf16(bf16 [4][160], bf16 [4][160])`
- **Why**: Text encoder FFN_HID=2560, tiled to [4][160] per core.
- **Test**: `test_silu_2560_bf16.py` — compare vs `torch.nn.SiLU()`.

### Kernel 3b: `silu_2048_bf16.cc`
- **Signature**: `silu_2048_bf16(bf16 [4][128], bf16 [4][128])`
- **Why**: Action expert FFN_HID=2048, tiled to [4][128] per core.
- **Test**: `test_silu_2048_bf16.py` — compare vs `torch.nn.SiLU()`.

### Kernel 4: `masked_softmax_128_bf16.cc`
- **Signature**: `masked_softmax_128_bf16(bf16 [32][128], int [1], bf16 [32][128])`
- **Why**: Text encoder causal self-attention score is [128, 128] per head.
  Need masked softmax over 128-col rows with causal mask. Current float32 kernel only does 64-col.
- **Impl**: Extend `masked_softmax.cc` pattern to 128 cols with bf16 input/output
  (internally promote to float32 for exp/division). Use 4 × `aie::vector<bfloat16, 32>` loads.
  Or: convert bf16→f32, apply mask, use existing float softmax pattern, convert back.
- **Alternative**: Use existing `softmax_bf16_32_128` for unmasked path (cross-attn),
  and write a new masked variant for the text encoder.
- **Test**: `test_masked_softmax_128.py` — compare vs `torch.softmax` with causal mask.

### Kernel 5: `softmax_32x128_bf16.cc` (unmasked)
- **Signature**: `softmax_32x128_bf16(bf16 [32][128], bf16 [32][128])`
- **Why**: Action expert cross-attention score is [32, 128] per head. No causal mask.
- **Impl**: Already exists! `softmax_bf16_32_128` in `cc/bf16_old/v2_softmax_bf16.cc`.
- **Test**: `test_softmax_32x128_bf16.py` — quick validation that existing kernel works.

### Kernel 6: `rms_norm_960_float32.cc` (for float32 versions)
- **Signature**: `rms_norm_960(f32 [4][960], f32 [960], f32 [4][960])`
- **Why**: Float32 text encoder needs width=960 RMSNorm.
- **Impl**: Copy from `cc/float/rms_norm.cc`, change 768→960.
- **Test**: `test_rms_norm_960.py`

### Kernel 7: `masked_softmax_128_float32.cc` (for float32 versions)
- **Signature**: `masked_softmax_128_float32(f32 [32][128], int [1], f32 [32][128])`
- **Why**: Float32 text encoder causal softmax over 128 cols.
- **Impl**: Extend existing `masked_softmax.cc` from 64 to 128 cols.
  Uses 4 × `aie::vector<float, 32>` instead of 2.
- **Test**: `test_masked_softmax_128.py`

## Phase 2: GEMM Configurations (bf16, using GEMM module)

All use `LINEAR_TILE=64`. The GEMM module handles tiling automatically.
Key constraint: `Pm % row_num == 0` and `Pn % col_num == 0`.

### Text Encoder GEMMs (SEQ=128, EMBD=960, FFN=2560)

| Name | M×N×K | Pm×Pn×Pk | row_num | col_num | Notes |
|---|---|---|---|---|---|
| gemm_text_q | 128×960×960 | 2×15×15 | 1 | 5 or 3 | 15%5=0, 15%3=0 |
| gemm_text_kv | 128×320×960 | 2×5×15 | 1 | 5 | 5%5=0 |
| gemm_text_out | 128×960×960 | 2×15×15 | 1 | 5 or 3 | same as q |
| gemm_text_ffn_up | 128×2560×960 | 2×40×15 | 1 | 5 | 40%5=0 |
| gemm_text_ffn_down | 128×960×2560 | 2×15×40 | 1 | 5 | Pk=40 → chain; or chunk K |
| gemm_text_attn | 128×128×64 | 4×4×2 (tile=32) | 2 | 2 | attn score & value |

**Note on ffn_down**: Pk=40 means 40 chained cores on K — too many. Chunk K:
`GEMM(128, 960, 960, 2, 15, 15)` and loop 2560/960 ≈ 2.67 — not clean.
Better: `GEMM(128, 960, 640, 2, 15, 10)` and chunk 2560/640=4. Or use
`GEMM(128, 960, 320, 2, 15, 5)` and chunk 2560/320=8.
Simplest: `GEMM(128, 960, 320, 2, 15, 5, col_num=5, row_num=1)` with 8 K-chunks.

### Action Expert GEMMs (SEQ=32, EMBD=768, FFN=2048)

| Name | M×N×K | Pm×Pn×Pk | row_num | col_num | Notes |
|---|---|---|---|---|---|
| gemm_exp_q | 32×960×768 | 1×15×12 | 1 | 5 or 3 | Self-attn: 768→960 |
| gemm_exp_kv_self | 32×320×768 | 1×5×12 | 1 | 5 | Self-attn K/V |
| gemm_exp_kv_cross | 128×320×320 | 2×5×5 | 1 | 5 | Cross-attn: text KV 320→320 |
| gemm_exp_out | 32×768×960 | 1×12×15 | 1 | 3 or 4 | 12%3=0, 12%4=0 |
| gemm_exp_ffn_up | 32×2048×768 | 1×32×12 | 1 | 4 | 32%4=0 |
| gemm_exp_ffn_down | 32×768×2048 | 1×12×32 | 1 | 3 or 4 | Pk=32 → chunk K |
| gemm_exp_attn_self | 32×32×64 | 1×1×2 (tile=32) | 1 | 1 | too small? |
| gemm_exp_attn_cross | 32×128×64 | 1×4×2 (tile=32) | 1 | 2 or 4 | Q[32] × K[128] |
| gemm_exp_attn_val | 32×64×128 | 1×2×4 (tile=32) | 1 | 2 | attn_weight × V |

**Note on exp_ffn_down**: Pk=32 is too many. Chunk:
`GEMM(32, 768, 256, 1, 12, 4, col_num=4, row_num=1)` with 2048/256=8 K-chunks.

**Note on exp_attn_self**: Pm=Pn=Pk=1 won't work with GEMM module (minimum mapping).
Use tile=32 with the bf16 attn_score external kernel instead (same as text: `transpose_matmul_with_scale_bf16`).

## Phase 3: RoPE

Existing RoPE kernels (radians, pack, sin, cos, copyL/R, join, mul, add, sub) all
use `SEQ_TILE=64, HEAD_DIM=64`. Both text encoder (HEAD_DIM=64) and action expert
(HEAD_DIM=64) match. The host-side `rope_apply_packed()` already loops over
sequence in 64-row tiles, so:
- Text encoder (SEQ=128): 2 tiles of 64 rows — works as-is.
- Action expert (SEQ=32): 1 tile of 64 rows with 32 valid rows, pad rest — works as-is.

No new RoPE kernels needed.

## Phase 4: End-to-End Layer Implementation

### Step 1: Text Encoder Layer (`text_encoder_bf16.py`)
Reference: `llama_block_rope_bf16.py`

```
Forward flow:
  x [128, 960]
  ├── RMSNorm(960) → x_norm
  ├── GEMM q_proj: [128, 960] × [960, 960] → Q [128, 960]
  ├── GEMM k_proj: [128, 960] × [960, 320] → K [128, 320]
  ├── GEMM v_proj: [128, 960] × [960, 320] → V [128, 320]
  ├── RoPE(Q, heads=15), RoPE(K, heads=5)
  ├── Attn score: per head [128, 64] × [128, 64]^T → [128, 128], tile=32
  ├── Masked softmax [128, 128] (128-col causal) ← NEW KERNEL
  ├── Attn value: [128, 128] × [128, 64] → [128, 64], per head
  ├── GEMM o_proj: [128, 960] × [960, 960] → [128, 960]
  ├── Residual add
  ├── RMSNorm(960) → x_norm2
  ├── GEMM gate: [128, 960] × [960, 2560] → [128, 2560]
  ├── GEMM up:   [128, 960] × [960, 2560] → [128, 2560]
  ├── SiLU(gate) * up
  ├── GEMM down: [128, 2560] × [2560, 960] → [128, 960] (K-chunked)
  └── Residual add → output [128, 960]
  Also returns: K [128, 320], V [128, 320] for action expert cross-attn
```

### Step 2: Action Expert Layer (`action_expert_bf16.py`)
Two modes: self-attention (even layers) and cross-attention (odd layers).

```
Self-attention forward (layer_idx % 2 == 0):
  x [32, 768]
  ├── RMSNorm(768) → x_norm
  ├── GEMM q_proj: [32, 768] × [768, 960] → Q [32, 960]
  ├── GEMM k_proj: [32, 768] × [768, 320] → K [32, 320]
  ├── GEMM v_proj: [32, 768] × [768, 320] → V [32, 320]
  ├── RoPE(Q, heads=15), RoPE(K, heads=5)
  ├── Attn score: [32, 64] × [32, 64]^T → [32, 32] per head
  ├── Masked softmax [32, 32] (32-col causal — use existing 64-col kernel with padding,
  │   or use softmax_bf16_64_64 with padding)
  ├── Attn value: [32, 32] × [32, 64] → [32, 64] per head
  ├── GEMM o_proj: [32, 960] × [960, 768] → [32, 768]
  ├── Residual add
  ├── RMSNorm(768) → x_norm2
  ├── GEMM gate: [32, 768] × [768, 2048] → [32, 2048]
  ├── GEMM up:   [32, 768] × [768, 2048] → [32, 2048]
  ├── SiLU(gate) * up
  ├── GEMM down: [32, 2048] × [2048, 768] → [32, 768] (K-chunked)
  └── Residual add → output [32, 768]

Cross-attention forward (layer_idx % 2 == 1):
  x [32, 768], text_K [128, 320], text_V [128, 320]
  ├── RMSNorm(768) → x_norm
  ├── GEMM q_proj: [32, 768] × [768, 960] → Q [32, 960]
  ├── GEMM k_proj: [128, 320] × [320, 320] → K [128, 320]
  ├── GEMM v_proj: [128, 320] × [320, 320] → V [128, 320]
  ├── RoPE(Q, heads=15)  ← Q only, K comes from text
  ├── Attn score: [32, 64] × [128, 64]^T → [32, 128] per head (rectangular!)
  ├── Unmasked softmax [32, 128] ← use existing softmax_bf16_32_128
  ├── Attn value: [32, 128] × [128, 64] → [32, 64] per head
  ├── GEMM o_proj: [32, 960] × [960, 768] → [32, 768]
  ├── Residual add
  ├── RMSNorm(768), GEMM gate/up/down, SiLU — same as self-attn MLP
  └── Residual add → output [32, 768]
```

## Kernel Implementation Order (Priority)

### Must-have (blocking end-to-end)

| # | Kernel | File | Test | Blocked by |
|---|---|---|---|---|
| 1 | RMSNorm bf16 width=960 | `cc/bf16/rms_norm_960_bf16.cc` | `test_rms_norm_960_bf16.py` | — |
| 2 | RMSNorm f32 width=960 | `cc/float/rms_norm_960.cc` | `test_rms_norm_960.py` | — |
| 3 | Masked softmax f32 128-col | `cc/float/masked_softmax_128.cc` | `test_masked_softmax_128.py` | — |
| 4 | SiLU bf16 tile [4][160] | `cc/bf16/silu_160_bf16.cc` | `test_silu_160_bf16.py` | — |
| 5 | SiLU bf16 tile [4][128] | `cc/bf16/silu_128_bf16.cc` | `test_silu_128_bf16.py` | — |
| 6 | SiLU f32 tile [4][160] | `cc/float/silu_160.cc` | `test_silu_160.py` | — |
| 7 | SiLU f32 tile [4][128] | `cc/float/silu_128.cc` | `test_silu_128.py` | — |

### Already available (no new kernel needed)

| Operation | Existing kernel | Notes |
|---|---|---|
| RMSNorm bf16 width=768 | `cc/bf16_old/rms_norm_bf16.cc` | Action expert |
| Unmasked softmax bf16 [32][128] | `cc/bf16_old/v2_softmax_bf16.cc` → `softmax_bf16_32_128` | Cross-attn |
| Attn score bf16 [32][64]×[32][64] | `cc/bf16_old/transpose_matmul_with_scale_bf16.cc` | HEAD_DIM=64 |
| Attn score f32 [32][64]×[32][64] | `cc/float/transpose_matmul_with_scale.cc` | HEAD_DIM=64 |
| Masked softmax f32 [32][64] | `cc/float/masked_softmax.cc` | 64-col only (action self-attn 32-col: pad to 64) |
| RoPE (full pipeline) | `cc/float/rope_vec_ops.cc`, `sine.cc`, `cosine.cc` | HEAD_DIM=64, 64-row tiles |
| GEMM bf16 | `allo.library.aie.modules.gemm.GEMM` | All linear projections |
| Hadamard (element-wise mul) | `allo.mul` built-in | SiLU(gate) * up |

### Nice-to-have (can use CPU fallback initially)

| # | Kernel | Notes |
|---|---|---|
| 8 | Masked softmax bf16 128-col | Currently using f32 masked softmax + bf16↔f32 conversion |
| 9 | Unmasked softmax f32 [32][128] | For float32 action expert cross-attn |


# result

 ┌─────┬────────────────────────────┬────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │            Test            │ Status │                                          Notes                                           │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ test_rms_norm_960_bf16.py  │ PASSED │ 149us avg, tile [4,960]                                                                  │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 2   │ test_rms_norm_960.py       │ PASSED │ 169us avg, tile [2,960] (reduced from [4,960] for memory)                                │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ test_masked_softmax_128.py │ SKIP   │ Program memory overflow — get_exp LUT + 128-col scalar loops too large for AIE. Will use │
  │     │                            │        │  bf16 masked softmax instead (see #3b)                                                    │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 3b  │ test_masked_softmax_128    │ PASSED │ 127us avg. bf16 LUT-based masked softmax. Two fixes needed:                               │
  │     │ _bf16.py                   │        │  (a) LUT exp(-inf)=1 not 0 — must zero out masked positions AFTER exp, not before          │
  │     │                            │        │  (b) objectfifo consumer buffer is read-only — copy to local buffer for in-place masking   │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ test_silu_160_bf16.py      │ PASSED │ 124us avg                                                                                │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ test_silu_128_bf16.py      │ PASSED │ 121us avg. Fixed: padded buffer from [4][128] to [4][256] to avoid 1024-byte (2^10)       │
  │     │                            │        │  DMA transfer edge case. Kernel still only processes first 128 columns.                   │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 6   │ test_silu_160.py           │ PASSED │ 1016us avg (LUT-based)                                                                   │
  ├─────┼────────────────────────────┼────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
  │ 7   │ test_silu_128.py           │ PASSED │ 838us avg (LUT-based)                                                                    │
  └─────┴────────────────────────────┴────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

  7 of 7 pass (including new bf16 masked softmax #3b replacing the skipped f32 version #3).

  Key findings:
  - [4][128] bf16 = 1024 bytes total causes DMA corruption — pad to [4][256] (2048 bytes) to workaround
  - LUT-based bf16 exp() returns exp(-inf)=1.0 (not 0) — must zero masked positions after exp, not before
  - AIE objectfifo consumer buffers appear read-only — use local buffer for in-place modifications

---

# Phase 4: Text Encoder — PASSED

  File: `vla/text_encoder_bf16.py`
  Status: Builds, runs, and matches PyTorch bf16 reference (atol=0.1, rtol=0.1)
  Allo bf16 forward time: ~21s (vs PyTorch bf16: ~7ms)

  Dimensions: SEQ=128, EMBD=960, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2560

  ┌────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │ Component  │            Reference (LLaMA)             │                       Text Encoder                       │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ RMSNorm    │ rms_norm_bf16 (768)                      │ rms_norm_960_bf16 (960)                                  │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ SiLU       │ silu_bf16 P1=4 (768/core)                │ silu_160_bf16 P0=1,P1=16 (160/core, float32 Horner)     │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Softmax    │ float32 masked_softmax, P0×P1 multi-core │ bf16 masked_softmax, [8,128] tiles, single core per tile │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Attn GEMM  │ One shared 64×64×64                      │ Two separate: score (128×128×64) + value (128×64×128)    │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ FFN down   │ 4 chunks of 768                          │ 8 chunks of 320                                          │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ RoPE tiles │ 64 rows (SEQ=64, 1 tile)                 │ 64 rows (SEQ=128, 2 tiles)                               │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Output     │ block output only                        │ (output, key, value) tuple                               │
  └────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

  Issues resolved during integration:
  1. SILU P0=4,P1=16 = 64 cores exceeded 16-core NPU limit → fixed: P0=1,P1=16 = 16 cores, SILU_SEQ_TILE=4
  2. bf16 SiLU Taylor series overflow: x^12+ exceeds bf16 max for |x|>2.83 → fixed: Horner's method with
     float32 internal computation for the exp(-x) Taylor series, bf16 for sigmoid/div/select.
     - Horner evaluates polynomial inside-out: t = c16; t = c15 + x*t; ... keeping intermediates bounded
     - Float32 needed because bf16 loses precision at each Horner step (7-bit mantissa too few)
     - aie::div only works for bf16 vectors on AIE2, not float32 → compute exp(-x) in float32,
       convert to bf16, then sigmoid/div/select in bf16
  3. GEMM Pn=40 for FFN up (2560/64) — builds and runs fine
  4. Softmax loop: 15 heads × 16 tiles = 240 kernel calls — works, main contributor to latency

# Phase 4: Action Expert — TODO

  Next step. Two modes: self-attention (even layers) and cross-attention (odd layers).

  What's Missing                                                             
                  
  1. Action Expert Layer (action_expert_bf16.py) — the main blocker          
  
  Per the plan in llama_build.md, this needs two modes:                      
  - Self-attention (even layers): SEQ=32, EMBD=768, FFN_HID=2048
  - Cross-attention (odd layers): Q from [32,768], K/V from text encoder     
  [128,320]                                                             
                                                                             
  New GEMM configs needed (different from text encoder and llama block):
  - gemm_exp_q: 32×960×768                                                   
  - gemm_exp_kv_self: 32×320×768                                             
  - gemm_exp_kv_cross: 128×320×320                                           
  - gemm_exp_out: 32×768×960                                                 
  - gemm_exp_ffn_up: 32×2048×768                                             
  - gemm_exp_ffn_down: 32×768×2048 (K-chunked)
  - Cross-attention GEMM: 32×128×64 (rectangular attention score)            
                                                                             
  Also needs a SiLU kernel with tile [4][128] (already tested and passed per 
  the log).                                                                  
                                                                             
  2. Dimension Mismatches in vla.py                                          
                  
  The current vla.py has several issues:                                     
  - EXPERT_HIDDEN = TEXT // 10 = 96 — likely a placeholder; the plan says
  action expert EMBD=768                                                     
  - joint_transformer() calls llama_block_rope (SEQ=64, EMBD=768) for both
  VLM and expert — should use text_encoder_forward for VLM and a new         
  action_expert_forward for the expert                                       
  - Params: params_exp and params_vlm both use EMBD_MM (768) dimensions, but
  VLM should use EMBD=960                                                    
  - main() returns early at line 256, short-circuiting the entire pipeline   
  
  3. Postprocessing                                                          
                  
  Currently uses EMBD=96 with rms_norm_96_bf16.cc — this needs to match the  
  action expert output dimension (768 per the plan).
                                                                             
  Suggested Next Steps

  1. Build the action expert layer — this is the critical path. It follows   
  the same pattern as text_encoder_bf16.py but with different dimensions and
  the added cross-attention mode. The kernel inventory is mostly ready (SiLU 
  [4][128] passed, RMSNorm 768 exists, softmax_bf16_32_128 exists for
  cross-attn).
  2. Fix vla.py dimensions and wiring — update to use text_encoder_forward
  for the VLM path and the new action expert for the expert path, fix        
  EXPERT_HIDDEN, fix postprocessing norm width.
  3. Remove the early return in main() and wire the full pipeline.

---

# Phase 4: Action Expert — PASSED

  File: `vla/action_expert_bf16.py`
  Status: Builds, runs, and matches PyTorch bf16 reference (atol=0.1, rtol=0.1)
  Tested: 2026-03-30

  Dimensions: SEQ=32, EMBD=768, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2048

  Two modes implemented:
  - Self-attention (even layers): causal self-attention, SEQ=32
  - Cross-attention (odd layers): Q from action [32,768], K/V from text encoder [128,320]

  ┌──────────────────┬─────────────┬─────────────┬───────────┐
  │ Mode             │  Max Error  │ Mean Error  │  NPU Time │
  ├──────────────────┼─────────────┼─────────────┼───────────┤
  │ Self-attention   │ 0.0469      │ 0.0079      │ ~7.5s     │
  ├──────────────────┼─────────────┼─────────────┼───────────┤
  │ Cross-attention  │ 0.0332      │ 0.0049      │ ~6.4s     │
  └──────────────────┴─────────────┴─────────────┴───────────┘

  GEMM configurations (10 total):

  ┌──────────────────────────┬─────────────────┬──────────────┬──────────────────────────────────┐
  │ GEMM                     │ Shape (MxNxK)   │ Pm x Pn x Pk │ Usage                            │
  ├──────────────────────────┼─────────────────┼──────────────┼──────────────────────────────────┤
  │ gemm_q                   │ 32×960×768      │ 1×15×12      │ Q projection (shared)            │
  │ gemm_kv_self             │ 32×320×768      │ 1×5×12       │ Self-attn K/V                    │
  │ gemm_kv_cross            │ 128×320×320     │ 2×5×5        │ Cross-attn K/V (320→320)         │
  │ gemm_out                 │ 32×768×960      │ 1×12×15      │ Output projection (shared)       │
  │ gemm_attn_self_score     │ 32×32×64        │ 1×1×2        │ Self-attn Q@K^T (tile=32)        │
  │ gemm_attn_self_value     │ 32×64×32        │ 1×2×1        │ Self-attn weights@V (tile=32)    │
  │ gemm_attn_cross_score    │ 32×128×64       │ 1×4×2        │ Cross-attn Q@K^T (tile=32)       │
  │ gemm_attn_cross_value    │ 32×64×128       │ 1×2×4        │ Cross-attn weights@V (tile=32)   │
  │ gemm_ffn_up              │ 32×2048×768     │ 1×32×12      │ Gate/Up projection (shared)      │
  │ gemm_ffn_down            │ 32×768×256      │ 1×12×4       │ FFN down, 8 K-chunks             │
  └──────────────────────────┴─────────────────┴──────────────┴──────────────────────────────────┘

  ┌────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │ Component  │            Reference (LLaMA)             │                      Action Expert                       │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ RMSNorm    │ rms_norm_bf16 (768)                      │ rms_norm_bf16 (768) — reused                             │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ SiLU       │ silu_bf16 P1=4 (768/core, old Taylor)    │ silu_256_bf16 P0=1,P1=8 (256/core, float32 Horner)      │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Softmax    │ float32 masked_softmax, P0×P1 multi-core │ Self: CPU causal masked [32,32]                          │
  │            │                                          │ Cross: NPU softmax_bf16_32_128 unmasked [32,128]         │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Attn GEMM  │ One shared 64×64×64                      │ Self: score(32×32×64) + value(32×64×32)                  │
  │            │                                          │ Cross: score(32×128×64) + value(32×64×128)               │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ FFN down   │ 4 chunks of 768                          │ 8 chunks of 256                                          │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ RoPE tiles │ 64 rows (SEQ=64, 1 tile)                 │ 64 rows (SEQ=32, 1 tile padded to 64)                    │
  ├────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Cross K/V  │ n/a                                      │ 320→320 projection via gemm_kv_cross (128×320×320)       │
  └────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

  Issues resolved during integration:
  1. silu_256_bf16.cc was a "test variant" using old bf16 Taylor series — overflow for |x|>2.0, max output 158.
     Fixed: rewrote with Horner's method + float32 internals (same as silu_160_bf16.cc). Now correct.
  2. NPU context limit: 23 module builds + another user's NPU process caused DRM_IOCTL timer expired errors.
     Workaround: added --mode self/cross CLI flag to split builds (~19 each). For VLA integration,
     need to manage NPU context lifetime (build/release modules in stages rather than all at once).
  3. Self-attn [32,32] softmax: too small for NPU kernel overhead. Used CPU masked softmax instead.
     Cross-attn [32,128] softmax: perfect fit for existing softmax_bf16_32_128 NPU kernel.
  4. Cross-attention RoPE: Q only (K/V from text encoder don't get RoPE), per SmolVLA architecture.

  Exported functions:
  - action_expert_self_forward(x, params) → [32, 768]
  - action_expert_cross_forward(x, text_k, text_v, params) → [32, 768]

---

# Phase 5: End-to-End VLA Integration — TODO

  File: `vla/vla.py`

  ## What works (NPU, individually tested)
  - Preprocessing (Conv2d): `preprocessing_bf16.py`
  - Vision Encoder (ViT): `vision_block_bf16.py` (SEQ=1024, EMBD=768)
  - Connector: `connector_bf16.py` (1024×768 → 64×960)
  - Text Encoder layer: `text_encoder_bf16.py` (SEQ=128, EMBD=960) — PASSED
  - Action Expert layer: `action_expert_bf16.py` (SEQ=32, EMBD=768) — PASSED

  ## What needs fixing in vla.py
  1. EXPERT_HIDDEN = 96 is a placeholder — actual action expert EMBD=768
  2. joint_transformer() uses llama_block_rope (SEQ=64, EMBD=768) for VLM —
     should use text_encoder_forward (SEQ=128, EMBD=960)
  3. joint_transformer() uses llama_block_rope for expert —
     should use action_expert_self_forward / action_expert_cross_forward
  4. params_vlm uses EMBD_MM=768 dims — should use text encoder EMBD=960
  5. params_exp uses EMBD_MM dims — should use action expert dims (768/320/2048)
  6. main() returns early at line 256, short-circuiting the pipeline
  7. Postprocessing uses rms_norm_96_bf16 — needs rms_norm_768_bf16
  8. NPU context management: all sub-modules build their own NPU kernels at import.
     Running the full pipeline may exceed NPU context limits (~23+ per module).
     Strategy: sequential import/build, or restructure to share kernel builds.

---

# Phase 5: VLA Integration — In Progress (2026-03-30)

  ## Cleanup: Removed debug `_check` from action_expert_bf16.py and text_encoder_bf16.py

  Both files had a `_check(name, arr)` function that printed "OK ..." for every
  intermediate tensor (rmsnorm, query, key, value, attn_weight, etc.). Removed
  the function definition and all call sites:
  - `action_expert_bf16.py`: 24 `_check()` calls removed
  - `text_encoder_bf16.py`: 16 `_check()` calls removed

  ## Rewrote vla.py for correct module wiring

  Previous state: vla.py imported `llama_block_rope_bf16` (SEQ=64, EMBD=768) for
  both VLM and expert paths, had placeholder dimensions (EXPERT_HIDDEN=96), and
  returned early at line 256.

  ### Changes made

  | Area | Old | New |
  |---|---|---|
  | Imports | `llama_block_rope_bf16` | `text_encoder_bf16` + `action_expert_bf16` |
  | VLM forward | `llama_block_rope()` | `text_encoder_forward()` (SEQ=128, EMBD=960) |
  | Expert forward | `llama_block_rope()` / `llama_block_rope_cross()` | `action_expert_self_forward()` (even layers) / `action_expert_cross_forward()` (odd layers) |
  | `EXPERT_HIDDEN` | 96 (placeholder) | Removed — uses `EMBD_EXP = 768` from action_expert_bf16 |
  | `params_vlm` | EMBD_MM=768, FFN=4×768 | EMBD=960, Q_H×HEAD_DIM=960, KV=320, FFN=2560 |
  | Expert params | Single `params_exp` dict | Split into `params_exp_self` (Wq/Wk/Wv) + `params_exp_cross` (Wq/Wk_cross/Wv_cross) |
  | Cross-attn weights | Missing | `Wk_cross` [320,320], `Wv_cross` [320,320] |
  | `action` input | `[32, 96]` with padding | `[32, 768]` |
  | Early return | `return` at line 256 | Removed — full pipeline runs |
  | Postprocessing GEMM | `GEMM(32, 32, 96, ...)` | `GEMM(32, 32, 768, 1, 1, 12)` |
  | Postprocessing norm | `rms_norm_96_bf16.cc` (width=96) | `rms_norm_bf16.cc` (width=768) |
  | CPU reference | `AttentionExpertBlock`/`CrossAttentionBlock` from llama_block_rope_bf16 | `TextEncoderBlock` from text_encoder_bf16, `ActionExpertSelfBlock`/`ActionExpertCrossBlock` from action_expert_bf16 |

  ### Pipeline flow (NPU)

  ```
  image [3, 512, 512]
    → preprocessing_block → [1024, 768]
    → vision_encoder (12 layers) → [1024, 768]
    → connector_block → [64, 960]
    → concat(llama_emb[64,960], text_emb[48,960], state_emb[1,960], zeros[15,960])
    → mm_seq [128, 960]

  joint_transformer (16 layers):
    for each layer i:
      text_encoder_forward(mm_seq) → (mm_seq [128,960], text_k [128,320], text_v [128,320])
      if i % 2 == 0: action_expert_self_forward(action) → action [32,768]
      if i % 2 == 1: action_expert_cross_forward(action, text_k, text_v) → action [32,768]

  postprocessing:
    rms_norm_768(action) → [32, 768]
    GEMM → [32, 32]  (action output)
  ```

  ### Known issue: NPU context limits
  The full pipeline imports preprocessing_bf16, vision_block_bf16, connector_bf16,
  text_encoder_bf16, and action_expert_bf16 — each builds multiple NPU modules at
  import time. This may exceed the NPU context limit (~23 concurrent modules).
  Not yet tested end-to-end on device.

  # Phase 5: End-to-End Test Results

  ## Test 1: LLAMA_NUM_LAYERS=11 (full pipeline) — FAILED (NaN)

  Date: 2026-04-06

  ┌─────────────────────────────┬────────────┐
  │ Stage                       │  NPU Time  │
  ├─────────────────────────────┼────────────┤
  │ Preprocessing               │ 449s       │
  │ Vision encoder (1L)         │ 50s        │
  │ Connector                   │ 96s        │
  │ Joint transformer (11L)     │ 310s       │
  │ Postprocessing              │ 0.06s      │
  │ Total                       │ 908s       │
  └─────────────────────────────┴────────────┘

  Result: FAILED — NPU output is all NaN.
  SiLU reports `Bad input range: [nan, nan]` — values already NaN before FFN gate.
  Root cause: 11 layers with random weights in bf16 causes values to grow unbounded
  and overflow to NaN. CPU reference (PyTorch bf16) produces valid values due to
  more robust intermediate accumulation.

  ## Test 2: LLAMA_NUM_LAYERS=1 (self-attention only) — FAILED (precision)

  Date: 2026-04-06

  ┌─────────────────────────────┬────────────┐
  │ Stage                       │  NPU Time  │
  ├─────────────────────────────┼────────────┤
  │ Preprocessing               │ 444s       │
  │ Vision encoder (1L)         │ 50s        │
  │ Connector                   │ 97s        │
  │ Joint transformer (1L)      │ 29s        │
  │ Postprocessing              │ 0.06s      │
  │ Total                       │ 620s       │
  └─────────────────────────────┴────────────┘

  Result: No NaN — real values produced. Pipeline runs end-to-end.
  Assertion failed: atol=0.1, rtol=0.1
  - Mismatched elements: 605 / 1024 (59.1%)
  - Max absolute difference: 38.5
  - Max relative difference: 878.1
  - SiLU: 281,600 bad values patched with CPU fallback
  Note: Only self-attention tested (layer 0, i%2==0). Cross-attention not exercised.

  ## Test 3: LLAMA_NUM_LAYERS=2 (self + cross attention) — FAILED (NaN)

  Date: 2026-04-06

  ┌─────────────────────────────┬────────────┐
  │ Stage                       │  NPU Time  │
  ├─────────────────────────────┼────────────┤
  │ Preprocessing               │ 445s       │
  │ Vision encoder (1L)         │ 50s        │
  │ Connector                   │ 97s        │
  │ Joint transformer (2L)      │ 57s        │
  │ Postprocessing              │ 0.06s      │
  │ Total                       │ 648s       │
  └─────────────────────────────┴────────────┘

  Result: FAILED — NPU output is all NaN (same as 11-layer).
  SiLU reports 281,600 bad values (layer 0 text encoder) then 327,680 (layer 1).
  NaN already appears in the first text encoder call, same as 1-layer test.
  The 1-layer test produced real values but with 59% mismatch — the error from
  layer 0 is large enough that layer 1 overflows to NaN.

  Key observation: The text encoder's first call (layer 0) already has SiLU overflow
  issues even with just 1 layer. The connector output (64×960) feeding into the
  text encoder contains large values from accumulating 192 bf16 GEMM tiles
  (K=12288/64=192), which then blow up through the text encoder's projections.