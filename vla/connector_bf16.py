"""
connector_bf16.py — pixel shuffle + linear projection on AMD NPU.

Transforms vision tokens [1024, 768] → [64, 960]:
  1. Pixel shuffle (NPU): [1024, 768] → [64, 12288] via 4 NPU calls  (OPT-B)
  2. GEMM (NPU):          [64, 12288] @ [12288, 960] → [64, 960]  (480 calls, n_m=2 × n_n=15 × n_k=16)
  3. Accumulation (CPU):  float32 accumulator across 16 K-tiles

OPT-B: pixel shuffle reduced from 256 NPU calls to 4 NPU calls.
Uses mapping=[16] with direct assignment (copy_region pattern) to avoid
ExternalModule asymmetric-shape limitation.

Pre-gather (CPU, no arithmetic): reorder input into [256, 768] sorted by
[core_group, output_row], where core k = j*4+g handles column chunk j, group g.
NPU redistributes to [16, 12288] in one dispatch via direct assignment.

Per-core SRAM for pixel_shuffle: A[16,768]=24KB + Out[16,768]=24KB = 48KB < 64KB ✓
Per-core SRAM for GEMM: A[32,192]×2 + B[192,32]×2 + C_pipe[32,32]×4 ≈ 57 KB < 64 KB ✓
"""

import time
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from allo.memory import Layout

np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

SEQ      = 1024
EMBD     = 768
NEW_SEQ  = 64          # 1024 / (4×4)
NEW_EMBD = 12288       # 768 × 4 × 4
TEXT     = 960
M_TILE   = 32          # M-tile (NEW_SEQ/M_TILE = 2 M-iterations)
K_TILE   = 768         # large K-tile: 16 K-iterations instead of 192
N_TILE   = 64          # N-tile per GEMM call (TEXT/N_TILE = 15 N-iterations)

# Pixel shuffle tile dimensions (OPT-B)
SHUFFLE_BATCH = 8      # output rows per NPU call (double-buf: 4×12KB=48KB < 64KB)
SHUFFLE_CORES = 4      # same as copy_region: 4 input-groups per call

Ty = bfloat16

# ---------------------------------------------------------------------------
# OPT-B: precompute row-index tables for 4-call NPU pixel shuffle.
#
# _PIXEL_ROW_IDX[j][i*4+r] = input row for output row i, input-group r, col-chunk j.
# _PS_ROW_IDX[b, k, :]     = 16 input rows for batch b, core k (k = j*4 + g).
#
# Pixel shuffle formula verified:
#   out[b*16+r, j*3072 + g*768 : j*3072 + (g+1)*768] = A[_PS_ROW_IDX[b, j*4+g, r], :]
# ---------------------------------------------------------------------------
_PIXEL_ROW_IDX = np.array(
    [[(i // 8) * 128 + (i % 8) * 4 + j * 32 + r
      for i in range(NEW_SEQ) for r in range(4)]
     for j in range(4)],
    dtype=np.intp,
)  # shape [4, 256]

# Pre-sorted row indices for NPU pixel shuffle: 4 col-chunks × 4 row-batches = 16 calls
# _PS_ROW_IDX[j, b, g, :] = SHUFFLE_BATCH input rows for col-chunk j, row-batch b, group g
_n_col_chunks  = 4
_n_row_batches = NEW_SEQ // SHUFFLE_BATCH   # 4
_PS_ROW_IDX = np.empty((_n_col_chunks, _n_row_batches, SHUFFLE_CORES, SHUFFLE_BATCH), dtype=np.intp)
for _j in range(_n_col_chunks):
    for _b in range(_n_row_batches):
        for _g in range(SHUFFLE_CORES):
            _PS_ROW_IDX[_j, _b, _g, :] = _PIXEL_ROW_IDX[_j][_g::4][_b * SHUFFLE_BATCH:(_b + 1) * SHUFFLE_BATCH]

# ---------------------------------------------------------------------------
# pixel_shuffle_region — OPT-B: 16 calls (4 col-chunks × 4 row-batches), was 256.
#
# Same pattern as copy_region but with B=16 rows per call instead of B=1.
# mapping=[4]: core g gets input rows [g*16:(g+1)*16] and output cols [g*768:(g+1)*768].
# Per-core shapes: [16, 768] → [16, 768] (symmetric) → direct assignment works.
# Per-core SRAM: 24KB in + 24KB out = 48KB < 64KB ✓
# ---------------------------------------------------------------------------
ps_in_layout  = [S(0), R]
ps_out_layout = [R, S(0)]

@df.region()
def pixel_shuffle_region(
    A:   Ty[SHUFFLE_CORES * SHUFFLE_BATCH, EMBD],
    Out: Ty[SHUFFLE_BATCH, SHUFFLE_CORES * EMBD],
):
    @df.kernel(mapping=[SHUFFLE_CORES], args=[A, Out])
    def mod(
        local_A:   Ty[SHUFFLE_CORES * SHUFFLE_BATCH, EMBD]  @ ps_in_layout,
        local_Out: Ty[SHUFFLE_BATCH, SHUFFLE_CORES * EMBD]  @ ps_out_layout,
    ):
        local_Out[:, :] = local_A[:, :]

pixel_shuffle_mod = df.build(
    pixel_shuffle_region, target="aie", project="connector/pixel_shuffle.prj"
)

# ---------------------------------------------------------------------------
# copy_mod — original single-row pixel shuffle tile (kept as unused reference)
# ---------------------------------------------------------------------------
linear_in_layout  = [S(0), R]
linear_out_layout = [R, S(0)]

@df.region()
def copy_region(A: Ty[4, EMBD], C: Ty[1, EMBD * 4]):
    @df.kernel(mapping=[4], args=[A, C])
    def mod(
        local_A: Ty[4, EMBD]     @ linear_in_layout,
        local_C: Ty[1, EMBD * 4] @ linear_out_layout,
    ):
        local_C[:, :] = local_A[:, :]

copy_mod = df.build(copy_region, target="aie", project="connector/copy.prj")

# ---------------------------------------------------------------------------
# GEMM module — built once at import time (NOT inside the hot-path function)
#
# GEMM(M=32, N=64, K=768, Pn=2, Pk=4): 8 cores (2 N-lanes × 4 K-chain).
# Per-core SRAM identical to preprocessing fused GEMM (~57 KB).
# ---------------------------------------------------------------------------
Pn = 2
Pk = 4

_gemm_top, _mapping_primitives = GEMM(M_TILE, N_TILE, K_TILE,
                                       Pm=1, Pn=Pn, Pk=Pk,
                                       TyI=Ty, TyO=Ty)
gemm_mod = df.build(
    _gemm_top,
    target="aie",
    project="connector/gemm.prj",
    mapping_primitives=_mapping_primitives,
)


def fused_op(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    t0 = time.time()

    # Pixel shuffle on NPU: [1024, 768] → [64, 12288]  (OPT-B: 16 calls, was 256)
    # 4 col-chunks × 4 row-batches. For each (j, b):
    #   1. CPU pre-gather: reorder 64 rows by group → [64, 768] (no arithmetic)
    #   2. NPU dispatch (mapping=[4]): direct copy to [16, 3072]
    A_  = np.empty((NEW_SEQ, NEW_EMBD), dtype=np_bfloat16)
    _in  = np.empty((SHUFFLE_CORES * SHUFFLE_BATCH, EMBD), dtype=np_bfloat16)
    _out = np.empty((SHUFFLE_BATCH, SHUFFLE_CORES * EMBD), dtype=np_bfloat16)

    for j in range(_n_col_chunks):
        for b in range(_n_row_batches):
            for g in range(SHUFFLE_CORES):
                _in[g * SHUFFLE_BATCH:(g + 1) * SHUFFLE_BATCH, :] = A[_PS_ROW_IDX[j, b, g], :]
            pixel_shuffle_mod(_in, _out)
            A_[b * SHUFFLE_BATCH:(b + 1) * SHUFFLE_BATCH,
               j * SHUFFLE_CORES * EMBD:(j + 1) * SHUFFLE_CORES * EMBD] = _out

    t1 = time.time()

    # GEMM: [64, 12288] @ [12288, 960] → [64, 960]
    # Outer loops: M-tiles (2) × N-tiles (15); inner K-loop (16) accumulates in f32.
    n_k = NEW_EMBD // K_TILE   # 16
    n_m = NEW_SEQ  // M_TILE   # 2
    n_n = TEXT     // N_TILE   # 15

    C_acc_f32 = np.zeros((NEW_SEQ, TEXT), dtype=np.float32)
    C_tmp     = np.empty((M_TILE, N_TILE), dtype=np_bfloat16)

    for m in range(n_m):
        for n in range(n_n):
            for k in range(n_k):
                A_tile = np.ascontiguousarray(
                    A_[m * M_TILE:(m + 1) * M_TILE,
                       k * K_TILE:(k + 1) * K_TILE])          # [32, 768]
                B_tile = np.ascontiguousarray(
                    B[k * K_TILE:(k + 1) * K_TILE,
                      n * N_TILE:(n + 1) * N_TILE])            # [768, 64]
                gemm_mod(A_tile, B_tile, C_tmp)
                C_acc_f32[m * M_TILE:(m + 1) * M_TILE,
                          n * N_TILE:(n + 1) * N_TILE] += C_tmp.astype(np.float32)

    C[:] = C_acc_f32.astype(np_bfloat16)

    t2 = time.time()
    n_calls = n_m * n_n * n_k
    n_ps_calls = _n_col_chunks * _n_row_batches
    print(f"pixel shuffle (NPU): {(t1 - t0) * 1e3:.2f} ms  ({n_ps_calls} calls, OPT-B mapping=[4])")
    print(f"GEMM          (NPU): {(t2 - t1) * 1e3:.2f} ms  ({n_calls} calls, {n_k} K-tiles/block, f32 accum)")
    print(f"total:               {(t2 - t0) * 1e3:.2f} ms")


def connector_block(x: np.ndarray, params: dict) -> np.ndarray:
    out = np.zeros((NEW_SEQ, TEXT), dtype=np_bfloat16)
    fused_op(x, params["W"], out)
    return out


# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np_bfloat16)
    w = np.random.randn(NEW_EMBD, TEXT).astype(np_bfloat16)
    out = connector_block(x, {"W": w})

    # Reference: CPU pixel shuffle + matmul
    x_ref = (x.astype(np.float32)
              .reshape(32, 32, EMBD)
              .reshape(32, 8, EMBD * 4)
              .transpose(1, 0, 2)
              .reshape(8, 8, NEW_EMBD)
              .transpose(1, 0, 2)
              .reshape(NEW_SEQ, NEW_EMBD))
    expected = x_ref @ w.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), expected, rtol=1e-1, atol=6.0)
    print("PASSED")
