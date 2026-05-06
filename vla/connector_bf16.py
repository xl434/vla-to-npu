"""
connector_bf16.py — pixel shuffle + linear projection on AMD NPU.

Transforms vision tokens [1024, 768] → [64, 960]:
  1. Pixel shuffle (NPU): [1024, 768] → [64, 12288]  via copy_mod (256 calls)
  2. GEMM (NPU):          [64, 12288] @ [12288, 960] → [64, 960]  (480 calls)
  3. Accumulation (CPU):  float32 accumulator across 16 K-tiles (not 192)

K_TILE=768 + Pk=4: each GEMM call accumulates 768 K-elements in AIE accfloat
registers before a single bf16 conversion → 16 K-tiles instead of 192, reducing
accumulated quantization error by ~12× (√192/√16 ≈ 3.5× in random-walk terms,
and 192/16=12× in worst-case bias terms).

Per-core SRAM: A[32,192]×2 + B[192,32]×2 + C_pipe[32,32]×4 ≈ 57 KB < 64 KB ✓
(identical layout to preprocessing fused GEMM)
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

KERNEL_LIB_PATH = "../cc/bf16_old/"

SEQ      = 1024
EMBD     = 768
NEW_SEQ  = 64          # 1024 / (4×4)
NEW_EMBD = 12288       # 768 × 4 × 4
TEXT     = 960
M_TILE   = 32          # M-tile (NEW_SEQ/M_TILE = 2 M-iterations)
K_TILE   = 768         # large K-tile: 16 K-iterations instead of 192
N_TILE   = 64          # N-tile per GEMM call (TEXT/N_TILE = 15 N-iterations)

Ty = bfloat16

# ---------------------------------------------------------------------------
# copy_mod — pixel shuffle tile rearrangement on NPU
# Copies [4, EMBD] → [1, EMBD*4] using 4 cores (unchanged from original)
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
# Each call accumulates K=768 elements in AIE accfloat → 1 bf16 conversion.
# ---------------------------------------------------------------------------
Pn = 2   # N-parallel lanes (N_TILE/Pn = 32 per lane)
Pk = 4   # K-chain depth    (K_TILE/Pk = 192 per stage)

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

    # Pixel shuffle on NPU: [1024, 768] → [64, 12288]  (256 copy_mod calls)
    A_ = np.zeros((NEW_SEQ, NEW_EMBD), dtype=np_bfloat16)
    for i in range(NEW_SEQ):
        offset = (i // 8) * 128 + (i % 8) * 4
        for j in range(4):
            copy_mod(
                A[offset + j * 32 : offset + j * 32 + 4, :],       # [4, 768]
                A_[i : i + 1, j * EMBD * 4 : (j + 1) * EMBD * 4], # [1, 3072]
            )

    t1 = time.time()

    # GEMM: [64, 12288] @ [12288, 960] → [64, 960]
    # Outer loops: M-tiles (2) × N-tiles (15); inner K-loop (16) accumulates in f32.
    # Both A_tile and B_tile need np.ascontiguousarray because they are column-slices
    # of row-major arrays (non-contiguous stride — Allo DMA ignores strides).
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
    print(f"pixel shuffle (NPU): {(t1 - t0) * 1e3:.2f} ms  (256 calls)")
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
    np.testing.assert_allclose(out.astype(np.float32), expected, rtol=1e-1)
    print("PASSED")
