import time
import torch
import torch.nn as nn
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

BATCH = 1
SEQ = 1024
EMBD = 768
SF = 4
NEW_SEQ = 64 # 1024 / 4 / 4
NEW_EMBD = 12288 # 768 * 4 * 4
TEXT = 960

# ===============================================================================
# Allo Version
# ===============================================================================

Ty = float32

# ----------------------------------------------------------------
# Linear
# ----------------------------------------------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 1, 8, 768
matmul_layout = [R, R]

@df.region()
def linear_matmul_kernel(A: Ty[LINEAR_M, LINEAR_K], B: Ty[LINEAR_K, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[1], args=[A, B, C])
    def gemm(
        local_A: Ty[LINEAR_M, LINEAR_K] @ matmul_layout,
        local_B: Ty[LINEAR_K, LINEAR_N] @ matmul_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ matmul_layout,
    ):
        tmp = allo.matmul(local_A, local_B)
        for i in range(8):
            local_C[0, i] = tmp[0, i]

M, N = 64, 64
accum_layout = [S(0), S(1)]
@df.region()
def linear_accumulate_kernel(A: Ty[M, N], B: Ty[M, N], C: Ty[M, N]):
    @df.kernel(mapping=[2,4], args=[A, B, C])
    def core(
        local_A: Ty[M, N] @ accum_layout,
        local_B: Ty[M, N] @ accum_layout,
        local_C: Ty[M, N] @ accum_layout,
    ):
        local_C[:, :] = allo.add(local_A, local_B)

# ##############################################################
# BUILD
# ##############################################################
linear_matmul_mod = df.build(linear_matmul_kernel, target="aie", project="linear_matmul.prj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie", project="linear_accumulate.prj")

# ##############################################################
# TOOL
# ##############################################################
def connector(A, B, C):
    # arr[16 partials, NEW_SEQ=64 rows, TEXT=960 cols]
    arr = np.zeros((16, NEW_SEQ, TEXT), dtype=np.float32)

    # Precompute all source row indices for each (i, j, k)
    # to avoid recomputing offset inside the hot loop
    for i in range(NEW_SEQ):
        offset = (i // 8) * 128 + (i % 8) * 4
        for j in range(4):
            for k in range(4):
                src_row     = offset + j * 32 + k
                partial_idx = j * 4 + k
                tile_A      = A[src_row : src_row + 1, :]  # (1, 768) — no copy, read-only

                for l in range(TEXT // LINEAR_N):           # 80 iterations
                    col_start = l * LINEAR_N
                    col_end   = col_start + LINEAR_N

                    tile_B = B[
                        partial_idx * EMBD : (partial_idx + 1) * EMBD,
                        col_start : col_end,
                    ]                                       # (768, 12) — no copy, read-only
                    tile_C = arr[partial_idx, i : i + 1, col_start : col_end]  # (1, 12) — write in-place

                    linear_matmul_mod(tile_A, tile_B, tile_C)
                    # tile_C is a view into arr, so result lands directly — no copy back needed

    # Accumulate 16 partial sums into C (64, 960)
    # Process in (64, 64) tiles to match compiled shape of linear_accumulate_mod
    C[:] = arr[0]
    for partial_idx in range(1, 16):
        for t in range(TEXT // N):                          # 960 / 64 = 15
            col_start = t * N
            col_end   = col_start + N
            result    = np.zeros((M, N), dtype=np.float32)
            linear_accumulate_mod(
                arr[partial_idx, :, col_start : col_end],   # (64, 64)
                C[:, col_start : col_end],                  # (64, 64)
                result,
            )
            C[:, col_start : col_end] = result

def linear_projection(A, B, C, M, N, K):
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
            for k in range(K // LINEAR_K):
                tile_A = A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                ]
                tile_B = B[
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ]
                linear_matmul_mod(tile_A, tile_B, C_tmp)
                linear_accumulate_mod(
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    C_tmp,
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )

# Weight matrix is size [12288, 960]
def connector_block(x_fp32: np.ndarray, params: dict):
    x = x_fp32.astype(np.float32)
    x = x.reshape(SEQ, EMBD)
    out = np.zeros((NEW_SEQ, TEXT), dtype=np.float32)
    t0 = time.time()
    connector(x, params["W"], out)
    t1 = time.time()
    print("total execution time: " + str(t1 - t0))
    return out

# ##############################################################
# TEST
# ##############################################################
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np.float32)
    w = np.random.randn(EMBD*(SF**2), TEXT).astype(np.float32)
    dict = {"W": w}
    out = connector_block(x, dict)

    # test python
    x = x.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    expected = x @ w
    np.testing.assert_allclose(out, expected, rtol=1e-5)