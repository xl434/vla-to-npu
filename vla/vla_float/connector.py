import time
import torch
import torch.nn as nn
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
#from allo.backend.aie.external_kernel import ExternalModule

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

SEQ = 1024
EMBD = 768
SCALE_FACTOR = 4
NEW_SEQ = 64 # 1024 / 4 / 4
NEW_EMBD = 12288 # 768 * 4 * 4
TEXT = 960

# ===============================================================================
# Allo Version
# ===============================================================================

Ty = float32

linear_in_layout = [S(0), R]
linear_out_layout = [R, S(0)]

@df.region()
def copy(A: Ty[4, EMBD], C: Ty[1, EMBD*4]):
    @df.kernel(mapping=[4], args=[A,C])
    def mod(
        local_A: Ty[4, EMBD] @ linear_in_layout,
        local_C: Ty[1, EMBD*4] @ linear_out_layout,
    ):
        local_C[:,:] = local_A[:,:]

# ----------------------------------------------------------------
# Linear
# ----------------------------------------------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
linear_A_layout = [S(0), R]
linear_B_layout = [R, S(1)]
linear_C_layout = [S(0), S(1)]

@df.region()
def linear_matmul_kernel(A: Ty[LINEAR_M, LINEAR_K], B: Ty[LINEAR_K, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[4, 4], args=[A, B, C])
    def gemm(
        local_A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
        local_B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        local_C[:, :] = allo.matmul(local_A, local_B)

@df.region()
def linear_accumulate_kernel(A: Ty[LINEAR_M, LINEAR_N], B: Ty[LINEAR_M, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[2, 4], args=[A, B, C])
    def core(
        local_A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        local_B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        local_C[:, :] = allo.add(local_A, local_B)

# ##############################################################
# BUILD
# ##############################################################
copy_mod = df.build(copy, target="aie", project="copy.prj")
linear_matmul_mod = df.build(linear_matmul_kernel, target="aie", project="linear_matmul.prj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie", project="linear_accumulate.prj")

# ##############################################################
# TOOL
# ##############################################################
def pixel_shuffle(A, C):
    for i in range(NEW_SEQ):
        offset = (i // 8) * 128 + (i % 8) * 4
        for k in range(4):
            tile_A = A[ 
                offset + k * 32 : offset + k * 32 + 4,
                :
            ]
            tile_C = C[
                i: i + 1,
                k * EMBD * 4 : (k + 1) * (EMBD * 4)
            ]
            copy_mod(tile_A, tile_C)

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
    x_shuffled = np.zeros((NEW_SEQ, NEW_EMBD), dtype=np.float32)
    out = np.zeros((NEW_SEQ, TEXT), dtype=np.float32)
    t0 = time.time()
    pixel_shuffle(x, x_shuffled)
    t1 = time.time()
    linear_projection(x_shuffled, params["W"], out, NEW_SEQ, TEXT, NEW_EMBD)
    t2 = time.time()
    print("pixel shuffle execution time: " + str(t1 - t0))
    print("matmul execution time: " + str(t2 - t1))
    print("total execution time: " + str(t2 - t0))
    return out

# ##############################################################
# TEST
# ##############################################################
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np.float32)
    w = np.random.randn(EMBD*(SCALE_FACTOR**2), TEXT).astype(np.float32)
    dict = {"W": w}
    out = connector_block(x, dict)

    # test python
    x = x.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    expected = x @ w
    np.testing.assert_allclose(out, expected, rtol=1e-1)