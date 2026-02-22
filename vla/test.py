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
input_layout = [R, R]
output_layout = [R, S(1)]

@df.region()
def linear_matmul_kernel(A: Ty[LINEAR_M, LINEAR_K], B: Ty[LINEAR_K, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[1], args=[A, B, C])
    def gemm(
        local_A: Ty[LINEAR_M, LINEAR_K] @ input_layout,
        local_B: Ty[LINEAR_K, LINEAR_N] @ input_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ input_layout,
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
    arr = np.zeros((16, 64, 240))
    for i in range(NEW_SEQ):
        offset = (i // 8) * 128 + (i % 8) * 4
        tile_C = np.zeros((NEW_SEQ, TEXT)).astype(np.float32)
        for j in range(4):
            for k in range(4):
                for l in range(120):
                    tile_A = A[ 
                        offset + j * 32 + k : offset + j * 32 + k + 1,
                        l*8:(l+1) * 8
                    ]
                    tile_B = B[
                        (j * 4 + k) * EMBD : (j * 4 + k + 1) * EMBD,
                        l*8:(l+1) * 8
                    ]
                    tile_C = arr[j * 4 + k, i:i+1, l*8:(l+1) * 8]
                    print(tile_A.shape, tile_B.shape, tile_C.shape)
                    linear_matmul_mod(tile_A, tile_B, tile_C)

    C = np.zeros((M, N)).astype(float32)
    for a in arr:
        for i in range(15):
            linear_accumulate_mod(a, C[:,i*64:(i+1)*64], C[:,i*64:(i+1)*64])

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