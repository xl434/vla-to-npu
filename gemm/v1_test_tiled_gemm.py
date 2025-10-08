# from: https://github.com/cornell-zhang/allo/blob/a21e78af4ca9597e11e2e4ccf8f2bb09cc4bd4c5/tests/dataflow/aie/gpt2/test_gpt2_block.py#L340

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import MultiHeadAttention
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16, int32, int8, int16
from allo.memory import Layout
from allo.backend.aie import ExternalModule
import time

torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Model Configuration
# ===============================================================================
USE_ALL_NPU_KERNELS = True  # if False, we will offload softmax and silu to cpu
KERNEL_LIB_PATH = "../cc/"
BATCH = 1  # fixme: don't care for now
SEQ = 64
EMBD = 768  # 64 * 12
Q_H = 15
KV_H = 5
HEAD_DIM = 64
FFN_HID = EMBD * 4


# ===============================================================================
# Allo Version
# ===============================================================================
# Ty = float32  # All tensors use float32
# N = BATCH * SEQ  # 16   flattened (batch*seq)

# map from allo dtype to numpy dtype
dtype_map = {
    float32: np.float32,
    bfloat16: np_bfloat16,
    int8: np.int8,
    int16: np.int16,
}


def test_tiled_gemm(M, N, K, TyI, TyO):
    assert TyI == TyO
    LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
    Mt, Nt = M // LINEAR_M, N // LINEAR_N

    linear_A_layout = Layout("S0R")
    linear_B_layout = Layout("RS1")
    linear_C_layout = Layout("S0S1")

    @df.region()
    def linear_matmul_kernel():
        @df.kernel(mapping=[4, 4])
        def gemm(
            A: TyI[LINEAR_M, LINEAR_K] @ linear_A_layout,
            B: TyI[LINEAR_K, LINEAR_N] @ linear_B_layout,
            C: TyO[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.matmul(A, B)


    @df.region()
    def linear_accumulate_kernel():
        @df.kernel(mapping=[2, 4])
        def core(
            A: TyO[LINEAR_M, LINEAR_N] @ linear_C_layout,
            B: TyO[LINEAR_M, LINEAR_N] @ linear_C_layout,
            C: TyO[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.add(A, B)

    linear_matmul_mod = df.build(
        linear_matmul_kernel, 
        target="aie", 
        project="llama/linear_matmul.prj",
        profile=True,
        warmup=20,
        num_iters=100,
    )
    linear_accumulate_mod = df.build(
        linear_accumulate_kernel, 
        target="aie", 
        project="llama/linear_accumulate.prj",
        profile=True,
        warmup=20,
        num_iters=100,
    )

    if TyI is bfloat16:
        A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
        B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
    elif TyI is float32:
        A = (np.random.random((M, K)) * 0.1).astype(np.float32)
        B = (np.random.random((K, N)) * 0.1).astype(np.float32)
        C = np.zeros((M, N)).astype(np.float32)
    elif TyI is int8:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int8)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
    elif TyI is int16:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
    else:
        raise ValueError(f"unsupported data type {TyI}")

    # run Allo
    start_time = time.perf_counter()
    for i in range(Mt):
        for j in range(Nt):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(dtype_map[TyO])
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
    end_time = time.perf_counter()
    print(f"Allo end-to-end time: {(end_time - start_time)*1e6:.3f}us")

    # run reference with numpy
    C_ref = A @ B
    # verification
    if TyI is bfloat16:
        np.testing.assert_allclose(
            C.astype(np.float32), C_ref.astype(np.float32), atol=1e-1, rtol=1e-2
        )
    else:
        np.testing.assert_allclose(
            C, C_ref, atol=1e-5, rtol=1e-2
        )
    print("PASSED!")


if __name__ == "__main__":
    # M = SEQ # 64
    # N = Q_H * HEAD_DIM # 15 * 64 = 960
    # K = EMBD # 768

    # M, N, K = 128, 128, 128

    M, N, K = 64, 960, 768




    # test_tiled_gemm(M, N, K, float32, float32)
    # test_tiled_gemm(M, N, K, int8, int8)


    test_tiled_gemm(M, N, K, bfloat16, bfloat16) # 
