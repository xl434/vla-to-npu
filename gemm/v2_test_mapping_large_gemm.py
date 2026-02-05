# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from allo.ir.types import int4, int8, int16, bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.backend.aie import is_available
import argparse


def test_pingpong_gemm(M, N, K, Pm, Pn, Pk, TyI, TyO):
    assert TyI == TyO or TyI is int4
    top, mapping_primitives = GEMM(M, N, K, Pm, Pn, Pk, TyI, TyO)

    if is_available():
        os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
        mod = df.build(
            top,
            project="top.prj",
            target="aie",
            mapping_primitives=mapping_primitives,
            profile=True,
            warmup=200,
            num_iters=1000,
        )
        if TyI is bfloat16:
            A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
            B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
            C = np.zeros((M, N)).astype(np_bfloat16)
        elif TyI in {int4, int8}:
            A = np.random.randint(-4, 4, (M, K)).astype(np.int8)
            B = np.random.randint(-4, 4, (K, N)).astype(np.int8)
            C = np.zeros((M, N)).astype(np.int8)
        elif TyI is int16:
            A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
            B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
            C = np.zeros((M, N)).astype(np.int16)
        else:
            raise ValueError(f"unsupported data type {TyI}")
        mod(A, B, C)
        if TyI is bfloat16:
            np.testing.assert_allclose(
                C.astype(np.float32), (A @ B).astype(np.float32), atol=1e-1
            )
        else:
            np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
        del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":

    # print("here")

    # parser = argparse.ArgumentParser(description="Test mapping GEMM with manual sizes")
    # parser.add_argument("--M", type=int, required=True)
    # parser.add_argument("--N", type=int, required=True)
    # parser.add_argument("--K", type=int, required=True)
    # parser.add_argument("--m", type=int, required=True)
    # parser.add_argument("--n", type=int, required=True)
    # parser.add_argument("--k", type=int, required=True)
    # args = parser.parse_args()
    # M, N, K = args.M, args.N, args.K
    # m, n, k = args.m, args.n, args.k
    M, N, K = 1024, 1024, 1024
    m, n, k = 64, 64, 64
    # - i8
    test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int8, int8)

    # - i16
    test_pingpong_gemm(M, N, K, M // m, N // n, K // k, int16, int16)

    # - bf16
    try:
        test_pingpong_gemm(M, N, K, M // m, N // n, K // k, bfloat16, bfloat16)
    except:
        print("[NOTE]: bfloat16 have accuracy issue")
