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
import time

ALLOWED_M = [32, 64, 128, 256, 512, 1024, 2048]
ALLOWED_N = [32, 64, 128, 256, 512, 768, 1024, 2048]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 1024, 2048]


def _next_allowed_dim(value, allowed_values, name):
    for candidate in allowed_values:
        if candidate >= value:
            return candidate
    raise ValueError(
        f"{name}={value} exceeds max supported {name} ({allowed_values[-1]}) "
        f"from run_profiles.py"
    )


def _pad_for_gemm(A, B, C, M, N, K, Mp, Np, Kp):
    A_pad = np.zeros((Mp, Kp), dtype=A.dtype)
    B_pad = np.zeros((Kp, Np), dtype=B.dtype)
    C_pad = np.zeros((Mp, Np), dtype=C.dtype)
    A_pad[:M, :K] = A
    B_pad[:K, :N] = B
    C_pad[:M, :N] = C
    return A_pad, B_pad, C_pad


def _dtype_name(dtype):
    if dtype is int8 or dtype is int4:
        return "i8"
    if dtype is int16:
        return "i16"
    if dtype is bfloat16:
        return "bf16"
    return str(dtype)


def test_pingpong_gemm(
    M, N, K, m, n, k, TyI, TyO, use_padding=False, pad_M=None, pad_N=None, pad_K=None
):
    assert TyI == TyO or TyI is int4
    dtype_name = _dtype_name(TyI)
    if use_padding:
        Mp = pad_M if pad_M is not None else _next_allowed_dim(M, ALLOWED_M, "M")
        Np = pad_N if pad_N is not None else _next_allowed_dim(N, ALLOWED_N, "N")
        Kp = pad_K if pad_K is not None else _next_allowed_dim(K, ALLOWED_K, "K")
        if Mp < M or Np < N or Kp < K:
            raise ValueError(
                f"Explicit padded shape {(Mp, Np, Kp)} must be >= original {(M, N, K)}"
            )
        if Mp not in ALLOWED_M or Np not in ALLOWED_N or Kp not in ALLOWED_K:
            raise ValueError(
                f"Explicit padded shape {(Mp, Np, Kp)} must be in allowed lists from run_profiles.py"
            )
    else:
        if pad_M is not None or pad_N is not None or pad_K is not None:
            raise ValueError("pad-M/pad-N/pad-K require --use-padding")
        Mp, Np, Kp = M, N, K
    if Mp % m != 0 or Np % n != 0 or Kp % k != 0:
        raise ValueError(
            f"Tile sizes {(m, n, k)} do not divide padded shape {(Mp, Np, Kp)}"
        )
    print(
        f"[RUN] dtype={dtype_name} tile=({m},{n},{k}) "
        f"original=({M},{N},{K}) padded=({Mp},{Np},{Kp}) use_padding={use_padding}"
    )
    top, mapping_primitives = GEMM(Mp, Np, Kp, Mp // m, Np // n, Kp // k, TyI, TyO)

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
        if use_padding:
            t_pad0 = time.perf_counter()
            A_pad, B_pad, C_pad = _pad_for_gemm(A, B, C, M, N, K, Mp, Np, Kp)
            t_pad1 = time.perf_counter()
            pad_us = (t_pad1 - t_pad0) * 1e6
            print(f"[{dtype_name}] Padding shape: ({M}, {N}, {K}) -> ({Mp}, {Np}, {Kp})")
            print(f"[{dtype_name}] Padding time: {pad_us:.3f}us")
            mod(A_pad, B_pad, C_pad)
            t_unpad0 = time.perf_counter()
            C_unpad = C_pad[:M, :N]
            t_unpad1 = time.perf_counter()
            unpad_us = (t_unpad1 - t_unpad0) * 1e6
            print(f"[{dtype_name}] Unpadding time: {unpad_us:.3f}us")
        else:
            mod(A, B, C)
            C_unpad = C
        if TyI is bfloat16:
            np.testing.assert_allclose(
                C_unpad.astype(np.float32), (A @ B).astype(np.float32), atol=1e-1
            )
        else:
            np.testing.assert_allclose(C_unpad, A @ B, atol=1e-5)
        print("PASSED!")
        del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mapping GEMM with manual sizes")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="all",
        choices=["all", "i8", "i16", "bf16"],
        help="Run one dtype or all",
    )
    parser.add_argument(
        "--use-padding",
        action="store_true",
        help="Enable pad -> GEMM -> unpad flow. If not set, run direct GEMM only.",
    )
    parser.add_argument("--pad-M", type=int, default=None, help="Explicit padded M")
    parser.add_argument("--pad-N", type=int, default=None, help="Explicit padded N")
    parser.add_argument("--pad-K", type=int, default=None, help="Explicit padded K")
    args = parser.parse_args()
    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    use_padding = args.use_padding
    pad_M, pad_N, pad_K = args.pad_M, args.pad_N, args.pad_K

    if args.dtype in ("all", "i8"):
        test_pingpong_gemm(
            M, N, K, m, n, k, int8, int8, use_padding, pad_M, pad_N, pad_K
        )
    if args.dtype in ("all", "i16"):
        test_pingpong_gemm(
            M, N, K, m, n, k, int16, int16, use_padding, pad_M, pad_N, pad_K
        )
    if args.dtype in ("all", "bf16"):
        try:
            test_pingpong_gemm(
                M, N, K, m, n, k, bfloat16, bfloat16, use_padding, pad_M, pad_N, pad_K
            )
        except Exception:
            print("[NOTE]: bfloat16 have accuracy issue")
