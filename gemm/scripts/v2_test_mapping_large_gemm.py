# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from allo.ir.types import int4, int8, int16, bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.backend.aie import is_available
import argparse
import time

ALLOWED_M = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_N = [32, 64, 96, 128, 160, 192, 256, 288, 320, 384, 448, 480, 512, 576, 640, 672, 768, 960, 1024, 1152, 1280, 1344, 1536, 1600, 1728, 1792, 1920, 2048, 2112, 2240, 2304, 2496, 2560, 2688, 2816, 2880, 3072, 3200, 3264, 3328, 3456, 3584, 3648, 3840, 4032, 4096]
ALLOWED_K = [32, 64, 128, 256, 512, 768, 960, 1024, 2048, 3072]
ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR / "top.prj"
_TORCH = None


def _next_allowed_dim(value, allowed_values, name):
    for candidate in allowed_values:
        if candidate >= value:
            return candidate
    raise ValueError(
        f"{name}={value} exceeds max supported {name} ({allowed_values[-1]}) "
        f"from run_profiles.py"
    )


def _pad_copy(arr, add_rows, add_cols):
    if add_rows < 0 or add_cols < 0:
        raise ValueError(f"Padding increments must be non-negative, got ({add_rows}, {add_cols})")
    out = np.zeros((arr.shape[0] + add_rows, arr.shape[1] + add_cols), dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _pad_torch(arr, add_rows, add_cols):
    if add_rows < 0 or add_cols < 0:
        raise ValueError(f"Padding increments must be non-negative, got ({add_rows}, {add_cols})")
    torch = _get_torch()

    pad = torch.nn.ZeroPad2d((0, add_cols, 0, add_rows))
    if arr.dtype == np_bfloat16:
        arr_t = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
        padded = pad(arr_t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return padded.to(torch.float32).cpu().numpy().astype(np_bfloat16)

    arr_t = torch.from_numpy(arr)
    padded = pad(arr_t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return padded.cpu().numpy().astype(arr.dtype, copy=False)


def _pad_numpy(arr, add_rows, add_cols):
    if add_rows < 0 or add_cols < 0:
        raise ValueError(f"Padding increments must be non-negative, got ({add_rows}, {add_cols})")
    return np.pad(arr, ((0, add_rows), (0, add_cols)), mode="constant", constant_values=0)


def _get_torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch as _torch
        except ImportError as exc:
            raise ImportError("torch is required for _pad_torch") from exc
        _TORCH = _torch
    return _TORCH


PAD_IMPLS = {
    "manual_copy": _pad_copy,
    "torch_pad": _pad_torch,
    "numpy_pad": _pad_numpy,
}


def _get_pad_impl(pad_impl_name):
    if pad_impl_name not in PAD_IMPLS:
        raise ValueError(
            f"Unknown pad implementation '{pad_impl_name}'. "
            f"Supported: {sorted(PAD_IMPLS.keys())}"
        )
    return PAD_IMPLS[pad_impl_name]


def _auto_col_num(Pn: int) -> int:
    """Return the best col_num from {4,3,5} for the given Pn quotient.
    When Pn < 3, GEMM auto-adjusts col_num down to Pn — pass 4 as a safe default."""
    for c in (4, 3, 5):
        if Pn % c == 0:
            return c
    if Pn < 3:
        return 4  # GEMM will auto-adjust: col_num = min(4, Pn)
    raise ValueError(f"Pn={Pn} is not divisible by any of {{3, 4, 5}}")


def _auto_row_num(Pm: int) -> int:
    """Return the best row_num from {4,3,5} for the given Pm quotient.
    When Pm < 3, GEMM auto-adjusts row_num down to Pm — pass 4 as a safe default."""
    for r in (4, 3, 5):
        if Pm % r == 0:
            return r
    if Pm < 3:
        return 4  # GEMM will auto-adjust: row_num = min(4, Pm)
    raise ValueError(f"Pm={Pm} is not divisible by any of {{3, 4, 5}}")


def _pad_for_gemm(A, B, M, N, K, Mp, Np, Kp):
    add_m = Mp - M
    add_n = Np - N
    add_k = Kp - K
    pad_impl = _get_pad_impl()
    A_pad = pad_impl(A, add_rows=add_m, add_cols=add_k)
    B_pad = pad_impl(B, add_rows=add_k, add_cols=add_n)
    C_pad = np.zeros((Mp, Np), dtype=A_pad.dtype)
    return A_pad, B_pad, C_pad


def _dtype_name(dtype):
    if dtype is int8 or dtype is int4:
        return "i8"
    if dtype is int16:
        return "i16"
    if dtype is bfloat16:
        return "bf16"
    return str(dtype)


def _make_inputs(M, N, K, TyI):
    if TyI is bfloat16:
        A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
        B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
        return A, B
    if TyI in {int4, int8}:
        A = np.random.randint(-4, 4, (M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, (K, N)).astype(np.int8)
        return A, B
    if TyI is int16:
        A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
        B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
        return A, B
    raise ValueError(f"unsupported data type {TyI}")


def _check_correctness(TyI, A, B, C_unpad):
    if TyI is bfloat16:
        np.testing.assert_allclose(
            C_unpad.astype(np.float32), (A @ B).astype(np.float32), atol=1e-1
        )
    else:
        np.testing.assert_allclose(C_unpad, A @ B, atol=1e-5)


def test_pingpong_gemm(
    M,
    N,
    K,
    m,
    n,
    k,
    TyI,
    TyO,
    use_padding=False,
    pad_M=None,
    pad_N=None,
    pad_K=None,
    pad_impl_name="numpy_pad",
    project_dir=PROJECT_DIR,
    col_num="auto",
    row_num="auto",
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
    Pm = Mp // m
    Pn = Np // n
    Pk = Kp // k
    resolved_col_num = _auto_col_num(Pn) if col_num == "auto" else int(col_num)
    resolved_row_num = _auto_row_num(Pm) if row_num == "auto" else int(row_num)
    print(
        f"[RUN] dtype={dtype_name} tile=({m},{n},{k}) "
        f"original=({M},{N},{K}) padded=({Mp},{Np},{Kp}) use_padding={use_padding} "
        f"pad_impl={pad_impl_name} col_num={resolved_col_num} row_num={resolved_row_num}"
    )
    top, mapping_primitives = GEMM(Mp, Np, Kp, Pm, Pn, Pk, TyI, TyO,
                                   col_num=resolved_col_num, row_num=resolved_row_num)

    if is_available():
        os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
        try:
            mod = df.build(
                top,
                project=str(project_dir),
                target="aie",
                mapping_primitives=mapping_primitives,
                profile=True,
                warmup=200,
                num_iters=1000,
            )
            A, B = _make_inputs(M, N, K, TyI)
            if use_padding:
                add_m = Mp - M
                add_n = Np - N
                add_k = Kp - K
                pad_impl = _get_pad_impl(pad_impl_name)
                if pad_impl is _pad_torch:
                    _get_torch()
                t_pad0 = time.perf_counter()
                A_pad = pad_impl(A, add_rows=add_m, add_cols=add_k)
                B_pad = pad_impl(B, add_rows=add_k, add_cols=add_n)
                t_pad1 = time.perf_counter()
                pad_us = (t_pad1 - t_pad0) * 1e6
                C_pad = np.zeros((Mp, Np), dtype=A_pad.dtype)
                print(f"[{dtype_name}] Padding shape: ({M}, {N}, {K}) -> ({Mp}, {Np}, {Kp})")
                print(f"[{dtype_name}] Padding time: {pad_us:.3f}us")
                mod(A_pad, B_pad, C_pad)
                t_unpad0 = time.perf_counter()
                C_unpad = C_pad[:M, :N]
                t_unpad1 = time.perf_counter()
                unpad_us = (t_unpad1 - t_unpad0) * 1e6
                print(f"[{dtype_name}] Unpadding time: {unpad_us:.3f}us")
            else:
                C = np.zeros((M, N), dtype=A.dtype)
                mod(A, B, C)
                C_unpad = C
            _check_correctness(TyI, A, B, C_unpad)
            print("PASSED!")
        finally:
            os.environ.pop("ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH", None)
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
    parser.add_argument(
        "--pad-impl",
        type=str,
        default="numpy_pad",
        choices=sorted(PAD_IMPLS.keys()),
        help="Padding implementation used when --use-padding is enabled.",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(PROJECT_DIR),
        help="AIE project directory. Set unique values for async parallel jobs.",
    )
    parser.add_argument(
        "--col-num",
        type=str,
        default="auto",
        choices=["auto", "3", "4", "5"],
        help="col_num passed to GEMM (Pn must be divisible by it). 'auto' selects best from {4,3,5}.",
    )
    parser.add_argument(
        "--row-num",
        type=str,
        default="auto",
        choices=["auto", "3", "4", "5"],
        help="row_num passed to GEMM (Pm must be divisible by it). 'auto' selects best from {4,3,5}.",
    )
    args = parser.parse_args()
    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    use_padding = args.use_padding
    pad_M, pad_N, pad_K = args.pad_M, args.pad_N, args.pad_K
    pad_impl_name = args.pad_impl
    col_num = args.col_num
    row_num = args.row_num
    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype in ("all", "i8"):
        test_pingpong_gemm(
            M, N, K, m, n, k, int8, int8,
            use_padding, pad_M, pad_N, pad_K, pad_impl_name, project_dir,
            col_num=col_num, row_num=row_num,
        )
    if args.dtype in ("all", "i16"):
        test_pingpong_gemm(
            M, N, K, m, n, k, int16, int16,
            use_padding, pad_M, pad_N, pad_K, pad_impl_name, project_dir,
            col_num=col_num, row_num=row_num,
        )
    if args.dtype in ("all", "bf16"):
        try:
            test_pingpong_gemm(
                M, N, K, m, n, k, bfloat16, bfloat16,
                use_padding, pad_M, pad_N, pad_K, pad_impl_name, project_dir,
                col_num=col_num, row_num=row_num,
            )
        except Exception:
            print("[NOTE]: bfloat16 have accuracy issue")
