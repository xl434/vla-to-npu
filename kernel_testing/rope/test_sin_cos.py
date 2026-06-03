# Test: float32 sin_cos kernel (sin_float32, cos_float32, sin_cos_float32) tile [32][64]

import os
import time
import numpy as np
import torch
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ty = float32

seq_tile     = 32
feature_tile = 64

RTOL = 1e-2
ATOL = 1e-3

KERNEL_PATH = "../../cc/float/sin_cos.cc"


def _mismatch_stats(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float):
    diff = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    mask = diff > tol
    total = mask.size
    mismatches = int(np.count_nonzero(mask))
    return 100.0 * mismatches / total if total else 0.0, mismatches, total


def _print_mismatch(label, output_allo, ref_numpy, input_tensor, rtol, atol):
    pct, mism, total = _mismatch_stats(output_allo, ref_numpy, rtol, atol)
    diff    = np.abs(output_allo - ref_numpy)
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    r, c    = max_idx
    print(f"{label} mismatch detected.")
    print(f"  Mismatch rate : {pct:.4f}%  ({mism}/{total})  (rtol={rtol}, atol={atol})")
    print(f"  Max abs diff  = {diff[max_idx]:.6e}  at index {max_idx}")
    print(f"  Input         = {input_tensor[r, c].item():.6f}")
    print(f"  Allo output   = {output_allo[r, c]:.6f}")
    print(f"  Ref  output   = {ref_numpy[r, c]:.6f}")


def _test_sin_float32():
    sine = ExternalModule(
        top="sin_float32",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(input_x: Ty[seq_tile, feature_tile], output_x: Ty[seq_tile, feature_tile]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(
            local_input_x:  Ty[seq_tile, feature_tile] @ Ly,
            local_output_x: Ty[seq_tile, feature_tile] @ Ly,
        ):
            sine(local_input_x, local_output_x)

    torch.manual_seed(0)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.float32) * 40.0) - 20.0

    # CPU execution time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.cpu().numpy()                     # input data prep
        ref_out = torch.sin(torch.from_numpy(input_numpy_cpu))          # compute
        ref_numpy = ref_out.cpu().numpy()                               # output retrieval
        end = time.perf_counter()
    cpu_time_us = (end - start) * 1_000_000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (sin_float32).")
        return

    mod = df.build(
        top,
        target="aie",
        profile=True,
        trace=[("core", (0, 0))],
        trace_size=65536,
    )
    output_allo = np.zeros((seq_tile, feature_tile), dtype=np.float32)
    mod(input_tensor.cpu().numpy(), output_allo)

    print(f"CPU execution time (sin): {cpu_time_us:.2f} us")
    try:
        np.testing.assert_allclose(output_allo, ref_numpy, rtol=RTOL, atol=ATOL)
        print(f"PASSED sin_float32!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        _print_mismatch("sin_float32", output_allo, ref_numpy, input_tensor, RTOL, ATOL)


def _test_cos_float32():
    cosine = ExternalModule(
        top="cos_float32",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(input_x: Ty[seq_tile, feature_tile], output_x: Ty[seq_tile, feature_tile]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(
            local_input_x:  Ty[seq_tile, feature_tile] @ Ly,
            local_output_x: Ty[seq_tile, feature_tile] @ Ly,
        ):
            cosine(local_input_x, local_output_x)

    torch.manual_seed(0)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.float32) * 40.0) - 20.0

    # CPU execution time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.cpu().numpy()                     # input data prep
        ref_out = torch.cos(torch.from_numpy(input_numpy_cpu))          # compute
        ref_numpy = ref_out.cpu().numpy()                               # output retrieval
        end = time.perf_counter()
    cpu_time_us = (end - start) * 1_000_000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (cos_float32).")
        return

    mod = df.build(
        top,
        target="aie",
        profile=True,
        trace=[("core", (0, 0))],
        trace_size=65536,
    )
    output_allo = np.zeros((seq_tile, feature_tile), dtype=np.float32)
    mod(input_tensor.cpu().numpy(), output_allo)

    print(f"CPU execution time (cos): {cpu_time_us:.2f} us")
    try:
        np.testing.assert_allclose(output_allo, ref_numpy, rtol=RTOL, atol=ATOL)
        print(f"PASSED cos_float32!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        _print_mismatch("cos_float32", output_allo, ref_numpy, input_tensor, RTOL, ATOL)


def _test_sin_cos_float32():
    sin_cos = ExternalModule(
        top="sin_cos_float32",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(
        input_x: Ty[seq_tile, feature_tile],
        out: Ty[2 * seq_tile, feature_tile],
    ):
        @df.kernel(mapping=[1, 1], args=[input_x, out])
        def core(
            local_input_x: Ty[seq_tile, feature_tile] @ Ly,
            local_out: Ty[2 * seq_tile, feature_tile] @ Ly,
        ):
            sin_cos(local_input_x, local_out)

    torch.manual_seed(0)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.float32) * 40.0) - 20.0

    with torch.no_grad():
        start = time.perf_counter()
        input_numpy = input_tensor.cpu().numpy()
        ref_sin_np = torch.sin(torch.from_numpy(input_numpy)).cpu().numpy()
        ref_cos_np = torch.cos(torch.from_numpy(input_numpy)).cpu().numpy()
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (sin_cos_float32).")
        return

    mod = df.build(top, target="aie")

    out_allo = np.zeros((2 * seq_tile, feature_tile), dtype=np.float32)
    mod(input_numpy, out_allo)

    sin_allo = out_allo[:seq_tile, :]
    cos_allo = out_allo[seq_tile:, :]

    print(f"CPU execution time (sin+cos): {cpu_time_us:.2f} us")

    sin_ok = cos_ok = True

    try:
        np.testing.assert_allclose(sin_allo, ref_sin_np, rtol=RTOL, atol=ATOL)
        print(f"PASSED sin path of sin_cos_float32!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        sin_ok = False
        _print_mismatch("sin_cos_float32 [sin path]", sin_allo, ref_sin_np, input_tensor, RTOL, ATOL)

    try:
        np.testing.assert_allclose(cos_allo, ref_cos_np, rtol=RTOL, atol=ATOL)
        print(f"PASSED cos path of sin_cos_float32!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        cos_ok = False
        _print_mismatch("sin_cos_float32 [cos path]", cos_allo, ref_cos_np, input_tensor, RTOL, ATOL)

    if sin_ok and cos_ok:
        print("PASSED sin_cos_float32 packed 2D output!")


if __name__ == "__main__":
    _test_sin_float32()
    _test_cos_float32()
    _test_sin_cos_float32()
