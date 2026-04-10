# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import torch.nn as nn
import ml_dtypes
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_LIB_PATH = "../cc/bf16_new/"

S  = Layout.Shard
R  = Layout.Replicate
Ly = [S(0), S(1)]
Ty = bfloat16

feature_tile = 768
seq_tile     = 4

RTOL = 1e-2
ATOL = 1e-3

def _to_bf16_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Bit-identical conversion: bfloat16 torch tensor → ml_dtypes.bfloat16 numpy
    array via int16 view.  Avoids the lossy float32 round-trip that
    tensor.cpu().numpy() performs on bfloat16 tensors.
    """
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16)

def _mismatch_stats(actual: np.ndarray, expected: np.ndarray,
                    rtol: float, atol: float):
    diff       = np.abs(actual - expected)
    tol        = atol + rtol * np.abs(expected)
    mask       = diff > tol
    total      = mask.size
    mismatches = int(np.count_nonzero(mask))
    return 100.0 * mismatches / total if total else 0.0, mismatches, total

def _print_mismatch_debug(output_allo: np.ndarray, ref_numpy: np.ndarray,
                          input_tensor: torch.Tensor,
                          rtol: float, atol: float, label: str = ""):
    pct, mism, total = _mismatch_stats(output_allo, ref_numpy, rtol, atol)
    diff    = np.abs(output_allo - ref_numpy)
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    r, c    = max_idx
    tag     = f"[{label}] " if label else ""
    print(f"{tag}GeLU mismatch detected.")
    print(f"  Mismatch rate : {pct:.4f}%  ({mism}/{total})  "
          f"(rtol={rtol}, atol={atol})")
    print(f"  Max abs diff  = {diff[max_idx]:.6e}  at index {max_idx}")
    print(f"  Input         = {input_tensor[r, c].item():.6f}")
    print(f"  Allo output   = {output_allo[r, c]:.6f}")
    print(f"  Ref  output   = {ref_numpy[r, c]:.6f}")

def _test_gelu_single_tile():
    gelu = ExternalModule(
        top="gelu",
        impl_path=KERNEL_LIB_PATH + "gelu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(input_x:  Ty[seq_tile, feature_tile],
            output_x: Ty[seq_tile, feature_tile]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(
            local_input_x:  Ty[seq_tile, feature_tile] @ Ly,
            local_output_x: Ty[seq_tile, feature_tile] @ Ly,
        ):
            gelu(local_input_x, local_output_x)

    torch.manual_seed(0)
    input_tensor = torch.randn(seq_tile, feature_tile, dtype=torch.bfloat16)

    # Reference: run nn.GELU in bfloat16 to match kernel precision
    gelu_model = nn.GELU()

    # CPU execution time
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start
    cpu_time_us = (total_time / 1000) * 1000000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (single_tile).")
        return

    mod = df.build(
        top,
        target="aie",
        profile=True,
        trace=[("core", (0, 0))],
        trace_size=65536,
    )

    # Bit-identical bfloat16 input; output buffer in ml_dtypes.bfloat16
    input_numpy = _to_bf16_numpy(input_tensor)
    output_allo = np.zeros((seq_tile, feature_tile), dtype=ml_dtypes.bfloat16)

    mod(input_numpy, output_allo)

    output_allo_f32 = output_allo.astype(np.float32)

    print(f"CPU execution time: {cpu_time_us:.2f} us")

    try:
        np.testing.assert_allclose(output_allo_f32, ref_numpy, rtol=RTOL, atol=ATOL)
        print(f"PASSED gelu single-tile!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        _print_mismatch_debug(output_allo_f32, ref_numpy, input_tensor,
                              RTOL, ATOL, label="single_tile")

def _test_gelu_tiling():
    gelu = ExternalModule(
        top="gelu",
        impl_path=KERNEL_LIB_PATH + "gelu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 4

    feature_dim = P0 * feature_tile
    seq         = P1 * seq_tile

    @df.region()
    def top(input_x: Ty[seq, feature_dim],
            output_x: Ty[seq, feature_dim]):
        @df.kernel(mapping=[P1, P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[seq, feature_dim] @ Ly,
            local_output_x: Ty[seq, feature_dim] @ Ly,
        ):
            gelu(local_input_x, local_output_x)

    torch.manual_seed(1)
    input_tensor = torch.randn(seq, feature_dim, dtype=torch.bfloat16)

    # CPU execution time
    gelu_model = nn.GELU()
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start
    cpu_time_us = (total_time / 1000) * 1000000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (tiling).")
        return

    mod = df.build(
        top,
        target="aie",
        profile=True,
        trace=[("core", (0, 0))],
        trace_size=65536,
    )

    input_numpy = _to_bf16_numpy(input_tensor)
    output_allo = np.zeros((seq, feature_dim), dtype=ml_dtypes.bfloat16)

    mod(input_numpy, output_allo)

    output_allo_f32 = output_allo.astype(np.float32)

    print(f"CPU execution time: {cpu_time_us:.2f} us")

    try:
        np.testing.assert_allclose(output_allo_f32, ref_numpy, rtol=RTOL, atol=ATOL)
        print(f"PASSED gelu tiling!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        _print_mismatch_debug(output_allo_f32, ref_numpy, input_tensor,
                              RTOL, ATOL, label="tiling")
        
def _test_gelu_tiling_small():
    gelu = ExternalModule(
        top="gelu_small",
        impl_path=KERNEL_LIB_PATH + "gelu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 4
    
    feature_tile = 192
    seq_tile     = 1

    feature_dim = P0 * feature_tile
    seq         = P1 * seq_tile

    @df.region()
    def top(input_x: Ty[seq, feature_dim],
            output_x: Ty[seq, feature_dim]):
        @df.kernel(mapping=[P1, P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[seq, feature_dim] @ Ly,
            local_output_x: Ty[seq, feature_dim] @ Ly,
        ):
            gelu(local_input_x, local_output_x)

    torch.manual_seed(1)
    input_tensor = torch.randn(seq, feature_dim, dtype=torch.bfloat16)

    # CPU execution time
    gelu_model = nn.GELU()
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            ref_out = gelu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
            ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start
    cpu_time_us = (total_time / 1000) * 1000000

    if "MLIR_AIE_INSTALL_DIR" not in os.environ:
        print("MLIR_AIE_INSTALL_DIR unset — skipping AIE run (tiling).")
        return

    mod = df.build(
        top,
        target="aie",
        profile=True,
        trace=[("core", (0, 0))],
        trace_size=65536,
    )

    input_numpy = _to_bf16_numpy(input_tensor)
    output_allo = np.zeros((seq, feature_dim), dtype=ml_dtypes.bfloat16)

    mod(input_numpy, output_allo)

    output_allo_f32 = output_allo.astype(np.float32)

    print(f"CPU execution time: {cpu_time_us:.2f} us")

    try:
        np.testing.assert_allclose(output_allo_f32, ref_numpy, rtol=RTOL, atol=ATOL)
        print(f"PASSED gelu tiling!  (rtol={RTOL}, atol={ATOL})")
    except AssertionError:
        _print_mismatch_debug(output_allo_f32, ref_numpy, input_tensor,
                              RTOL, ATOL, label="tiling")


if __name__ == "__main__":
    # _test_gelu_single_tile()
    # _test_gelu_tiling()
    _test_gelu_tiling_small()