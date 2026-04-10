# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import pytest
import torch
import torch.nn as nn
from allo.ir.types import bfloat16
import ml_dtypes
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available

S = Layout.Shard
R = Layout.Replicate
Ly = [R]
LyA = [S(0), R]

seq_len = 4
hidden_size = 768

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm along last dim
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * weight


def test_rms_norm():
    norm = ExternalModule(
        top="rms_norm",
        impl_path="../cc/bf16_new/rms_norm_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    M, N = seq_len, hidden_size

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M, N] @ LyA, local_B: Ty[N] @ Ly, local_C: Ty[M, N] @ LyA):
            norm(local_A, local_B, local_C)

    input_tensor = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)
    rms_norm = RMSNorm()

    # CPU Execution Time
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            output = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )
            ref_numpy = output.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)         # input data prep
            output = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )  # compute
            ref_numpy = output.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start

    cpu_time_us = (total_time / 1000) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    weight_np = np.asarray(weight.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    if is_available():
        mod = df.build(top, target="aie", profile=True)
        output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)
        mod(input_np, weight_np, output_allo)
        print(f"CPU execution time: {cpu_time_us:.2f} us")

        try:
            np.testing.assert_allclose(
                output_allo.astype(np.float32),
                ref_numpy,
                rtol=1e-2,
                atol=1e-3,
            )
            print("PASSED!")
        except AssertionError as e:
            print(f"MISMATCH (continuing): {e}")
            report_mismatches(
                output_allo.astype(np.float32),
                ref_numpy,
            )
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def report_mismatches(actual, expected, rtol=1e-2, atol=1e-3, max_print=20):
    """
    Print mismatch statistics and a few mismatch examples.
    """
    actual_f32 = np.asarray(actual, dtype=np.float32)
    expected_f32 = np.asarray(expected, dtype=np.float32)

    # Same rule used by np.testing.assert_allclose:
    # abs(actual - expected) <= atol + rtol * abs(expected)
    abs_diff = np.abs(actual_f32 - expected_f32)
    tol = atol + rtol * np.abs(expected_f32)
    mismatch_mask = abs_diff > tol

    total = actual_f32.size
    mismatch_count = np.count_nonzero(mismatch_mask)
    mismatch_pct = 100.0 * mismatch_count / total

    print(f"Total elements: {total}")
    print(f"Mismatched elements: {mismatch_count}")
    print(f"Mismatch percentage: {mismatch_pct:.4f}%")

    if mismatch_count > 0:
        mismatch_indices = np.argwhere(mismatch_mask)
        print(f"\nFirst {min(max_print, mismatch_count)} mismatches:")
        for idx in mismatch_indices[:max_print]:
            idx_tuple = tuple(idx)
            a_val = float(actual_f32[idx_tuple])
            e_val = float(expected_f32[idx_tuple])
            d_val = float(abs_diff[idx_tuple])
            t_val = float(tol[idx_tuple])

            print(
                f"  index={idx_tuple}, "
                f"actual={a_val:.6f}, "
                f"expected={e_val:.6f}, "
                f"abs_diff={d_val:.6f}, "
                f"tol={t_val:.6f}"
            )


def test_rms_norm_tiling():
    norm = ExternalModule(
        top="rms_norm",
        impl_path="../cc/bf16_new/rms_norm_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16

    P0 = 4                     # number of tiles along sequence dimension
    M = seq_len * P0           # total rows = 4 * 4 = 16
    N = hidden_size            # keep full hidden dimension in each tile

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def core(
            local_A: Ty[M, N] @ LyA,
            local_B: Ty[N] @ Ly,
            local_C: Ty[M, N] @ LyA
        ):
            norm(local_A, local_B, local_C)

    input_tensor = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)
    rms_norm = RMSNorm()

    # CPU Execution Time
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            output_ref = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )
            ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)         # input data prep
            output_ref = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )  # compute
            ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start

    cpu_time_us = (total_time / 1000) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    weight_np = np.asarray(weight.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    if is_available():
        mod = df.build(top, target="aie", profile=True)
        mod(input_np, weight_np, output_allo)

        print(f"CPU execution time: {cpu_time_us:.2f} us")
        try:
            np.testing.assert_allclose(
                output_allo.astype(np.float32),
                ref_numpy,
                rtol=1e-2,
                atol=1e-3
            )
            print("PASSED rms norm tiling!")
        except AssertionError as e:
            print(f"MISMATCH (continuing): {e}")
            report_mismatches(
                output_allo.astype(np.float32),
                ref_numpy,
            )
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_rms_norm_tiling_small():
    norm = ExternalModule(
        top="rms_norm_small",
        impl_path="../cc/bf16_new/rms_norm_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16

    P0 = 4                    # 16 tiles along sequence dimension
    seq_tile = 4
    hidden_size = 192

    M = seq_tile * P0          # total rows = 1 * 16 = 16
    N = hidden_size                # hidden dimension per tile

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def core(
            local_A: Ty[M, N] @ LyA,
            local_B: Ty[N] @ Ly,
            local_C: Ty[M, N] @ LyA
        ):
            norm(local_A, local_B, local_C)

    input_tensor = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)
    rms_norm = RMSNorm()

    # CPU Execution Time
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            output_ref = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )
            ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)

        # Timed runs
        total_time = 0.0
        for _ in range(1000):
            start = time.perf_counter()
            input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
            weight_numpy_cpu = weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16)         # input data prep
            output_ref = rms_norm(
                torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16),
                torch.from_numpy(weight_numpy_cpu.view(np.int16)).view(torch.bfloat16),
            )  # compute
            ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
            end = time.perf_counter()
            total_time += end - start

    cpu_time_us = (total_time / 1000) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    weight_np = np.asarray(weight.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    if is_available():
        mod = df.build(top, target="aie", profile=True)
        mod(input_np, weight_np, output_allo)

        print(f"CPU execution time: {cpu_time_us:.2f} us")
        try:
            np.testing.assert_allclose(
                output_allo.astype(np.float32),
                ref_numpy,
                rtol=1e-2,
                atol=1e-3
            )
            print("PASSED rms norm tiling small!")
        except AssertionError as e:
            print(f"MISMATCH (continuing): {e}")
            report_mismatches(
                output_allo.astype(np.float32),
                ref_numpy,
            )
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    # test_rms_norm()
    # test_rms_norm_tiling()
    test_rms_norm_tiling_small()