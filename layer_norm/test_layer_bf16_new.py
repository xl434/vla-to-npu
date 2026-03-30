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

def layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: shape [..., dim] - input tensor
    weight: shape [dim] - scale parameter γ
    eps: small constant for numerical stability
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * weight


def test_layer_norm():
    norm = ExternalModule(
        top="layer_norm",
        impl_path="../cc/bf16_new/layer_norm_bf16.cc",
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

    # CPU execution time
    with torch.no_grad():
        start = time.perf_counter()
        output_ref = layernorm(input_tensor, weight)
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    weight_np = np.asarray(weight.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    if is_available():
        mod = df.build(top, target="aie", profile=True)
        mod(input_np, weight_np, output_allo)
        print(f"CPU execution time: {cpu_time_us:.2f} us")
        np.testing.assert_allclose(
            output_allo.astype(np.float32),
            output_ref.float().detach().cpu().numpy(),
            rtol=1e-2, atol=1e-3
        )
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def report_mismatches(actual, expected, rtol=1e-2, atol=1e-3, max_print=20):
    """
    Print mismatch statistics and a few mismatch examples.
    """
    actual_f32 = actual.astype(ml_dtypes.bfloat16)
    expected_f32 = expected.astype(ml_dtypes.bfloat16)

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
            print(
                f"  index={idx_tuple}, "
                f"actual={actual_f32[idx_tuple]:.6f}, "
                f"expected={expected_f32[idx_tuple]:.6f}, "
                f"abs_diff={abs_diff[idx_tuple]:.6f}, "
                f"tol={tol[idx_tuple]:.6f}"
            )

if __name__ == "__main__":
    test_layer_norm()