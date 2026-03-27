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
        profile = True,
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
    output = rms_norm(input_tensor, weight)

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    weight_np = np.asarray(weight.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    # CPU execution time
    with torch.no_grad():
        start = time.perf_counter()
        end = time.perf_counter()
    cpu_time_us = (end - start) * 1000000

    if is_available():
        mod = df.build(top, target="aie", profile=True)
        output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)
        mod(input_np, weight_np, output_allo)
        print(f"CPU execution time: {cpu_time_us:.2f} us")

        np.testing.assert_allclose(
            output_allo.astype(np.float32),
            output.float().detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_rms_norm()