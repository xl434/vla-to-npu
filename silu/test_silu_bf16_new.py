# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import ml_dtypes
import torch.nn as nn
import numpy as np
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import bfloat16
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ty = bfloat16

feature_tile = 768
seq_tile = 4

KERNEL_LIB_PATH = "../cc/bf16_new/"

def _test_silu_bf16_new_single_tile():
    silu = ExternalModule(
        top="silu_bf16",
        impl_path=KERNEL_LIB_PATH + "silu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(input_x: Ty[seq_tile, feature_tile], output_x: Ty[seq_tile, feature_tile]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(
            local_input_x: Ty[seq_tile, feature_tile] @ Ly,
            local_output_x: Ty[seq_tile, feature_tile] @ Ly,
        ):
            silu(local_input_x, local_output_x)

    # Reference PyTorch SiLU
    silu_model = nn.SiLU().cpu()
    input_tensor = torch.tensor(
        [[6.5] * feature_tile for _ in range(seq_tile)],
        dtype=torch.bfloat16
    )
    
    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        ref_out = silu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
        ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1000000

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            trace=[
                ("core", (0, 0)),
            ],
            trace_size=4096,
        )
        output_allo = np.zeros((seq_tile, feature_tile), dtype=ml_dtypes.bfloat16)
        input_numpy = input_tensor.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        mod(input_numpy, output_allo)

        diff = np.abs(output_allo - ref_numpy)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        r, c = max_idx
        print(f"Input        = {input_tensor[r, c].item():.6f}")
        print(f"CPU execution time: {cpu_time_us:.2f} us")
        np.testing.assert_allclose(output_allo, ref_numpy, rtol=1e-2, atol=1e-3)
        print("PASSED SiLU bf16_new!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

def _test_silu_bf16_new_tiling():
    silu = ExternalModule(
        top="silu_bf16",
        impl_path=KERNEL_LIB_PATH + "silu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 4
    seq = seq_tile * P0
    feature_dim = feature_tile * P1

    @df.region()
    def top(input_x: Ty[seq, feature_dim], output_x: Ty[seq, feature_dim]):
        @df.kernel(mapping=[P0, P1], args=[input_x, output_x])
        def core(
            local_input_x: Ty[seq, feature_dim] @ Ly,
            local_output_x: Ty[seq, feature_dim] @ Ly,
        ):
            silu(local_input_x, local_output_x)

    # Reference PyTorch SiLU
    silu_model = nn.SiLU().cpu()
    input_tensor = torch.randn(seq, feature_dim, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output = silu_model(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
        ref_numpy = output.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            trace=[
                ("core", (0, 0)),
            ],
            trace_size=4096,
        )
        output_allo = np.zeros((seq, feature_dim), dtype=ml_dtypes.bfloat16)
        input_numpy = input_tensor.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        mod(input_numpy, output_allo)
        print(f"CPU execution time: {cpu_time_us:.2f} us")
        np.testing.assert_allclose(output_allo, ref_numpy, rtol=1e-2, atol=1e-3)
        print("PASSED SiLU bf16_new tiling!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_silu_bf16_new_single_tile()
    _test_silu_bf16_new_tiling()