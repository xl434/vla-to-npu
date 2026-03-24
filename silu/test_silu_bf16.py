# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
from allo.ir.types import bfloat16
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

KERNEL_LIB_PATH = "../cc/bf16_old/"

S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ty = bfloat16

feature_tile = 768
seq_tile = 4


def _test_silu_bf16_single_tile():
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

    silu_model = nn.SiLU()

    # Generate bf16 input
    torch.manual_seed(0)
    input_np = np.random.randn(seq_tile, feature_tile).astype(np_bfloat16)
    input_torch = torch.from_numpy(input_np.astype(np.float32)).to(torch.bfloat16)
    ref_output = silu_model(input_torch.float()).detach().to(torch.bfloat16).float().numpy()

    mod = df.build(top, target="aie")
    output_allo = np.zeros((seq_tile, feature_tile), dtype=np_bfloat16)
    mod(input_np, output_allo)
    np.testing.assert_allclose(
        output_allo.astype(np.float32), ref_output, atol=1e-1, rtol=1e-2
    )
    print("PASS! SiLU bf16 single tile matches PyTorch reference.")


def _test_silu_bf16_tiling():
    silu = ExternalModule(
        top="silu_bf16",
        impl_path=KERNEL_LIB_PATH + "silu_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 4
    feature_dim = P0 * feature_tile
    seq = P1 * seq_tile

    @df.region()
    def top(input_x: Ty[seq, feature_dim], output_x: Ty[seq, feature_dim]):
        @df.kernel(mapping=[P1, P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[seq, feature_dim] @ Ly,
            local_output_x: Ty[seq, feature_dim] @ Ly,
        ):
            silu(local_input_x, local_output_x)

    silu_model = nn.SiLU()

    # Generate bf16 input
    torch.manual_seed(0)
    input_np = np.random.randn(seq, feature_dim).astype(np_bfloat16)
    input_torch = torch.from_numpy(input_np.astype(np.float32)).to(torch.bfloat16)
    ref_output = silu_model(input_torch.float()).detach().to(torch.bfloat16).float().numpy()

    mod = df.build(top, target="aie")
    output_allo = np.zeros((seq, feature_dim), dtype=np_bfloat16)
    mod(input_np, output_allo)
    np.testing.assert_allclose(
        output_allo.astype(np.float32), ref_output, atol=1e-1, rtol=1e-2
    )
    print("PASS! SiLU bf16 tiling matches PyTorch reference.")


if __name__ == "__main__":
    _test_silu_bf16_single_tile()
    _test_silu_bf16_tiling()
