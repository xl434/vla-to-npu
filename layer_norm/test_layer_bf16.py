# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import torch.nn as nn
import allo
import allo.dataflow as df
from allo.ir.types import bfloat16, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

# ----------------------------
# Config
# ----------------------------
S = Layout.Shard
R = Layout.Replicate
Ty = bfloat16
SEQ = 64
EMBD = 768

KERNEL_LIB_PATH = "../cc/bf16_old/"

norm = ExternalModule(
    top="layer_norm_bf16",
    impl_path=KERNEL_LIB_PATH + "layer_norm_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0

norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def layer_norm_kernel(
    input_x: Ty[NORM_SEQ_TILE, EMBD],
    weight: Ty[EMBD],
    bias: Ty[EMBD],
    output_x: Ty[NORM_SEQ_TILE, EMBD],
):
    pipe: Stream[Ty[NORM_TILE, EMBD], 1][NORM_P0]

    @df.kernel(mapping=[NORM_P0], args=[input_x, weight])
    def norm_no_bias(
        local_input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_weight: Ty[EMBD] @ norm_arg_layout,
    ):
        pi = df.get_pid()
        tmp: Ty[NORM_TILE, EMBD] = 0
        norm(local_input_x, local_weight, tmp)
        pipe[pi].put(tmp)

    @df.kernel(mapping=[NORM_P0], args=[bias, output_x])
    def norm_add_bias(
        local_bias: Ty[EMBD] @ norm_arg_layout,
        local_output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        pi = df.get_pid()
        data = pipe[pi].get()
        local_output_x[:, :] = allo.add(data, local_bias)


def test_layer_norm_bf16():
    layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm_bf16.prj")

    # Create reference model
    ln = nn.LayerNorm(EMBD, elementwise_affine=True)
    with torch.no_grad():
        weight_np = ln.weight.numpy().astype(np_bfloat16)
        bias_np = ln.bias.numpy().astype(np_bfloat16)

    # Random input
    torch.manual_seed(0)
    input_np = np.random.randn(SEQ, EMBD).astype(np_bfloat16)

    # PyTorch reference in bf16
    input_torch = torch.from_numpy(input_np.astype(np.float32)).to(torch.bfloat16)
    weight_torch = torch.from_numpy(weight_np.astype(np.float32)).to(torch.bfloat16)
    bias_torch = torch.from_numpy(bias_np.astype(np.float32)).to(torch.bfloat16)
    ref_ln = nn.LayerNorm(EMBD, elementwise_affine=True)
    with torch.no_grad():
        ref_ln.weight.copy_(weight_torch.float())
        ref_ln.bias.copy_(bias_torch.float())
    ref_out = ref_ln(input_torch.float()).detach().to(torch.bfloat16).float().numpy()

    # Allo output
    out = np.empty((SEQ, EMBD), dtype=np_bfloat16)
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_in = input_np[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        tile_out = out[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        layer_norm_mod(tile_in, weight_np, bias_np, tile_out)

    np.testing.assert_allclose(
        out.astype(np.float32), ref_out, atol=1e-1, rtol=1e-2
    )
    print("PASS! LayerNorm bf16 matches PyTorch reference within tolerance.")


if __name__ == "__main__":
    test_layer_norm_bf16()
