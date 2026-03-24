# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import allo
import allo.dataflow as df
from allo.ir.types import bfloat16, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

S = Layout.Shard
R = Layout.Replicate
Ty = bfloat16
SEQ = 64
EMBD = 768

KERNEL_LIB_PATH = "../cc/bf16_old/"

norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path=KERNEL_LIB_PATH + "rms_norm_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0

norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def rms_norm_kernel(
    input_x: Ty[NORM_SEQ_TILE, EMBD],
    weight: Ty[EMBD],
    output_x: Ty[NORM_SEQ_TILE, EMBD],
):
    @df.kernel(mapping=[NORM_P0], args=[input_x, weight, output_x])
    def core(
        local_input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_weight: Ty[EMBD] @ norm_arg_layout,
        local_output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        norm(local_input_x, local_weight, local_output_x)


def test_rms_norm_bf16():
    rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="rms_norm_bf16.prj", profile=True)

    # PyTorch reference
    rms = nn.RMSNorm(EMBD, elementwise_affine=True)
    with torch.no_grad():
        weight_np = rms.weight.numpy().astype(np_bfloat16)

    # Random input
    torch.manual_seed(0)
    input_np = np.random.randn(SEQ, EMBD).astype(np_bfloat16)

    # PyTorch bf16 reference
    input_torch = torch.from_numpy(input_np.astype(np.float32)).to(torch.bfloat16)
    weight_torch = torch.from_numpy(weight_np.astype(np.float32)).to(torch.bfloat16)
    ref_rms = nn.RMSNorm(EMBD, elementwise_affine=True)
    with torch.no_grad():
        ref_rms.weight.copy_(weight_torch.float())
    ref_out = ref_rms(input_torch.float()).detach().to(torch.bfloat16).float().numpy()

    # Allo output
    out = np.empty((SEQ, EMBD), dtype=np_bfloat16)
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_in = input_np[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        tile_out = out[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        rms_norm_mod(tile_in, weight_np, tile_out)

    np.testing.assert_allclose(
        out.astype(np.float32), ref_out, atol=1e-1, rtol=1e-2
    )
    print("PASS! RMSNorm bf16 matches PyTorch reference within tolerance.")


if __name__ == "__main__":
    test_rms_norm_bf16()
