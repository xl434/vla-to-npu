# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn.functional as F
from typing import Annotated
from allo.ir.types import float32, int32, bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
import time

KERNEL_LIB_PATH = "../cc/"

S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ly_1 = [S(0)]

# -----------------------------
# PyTorch reference (UNMASKED)
# -----------------------------
def softmax_torch(
    attention_score_tile: Annotated[torch.Tensor, "shape: (64, 64*HEAD_TILE), dtype: float32"],
    SEQ_TILED, HEAD_TILE, SEQ, 
) -> Annotated[torch.Tensor, "shape: (64, 64*HEAD_TILE), dtype: float32"]:
        
    x = attention_score_tile.view(SEQ_TILED, HEAD_TILE, SEQ)
    y = F.softmax(x.float(), dim=-1)
    return y

# -----------------------------
# Allo test harness (UNMASKED)
# -----------------------------
def _test_softmax_float_64_64():
    SEQ_TILED = 64
    SEQ = 64
    HEAD_TILE = 1

    softmax_kernel = ExternalModule(
        top="softmax_float32",     
        impl_path=KERNEL_LIB_PATH + "v1_softmax_float.cc",
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32
    Ty_1 = int32

    @df.region()
    def top(Input: Ty[SEQ_TILED, SEQ * HEAD_TILE], Output: Ty[SEQ_TILED, SEQ * HEAD_TILE]):
        @df.kernel(mapping=[2, HEAD_TILE], args=[Input, Output])
        def core(
            local_Input: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
            local_Output: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
        ):
            softmax_kernel(local_Input, local_Output)

    # Random input
    torch.manual_seed(0)
    input_tensor = torch.randn(SEQ_TILED, SEQ * HEAD_TILE, dtype=torch.float32)

    ref = softmax_torch(input_tensor, SEQ_TILED, HEAD_TILE, SEQ)  # [64, H*64]

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np.float32)

    softmax_mod(input_tensor.cpu().numpy(), output_allo)

    # Compare
    ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy()
    np.testing.assert_allclose(output_allo, ref_flat, rtol=1e-2, atol=0.0)

    warmup=20
    iters=100
    label="cpu softmax"
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for _ in range(warmup):
            ref = softmax_torch(input_tensor, SEQ_TILED, HEAD_TILE, SEQ)
        t0 = time.perf_counter()
        for _ in range(iters):
            ref = softmax_torch(input_tensor, SEQ_TILED, HEAD_TILE, SEQ)
        t1 = time.perf_counter()
    avg_us = (t1 - t0) / iters * 1e6  # convert to microseconds
    print(f"{label} avg over {iters} iters (after {warmup} warmup): {avg_us:.1f} µs")
    print("PASS! Unmasked softmax matches PyTorch reference within tolerance.")

if __name__ == "__main__":
    _test_softmax_float_64_64()