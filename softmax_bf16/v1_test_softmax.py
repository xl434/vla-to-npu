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

KERNEL_LIB_PATH = "../cc/bf16_old/"

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
def _test_softmax_bf16_64_64():
    softmax_kernel = ExternalModule(
        top="softmax_bf16_64_64",     
        impl_path=KERNEL_LIB_PATH + "v1_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    SEQ_TILED = 64
    SEQ = 64
    HEAD_TILE = 1

    Ty = bfloat16

    @df.region()
    def top(Input: Ty[SEQ_TILED, SEQ * HEAD_TILE], Output: Ty[SEQ_TILED, SEQ * HEAD_TILE]):
        @df.kernel(mapping=[HEAD_TILE, HEAD_TILE], args=[Input, Output])
        def core(
            local_Input: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
            local_Output: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
        ):
            softmax_kernel(local_Input, local_Output)

    # Random input
    torch.manual_seed(0)
    input_tensor = np.random.randn(SEQ_TILED, SEQ * HEAD_TILE).astype(np_bfloat16)

    # Reference (unmasked)
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    ref = softmax_torch(torch_in, SEQ_TILED, HEAD_TILE, SEQ)  # [64, H*64]

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np_bfloat16)

    softmax_mod(input_tensor, output_allo)

    # Compare
    ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy().astype(np.float32)
    np.testing.assert_allclose(output_allo.astype(np.float32), ref_flat, atol=1e-1, rtol=1e-2,)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.copy()                                               # input data prep
        torch_tmp = torch.from_numpy(input_numpy_cpu.astype(np.float32)).to(torch.bfloat16) # input data prep
        ref = softmax_torch(torch_tmp, SEQ_TILED, HEAD_TILE, SEQ)                           # compute
        ref_numpy = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy().astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000
    print(f"CPU execution time: {cpu_time_us:.2f} us")
    print("PASS! softmax matches PyTorch reference within tolerance.")

def _test_softmax_bf16_32_64():
    softmax_kernel = ExternalModule(
        top="softmax_bf16_32_64",     
        impl_path=KERNEL_LIB_PATH + "v1_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    SEQ_TILED = 64
    SEQ = 64
    HEAD_TILE = 1

    Ty = bfloat16

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
    input_tensor = np.random.randn(SEQ_TILED, SEQ * HEAD_TILE).astype(np_bfloat16)

    # Reference (unmasked)
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    ref = softmax_torch(torch_in, SEQ_TILED, HEAD_TILE, SEQ)  # [64, H*64]

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np_bfloat16)

    softmax_mod(input_tensor, output_allo)

    # Compare
    ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy().astype(np.float32)
    np.testing.assert_allclose(output_allo.astype(np.float32), ref_flat, atol=1e-1, rtol=1e-2,)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.copy()                                               # input data prep
        torch_tmp = torch.from_numpy(input_numpy_cpu.astype(np.float32)).to(torch.bfloat16) # input data prep
        ref = softmax_torch(torch_tmp, SEQ_TILED, HEAD_TILE, SEQ)                           # compute
        ref_numpy = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy().astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000
    print(f"CPU execution time: {cpu_time_us:.2f} us")
    print("PASS! softmax matches PyTorch reference within tolerance.")

if __name__ == "__main__":
    # _test_softmax_float_64_64()
    _test_softmax_bf16_32_64()