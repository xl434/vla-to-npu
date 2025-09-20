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

Ly = Layout("S1S0")
Ly_1 = Layout("S1")

# Softmax dimensions (same as masked test)
SEQ_TILED = 64
SEQ = 64
HEAD_TILE = 1

# -----------------------------
# PyTorch reference (UNMASKED)
# -----------------------------
def softmax_tiled_(
    attention_score_tile: Annotated[torch.Tensor, "shape: (64, 64*HEAD_TILE), dtype: float32"],
    tile_row_start: Annotated[int, "shape: (1), dtype: int32"],  # kept for parity; unused
) -> Annotated[torch.Tensor, "shape: (64, 64*HEAD_TILE), dtype: float32"]:

    x = attention_score_tile.view(SEQ_TILED, HEAD_TILE, SEQ)
    y = F.softmax(x.float(), dim=-1)
    return y

# -----------------------------
# Allo test harness (UNMASKED)
# -----------------------------
def _test_softmax_float():
    softmax_kernel = ExternalModule(
        top="softmax_float32",     
        impl_path=KERNEL_LIB_PATH + "softmax.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Ty_1 = int32

    @df.region()
    def top():
        @df.kernel(mapping=[2, HEAD_TILE])  # same mapping as masked test
        def core(
            Input: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
            TileRowStart: Ty_1[2] @ Ly_1,   # kept for ABI parity (ignored by kernel)
            Output: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
        ):
            softmax_kernel(Input, TileRowStart, Output)

    # Random input
    torch.manual_seed(0)
    input_tensor = torch.randn(SEQ_TILED, SEQ * HEAD_TILE, dtype=torch.float32)

    # Reference (unmasked)
    tile_row_start = 0
    tile_row_start_tensor = torch.tensor([tile_row_start], dtype=torch.int32)
    ref = softmax_tiled_(input_tensor, tile_row_start_tensor)  # [64, H*64]

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np.float32)

    softmax_mod(input_tensor.cpu().numpy(), np.array([0, 32], dtype=np.int32), output_allo)

    # Compare
    ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy()
    np.testing.assert_allclose(output_allo, ref_flat, rtol=1e-2, atol=0.0)

    warmup=20
    iters=100
    label="cpu softmax"
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for _ in range(warmup):
            ref = softmax_tiled_(input_tensor, tile_row_start_tensor)
        t0 = time.perf_counter()
        for _ in range(iters):
            ref = softmax_tiled_(input_tensor, tile_row_start_tensor)
        t1 = time.perf_counter()
    avg_us = (t1 - t0) / iters * 1e6  # convert to microseconds
    print(f"{label} avg over {iters} iters (after {warmup} warmup): {avg_us:.1f} µs")
    print("PASS! Unmasked softmax matches PyTorch reference within tolerance.")

def _test_softmax_bf16():
    softmax_kernel = ExternalModule(
        top="softmax_bf16",     
        impl_path=KERNEL_LIB_PATH + "softmax_old_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    Ty_1 = int32

    @df.region()
    def top():
        @df.kernel(mapping=[2, HEAD_TILE])  # same mapping as masked test
        def core(
            Input: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
            TileRowStart: Ty_1[2] @ Ly_1,   # kept for ABI parity (ignored by kernel)
            Output: Ty[SEQ_TILED, SEQ * HEAD_TILE] @ Ly,
        ):
            softmax_kernel(Input, TileRowStart, Output)

    # Random input
    torch.manual_seed(0)
    input_tensor = np.random.randn(SEQ_TILED, SEQ * HEAD_TILE).astype(np_bfloat16)

    # Reference (unmasked)
    tile_row_start = 0
    tile_row_start_tensor = torch.tensor([tile_row_start], dtype=torch.int32)
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    ref = softmax_tiled_(torch_in, tile_row_start_tensor)  # [64, H*64]

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np_bfloat16)

    softmax_mod(input_tensor, np.array([0, 32], dtype=np.int32), output_allo)

    # Compare
    ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy().astype(np.float32)
    np.testing.assert_allclose(output_allo.astype(np.float32), ref_flat, atol=1e-1, rtol=1e-2,)

    warmup=20
    iters=100
    label="cpu softmax"
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for _ in range(warmup):
            ref = softmax_tiled_(torch_in, tile_row_start_tensor)
        t0 = time.perf_counter()
        for _ in range(iters):
            ref = softmax_tiled_(torch_in, tile_row_start_tensor)
        t1 = time.perf_counter()
    avg_us = (t1 - t0) / iters * 1e6  # convert to microseconds
    print(f"{label} avg over {iters} iters (after {warmup} warmup): {avg_us:.1f} µs")
    print("PASS! Unmasked softmax matches PyTorch reference within tolerance.")

if __name__ == "__main__":
    # _test_softmax_float()
    _test_softmax_bf16()