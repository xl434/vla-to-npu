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

KERNEL_LIB_PATH = "../cc/float/"

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

    # CPU reference
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.cpu().numpy()                                        # input data prep
        ref = softmax_torch(torch.from_numpy(input_numpy_cpu), SEQ_TILED, HEAD_TILE, SEQ)  # compute
        ref_flat = ref.view(SEQ_TILED, HEAD_TILE * SEQ).detach().cpu().numpy()              # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    softmax_mod = df.build(
        top,
        target="aie",
        profile=True,
        warmup=20,
        num_iters=1000,  # execute multiple times for stable perf
    )

    input_np = input_tensor.cpu().numpy()
    num_runs = 1000
    npu_times = []
    for _ in range(num_runs):
        output_allo = np.zeros((SEQ_TILED, SEQ * HEAD_TILE), dtype=np.float32)
        t0 = time.perf_counter()
        softmax_mod(input_np, output_allo)
        t1 = time.perf_counter()
        npu_times.append((t1 - t0) * 1e6)

    # Compare
    np.testing.assert_allclose(output_allo, ref_flat, rtol=1e-2, atol=0.0)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    print(f"Average NPU execution time: {sum(npu_times)/len(npu_times):.2f} us")
    print(f"Min NPU execution time: {min(npu_times):.2f} us")

def _test_softmax_float_1024_1024():
    # Logical shape: [1024, 1024]. Softmax over dim=-1 (1024 elements).
    # Each logical row of 1024 is stored as 2 physical rows of 512.
    # Kernel takes [4][512] = 2 logical rows per invocation.
    # With 4 cores: each call processes 4 cores * 2 logical rows = 8 logical rows.
    # Loop 1024/8 = 128 times in Python.
    LOGICAL_ROWS = 1024
    LOGICAL_SEQ = 1024
    PHYS_COLS = 512
    PHYS_ROWS_PER_LOGICAL = LOGICAL_SEQ // PHYS_COLS  # 2
    NUM_CORES = 4
    KERNEL_PHYS_ROWS = 4  # C kernel signature: float[4][512]
    # Per-call: 4 cores * 4 phys rows = 16 phys rows = 8 logical rows
    BATCH_PHYS_ROWS = NUM_CORES * KERNEL_PHYS_ROWS  # 16
    BATCH_LOGICAL_ROWS = BATCH_PHYS_ROWS // PHYS_ROWS_PER_LOGICAL  # 8
    NUM_BATCHES = LOGICAL_ROWS // BATCH_LOGICAL_ROWS  # 128

    softmax_kernel = ExternalModule(
        top="softmax_float32_seq1024",
        impl_path=KERNEL_LIB_PATH + "v1_softmax_float.cc",
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32

    @df.region()
    def top(Input: Ty[BATCH_PHYS_ROWS, PHYS_COLS], Output: Ty[BATCH_PHYS_ROWS, PHYS_COLS]):
        @df.kernel(mapping=[NUM_CORES, 1], args=[Input, Output])
        def core(
            local_Input: Ty[BATCH_PHYS_ROWS, PHYS_COLS] @ Ly,
            local_Output: Ty[BATCH_PHYS_ROWS, PHYS_COLS] @ Ly,
        ):
            softmax_kernel(local_Input, local_Output)

    # Build once
    softmax_mod = df.build(
        top,
        target="aie",
        profile=False,
    )

    # Random input: logical [1024, 1024]
    torch.manual_seed(0)
    input_logical = torch.randn(LOGICAL_ROWS, LOGICAL_SEQ, dtype=torch.float32)

    # CPU reference
    with torch.no_grad():
        start = time.perf_counter()
        input_cpu = input_logical.cpu().numpy()                          # input data prep
        ref = F.softmax(input_logical.float(), dim=-1)                   # compute
        ref_np = ref.detach().cpu().numpy()                              # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    # Reshape to physical: [2048, 512]
    input_phys = input_logical.numpy().reshape(-1, PHYS_COLS)
    output_phys = np.zeros_like(input_phys)

    # Process in batches of [16, 512] (= 8 logical rows per batch)
    batch_times = []
    for b in range(NUM_BATCHES):
        start = b * BATCH_PHYS_ROWS
        end = start + BATCH_PHYS_ROWS
        batch_in = np.ascontiguousarray(input_phys[start:end])
        batch_out = np.zeros((BATCH_PHYS_ROWS, PHYS_COLS), dtype=np.float32)
        t0 = time.perf_counter()
        softmax_mod(batch_in, batch_out)
        t1 = time.perf_counter()
        batch_times.append((t1 - t0) * 1e6)
        output_phys[start:end] = batch_out

    # Compare against PyTorch reference
    output_logical = output_phys.reshape(LOGICAL_ROWS, LOGICAL_SEQ)
    np.testing.assert_allclose(output_logical, ref_np, rtol=1e-2, atol=0.0)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    print(f"Average NPU execution time: {sum(batch_times)/len(batch_times):.2f} us")
    print(f"Min NPU execution time: {min(batch_times):.2f} us")


if __name__ == "__main__":
    # _test_softmax_float_64_64()
    _test_softmax_float_1024_1024()