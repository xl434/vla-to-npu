# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import torch.nn.functional as F
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

KERNEL_LIB_PATH = "../cc/bf16_old/"

S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ty = bfloat16


def _test_softmax_bf16_1024_1024():
    """
    Logical shape: [1024, 1024]. Softmax over dim=-1 (1024 elements).
    Each logical row of 1024 is stored as 2 physical rows of 512.
    Kernel takes [4][512] = 2 logical rows per invocation.
    With 4 cores: each call processes 4 cores * 2 logical rows = 8 logical rows.
    Loop 1024/8 = 128 times in Python.
    """
    LOGICAL_ROWS = 1024
    LOGICAL_SEQ = 1024
    PHYS_COLS = 512
    PHYS_ROWS_PER_LOGICAL = LOGICAL_SEQ // PHYS_COLS  # 2
    NUM_CORES = 4
    KERNEL_PHYS_ROWS = 4  # C kernel signature: bfloat16[4][512]
    BATCH_PHYS_ROWS = NUM_CORES * KERNEL_PHYS_ROWS  # 16
    BATCH_LOGICAL_ROWS = BATCH_PHYS_ROWS // PHYS_ROWS_PER_LOGICAL  # 8
    NUM_BATCHES = LOGICAL_ROWS // BATCH_LOGICAL_ROWS  # 128

    softmax_kernel = ExternalModule(
        top="softmax_bf16_seq1024",
        impl_path=KERNEL_LIB_PATH + "v1_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(
        Input: Ty[BATCH_PHYS_ROWS, PHYS_COLS],
        Output: Ty[BATCH_PHYS_ROWS, PHYS_COLS],
    ):
        @df.kernel(mapping=[NUM_CORES, 1], args=[Input, Output])
        def core(
            local_Input: Ty[BATCH_PHYS_ROWS, PHYS_COLS] @ Ly,
            local_Output: Ty[BATCH_PHYS_ROWS, PHYS_COLS] @ Ly,
        ):
            softmax_kernel(local_Input, local_Output)

    # Build once
    softmax_mod = df.build(top, target="aie")

    # Random input: logical [1024, 1024]
    torch.manual_seed(0)
    input_logical = torch.randn(LOGICAL_ROWS, LOGICAL_SEQ, dtype=torch.float32)
    input_bf16 = input_logical.numpy().astype(np_bfloat16)

    # Reshape to physical: [2048, 512]
    input_phys = input_bf16.reshape(-1, PHYS_COLS)
    output_phys = np.zeros_like(input_phys)

    # Process in batches of [16, 512] (= 8 logical rows per batch)
    t0 = time.perf_counter()
    for b in range(NUM_BATCHES):
        start = b * BATCH_PHYS_ROWS
        end = start + BATCH_PHYS_ROWS
        batch_in = np.ascontiguousarray(input_phys[start:end])
        batch_out = np.zeros((BATCH_PHYS_ROWS, PHYS_COLS), dtype=np_bfloat16)
        softmax_mod(batch_in, batch_out)
        output_phys[start:end] = batch_out
    t1 = time.perf_counter()
    npu_us = (t1 - t0) * 1e6
    print(
        f"NPU softmax bf16 [1024x1024] total: {npu_us:.1f} us ({NUM_BATCHES} batches)"
    )

    # Compare against PyTorch reference (compute in float, compare in float)
    input_torch_bf16 = torch.from_numpy(input_bf16.astype(np.float32)).to(
        torch.bfloat16
    )
    ref = F.softmax(input_torch_bf16.float(), dim=-1).to(torch.bfloat16)
    output_logical = output_phys.reshape(LOGICAL_ROWS, LOGICAL_SEQ)
    ref_np = ref.numpy().astype(np.float32)
    np.testing.assert_allclose(
        output_logical.astype(np.float32), ref_np, atol=1e-1, rtol=1e-2
    )

    # CPU baseline timing
    warmup = 20
    iters = 100
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = F.softmax(input_torch_bf16.float(), dim=-1)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.softmax(input_torch_bf16.float(), dim=-1)
        t1 = time.perf_counter()
    avg_us = (t1 - t0) / iters * 1e6
    print(
        f"CPU softmax avg over {iters} iters (after {warmup} warmup): {avg_us:.1f} us"
    )
    print("PASS! Softmax bf16 [1024x1024] matches PyTorch reference within tolerance.")


if __name__ == "__main__":
    _test_softmax_bf16_1024_1024()
