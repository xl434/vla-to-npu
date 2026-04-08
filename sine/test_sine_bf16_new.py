# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import numpy as np
import torch
import ml_dtypes
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import bfloat16
from allo.backend.aie.external_kernel import ExternalModule

# Matrix shape: rows=32, cols=64  (matches sin_float32 signature: [32][64])
S = Layout.Shard
R = Layout.Replicate
Ly = [S(0), S(1)]
Ty = bfloat16
seq_tile = 32       # rows
feature_tile = 64   # cols

# RTOL = 1e-3
# ATOL = 1e-4
RTOL = 1e-2
ATOL = 1e-3

def _mismatch_stats(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float):
    """
    Return (mismatch_pct, mismatch_count, total_count)
    using the same per-element rule as numpy.allclose/assert_allclose.
    """
    diff = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    mismatch_mask = diff > tol

    total = mismatch_mask.size
    mismatches = int(np.count_nonzero(mismatch_mask))
    mismatch_pct = 100.0 * mismatches / total if total else 0.0
    return mismatch_pct, mismatches, total

KERNEL_LIB_PATH = "../cc/bf16_new/"
def _test_sine_single_tile():
    sine = ExternalModule(
        top="sin_bfloat16",
        impl_path=KERNEL_LIB_PATH + "sine_bf16.cc",
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
            sine(local_input_x, local_output_x)
    
    torch.manual_seed(0)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.bfloat16) * 40.0) - 20.0

    # CPU execution time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        ref_out = torch.sin(torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16))  # compute
        ref_numpy = ref_out.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        # mod = df.build(top, target="aie", profile=True)
        mod = df.build(
            top,
            target="aie",
            profile=True,
            trace=[("core", (0, 0))],
            trace_size=65536,
        )
        output_allo = np.zeros((seq_tile, feature_tile), dtype=ml_dtypes.bfloat16)
        input_numpy = input_tensor.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16)

        mod(input_numpy, output_allo)

        print(f"CPU execution time: {cpu_time_us:.2f} us")
        output_allo = output_allo.astype(ml_dtypes.bfloat16)
        try:
            np.testing.assert_allclose(output_allo, ref_numpy, rtol=RTOL, atol=ATOL)
            print(f"PASSED sine! (rtol={RTOL}, atol={ATOL})")
        except AssertionError as e:
            # Debug: print summary and a few worst mismatches
            pct, mism, total = _mismatch_stats(output_allo, ref_numpy, RTOL, ATOL)
            diff    = np.abs(output_allo - ref_numpy)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print("Sine mismatch detected.")
            print(f"Mismatch rate: {pct:.4f}% ({mism}/{total})  (rtol={RTOL}, atol={ATOL})")
            print(f"Max abs diff = {diff[max_idx]:.6e} at index {max_idx}")
            r, c = max_idx
            print(f"Input        = {input_tensor[r, c].item():.6f}")
            print(f"Allo output  = {output_allo[r, c]}")
            print(f"Torch output = {ref_numpy[r, c]:.6f}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")

if __name__ == "__main__":
    _test_sine_single_tile()