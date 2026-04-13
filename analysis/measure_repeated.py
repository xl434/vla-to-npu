"""
Run several kernels 10 times each. For each call, record:
  - Wall-clock time (timer around the full function call)
  - NPU time (printed by C++ harness: kernel launch + wait)
  - Overhead = Wall-clock - NPU = Context creation + Arg preparation + DMA sync
Save data to JSON for plotting separately.

Usage:
  cd vla && python ../analysis/measure_repeated.py
"""

import os
import sys
import re
import json
import time
import tempfile
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

# Add vla directory to path so we can import kernel modules
VLA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vla")
sys.path.insert(0, VLA_DIR)
os.chdir(VLA_DIR)

from preprocessing_bf16 import add_mod as preproc_add_mod
from text_encoder_bf16 import (
    rms_norm_mod as text_rms_mod,
    gemm_q_mod,
    silu_mod as text_silu_mod,
    copyL_mod,
)
from vision_block_bf16 import gemm_embd_embd_mod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "repeated_calls_data.json")


def capture_call(fn):
    """Single call: capture wall-clock and NPU time via OS fd redirect."""
    sys.stdout.flush()
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+")
    os.dup2(tmp.fileno(), 1)

    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()

    sys.stdout.flush()
    os.dup2(old_fd, 1)
    os.close(old_fd)
    tmp.seek(0)
    output = tmp.read()
    tmp.close()

    wall_ms = (t1 - t0) * 1000
    match = re.search(r"NPU execution time:\s*([\d.]+)us", output)
    npu_ms = float(match.group(1)) / 1000.0 if match else None
    return wall_ms, npu_ms


def run_kernel(name, fn, n=10):
    """Run kernel n times, return list of dicts."""
    fn()  # 1 warmup (not recorded)
    records = []
    for i in range(n):
        wall, npu = capture_call(fn)
        overhead = (wall - npu) if npu is not None else None
        rec = {
            "trial": i + 1,
            "wall_ms": round(wall, 2),
            "npu_ms": round(npu, 2) if npu is not None else None,
            "overhead_ms": round(overhead, 2) if overhead is not None else None,
        }
        records.append(rec)
        npu_s = f"{npu:.2f}" if npu is not None else "N/A"
        oh_s = f"{overhead:.2f}" if overhead is not None else "N/A"
        print(f"  trial {i+1:2d}: wall={wall:.1f}ms  npu={npu_s}ms  overhead={oh_s}ms")
    return records


def main():
    rng = np.random.default_rng(42)
    bf16 = np_bfloat16
    N = 10

    # Buffers
    add_a = rng.standard_normal((32, 32)).astype(bf16)
    add_b = rng.standard_normal((32, 32)).astype(bf16)
    add_c = np.zeros((32, 32), dtype=bf16)

    norm_in = rng.standard_normal((16, 960)).astype(bf16)
    norm_w = rng.standard_normal((960,)).astype(bf16)
    norm_out = np.zeros((16, 960), dtype=bf16)

    silu_in = rng.standard_normal((4, 2560)).astype(bf16)
    silu_out = np.zeros((4, 2560), dtype=bf16)

    rope_in = np.zeros((64, 64), dtype=np.float32)
    rope_out = np.zeros((64, 32), dtype=np.float32)

    gq_a = rng.standard_normal((128, 960)).astype(bf16)
    gq_b = rng.standard_normal((960, 960)).astype(bf16)
    gq_c = np.zeros((128, 960), dtype=bf16)

    vg_a = rng.standard_normal((1024, 768)).astype(bf16)
    vg_b = rng.standard_normal((768, 768)).astype(bf16)
    vg_c = np.zeros((1024, 768), dtype=bf16)

    kernels = [
        ("add 32x32 (1 tile, 2KB)",    lambda: preproc_add_mod(add_a, add_b, add_c)),
        ("RMSNorm 16x960 (4 tiles)",   lambda: text_rms_mod(norm_in, norm_w, norm_out)),
        ("SiLU 4x2560 (16 tiles)",     lambda: text_silu_mod(silu_in, silu_out)),
        ("RoPE copyL (1 tile, f32)",   lambda: copyL_mod(rope_in, rope_out)),
        ("GEMM Q 128x960x960",         lambda: gemm_q_mod(gq_a, gq_b, gq_c)),
        ("GEMM vision 1024x768x768",   lambda: gemm_embd_embd_mod(vg_a, vg_b, vg_c)),
    ]

    all_data = {}
    for name, fn in kernels:
        print(f"\n{name}:")
        all_data[name] = run_kernel(name, fn, N)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nData saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
