"""
Capture both wall-clock time and NPU execution time for each kernel call.
Plot the breakdown: Context+Preparation vs Launch+Execution (NPU time).

The C++ harness prints "NPU execution time: XXXus" to stdout for each call.
Wall-clock = Context Creation + Preparation + Launch + Execution
NPU time   = Launch + Execution
Overhead   = Wall-clock - NPU time = Context Creation + Preparation
"""

import os
import sys
import io
import re
import time
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import kernels ──────────────────────────────────────────────────────────

from preprocessing_bf16 import add_mod as preproc_add_mod
from text_encoder_bf16 import (
    rms_norm_mod as text_rms_mod,
    masked_softmax_mod,
    silu_mod as text_silu_mod,
    gemm_kv_mod, gemm_q_mod, gemm_ffn_up_mod,
    radians_mod, pack_mod, sin_mod, cos_mod,
    copyL_mod, mul32_mod, join_mod,
)
from action_expert_bf16 import gemm_q_mod as exp_gemm_q_mod
from vision_block_bf16 import gemm_embd_embd_mod


def capture_npu_time(fn):
    """Call fn() while capturing stdout at OS fd level to extract NPU execution time.
    The C++ harness writes directly to fd 1, bypassing Python's sys.stdout."""
    import tempfile

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

    # Parse "NPU execution time: XXXus"
    match = re.search(r"NPU execution time:\s*([\d.]+)us", output)
    npu_us = float(match.group(1)) if match else None
    npu_ms = npu_us / 1000.0 if npu_us is not None else None

    return wall_ms, npu_ms


def bench_kernel(name, fn, warmup=3, iters=20):
    """Benchmark a kernel, capturing both wall-clock and NPU time."""
    for _ in range(warmup):
        fn()

    records = []
    for _ in range(iters):
        wall_ms, npu_ms = capture_npu_time(fn)
        records.append({"wall_ms": wall_ms, "npu_ms": npu_ms})

    walls = [r["wall_ms"] for r in records]
    npus = [r["npu_ms"] for r in records if r["npu_ms"] is not None]

    return {
        "name": name,
        "wall_min": min(walls),
        "wall_median": sorted(walls)[len(walls) // 2],
        "npu_min": min(npus) if npus else None,
        "npu_median": sorted(npus)[len(npus) // 2] if npus else None,
        "overhead_min": min(walls) - min(npus) if npus else None,
        "all_walls": walls,
        "all_npus": npus,
    }


def main():
    rng = np.random.default_rng(42)
    bf16 = np_bfloat16

    # ── Prepare buffers ─────────────────────────────────────────────────────
    add_a = rng.standard_normal((32, 32)).astype(bf16)
    add_b = rng.standard_normal((32, 32)).astype(bf16)
    add_c = np.zeros((32, 32), dtype=bf16)

    norm_in = rng.standard_normal((16, 960)).astype(bf16)
    norm_w = rng.standard_normal((960,)).astype(bf16)
    norm_out = np.zeros((16, 960), dtype=bf16)

    sm_in = rng.standard_normal((8, 128)).astype(bf16)
    sm_row = np.array([0], dtype=np.int32)
    sm_out = np.zeros((8, 128), dtype=bf16)

    silu_in = rng.standard_normal((4, 2560)).astype(bf16)
    silu_out = np.zeros((4, 2560), dtype=bf16)

    gkv_a = rng.standard_normal((128, 960)).astype(bf16)
    gkv_b = rng.standard_normal((960, 320)).astype(bf16)
    gkv_c = np.zeros((128, 320), dtype=bf16)

    gq_a = rng.standard_normal((128, 960)).astype(bf16)
    gq_b = rng.standard_normal((960, 960)).astype(bf16)
    gq_c = np.zeros((128, 960), dtype=bf16)

    gfu_a = rng.standard_normal((128, 960)).astype(bf16)
    gfu_b = rng.standard_normal((960, 2560)).astype(bf16)
    gfu_c = np.zeros((128, 2560), dtype=bf16)

    rope_pos = np.arange(64, dtype=np.float32)
    rope_inv = np.ones(32, dtype=np.float32)
    rope_rad32 = np.zeros((64, 32), dtype=np.float32)
    rope_rad64 = np.zeros((64, 64), dtype=np.float32)
    rope_sin = np.zeros((64, 64), dtype=np.float32)
    rope_cos = np.zeros((64, 64), dtype=np.float32)
    rope_half = np.zeros((64, 32), dtype=np.float32)
    rope_half2 = np.zeros((64, 32), dtype=np.float32)
    rope_join = np.zeros((64, 64), dtype=np.float32)

    eq_a = rng.standard_normal((32, 768)).astype(bf16)
    eq_b = rng.standard_normal((768, 960)).astype(bf16)
    eq_c = np.zeros((32, 960), dtype=bf16)

    vg_a = rng.standard_normal((1024, 768)).astype(bf16)
    vg_b = rng.standard_normal((768, 768)).astype(bf16)
    vg_c = np.zeros((1024, 768), dtype=bf16)

    # ── Define benchmarks ───────────────────────────────────────────────────
    benchmarks = [
        ("add 32x32\n(1 tile, 2KB)",         lambda: preproc_add_mod(add_a, add_b, add_c)),
        ("RMSNorm 16x960\n(4 tiles, 30KB)",  lambda: text_rms_mod(norm_in, norm_w, norm_out)),
        ("softmax 8x128\n(1 tile, 2KB)",     lambda: masked_softmax_mod(sm_in, sm_row, sm_out)),
        ("SiLU 4x2560\n(16 tiles, 20KB)",    lambda: text_silu_mod(silu_in, silu_out)),
        ("RoPE radians\n(1 tile, f32)",       lambda: radians_mod(rope_pos, rope_inv, rope_rad32)),
        ("RoPE sin\n(2 tiles, f32)",          lambda: sin_mod(rope_rad64, rope_sin)),
        ("RoPE copyL\n(1 tile, f32)",         lambda: copyL_mod(rope_rad64, rope_half)),
        ("RoPE mul32\n(1 tile, f32)",         lambda: mul32_mod(rope_half, rope_half2, rope_half)),
        ("RoPE join\n(2 tiles, f32)",         lambda: join_mod(rope_half, rope_half2, rope_join)),
        ("GEMM KV\n128x320x960",             lambda: gemm_kv_mod(gkv_a, gkv_b, gkv_c)),
        ("GEMM Q\n128x960x960",              lambda: gemm_q_mod(gq_a, gq_b, gq_c)),
        ("GEMM FFN_up\n128x2560x960",        lambda: gemm_ffn_up_mod(gfu_a, gfu_b, gfu_c)),
        ("GEMM exp_Q\n32x960x768",           lambda: exp_gemm_q_mod(eq_a, eq_b, eq_c)),
        ("GEMM vision\n1024x768x768",        lambda: gemm_embd_embd_mod(vg_a, vg_b, vg_c)),
    ]

    # ── Run ─────────────────────────────────────────────────────────────────
    print("Running benchmarks (capturing NPU time from stdout)...")
    results = []
    for name, fn in benchmarks:
        print(f"  {name.replace(chr(10), ' ')}...", end=" ", flush=True)
        r = bench_kernel(name, fn)
        overhead_str = f"{r['overhead_min']:.1f}" if r['overhead_min'] is not None else "N/A"
        npu_str = f"{r['npu_min']:.2f}" if r['npu_min'] is not None else "N/A"
        print(f"wall={r['wall_min']:.1f}ms  npu={npu_str}ms  overhead={overhead_str}ms")
        results.append(r)

    # ── Print table ─────────────────────────────────────────────────────────
    print()
    print(f"{'Kernel':<30s} {'Wall(ms)':>10s} {'NPU(ms)':>10s} {'Overhead(ms)':>13s} {'NPU%':>7s}")
    print("-" * 75)
    for r in results:
        wall = r["wall_min"]
        npu = r["npu_min"]
        if npu is not None:
            overhead = wall - npu
            pct = (npu / wall) * 100
            print(f"{r['name'].replace(chr(10), ' '):<30s} {wall:10.2f} {npu:10.2f} {overhead:13.2f} {pct:6.1f}%")
        else:
            print(f"{r['name'].replace(chr(10), ' '):<30s} {wall:10.2f} {'N/A':>10s} {'N/A':>13s} {'N/A':>7s}")

    # ── Plot 1: Stacked bar — NPU time vs Overhead ─────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    names = [r["name"] for r in results]
    walls = [r["wall_min"] for r in results]
    npus = [r["npu_min"] if r["npu_min"] is not None else 0 for r in results]
    overheads = [w - n for w, n in zip(walls, npus)]

    x = np.arange(len(names))

    ax1 = axes[0]
    bars_npu = ax1.bar(x, npus, color="#2C7BE5", label="NPU time (Launch + Execution)")
    bars_oh = ax1.bar(x, overheads, bottom=npus, color="#F59E0B", label="Overhead (Context + Preparation)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7.5, ha="center")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Per-Kernel Time Breakdown: NPU Compute vs Dispatch Overhead", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Annotate overhead percentage
    for i, (w, n) in enumerate(zip(walls, npus)):
        if n > 0:
            pct = ((w - n) / w) * 100
            ax1.text(i, w + 0.3, f"{pct:.0f}%\nOH", ha="center", va="bottom", fontsize=7, color="#B45309")

    # ── Plot 2: NPU time zoomed in (log scale to see variation) ────────────
    ax2 = axes[1]

    # Box plot of NPU times
    npu_data = [r["all_npus"] if r["all_npus"] else [0] for r in results]
    wall_data = [r["all_walls"] for r in results]

    bp1 = ax2.boxplot(wall_data, positions=x - 0.15, widths=0.25, patch_artist=True,
                       boxprops=dict(facecolor="#FEF3C7", edgecolor="#F59E0B"),
                       medianprops=dict(color="#B45309"), flierprops=dict(markersize=3))
    bp2 = ax2.boxplot(npu_data, positions=x + 0.15, widths=0.25, patch_artist=True,
                       boxprops=dict(facecolor="#DBEAFE", edgecolor="#2C7BE5"),
                       medianprops=dict(color="#1E40AF"), flierprops=dict(markersize=3))

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7.5, ha="center")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Distribution: Wall-Clock (yellow) vs NPU Time (blue)", fontsize=13, fontweight="bold")
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Wall-clock", "NPU time"], loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("dispatch_breakdown.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: dispatch_breakdown.png")

    # ── Plot 3: Pie chart of where time goes in full pipeline ──────────────
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # Use median NPU times for the kernel categories
    categories = {
        "RoPE sub-ops": [],
        "Element-wise\n(norm/softmax/SiLU)": [],
        "GEMM": [],
    }
    for r in results:
        n = r["name"]
        if "RoPE" in n:
            categories["RoPE sub-ops"].append(r)
        elif "GEMM" in n:
            categories["GEMM"].append(r)
        else:
            categories["Element-wise\n(norm/softmax/SiLU)"].append(r)

    # Pie: overhead vs NPU for a "typical" dispatch
    avg_wall = np.mean(walls)
    avg_npu = np.mean([n for n in npus if n > 0])
    avg_overhead = avg_wall - avg_npu

    ax3.pie([avg_npu, avg_overhead],
            labels=[f"NPU compute\n{avg_npu:.1f}ms", f"Dispatch overhead\n{avg_overhead:.1f}ms"],
            colors=["#2C7BE5", "#F59E0B"],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax3.set_title("Average Per-Call Time Split", fontsize=13, fontweight="bold")

    # Pie: estimated full-pipeline time by component
    dispatch_counts = {
        "Preprocessing\n(14,592 calls)": 14592,
        "Connector\n(3,328 calls)": 3328,
        "Vision Block\n(1,761 calls)": 1761,
        "Text Encoder\n(1,576 calls)": 1576,
        "Action Expert\n(524 calls)": 524,
    }
    colors_pipe = ["#EF4444", "#F59E0B", "#10B981", "#2C7BE5", "#7C3AED"]
    sizes = [count * avg_wall / 1000 for count in dispatch_counts.values()]  # seconds

    ax4.pie(sizes, labels=[f"{k}\n~{s:.0f}s" for k, s in zip(dispatch_counts.keys(), sizes)],
            colors=colors_pipe, autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9})
    ax4.set_title("Estimated Full-Pipeline Time by Component", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig("pipeline_breakdown.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to: pipeline_breakdown.png")


if __name__ == "__main__":
    main()
