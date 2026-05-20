"""
Measure dispatch overhead across different kernels.
Compares fixed vs variable components of dispatch time.
"""

import time
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

# ── Tiny: add 32x32 (1 tile, 2KB) ──────────────────────────────────────────
from preprocessing_bf16 import add_mod as preproc_add_mod

# ── Small: RMSNorm 16x768 (4 tiles, 24KB) ──────────────────────────────────
from text_encoder_bf16 import rms_norm_mod as text_rms_mod

# ── Small: masked softmax 8x128 (1 tile, 2KB) ──────────────────────────────
from text_encoder_bf16 import masked_softmax_mod

# ── Small: SiLU 4x2560 (16 tiles, 20KB) ────────────────────────────────────
from text_encoder_bf16 import silu_mod as text_silu_mod

# ── Medium: GEMM KV 128x320x960 (many tiles) ───────────────────────────────
from text_encoder_bf16 import gemm_kv_mod

# ── Large: GEMM Q 128x960x960 (many tiles) ─────────────────────────────────
from text_encoder_bf16 import gemm_q_mod

# ── Large: GEMM FFN up 128x2560x960 (many tiles) ───────────────────────────
from text_encoder_bf16 import gemm_ffn_up_mod

# ── RoPE sub-ops (1-2 tiles, tiny data) ────────────────────────────────────
from text_encoder_bf16 import (
    radians_mod, pack_mod, sin_mod, cos_mod,
    copyL_mod, mul32_mod, join_mod,
)

# ── Action expert small GEMM: 32x960x768 ───────────────────────────────────
from action_expert_bf16 import gemm_q_mod as exp_gemm_q_mod

# ── Vision block GEMM: 1024x768x768 ────────────────────────────────────────
from vision_block_bf16 import gemm_embd_embd_mod


def bench(name, fn, warmup=5, iters=30):
    """Time a callable, return stats."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    times.sort()
    return {
        "name": name,
        "min": times[0],
        "median": times[len(times) // 2],
        "mean": sum(times) / len(times),
        "max": times[-1],
        "p95": times[int(len(times) * 0.95)],
    }


def main():
    rng = np.random.default_rng(42)
    bf16 = np_bfloat16

    # ── Prepare buffers ─────────────────────────────────────────────────────

    # Tiny: add 32x32
    add_a = rng.standard_normal((32, 32)).astype(bf16)
    add_b = rng.standard_normal((32, 32)).astype(bf16)
    add_c = np.zeros((32, 32), dtype=bf16)

    # RMSNorm 16x960
    norm_in = rng.standard_normal((16, 960)).astype(bf16)
    norm_w = rng.standard_normal((960,)).astype(bf16)
    norm_out = np.zeros((16, 960), dtype=bf16)

    # Masked softmax 8x128
    sm_in = rng.standard_normal((8, 128)).astype(bf16)
    sm_row = np.array([0], dtype=np.int32)
    sm_out = np.zeros((8, 128), dtype=bf16)

    # SiLU 4x2560
    silu_in = rng.standard_normal((4, 2560)).astype(bf16)
    silu_out = np.zeros((4, 2560), dtype=bf16)

    # GEMM KV: 128x320 = A[128,960] x B[960,320]
    gkv_a = rng.standard_normal((128, 960)).astype(bf16)
    gkv_b = rng.standard_normal((960, 320)).astype(bf16)
    gkv_c = np.zeros((128, 320), dtype=bf16)

    # GEMM Q: 128x960 = A[128,960] x B[960,960]
    gq_a = rng.standard_normal((128, 960)).astype(bf16)
    gq_b = rng.standard_normal((960, 960)).astype(bf16)
    gq_c = np.zeros((128, 960), dtype=bf16)

    # GEMM FFN up: 128x2560 = A[128,960] x B[960,2560]
    gfu_a = rng.standard_normal((128, 960)).astype(bf16)
    gfu_b = rng.standard_normal((960, 2560)).astype(bf16)
    gfu_c = np.zeros((128, 2560), dtype=bf16)

    # RoPE sub-ops (float32, ROPE_TILE=64)
    rope_pos = np.arange(64, dtype=np.float32)
    rope_inv = np.ones(32, dtype=np.float32)
    rope_rad32 = np.zeros((64, 32), dtype=np.float32)
    rope_rad64 = np.zeros((64, 64), dtype=np.float32)
    rope_sin = np.zeros((64, 64), dtype=np.float32)
    rope_cos = np.zeros((64, 64), dtype=np.float32)
    rope_half = np.zeros((64, 32), dtype=np.float32)
    rope_half2 = np.zeros((64, 32), dtype=np.float32)
    rope_join = np.zeros((64, 64), dtype=np.float32)

    # Action expert GEMM Q: 32x960 = A[32,768] x B[768,960]
    eq_a = rng.standard_normal((32, 768)).astype(bf16)
    eq_b = rng.standard_normal((768, 960)).astype(bf16)
    eq_c = np.zeros((32, 960), dtype=bf16)

    # Vision GEMM: 1024x768 = A[1024,768] x B[768,768]
    vg_a = rng.standard_normal((1024, 768)).astype(bf16)
    vg_b = rng.standard_normal((768, 768)).astype(bf16)
    vg_c = np.zeros((1024, 768), dtype=bf16)

    # ── Run benchmarks ──────────────────────────────────────────────────────

    results = []

    results.append(bench(
        "add 32x32 (1 tile, 2KB)",
        lambda: preproc_add_mod(add_a, add_b, add_c),
    ))
    results.append(bench(
        "RMSNorm 16x960 (4 tiles, 30KB)",
        lambda: text_rms_mod(norm_in, norm_w, norm_out),
    ))
    results.append(bench(
        "masked_softmax 8x128 (1 tile, 2KB)",
        lambda: masked_softmax_mod(sm_in, sm_row, sm_out),
    ))
    results.append(bench(
        "SiLU 4x2560 (16 tiles, 20KB)",
        lambda: text_silu_mod(silu_in, silu_out),
    ))
    results.append(bench(
        "RoPE radians 64x32 (1 tile, f32)",
        lambda: radians_mod(rope_pos, rope_inv, rope_rad32),
    ))
    results.append(bench(
        "RoPE pack 64x32->64x64 (1 tile, f32)",
        lambda: pack_mod(rope_rad32, rope_rad64),
    ))
    results.append(bench(
        "RoPE sin 64x64 (2 tiles, f32)",
        lambda: sin_mod(rope_rad64, rope_sin),
    ))
    results.append(bench(
        "RoPE cos 64x64 (2 tiles, f32)",
        lambda: cos_mod(rope_rad64, rope_cos),
    ))
    results.append(bench(
        "RoPE copyL 64x64->64x32 (1 tile, f32)",
        lambda: copyL_mod(rope_rad64, rope_half),
    ))
    results.append(bench(
        "RoPE mul32 64x32 (1 tile, f32)",
        lambda: mul32_mod(rope_half, rope_half2, rope_half),
    ))
    results.append(bench(
        "RoPE join 2x64x32->64x64 (2 tiles, f32)",
        lambda: join_mod(rope_half, rope_half2, rope_join),
    ))
    results.append(bench(
        "GEMM KV 128x320x960 (medium)",
        lambda: gemm_kv_mod(gkv_a, gkv_b, gkv_c),
    ))
    results.append(bench(
        "GEMM Q 128x960x960 (large)",
        lambda: gemm_q_mod(gq_a, gq_b, gq_c),
    ))
    results.append(bench(
        "GEMM FFN_up 128x2560x960 (xlarge)",
        lambda: gemm_ffn_up_mod(gfu_a, gfu_b, gfu_c),
    ))
    results.append(bench(
        "GEMM exp_Q 32x960x768 (action)",
        lambda: exp_gemm_q_mod(eq_a, eq_b, eq_c),
    ))
    results.append(bench(
        "GEMM vision 1024x768x768 (huge)",
        lambda: gemm_embd_embd_mod(vg_a, vg_b, vg_c),
    ))

    # ── Print results ───────────────────────────────────────────────────────

    print()
    print(f"{'Kernel':<42s} {'Min':>8s} {'Median':>8s} {'Mean':>8s} {'P95':>8s} {'Max':>8s}")
    print("-" * 82)
    for r in results:
        print(f"{r['name']:<42s} {r['min']:8.2f} {r['median']:8.2f} {r['mean']:8.2f} {r['p95']:8.2f} {r['max']:8.2f}")

    print()
    mins = [r["min"] for r in results]
    print(f"Fastest dispatch (floor):  {min(mins):.2f} ms")
    print(f"Slowest dispatch:          {max(mins):.2f} ms")
    print(f"Range:                     {max(mins) - min(mins):.2f} ms")
    print()

    # Separate into categories
    rope_results = [r for r in results if "RoPE" in r["name"]]
    gemm_results = [r for r in results if "GEMM" in r["name"]]
    elem_results = [r for r in results if r not in rope_results and r not in gemm_results]

    if rope_results:
        rope_mins = [r["min"] for r in rope_results]
        print(f"RoPE sub-ops avg min:      {sum(rope_mins)/len(rope_mins):.2f} ms  (range {min(rope_mins):.2f} - {max(rope_mins):.2f})")
    if elem_results:
        elem_mins = [r["min"] for r in elem_results]
        print(f"Element-wise avg min:      {sum(elem_mins)/len(elem_mins):.2f} ms  (range {min(elem_mins):.2f} - {max(elem_mins):.2f})")
    if gemm_results:
        gemm_mins = [r["min"] for r in gemm_results]
        print(f"GEMM avg min:              {sum(gemm_mins)/len(gemm_mins):.2f} ms  (range {min(gemm_mins):.2f} - {max(gemm_mins):.2f})")

    print()
    print("All times in milliseconds. 'Min' is closest to pure dispatch+compute.")
    print("If min times are similar across sizes, dispatch overhead dominates.")
    print("If min times scale with data size, DMA transfer time matters.")


if __name__ == "__main__":
    main()
