"""
Estimate SmolVLA inference time if allo had a persistent XRT runtime
(no subprocess per call, no XRT re-initialization).

Uses measured warm steady-state kernel times (profile mode, 50 warmup, 200 iters).
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════
# Measured warm kernel times (μs) — from profile mode on actual NPU
# ═══════════════════════════════════════════════════════════════════════

# Preprocessing kernels
CONV_US        = 128    # conv 256x256 → 16x16
ADD_32x32_US   = 150    # element-wise add 32x32
COPY_PREPROC_US = 119   # copy/reshape

# Connector kernels
CON_GEMM_US    = 157    # GEMM 64x960 from 64x64 × 64x960
CON_ADD_US     = 257    # add 64x64
CON_COPY_US    = 140    # copy/pixel shuffle

# Vision block kernels
VIS_GEMM_US    = 761    # GEMM 1024x768x768
VIS_SCORE_US   = 401    # GEMM score 1024x1024x64
VIS_LN_US      = 318    # LayerNorm 16x768
VIS_SOFTMAX_US = 3435   # Softmax float32 (per batch)
VIS_GELU_US    = 7353   # GELU 16x3072
VIS_GEMM_HID_US = 668   # GEMM FFN up (reuse text encoder as proxy)

# Text encoder kernels
TXT_RMSNORM_US = 173    # RMSNorm 16x960
TXT_GEMM_Q_US  = 503    # GEMM Q 128x960x960
TXT_GEMM_KV_US = 284    # GEMM KV 128x320x960
TXT_GEMM_SCORE_US = 137 # GEMM attn score 128x128x64
TXT_GEMM_VALUE_US = 138 # GEMM attn value 128x64x128
TXT_GEMM_OUT_US = 503   # GEMM out 128x960x960 (same shape as Q)
TXT_GEMM_FFN_UP_US = 668  # GEMM FFN up 128x2560x960
TXT_GEMM_FFN_DOWN_US = 266  # GEMM FFN down 128x960x320
TXT_SOFTMAX_US = 122    # Masked softmax 8x128
TXT_SILU_US    = 180    # SiLU 4x2560
ROPE_US        = 131    # Average RoPE sub-op (radians/sin/cos/copy/mul/add/sub/join/pack)

# Action expert kernels (use text encoder proxies, slightly smaller)
EXP_RMSNORM_US = 173    # Same kernel width=768
EXP_GEMM_Q_US  = 503    # 32x960x768 (similar)
EXP_GEMM_KV_US = 284    # 32x320x768
EXP_GEMM_SCORE_US = 137
EXP_GEMM_VALUE_US = 138
EXP_GEMM_OUT_US = 503
EXP_GEMM_FFN_UP_US = 668
EXP_GEMM_FFN_DOWN_US = 266
EXP_SOFTMAX_CROSS_US = 122
EXP_SILU_US    = 180

# ═══════════════════════════════════════════════════════════════════════
# Dispatch counts per component (from kernel count analysis)
# ═══════════════════════════════════════════════════════════════════════

def preprocessing():
    """768 embd × (3 channels × 4 conv + 3 add + 4 copy)"""
    conv_calls = 768 * 3 * 4   # 9216
    add_calls  = 768 * 3        # 2304
    copy_calls = 768 * 4        # 3072
    total_calls = conv_calls + add_calls + copy_calls  # 14592

    time_us = (conv_calls * CONV_US +
               add_calls * ADD_32x32_US +
               copy_calls * COPY_PREPROC_US)
    return total_calls, time_us

def connector():
    """192 K-chunks × (1 GEMM + 15 adds) + 64×4 copies"""
    k_chunks = 192
    gemm_calls = k_chunks       # 192
    add_calls  = k_chunks * 15  # 2880
    copy_calls = 64 * 4         # 256
    total_calls = gemm_calls + add_calls + copy_calls  # 3328

    time_us = (gemm_calls * CON_GEMM_US +
               add_calls * CON_ADD_US +
               copy_calls * CON_COPY_US)
    return total_calls, time_us

def vision_block():
    """1 layer of ViT: SEQ=1024, EMBD=768, 12 heads"""
    N_HEAD = 12
    SEQ = 1024
    NORM_TILES = SEQ // 16       # 64
    GELU_TILES = SEQ // 16       # 64
    FFN_DOWN_CHUNKS = 4
    SOFTMAX_BATCHES = 128        # per head

    ln1_calls = NORM_TILES                          # 64
    qkv_calls = 3                                    # 3 GEMM
    score_calls = N_HEAD                             # 12
    softmax_calls = N_HEAD * SOFTMAX_BATCHES         # 1536
    value_calls = N_HEAD                             # 12
    out_calls = 1                                    # 1 GEMM
    ln2_calls = NORM_TILES                           # 64
    ffn_up_calls = 1                                 # 1 GEMM
    gelu_calls = GELU_TILES                          # 64
    ffn_down_calls = FFN_DOWN_CHUNKS                 # 4

    total_calls = (ln1_calls + qkv_calls + score_calls + softmax_calls +
                   value_calls + out_calls + ln2_calls + ffn_up_calls +
                   gelu_calls + ffn_down_calls)  # 1761

    time_us = (ln1_calls * VIS_LN_US +
               3 * VIS_GEMM_US +               # QKV
               score_calls * VIS_SCORE_US +
               softmax_calls * VIS_SOFTMAX_US +
               value_calls * VIS_GEMM_US +      # approx
               out_calls * VIS_GEMM_US +
               ln2_calls * VIS_LN_US +
               ffn_up_calls * VIS_GEMM_HID_US +
               gelu_calls * VIS_GELU_US +
               ffn_down_calls * VIS_GEMM_US)
    return total_calls, time_us

def text_encoder():
    """1 layer: SEQ=128, EMBD=960, Q_H=15, KV_H=5"""
    Q_H = 15
    KV_H = 5
    SEQ = 128
    NORM_TILES = SEQ // 16        # 8
    SILU_TILES = SEQ // 4         # 32
    SOFTMAX_TILES = SEQ // 8      # 16
    FFN_DOWN_CHUNKS = 8
    ROPE_TILE = 64
    ROPE_SEQ_TILES = SEQ // ROPE_TILE  # 2

    # RoPE: per tile = 4 shared + heads×11
    rope_q_calls = ROPE_SEQ_TILES * (4 + Q_H * 11)   # 2*(4+165) = 338
    rope_k_calls = ROPE_SEQ_TILES * (4 + KV_H * 11)  # 2*(4+55) = 118

    norm1_calls = NORM_TILES                    # 8
    qkv_calls = 3                                # Q + K + V
    score_calls = Q_H                            # 15
    softmax_calls = Q_H * SOFTMAX_TILES          # 15*16 = 240
    value_calls = Q_H                            # 15
    out_calls = 1
    norm2_calls = NORM_TILES                     # 8
    gate_calls = 1
    up_calls = 1
    silu_calls = SILU_TILES                      # 32
    ffn_down_calls = FFN_DOWN_CHUNKS             # 8

    total_calls = (norm1_calls + qkv_calls + rope_q_calls + rope_k_calls +
                   score_calls + softmax_calls + value_calls + out_calls +
                   norm2_calls + gate_calls + up_calls + silu_calls + ffn_down_calls)

    time_us = (norm1_calls * TXT_RMSNORM_US +
               TXT_GEMM_Q_US + 2 * TXT_GEMM_KV_US +           # QKV
               (rope_q_calls + rope_k_calls) * ROPE_US +        # RoPE
               score_calls * TXT_GEMM_SCORE_US +
               softmax_calls * TXT_SOFTMAX_US +
               value_calls * TXT_GEMM_VALUE_US +
               out_calls * TXT_GEMM_OUT_US +
               norm2_calls * TXT_RMSNORM_US +
               gate_calls * TXT_GEMM_FFN_UP_US +
               up_calls * TXT_GEMM_FFN_UP_US +
               silu_calls * TXT_SILU_US +
               ffn_down_calls * TXT_GEMM_FFN_DOWN_US)
    return total_calls, time_us

def action_expert_self():
    """Self-attention: SEQ=32, EMBD=768"""
    Q_H = 15
    KV_H = 5
    SEQ = 32
    NORM_TILES = SEQ // 16        # 2
    SILU_TILES = SEQ // 4         # 8
    FFN_DOWN_CHUNKS = 8
    ROPE_TILE = 64
    # SEQ=32 < ROPE_TILE=64, still 1 tile (padded)
    rope_q_calls = 1 * (4 + Q_H * 11)   # 169
    rope_k_calls = 1 * (4 + KV_H * 11)  # 59

    norm1 = NORM_TILES          # 2
    qkv = 3
    score = Q_H                 # 15 (masked softmax on CPU, 0 NPU calls)
    value = Q_H                 # 15
    out = 1
    norm2 = NORM_TILES          # 2
    gate = 1
    up = 1
    silu = SILU_TILES           # 8
    ffn_down = FFN_DOWN_CHUNKS  # 8

    total_calls = (norm1 + qkv + rope_q_calls + rope_k_calls +
                   score + value + out + norm2 + gate + up + silu + ffn_down)

    time_us = (norm1 * EXP_RMSNORM_US +
               EXP_GEMM_Q_US + 2 * EXP_GEMM_KV_US +
               (rope_q_calls + rope_k_calls) * ROPE_US +
               score * EXP_GEMM_SCORE_US +
               # masked softmax on CPU — not counted
               value * EXP_GEMM_VALUE_US +
               out * EXP_GEMM_OUT_US +
               norm2 * EXP_RMSNORM_US +
               gate * EXP_GEMM_FFN_UP_US + up * EXP_GEMM_FFN_UP_US +
               silu * EXP_SILU_US +
               ffn_down * EXP_GEMM_FFN_DOWN_US)
    return total_calls, time_us

def action_expert_cross():
    """Cross-attention: SEQ=32, TEXT_SEQ=128"""
    Q_H = 15
    SEQ = 32
    NORM_TILES = SEQ // 16        # 2
    SILU_TILES = SEQ // 4         # 8
    FFN_DOWN_CHUNKS = 8
    rope_q_calls = 1 * (4 + Q_H * 11)  # 169

    norm1 = NORM_TILES          # 2
    q_proj = 1
    kv_cross = 2                # K_cross + V_cross
    score = Q_H                 # 15
    softmax = Q_H               # 15 (unmasked NPU softmax)
    value = Q_H                 # 15
    out = 1
    norm2 = NORM_TILES          # 2
    gate = 1
    up = 1
    silu = SILU_TILES           # 8
    ffn_down = FFN_DOWN_CHUNKS  # 8

    total_calls = (norm1 + q_proj + kv_cross + rope_q_calls +
                   score + softmax + value + out + norm2 + gate + up + silu + ffn_down)

    time_us = (norm1 * EXP_RMSNORM_US +
               q_proj * EXP_GEMM_Q_US +
               kv_cross * EXP_GEMM_KV_US +
               rope_q_calls * ROPE_US +
               score * EXP_GEMM_SCORE_US +
               softmax * EXP_SOFTMAX_CROSS_US +
               value * EXP_GEMM_VALUE_US +
               out * EXP_GEMM_OUT_US +
               norm2 * EXP_RMSNORM_US +
               gate * EXP_GEMM_FFN_UP_US + up * EXP_GEMM_FFN_UP_US +
               silu * EXP_SILU_US +
               ffn_down * EXP_GEMM_FFN_DOWN_US)
    return total_calls, time_us

def postprocessing():
    """RMSNorm + 1 GEMM"""
    return 2, EXP_RMSNORM_US + 503  # rough


def main():
    OVERHEAD_PER_CALL_MS = 28.0  # current measured

    components = {
        "Preprocessing":        preprocessing(),
        "Vision Encoder (1L)":  vision_block(),
        "Connector":            connector(),
        "Text Encoder (×2)":    (lambda c, t: (c*2, t*2))(*text_encoder()),
        "Action Expert Self":   action_expert_self(),
        "Action Expert Cross":  action_expert_cross(),
        "Postprocessing":       postprocessing(),
    }

    print(f"{'Component':<25s} {'Calls':>7s} {'Current (s)':>12s} {'Warm NPU (s)':>13s} {'Speedup':>8s}")
    print("=" * 70)

    total_calls = 0
    total_current_s = 0
    total_warm_s = 0

    rows = []
    for name, (calls, time_us) in components.items():
        current_s = calls * OVERHEAD_PER_CALL_MS / 1000
        warm_s = time_us / 1e6
        speedup = current_s / warm_s if warm_s > 0 else float('inf')
        total_calls += calls
        total_current_s += current_s
        total_warm_s += warm_s
        rows.append((name, calls, current_s, warm_s, speedup))
        print(f"{name:<25s} {calls:>7d} {current_s:>11.1f}s {warm_s:>12.3f}s {speedup:>7.0f}x")

    print("=" * 70)
    total_speedup = total_current_s / total_warm_s
    print(f"{'TOTAL':<25s} {total_calls:>7d} {total_current_s:>11.1f}s {total_warm_s:>12.3f}s {total_speedup:>7.0f}x")

    print(f"\n--- Summary ---")
    print(f"Total kernel dispatches:  {total_calls}")
    print(f"Current (subprocess/call): {total_current_s:.1f}s")
    print(f"Persistent XRT runtime:    {total_warm_s:.3f}s  ({total_speedup:.0f}x speedup)")
    print(f"NPU utilization current:   {total_warm_s/total_current_s*100:.2f}%")
    print(f"NPU utilization persistent: ~100%")

    # Save
    data = {
        "components": {name: {"calls": c, "current_s": round(cs, 2), "warm_npu_s": round(ws, 4), "speedup": round(sp, 1)}
                       for (name, c, cs, ws, sp) in rows},
        "total": {"calls": total_calls, "current_s": round(total_current_s, 1), "warm_npu_s": round(total_warm_s, 3), "speedup": round(total_speedup, 0)},
    }
    out_path = os.path.join(SCRIPT_DIR, "persistent_runtime_estimate.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved to: {out_path}")


if __name__ == "__main__":
    main()
