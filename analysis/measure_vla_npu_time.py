"""
Run vla.py's full pipeline, capturing NPU execution times from stdout
for each component. Sums up the NPU times per component.

Usage:
  python measure_vla_npu_time.py
"""

import os
import sys
import re
import time
import tempfile
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

VLA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vla")
sys.path.insert(0, VLA_DIR)
os.chdir(VLA_DIR)


def capture_npu_times(fn):
    """Call fn(), capture all NPU execution time prints, return (wall_ms, [npu_us_list])."""
    sys.stdout.flush()
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+")
    os.dup2(tmp.fileno(), 1)

    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()

    sys.stdout.flush()
    os.dup2(old_fd, 1)
    os.close(old_fd)
    tmp.seek(0)
    output = tmp.read()
    tmp.close()

    wall_ms = (t1 - t0) * 1000
    npu_times_us = [float(m) for m in re.findall(r"NPU execution time:\s*([\d.]+)us", output)]
    return wall_ms, npu_times_us, result


def main():
    # Import everything from vla.py
    from vla import (
        preprocessing_block, vision_encoder, connector_block,
        joint_transformer, postprocessing,
        create_text_emb, create_state_emb,
        VIT_NUM_LAYERS, LLAMA_NUM_LAYERS,
        CH, PIX, EMBD_P, KERNEL_DIM,
        EMBD_C, TEXT,
        TEXT_ENC_SEQ, EMBD_TEXT,
        EMBD_V,
        EMBD_EXP, EXP_Q_H, EXP_HEAD_DIM, EXP_FFN_HID, EXP_KV_DIM,
        CHUNK_SIZE, MAX_STATE_DIM, EMBD_S, SEQ_T, SEQ_S, PADDING,
        TEXT_VOCAB_SIZE,
    )

    rng = np.random.default_rng(0)
    np.random.seed(0)

    def rand_mat(m, n): return (rng.standard_normal((m, n)) / np.sqrt(m)).astype(np_bfloat16)
    def rand_vec(n):    return rng.standard_normal((n,)).astype(np_bfloat16)

    # --- Params (same as vla.py main) ---
    params_proc = dict(
        kernel=(rng.standard_normal((EMBD_P, CH, KERNEL_DIM, KERNEL_DIM)) / np.sqrt(CH * KERNEL_DIM * KERNEL_DIM)).astype(np_bfloat16)
    )
    params_con = dict(W=rand_mat(EMBD_C, TEXT))
    params_vit = dict(
        Wq=rand_mat(EMBD_V, EMBD_V), Wk=rand_mat(EMBD_V, EMBD_V), Wv=rand_mat(EMBD_V, EMBD_V),
        Wo=rand_mat(EMBD_V, EMBD_V),
        W_up=rand_mat(EMBD_V, 4*EMBD_V), W_down=rand_mat(4*EMBD_V, EMBD_V),
        W_norm_1=rand_vec(EMBD_V), b_norm_1=rand_vec(EMBD_V),
        W_norm_2=rand_vec(EMBD_V), b_norm_2=rand_vec(EMBD_V),
    )
    params_vlm = dict(
        Wq=rand_mat(EMBD_TEXT, 960), Wk=rand_mat(EMBD_TEXT, 320), Wv=rand_mat(EMBD_TEXT, 320),
        Wo=rand_mat(960, EMBD_TEXT),
        W_gate=rand_mat(EMBD_TEXT, 2560), W_up=rand_mat(EMBD_TEXT, 2560), W_down=rand_mat(2560, EMBD_TEXT),
        W_norm_1=rand_vec(EMBD_TEXT), W_norm_2=rand_vec(EMBD_TEXT),
    )
    params_exp_self = dict(
        Wq=rand_mat(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM), Wk=rand_mat(EMBD_EXP, EXP_KV_DIM),
        Wv=rand_mat(EMBD_EXP, EXP_KV_DIM), Wo=rand_mat(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP),
        W_gate=rand_mat(EMBD_EXP, EXP_FFN_HID), W_up=rand_mat(EMBD_EXP, EXP_FFN_HID),
        W_down=rand_mat(EXP_FFN_HID, EMBD_EXP),
        W_norm_1=rand_vec(EMBD_EXP), W_norm_2=rand_vec(EMBD_EXP),
    )
    params_exp_cross = dict(
        Wq=rand_mat(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM),
        Wk_cross=rand_mat(EXP_KV_DIM, EXP_KV_DIM), Wv_cross=rand_mat(EXP_KV_DIM, EXP_KV_DIM),
        Wo=rand_mat(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP),
        W_gate=rand_mat(EMBD_EXP, EXP_FFN_HID), W_up=rand_mat(EMBD_EXP, EXP_FFN_HID),
        W_down=rand_mat(EXP_FFN_HID, EMBD_EXP),
        W_norm_1=rand_vec(EMBD_EXP), W_norm_2=rand_vec(EMBD_EXP),
    )
    params_out = dict(
        W_exp_norm=rand_vec(EMBD_EXP),
        W_action_out=rand_mat(EMBD_EXP, MAX_STATE_DIM),
    )

    # --- Inputs ---
    image_rgb = rng.random((CH, PIX, PIX)).astype(np_bfloat16)
    import torch; torch.manual_seed(0)
    text_emb = create_text_emb(TEXT_VOCAB_SIZE, TEXT, SEQ_T)
    state_input = rand_mat(SEQ_S, MAX_STATE_DIM)
    weights = rand_mat(MAX_STATE_DIM, EMBD_S)
    state_emb = create_state_emb(state_input, weights)
    zeros = np.zeros((PADDING, EMBD_S)).astype(np_bfloat16)
    action = rand_mat(CHUNK_SIZE, EMBD_EXP)

    # ═══════════════════════════════════════════════════════════════════
    # Run each component, capturing NPU times
    # ═══════════════════════════════════════════════════════════════════

    print("Running SmolVLA pipeline, capturing NPU times per component...\n")

    # 1. Preprocessing
    wall, npu_times, conv_emb = capture_npu_times(
        lambda: preprocessing_block(image_rgb, params_proc)
    )
    preproc = {"wall_ms": wall, "npu_us": npu_times, "npu_sum_ms": sum(npu_times)/1000, "count": len(npu_times)}
    print(f"Preprocessing:    wall={wall/1000:.2f}s  npu_sum={preproc['npu_sum_ms']/1000:.3f}s  dispatches={preproc['count']}")

    # 2. Vision Encoder
    wall, npu_times, vision_emb = capture_npu_times(
        lambda: vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)
    )
    vision = {"wall_ms": wall, "npu_us": npu_times, "npu_sum_ms": sum(npu_times)/1000, "count": len(npu_times)}
    print(f"Vision Encoder:   wall={wall/1000:.2f}s  npu_sum={vision['npu_sum_ms']/1000:.3f}s  dispatches={vision['count']}")

    # 3. Connector
    wall, npu_times, llama_emb = capture_npu_times(
        lambda: connector_block(vision_emb, params_con)
    )
    connector = {"wall_ms": wall, "npu_us": npu_times, "npu_sum_ms": sum(npu_times)/1000, "count": len(npu_times)}
    print(f"Connector:        wall={wall/1000:.2f}s  npu_sum={connector['npu_sum_ms']/1000:.3f}s  dispatches={connector['count']}")

    # 4. Joint Transformer
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)
    wall, npu_times, out = capture_npu_times(
        lambda: joint_transformer(LLAMA_NUM_LAYERS, mm_seq, action, params_vlm, params_exp_self, params_exp_cross)
    )
    joint = {"wall_ms": wall, "npu_us": npu_times, "npu_sum_ms": sum(npu_times)/1000, "count": len(npu_times)}
    print(f"Joint Transformer: wall={wall/1000:.2f}s  npu_sum={joint['npu_sum_ms']/1000:.3f}s  dispatches={joint['count']}")

    # 5. Postprocessing
    wall, npu_times, v_t = capture_npu_times(
        lambda: postprocessing(out, params_out)
    )
    post = {"wall_ms": wall, "npu_us": npu_times, "npu_sum_ms": sum(npu_times)/1000, "count": len(npu_times)}
    print(f"Postprocessing:   wall={wall/1000:.2f}s  npu_sum={post['npu_sum_ms']/1000:.3f}s  dispatches={post['count']}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    components = {
        "Preprocessing": preproc,
        "Vision Encoder": vision,
        "Connector": connector,
        "Joint Transformer": joint,
        "Postprocessing": post,
    }

    total_wall = sum(c["wall_ms"] for c in components.values())
    total_npu = sum(c["npu_sum_ms"] for c in components.values())
    total_dispatches = sum(c["count"] for c in components.values())

    print(f"\n{'='*75}")
    print(f"{'Component':<22s} {'Wall (s)':>10s} {'NPU sum (s)':>12s} {'Overhead (s)':>13s} {'Dispatches':>11s} {'NPU%':>6s}")
    print(f"{'='*75}")
    for name, c in components.items():
        wall_s = c["wall_ms"] / 1000
        npu_s = c["npu_sum_ms"] / 1000
        oh_s = wall_s - npu_s
        pct = (npu_s / wall_s * 100) if wall_s > 0 else 0
        print(f"{name:<22s} {wall_s:>9.2f}s {npu_s:>11.3f}s {oh_s:>12.2f}s {c['count']:>11d} {pct:>5.1f}%")
    print(f"{'='*75}")
    wall_s = total_wall / 1000
    npu_s = total_npu / 1000
    oh_s = wall_s - npu_s
    pct = (npu_s / wall_s * 100) if wall_s > 0 else 0
    print(f"{'TOTAL':<22s} {wall_s:>9.2f}s {npu_s:>11.3f}s {oh_s:>12.2f}s {total_dispatches:>11d} {pct:>5.1f}%")

    print(f"\nNote: NPU times are cold first-call times (~1ms each).")
    print(f"Warm steady-state times are ~6x lower (~160μs each).")
    print(f"Warm NPU total estimate: ~{total_npu/6/1000:.1f}s")

    # Save
    import json
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(SCRIPT_DIR, "vla_npu_time_breakdown.json")
    save_data = {}
    for name, c in components.items():
        save_data[name] = {
            "wall_ms": round(c["wall_ms"], 2),
            "npu_sum_ms": round(c["npu_sum_ms"], 2),
            "overhead_ms": round(c["wall_ms"] - c["npu_sum_ms"], 2),
            "dispatches": c["count"],
            "npu_times_us": c["npu_us"],
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nData saved to: {out_path}")


if __name__ == "__main__":
    main()
