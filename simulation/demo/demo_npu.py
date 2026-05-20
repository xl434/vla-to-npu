"""
NPU Demo: Fake VLA inference on NPU with PushT visualization.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  PushT Environment        │  VLA Image Input             │
  │  (live simulation)        │  (what model sees)           │
  │                           │                              │
  ├───────────────────────────┴──────────────────────────────┤
  │  Text: "push the T block to the green target"           │
  ├─────────────────────────────────────────────────────────-┤
  │  NPU Terminal Log                                        │
  │  > NPU execution time: 1432us                            │
  │  > NPU execution time: 891us                             │
  │  > ...                                                   │
  │  > Action chunks: [(3.2, -1.4), (2.8, -0.9), ...]      │
  └──────────────────────────────────────────────────────────┘

Usage:
  cd simulation && python demo_npu.py
  cd simulation && python demo_npu.py --steps 256 --output npu_demo.gif
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env.pusht_env import PushTEnv


# ═══════════════════════════════════════════════════════════════════════
# Fake NPU kernel dispatch data (from real measurements)
# ═══════════════════════════════════════════════════════════════════════

# Kernel names and typical NPU times (μs) from profile mode
KERNEL_SEQUENCE = [
    # Preprocessing (sampled — real has 14592 calls)
    ("conv_bf16",           128),
    ("add_32x32_bf16",      150),
    ("copy_reshape",        119),
    # Vision encoder
    ("layer_norm_bf16",     318),
    ("gemm_qkv_768x768",   761),
    ("attn_score_1024x64",  401),
    ("softmax_f32",        3435),
    ("gemm_value_1024x64",  761),
    ("gemm_out_768x768",    761),
    ("gelu_bf16",          7353),
    ("gemm_ffn_3072x768",   668),
    # Connector
    ("pixel_shuffle_copy",  140),
    ("gemm_conn_64x960",    157),
    ("add_64x64_accum",     257),
    # Text encoder
    ("rms_norm_960_bf16",   173),
    ("gemm_q_128x960",      503),
    ("gemm_kv_128x320",     284),
    ("rope_radians_f32",    131),
    ("rope_sin_f32",        212),
    ("rope_cos_f32",        212),
    ("rope_mul_f32",        140),
    ("gemm_score_128x128",  137),
    ("masked_softmax_bf16", 122),
    ("gemm_value_128x64",   138),
    ("gemm_out_128x960",    503),
    ("silu_160_bf16",       180),
    ("gemm_ffn_up_2560",    668),
    ("gemm_ffn_down_320",   266),
    # Action expert
    ("rms_norm_768_bf16",   173),
    ("gemm_q_32x960",       503),
    ("gemm_kv_32x320",      284),
    ("rope_fused_bf16",     140),
    ("gemm_score_32x32",    137),
    ("softmax_32x128_bf16", 122),
    ("gemm_value_32x64",    138),
    ("gemm_out_32x768",     503),
    ("silu_128_bf16",       180),
    ("gemm_ffn_2048",       668),
    # Postprocessing
    ("rms_norm_final",      173),
    ("gemm_proj_768x32",    157),
]


def generate_npu_log_lines(inference_idx, total_time_s):
    """Generate realistic NPU log lines for one inference.
    Distributes kernel calls to fill the total_time_s budget."""
    lines = []
    n_repeats = max(1, int(total_time_s * 1000 / 28))  # ~28ms per dispatch
    n_repeats = min(n_repeats, 800)  # cap for display

    for i in range(n_repeats):
        name, base_us = KERNEL_SEQUENCE[i % len(KERNEL_SEQUENCE)]
        # Add noise
        npu_us = base_us + np.random.randint(-50, 80)
        npu_us = max(80, npu_us)
        lines.append(f"NPU execution time: {npu_us}us  [{name}]")
    return lines


class ScriptedPolicy:
    """Proportional controller to push T toward target."""
    def __init__(self, speed=6.0, noise_std=0.3):
        self.speed = speed
        self.noise_std = noise_std

    def get_actions(self, env, chunk_size=32):
        actions = []
        ee = np.array([env.ee_x, env.ee_y])
        block = np.array([env.block_x, env.block_y])
        target = np.array([env.target_x, env.target_y])

        for _ in range(chunk_size):
            b2t = target - block
            dist = np.linalg.norm(b2t)
            if dist < 1.0:
                action = np.random.randn(2) * 0.5
            else:
                push_dir = b2t / dist
                approach = block - push_dir * 40
                if np.linalg.norm(approach - ee) > 20:
                    d = approach - ee
                    action = d / max(np.linalg.norm(d), 1e-6) * self.speed
                else:
                    action = push_dir * self.speed
            action += np.random.randn(2) * self.noise_std
            actions.append(action)
            ee = np.clip(ee + action, 15, 497)
            if np.linalg.norm(ee - block) < 45:
                block += action * 0.15
        return np.array(actions, dtype=np.float32)


def render_demo_frame(env, vla_image, text_prompt, log_lines, log_scroll,
                      inference_idx, step_in_chunk, chunk_size,
                      inference_time_s, is_inferring, actions_2d=None):
    """Render a full demo frame with all panels."""
    fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")

    # Layout: 2 rows, 2 columns
    # Top-left: PushT env
    # Top-right: VLA input image
    # Middle: text prompt bar
    # Bottom: NPU terminal log

    gs = fig.add_gridspec(3, 2, height_ratios=[5, 0.6, 3],
                          hspace=0.15, wspace=0.08,
                          left=0.03, right=0.97, top=0.95, bottom=0.03)

    # ── Top-left: PushT Environment ────────────────────────────────────
    ax_env = fig.add_subplot(gs[0, 0])
    env_frame = env.render()
    ax_env.imshow(env_frame)
    ax_env.set_title("PushT Environment (Live)", color="white", fontsize=11,
                     fontweight="bold", pad=8)
    ax_env.axis("off")

    # Overlay step info
    status = "INFERRING..." if is_inferring else f"Executing step {step_in_chunk}/{chunk_size}"
    ax_env.text(10, 20, f"Inference #{inference_idx}  |  {status}",
                fontsize=8, color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa", edgecolor="none"))
    if inference_time_s > 0:
        ax_env.text(10, 45, f"Last inference: {inference_time_s:.1f}s",
                    fontsize=8, color="#F59E0B", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa", edgecolor="none"))

    # ── Top-right: VLA Image Input ─────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 1])
    if vla_image is not None:
        # CHW → HWC for display
        if vla_image.ndim == 3 and vla_image.shape[0] == 3:
            disp = np.clip(vla_image.transpose(1, 2, 0), 0, 1)
        else:
            disp = np.clip(vla_image, 0, 1)
        ax_img.imshow(disp)
    else:
        ax_img.imshow(np.zeros((512, 512, 3)))
    ax_img.set_title("VLA Image Input [3, 512, 512]", color="white", fontsize=11,
                     fontweight="bold", pad=8)
    ax_img.axis("off")
    ax_img.text(10, 20, "Updated each inference",
                fontsize=8, color="#88ccff", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa", edgecolor="none"))

    # ── Middle: Text prompt bar ────────────────────────────────────────
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.set_facecolor("#16213e")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    ax_text.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.1), 0.98, 0.8, boxstyle="round,pad=0.02",
        facecolor="#16213e", edgecolor="#4a90d9", linewidth=1.5))
    ax_text.text(0.02, 0.5, "Text Input: ", fontsize=10, color="#4a90d9",
                 fontweight="bold", va="center", fontfamily="monospace")
    ax_text.text(0.12, 0.5, f'"{text_prompt}"', fontsize=10, color="white",
                 va="center", fontfamily="monospace")

    # ── Bottom: NPU Terminal Log ───────────────────────────────────────
    ax_log = fig.add_subplot(gs[2, :])
    ax_log.set_facecolor("#0a0a0a")
    ax_log.set_xlim(0, 1)
    ax_log.set_ylim(0, 1)
    ax_log.axis("off")
    ax_log.add_patch(mpatches.FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96, boxstyle="round,pad=0.01",
        facecolor="#0d1117", edgecolor="#30363d", linewidth=1))

    ax_log.text(0.01, 0.97, "NPU Terminal", fontsize=9, color="#58a6ff",
                fontweight="bold", va="top", fontfamily="monospace")

    # Show last N log lines
    max_lines = 10
    visible = log_lines[max(0, log_scroll - max_lines):log_scroll]

    for i, line in enumerate(visible):
        y = 0.87 - i * 0.08
        if y < 0.05:
            break
        color = "#7ee787" if "Action" in line else "#c9d1d9"
        if "Total" in line:
            color = "#F59E0B"
        ax_log.text(0.02, y, f"> {line}", fontsize=7.5, color=color,
                    va="top", fontfamily="monospace")

    # Show action chunks if available
    if actions_2d is not None and not is_inferring:
        chunk_str = "  ".join([f"({a[0]:+.1f}, {a[1]:+.1f})" for a in actions_2d[:8]])
        ax_log.text(0.02, 0.05, f"Actions: [{chunk_str} ...]",
                    fontsize=7, color="#d2a8ff", va="bottom", fontfamily="monospace")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    frame = buf
    plt.close(fig)
    return frame


def run_demo(text_prompt, num_steps=192, output_path="npu_demo.gif", seed=42):
    """Run the fake NPU demo."""
    np.random.seed(seed)
    env = PushTEnv(seed=seed)
    state = env.reset()
    policy = ScriptedPolicy(speed=6.0, noise_std=0.2)

    chunk_size = 32
    frames = []
    action_buffer = []
    all_log_lines = []
    inference_idx = 0
    inference_time_s = 0
    vla_image = None
    actions_2d = None

    print(f"\nGenerating NPU demo: {num_steps} steps...")
    print(f"Text prompt: \"{text_prompt}\"")
    print()

    for step in range(num_steps):
        if len(action_buffer) == 0:
            inference_idx += 1
            vla_image = env.render_for_vla()

            # Fake inference time (~580-680s with randomness)
            inference_time_s = 600 + np.random.uniform(-40, 60)

            # Generate NPU log lines
            npu_lines = generate_npu_log_lines(inference_idx, inference_time_s)
            all_log_lines.append(f"")
            all_log_lines.append(f"═══ Inference #{inference_idx} ═══")
            all_log_lines.append(f"Loading image [3, 512, 512] bf16...")
            all_log_lines.append(f"Text: \"{text_prompt}\"")
            all_log_lines.extend(npu_lines[-15:])  # show last 15 kernel calls

            # Generate real actions from scripted policy
            actions_2d = policy.get_actions(env, chunk_size)
            action_buffer = list(actions_2d)

            total_npu_ms = sum(
                KERNEL_SEQUENCE[i % len(KERNEL_SEQUENCE)][1]
                for i in range(min(800, int(inference_time_s * 1000 / 28)))
            ) / 1000

            all_log_lines.append(f"Total wall-clock: {inference_time_s:.1f}s  |  NPU compute: {total_npu_ms:.1f}ms  |  21,783 dispatches")
            chunk_str = "  ".join([f"({a[0]:+.1f}, {a[1]:+.1f})" for a in actions_2d[:6]])
            all_log_lines.append(f"Action chunks [32×(dx,dy)]: [{chunk_str} ...]")

            # Render "inferring" frame
            frame = render_demo_frame(
                env, vla_image, text_prompt, all_log_lines, len(all_log_lines),
                inference_idx, 0, chunk_size, inference_time_s,
                is_inferring=True, actions_2d=actions_2d)
            frames.append(frame)

            print(f"  Inference #{inference_idx}: {inference_time_s:.1f}s (fake), "
                  f"{len(action_buffer)} actions generated")

        # Execute action
        action = action_buffer.pop(0)
        state = env.step(action)
        step_in_chunk = chunk_size - len(action_buffer)

        # Render execution frame
        frame = render_demo_frame(
            env, vla_image, text_prompt, all_log_lines, len(all_log_lines),
            inference_idx, step_in_chunk, chunk_size, inference_time_s,
            is_inferring=False, actions_2d=actions_2d)
        frames.append(frame)

    # Save
    print(f"\nSaving {len(frames)} frames to {output_path}...")
    imgs = [Image.fromarray(f) for f in frames]
    if output_path.endswith(".gif"):
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                     duration=80, loop=0)
    else:
        import matplotlib.animation as animation
        fig_out, ax_out = plt.subplots(figsize=(14, 9))
        ax_out.axis("off")
        im = ax_out.imshow(frames[0])
        def update(i):
            im.set_data(frames[i])
            return [im]
        ani = animation.FuncAnimation(fig_out, update, frames=len(frames), blit=True)
        ani.save(output_path, writer="ffmpeg", fps=12)
        plt.close(fig_out)

    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PushT NPU Demo (Fake)")
    parser.add_argument("--steps", type=int, default=192,
                        help="Total environment steps (32 per inference)")
    parser.add_argument("--output", type=str, default="npu_demo.gif")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Prompt user for text input
    print("=" * 60)
    print("  SmolVLA on NPU — PushT Demo")
    print("=" * 60)
    text = input("\nEnter text prompt (or press Enter for default): ").strip()
    if not text:
        text = "push the T block to the green target"
    print()

    run_demo(text_prompt=text, num_steps=args.steps,
             output_path=args.output, seed=args.seed)
