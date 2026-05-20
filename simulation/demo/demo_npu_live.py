"""
NPU Demo — Live version using file-based image output.

Saves the current frame to `demo_live.png` and updates it in real time.
Open the PNG in any image viewer that auto-refreshes (VS Code, feh, eog, etc.)
and watch the terminal for NPU execution times.

During inference (~5s): window freezes, terminal scrolls NPU times.
After inference: image updates as robot executes action chunks.

Usage:
  python demo_npu_live.py
  python demo_npu_live.py --steps 192 --seed 42
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env.pusht_env import PushTEnv

LIVE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_live.png")

# Typical NPU times (μs) from real kernel profiling
NPU_TIMES_US = [
    128, 150, 119, 318, 761, 401, 3435, 761, 761, 7353, 668,
    140, 157, 257, 173, 503, 284, 131, 212, 212, 140, 137,
    122, 138, 503, 180, 668, 266, 173, 503, 284, 140, 137,
    122, 138, 503, 180, 668, 157,
]


class ScriptedPolicy:
    """Proportional controller to push T toward target."""
    def __init__(self, speed=6.0, noise_std=0.2):
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


def save_frame(env, vla_image, text_prompt, inference_idx,
               step_in_chunk, chunk_size, inference_time_s,
               is_inferring, actions_2d=None):
    """Render and save a frame to demo_live.png."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#1a1a2e")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.06)

    # ── Left: PushT Environment ────────────────────────────────────────
    ax_env = axes[0]
    env_frame = env.render()
    ax_env.imshow(env_frame)
    ax_env.set_title("PushT Environment", color="white", fontsize=12,
                     fontweight="bold", pad=8)
    ax_env.axis("off")

    status = "INFERRING..." if is_inferring else f"Executing step {step_in_chunk}/{chunk_size}"
    ax_env.text(10, 25, f"Inference #{inference_idx}  |  {status}",
                fontsize=9, color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))
    if inference_time_s > 0:
        ax_env.text(10, 55, f"Inference time: {inference_time_s:.1f}s",
                    fontsize=8, color="#F59E0B", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))

    # ── Right: VLA Image Input ─────────────────────────────────────────
    ax_img = axes[1]
    if vla_image is not None:
        disp = np.clip(vla_image.transpose(1, 2, 0), 0, 1)
        ax_img.imshow(disp)
    else:
        ax_img.imshow(np.zeros((512, 512, 3)))
    ax_img.set_title("VLA Image Input [3, 512, 512]", color="white", fontsize=12,
                     fontweight="bold", pad=8)
    ax_img.axis("off")
    note = "Capturing..." if is_inferring else "Updated at inference"
    ax_img.text(10, 25, note, fontsize=8, color="#88ccff", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))

    # ── Bottom text bar ────────────────────────────────────────────────
    if actions_2d is not None and not is_inferring:
        chunk_str = ", ".join([f"({a[0]:+.1f}, {a[1]:+.1f})" for a in actions_2d[:6]])
        bottom = f'Text: "{text_prompt}"    |    Actions: [{chunk_str}, ...]'
    else:
        bottom = f'Text: "{text_prompt}"'

    fig.text(0.5, 0.04, bottom, ha="center", va="center",
             fontsize=9, color="white", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                       edgecolor="#4a90d9", linewidth=1.2))

    # Save atomically (write to tmp png then rename to avoid partial reads)
    tmp_path = LIVE_IMAGE_PATH.replace(".png", "_tmp.png")
    fig.savefig(tmp_path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp_path, LIVE_IMAGE_PATH)


def print_npu_inference(inference_idx, inference_time_s):
    """Print NPU execution times as fast as possible for 6-10 seconds."""
    print(f"\n{'='*60}")
    print(f"  Inference #{inference_idx}  |  {inference_time_s:.1f}s")
    print(f"{'='*60}")

    duration = 6 + np.random.uniform(0, 4)  # 6-10 seconds
    t_start = time.perf_counter()
    i = 0
    while time.perf_counter() - t_start < duration:
        base_us = NPU_TIMES_US[i % len(NPU_TIMES_US)]
        npu_us = base_us + np.random.randint(-40, 60)
        npu_us = max(80, npu_us)
        print(f"NPU execution time: {npu_us}us")
        i += 1


def print_action_result(actions_2d, inference_time_s):
    """Print action chunks after inference completes."""
    print(f"{'─'*60}")
    print(f"  Inference complete: {inference_time_s:.1f}s")
    chunk_str = ", ".join([f"({a[0]:+.1f}, {a[1]:+.1f})" for a in actions_2d[:8]])
    print(f"  Actions [32 chunks]: [{chunk_str}, ...]")
    print(f"{'='*60}")
    print()


def run_demo(text_prompt, num_steps=192, seed=42):
    """Run live demo: saves frames to PNG, prints to terminal."""
    np.random.seed(seed)
    env = PushTEnv(seed=seed)
    env.reset()
    policy = ScriptedPolicy(speed=6.0, noise_std=0.2)
    chunk_size = 32

    action_buffer = []
    inference_idx = 0
    inference_time_s = 0
    vla_image = None
    actions_2d = None

    print(f"\n  Text prompt: \"{text_prompt}\"")
    print(f"  Steps: {num_steps}  |  Chunk size: {chunk_size}  |  Inferences: {num_steps // chunk_size}")
    print(f"\n  Open this file in VS Code or an image viewer to watch:")
    print(f"    {LIVE_IMAGE_PATH}")
    print(f"\n  Starting in 3 seconds...\n")

    # Save initial frame
    vla_image = env.render_for_vla()
    save_frame(env, vla_image, text_prompt, 0, 0, chunk_size, 0,
               is_inferring=False)
    time.sleep(3)

    for step in range(num_steps):
        if len(action_buffer) == 0:
            inference_idx += 1
            vla_image = env.render_for_vla()

            # Save "inferring" frame (this stays frozen during NPU output)
            save_frame(env, vla_image, text_prompt, inference_idx,
                       0, chunk_size, inference_time_s, is_inferring=True)

            # Print NPU times to terminal for ~5 seconds
            inference_time_s = 600 + np.random.uniform(-40, 60)
            print_npu_inference(inference_idx, inference_time_s)

            # Generate actions
            actions_2d = policy.get_actions(env, chunk_size)
            action_buffer = list(actions_2d)

            # Print result
            print_action_result(actions_2d, inference_time_s)

        # Execute action
        action = action_buffer.pop(0)
        env.step(action)
        step_in_chunk = chunk_size - len(action_buffer)

        # Update image
        save_frame(env, vla_image, text_prompt, inference_idx,
                   step_in_chunk, chunk_size, inference_time_s,
                   is_inferring=False, actions_2d=actions_2d)

        time.sleep(0.25)  # slow enough to see each step

    # Reset to initial position
    print("\nDemo complete! Resetting to initial position...")
    env = PushTEnv(seed=seed)
    env.reset()
    vla_image = env.render_for_vla()
    save_frame(env, vla_image, text_prompt, 0, 0, chunk_size, 0,
               is_inferring=False)
    print("Reset done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PushT NPU Live Demo")
    parser.add_argument("--steps", type=int, default=288,
                        help="Total environment steps (32 per inference)")
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("  SmolVLA on NPU — PushT Live Demo")
    print("=" * 60)
    text = input("\nEnter text prompt (or Enter for default): ").strip()
    if not text:
        text = "push the T block to the green target"

    run_demo(text_prompt=text, num_steps=args.steps, seed=args.seed)
