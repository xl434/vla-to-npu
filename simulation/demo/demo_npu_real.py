"""
Real NPU Demo — actual inference on NPU with trained checkpoint weights.

Loads the trained action expert + state encoder from checkpoint, runs the
full NPU pipeline (~5s/inference), and drives the PushT environment in a
closed loop with actions coming directly from NPU output.

Architecture:
  1. CPU: state_encoder(env_state [5]) → action_queries [32, 768]
  2. NPU: image → preprocessing → vision (1L) → connector → text_encoder (1L)
                action_queries → action_expert_self (trained) → [32, 768]
             → postprocessing (trained) → [32, 32]
  3. actions = output[:, :2]  (first 2 dims = dx, dy)

Updates simulation/demo_live.png in real time.
Open it in VS Code (Image Preview extension) to watch the live demo.

Usage:
  cd /home/xl434/vla-to-npu
  python simulation/demo/demo_npu_real.py
  python simulation/demo/demo_npu_real.py --steps 96 --seed 42
  python simulation/demo/demo_npu_real.py --ckpt simulation/checkpoints/ckpt_step10000.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from ml_dtypes import bfloat16 as np_bfloat16

# ── path setup ──────────────────────────────────────────────────────────────
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR  = os.path.abspath(os.path.join(_DEMO_DIR, ".."))
_VLA_DIR  = os.path.abspath(os.path.join(_SIM_DIR, "..", "vla"))

sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, _VLA_DIR)

from env.pusht_env import PushTEnv
import vla_cpp as cpp
from vla_standalone import postprocessing

# ── constants ────────────────────────────────────────────────────────────────
VIT_NUM_LAYERS = 1
CHUNK_SIZE     = 32
EXP_EMBD       = 768
TEXT_EMBD      = 960
STATE_DIM      = 32
PADDING        = 15

LIVE_IMAGE_PATH = os.path.join(_SIM_DIR, "demo_live.png")
CKPT_DEFAULT    = os.path.join(_SIM_DIR, "checkpoints", "latest.pt")


# ══════════════════════════════════════════════════════════════════════════════
# Weight loading
# ══════════════════════════════════════════════════════════════════════════════

def _b(t: torch.Tensor) -> np.ndarray:
    """Tensor → bf16 numpy, no transpose."""
    return t.detach().float().numpy().astype(np_bfloat16)


def _bT(t: torch.Tensor) -> np.ndarray:
    """2D tensor → bf16 numpy, transposed: PyTorch [out, in] → NPU [in, out]."""
    return t.detach().float().numpy().T.astype(np_bfloat16)


def load_weights(ckpt_path: str):
    """Load checkpoint and return all NPU weight dicts + CPU state encoder."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    sd = ckpt["model_state"]
    step = ckpt.get("step", "?")
    loss_hist = ckpt.get("loss_history", [])
    best = min(loss_hist) if loss_hist else float("nan")
    print(f"  step={step}  best_loss={best:.4f}")

    # Preprocessing  conv.weight [768, 3, 16, 16] — same layout as NPU kernel
    params_proc = dict(kernel=_b(sd["conv.weight"]))

    # ViT — split combined in_proj_weight [2304, 768] → Wq/Wk/Wv each [768, 768]
    inproj = sd["vit.attn.in_proj_weight"]
    params_vit = dict(
        Wq      = _bT(inproj[:768, :]),
        Wk      = _bT(inproj[768:1536, :]),
        Wv      = _bT(inproj[1536:, :]),
        Wo      = _bT(sd["vit.attn.out_proj.weight"]),
        W_up    = _bT(sd["vit.ffn_up.weight"]),
        W_down  = _bT(sd["vit.ffn_down.weight"]),
        W_norm_1= _b(sd["vit.ln_1.weight"]),
        b_norm_1= _b(sd["vit.ln_1.bias"]),
        W_norm_2= _b(sd["vit.ln_2.weight"]),
        b_norm_2= _b(sd["vit.ln_2.bias"]),
    )

    # Connector  connector_w [12288, 960] is already [in, out]
    params_con = dict(W=_b(sd["connector_w"]))

    # Text encoder (frozen in training — random init, but consistent with CPU model)
    params_vlm = dict(
        Wq      = _bT(sd["text_enc.q_proj.weight"]),    # [960, 960]
        Wk      = _bT(sd["text_enc.k_proj.weight"]),    # [960, 320]
        Wv      = _bT(sd["text_enc.v_proj.weight"]),    # [960, 320]
        Wo      = _bT(sd["text_enc.o_proj.weight"]),    # [960, 960]
        W_gate  = _bT(sd["text_enc.gate_proj.weight"]), # [960, 2560]
        W_up    = _bT(sd["text_enc.up_proj.weight"]),   # [960, 2560]
        W_down  = _bT(sd["text_enc.down_proj.weight"]), # [2560, 960]
        W_norm_1= _b(sd["text_enc.ln_1.weight"]),
        W_norm_2= _b(sd["text_enc.ln_2.weight"]),
    )

    # Action expert self  (TRAINED — behavioral cloning target)
    params_exp_self = dict(
        Wq      = _bT(sd["exp_self.q_proj.weight"]),    # [768, 960]
        Wk      = _bT(sd["exp_self.k_proj.weight"]),    # [768, 320]
        Wv      = _bT(sd["exp_self.v_proj.weight"]),    # [768, 320]
        Wo      = _bT(sd["exp_self.o_proj.weight"]),    # [960, 768]
        W_gate  = _bT(sd["exp_self.gate_proj.weight"]), # [768, 2048]
        W_up    = _bT(sd["exp_self.up_proj.weight"]),   # [768, 2048]
        W_down  = _bT(sd["exp_self.down_proj.weight"]), # [2048, 768]
        W_norm_1= _b(sd["exp_self.ln_1.weight"]),
        W_norm_2= _b(sd["exp_self.ln_2.weight"]),
    )

    # Postprocessing  (TRAINED)
    params_out = dict(
        W_exp_norm  = _b(sd["post_norm.weight"]),
        W_action_out= _bT(sd["post_proj.weight"]),      # [768, 32]
    )

    # Text embedding [48, 960] and state projection [32, 960] — keep as numpy
    text_emb_bf16 = _b(sd["text_emb"])                  # [48, 960] bf16
    state_w_f32   = sd["state_w"].float().numpy()        # [32, 960] float32 for CPU matmul

    # CPU state encoder  (TRAINED — maps env state → action queries)
    state_encoder = nn.Sequential(
        nn.Linear(5, 256), nn.SiLU(),
        nn.Linear(256, 512), nn.SiLU(),
        nn.Linear(512, EXP_EMBD),
    )
    state_encoder.load_state_dict(ckpt["state_encoder_state"])
    state_encoder.eval()

    print("  All weights loaded and converted to BF16.")
    return (params_proc, params_vit, params_con, params_vlm,
            params_exp_self, params_out, text_emb_bf16, state_w_f32,
            state_encoder)


# ══════════════════════════════════════════════════════════════════════════════
# NPU inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _action_queries(state_encoder: nn.Sequential, env_state: np.ndarray) -> np.ndarray:
    """env_state [5] → action queries [32, 768] bf16, matching BC training path."""
    s   = torch.tensor(env_state, dtype=torch.float32).unsqueeze(0)  # [1, 5]
    emb = state_encoder(s)                                            # [1, 768]
    t_pos = torch.linspace(0, 1, CHUNK_SIZE).unsqueeze(1)            # [32, 1]
    return (emb.expand(CHUNK_SIZE, EXP_EMBD) + t_pos * 0.1).numpy().astype(np_bfloat16)


_NPU_ERROR_PATTERNS = ("Timer expired", "No such device", "exit -6", "xrt_core")


def _reset_npu_driver() -> bool:
    """Unload and reload the amdxdna kernel module to clear a stuck NPU.

    Runs sudo modprobe without capturing output so the password prompt is
    visible in the terminal.  Returns True if the device is fully ready.
    """
    import subprocess
    print("\n  *** NPU hardware error detected — attempting driver reset ***")
    print("  Running: sudo modprobe -r amdxdna")
    r = subprocess.run(["sudo", "modprobe", "-r", "amdxdna"])
    if r.returncode != 0:
        print("  ERROR: modprobe -r failed. Run manually and retry.")
        return False

    print("  Waiting 3s for driver to unload...")
    time.sleep(3)

    print("  Running: sudo modprobe amdxdna")
    r = subprocess.run(["sudo", "modprobe", "amdxdna"])
    if r.returncode != 0:
        print("  ERROR: modprobe amdxdna failed. Run manually and retry.")
        return False

    # Poll until XRT firmware is truly ready (not just the device node present).
    # /dev/accel/accel0 reappearing is necessary but not sufficient — the firmware
    # load is async and the IOCTL will return EPIPE until it completes.
    print("  Waiting for NPU firmware to initialize (up to 20s)...", flush=True)
    deadline = time.perf_counter() + 20.0
    ready = False
    while time.perf_counter() < deadline:
        time.sleep(1)
        if not os.path.exists("/dev/accel/accel0"):
            continue
        # Use xrt-smi to confirm XRT can talk to the device.
        probe = subprocess.run(
            ["xrt-smi", "examine"],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            ready = True
            break
        print("  . ", end="", flush=True)

    print()
    if ready:
        print("  NPU driver reset OK — device is responsive.\n")
        return True
    elif os.path.exists("/dev/accel/accel0"):
        # xrt-smi unavailable or kept failing; give it one more fixed sleep.
        print("  xrt-smi check failed; waiting an extra 5s and trying anyway...")
        time.sleep(5)
        return True
    else:
        print("  WARNING: /dev/accel/accel0 still missing after reload.")
        return False


def _is_npu_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return any(pat in msg for pat in _NPU_ERROR_PATTERNS)


def npu_inference_with_retry(env: PushTEnv, weights, max_resets: int = 1) -> tuple:
    """Call npu_inference, resetting the driver once on hardware errors."""
    for attempt in range(max_resets + 1):
        try:
            return npu_inference(env, weights)
        except RuntimeError as exc:
            if attempt < max_resets and _is_npu_error(exc):
                print(f"\n  NPU error: {str(exc)[:120]}")
                if _reset_npu_driver():
                    print(f"  Retrying inference...\n")
                    continue
            raise


def npu_inference(env: PushTEnv, weights) -> tuple:
    """Run one full NPU inference.  Returns ([32, 2] float32 actions, timings dict)."""
    (params_proc, params_vit, params_con, params_vlm,
     params_exp_self, params_out, text_emb_bf16, state_w_f32, state_encoder) = weights

    env_state = env.get_state()  # [5] normalized

    # State embedding for multimodal sequence (CPU matmul, [1, 32] @ [32, 960])
    state_pad = np.zeros(STATE_DIM, dtype=np.float32)
    state_pad[:5] = env_state
    state_emb = (state_pad[None] @ state_w_f32).astype(np_bfloat16)  # [1, 960]

    # Action queries from trained CPU state encoder
    action_q = _action_queries(state_encoder, env_state)  # [32, 768] bf16

    # Image from environment
    image_bf16 = env.render_for_vla().astype(np_bfloat16)  # [3, 512, 512]

    timings = {}
    t_start = time.perf_counter()

    print("  Preprocessing ...", end="", flush=True)
    t0 = time.perf_counter()
    conv_emb = cpp.preprocessing_block(image_bf16, params_proc)     # [1024, 768]
    t1 = time.perf_counter()
    timings["preprocessing"] = t1 - t0
    print(f" {timings['preprocessing']:.3f}s")

    print("  Vision encoder ...", end="", flush=True)
    vision_emb = cpp.vision_block(conv_emb, params_vit)             # [1024, 768]
    t2 = time.perf_counter()
    timings["vision"] = t2 - t1
    print(f" {timings['vision']:.3f}s")

    print("  Connector ...", end="", flush=True)
    connector_emb = cpp.connector_block(vision_emb, params_con)     # [64, 960]
    t3 = time.perf_counter()
    timings["connector"] = t3 - t2
    print(f" {timings['connector']:.3f}s")

    # Assemble multimodal sequence [128, 960]
    zeros  = np.zeros((PADDING, TEXT_EMBD), dtype=np_bfloat16)
    mm_seq = np.concatenate([connector_emb, text_emb_bf16, state_emb, zeros], axis=0)
    assert mm_seq.shape == (128, 960)

    print("  Text encoder + action expert ...", end="", flush=True)
    # 1 transformer layer: text self-attn, then action expert self-attn (trained)
    cpp.text_encoder_forward(mm_seq, params_vlm)                     # context pass
    action_out = cpp.action_expert_self_forward(action_q, params_exp_self)  # [32, 768]
    t4 = time.perf_counter()
    timings["transformer"] = t4 - t3
    print(f" {timings['transformer']:.3f}s")

    print("  Postprocessing ...", end="", flush=True)
    v_t = postprocessing(action_out, params_out)                     # [32, 32] bf16
    t5 = time.perf_counter()
    timings["postprocessing"] = t5 - t4
    print(f" {timings['postprocessing']:.3f}s")

    timings["total"] = t5 - t_start
    actions = v_t.astype(np.float32)[:, :2]                         # [32, 2] (dx, dy)
    return actions, timings


# ══════════════════════════════════════════════════════════════════════════════
# Live PNG visualization
# ══════════════════════════════════════════════════════════════════════════════

def save_frame(env, vla_image, text_prompt, inference_idx,
               step_in_chunk, chunk_size, inference_time_s,
               is_inferring, timings=None, actions_2d=None):
    """Render and atomically save a frame to demo_live.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#1a1a2e")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.13, wspace=0.06)

    # ── Left: PushT environment ──────────────────────────────────────────────
    ax_env = axes[0]
    env_frame = env.render()
    ax_env.imshow(env_frame)
    ax_env.set_title("PushT Environment", color="white", fontsize=12,
                     fontweight="bold", pad=8)
    ax_env.axis("off")

    status = "INFERRING ON NPU..." if is_inferring else f"Executing step {step_in_chunk}/{chunk_size}"
    status_col = "#F59E0B" if is_inferring else "white"
    ax_env.text(10, 25, f"Inference #{inference_idx}  |  {status}",
                fontsize=9, color=status_col, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))
    if inference_time_s > 0:
        ax_env.text(10, 55, f"NPU inference: {inference_time_s:.2f}s",
                    fontsize=8, color="#7ee787", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))

    # ── Right: VLA image input ───────────────────────────────────────────────
    ax_img = axes[1]
    if vla_image is not None:
        disp = np.clip(vla_image.transpose(1, 2, 0), 0, 1)
        ax_img.imshow(disp)
    else:
        ax_img.imshow(np.zeros((512, 512, 3)))
    ax_img.set_title("VLA Image Input  [3 × 512 × 512]", color="white", fontsize=12,
                     fontweight="bold", pad=8)
    ax_img.axis("off")
    note = "Capturing..." if is_inferring else "Captured at inference"
    ax_img.text(10, 25, note, fontsize=8, color="#88ccff", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000bb", edgecolor="none"))

    # ── Timing breakdown (if available) ─────────────────────────────────────
    if timings and not is_inferring:
        timing_str = (
            f"prep {timings['preprocessing']:.3f}s  "
            f"vit {timings['vision']:.3f}s  "
            f"conn {timings['connector']:.3f}s  "
            f"tfm {timings['transformer']:.3f}s  "
            f"post {timings['postprocessing']:.3f}s"
        )
        fig.text(0.5, 0.085, timing_str, ha="center", va="center",
                 fontsize=8, color="#8b949e", fontfamily="monospace")

    # ── Bottom text bar ──────────────────────────────────────────────────────
    if actions_2d is not None and not is_inferring:
        chunk_str = ", ".join([f"({a[0]:+.2f},{a[1]:+.2f})" for a in actions_2d[:5]])
        bottom = f'Prompt: "{text_prompt}"    |    NPU actions: [{chunk_str}, ...]'
    else:
        bottom = f'Prompt: "{text_prompt}"'

    fig.text(0.5, 0.04, bottom, ha="center", va="center",
             fontsize=9, color="white", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                       edgecolor="#4a90d9", linewidth=1.2))

    tmp = LIVE_IMAGE_PATH.replace(".png", "_tmp.png")
    fig.savefig(tmp, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, LIVE_IMAGE_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Main demo loop
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(ckpt_path: str, text_prompt: str, num_steps: int, seed: int):
    weights = load_weights(ckpt_path)

    env = PushTEnv(seed=seed)
    state = env.reset()
    np.random.seed(seed)

    action_buffer = []
    inference_idx = 0
    inference_time_s = 0.0
    last_timings = {}
    vla_image = None
    actions_2d = None

    print(f"\n  Text prompt : \"{text_prompt}\"")
    print(f"  Steps       : {num_steps}  ({num_steps // CHUNK_SIZE} inferences)")
    print(f"  Output PNG  : {LIVE_IMAGE_PATH}")
    print(f"\n  Open {LIVE_IMAGE_PATH} in VS Code (Image Preview) to watch.")
    print(f"\n  Starting in 3 seconds...\n")

    vla_image = env.render_for_vla()
    save_frame(env, vla_image, text_prompt, 0, 0, CHUNK_SIZE, 0, is_inferring=False)
    time.sleep(3)

    for step in range(num_steps):
        if len(action_buffer) == 0:
            inference_idx += 1
            vla_image = env.render_for_vla()

            print(f"\n{'='*60}")
            print(f"  Inference #{inference_idx}  |  Running NPU pipeline...")
            print(f"{'='*60}")

            # Show "inferring" frame while NPU runs
            save_frame(env, vla_image, text_prompt, inference_idx,
                       0, CHUNK_SIZE, inference_time_s, is_inferring=True)

            # Real NPU inference (retries once with driver reset on hardware errors)
            actions_2d, last_timings = npu_inference_with_retry(env, weights)
            inference_time_s = last_timings["total"]
            action_buffer = list(actions_2d)

            print(f"{'─'*60}")
            print(f"  Total: {inference_time_s:.3f}s")
            chunk_str = ", ".join([f"({a[0]:+.2f},{a[1]:+.2f})" for a in actions_2d[:6]])
            print(f"  NPU actions [32×(dx,dy)]: [{chunk_str}, ...]")
            print(f"{'='*60}\n")

        # Execute action
        action = action_buffer.pop(0)
        state = env.step(action)
        step_in_chunk = CHUNK_SIZE - len(action_buffer)

        save_frame(env, vla_image, text_prompt, inference_idx,
                   step_in_chunk, CHUNK_SIZE, inference_time_s,
                   is_inferring=False, timings=last_timings, actions_2d=actions_2d)

        time.sleep(0.15)

    print("\nDemo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PushT Real NPU Demo")
    parser.add_argument("--steps", type=int, default=192,
                        help="Total environment steps (default 192 = 6 inferences)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=CKPT_DEFAULT,
                        help="Path to checkpoint .pt file")
    args = parser.parse_args()

    print("=" * 60)
    print("  SmolVLA on NPU — PushT Real Inference Demo")
    print("=" * 60)
    text = input("\nEnter text prompt (or Enter for default): ").strip()
    if not text:
        text = "push the T block to the green target"
    print()

    run_demo(
        ckpt_path=args.ckpt,
        text_prompt=text,
        num_steps=args.steps,
        seed=args.seed,
    )
