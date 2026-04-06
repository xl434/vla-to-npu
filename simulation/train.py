"""
Background training for VLA PushT demo.

Trains the full VLA model (1 layer each: ViT, text encoder, action expert)
via behavioral cloning against a scripted expert policy.

Saves checkpoints every N steps. Resumable.

Usage:
    python train.py                          # train 10k steps
    python train.py --steps 20000            # more steps
    python train.py --resume                 # resume from latest checkpoint
    python train.py --eval                   # quick eval of latest checkpoint

Checkpoints saved to: simulation/checkpoints/
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pusht_env import PushTEnv, WORKSPACE

# Import model + policy from demo
from demo_cpu import (
    VLACpuRef, ScriptedPolicy,
    CHUNK_SIZE, EXP_EMBD, ACTION_DIM, EXP_SEQ,
)

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None, 0
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda f: int(f.split("_step")[1].split(".")[0]))
    latest = os.path.join(CKPT_DIR, ckpts[-1])
    step = int(ckpts[-1].split("_step")[1].split(".")[0])
    return latest, step


def save_checkpoint(model, state_encoder, optimizer, scheduler, step, loss_history):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"ckpt_step{step}.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "state_encoder_state": state_encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "loss_history": loss_history,
    }, path)
    # Also save as "latest" for easy loading
    latest_path = os.path.join(CKPT_DIR, "latest.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "state_encoder_state": state_encoder.state_dict(),
        "loss_history": loss_history,
    }, latest_path)
    print(f"  checkpoint saved: {path}")


def load_checkpoint(model, state_encoder, optimizer=None, scheduler=None, path=None):
    if path is None:
        path, _ = get_latest_checkpoint()
    if path is None:
        return 0, []
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    state_encoder.load_state_dict(ckpt["state_encoder_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    step = ckpt["step"]
    loss_history = ckpt.get("loss_history", [])
    print(f"  resumed from step {step} (loss history: {len(loss_history)} entries)")
    return step, loss_history


def make_state_encoder():
    return nn.Sequential(
        nn.Linear(5, 256),
        nn.SiLU(),
        nn.Linear(256, 512),
        nn.SiLU(),
        nn.Linear(512, EXP_EMBD),
    )


def train(total_steps=10000, lr=1e-3, batch_size=32, save_every=1000,
          log_every=100, resume=False, seed=0):

    print(f"Setting up VLA model (1 layer each)...")
    model = VLACpuRef(num_vlm_layers=1, num_vit_layers=1, seed=seed)
    state_encoder = make_state_encoder()
    policy = ScriptedPolicy(speed=6.0, noise_std=0.3)
    rng = np.random.default_rng(seed)

    # Freeze everything except action expert + postprocessing
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(k in name for k in ["exp_self", "post_norm", "post_proj"]):
            p.requires_grad = True

    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    train_params += list(state_encoder.parameters())
    trainable = sum(p.numel() for p in train_params)
    print(f"  trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )

    start_step = 0
    loss_history = []

    if resume:
        start_step, loss_history = load_checkpoint(
            model, state_encoder, optimizer, scheduler
        )
        if start_step >= total_steps:
            print(f"Already at step {start_step} >= {total_steps}, nothing to do.")
            return

    model.train()
    t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(batch_size):
            env = PushTEnv(seed=int(rng.integers(0, 1_000_000)))
            env.reset()
            # Diverse positions
            env.ee_x = rng.uniform(30, WORKSPACE - 30)
            env.ee_y = rng.uniform(30, WORKSPACE - 30)
            env.block_x = WORKSPACE / 2 + rng.uniform(-100, 100)
            env.block_y = WORKSPACE / 2 + rng.uniform(-100, 100)
            env.block_theta = rng.uniform(-0.5, 0.5)

            target = torch.tensor(policy.get_actions(env))  # [32, 2]
            state = torch.tensor(env.get_state()).unsqueeze(0)  # [1, 5]

            act_emb = state_encoder(state)  # [1, 768]
            t_pos = torch.linspace(0, 1, CHUNK_SIZE).unsqueeze(1)
            act_input = (act_emb.expand(CHUNK_SIZE, EXP_EMBD) + t_pos * 0.1).unsqueeze(0)

            act_out = model.exp_self(act_input)
            pred = model.post_proj(model.post_norm(act_out.squeeze(0)))
            pred_2d = pred[:, :ACTION_DIM]

            loss = nn.functional.mse_loss(pred_2d, target) / batch_size
            loss.backward()
            total_loss += loss.item()

        nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss)

        if (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            recent = np.mean(loss_history[-log_every:])
            best = min(loss_history)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step+1}/{total_steps}  loss: {recent:.4f}  "
                  f"best: {best:.4f}  lr: {lr_now:.6f}  "
                  f"({elapsed:.0f}s elapsed, {elapsed/(step+1-start_step):.2f}s/step)")

        if (step + 1) % save_every == 0:
            save_checkpoint(model, state_encoder, optimizer, scheduler,
                            step + 1, loss_history)

    # Final save
    save_checkpoint(model, state_encoder, optimizer, scheduler,
                    total_steps, loss_history)

    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete: {total_steps} steps in {elapsed:.0f}s")
    print(f"  final loss (avg last 100): {np.mean(loss_history[-100:]):.4f}")
    print(f"  best loss: {min(loss_history):.4f}")


def evaluate(num_episodes=3, steps_per_ep=256):
    """Quick evaluation: load latest checkpoint and run demo."""
    path = os.path.join(CKPT_DIR, "latest.pt")
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return

    print(f"Loading checkpoint from {path}...")
    model = VLACpuRef(num_vlm_layers=1, num_vit_layers=1, seed=0)
    state_encoder = make_state_encoder()
    load_checkpoint(model, state_encoder, path=path)
    model.eval()
    model._state_encoder = state_encoder

    from demo_cpu import run_demo_with_model
    run_demo_with_model(model, num_steps=steps_per_ep, output_path="pusht_eval.mp4")


def show_progress():
    """Print training progress from latest checkpoint."""
    path = os.path.join(CKPT_DIR, "latest.pt")
    if not os.path.exists(path):
        print("No checkpoint found.")
        return
    ckpt = torch.load(path, weights_only=False)
    step = ckpt["step"]
    history = ckpt.get("loss_history", [])
    print(f"Training progress: step {step}")
    if history:
        print(f"  latest loss: {history[-1]:.4f}")
        print(f"  best loss:   {min(history):.4f}")
        # Show loss at intervals
        milestones = [100, 500, 1000, 2000, 5000, 10000]
        for m in milestones:
            if m <= len(history):
                avg = np.mean(history[max(0, m-50):m])
                print(f"  step {m}: {avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLA for PushT")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.progress:
        show_progress()
    elif args.eval:
        evaluate()
    else:
        train(
            total_steps=args.steps,
            lr=args.lr,
            batch_size=args.batch,
            save_every=args.save_every,
            resume=args.resume,
            seed=args.seed,
        )
