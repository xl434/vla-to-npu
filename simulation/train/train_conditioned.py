"""
Text-conditioned training for multi-target PushT.

Phase 1 (BC): Behavioral cloning against a scripted expert.
              Trains text_embed, state_encoder, text_enc, exp_self, exp_cross,
              post_norm, post_proj.  Vision backbone frozen (zeros during training).
Phase 2 (RL): REINFORCE fine-tuning — maximises block-to-target reward.

Usage (run from simulation/):
  python train/train_conditioned.py                     # BC 10k steps then RL 3k ep
  python train/train_conditioned.py --bc_steps 20000
  python train/train_conditioned.py --phase rl          # RL only (load latest BC ckpt)
  python train/train_conditioned.py --phase bc          # BC only
  python train/train_conditioned.py --eval              # CPU eval (deterministic)

Checkpoints saved to: simulation/checkpoints/conditioned/
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.abspath(os.path.join(_TRAIN_DIR, ".."))
sys.path.insert(0, _SIM_DIR)

from env.pusht_env import WORKSPACE, make_t_shape
from env.pusht_env_multi import PushTEnvMulti, COMMANDS, COMMAND_PROMPTS, TARGETS
from shapely.geometry import Polygon as ShapelyPolygon
from model.model_conditioned import (
    VLAConditioned, tokenize,
    CHUNK_SIZE, ACTION_DIM, STATE_DIM,
)
from demo.demo_cpu import ScriptedPolicy

# ── Directory & checkpoint helpers ───────────────────────────────────────────

CKPT_DIR = os.path.join(_SIM_DIR, "checkpoints", "conditioned")


def _latest_ckpt(prefix: str = "bc_step"):
    if not os.path.exists(CKPT_DIR):
        return None, 0
    files = [f for f in os.listdir(CKPT_DIR)
             if f.startswith(prefix) and f.endswith(".pt")]
    if not files:
        return None, 0
    files.sort(key=lambda f: int(f.split(prefix)[1].split(".")[0]))
    path = os.path.join(CKPT_DIR, files[-1])
    step = int(files[-1].split(prefix)[1].split(".")[0])
    return path, step


def _save(model: VLAConditioned, optimizer, scheduler, step: int,
          history: list, tag: str = "bc_step"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    state = dict(
        step=step,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler else None,
        history=history,
    )
    torch.save(state, os.path.join(CKPT_DIR, f"{tag}{step}.pt"))
    torch.save(state, os.path.join(CKPT_DIR, "latest.pt"))
    print(f"  → checkpoint saved: {tag}{step}.pt")


def _load(model: VLAConditioned, optimizer=None, scheduler=None,
          path: str = None, prefix: str = "bc_step"):
    if path is None:
        path, _ = _latest_ckpt(prefix)
    if path is None or not os.path.exists(path):
        return 0, []
    ckpt = torch.load(path, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:4]}")
    if optimizer and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
    if scheduler and ckpt.get("scheduler_state"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    step = ckpt.get("step", 0)
    history = ckpt.get("history", [])
    print(f"  loaded checkpoint: {os.path.basename(path)}  (step={step})")
    return step, history


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Behavioral Cloning
# ══════════════════════════════════════════════════════════════════════════════

def _scripted_chunk(env: PushTEnvMulti, policy: ScriptedPolicy) -> np.ndarray:
    """Run scripted policy for CHUNK_SIZE steps. Returns [CHUNK_SIZE, 2] actions."""
    # ScriptedPolicy.get_actions advances the env internally
    actions = policy.get_actions(env, chunk_size=CHUNK_SIZE)   # [CHUNK_SIZE, 2]
    return np.array(actions, dtype=np.float32)


def train_bc(total_steps: int = 10_000,
             lr: float = 3e-4,
             resume: bool = False,
             save_every: int = 1000,
             seed: int = 0,
             noise_std: float = 0.3):
    """BC against scripted expert across all 4 commands."""

    model = VLAConditioned(seed=seed)
    model.freeze_vision()
    model.train()

    optimizer = torch.optim.AdamW(model.trainable_params(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.05
    )

    start_step, loss_history = 0, []
    if resume:
        start_step, loss_history = _load(model, optimizer, scheduler)

    n_trainable = sum(p.numel() for p in model.trainable_params())
    print(f"\nBC training: {total_steps} steps  lr={lr}  trainable={n_trainable:,}")
    print(f"  commands: {COMMANDS}")
    print()

    scripted = ScriptedPolicy(speed=6.0, noise_std=noise_std)
    rng = np.random.default_rng(seed)
    env = PushTEnvMulti()
    t0 = time.perf_counter()

    for step in range(start_step, total_steps):
        # Sample random command + randomized initial state
        cmd = rng.choice(COMMANDS)
        env.reset(command=cmd)
        env.ee_x    = float(rng.uniform(40, WORKSPACE - 40))
        env.ee_y    = float(rng.uniform(40, WORKSPACE - 40))
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-100, 100))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_theta = float(rng.uniform(-0.8, 0.8))

        # Capture state BEFORE scripted policy advances the env
        state7  = env.get_state()                             # [7] float32
        prompt  = COMMAND_PROMPTS[cmd]

        # Ground-truth actions from scripted expert (runs the env forward)
        gt_actions = _scripted_chunk(env, scripted)           # [CHUNK_SIZE, 2]

        tokens = torch.tensor(tokenize(prompt), dtype=torch.long)
        state7_t = torch.tensor(state7, dtype=torch.float32)

        pred = model(tokens, state7_t)                        # [32, 32]
        pred_actions = pred[:, :ACTION_DIM]                   # [32, 2]

        gt = torch.tensor(gt_actions, dtype=torch.float32)    # [32, 2]
        loss = nn.functional.mse_loss(pred_actions, gt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if (step + 1) % 200 == 0:
            recent = np.mean(loss_history[-200:])
            elapsed = time.perf_counter() - t0
            print(f"  step {step+1:6d}/{total_steps}  loss={recent:.4f}  ({elapsed:.0f}s)")

        if (step + 1) % save_every == 0:
            _save(model, optimizer, scheduler, step + 1, loss_history, tag="bc_step")

    model.eval()
    print(f"\nBC complete. Final loss (last 500): {np.mean(loss_history[-500:]):.4f}")
    return model, loss_history


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — REINFORCE RL fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

GAMMA         = 0.95
STEPS_EP      = 256
SUCCESS_PX    = 30      # primary success: block center within this of target
SUCCESS_BONUS = 2.0     # reward bonus when solved
LOG_STD_INIT  = -1.0
LOG_STD_MIN   = -3.0
BASELINE_ALPHA = 0.05


def _overlap_iou(env: PushTEnvMulti) -> float:
    """IoU between block T-shape and target T-shape (0.0 – 1.0)."""
    block_poly  = ShapelyPolygon(make_t_shape(env.block_x,  env.block_y,  env.block_theta))
    target_poly = ShapelyPolygon(make_t_shape(env.target_x, env.target_y, 0.0))
    if not block_poly.is_valid or not target_poly.is_valid:
        return 0.0
    inter = block_poly.intersection(target_poly).area
    union = block_poly.union(target_poly).area
    return float(inter / union) if union > 0 else 0.0


class RLPolicyConditioned(nn.Module):
    """Stochastic wrapper around VLAConditioned for REINFORCE."""

    def __init__(self, model: VLAConditioned):
        super().__init__()
        self.model   = model
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), LOG_STD_INIT))

    def sample(self, prompt: str, state7_np: np.ndarray):
        tokens  = torch.tensor(tokenize(prompt), dtype=torch.long)
        state7  = torch.tensor(state7_np, dtype=torch.float32)

        pred    = self.model(tokens, state7)         # [32, 32]
        mean    = pred[:, :ACTION_DIM]               # [32, 2]
        std     = torch.exp(self.log_std.clamp(min=LOG_STD_MIN))

        noise   = torch.randn_like(mean)
        actions = (mean + noise * std).detach().numpy().astype(np.float32)

        log_prob = (-0.5 * noise.pow(2) - self.log_std
                    - 0.5 * torch.tensor(2 * math.pi).log()).sum()
        entropy  = (self.log_std + 0.5 * torch.tensor(2 * math.pi * math.e).log()).sum()
        return actions, log_prob, entropy


def _step_reward(env: PushTEnvMulti) -> tuple:
    """Returns (reward, solved).

    Dense distance reward (learnable from anywhere) + soft IoU bonus
    (orientation guidance once close) + success bonus.

      distance:  -dist/WORKSPACE          always informative  [-1, 0]
      iou bonus: +0.5 * iou               soft orientation    [0, 0.5]
      bonus:     +2.0 if dist < 30px      strong success spike
    """
    dist   = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
    solved = dist < SUCCESS_PX

    r  = -dist / WORKSPACE          # dense distance signal
    r += 0.5 * _overlap_iou(env)    # soft orientation bonus
    if solved:
        r += SUCCESS_BONUS
    return r, solved


def _rollout(policy: RLPolicyConditioned, env: PushTEnvMulti, prompt: str):
    log_probs, chunk_rewards = [], []
    solved = False
    for _ in range(STEPS_EP // CHUNK_SIZE):
        state7 = env.get_state()
        actions, lp, _ = policy.sample(prompt, state7)

        step_r = []
        for t in range(CHUNK_SIZE):
            env.step(actions[t])
            r, s = _step_reward(env)
            step_r.append(r)
            if s:
                solved = True

        log_probs.append(lp)
        chunk_rewards.append(float(np.mean(step_r)))

    return log_probs, chunk_rewards, solved


def train_rl(total_episodes: int = 3000,
             lr: float = 1e-4,
             entropy_coef: float = 0.01,
             save_every: int = 500,
             seed: int = 0,
             resume: bool = False,
             cmd_weights: dict = None):
    """REINFORCE fine-tuning starting from the latest BC checkpoint."""

    model = VLAConditioned(seed=seed)
    model.freeze_vision()

    # Warm-start from best available checkpoint
    latest_bc = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(latest_bc):
        _load(model, path=latest_bc)
    else:
        print("  [warn] no BC checkpoint found — RL starts from random weights")

    policy = RLPolicyConditioned(model)
    model.train()

    train_params = list(model.trainable_params()) + [policy.log_std]
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_episodes, eta_min=lr * 0.05
    )

    start_ep, reward_history, solved_history = 0, [], []
    # Per-command EMA baselines — up/down have different reward scales
    baselines = {cmd: -0.3 for cmd in COMMANDS}

    # Command sampling weights (default uniform; oversample hard commands)
    if cmd_weights is None:
        cmd_weights = {cmd: 1.0 for cmd in COMMANDS}
    cmd_names = list(cmd_weights.keys())
    cmd_probs  = np.array([cmd_weights[c] for c in cmd_names], dtype=float)
    cmd_probs /= cmd_probs.sum()

    if resume:
        rl_path, _ = _latest_ckpt("rl_ep")
        if rl_path:
            ckpt = torch.load(rl_path, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            policy.log_std.data = ckpt["log_std"]
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_ep = ckpt["episode"]
            # Handle both old (scalar "baseline") and new (dict "baselines") formats
            if "baselines" in ckpt:
                baselines = ckpt["baselines"]
            elif "baseline" in ckpt:
                baselines = {cmd: ckpt["baseline"] for cmd in COMMANDS}
            reward_history = ckpt.get("reward_history", [])
            solved_history = ckpt.get("solved_history", [])
            print(f"  resumed RL from episode {start_ep}")

    n_trainable = sum(p.numel() for p in train_params)
    print(f"\nRL training: {total_episodes} episodes  lr={lr}  trainable={n_trainable:,}")
    print(f"  exploration std = {torch.exp(policy.log_std).detach().numpy()}")
    print()

    rng = np.random.default_rng(seed)
    env = PushTEnvMulti()
    t0  = time.perf_counter()

    for ep in range(start_ep, total_episodes):
        cmd    = rng.choice(cmd_names, p=cmd_probs)
        prompt = COMMAND_PROMPTS[cmd]
        env.reset(command=cmd)
        env.ee_x    = float(rng.uniform(40, WORKSPACE - 40))
        env.ee_y    = float(rng.uniform(40, WORKSPACE - 40))
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-100, 100))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_theta = float(rng.uniform(-0.8, 0.8))

        log_probs, chunk_rewards, solved = _rollout(policy, env, prompt)

        G = 0.0
        returns = []
        for r in reversed(chunk_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Per-command EMA baseline — keeps gradients meaningful across directions
        ep_return = returns_t[0].item()
        baselines[cmd] = ((1 - BASELINE_ALPHA) * baselines[cmd]
                          + BASELINE_ALPHA * ep_return)
        advantages = returns_t - baselines[cmd]
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = sum(-adv * lp for adv, lp in zip(advantages.tolist(), log_probs))
        dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
        _, _, entropy = policy.sample(prompt, dummy_state)
        loss = pg_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(train_params, max_norm=0.5)
        optimizer.step()
        scheduler.step()

        reward_history.append(float(np.sum(chunk_rewards)))
        solved_history.append(int(solved))

        if (ep + 1) % 50 == 0:
            recent_r   = np.mean(reward_history[-50:])
            solve_rate = np.mean(solved_history[-50:])
            std_now    = torch.exp(policy.log_std[0]).item()
            elapsed    = time.perf_counter() - t0
            bl_str     = " ".join(f"{c[0]}:{baselines[c]:.2f}" for c in COMMANDS)
            print(f"  ep {ep+1:5d}/{total_episodes}  "
                  f"reward={recent_r:+.3f}  "
                  f"solved={solve_rate:.0%}  "
                  f"std={std_now:.3f}  "
                  f"bl=[{bl_str}]  "
                  f"({elapsed:.0f}s)  cmd={cmd}")

        if (ep + 1) % save_every == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            ckpt_state = dict(
                episode=ep + 1,
                model_state=model.state_dict(),
                log_std=policy.log_std.data,
                optimizer_state=optimizer.state_dict(),
                baselines=baselines,
                reward_history=reward_history,
                solved_history=solved_history,
            )
            ep_path = os.path.join(CKPT_DIR, f"rl_ep{ep+1}.pt")
            torch.save(ckpt_state, ep_path)
            torch.save(ckpt_state, os.path.join(CKPT_DIR, "latest_rl.pt"))
            # Write model-only checkpoint for demo loading
            model.save(os.path.join(CKPT_DIR, "latest.pt"))
            print(f"  → saved rl checkpoint (ep {ep+1})")

    print(f"\nRL complete: {total_episodes} episodes.")
    solve_rate = np.mean(solved_history[-200:]) if len(solved_history) >= 200 else np.mean(solved_history)
    print(f"  Final solve rate (last 200 eps): {solve_rate:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# CPU evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_cpu(ckpt_path: str = None, num_episodes: int = 20, seed: int = 0):
    """Deterministic CPU evaluation across all 4 commands."""
    model = VLAConditioned.load(ckpt_path or os.path.join(CKPT_DIR, "latest.pt"))

    rng  = np.random.default_rng(seed)
    env  = PushTEnvMulti()
    results = {cmd: [] for cmd in COMMANDS}

    for ep in range(num_episodes):
        cmd    = COMMANDS[ep % len(COMMANDS)]
        prompt = COMMAND_PROMPTS[cmd]
        env.reset(command=cmd)
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-60, 60))

        ep_solved = False
        for _ in range(256 // CHUNK_SIZE):
            state7 = env.get_state()
            actions = model.predict(prompt, state7)   # [32, 2]
            for t in range(CHUNK_SIZE):
                env.step(actions[t])
            dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
            if dist < SUCCESS_PX:
                ep_solved = True

        iou_final  = _overlap_iou(env)
        dist_final = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
        results[cmd].append(ep_solved)
        print(f"  ep {ep+1:3d} [{cmd:5s}] {'SOLVED' if ep_solved else 'failed'} "
              f"(iou={iou_final:.2f}  dist={dist_final:.1f}px)")

    print("\nSolve rates by command:")
    total = 0
    for cmd in COMMANDS:
        r = results[cmd]
        rate = np.mean(r) if r else 0.0
        total += sum(r)
        print(f"  {cmd:5s}: {sum(r)}/{len(r)} = {rate:.0%}")
    print(f"  TOTAL: {total}/{num_episodes} = {total/num_episodes:.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-conditioned VLA training")
    parser.add_argument("--phase",       choices=["bc", "rl", "both"], default="both")
    parser.add_argument("--bc_steps",   type=int,   default=10_000)
    parser.add_argument("--rl_episodes", type=int,   default=3_000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--rl_lr",       type=float, default=1e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--save_every",  type=int,   default=1000)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--oversample_hard", action="store_true",
                        help="Sample up/down 2x more than left/right")
    parser.add_argument("--eval",        action="store_true",
                        help="CPU evaluation of latest checkpoint")
    parser.add_argument("--ckpt",        type=str,   default=None)
    parser.add_argument("--noise_std",   type=float, default=0.3,
                        help="Scripted policy noise during BC")
    args = parser.parse_args()

    if args.eval:
        eval_cpu(args.ckpt)
        sys.exit(0)

    if args.phase in ("bc", "both"):
        train_bc(
            total_steps=args.bc_steps,
            lr=args.lr,
            resume=args.resume,
            save_every=args.save_every,
            seed=args.seed,
            noise_std=args.noise_std,
        )

    if args.phase in ("rl", "both"):
        cmd_weights = ({"up": 2.0, "down": 2.0, "left": 1.0, "right": 1.0}
                       if args.oversample_hard else None)
        train_rl(
            total_episodes=args.rl_episodes,
            lr=args.rl_lr,
            entropy_coef=args.entropy_coef,
            save_every=args.save_every,
            seed=args.seed,
            resume=args.resume and args.phase == "rl",
            cmd_weights=cmd_weights,
        )
