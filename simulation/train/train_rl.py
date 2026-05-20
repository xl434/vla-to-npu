"""
RL fine-tuning for PushT action expert using REINFORCE.

Problem with BC: model copies the average push direction but doesn't learn
to disengage and find a new contact point. RL fixes this by directly
optimizing block-to-target distance.

Training runs on CPU (fast, differentiable). Every --npu_eval_every episodes
the script runs a real NPU validation pass so NPU inference is in the loop.

Architecture:
  state_encoder(5→768) → exp_self → post_norm → post_proj → actions[:, :2]
  Same weights as demo_npu_real.py — training changes carry over directly.

Reward (dense, per step):
  -distance(block, target) / WORKSPACE    in range [-1, 0]
  +2.0 bonus if block within success_px of target (task solved)

Usage:
  cd /home/xl434/vla-to-npu/simulation
  python train/train_rl.py
  python train/train_rl.py --episodes 5000 --lr 3e-4
  python train/train_rl.py --resume
  python train/train_rl.py --eval_npu   # one NPU validation run with latest ckpt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.abspath(os.path.join(_TRAIN_DIR, ".."))
_VLA_DIR   = os.path.abspath(os.path.join(_SIM_DIR, "..", "vla"))

sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, _VLA_DIR)

from env.pusht_env import PushTEnv, WORKSPACE
from demo.demo_cpu import VLACpuRef, CHUNK_SIZE, EXP_EMBD, ACTION_DIM
from train.train import make_state_encoder, load_checkpoint, save_checkpoint, CKPT_DIR

# ── constants ────────────────────────────────────────────────────────────────
GAMMA        = 0.95    # discount across chunks within one episode
STEPS_EP     = 192     # env steps per episode = 6 chunks of 32
SUCCESS_PX   = 30      # pixels — task is "solved" if block within this of target
SUCCESS_BONUS = 2.0    # extra reward per step when solved
LOG_STD_INIT = -1.0    # initial action exploration (std ≈ 0.37)
LOG_STD_MIN  = -3.0    # minimum allowed log_std (std ≈ 0.05)
BASELINE_ALPHA = 0.05  # EMA coefficient for value baseline


# ══════════════════════════════════════════════════════════════════════════════
# Policy
# ══════════════════════════════════════════════════════════════════════════════

class RLPolicy(nn.Module):
    """Wraps the trained action expert as a stochastic policy for REINFORCE."""

    def __init__(self, state_encoder: nn.Sequential, model: VLACpuRef):
        super().__init__()
        self.state_enc = state_encoder
        self.exp_self  = model.exp_self
        self.post_norm = model.post_norm
        self.post_proj = model.post_proj
        self.log_std   = nn.Parameter(torch.full((ACTION_DIM,), LOG_STD_INIT))

    def mean_actions(self, state5: torch.Tensor) -> torch.Tensor:
        """state5 [1, 5] → mean action chunk [32, 2]."""
        emb    = self.state_enc(state5)                                  # [1, 768]
        t_pos  = torch.linspace(0, 1, CHUNK_SIZE, device=state5.device).unsqueeze(1)
        act_in = (emb.expand(CHUNK_SIZE, EXP_EMBD) + t_pos * 0.1).unsqueeze(0)  # [1,32,768]
        act_out = self.exp_self(act_in).squeeze(0)                       # [32, 768]
        return self.post_proj(self.post_norm(act_out))[:, :ACTION_DIM]  # [32, 2]

    def sample(self, state5: torch.Tensor):
        """Sample action chunk and compute log probability.

        Returns:
          actions_np: [32, 2] numpy float32 — for env.step()
          log_prob:   scalar tensor — gradient flows through mean + log_std
        """
        std = torch.exp(self.log_std.clamp(min=LOG_STD_MIN))             # [2]
        mean = self.mean_actions(state5)                                  # [32, 2]

        # Sample WITHOUT reparameterization so env actions are detached,
        # but log_prob retains gradient through mean and log_std.
        noise = torch.randn_like(mean)                                   # [32, 2], no grad
        actions = (mean + noise * std).detach()                          # [32, 2], no grad

        # log N(a; mean, std) = -0.5*(noise/std)^2 - log(std) - 0.5*log(2π)
        log_prob = (-0.5 * noise.pow(2) - self.log_std
                    - 0.5 * torch.tensor(2 * torch.pi).log()).sum()
        entropy  = (self.log_std + 0.5 * torch.tensor(2 * torch.pi * torch.e).log()).sum()

        return actions.numpy().astype(np.float32), log_prob, entropy


# ══════════════════════════════════════════════════════════════════════════════
# Reward
# ══════════════════════════════════════════════════════════════════════════════

def step_reward(env: PushTEnv) -> float:
    dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
    r = -dist / WORKSPACE
    if dist < SUCCESS_PX:
        r += SUCCESS_BONUS
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Rollout (CPU)
# ══════════════════════════════════════════════════════════════════════════════

def rollout_cpu(policy: RLPolicy, env: PushTEnv) -> tuple:
    """Collect one episode.  Returns (log_probs, chunk_rewards, solved)."""
    log_probs, chunk_rewards = [], []
    solved = False

    for _ in range(STEPS_EP // CHUNK_SIZE):
        state = env.get_state()
        s5 = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, 5]

        actions_np, log_prob, entropy = policy.sample(s5)

        step_rewards = []
        for t in range(CHUNK_SIZE):
            env.step(actions_np[t])
            r = step_reward(env)
            step_rewards.append(r)
            dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
            if dist < SUCCESS_PX:
                solved = True

        log_probs.append(log_prob)
        chunk_rewards.append(float(np.mean(step_rewards)))

    return log_probs, chunk_rewards, solved


# ══════════════════════════════════════════════════════════════════════════════
# NPU validation
# ══════════════════════════════════════════════════════════════════════════════

def run_npu_validation(ckpt_path: str, seed: int = 99) -> dict:
    """Save checkpoint → run one demo episode through real NPU → return metrics.

    Imports demo_npu_real lazily to avoid pulling in vla_cpp at module load time.
    """
    print("\n" + "═" * 60)
    print("  NPU Validation — running real inference on hardware")
    print("═" * 60)

    from demo.demo_npu_real import load_weights, npu_inference

    weights = load_weights(ckpt_path)
    env = PushTEnv(seed=seed)
    env.reset()

    total_reward = 0.0
    solved = False
    inference_times = []

    for chunk_idx in range(STEPS_EP // CHUNK_SIZE):
        actions_np, timings = npu_inference(env, weights)
        inference_times.append(timings["total"])

        for t in range(CHUNK_SIZE):
            env.step(actions_np[t])
            r = step_reward(env)
            total_reward += r
            dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
            if dist < SUCCESS_PX:
                solved = True

    mean_npu_time = np.mean(inference_times)
    result = dict(
        total_reward=total_reward,
        solved=solved,
        mean_npu_time_s=mean_npu_time,
        final_block_dist=np.hypot(env.block_x - env.target_x,
                                   env.block_y - env.target_y),
    )

    print(f"\n  NPU result: total_reward={total_reward:.3f}  "
          f"solved={solved}  "
          f"final_dist={result['final_block_dist']:.1f}px  "
          f"avg_inference={mean_npu_time:.2f}s")
    print("═" * 60 + "\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(total_episodes: int = 3000, lr: float = 3e-4,
          resume: bool = False, npu_eval_every: int = 500,
          save_every: int = 500, seed: int = 0,
          entropy_coef: float = 0.01):

    model = VLACpuRef(num_vlm_layers=1, num_vit_layers=1, seed=seed)
    state_encoder = make_state_encoder()

    # Warm-start from best available BC checkpoint
    ckpt_bc = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(ckpt_bc):
        load_checkpoint(model, state_encoder, path=ckpt_bc)
        print(f"Warm-started from BC checkpoint: {ckpt_bc}")
    else:
        print("No BC checkpoint found — starting from random weights")

    policy = RLPolicy(state_encoder, model)

    # Trainable: state_encoder, exp_self, post_norm, post_proj, log_std
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(k in name for k in ["exp_self", "post_norm", "post_proj"]):
            p.requires_grad = True

    train_params = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(state_encoder.parameters()) +
        [policy.log_std]
    )
    n_trainable = sum(p.numel() for p in train_params)
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_episodes, eta_min=lr * 0.05
    )

    # Resume RL checkpoint if available
    rl_ckpt_dir = os.path.join(CKPT_DIR, "rl")
    start_ep = 0
    baseline = -0.3   # initial baseline (typical episode reward)
    reward_history = []
    solved_history = []

    if resume:
        rl_latest = os.path.join(rl_ckpt_dir, "latest_rl.pt")
        if os.path.exists(rl_latest):
            rl_ckpt = torch.load(rl_latest, weights_only=False)
            model.load_state_dict(rl_ckpt["model_state"])
            state_encoder.load_state_dict(rl_ckpt["state_encoder_state"])
            policy.log_std.data = rl_ckpt["log_std"]
            optimizer.load_state_dict(rl_ckpt["optimizer_state"])
            start_ep = rl_ckpt["episode"]
            baseline = rl_ckpt.get("baseline", baseline)
            reward_history = rl_ckpt.get("reward_history", [])
            solved_history = rl_ckpt.get("solved_history", [])
            print(f"Resumed RL from episode {start_ep}")

    print(f"\nRL training: {total_episodes} episodes, lr={lr}, "
          f"trainable={n_trainable:,}, npu_eval_every={npu_eval_every}")
    print(f"  exploration std = {torch.exp(policy.log_std).detach().numpy()}")
    print()

    rng = np.random.default_rng(seed)
    env = PushTEnv(seed=seed)
    t0 = time.perf_counter()

    for ep in range(start_ep, total_episodes):
        # Randomize initial env state for diversity
        env.reset()
        env.ee_x  = float(rng.uniform(40, WORKSPACE - 40))
        env.ee_y  = float(rng.uniform(40, WORKSPACE - 40))
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-100, 100))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_theta = float(rng.uniform(-0.8, 0.8))

        # Collect episode
        log_probs, chunk_rewards, solved = rollout_cpu(policy, env)

        # Discounted returns per chunk
        G = 0.0
        returns = []
        for r in reversed(chunk_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Update EMA baseline
        ep_return = returns_t[0].item()
        baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * ep_return
        advantages = returns_t - baseline
        # Normalize advantages for training stability
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # REINFORCE loss = -Σ_c advantage_c * log_prob_c
        # Entropy bonus encourages exploration
        pg_loss = sum(-adv * lp for adv, lp in zip(advantages.tolist(), log_probs))
        _, _, entropy = policy.sample(torch.zeros(1, 5))  # entropy (same for all states)
        loss = pg_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(train_params, max_norm=0.5)
        optimizer.step()
        scheduler.step()

        reward_history.append(float(np.sum(chunk_rewards)))
        solved_history.append(int(solved))

        # Logging
        if (ep + 1) % 50 == 0:
            recent_r  = np.mean(reward_history[-50:])
            solve_rate = np.mean(solved_history[-50:])
            std_now    = float(torch.exp(policy.log_std[0]))
            elapsed    = time.perf_counter() - t0
            print(f"  ep {ep+1:5d}/{total_episodes}  "
                  f"reward={recent_r:+.3f}  "
                  f"solved={solve_rate:.0%}  "
                  f"std={std_now:.3f}  "
                  f"baseline={baseline:.3f}  "
                  f"({elapsed:.0f}s)")

        # Save checkpoint
        if (ep + 1) % save_every == 0:
            os.makedirs(rl_ckpt_dir, exist_ok=True)
            rl_path = os.path.join(rl_ckpt_dir, f"rl_ep{ep+1}.pt")
            rl_latest_path = os.path.join(rl_ckpt_dir, "latest_rl.pt")
            state = dict(
                episode=ep + 1,
                model_state=model.state_dict(),
                state_encoder_state=state_encoder.state_dict(),
                log_std=policy.log_std.data,
                optimizer_state=optimizer.state_dict(),
                baseline=baseline,
                reward_history=reward_history,
                solved_history=solved_history,
            )
            torch.save(state, rl_path)
            torch.save(state, rl_latest_path)

            # Also write to BC-compatible format so demo_npu_real.py can load it
            bc_path = os.path.join(CKPT_DIR, "rl_latest.pt")
            torch.save(dict(
                step=ep + 1,
                model_state=model.state_dict(),
                state_encoder_state=state_encoder.state_dict(),
                loss_history=reward_history,
            ), bc_path)
            print(f"  → saved rl checkpoint (ep {ep+1})")

        # NPU validation run
        if npu_eval_every > 0 and (ep + 1) % npu_eval_every == 0:
            bc_path = os.path.join(CKPT_DIR, "rl_latest.pt")
            if os.path.exists(bc_path):
                run_npu_validation(bc_path, seed=99 + ep)

    print(f"\nTraining complete: {total_episodes} episodes.")
    recent_r  = np.mean(reward_history[-200:]) if len(reward_history) >= 200 else np.mean(reward_history)
    solve_rate = np.mean(solved_history[-200:]) if len(solved_history) >= 200 else np.mean(solved_history)
    print(f"  Final solve rate (last 200 eps): {solve_rate:.1%}")
    print(f"  Final mean reward:               {recent_r:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def eval_cpu(ckpt_path: str, num_episodes: int = 10, seed: int = 0):
    """Quick CPU evaluation — prints solve rate without running NPU."""
    model = VLACpuRef(num_vlm_layers=1, num_vit_layers=1, seed=0)
    state_encoder = make_state_encoder()

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    state_encoder.load_state_dict(ckpt["state_encoder_state"])

    policy = RLPolicy(state_encoder, model)
    # Use mean actions only (deterministic) for evaluation
    policy.log_std.data.fill_(LOG_STD_MIN)

    rng = np.random.default_rng(seed)
    env = PushTEnv(seed=seed)
    solved = 0

    for ep in range(num_episodes):
        env.reset()
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-60, 60))
        ep_solved = False

        for _ in range(STEPS_EP // CHUNK_SIZE):
            state = env.get_state()
            s5 = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean = policy.mean_actions(s5)
            for t in range(CHUNK_SIZE):
                env.step(mean[t].numpy())
            dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
            if dist < SUCCESS_PX:
                ep_solved = True

        solved += int(ep_solved)
        print(f"  ep {ep+1}: {'SOLVED' if ep_solved else 'failed'} "
              f"(dist={np.hypot(env.block_x-env.target_x, env.block_y-env.target_y):.1f}px)")

    print(f"\nSolve rate: {solved}/{num_episodes} = {solved/num_episodes:.0%}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training for PushT action expert")
    parser.add_argument("--episodes",       type=int,   default=3000)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--entropy_coef",   type=float, default=0.01)
    parser.add_argument("--npu_eval_every", type=int,   default=500,
                        help="Run NPU validation every N episodes (0 to disable)")
    parser.add_argument("--save_every",     type=int,   default=500)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from latest RL checkpoint")
    parser.add_argument("--eval_cpu",       action="store_true",
                        help="CPU eval of latest checkpoint")
    parser.add_argument("--eval_npu",       action="store_true",
                        help="One NPU validation run with latest ckpt")
    parser.add_argument("--ckpt",           type=str,
                        default=os.path.join(CKPT_DIR, "rl_latest.pt"))
    args = parser.parse_args()

    if args.eval_cpu:
        ckpt = args.ckpt if os.path.exists(args.ckpt) else os.path.join(CKPT_DIR, "latest.pt")
        eval_cpu(ckpt)
    elif args.eval_npu:
        ckpt = args.ckpt if os.path.exists(args.ckpt) else os.path.join(CKPT_DIR, "rl_latest.pt")
        run_npu_validation(ckpt)
    else:
        train(
            total_episodes=args.episodes,
            lr=args.lr,
            resume=args.resume,
            npu_eval_every=args.npu_eval_every,
            save_every=args.save_every,
            seed=args.seed,
            entropy_coef=args.entropy_coef,
        )
