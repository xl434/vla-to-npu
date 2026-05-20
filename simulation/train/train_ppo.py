"""
PPO training for text-conditioned multi-target PushT.

Why PPO over REINFORCE:
  - Reuses each rollout for multiple gradient updates (4 epochs × N minibatches)
  - Critic network gives low-variance advantage estimates (vs EMA baseline)
  - Clipped objective prevents destructive large updates
  - Typically 5–10× more sample efficient than REINFORCE for this task

Architecture:
  Actor:  VLAConditioned  (same as train_conditioned.py)
  Critic: small MLP  7D state → 256 → 128 → 1  (scalar value estimate)

Usage (run from simulation/):
  python train/train_ppo.py                        # fresh run
  python train/train_ppo.py --resume               # resume from latest PPO ckpt
  python train/train_ppo.py --eval                 # evaluate latest checkpoint
  python train/train_ppo.py --warmstart            # load VLAConditioned BC weights first

Checkpoints: simulation/checkpoints/conditioned/ppo_latest.pt
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.abspath(os.path.join(_TRAIN_DIR, ".."))
sys.path.insert(0, _SIM_DIR)

from env.pusht_env import WORKSPACE, make_t_shape
from env.pusht_env_multi import PushTEnvMulti, COMMANDS, COMMAND_PROMPTS
from model.model_conditioned import (
    VLAConditioned, tokenize,
    CHUNK_SIZE, ACTION_DIM, STATE_DIM,
)
from shapely.geometry import Polygon as ShapelyPolygon

# ── Hyperparameters ───────────────────────────────────────────────────────────

STEPS_EP      = 256       # env steps per episode
SUCCESS_PX    = 30        # success threshold (pixels)
SUCCESS_BONUS = 2.0

GAMMA         = 0.99      # discount (higher than REINFORCE — critic handles variance)
GAE_LAMBDA    = 0.95      # GAE smoothing (0=pure TD, 1=pure MC)
CLIP_EPS      = 0.2       # PPO clip range
VALUE_COEF    = 0.5       # critic loss weight
ENTROPY_COEF  = 0.01      # entropy bonus weight
MAX_GRAD_NORM = 0.5

PPO_EPOCHS    = 4         # gradient updates per rollout
MINIBATCH     = 8         # chunks per minibatch
ROLLOUT_EPS   = 16        # episodes per rollout batch

CKPT_DIR = os.path.join(_SIM_DIR, "checkpoints", "conditioned")


# ── Reward ────────────────────────────────────────────────────────────────────

def _iou(env: PushTEnvMulti) -> float:
    bp = ShapelyPolygon(make_t_shape(env.block_x, env.block_y, env.block_theta))
    tp = ShapelyPolygon(make_t_shape(env.target_x, env.target_y, 0.0))
    if not bp.is_valid or not tp.is_valid:
        return 0.0
    inter = bp.intersection(tp).area
    union = bp.union(tp).area
    return float(inter / union) if union > 0 else 0.0


def step_reward(env: PushTEnvMulti):
    dist   = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
    solved = dist < SUCCESS_PX
    r = -dist / WORKSPACE + 0.5 * _iou(env)
    if solved:
        r += SUCCESS_BONUS
    return r, solved


# ── Critic (value network) ────────────────────────────────────────────────────

class Critic(nn.Module):
    """Maps (text_tokens, state7) → scalar value estimate."""

    def __init__(self, text_embd: int = 960):
        super().__init__()
        # Simple MLP — takes mean-pooled text embedding + state
        self.text_proj = nn.Linear(text_embd, 64)
        self.net = nn.Sequential(
            nn.Linear(64 + STATE_DIM, 256), nn.Tanh(),
            nn.Linear(256, 128),            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, text_emb_mean: torch.Tensor, state7: torch.Tensor) -> torch.Tensor:
        """text_emb_mean [B, 960], state7 [B, 7] → value [B]"""
        t = F.tanh(self.text_proj(text_emb_mean))
        x = torch.cat([t, state7], dim=-1)
        return self.net(x).squeeze(-1)


# ── Actor wrapper ─────────────────────────────────────────────────────────────

class PPOActor(nn.Module):
    def __init__(self, model: VLAConditioned):
        super().__init__()
        self.model   = model
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def forward(self, tokens: torch.Tensor, state7: torch.Tensor):
        """Returns mean [CHUNK, 2] and current std [2]."""
        pred = self.model(tokens, state7)          # [32, 32]
        mean = pred[:, :ACTION_DIM]                # [32, 2]
        std  = torch.exp(self.log_std.clamp(-3.0, 0.5))
        return mean, std

    def sample(self, tokens: torch.Tensor, state7: torch.Tensor):
        mean, std = self.forward(tokens, state7)
        noise    = torch.randn_like(mean)
        actions  = (mean + noise * std).detach().numpy().astype(np.float32)
        # log N(action; mean, std) = -0.5*noise^2 - log_std - 0.5*log(2π), summed
        log_prob = (-0.5 * noise.pow(2) - self.log_std
                    - 0.5 * math.log(2 * math.pi)).sum()
        return actions, log_prob.detach(), actions

    def log_prob_for_actions(self, tokens: torch.Tensor, state7: torch.Tensor,
                             actions: torch.Tensor) -> torch.Tensor:
        """Recompute log_prob for stored actions under current policy.
        tokens:  [seq_len]          long
        state7:  [7]                float32
        actions: [CHUNK_SIZE, 2]    float32 — the actual actions taken at collection time
        Returns scalar log prob under current (updated) policy mean + std.
        """
        mean, std = self.forward(tokens, state7)       # mean: [CHUNK, 2], std: [2]
        normalized = (actions - mean) / std            # [CHUNK, 2]
        return (-0.5 * normalized.pow(2) - self.log_std
                - 0.5 * math.log(2 * math.pi)).sum()

    def entropy(self) -> torch.Tensor:
        return (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(actor: PPOActor, critic: Critic,
                    rng: np.random.Generator,
                    cmd_probs: np.ndarray, cmd_names: list,
                    n_episodes: int = ROLLOUT_EPS):
    """
    Collect n_episodes, return everything needed for PPO update.

    Returns dict of lists (one entry per chunk across all episodes):
      tokens, state7s, noises, old_log_probs, rewards, values, dones
    Also returns: episode solve rate, mean episode reward
    """
    env = PushTEnvMulti()

    all_tokens     = []
    all_state7s    = []
    all_actions    = []   # actual actions taken (for proper PPO ratio)
    all_log_probs  = []
    all_values     = []
    all_rewards    = []
    all_dones      = []   # 1.0 at last chunk of episode

    ep_rewards, ep_solved = [], []

    for _ in range(n_episodes):
        cmd    = rng.choice(cmd_names, p=cmd_probs)
        prompt = COMMAND_PROMPTS[cmd]
        env.reset(command=cmd)
        env.ee_x    = float(rng.uniform(40, WORKSPACE - 40))
        env.ee_y    = float(rng.uniform(40, WORKSPACE - 40))
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-100, 100))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_theta = float(rng.uniform(-0.8, 0.8))

        tokens_np = tokenize(prompt)
        tokens_t  = torch.tensor(tokens_np, dtype=torch.long)

        ep_r, solved = 0.0, False
        n_chunks = STEPS_EP // CHUNK_SIZE

        for ci in range(n_chunks):
            state7_np = env.get_state()
            state7_t  = torch.tensor(state7_np, dtype=torch.float32)

            with torch.no_grad():
                actions, log_prob, stored_actions = actor.sample(tokens_t, state7_t)

                # Critic value estimate
                text_emb = actor.model.text_embed(tokens_t).float()  # [48, 960]
                text_mean = text_emb.mean(0, keepdim=True)           # [1, 960]
                value = critic(text_mean, state7_t.unsqueeze(0)).item()

            step_r = []
            for t in range(CHUNK_SIZE):
                env.step(actions[t])
                r, s = step_reward(env)
                step_r.append(r)
                if s:
                    solved = True

            chunk_r = float(np.mean(step_r))
            ep_r   += chunk_r

            all_tokens.append(tokens_np)
            all_state7s.append(state7_np)
            all_actions.append(stored_actions)    # [CHUNK_SIZE, 2] float32
            all_log_probs.append(float(log_prob))
            all_values.append(value)
            all_rewards.append(chunk_r)
            all_dones.append(1.0 if ci == n_chunks - 1 else 0.0)

        ep_rewards.append(ep_r)
        ep_solved.append(int(solved))

    return dict(
        tokens    = np.array(all_tokens,    dtype=np.int64),
        state7s   = np.array(all_state7s,   dtype=np.float32),
        actions   = np.array(all_actions,   dtype=np.float32),
        log_probs = np.array(all_log_probs, dtype=np.float32),
        values    = np.array(all_values,    dtype=np.float32),
        rewards   = np.array(all_rewards,   dtype=np.float32),
        dones     = np.array(all_dones,     dtype=np.float32),
    ), np.mean(ep_rewards), np.mean(ep_solved)


# ── GAE advantage computation ─────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized Advantage Estimation.

    Returns advantages and value targets (advantages + values).
    GAE gives lower variance than pure MC returns while keeping some bias.
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    gae = 0.0
    for i in reversed(range(N)):
        next_val = 0.0 if dones[i] else values[i + 1] if i + 1 < N else 0.0
        delta    = rewards[i] + gamma * next_val - values[i]
        gae      = delta + gamma * lam * (1.0 - dones[i]) * gae
        advantages[i] = gae
    returns = advantages + values
    return advantages, returns


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(actor: PPOActor, critic: Critic,
               optimizer: torch.optim.Optimizer,
               rollout: dict, n_epochs: int = PPO_EPOCHS,
               minibatch: int = MINIBATCH):
    """Run n_epochs of PPO updates on the collected rollout."""

    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"]
    )
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    tokens_t    = torch.tensor(rollout["tokens"],    dtype=torch.long)
    state7s_t   = torch.tensor(rollout["state7s"],   dtype=torch.float32)
    actions_t   = torch.tensor(rollout["actions"],   dtype=torch.float32)
    old_lp_t    = torch.tensor(rollout["log_probs"], dtype=torch.float32)
    adv_t       = torch.tensor(advantages,           dtype=torch.float32)
    ret_t       = torch.tensor(returns,              dtype=torch.float32)

    N = len(tokens_t)
    total_pg, total_val, total_ent = 0.0, 0.0, 0.0

    for _ in range(n_epochs):
        idx = np.random.permutation(N)
        for start in range(0, N, minibatch):
            mb = idx[start:start + minibatch]
            if len(mb) == 0:
                continue

            mb_tokens  = tokens_t[mb]
            mb_state7  = state7s_t[mb]
            mb_actions = actions_t[mb]
            mb_old_lp  = old_lp_t[mb]
            mb_adv     = adv_t[mb]
            mb_ret     = ret_t[mb]

            # New log probs using actual actions under current policy mean
            new_lp_list = []
            for i in range(len(mb)):
                lp = actor.log_prob_for_actions(mb_tokens[i], mb_state7[i], mb_actions[i])
                new_lp_list.append(lp)
            new_lp = torch.stack(new_lp_list)

            # PPO clipped policy loss
            ratio    = torch.exp(new_lp - mb_old_lp)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (critic)
            text_means = []
            for i in range(len(mb)):
                emb = actor.model.text_embed(mb_tokens[i]).float().mean(0)
                text_means.append(emb)
            text_mean_t = torch.stack(text_means)
            val_pred = critic(text_mean_t, mb_state7)
            val_loss = F.mse_loss(val_pred, mb_ret)

            # Entropy bonus
            entropy = actor.entropy()

            loss = pg_loss + VALUE_COEF * val_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                MAX_GRAD_NORM
            )
            optimizer.step()

            total_pg  += pg_loss.item()
            total_val += val_loss.item()
            total_ent += entropy.item()

    n_updates = n_epochs * max(1, N // minibatch)
    return total_pg / n_updates, total_val / n_updates, total_ent / n_updates


# ── Training loop ─────────────────────────────────────────────────────────────

def train(total_rollouts: int = 500,
          lr: float = 3e-4,
          seed: int = 0,
          resume: bool = False,
          warmstart: bool = True,
          oversample_hard: bool = True,
          save_every: int = 50):

    # ── Build actor + critic ──────────────────────────────────────────────────
    model = VLAConditioned(seed=seed)
    model.freeze_vision()

    if warmstart:
        bc_path = os.path.join(CKPT_DIR, "latest.pt")
        if os.path.exists(bc_path):
            ckpt = torch.load(bc_path, weights_only=False)
            missing, _ = model.load_state_dict(ckpt["model_state"], strict=False)
            print(f"  warm-started from {bc_path}")
            if missing:
                print(f"  missing keys (expected for critic): {missing[:3]}")
        else:
            print("  [warn] no BC checkpoint found, starting from random")

    actor  = PPOActor(model)
    critic = Critic(text_embd=960)
    model.train()

    all_params = (list(actor.parameters()) + list(critic.parameters()))
    optimizer  = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_rollouts, eta_min=lr * 0.1
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_rollout  = 0
    reward_history = []
    solved_history = []

    ppo_latest = os.path.join(CKPT_DIR, "ppo_latest.pt")
    if resume and os.path.exists(ppo_latest):
        ckpt = torch.load(ppo_latest, weights_only=False)
        actor.model.load_state_dict(ckpt["model_state"])
        actor.log_std.data = ckpt["log_std"]
        critic.load_state_dict(ckpt["critic_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_rollout  = ckpt["rollout"]
        reward_history = ckpt.get("reward_history", [])
        solved_history = ckpt.get("solved_history", [])
        print(f"  resumed from rollout {start_rollout}")

    # ── Command sampling ──────────────────────────────────────────────────────
    if oversample_hard:
        cmd_w = {"up": 2.0, "down": 2.0, "left": 1.0, "right": 1.0}
    else:
        cmd_w = {cmd: 1.0 for cmd in COMMANDS}
    cmd_names = list(cmd_w.keys())
    cmd_probs = np.array([cmd_w[c] for c in cmd_names], dtype=float)
    cmd_probs /= cmd_probs.sum()

    n_trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"\nPPO training: {total_rollouts} rollouts × {ROLLOUT_EPS} eps "
          f"({total_rollouts * ROLLOUT_EPS} total episodes)")
    print(f"  trainable params: {n_trainable:,}  lr={lr}")
    print(f"  PPO epochs={PPO_EPOCHS}  minibatch={MINIBATCH}  clip={CLIP_EPS}")
    print(f"  oversample_hard={oversample_hard}")
    print()

    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    for rollout_idx in range(start_rollout, total_rollouts):
        # Collect rollout
        rollout, mean_r, solve_rate = collect_rollout(
            actor, critic, rng, cmd_probs, cmd_names, n_episodes=ROLLOUT_EPS
        )

        # PPO update
        pg_loss, val_loss, entropy = ppo_update(
            actor, critic, optimizer, rollout
        )
        scheduler.step()

        reward_history.append(float(mean_r))
        solved_history.append(float(solve_rate))

        if (rollout_idx + 1) % 5 == 0:
            recent_r    = np.mean(reward_history[-10:])
            recent_solv = np.mean(solved_history[-10:])
            std_now     = torch.exp(actor.log_std[0]).item()
            elapsed     = time.perf_counter() - t0
            episodes_so_far = (rollout_idx + 1) * ROLLOUT_EPS
            print(f"  rollout {rollout_idx+1:4d}/{total_rollouts}  "
                  f"ep={episodes_so_far:6d}  "
                  f"reward={recent_r:+.3f}  "
                  f"solved={recent_solv:.0%}  "
                  f"pg={pg_loss:.3f}  val={val_loss:.3f}  "
                  f"std={std_now:.3f}  "
                  f"({elapsed:.0f}s)")

        if (rollout_idx + 1) % save_every == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(dict(
                rollout        = rollout_idx + 1,
                model_state    = actor.model.state_dict(),
                log_std        = actor.log_std.data,
                critic_state   = critic.state_dict(),
                optimizer_state= optimizer.state_dict(),
                reward_history = reward_history,
                solved_history = solved_history,
            ), ppo_latest)
            # Also write VLAConditioned-compatible checkpoint for demo/eval
            actor.model.save(os.path.join(CKPT_DIR, "latest.pt"))
            print(f"  → checkpoint saved (rollout {rollout_idx+1})")

    print(f"\nPPO training complete.")
    if solved_history:
        print(f"  Final solve rate (last 20 rollouts): "
              f"{np.mean(solved_history[-20:]):.1%}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_cpu(ckpt_path: str = None, num_episodes: int = 20, seed: int = 0):
    path = ckpt_path or os.path.join(CKPT_DIR, "latest.pt")
    model = VLAConditioned.load(path)

    rng = np.random.default_rng(seed)
    env = PushTEnvMulti()
    results = {cmd: [] for cmd in COMMANDS}

    for ep in range(num_episodes):
        cmd    = COMMANDS[ep % 4]
        prompt = COMMAND_PROMPTS[cmd]
        env.reset(command=cmd)
        env.block_x = WORKSPACE / 2 + float(rng.uniform(-80, 80))
        env.block_y = WORKSPACE / 2 + float(rng.uniform(-60, 60))

        ep_solved = False
        for _ in range(STEPS_EP // CHUNK_SIZE):
            state7  = env.get_state()
            actions = model.predict(prompt, state7)
            for t in range(CHUNK_SIZE):
                env.step(actions[t])
            dist = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
            if dist < SUCCESS_PX:
                ep_solved = True

        iou_f  = _iou(env)
        dist_f = np.hypot(env.block_x - env.target_x, env.block_y - env.target_y)
        results[cmd].append(ep_solved)
        print(f"  ep {ep+1:3d} [{cmd:5s}] {'SOLVED' if ep_solved else 'failed'} "
              f"(iou={iou_f:.2f}  dist={dist_f:.1f}px)")

    print("\nSolve rates by command:")
    total = 0
    for cmd in COMMANDS:
        r = results[cmd]
        total += sum(r)
        print(f"  {cmd:5s}: {sum(r)}/{len(r)} = {np.mean(r):.0%}")
    print(f"  TOTAL: {total}/{num_episodes} = {total/num_episodes:.0%}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for text-conditioned VLA")
    parser.add_argument("--rollouts",        type=int,   default=500,
                        help="Number of rollout batches (each = 16 episodes)")
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--warmstart",       action="store_true", default=True,
                        help="Load VLAConditioned BC weights before PPO")
    parser.add_argument("--no_warmstart",    action="store_false", dest="warmstart")
    parser.add_argument("--oversample_hard", action="store_true", default=True)
    parser.add_argument("--save_every",      type=int,   default=50)
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--ckpt",            type=str,   default=None)
    args = parser.parse_args()

    if args.eval:
        eval_cpu(args.ckpt)
    else:
        train(
            total_rollouts  = args.rollouts,
            lr              = args.lr,
            seed            = args.seed,
            resume          = args.resume,
            warmstart       = args.warmstart,
            oversample_hard = args.oversample_hard,
            save_every      = args.save_every,
        )
