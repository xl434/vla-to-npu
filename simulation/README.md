# PushT Simulation — SmolVLA Training & Demo

2D simulation of the SmolVLA-Mini pipeline on a PushT task (push a T-shaped block to a target
with a circular end-effector). Used to train the model on CPU and validate NPU inference.

---

## Directory layout

```
simulation/
  env/
    pusht_env.py          single-target PushT (state: ee_x, ee_y, block_x, block_y, block_theta)
    pusht_env_multi.py    4-target extension (commands: up/down/left/right)
  model/
    model.py              SmolVLAMini — matches vla/vla.py architecture exactly (42.7M params)
    model_conditioned.py  VLAConditioned — text-conditioned variant (23.7M trainable)
    test_model.py         44 unit tests for shapes, numerics, weight sharing, export round-trip
  train/
    train.py              Phase 1: BC on single-target PushT with scripted expert
    train_conditioned.py  Phase 2: BC + REINFORCE on 4-target text-conditioned PushT
    train_ppo.py          Phase 3: PPO on text-conditioned PushT (more sample-efficient)
    train_rl.py           Standalone REINFORCE RL fine-tuning for the basic model
  demo/
    demo_cpu.py           CPU inference demo (no NPU), includes VLACpuRef and ScriptedPolicy
    demo_npu.py           NPU inference demo
    demo_npu_live.py      Live NPU demo with matplotlib animation
    demo_npu_final.py     Final polished NPU demo
  utils/
    show_actions.py       Visualize action chunk predictions
  checkpoints/            Saved weights — not committed (8.5 GB, retrain from scratch)
  outputs/                Generated GIFs, PNGs, training logs — not committed
  PLAN.md                 Original implementation plan (6 stages)
  PROGRESS.md             Detailed progress log with bugs found/fixed
  README.md               This file
```

---

## Architecture

Matches `vla/vla.py` exactly so CPU-trained weights load directly onto the NPU.

| Component | Details |
|---|---|
| Preprocessing | `Conv2d(3, 768, kernel=16, stride=16)` → `[1024, 768]` |
| ViT | 1 layer, EMBD=768, 12 heads, FFN=3072, LayerNorm |
| Connector | Pixel-shuffle + linear → `[64, 960]` |
| Text embedding | `Embedding(256, 960)`, character-level, no external tokenizer |
| Text encoder | 1 block (weight-shared × 2), GQA Q_H=15/KV_H=5, RoPE, SwiGLU, EMBD=960 |
| Action expert | Self-attn + cross-attn (weight-shared), EMBD=768, SEQ=32 |
| Action queries | Learned `[32, 768]` parameter (replaces zero init — prevents dead gradients) |
| Postprocessing | RMSNorm(768) + Linear(768→32) |
| **Total** | **42.7M params, 23.2M trainable** |

**Frozen:** conv, ViT, connector (random weights — vision features are meaningless but fixed)  
**Trained:** text_emb, text_enc, action_queries, exp_self, exp_cross, state_w, post_norm, post_proj

Action output `[32, 32]`: 32-step chunk, first 2 dims used as `(dx, dy)` end-effector deltas.

---

## Dependencies

```bash
pip install torch numpy matplotlib shapely
```

No CUDA needed — all training runs on CPU. No external tokenizer or HuggingFace dependencies.

---

## Training from scratch

All training scripts run from `simulation/`. Checkpoints are saved to `simulation/checkpoints/`.

### Phase 1 — Basic behavioral cloning (single-target, `train.py`)

Trains the action expert + a small state encoder `5→256→512→768` against a scripted
proportional controller. Vision and text backbone stay frozen (random weights).

```bash
cd /home/xl434/vla-to-npu/simulation

# Standard run: 10k steps, ~30 min
python train/train.py --steps 10000 --batch 32 --lr 1e-3

# Longer run for better convergence (we ran to 50k)
python train/train.py --steps 50000 --batch 32 --lr 1e-3

# Resume from latest checkpoint
python train/train.py --resume --steps 50000

# Check progress without running
python train/train.py --progress

# Quick eval
python train/train.py --eval
```

**Convergence:** loss drops from ~20 → ~3 in the first 1k steps, then slowly to ~0.8–1.2 by 50k.
Checkpoints saved every 1000 steps to `checkpoints/ckpt_stepN.pt` and `checkpoints/latest.pt`.

---

### Phase 2 — Text-conditioned BC + REINFORCE (`train_conditioned.py`)

Extends to 4 named targets (up/down/left/right). Trains the full model including text encoder.
Two sub-phases run back to back by default.

```bash
cd /home/xl434/vla-to-npu/simulation

# Full run: BC 10k steps then REINFORCE 3k episodes (~2–3 hours)
python train/train_conditioned.py

# BC only, longer run
python train/train_conditioned.py --phase bc --bc_steps 20000

# REINFORCE only (loads latest BC checkpoint)
python train/train_conditioned.py --phase rl --rl_episodes 10000

# Resume RL from where it left off
python train/train_conditioned.py --phase rl --resume

# Evaluate latest checkpoint (CPU, deterministic)
python train/train_conditioned.py --eval
```

**BC hyperparameters:** lr=3e-4, cosine annealing, weight_decay=1e-5, no batching (1 sample/step).  
**RL hyperparameters:** lr=1e-4, entropy_coef=0.01, GAMMA=0.95, EMA baseline per command.  
**Reward:** `−dist/WORKSPACE` (dense) + `0.5 × IoU` (orientation) + `2.0` bonus on solve.

Checkpoints saved to `checkpoints/conditioned/bc_stepN.pt` and `latest.pt`.

**Results we achieved:** 40% overall solve rate (20 episodes, all 4 directions) after BC 15k + RL 30k.

```
up  : 40%   down: 20%   left: 40%   right: 60%   TOTAL: 40%
```

---

### Phase 3 — PPO fine-tuning (`train_ppo.py`)

PPO with a value network (critic) for lower-variance advantage estimates. Typically 5–10× more
sample-efficient than REINFORCE. Warm-starts from the latest BC checkpoint.

```bash
cd /home/xl434/vla-to-npu/simulation

# Standard run: 500 rollouts × 16 episodes = 8k episodes total
python train/train_ppo.py

# Resume PPO from latest PPO checkpoint
python train/train_ppo.py --resume

# Start without BC warm-start (random weights)
python train/train_ppo.py --no_warmstart

# Evaluate
python train/train_ppo.py --eval
```

**PPO hyperparameters:** GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2, VALUE_COEF=0.5,
ENTROPY_COEF=0.01, PPO_EPOCHS=4, MINIBATCH=8, ROLLOUT_EPS=16.  
Checkpoints to `checkpoints/conditioned/ppo_latest.pt`.

**Recommended pipeline:**
```bash
# Step 1: BC warm-start
python train/train_conditioned.py --phase bc --bc_steps 15000

# Step 2: PPO fine-tuning
python train/train_ppo.py --warmstart --rollouts 500
```

---

## Running demos

```bash
cd /home/xl434/vla-to-npu/simulation

# CPU demo (no NPU required)
python demo/demo_cpu.py --checkpoint checkpoints/latest.pt --steps 256

# NPU demo (requires compiled kernels in vla/)
python demo/demo_npu.py

# Live NPU demo with matplotlib animation
python demo/demo_npu_live.py
```

---

## Evaluating a checkpoint

```bash
cd /home/xl434/vla-to-npu/simulation

# Evaluate conditioned model (20 episodes, all 4 commands)
python train/train_conditioned.py --eval

# Evaluate PPO checkpoint
python train/train_ppo.py --eval --ckpt checkpoints/conditioned/ppo_latest.pt
```

---

## Environment details

| Property | Value |
|---|---|
| Workspace | 512 × 512 pixels |
| End-effector | Red circle, radius=15 |
| T-block | Blue polygon (80×30 bar + stem), contact physics |
| Target (single) | Green dashed T at `(256, 400)` |
| Targets (multi) | up `(256, 130)`, down `(256, 400)`, left `(128, 256)`, right `(384, 256)` |
| Physics | Push coeff=0.15, friction=0.7, collision resolution |
| State (single) | `[ee_x, ee_y, block_x, block_y, block_theta]` normalized to `[0, 1]`, dim=5 |
| State (multi) | Same + `[target_x, target_y]`, dim=7 |
| Success threshold | `dist(block_center, target_center) < 30 px` |
| Text commands | `"push the block up/down/left/right"` |

---

## Training tips

- **Start with BC before RL/PPO.** RL from random weights rarely converges on this task.
- **BC 10–15k steps is enough** to warm-start RL. More BC doesn't help much — RL takes over.
- **PPO > REINFORCE** for sample efficiency. If time-limited, use `train_ppo.py`.
- **Oversample hard directions.** `--oversample_hard` doubles the sampling weight for up/down
  (harder to reach than left/right). This is default in PPO and optional in REINFORCE.
- **Don't train vision.** Conv and ViT are frozen with random weights throughout. The state
  encoder (direct state→768 MLP) is the shortcut that makes training tractable. When real
  SmolVLA weights are available, remove the state encoder and train the full pipeline end-to-end.
- **Checkpoint early and often.** Default is every 1000 BC steps or 500 RL episodes. Decrease
  `--save_every` if the server might restart.

---

## Notes on checkpoints

Checkpoints are ~220 MB each (full model + optimizer state) and are not committed to git.
To retrain from scratch, run Phase 2 (conditioned BC + RL) which takes ~3–5 hours on CPU.
The full 50k basic training run took several hours and reached a final loss of ~0.8.
