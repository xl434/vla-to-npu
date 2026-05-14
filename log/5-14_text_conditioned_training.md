# Text-Conditioned VLA — Implementation & Training Notes
**Date:** 2026-05-14  
**Context:** Adding text instructions ("push the block up/down/left/right") to the NPU-accelerated VLA demo.

---

## The Big Picture

We have a robot arm in a 2D simulation called **PushT**. The robot needs to push a T-shaped block to a target zone. Previously the model always aimed for the same target. Now we want to give it a natural language command — *"push the block left"* — and have it understand and execute that.

The full pipeline is:

```
Text command  ──────────────────────────────────────┐
                                                     ▼
Camera image  →  Vision (ViT)  →  Connector  →  Transformer  →  32 (dx, dy) actions
                                                     ▲
Robot state (position, block pos, target pos) ───────┘
```

Each "action" is a small (dx, dy) displacement for the robot's end-effector. We predict 32 of them at once (called a **chunk**), execute them one by one, then predict the next 32. This repeats for 192 steps total (6 chunks × 32).

---

## Why We Need Training

The pretrained VLA was trained on robot manipulation data with image + text. But:

1. The original model never saw PushT — it doesn't know how to push blocks.
2. We simplified the model significantly for NPU (1 transformer layer instead of many), so the text conditioning weights need to be re-learned from scratch.
3. The model needs to connect the word "left" to "move the block leftward" in this specific environment.

---

## The Environment: PushT

`PushTEnvMulti` is a 2D physics simulation. The state is 7 numbers:

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | End-effector X (robot arm tip) | 0–1 (normalized) |
| 1 | End-effector Y | 0–1 |
| 2 | Block X | 0–1 |
| 3 | Block Y | 0–1 |
| 4 | Block rotation angle | −1 to 1 |
| 5 | Target X | 0–1 |
| 6 | Target Y | 0–1 |

The 4 target zones (in pixel space, workspace = 512px):

| Command | Target position | Prompt |
|---------|-----------------|--------|
| `up`    | (256, 102) — top center | "push the block up" |
| `down`  | (256, 400) — bottom center | "push the block down" |
| `left`  | (102, 256) — left center | "push the block left" |
| `right` | (410, 256) — right center | "push the block right" |

---

## The Model: VLAConditioned

The trainable model (`model/model_conditioned.py`) has three new components compared to the original VLA:

### 1. Text Embedding (`text_embed`)
- Converts a text command into numbers the transformer can process.
- We use **character-level tokenization**: each character becomes its ASCII code (0–255). "push the block up" → `[112, 117, 115, 104, ...]`.
- A learnable lookup table (`nn.Embedding(256, 960)`) maps each character code to a 960-dimensional vector.
- The full text sequence is padded to 48 characters → shape `[48, 960]`.
- This is much simpler than full word-piece tokenization (like GPT), but sufficient for 4 short commands.

### 2. 7D State Encoder
- A small neural network: `7 → 256 → 512 → 768` with SiLU activations.
- Converts the 7D environment state into a 768D vector — one per time step in the action chunk.
- Expands to `[32, 768]` (one embedding per action in the chunk) with a small linear position offset.
- This becomes the **action queries**: what the transformer "asks" about when deciding each action.

### 3. Two-Layer Transformer
The transformer processes everything in two passes:

**Layer 0 (self-attention):**
- Text encoder processes `[vision(64) + text(48) + state(1) + padding(15)] = [128, 960]`
- Action expert self-attention over `[32, 768]` — actions "talk to each other"

**Layer 1 (cross-attention = text conditioning):**
- Text encoder runs again on layer-0 output → produces Key and Value tensors `[128, 320]`
- Action expert **cross-attends** to those K/V tensors
- This is where the text actually influences the actions: each action query looks at the text representation and picks out the relevant information

Finally: `[32, 768] → RMSNorm → Linear(768→32) → take first 2 columns → [32, 2]` (dx, dy actions).

**Frozen during training:** The vision backbone (Conv2D + ViT + connector) — we replace it with zeros during training. Vision weights are from the original pretrained VLA and don't need updating for this task.

**Trainable:** `text_embed`, `state_encoder`, `text_enc`, `exp_self`, `exp_cross`, `post_norm`, `post_proj` — about **23.7M parameters**.

---

## Phase 1: Behavioral Cloning (BC)

**Concept:** "Learn by watching an expert."

We have a hand-coded **scripted policy** that already knows how to push blocks. Its logic is simple: move the end-effector behind the block, then push toward the target. It's not perfect (noisy, sometimes loses the block) but it demonstrates the right behavior.

**Training loop (10,000 steps):**

```
for each step:
    1. Pick a random command: "up", "down", "left", or "right"
    2. Place the robot + block at a random starting position
    3. Ask the scripted expert: "what would you do from here?" → 32 actions
    4. Ask the model: "what would YOU do?" (given text + 7D state) → 32 actions
    5. Loss = mean squared error between model's actions and expert's actions
    6. Backpropagation: nudge model weights to reduce the error
```

**Why MSE loss?** Actions are continuous (dx, dy), so we treat this as regression — minimize the average squared difference between predicted and ground-truth action vectors.

**Key hyperparameters:**
- `lr = 3e-4` — learning rate, how big each weight update step is. Too high → training unstable, too low → trains slowly.
- `save_every = 1000` — save a checkpoint every 1000 steps so we can resume or roll back.
- `noise_std = 0.3` — noise added to the scripted policy's actions, so the model learns robustly rather than memorizing exact trajectories.

**What BC achieves:** The model learns to imitate the expert — it will push the block roughly in the right direction for each command. But it may not complete the task reliably because BC learns the average behavior, not the optimal one.

---

## Phase 2: Reinforcement Learning (RL)

**Concept:** "Learn by trial and error, maximize reward."

Instead of imitating an expert, the model tries things and gets scored based on how well they worked. Over many episodes it figures out what actually works.

**Algorithm: REINFORCE (policy gradient)**

```
for each episode:
    1. Pick a random command + random starting position
    2. Run the model for 6 chunks (192 steps total) — this is a "rollout"
    3. After each step, compute a reward:
         reward = -(distance from block to target) / 512
         bonus  = +2.0 if distance < 30 pixels  (task "solved")
    4. Compute "returns": weighted sum of future rewards at each chunk
    5. Compare return to a running average (baseline) → "advantage"
       - positive advantage = this episode went better than average
       - negative advantage = worse than average
    6. Update weights: reinforce actions that led to positive advantage,
       discourage actions that led to negative advantage
```

**Why a baseline?** Raw returns are noisy. By subtracting the running average (EMA baseline), we focus on "was this better or worse than usual?" instead of the absolute reward value. This dramatically reduces variance and makes training more stable.

**Entropy bonus:** A small regularization that encourages the model to stay somewhat random (not collapse to always taking the same action). `entropy_coef = 0.01` — small enough to not dominate the task reward.

**Exploration:** The model outputs a mean action; we add Gaussian noise with learnable standard deviation (`log_std`, initialized to −1.0 → std ≈ 0.37). This exploration is essential — without it the model can't discover better strategies.

**Key hyperparameters:**
- `rl_lr = 1e-4` — lower than BC learning rate; RL gradients are noisier so we take smaller steps.
- `gamma = 0.95` — discount factor, how much future rewards matter vs. immediate ones.
- `episodes = 3000` — each episode is one full 192-step trial.

---

## Training Run (2026-05-14)

Launched in tmux session `vla_train`:

```bash
python train/train_conditioned.py \
    --phase both \
    --bc_steps 10000 \
    --rl_episodes 3000 \
    --save_every 1000
```

### BC Results (observed)

| Milestone | Loss | Time |
|-----------|------|------|
| Step 200  | 14.43 | 16s |
| Step 1000 | 5.39  | 80s |
| Step 5000 | 4.85  | 403s |
| Step 10000 (final) | **4.06** | 800s (~13 min) |

Loss dropped from ~14 → ~4 and plateaued around step 3000–4000. The plateau is expected: the scripted expert is stochastic (random noise), so perfect imitation is impossible. A loss of ~4 means the model is capturing the general direction of pushing but not matching exact pixel displacements.

### RL Results (observed, in progress at time of writing)

| Episode | Reward | Solve rate (last 50 eps) |
|---------|--------|--------------------------|
| 50   | −1.96 | 2%  |
| 500  | −2.16 | 0%  |
| 1400 | −1.60 | **10%** (best so far) |
| 2000 | −2.09 | 2%  |
| 2650 | −1.89 | 6%  |

The solve rate is low (0–10%). The reward is around −2.0 throughout, which means the block is typically 1000px away from the target on average (workspace = 512px, so −2.0 means ~2× workspace distance — the block is often far off). The RL is not making strong progress, likely because:
- 3000 episodes is a short budget for REINFORCE, which is a high-variance algorithm
- The scripted expert sometimes doesn't complete the task itself (noisy)
- The model may need more BC steps before RL can reliably improve it

**Possible next steps if performance is insufficient:**
- Increase `--bc_steps 30000` — more imitation before RL
- Use a scripted-only eval to see if the env is solvable from random positions at all
- Increase `--rl_episodes 10000` — RL needs more time to find the signal

Log: `simulation/outputs/train_conditioned.log`  
Checkpoints: `simulation/checkpoints/conditioned/latest.pt`

**To evaluate after training:**
```bash
# From simulation/
python train/train_conditioned.py --eval
```
This runs 20 deterministic episodes (5 per command) and prints solve rate per direction.

**To resume RL if interrupted:**
```bash
python train/train_conditioned.py --phase rl --resume
```

---

## How Text Conditioning Actually Works (Intuition)

Imagine the transformer's attention mechanism as "asking questions":

- Each of the 32 action tokens says: *"I'm action step #N — what from the text is relevant to me?"*
- The text tokens (which encode "push the block left") respond with: *"left means negative-X direction"*
- The cross-attention weights tell us *how much* each action step attends to each text token

After training, the model learns that when it reads "left", it should weight the end-effector and block positions differently and produce actions that move the block in the −X direction. The mapping from character sequences → spatial behavior is entirely learned from the BC + RL signal.

---

## Files

| File | Purpose |
|------|---------|
| `simulation/env/pusht_env_multi.py` | PushT with 4 named target zones |
| `simulation/model/model_conditioned.py` | `VLAConditioned` — full model definition |
| `simulation/train/train_conditioned.py` | BC + RL training script |
| `simulation/checkpoints/conditioned/` | Saved checkpoints |
| `simulation/demo/demo_npu_real.py` | NPU demo (to be updated for text conditioning) |
