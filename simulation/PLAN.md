# SmolVLA-Mini: CPU Training → NPU Deployment Plan

## Goal

Train a small VLA model on CPU using the **exact same architecture** already implemented in
`vla/vla.py`, then load the same weights on the NPU to demonstrate real inference performance.
The CPU simulation serves as both a correctness testbed and a timing baseline.

---

## Current State

| File | Status | Role |
|---|---|---|
| `vla/vla.py` | Working, random weights | NPU inference pipeline (BF16) |
| `vla/preprocessing_bf16.py` | Working | Conv2d patch embedding on NPU |
| `vla/vision_block_bf16.py` | Working | ViT block on NPU |
| `vla/connector_bf16.py` | Working | Pixel shuffle + linear on NPU |
| `vla/text_encoder_bf16.py` | Working | LLM/text encoder block on NPU |
| `vla/action_expert_bf16.py` | Working | Action expert self+cross on NPU |
| `simulation/demo_cpu.py` | Working, random weights | PyTorch CPU reference |
| `simulation/pusht_env.py` | Working | 2D push environment |
| `simulation/train.py` | Working | Behavioral cloning (no text cond.) |

**Key gaps**: (1) weights are random — model doesn't do anything meaningful; (2) text is a
fixed random buffer — language does not condition behavior; (3) no bridge between PyTorch
training format and NPU numpy-dict format.

---

## Architecture Spec

Must match `vla/vla.py` exactly so the same weights run on both CPU and NPU.

```
Image [3, 512, 512]
  → Conv2d(3, 768, kernel=16, stride=16)       [1024, 768]   preprocessing_bf16
  → ViT block × 1  (EMBD=768, H=12, FFN=3072)  [1024, 768]   vision_block_bf16
  → Connector (pixel-shuffle + linear)          [64, 960]     connector_bf16
  ↓
Multimodal sequence assembly:
  [64 vision] + [48 text] + [1 state] + [15 pad] = [128, 960]
  ↓
Joint transformer × 2 layers (weight-shared):
  Layer 0 (i%2==0): text_encoder self-attn + action_expert self-attn
  Layer 1 (i%2==1): text_encoder self-attn + action_expert cross-attn ← attends to text K/V
  ↓
Postprocessing: RMSNorm(768) + Linear(768→32)  [32, 32]
  → first 2 columns used as (dx, dy) action
```

**Dimensions** (from NPU kernel files):

| Component | Key dims |
|---|---|
| ViT | SEQ=1024, EMBD=768, N_HEAD=12, FFN=3072 |
| Connector | in=12288 (768×4×4), out=960 |
| Text encoder | SEQ=128, EMBD=960, Q_H=15, KV_H=5, HEAD_DIM=64, FFN=2560 |
| Action expert | SEQ=32, EMBD=768, Q_H=15, KV_H=5, HEAD_DIM=64, FFN=2048 |
| Multimodal seq | 64 vision + 48 text + 1 state + 15 pad = 128 |
| Action output | [32 steps, 32 dims] → use [:, :2] as (dx, dy) |

**Note on weight sharing**: Both transformer layers share the same weight tensors (one
`TextEncoderBlock`, one `ActionExpertSelfBlock`, one `ActionExpertCrossBlock`). This matches
the existing NPU code and reduces parameter count.

---

## Text Conditioning Approach

**No external tokenizer needed.** Use character-level encoding into a small embedding table:

- Vocabulary size: **256** (ASCII byte values)
- Embedding table: `nn.Embedding(256, 960)` → 0.5 MB in BF16, vs 94 MB for 49280-vocab
- Tokenization: `tokens = [ord(c) % 256 for c in text]`, padded/truncated to **48 tokens**
- In `vla.py`, replace `TEXT_VOCAB_SIZE = 49280` with `256` and update the embedding creation

**Command set** for PushT training:

| Text prompt | Target zone |
|---|---|
| `"push the block to the center"` | (256, 180) |
| `"push the block to the left"` | (128, 256) |
| `"push the block to the right"` | (384, 256) |
| `"push the block up"` | (256, 130) |

At training time, a command is sampled per episode. The scripted policy uses the matching
target zone. The model learns to associate the language context with the correct direction.

---

## Shared Weight Format

All weights saved as a single `.npz` file with flat string keys. The NPU code loads this
instead of generating random arrays.

```
weights.npz
  proc/kernel          float32  [768, 3, 16, 16]
  text_emb/weight      float32  [256, 960]          ← character embedding table

  vit/Wq               float32  [768, 768]
  vit/Wk               float32  [768, 768]
  vit/Wv               float32  [768, 768]
  vit/Wo               float32  [768, 768]
  vit/W_up             float32  [768, 3072]
  vit/W_down           float32  [3072, 768]
  vit/W_norm_1         float32  [768]
  vit/b_norm_1         float32  [768]
  vit/W_norm_2         float32  [768]
  vit/b_norm_2         float32  [768]

  con/W                float32  [12288, 960]

  vlm/Wq               float32  [960, 960]
  vlm/Wk               float32  [960, 320]
  vlm/Wv               float32  [960, 320]
  vlm/Wo               float32  [960, 960]
  vlm/W_gate           float32  [960, 2560]
  vlm/W_up             float32  [960, 2560]
  vlm/W_down           float32  [2560, 960]
  vlm/W_norm_1         float32  [960]
  vlm/W_norm_2         float32  [960]

  exp_self/Wq          float32  [768, 960]
  exp_self/Wk          float32  [768, 320]
  exp_self/Wv          float32  [768, 320]
  exp_self/Wo          float32  [960, 768]
  exp_self/W_gate      float32  [768, 2048]
  exp_self/W_up        float32  [768, 2048]
  exp_self/W_down      float32  [2048, 768]
  exp_self/W_norm_1    float32  [768]
  exp_self/W_norm_2    float32  [768]

  exp_cross/Wq         float32  [768, 960]
  exp_cross/Wk_cross   float32  [320, 320]
  exp_cross/Wv_cross   float32  [320, 320]
  exp_cross/Wo         float32  [960, 768]
  exp_cross/W_gate     float32  [768, 2048]
  exp_cross/W_up       float32  [768, 2048]
  exp_cross/W_down     float32  [2048, 768]
  exp_cross/W_norm_1   float32  [768]
  exp_cross/W_norm_2   float32  [768]

  out/W_exp_norm       float32  [768]
  out/W_action_out     float32  [768, 32]
```

Stored in FP32. The NPU loader casts to BF16 at load time via `.astype(np_bfloat16)`.

---

## Implementation Stages

### Stage 1 — `model.py` (new file)

Define `SmolVLAMini` as a `nn.Module` matching the NPU architecture exactly.

**Tasks:**
- `SmolVLAMini.__init__()`: all weight tensors as `nn.Parameter` or `nn.Module`, matching the
  `.npz` key names above
- `SmolVLAMini.forward(image, text_tokens, state)` → `[32, 32]` raw action logits
- `SmolVLAMini.tokenize(text: str) → np.ndarray` of shape `[48]` (character-level)
- `export_weights(model, path)` → saves `.npz` with FP32 weights using the key schema above
- `load_weights(model, path)` → inverse of above (for resuming training)

The ViT and connector can be initialized randomly and **frozen during training** — they act
as a fixed feature extractor. Only text_emb, text encoder, action expert, and postprocessing
are trained.

**What to freeze / train:**

| Module | Frozen | Reason |
|---|---|---|
| Conv2d preprocessing | yes | pure geometry, task-agnostic |
| ViT | yes | expensive, random init is ok as fixed features |
| Connector | yes | fixed linear projection |
| `text_emb` (char embedding) | **no** | must learn to embed commands |
| Text encoder | **no** | must learn to contextualize text+vision |
| Action expert (self + cross) | **no** | core policy |
| Postprocessing | **no** | maps to action space |

---

### Stage 2 — `pusht_env_multi.py` (new file)

Extend PushT with multiple named target zones and text-conditioned resets.

**Tasks:**
- Add 4 target zones: center, left, right, up (fixed positions in workspace)
- `PushTEnvMulti.reset(command)` → sets active target zone based on command string
- `PushTEnvMulti.render()` → highlight the active target zone in a distinct color
- Keep all physics from `pusht_env.py` unchanged

---

### Stage 3 — `train_conditioned.py` (new file, replaces `train.py` for this task)

Behavioral cloning with text conditioning.

**Tasks:**
- Data generation: for each episode, sample a random command → get scripted policy actions
  toward the corresponding target → record `(image, text_tokens, state, actions)` tuples
- Loss: MSE between predicted `actions[:, :2]` and scripted `(dx, dy)` actions
- Optimizer: Adam, lr=1e-3, cosine decay
- Batch: collect N episodes per step, compute loss, backward
- Checkpoint: `export_weights(model, "checkpoints/conditioned_stepXXX.npz")` every 1000 steps
- Log: loss, per-command accuracy (does the robot go toward the right zone?)

**Training target** (~10k steps on CPU):
- Loss should drop from ~50 (random) to ~3 (reasonable behavior)
- Qualitatively: text `"push left"` should produce leftward bias in predicted actions

---

### Stage 4 — `demo_conditioned.py` (new file)

CPU simulation loop with trained weights and real text input.

**Tasks:**
- Load `checkpoints/conditioned_latest.npz` via `load_weights()`
- Accept text prompt from terminal (or command-line arg)
- Run inference loop: every 32 steps, call `model.forward()`, get action chunk
- Render and save frames to `outputs/demo_conditioned_<command>.gif`
- Print per-inference timing

**Success criterion**: robot visibly goes to different zones for different text inputs.

---

### Stage 5 — NPU weight loading (modify `vla/vla.py`)

Replace random weight generation in `vla.py` with loading from `.npz`.

**Tasks:**
- Add `load_params_from_npz(path)` that reads the `.npz` and returns the param dicts expected
  by each NPU kernel (`params_proc`, `params_vit`, `params_con`, `params_vlm`, etc.)
- Cast each tensor to BF16 at load time
- Update `TEXT_VOCAB_SIZE` to 256 in `vla.py`'s text embedding creation
- Add `run_inference(image_bf16, text_tokens, state_bf16)` function that wraps the current
  pipeline and returns `[32, 32]` actions

---

### Stage 6 — `demo_npu_real.py` (replace fake demo)

Real NPU demo with trained weights.

**Tasks:**
- Load `checkpoints/conditioned_latest.npz` via `load_params_from_npz()`
- Accept text prompt
- Inference loop connected to `PushTEnvMulti`
- Print real NPU kernel timings per inference call
- Side-by-side output: CPU timing (from `demo_conditioned.py` run) vs NPU timing

---

## Precision Considerations

Training is in FP32. NPU runs in BF16. The key concern is whether BF16 degrades action
quality enough to change behavior.

**Mitigation already in place:**
- Xavier-style initialization (`1/sqrt(fan_in)`) prevents accumulation blowup in BF16 GEMMs
  (noted in `vla.py` comments)
- The connector's 192-tile GEMM (K=12288) is the highest-risk layer; scaling is critical there

**Validation step** (add to `demo_conditioned.py`):
- Run the same input through the model in FP32 and in BF16 (cast all weights before inference)
- Compare output actions: expect max error < 0.5 on a scale of ~6 units
- If error is too high: add a learned scale factor per layer, or train with BF16-aware weight
  initialization

---

## File Summary

```
simulation/
  env/
    __init__.py
    pusht_env.py              keep (single-zone PushT environment)
    pusht_env_multi.py        NEW  PushT with 4 named target zones
  model/
    __init__.py
    model.py                  NEW  SmolVLAMini nn.Module + export/load helpers
  train/
    __init__.py
    train.py                  keep (older unconditioned training, for reference)
    train_conditioned.py      NEW  text-conditioned behavioral cloning
  demo/
    __init__.py
    demo_cpu.py               keep (CPU reference with random weights)
    demo_npu.py               keep (fake NPU demo, gif output)
    demo_npu_live.py          keep → eventually replaced by demo_npu_real.py
    demo_conditioned.py       NEW  CPU simulation with trained weights
    demo_npu_real.py          NEW  real NPU demo with trained weights
  utils/
    __init__.py
    show_actions.py           keep
  checkpoints/
    conditioned_step1000.npz
    conditioned_step2000.npz
    ...
    conditioned_latest.npz    ← symlink to latest
  outputs/                    all generated .gif / .png go here
  PLAN.md
  README.md
  demo_live.png               live-updating preview (written by demo_npu_live.py)

vla/
  vla.py                      MODIFY: add load_params_from_npz(), run_inference()
```

---

## Suggested Order of Work

1. `model.py` — define `SmolVLAMini` and `export_weights` / `load_weights`
2. Sanity check: export random weights → load in `vla.py` → forward pass matches CPU reference
3. `pusht_env_multi.py` — multi-zone environment
4. `train_conditioned.py` — run 2k steps, verify loss drops and directional bias appears
5. `demo_conditioned.py` — visual check: does the robot follow the text?
6. `vla.py` modifications — load `.npz`, add `run_inference()`
7. `demo_npu_real.py` — end-to-end NPU demo with real timing

---

## Open Questions

- **Does weight sharing across transformer layers hurt quality?** It's an unusual choice
  (shared weights = same layer applied multiple times, like looped attention). For a small
  demo task it should be fine; revisit if loss plateaus.
- **Is 10k training steps enough?** Previous experiments showed loss dropping from ~20 to ~3
  in 10k steps for the unconditioned case. Text conditioning adds complexity; may need 15-20k.
- **NPU function call overhead**: noted as the current key bottleneck. Staging (this plan)
  avoids it by solving CPU training first; NPU integration comes last after weights are validated.
