# Implementation Progress

**Last updated:** 2026-05-07  
**Goal:** Train SmolVLA-Mini on CPU → simulate on CPU → load same weights on NPU → demonstrate performance.

---

## Completed

### Folder reorganization
All Python scripts moved into logical subdirectories. All imports updated and smoke-tested.

```
simulation/
  env/      pusht_env.py
  model/    model.py, test_model.py
  train/    train.py
  demo/     demo_cpu.py, demo_npu.py, demo_npu_live.py
  utils/    show_actions.py
  model/    (ready for new files)
  outputs/  all generated .gif / .png
  checkpoints/
  PLAN.md, PROGRESS.md, README.md
```

### PLAN.md
Full implementation plan written covering 6 stages, weight format spec, architecture dimensions, text conditioning approach, and precision considerations. Lives at `simulation/PLAN.md`.

### `model/model.py` — SmolVLAMini
Architecture exactly matches `vla/vla.py` (same dimensions, same weight-sharing pattern).

| Component | Details |
|---|---|
| Preprocessing | `Conv2d(3, 768, kernel=16, stride=16)` → `[1024, 768]` |
| ViT | 1 layer, EMBD=768, 12 heads, FFN=3072, LayerNorm |
| Connector | Pixel-shuffle + linear, `[1024,768]` → `[64,960]` |
| Text embedding | `Embedding(256, 960)`, character-level tokenization (no external tokenizer) |
| Text encoder | 1 block (weight-shared × 2), GQA Q_H=15/KV_H=5, RoPE, SwiGLU, EMBD=960 |
| Action expert | Self-attn block + cross-attn block (weight-shared), EMBD=768 |
| Action queries | Learned `[32, 768]` parameter (replaces zero init — see bugs below) |
| Postprocessing | RMSNorm(768) + Linear(768→32) |
| **Total params** | **42.7M** |
| **Trainable** | **23.2M** (text_emb, text_enc, action_queries, exp_self, exp_cross, state_w, post_norm, post_proj) |
| **Frozen** | conv, ViT, connector |

Key functions:
- `tokenize(text)` → `int64 [48]`, character-level, vocab=256, no deps
- `export_weights(model, path)` → `.npz`, 44 tensors, float32, NPU `[in,out]` convention
- `load_weights(model, path)` → inverse, strict load, round-trip verified
- `model.inference(image_np, text, state_np)` → `[32, 2]` numpy (dx, dy) actions

### `model/test_model.py` — All 44 tests passing

| Test section | Result |
|---|---|
| 1. Tokenizer | PASS — shape, dtype, values, padding, truncation |
| 2. Intermediate shapes | PASS — ViT `[1,1024,768]`, connector `[64,960]`, output `[32,32]` |
| 3. Numerical sanity | PASS — no NaN/inf across 4 random seeds |
| 4. Text conditioning | PASS — 4 different commands produce 4 distinct outputs |
| 5. Frozen / trainable params | PASS — all 11 frozen params have zero grad; all 33 trainable params have non-zero grad |
| 6. Weight sharing | PASS — modifying one shared weight changes both transformer passes |
| 7. Export / load round-trip | PASS — all 44 keys present, correct NPU shapes, 0.00e+00 max diff |
| 8. inference() wrapper | PASS — shape `[32,2]`, float32, no NaN |

---

## Bugs Found and Fixed

### Bug 1 — Zero action input causes dead gradients in `exp_self`
**Root cause:** `act = torch.zeros(1, EXP_SEQ, EXP_EMBD)` in `forward()`. For any linear layer, `x @ W = 0` when `x = 0`, so every weight in `exp_self` received zero gradient during backward. The block was effectively a no-op and would never learn.

**Fix:** Replaced zero initialization with a learned `action_queries` parameter (`nn.Parameter([32, 768], std=0.02`)), similar to DETR object queries. The action expert now has a trainable non-zero starting point, and all `exp_self` weights receive meaningful gradients.

**Impact:** Also added `action_queries` to `export_weights` / `load_weights` (now 44 tensors instead of 43).

### Bug 2 — Test fixture used zero state vector
**Root cause:** Test used `state = torch.zeros(1, 32)`, so `state @ state_w = 0` and `state_w` appeared to have zero gradient — falsely suggesting `state_w` was dead.

**Fix:** Changed test fixture to `state = torch.randn(1, STATE_DIM)`. The model itself was correct; only the test was misleading.

---

## Open Questions / Known Limitations

- **Weight sharing quality:** Both transformer layers share the same `TextEncoderBlock` / `ActionExpertSelfBlock` / `ActionExpertCrossBlock` weights (same block applied twice). This is an unusual design that matches `vla/vla.py` but may limit expressiveness. Revisit if training loss plateaus — the fix would be separate weights per layer, requiring updates to both `model.py` and `vla/vla.py`.

- **Vision backbone is random + frozen:** Conv2d and ViT are initialized randomly and never trained. The features they produce are arbitrary but fixed — the text encoder and action expert must learn to work with whatever the frozen backbone outputs. This may slow convergence but is acceptable for a demo.

- **Action output scale unknown:** With random frozen vision weights, the scale of visual features fed into the model is uncontrolled. Xavier initialization on the connector (`std=1/sqrt(12288)`) mitigates blowup, but action output magnitude before training is ~±2 units vs the environment's action scale of ~6 units. May need output scaling during training.

---

## Next Stages

### Stage 2 — `env/pusht_env_multi.py`
Extend PushT with 4 named target zones so text commands map to distinct robot behavior.

```
Zone           Position (x, y)    Command
"center"       (256, 180)         "push the block to the center"
"left"         (128, 256)         "push the block to the left"
"right"        (384, 256)         "push the block to the right"
"up"           (256, 130)         "push the block up"
```

- `PushTEnvMulti.reset(command)` → sets active target from command string
- `PushTEnvMulti.render()` → highlights active target in a distinct color

### Stage 3 — `train/train_conditioned.py`
Behavioral cloning with text conditioning.

- Sample random command per episode → set target zone → run scripted policy → record `(image, text_tokens, state, actions)`
- Trainable: text_emb, text_enc, action_queries, exp_self, exp_cross, state_w, post_norm, post_proj
- Frozen: conv, ViT, connector
- Loss: MSE on `actions[:, :2]` vs scripted `(dx, dy)`
- Save checkpoint as `.npz` every 1000 steps via `export_weights()`

### Stage 4 — `demo/demo_conditioned.py`
CPU simulation loop with trained weights and real text input.

- Load `.npz` via `load_weights()`
- Accept text prompt from terminal
- Run inference every 32 steps, render environment
- Save to `outputs/demo_conditioned_<command>.gif`

### Stage 5 — Modify `vla/vla.py`
Load trained `.npz` on NPU instead of random weights.

- Add `load_params_from_npz(path)` → returns param dicts expected by each NPU kernel
- Cast to BF16 at load time
- Add `run_inference(image, text_tokens, state)` function

### Stage 6 — `demo/demo_npu_real.py`
Full NPU demo with real timing comparison.

- CPU baseline timing (from Stage 4)
- NPU inference timing (from Stage 5)
- Text prompt conditions actual robot behavior
