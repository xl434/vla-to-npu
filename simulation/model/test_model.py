"""
Model tests for SmolVLAMini.

Run from simulation/:
    python model/test_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import math
import tempfile
import numpy as np
import torch

from model.model import (
    SmolVLAMini, tokenize, export_weights, load_weights,
    CH, PIX, VIT_SEQ, VIT_EMBD, VIS_SEQ, CONN_OUT, LLM_EMBD,
    MM_SEQ, EXP_SEQ, EXP_EMBD, CHUNK_SIZE, STATE_DIM, ACTION_DIM,
    TEXT_SEQ, TEXT_VOCAB,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return cond

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

all_passed = True

# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────
torch.manual_seed(0)
model = SmolVLAMini()
model.eval()

image  = torch.randn(1, CH, PIX, PIX)
tok_a  = tokenize("push the block to the center")
tok_b  = tokenize("push the block to the left")
state  = torch.randn(1, STATE_DIM)   # non-zero so state_w receives a real gradient


# ─────────────────────────────────────────────────────────
# 1. Tokenizer
# ─────────────────────────────────────────────────────────
section("1. Tokenizer")

t = tokenize("abc")
all_passed &= check("shape", t.shape == (TEXT_SEQ,), f"{t.shape}")
all_passed &= check("dtype", t.dtype == torch.long)
all_passed &= check("values in [0, 255]", t.min() >= 0 and t.max() <= 255)
all_passed &= check("padding zeros", t[3:].sum().item() == 0,
                    f"t[3:]={t[3:6].tolist()}...")
all_passed &= check("'a'→97", t[0].item() == 97)

long_text = "x" * 100
t_long = tokenize(long_text)
all_passed &= check("truncation to TEXT_SEQ", t_long.shape == (TEXT_SEQ,))


# ─────────────────────────────────────────────────────────
# 2. Intermediate tensor shapes
# ─────────────────────────────────────────────────────────
section("2. Intermediate shapes")

with torch.no_grad():
    # Vision backbone
    vis_raw = model._vision_forward(image)
    all_passed &= check("ViT output", vis_raw.shape == (1, VIT_SEQ, VIT_EMBD),
                        str(vis_raw.shape))

    vis_conn = model._connector(vis_raw)
    all_passed &= check("Connector output", vis_conn.shape == (VIS_SEQ, CONN_OUT),
                        str(vis_conn.shape))

    # Text embedding
    text_emb = model.text_emb(tok_a) * math.sqrt(LLM_EMBD)
    all_passed &= check("Text embedding", text_emb.shape == (TEXT_SEQ, LLM_EMBD),
                        str(text_emb.shape))

    # Full forward
    out = model(image, tok_a, state)
    all_passed &= check("Full forward output", out.shape == (CHUNK_SIZE, STATE_DIM),
                        str(out.shape))


# ─────────────────────────────────────────────────────────
# 3. Numerical sanity (no NaN / inf)
# ─────────────────────────────────────────────────────────
section("3. Numerical sanity")

with torch.no_grad():
    # Multiple random inputs
    for seed in [0, 1, 42, 99]:
        torch.manual_seed(seed)
        img_r  = torch.randn(1, CH, PIX, PIX)
        tok_r  = torch.randint(0, TEXT_VOCAB, (TEXT_SEQ,))
        st_r   = torch.randn(1, STATE_DIM)
        out_r  = model(img_r, tok_r, st_r)
        ok = torch.isfinite(out_r).all().item()
        all_passed &= check(f"seed={seed}: no NaN/inf", ok,
                            f"range [{out_r.min():.2f}, {out_r.max():.2f}]")


# ─────────────────────────────────────────────────────────
# 4. Text conditioning — different text → different actions
# ─────────────────────────────────────────────────────────
section("4. Text conditioning")

with torch.no_grad():
    out_a = model(image, tok_a, state)
    out_b = model(image, tok_b, state)

diff = (out_a - out_b).abs()
all_passed &= check("Different text → different output",
                    diff.max().item() > 1e-6,
                    f"max diff = {diff.max().item():.4f}")
all_passed &= check("Same text → identical output (deterministic)",
                    (model(image, tok_a, state) - out_a).abs().max().item() == 0.0)

# Four different commands should all produce distinct outputs
commands = [
    "push the block to the center",
    "push the block to the left",
    "push the block to the right",
    "push the block up",
]
with torch.no_grad():
    outs = [model(image, tokenize(c), state) for c in commands]
pairs_differ = all(
    (outs[i] - outs[j]).abs().max().item() > 1e-6
    for i in range(len(outs)) for j in range(i + 1, len(outs))
)
all_passed &= check("All 4 commands produce distinct outputs", pairs_differ)


# ─────────────────────────────────────────────────────────
# 5. Frozen parameters (no gradient on vision layers)
# ─────────────────────────────────────────────────────────
section("5. Frozen / trainable parameters")

# Run a backward pass through a dummy loss
model.train()
out_train = model(image, tok_a, state)
loss = out_train.sum()
loss.backward()

frozen_names  = ["conv", "vit", "connector_w"]
trained_names = ["text_emb", "state_w", "text_enc", "exp_self", "exp_cross",
                 "post_norm", "post_proj"]

for name, param in model.named_parameters():
    root = name.split(".")[0]
    if any(name.startswith(f) for f in frozen_names):
        ok = param.grad is None
        all_passed &= check(f"frozen:  {name}", ok,
                            "no grad" if ok else f"UNEXPECTED grad norm={param.grad.norm():.3f}")
    elif any(name.startswith(t) for t in trained_names):
        ok = param.grad is not None and param.grad.norm().item() > 0
        all_passed &= check(f"trained: {name}", ok,
                            f"grad norm={param.grad.norm():.3f}" if param.grad is not None else "MISSING grad")

model.eval()
model.zero_grad()


# ─────────────────────────────────────────────────────────
# 6. Weight sharing across transformer layers
# ─────────────────────────────────────────────────────────
section("6. Weight sharing")

# The same nn.Module instances are called twice in forward().
# Verify that modifying a weight is reflected in both "layers".
orig_w = model.text_enc.q_proj.weight.data.clone()
model.text_enc.q_proj.weight.data.fill_(0.0)

with torch.no_grad():
    out_zeroed = model(image, tok_a, state)

model.text_enc.q_proj.weight.data.copy_(orig_w)

# If weight sharing is real, zeroing q_proj changes both layer-0 and layer-1 passes.
# The output should differ from the original.
diff_zero = (out_zeroed - out_a).abs().max().item()
all_passed &= check("Zeroing shared weight changes output", diff_zero > 1e-6,
                    f"max diff = {diff_zero:.4f}")


# ─────────────────────────────────────────────────────────
# 7. Export / load round-trip (weight-level)
# ─────────────────────────────────────────────────────────
section("7. Export / load round-trip")

with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
    tmp = f.name[:-4]

export_weights(model, tmp)
npz = np.load(tmp + ".npz")

expected_keys = {
    "proc/kernel", "text_emb/weight",
    "vit/Wq", "vit/Wk", "vit/Wv", "vit/Wo",
    "vit/W_up", "vit/W_down", "vit/W_norm_1", "vit/b_norm_1",
    "vit/W_norm_2", "vit/b_norm_2",
    "con/W", "state/W", "action_queries",
    "vlm/Wq", "vlm/Wk", "vlm/Wv", "vlm/Wo",
    "vlm/W_gate", "vlm/W_up", "vlm/W_down",
    "vlm/W_norm_1", "vlm/W_norm_2",
    "exp_self/Wq", "exp_self/Wk", "exp_self/Wv", "exp_self/Wo",
    "exp_self/W_gate", "exp_self/W_up", "exp_self/W_down",
    "exp_self/W_norm_1", "exp_self/W_norm_2",
    "exp_cross/Wq", "exp_cross/Wk_cross", "exp_cross/Wv_cross", "exp_cross/Wo",
    "exp_cross/W_gate", "exp_cross/W_up", "exp_cross/W_down",
    "exp_cross/W_norm_1", "exp_cross/W_norm_2",
    "out/W_exp_norm", "out/W_action_out",
}
actual_keys = set(npz.files)
all_passed &= check("All expected keys present",
                    expected_keys == actual_keys,
                    f"missing={expected_keys-actual_keys}, extra={actual_keys-expected_keys}")

# Spot-check shapes (NPU [in, out] convention)
shape_checks = [
    ("vit/Wq",           (VIT_EMBD, VIT_EMBD)),
    ("vit/W_up",         (VIT_EMBD, VIT_EMBD * 4)),
    ("con/W",            (12288, CONN_OUT)),
    ("vlm/Wq",           (LLM_EMBD, LLM_EMBD)),
    ("vlm/Wk",           (LLM_EMBD, 320)),
    ("exp_self/Wq",      (EXP_EMBD, 960)),
    ("exp_cross/Wk_cross", (320, 320)),
    ("out/W_action_out", (EXP_EMBD, STATE_DIM)),
]
for key, expected_shape in shape_checks:
    actual = npz[key].shape
    all_passed &= check(f"shape {key}", actual == expected_shape,
                        f"{actual} vs {expected_shape}")

# Load into a fresh model and verify outputs match
model2 = SmolVLAMini()
load_weights(model2, tmp + ".npz")
model2.eval()
with torch.no_grad():
    out_reloaded = model2(image, tok_a, state)
rt_diff = (out_reloaded - out_a).abs().max().item()
all_passed &= check("Reloaded model output identical", rt_diff < 1e-5,
                    f"max diff = {rt_diff:.2e}")


# ─────────────────────────────────────────────────────────
# 8. inference() wrapper
# ─────────────────────────────────────────────────────────
section("8. inference() wrapper")

actions = model.inference(
    image.squeeze(0).numpy(), "push the block to the left", state.numpy()[0, :5]
)
all_passed &= check("Output shape", actions.shape == (CHUNK_SIZE, ACTION_DIM),
                    str(actions.shape))
all_passed &= check("dtype float32", actions.dtype == np.float32)
all_passed &= check("No NaN/inf", bool(np.isfinite(actions).all()))


# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
if all_passed:
    print(f"  {PASS}  All tests passed.")
else:
    print(f"  {FAIL}  Some tests failed — see above.")
print(f"{'═'*55}\n")
sys.exit(0 if all_passed else 1)
