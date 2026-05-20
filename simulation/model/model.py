"""
SmolVLAMini — model definition and weight I/O.

Architecture exactly matches vla/vla.py so the same .npz weights
run on CPU (this file) and on the NPU.

Key design choices:
- Character-level tokenization (vocab=256): no external tokenizer needed.
- text_emb is a trainable Embedding — language genuinely conditions behavior.
- Conv, ViT, and connector are frozen during training (fixed feature extractors).
- The two joint-transformer layers share weights (same block applied twice),
  matching the weight-sharing in vla/vla.py.
- Action expert input is zeros; the model learns to produce actions purely
  from the context (simpler than flow-matching for behavioral cloning).

Weight format (.npz):
- All tensors stored as float32.
- NPU convention: weight matrices are [in_dim, out_dim] (used as x @ W).
- PyTorch convention: nn.Linear.weight is [out_dim, in_dim].
- export_weights transposes on save; load_weights transposes on load.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants — must match vla/vla.py exactly
# ─────────────────────────────────────────────────────────────────────────────
CH           = 3
PIX          = 512
KERNEL_DIM   = 16            # conv patch size → 32×32 patches

VIT_SEQ      = 1024          # 32×32 patches
VIT_EMBD     = 768
VIT_HEADS    = 12
VIT_FFN      = VIT_EMBD * 4  # 3072

VIS_SEQ      = 64            # vision tokens after connector (8×8 super-patches)
CONN_IN      = 12288         # 768 × 4×4 (pixel-shuffle expanded dim)
CONN_OUT     = 960           # == LLM_EMBD

TEXT_VOCAB   = 256           # character-level ASCII (no external tokenizer)
TEXT_SEQ     = 48            # tokens per prompt

LLM_EMBD    = 960
LLM_Q_H     = 15
LLM_KV_H    = 5
LLM_HEAD_DIM = 64
LLM_FFN     = 2560

STATE_DIM    = 32            # env state padded to this many dims
STATE_SEQ    = 1
MM_PAD       = 15            # padding so mm_seq = 64+48+1+15 = 128
MM_SEQ       = VIS_SEQ + TEXT_SEQ + STATE_SEQ + MM_PAD  # 128

EXP_SEQ      = 32
EXP_EMBD     = 768
EXP_Q_H      = 15
EXP_KV_H     = 5
EXP_HEAD_DIM = 64
EXP_KV_DIM   = EXP_KV_H * EXP_HEAD_DIM  # 320
EXP_FFN      = 2048

CHUNK_SIZE   = EXP_SEQ      # action steps per inference call
ACTION_DIM   = 2             # (dx, dy) taken from first 2 dims of 32-dim output

VIT_NUM_LAYERS = 1
LLM_NUM_LAYERS = 2
SKIP           = 2           # cross-attention on every odd layer (0→self, 1→cross)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> torch.Tensor:
    """text → int64 [TEXT_SEQ]. Character-level: ord(c) % 256, padded with 0."""
    ids = [ord(c) % TEXT_VOCAB for c in text[:TEXT_SEQ]]
    ids += [0] * (TEXT_SEQ - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helper (shared by all GQA blocks)
# ─────────────────────────────────────────────────────────────────────────────

def _rope(x: torch.Tensor, rope_ts: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x: [B, L, H, D]."""
    B, L, H, D = x.shape
    d2 = D // 2
    pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    rad = pos[..., None] / rope_ts[None, None, :]   # [B, L, d2]
    rad = rad[..., None, :]                          # [B, L, 1, d2] — broadcast over H
    x = x.float()
    x1, x2 = x[..., :d2], x[..., d2:]
    s, c = rad.sin(), rad.cos()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class ViTBlock(nn.Module):
    """Standard ViT block: LayerNorm → full MHA → LayerNorm → GELU FFN."""

    def __init__(self):
        super().__init__()
        self.ln_1     = nn.LayerNorm(VIT_EMBD)
        self.attn     = nn.MultiheadAttention(VIT_EMBD, VIT_HEADS, batch_first=True, bias=False)
        self.ln_2     = nn.LayerNorm(VIT_EMBD)
        self.ffn_up   = nn.Linear(VIT_EMBD, VIT_FFN,  bias=False)
        self.ffn_down = nn.Linear(VIT_FFN,  VIT_EMBD, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, VIT_SEQ, VIT_EMBD]
        r = x
        x = self.ln_1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + r
        r = x
        x = self.ln_2(x)
        return self.ffn_down(F.gelu(self.ffn_up(x))) + r


class TextEncoderBlock(nn.Module):
    """LLM block: RMSNorm → GQA + RoPE (causal) → SwiGLU FFN.
    Returns (x_out, k_out, v_out) where k/v are exposed for action cross-attn.
    """

    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(LLM_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(LLM_EMBD, LLM_Q_H  * LLM_HEAD_DIM, bias=False)
        self.k_proj    = nn.Linear(LLM_EMBD, LLM_KV_H * LLM_HEAD_DIM, bias=False)
        self.v_proj    = nn.Linear(LLM_EMBD, LLM_KV_H * LLM_HEAD_DIM, bias=False)
        self.o_proj    = nn.Linear(LLM_Q_H * LLM_HEAD_DIM, LLM_EMBD,  bias=False)
        self.ln_2      = nn.RMSNorm(LLM_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(LLM_EMBD, LLM_FFN, bias=False)
        self.up_proj   = nn.Linear(LLM_EMBD, LLM_FFN, bias=False)
        self.down_proj = nn.Linear(LLM_FFN,  LLM_EMBD, bias=False)

        d_half = LLM_HEAD_DIM // 2
        freq = (2.0 / LLM_HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        self.register_buffer("rope_ts", 10000.0 ** freq, persistent=False)

    def forward(self, x: torch.Tensor):
        residual = x
        h = self.ln_1(x)
        B, L, _ = h.shape

        q = _rope(self.q_proj(h).view(B, L, LLM_Q_H,  LLM_HEAD_DIM), self.rope_ts)
        k = _rope(self.k_proj(h).view(B, L, LLM_KV_H, LLM_HEAD_DIM), self.rope_ts)
        v = self.v_proj(h).view(B, L, LLM_KV_H, LLM_HEAD_DIM)

        kv_map = torch.div(
            torch.arange(LLM_Q_H, device=x.device) * LLM_KV_H, LLM_Q_H, rounding_mode="floor"
        )
        qh = q.transpose(1, 2).float()
        kh = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(LLM_HEAD_DIM)
        mask = torch.ones(L, L, device=x.device).triu(1).bool()
        scores.masked_fill_(mask, float("-inf"))
        attn_w = scores.softmax(-1)
        vh = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn_w @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)

        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        x = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + residual

        # Expose K/V from post-block state for action expert cross-attention
        h_post = self.ln_1(x)
        k_out = self.k_proj(h_post).view(B, L, LLM_KV_H * LLM_HEAD_DIM)
        v_out = self.v_proj(h_post).view(B, L, LLM_KV_H * LLM_HEAD_DIM)
        return x, k_out, v_out


class ActionExpertSelfBlock(nn.Module):
    """Action expert self-attention: GQA + RoPE (causal) + SwiGLU."""

    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(EXP_EMBD, EXP_Q_H  * EXP_HEAD_DIM, bias=False)
        self.k_proj    = nn.Linear(EXP_EMBD, EXP_KV_DIM,               bias=False)
        self.v_proj    = nn.Linear(EXP_EMBD, EXP_KV_DIM,               bias=False)
        self.o_proj    = nn.Linear(EXP_Q_H * EXP_HEAD_DIM, EXP_EMBD,  bias=False)
        self.ln_2      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.up_proj   = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.down_proj = nn.Linear(EXP_FFN,  EXP_EMBD, bias=False)

        d_half = EXP_HEAD_DIM // 2
        freq = (2.0 / EXP_HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        self.register_buffer("rope_ts", 10000.0 ** freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.ln_1(x)
        B, L, _ = h.shape

        q = _rope(self.q_proj(h).view(B, L, EXP_Q_H,  EXP_HEAD_DIM), self.rope_ts)
        k = _rope(self.k_proj(h).view(B, L, EXP_KV_H, EXP_HEAD_DIM), self.rope_ts)
        v = self.v_proj(h).view(B, L, EXP_KV_H, EXP_HEAD_DIM)

        kv_map = torch.div(
            torch.arange(EXP_Q_H, device=x.device) * EXP_KV_H, EXP_Q_H, rounding_mode="floor"
        )
        qh = q.transpose(1, 2).float()
        kh = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(EXP_HEAD_DIM)
        mask = torch.ones(L, L, device=x.device).triu(1).bool()
        scores.masked_fill_(mask, float("-inf"))
        attn_w = scores.softmax(-1)
        vh = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn_w @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)

        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + residual


class ActionExpertCrossBlock(nn.Module):
    """Action expert cross-attention: action queries attend to text K/V + SwiGLU."""

    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(EXP_EMBD,  EXP_Q_H * EXP_HEAD_DIM, bias=False)
        self.k_proj    = nn.Linear(EXP_KV_DIM, EXP_KV_DIM,             bias=False)  # projects text K
        self.v_proj    = nn.Linear(EXP_KV_DIM, EXP_KV_DIM,             bias=False)  # projects text V
        self.o_proj    = nn.Linear(EXP_Q_H * EXP_HEAD_DIM, EXP_EMBD,  bias=False)
        self.ln_2      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.up_proj   = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.down_proj = nn.Linear(EXP_FFN,  EXP_EMBD, bias=False)

        d_half = EXP_HEAD_DIM // 2
        freq = (2.0 / EXP_HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        self.register_buffer("rope_ts", 10000.0 ** freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,       # [B, EXP_SEQ, EXP_EMBD]
        text_k: torch.Tensor,  # [B, MM_SEQ, EXP_KV_DIM]
        text_v: torch.Tensor,  # [B, MM_SEQ, EXP_KV_DIM]
    ) -> torch.Tensor:
        residual = x
        h = self.ln_1(x)
        B, L, _ = h.shape
        _, Lc, _ = text_k.shape

        q = _rope(self.q_proj(h).view(B, L, EXP_Q_H, EXP_HEAD_DIM), self.rope_ts)
        k = self.k_proj(text_k).view(B, Lc, EXP_KV_H, EXP_HEAD_DIM)
        v = self.v_proj(text_v).view(B, Lc, EXP_KV_H, EXP_HEAD_DIM)

        kv_map = torch.div(
            torch.arange(EXP_Q_H, device=x.device) * EXP_KV_H, EXP_Q_H, rounding_mode="floor"
        )
        qh = q.transpose(1, 2).float()
        kh = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(EXP_HEAD_DIM)
        attn_w = scores.softmax(-1)   # no causal mask: action attends to all text
        vh = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn_w @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)

        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class SmolVLAMini(nn.Module):
    """
    Minimal SmolVLA matching vla/vla.py exactly.

    Pipeline:
      image [1,3,512,512]
        → Conv2d(3,768,16,16)   [1,768,32,32] → flatten → [1,1024,768]
        → ViTBlock × VIT_NUM_LAYERS             [1,1024,768]
        → pixel-shuffle + linear               [64,960]
      text  [TEXT_SEQ]
        → Embedding(256,960)                   [48,960]
      state [1,STATE_DIM]
        → @ state_w                            [1,960]
      concat + zero-pad → mm_seq               [1,128,960]

      joint transformer × LLM_NUM_LAYERS (weight-shared):
        layer 0 (i%2==0): TextEncoder self-attn + ActionExpert self-attn
        layer 1 (i%2==1): TextEncoder self-attn + ActionExpert cross-attn

      RMSNorm + Linear → [32,32] raw action logits
      first ACTION_DIM=2 columns used as (dx,dy)
    """

    def __init__(self):
        super().__init__()

        # Vision backbone (frozen after init)
        self.conv        = nn.Conv2d(CH, VIT_EMBD, KERNEL_DIM, stride=KERNEL_DIM, bias=False)
        self.vit         = ViTBlock()
        self.connector_w = nn.Parameter(
            torch.empty(CONN_IN, CONN_OUT).normal_(std=1.0 / math.sqrt(CONN_IN))
        )

        # Language (trainable)
        self.text_emb = nn.Embedding(TEXT_VOCAB, LLM_EMBD)

        # State projection (trainable): [STATE_DIM, LLM_EMBD] used as state @ state_w
        self.state_w = nn.Parameter(
            torch.empty(STATE_DIM, LLM_EMBD).normal_(std=1.0 / math.sqrt(STATE_DIM))
        )

        # Learned action query tokens (replaces zero initialization so exp_self
        # receives a non-zero, trainable input and produces meaningful gradients)
        self.action_queries = nn.Parameter(
            torch.empty(EXP_SEQ, EXP_EMBD).normal_(std=0.02)
        )

        # Joint transformer — weight-shared across LLM_NUM_LAYERS
        self.text_enc  = TextEncoderBlock()
        self.exp_self  = ActionExpertSelfBlock()
        self.exp_cross = ActionExpertCrossBlock()

        # Postprocessing
        self.post_norm = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.post_proj = nn.Linear(EXP_EMBD, STATE_DIM, bias=False)

        self._freeze_vision()

    def _freeze_vision(self):
        for p in self.conv.parameters():
            p.requires_grad = False
        for p in self.vit.parameters():
            p.requires_grad = False
        self.connector_w.requires_grad = False

    def _vision_forward(self, image: torch.Tensor) -> torch.Tensor:
        """[1,3,512,512] → [1,1024,768]."""
        x = self.conv(image)                  # [1,768,32,32]
        x = x.flatten(2).transpose(1, 2)     # [1,1024,768]
        for _ in range(VIT_NUM_LAYERS):
            x = self.vit(x)
        return x

    def _connector(self, x: torch.Tensor) -> torch.Tensor:
        """[1,1024,768] → [64,960]: pixel-shuffle then linear."""
        x = x.squeeze(0)                      # [1024,768]
        x = x.reshape(32, 32, 768).reshape(32, 8, 3072).permute(1, 0, 2)
        x = x.reshape(8, 8, 12288).permute(1, 0, 2).reshape(64, 12288)
        return x @ self.connector_w           # [64,960]

    def forward(
        self,
        image: torch.Tensor,        # [1, 3, 512, 512]  float32
        text_tokens: torch.Tensor,  # [TEXT_SEQ]         int64
        state: torch.Tensor,        # [1, STATE_DIM]     float32
    ) -> torch.Tensor:
        """Returns [CHUNK_SIZE, STATE_DIM] raw action logits."""
        vis  = self._vision_forward(image)         # [1,1024,768]
        vis  = self._connector(vis)                # [64,960]

        text = self.text_emb(text_tokens) * math.sqrt(LLM_EMBD)  # [48,960]
        st   = state @ self.state_w                # [1,960]
        pad  = torch.zeros(MM_PAD, LLM_EMBD, device=image.device, dtype=image.dtype)

        mm = torch.cat([vis, text, st, pad], dim=0).unsqueeze(0)  # [1,128,960]

        # Action expert starts from learned query tokens (deterministic, trainable)
        act = self.action_queries.unsqueeze(0).to(dtype=image.dtype)

        for i in range(LLM_NUM_LAYERS):
            mm, text_k, text_v = self.text_enc(mm)
            if i % SKIP == 0:
                act = self.exp_self(act)
            else:
                act = self.exp_cross(act, text_k, text_v)

        out = self.post_norm(act.squeeze(0))       # [32,768]
        return self.post_proj(out)                 # [32,32]

    @torch.no_grad()
    def inference(
        self,
        image_np: np.ndarray,   # [3,512,512] float32
        text: str,
        state_np: np.ndarray,   # 1-D float32, length ≤ STATE_DIM
    ) -> np.ndarray:
        """Numpy in → [CHUNK_SIZE, ACTION_DIM] numpy out. Used by demo loop."""
        self.eval()
        img = torch.from_numpy(image_np).float().unsqueeze(0)
        tok = tokenize(text)
        sp  = np.zeros(STATE_DIM, dtype=np.float32)
        sp[:len(state_np)] = state_np
        st  = torch.from_numpy(sp).float().unsqueeze(0)

        actions = self(img, tok, st)               # [32,32]
        return actions.numpy()[:, :ACTION_DIM]     # [32,2]


# ─────────────────────────────────────────────────────────────────────────────
# Weight I/O
# ─────────────────────────────────────────────────────────────────────────────
# NPU weight convention: matrices are [in_dim, out_dim] (used as x @ W).
# PyTorch nn.Linear.weight is [out_dim, in_dim].
# Conversion: NPU_W = torch_weight.T  (and vice-versa on load).

def export_weights(model: SmolVLAMini, path: str) -> None:
    """Save all weights to <path>.npz as float32 in NPU [in, out] convention."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}

    def npu(key):
        """PyTorch [out,in] → NPU [in,out]."""
        return sd[key].T

    def raw(key):
        """Weights already in NPU [in,out] format (parameters, not nn.Linear)."""
        return sd[key]

    p = {}

    # Preprocessing
    p["proc/kernel"]     = raw("conv.weight")          # [768,3,16,16] — conv format

    # Text embedding
    p["text_emb/weight"] = raw("text_emb.weight")      # [256,960]

    # ViT — split in_proj_weight [3*768,768] into Wq,Wk,Wv then transpose
    ipw = sd["vit.attn.in_proj_weight"]                # [2304,768] = [3*out,in]
    p["vit/Wq"]      = ipw[:VIT_EMBD,        :].T      # [768,768]
    p["vit/Wk"]      = ipw[VIT_EMBD:2*VIT_EMBD, :].T
    p["vit/Wv"]      = ipw[2*VIT_EMBD:,     :].T
    p["vit/Wo"]      = npu("vit.attn.out_proj.weight") # [768,768]
    p["vit/W_up"]    = npu("vit.ffn_up.weight")        # [768,3072]
    p["vit/W_down"]  = npu("vit.ffn_down.weight")      # [3072,768]
    p["vit/W_norm_1"] = raw("vit.ln_1.weight")
    p["vit/b_norm_1"] = raw("vit.ln_1.bias")
    p["vit/W_norm_2"] = raw("vit.ln_2.weight")
    p["vit/b_norm_2"] = raw("vit.ln_2.bias")

    # Connector
    p["con/W"]  = raw("connector_w")                   # [12288,960] — already [in,out]

    # State projection & action queries
    p["state/W"]        = raw("state_w")               # [32,960] — already [in,out]
    p["action_queries"] = raw("action_queries")        # [32,768]

    # Text encoder
    p["vlm/Wq"]     = npu("text_enc.q_proj.weight")    # [960,960]
    p["vlm/Wk"]     = npu("text_enc.k_proj.weight")    # [960,320]
    p["vlm/Wv"]     = npu("text_enc.v_proj.weight")    # [960,320]
    p["vlm/Wo"]     = npu("text_enc.o_proj.weight")    # [960,960]
    p["vlm/W_gate"] = npu("text_enc.gate_proj.weight") # [960,2560]
    p["vlm/W_up"]   = npu("text_enc.up_proj.weight")   # [960,2560]
    p["vlm/W_down"] = npu("text_enc.down_proj.weight") # [2560,960]
    p["vlm/W_norm_1"] = raw("text_enc.ln_1.weight")
    p["vlm/W_norm_2"] = raw("text_enc.ln_2.weight")

    # Action expert — self-attention
    p["exp_self/Wq"]      = npu("exp_self.q_proj.weight")    # [768,960]
    p["exp_self/Wk"]      = npu("exp_self.k_proj.weight")    # [768,320]
    p["exp_self/Wv"]      = npu("exp_self.v_proj.weight")    # [768,320]
    p["exp_self/Wo"]      = npu("exp_self.o_proj.weight")    # [960,768]
    p["exp_self/W_gate"]  = npu("exp_self.gate_proj.weight") # [768,2048]
    p["exp_self/W_up"]    = npu("exp_self.up_proj.weight")   # [768,2048]
    p["exp_self/W_down"]  = npu("exp_self.down_proj.weight") # [2048,768]
    p["exp_self/W_norm_1"] = raw("exp_self.ln_1.weight")
    p["exp_self/W_norm_2"] = raw("exp_self.ln_2.weight")

    # Action expert — cross-attention
    p["exp_cross/Wq"]       = npu("exp_cross.q_proj.weight")    # [768,960]
    p["exp_cross/Wk_cross"] = npu("exp_cross.k_proj.weight")    # [320,320]
    p["exp_cross/Wv_cross"] = npu("exp_cross.v_proj.weight")    # [320,320]
    p["exp_cross/Wo"]       = npu("exp_cross.o_proj.weight")    # [960,768]
    p["exp_cross/W_gate"]   = npu("exp_cross.gate_proj.weight")
    p["exp_cross/W_up"]     = npu("exp_cross.up_proj.weight")
    p["exp_cross/W_down"]   = npu("exp_cross.down_proj.weight")
    p["exp_cross/W_norm_1"] = raw("exp_cross.ln_1.weight")
    p["exp_cross/W_norm_2"] = raw("exp_cross.ln_2.weight")

    # Postprocessing
    p["out/W_exp_norm"]   = raw("post_norm.weight")    # [768]
    p["out/W_action_out"] = npu("post_proj.weight")    # [768,32]

    if not path.endswith(".npz"):
        path = path + ".npz"
    np.savez(path, **p)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {len(p)} tensors → {path}  ({size_mb:.1f} MB)")


def load_weights(model: SmolVLAMini, path: str) -> None:
    """Load weights from .npz (inverse of export_weights)."""
    if not path.endswith(".npz"):
        path = path + ".npz"
    p = np.load(path)

    def pt(key):
        """NPU [in,out] → PyTorch [out,in]."""
        return torch.from_numpy(p[key].T).float()

    def raw(key):
        return torch.from_numpy(p[key]).float()

    sd = model.state_dict()

    # Preprocessing
    sd["conv.weight"] = raw("proc/kernel")

    # Text embedding
    sd["text_emb.weight"] = raw("text_emb/weight")

    # ViT — reassemble in_proj_weight from split Wq/Wk/Wv
    sd["vit.attn.in_proj_weight"] = torch.cat(
        [pt("vit/Wq"), pt("vit/Wk"), pt("vit/Wv")], dim=0
    )                                                  # [2304,768]
    sd["vit.attn.out_proj.weight"] = pt("vit/Wo")
    sd["vit.ffn_up.weight"]        = pt("vit/W_up")
    sd["vit.ffn_down.weight"]      = pt("vit/W_down")
    sd["vit.ln_1.weight"]          = raw("vit/W_norm_1")
    sd["vit.ln_1.bias"]            = raw("vit/b_norm_1")
    sd["vit.ln_2.weight"]          = raw("vit/W_norm_2")
    sd["vit.ln_2.bias"]            = raw("vit/b_norm_2")

    # Connector, state & action queries
    sd["connector_w"]    = raw("con/W")
    sd["state_w"]        = raw("state/W")
    sd["action_queries"] = raw("action_queries")

    # Text encoder
    sd["text_enc.q_proj.weight"]    = pt("vlm/Wq")
    sd["text_enc.k_proj.weight"]    = pt("vlm/Wk")
    sd["text_enc.v_proj.weight"]    = pt("vlm/Wv")
    sd["text_enc.o_proj.weight"]    = pt("vlm/Wo")
    sd["text_enc.gate_proj.weight"] = pt("vlm/W_gate")
    sd["text_enc.up_proj.weight"]   = pt("vlm/W_up")
    sd["text_enc.down_proj.weight"] = pt("vlm/W_down")
    sd["text_enc.ln_1.weight"]      = raw("vlm/W_norm_1")
    sd["text_enc.ln_2.weight"]      = raw("vlm/W_norm_2")

    # Action expert self
    sd["exp_self.q_proj.weight"]    = pt("exp_self/Wq")
    sd["exp_self.k_proj.weight"]    = pt("exp_self/Wk")
    sd["exp_self.v_proj.weight"]    = pt("exp_self/Wv")
    sd["exp_self.o_proj.weight"]    = pt("exp_self/Wo")
    sd["exp_self.gate_proj.weight"] = pt("exp_self/W_gate")
    sd["exp_self.up_proj.weight"]   = pt("exp_self/W_up")
    sd["exp_self.down_proj.weight"] = pt("exp_self/W_down")
    sd["exp_self.ln_1.weight"]      = raw("exp_self/W_norm_1")
    sd["exp_self.ln_2.weight"]      = raw("exp_self/W_norm_2")

    # Action expert cross
    sd["exp_cross.q_proj.weight"]    = pt("exp_cross/Wq")
    sd["exp_cross.k_proj.weight"]    = pt("exp_cross/Wk_cross")
    sd["exp_cross.v_proj.weight"]    = pt("exp_cross/Wv_cross")
    sd["exp_cross.o_proj.weight"]    = pt("exp_cross/Wo")
    sd["exp_cross.gate_proj.weight"] = pt("exp_cross/W_gate")
    sd["exp_cross.up_proj.weight"]   = pt("exp_cross/W_up")
    sd["exp_cross.down_proj.weight"] = pt("exp_cross/W_down")
    sd["exp_cross.ln_1.weight"]      = raw("exp_cross/W_norm_1")
    sd["exp_cross.ln_2.weight"]      = raw("exp_cross/W_norm_2")

    # Postprocessing
    sd["post_norm.weight"] = raw("out/W_exp_norm")
    sd["post_proj.weight"] = pt("out/W_action_out")

    model.load_state_dict(sd, strict=True)
    print(f"Loaded weights from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("Building model...")
    model = SmolVLAMini()
    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}")

    print("Forward pass...")
    image = torch.randn(1, CH, PIX, PIX)
    tok   = tokenize("push the block to the center")
    state = torch.zeros(1, STATE_DIM)
    with torch.no_grad():
        out = model(image, tok, state)
    print(f"  Output shape: {out.shape}  (expect [32, 32])")
    assert out.shape == (CHUNK_SIZE, STATE_DIM), f"unexpected shape {out.shape}"

    print("Export / load round-trip...")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name[:-4]  # export_weights adds .npz
    export_weights(model, tmp)
    model2 = SmolVLAMini()
    load_weights(model2, tmp + ".npz")
    with torch.no_grad():
        out2 = model2(image, tok, state)
    diff = (out - out2).abs().max().item()
    print(f"  Max diff after round-trip: {diff:.2e}  (expect < 1e-5)")
    assert diff < 1e-5, f"round-trip failed: max diff = {diff}"

    print("inference() wrapper...")
    actions = model.inference(image.squeeze(0).numpy(), "push the block left", state.numpy()[0])
    print(f"  Action chunk shape: {actions.shape}  (expect [32, 2])")
    assert actions.shape == (CHUNK_SIZE, ACTION_DIM)

    print("\nAll checks passed.")
