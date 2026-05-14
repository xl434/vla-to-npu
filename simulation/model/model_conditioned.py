"""
VLAConditioned — text-conditioned VLA for multi-target PushT.

Key differences from VLACpuRef (demo_cpu.py):
  - text_embed: nn.Embedding(256, 960)  (char-level, learnable, replaces fixed buffer)
  - state_encoder: 7D input (adds normalized target_x, target_y)
  - 2-layer joint transformer (layer 0: self-attn, layer 1: cross-attn to text K/V)
  - tokenize(text) helper for char-level tokenization

Architecture on NPU (same weights, BF16):
  image  → preprocessing → vision (1L) → connector      [64, 960]
  text   → tokenize → text_embed lookup                  [48, 960]
  state  → state_encoder (CPU)                           [32, 768]
  mm_seq = [vision, text, state_emb, pad]                [128, 960]
  layer 0: text_enc(mm_seq) → K/V;  exp_self(action_q)
  layer 1: text_enc(mm_out) → K/V;  exp_cross(action, K, V)
  → post_norm → post_proj → [32, 32] → actions[:, :2]
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ── Architecture constants (must match vla/vla_standalone.py) ────────────────
SEQ_T       = 48      # text token length (char-level, zero-padded)
TEXT_EMBD   = 960     # text / LLM embedding dimension
TEXT_Q_H    = 15      # text encoder Q heads
TEXT_KV_H   = 5       # text encoder KV heads
TEXT_HD     = 64      # head dimension
TEXT_FFN    = 2560    # text encoder FFN dimension

VIT_EMBD    = 768     # ViT embedding dimension
VIT_HEADS   = 12
VIT_FFN     = 3072

EXP_EMBD    = 768     # action expert embedding
EXP_Q_H     = 15
EXP_KV_H    = 5
EXP_HD      = 64
EXP_KV_DIM  = EXP_KV_H * EXP_HD  # = 320
EXP_FFN     = 2048

CHUNK_SIZE  = 32      # action chunk size
ACTION_DIM  = 2       # (dx, dy)
STATE_DIM   = 7       # [ee_x, ee_y, block_x, block_y, theta, target_x, target_y]
PADDING     = 15      # zero-pad rows in mm_seq
MM_LEN      = 128     # 64 vision + 48 text + 1 state + 15 pad


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str, max_len: int = SEQ_T) -> np.ndarray:
    """Character-level tokenizer.  text → int64 [max_len], vocab 0-255, zero-padded."""
    tokens = [ord(c) & 0xFF for c in text[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)


# ── Sub-blocks (match demo_cpu.py architecture exactly) ──────────────────────

def _rope(x: torch.Tensor, rope_ts: torch.Tensor) -> torch.Tensor:
    B, L, H, D = x.shape
    d2  = D // 2
    pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    rad = (pos[..., None] / rope_ts[None, None, :])[..., None, :]
    x   = x.float()
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * rad.cos() - x2 * rad.sin(),
                      x2 * rad.cos() + x1 * rad.sin()], dim=-1).to(x.dtype)


class TextEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(TEXT_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(TEXT_EMBD, TEXT_Q_H * TEXT_HD, bias=False)
        self.k_proj    = nn.Linear(TEXT_EMBD, TEXT_KV_H * TEXT_HD, bias=False)
        self.v_proj    = nn.Linear(TEXT_EMBD, TEXT_KV_H * TEXT_HD, bias=False)
        self.o_proj    = nn.Linear(TEXT_Q_H * TEXT_HD, TEXT_EMBD, bias=False)
        self.ln_2      = nn.RMSNorm(TEXT_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(TEXT_EMBD, TEXT_FFN, bias=False)
        self.up_proj   = nn.Linear(TEXT_EMBD, TEXT_FFN, bias=False)
        self.down_proj = nn.Linear(TEXT_FFN, TEXT_EMBD, bias=False)
        self.silu      = nn.SiLU()
        d2 = TEXT_HD // 2
        self.register_buffer(
            "rope_ts",
            10000.0 ** ((2.0 / TEXT_HD) * torch.arange(d2, dtype=torch.float32)),
            persistent=False,
        )
        self._kv_map = None

    def _get_kv_map(self, device):
        if self._kv_map is None or self._kv_map.device != device:
            self._kv_map = torch.div(
                torch.arange(TEXT_Q_H, device=device) * TEXT_KV_H,
                TEXT_Q_H, rounding_mode="floor"
            )
        return self._kv_map

    def forward(self, x: torch.Tensor):
        """x [B, L, 960] → (out [B,L,960], key [B,L,320], val [B,L,320])"""
        residual = x
        h  = self.ln_1(x)
        B, L, _ = h.shape
        kv_map = self._get_kv_map(x.device)

        q = _rope(self.q_proj(h).view(B, L, TEXT_Q_H, TEXT_HD), self.rope_ts)
        k = _rope(self.k_proj(h).view(B, L, TEXT_KV_H, TEXT_HD), self.rope_ts)
        v = self.v_proj(h).view(B, L, TEXT_KV_H, TEXT_HD)

        qh  = q.transpose(1, 2).float()
        kh  = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(TEXT_HD)
        mask = torch.ones(L, L, device=x.device).triu(1).bool()
        scores.masked_fill_(mask, float("-inf"))
        attn = scores.softmax(-1)
        vh  = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)

        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        x = self.down_proj(self.silu(self.gate_proj(h)) * self.up_proj(h)) + residual

        # Return last-layer K/V for action expert cross-attention
        h2  = self.ln_1(x)
        k_out = self.k_proj(h2).view(B, L, TEXT_KV_H * TEXT_HD)
        v_out = self.v_proj(h2).view(B, L, TEXT_KV_H * TEXT_HD)
        return x, k_out, v_out


class ActionExpertSelfBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(EXP_EMBD, EXP_Q_H * EXP_HD, bias=False)
        self.k_proj    = nn.Linear(EXP_EMBD, EXP_KV_DIM, bias=False)
        self.v_proj    = nn.Linear(EXP_EMBD, EXP_KV_DIM, bias=False)
        self.o_proj    = nn.Linear(EXP_Q_H * EXP_HD, EXP_EMBD, bias=False)
        self.ln_2      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.up_proj   = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.down_proj = nn.Linear(EXP_FFN, EXP_EMBD, bias=False)
        self.silu      = nn.SiLU()
        d2 = EXP_HD // 2
        self.register_buffer(
            "rope_ts",
            10000.0 ** ((2.0 / EXP_HD) * torch.arange(d2, dtype=torch.float32)),
            persistent=False,
        )
        self._kv_map = None

    def _get_kv_map(self, device):
        if self._kv_map is None or self._kv_map.device != device:
            self._kv_map = torch.div(
                torch.arange(EXP_Q_H, device=device) * EXP_KV_H,
                EXP_Q_H, rounding_mode="floor"
            )
        return self._kv_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h  = self.ln_1(x)
        B, L, _ = h.shape
        kv_map = self._get_kv_map(x.device)
        q = _rope(self.q_proj(h).view(B, L, EXP_Q_H, EXP_HD), self.rope_ts)
        k = _rope(self.k_proj(h).view(B, L, EXP_KV_H, EXP_HD), self.rope_ts)
        v = self.v_proj(h).view(B, L, EXP_KV_H, EXP_HD)
        qh = q.transpose(1, 2).float()
        kh = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(EXP_HD)
        mask = torch.ones(L, L, device=x.device).triu(1).bool()
        scores.masked_fill_(mask, float("-inf"))
        attn = scores.softmax(-1)
        vh  = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)
        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        return self.down_proj(self.silu(self.gate_proj(h)) * self.up_proj(h)) + residual


class ActionExpertCrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.q_proj    = nn.Linear(EXP_EMBD, EXP_Q_H * EXP_HD, bias=False)
        self.k_proj    = nn.Linear(EXP_KV_DIM, EXP_KV_DIM, bias=False)
        self.v_proj    = nn.Linear(EXP_KV_DIM, EXP_KV_DIM, bias=False)
        self.o_proj    = nn.Linear(EXP_Q_H * EXP_HD, EXP_EMBD, bias=False)
        self.ln_2      = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.up_proj   = nn.Linear(EXP_EMBD, EXP_FFN, bias=False)
        self.down_proj = nn.Linear(EXP_FFN, EXP_EMBD, bias=False)
        self.silu      = nn.SiLU()
        d2 = EXP_HD // 2
        self.register_buffer(
            "rope_ts",
            10000.0 ** ((2.0 / EXP_HD) * torch.arange(d2, dtype=torch.float32)),
            persistent=False,
        )
        self._kv_map = None

    def _get_kv_map(self, device):
        if self._kv_map is None or self._kv_map.device != device:
            self._kv_map = torch.div(
                torch.arange(EXP_Q_H, device=device) * EXP_KV_H,
                EXP_Q_H, rounding_mode="floor"
            )
        return self._kv_map

    def forward(self, x: torch.Tensor, text_k: torch.Tensor, text_v: torch.Tensor):
        residual = x
        h = self.ln_1(x)
        B, L, _  = h.shape
        _, Lc, _ = text_k.shape
        kv_map = self._get_kv_map(x.device)
        q = _rope(self.q_proj(h).view(B, L, EXP_Q_H, EXP_HD), self.rope_ts)
        k = self.k_proj(text_k).view(B, Lc, EXP_KV_H, EXP_HD)
        v = self.v_proj(text_v).view(B, Lc, EXP_KV_H, EXP_HD)
        qh = q.transpose(1, 2).float()
        kh = k.index_select(2, kv_map).transpose(1, 2).float()
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(EXP_HD)
        attn = scores.softmax(-1)
        vh  = v.index_select(2, kv_map).transpose(1, 2).float()
        ctx = (attn @ vh).transpose(1, 2).contiguous().view(B, L, -1).to(x.dtype)
        x = self.o_proj(ctx) + residual
        residual = x
        h = self.ln_2(x)
        return self.down_proj(self.silu(self.gate_proj(h)) * self.up_proj(h)) + residual


# ── Main model ────────────────────────────────────────────────────────────────

class VLAConditioned(nn.Module):
    """Text-conditioned VLA. Frozen: conv, vit, connector. Trainable: everything else."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)

        # ── Frozen vision backbone ────────────────────────────────────────────
        self.conv        = nn.Conv2d(3, VIT_EMBD, 16, stride=16, padding=0)
        self.vit_attn    = nn.MultiheadAttention(VIT_EMBD, VIT_HEADS, batch_first=True)
        self.vit_ln1     = nn.LayerNorm(VIT_EMBD)
        self.vit_ln2     = nn.LayerNorm(VIT_EMBD)
        self.vit_ffn_up  = nn.Linear(VIT_EMBD, VIT_FFN, bias=False)
        self.vit_ffn_dn  = nn.Linear(VIT_FFN, VIT_EMBD, bias=False)
        self.connector_w = nn.Parameter(torch.randn(12288, TEXT_EMBD) * 0.01)

        # ── Trainable text / state components ────────────────────────────────
        # Char-level learnable embedding (replaces fixed random buffer)
        self.text_embed = nn.Embedding(256, TEXT_EMBD)
        nn.init.normal_(self.text_embed.weight, std=0.02)

        # State projection: [STATE_DIM padded to 32, 960]
        self.state_w = nn.Parameter(torch.randn(32, TEXT_EMBD) * 0.01)

        # State encoder: 7D env state → 768D action queries
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.SiLU(),
            nn.Linear(256, 512),       nn.SiLU(),
            nn.Linear(512, EXP_EMBD),
        )

        # ── Transformer blocks (all trainable) ───────────────────────────────
        self.text_enc  = TextEncoderBlock()
        self.exp_self  = ActionExpertSelfBlock()
        self.exp_cross = ActionExpertCrossBlock()

        # ── Postprocessing ────────────────────────────────────────────────────
        self.post_norm = nn.RMSNorm(EXP_EMBD, elementwise_affine=True)
        self.post_proj = nn.Linear(EXP_EMBD, 32, bias=False)

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_vision(self):
        for name in ("conv", "vit_attn", "vit_ln1", "vit_ln2",
                     "vit_ffn_up", "vit_ffn_dn"):
            for p in getattr(self, name).parameters():
                p.requires_grad = False
        self.connector_w.requires_grad = False

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ── Forward pass for training (skips vision for speed) ───────────────────

    def forward(self, text_tokens: torch.Tensor, state7: torch.Tensor):
        """Training forward — skips frozen vision, returns [CHUNK_SIZE, 32] actions.

        Args:
            text_tokens: [SEQ_T] int64
            state7:      [7] float32

        Returns:
            actions: [CHUNK_SIZE, 32] — use first 2 cols as (dx, dy)
        """
        # Text embedding [SEQ_T, 960], scaled like original paper
        text_emb = self.text_embed(text_tokens) * math.sqrt(TEXT_EMBD)  # [48, 960]

        # State for mm_seq (state_w path — same as NPU)
        state_pad = torch.zeros(32, device=state7.device)
        state_pad[:STATE_DIM] = state7
        state_emb = (state_pad @ self.state_w).unsqueeze(0)  # [1, 960]

        # Vision placeholder (zeros — conv/vit frozen+random, not needed for training)
        vision_emb = torch.zeros(64, TEXT_EMBD, device=state7.device)

        # Assemble mm_seq [128, 960]
        zeros  = torch.zeros(PADDING, TEXT_EMBD, device=state7.device)
        mm_seq = torch.cat([vision_emb, text_emb, state_emb, zeros], dim=0)
        mm_seq = mm_seq.unsqueeze(0)  # [1, 128, 960]

        # Action queries from state encoder
        act_q = self.state_encoder(state7.unsqueeze(0))          # [1, 768]
        t_pos = torch.linspace(0, 1, CHUNK_SIZE, device=state7.device).unsqueeze(1)
        act_in = (act_q.expand(CHUNK_SIZE, EXP_EMBD) + t_pos * 0.1).unsqueeze(0)  # [1,32,768]

        # Layer 0: text encoder self-attn + action expert self-attn
        mm_out, _, _ = self.text_enc(mm_seq)
        act_out = self.exp_self(act_in)  # [1, 32, 768]

        # Layer 1: text encoder again + action expert cross-attn (text conditioning)
        mm_out2, text_k, text_v = self.text_enc(mm_out)
        act_out = self.exp_cross(act_out, text_k, text_v)  # [1, 32, 768]

        return self.post_proj(self.post_norm(act_out.squeeze(0)))  # [32, 32]

    # ── Inference wrapper (numpy I/O) ─────────────────────────────────────────

    @torch.no_grad()
    def predict(self, text: str, state7_np: np.ndarray) -> np.ndarray:
        """text + 7D env state → [CHUNK_SIZE, 2] float32 (dx, dy) actions."""
        tokens = torch.tensor(tokenize(text), dtype=torch.long)
        state7 = torch.tensor(state7_np, dtype=torch.float32)
        out = self.forward(tokens, state7)           # [32, 32]
        return out[:, :ACTION_DIM].numpy()           # [32, 2]

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state": self.state_dict(),
            "arch":        "VLAConditioned",
        }, path)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "VLAConditioned":
        model = cls(seed=seed)
        ckpt  = torch.load(path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
