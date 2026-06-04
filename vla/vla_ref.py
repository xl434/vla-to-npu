"""
Pure PyTorch CPU reference implementations for VLA.
This module contains NO Allo imports, so it won't trigger any kernel builds.
"""

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import MultiHeadAttention

# VLA Constants (from vla.py and component files)
SEQ_T = 48
EMBD_S = 960  # TEXT (no import from connector_bf16)
SEQ_S = 1
PADDING = 15
VIT_NUM_LAYERS = 12
LLAMA_NUM_LAYERS = 12
SKIP = 2
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
CHUNK_SIZE = 32

# Text Encoder dimensions
EMBD_TEXT = 960
TEXT_Q_H = 15
TEXT_KV_H = 5
TEXT_HEAD_DIM = 64
TEXT_FFN_HID = 2560

# Action Expert dimensions
EMBD_EXP = 768
EXP_Q_H = 12
EXP_KV_H = 4
EXP_HEAD_DIM = 64
EXP_FFN_HID = 2048

# Vision (ViT) dimensions
EMBD_VIT = 768
VIT_Q_H = 12
VIT_KV_H = 12
VIT_HEAD_DIM = 64
VIT_FFN_HID = 3072

# Preprocessing
CH = 3
PIX = 512
KERNEL_DIM = 16
EMBD_P = 768

# ============================================================
# PyTorch Model Classes (no Allo imports)
# ============================================================

class TextEncoderBlock(nn.Module):
    """Text encoder transformer block with RoPE."""
    def __init__(self):
        super().__init__()
        q_proj = nn.Linear(EMBD_TEXT, TEXT_Q_H * TEXT_HEAD_DIM, bias=False)
        k_proj = nn.Linear(EMBD_TEXT, TEXT_KV_H * TEXT_HEAD_DIM, bias=False)
        v_proj = nn.Linear(EMBD_TEXT, TEXT_KV_H * TEXT_HEAD_DIM, bias=False)
        o_proj = nn.Linear(TEXT_Q_H * TEXT_HEAD_DIM, EMBD_TEXT, bias=False)
        self.attn = MultiHeadAttention(
            embed_dim=TEXT_Q_H * TEXT_HEAD_DIM, num_heads=TEXT_Q_H, num_kv_heads=TEXT_KV_H,
            head_dim=TEXT_HEAD_DIM, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
            output_proj=o_proj, is_causal=True,
        )
        self.gate_proj = nn.Linear(EMBD_TEXT, TEXT_FFN_HID, bias=False)
        self.ln_1 = nn.RMSNorm(EMBD_TEXT, elementwise_affine=True)
        self.up_proj = nn.Linear(EMBD_TEXT, TEXT_FFN_HID, bias=False)
        self.down_proj = nn.Linear(TEXT_FFN_HID, EMBD_TEXT, bias=False)
        self.silu = nn.SiLU()
        self.ln_2 = nn.RMSNorm(EMBD_TEXT, elementwise_affine=True)
        self.max_wavelength = 10_000.0
        self.head_dim = TEXT_HEAD_DIM
        d_half = self.head_dim // 2
        freq_exponents = (2.0 / self.head_dim) * torch.arange(d_half, dtype=torch.float32)
        timescale = self.max_wavelength ** freq_exponents
        self.register_buffer("rope_timescale", timescale, persistent=False)
        self.register_buffer("pos_cache", torch.arange(0, 1, dtype=torch.float32), persistent=False)

    def _positions(self, L, device):
        if self.pos_cache.numel() < L:
            self.pos_cache = torch.arange(L, dtype=torch.float32, device=device)
        return self.pos_cache[:L].unsqueeze(0)

    def apply_rope(self, x, positions):
        B, L, H, D = x.shape
        d_half = D // 2
        ts = self.rope_timescale.to(x.device)
        radians = positions.to(torch.float32)[..., None] / ts[None, None, :]
        radians = radians[..., None, :]
        x = x.to(torch.float32)
        x1, x2 = x.split(d_half, dim=-1)
        s, c = torch.sin(radians), torch.cos(radians)
        out = torch.empty_like(x)
        out[..., :d_half] = x1 * c - x2 * s
        out[..., d_half:] = x2 * c + x1 * s
        return out.to(x.dtype)

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        B, L, _ = x.shape
        D = self.head_dim
        q = self.attn.q_proj(x).view(B, L, TEXT_Q_H, D)
        k = self.attn.k_proj(x).view(B, L, TEXT_KV_H, D)
        v = self.attn.v_proj(x).view(B, L, TEXT_KV_H, D)
        pos = self._positions(L, x.device).expand(B, -1)
        q = self.apply_rope(q, pos)
        k = self.apply_rope(k, pos)
        kv_map = torch.div(torch.arange(TEXT_Q_H, device=x.device) * TEXT_KV_H, TEXT_Q_H, rounding_mode='floor')
        k_sel = k.index_select(dim=2, index=kv_map)
        v_sel = v.index_select(dim=2, index=kv_map)
        q_h = q.transpose(1, 2)
        k_h = k_sel.transpose(1, 2)
        scores = torch.matmul(q_h.float(), k_h.float().transpose(-2, -1)) / (D ** 0.5)
        scores.masked_fill_(torch.ones(L, L, device=x.device).triu(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16).transpose(1, 2).contiguous().view(B, L, TEXT_Q_H * D)
        x = self.attn.output_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        act = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(act) + residual
        k_out = self.attn.k_proj(self.ln_1(residual)).view(B, L, TEXT_KV_H, D)
        v_out = self.attn.v_proj(self.ln_1(residual)).view(B, L, TEXT_KV_H, D)
        return x, k_out.squeeze(0).reshape(L, TEXT_KV_H * D), v_out.squeeze(0).reshape(L, TEXT_KV_H * D)


class ActionExpertSelfBlock(nn.Module):
    """Action expert self-attention block."""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
        self.q_proj = nn.Linear(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(EMBD_EXP, EXP_KV_H * EXP_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(EMBD_EXP, EXP_KV_H * EXP_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP, bias=False)
        self.ln_2 = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
        self.gate_proj = nn.Linear(EMBD_EXP, EXP_FFN_HID, bias=False)
        self.up_proj = nn.Linear(EMBD_EXP, EXP_FFN_HID, bias=False)
        self.down_proj = nn.Linear(EXP_FFN_HID, EMBD_EXP, bias=False)
        self.silu = nn.SiLU()
        self.max_wavelength = 10_000.0
        d_half = EXP_HEAD_DIM // 2
        freq_exponents = (2.0 / EXP_HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
        timescale = self.max_wavelength ** freq_exponents
        self.register_buffer("rope_timescale", timescale, persistent=False)

    def apply_rope(self, x, L):
        B, _, H, D = x.shape
        d_half = D // 2
        ts = self.rope_timescale.to(x.device)
        positions = torch.arange(L, dtype=torch.float32, device=x.device).unsqueeze(0).expand(B, -1)
        radians = positions.to(torch.float32)[..., None] / ts[None, None, :]
        radians = radians[..., None, :]
        x = x.to(torch.float32)
        x1, x2 = x.split(d_half, dim=-1)
        s, c = torch.sin(radians), torch.cos(radians)
        out = torch.empty_like(x)
        out[..., :d_half] = x1 * c - x2 * s
        out[..., d_half:] = x2 * c + x1 * s
        return out.to(x.dtype)

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, EXP_Q_H, EXP_HEAD_DIM)
        k = self.k_proj(x).view(B, L, EXP_KV_H, EXP_HEAD_DIM)
        v = self.v_proj(x).view(B, L, EXP_KV_H, EXP_HEAD_DIM)
        q = self.apply_rope(q, L)
        k = self.apply_rope(k, L)
        kv_map = torch.div(torch.arange(EXP_Q_H, device=x.device) * EXP_KV_H, EXP_Q_H, rounding_mode='floor')
        k_sel = k.index_select(2, kv_map)
        v_sel = v.index_select(2, kv_map)
        q_h = q.transpose(1, 2).float()
        k_h = k_sel.transpose(1, 2).float()
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (EXP_HEAD_DIM ** 0.5)
        scores.masked_fill_(torch.ones(L, L, device=x.device).triu(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, EXP_Q_H * EXP_HEAD_DIM)
        x = self.o_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


class ActionExpertCrossBlock(nn.Module):
    """Action expert cross-attention block."""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
        self.q_proj = nn.Linear(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(TEXT_KV_H * TEXT_HEAD_DIM, TEXT_KV_H * TEXT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TEXT_KV_H * TEXT_HEAD_DIM, TEXT_KV_H * TEXT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP, bias=False)
        self.ln_2 = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
        self.gate_proj = nn.Linear(EMBD_EXP, EXP_FFN_HID, bias=False)
        self.up_proj = nn.Linear(EMBD_EXP, EXP_FFN_HID, bias=False)
        self.down_proj = nn.Linear(EXP_FFN_HID, EMBD_EXP, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x, k, v):
        residual = x
        x = self.ln_1(x)
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, EXP_Q_H, EXP_HEAD_DIM)
        k_proj = k.view(B, -1, TEXT_KV_H, TEXT_HEAD_DIM)
        v_proj = v.view(B, -1, TEXT_KV_H, TEXT_HEAD_DIM)
        kv_map = torch.div(torch.arange(EXP_Q_H, device=x.device) * TEXT_KV_H, EXP_Q_H, rounding_mode='floor')
        k_sel = k_proj.index_select(2, kv_map)
        v_sel = v_proj.index_select(2, kv_map)
        q_h = q.transpose(1, 2).float()
        k_h = k_sel.transpose(1, 2).float()
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (EXP_HEAD_DIM ** 0.5)
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, EXP_Q_H * EXP_HEAD_DIM)
        x = self.o_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


class MiniVit(nn.Module):
    """Vision ViT block."""
    def __init__(self):
        super().__init__()
        q_proj = nn.Linear(EMBD_VIT, VIT_Q_H * VIT_HEAD_DIM, bias=False)
        k_proj = nn.Linear(EMBD_VIT, VIT_KV_H * VIT_HEAD_DIM, bias=False)
        v_proj = nn.Linear(EMBD_VIT, VIT_KV_H * VIT_HEAD_DIM, bias=False)
        o_proj = nn.Linear(VIT_Q_H * VIT_HEAD_DIM, EMBD_VIT, bias=False)
        self.attn = MultiHeadAttention(
            embed_dim=VIT_Q_H * VIT_HEAD_DIM, num_heads=VIT_Q_H, num_kv_heads=VIT_KV_H,
            head_dim=VIT_HEAD_DIM, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
            output_proj=o_proj, is_causal=False,
        )
        self.ffn_up = nn.Linear(EMBD_VIT, VIT_FFN_HID, bias=False)
        self.ffn_down = nn.Linear(VIT_FFN_HID, EMBD_VIT, bias=False)
        self.ln_1 = nn.RMSNorm(EMBD_VIT, elementwise_affine=True)
        self.ln_2 = nn.RMSNorm(EMBD_VIT, elementwise_affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        B, L, _ = x.shape
        D = VIT_HEAD_DIM
        q = self.attn.q_proj(x).view(B, L, VIT_Q_H, D)
        k = self.attn.k_proj(x).view(B, L, VIT_KV_H, D)
        v = self.attn.v_proj(x).view(B, L, VIT_KV_H, D)
        kv_map = torch.div(torch.arange(VIT_Q_H, device=x.device) * VIT_KV_H, VIT_Q_H, rounding_mode='floor')
        k_sel = k.index_select(dim=2, index=kv_map)
        v_sel = v.index_select(dim=2, index=kv_map)
        q_h = q.transpose(1, 2)
        k_h = k_sel.transpose(1, 2)
        scores = torch.matmul(q_h.float(), k_h.float().transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16).transpose(1, 2).contiguous().view(B, L, VIT_Q_H * D)
        x = self.attn.output_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        x = self.ffn_down(self.gelu(self.ffn_up(x))) + residual
        return x


# ============================================================
# Pure PyTorch Reference Functions (CPU only, no NPU/Allo)
# ============================================================

def preproc_ref(input: np.ndarray, params: dict):
    """Preprocessing: Conv2d on CPU."""
    input_torch = torch.tensor(input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    conv = nn.Conv2d(
        in_channels=CH, out_channels=EMBD_P,
        kernel_size=KERNEL_DIM, stride=KERNEL_DIM, padding=0,
    ).to(torch.bfloat16)
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.tensor(params["kernel"].astype(np.float32)).to(torch.bfloat16))
        conv.bias   = nn.Parameter(torch.zeros(EMBD_P, dtype=torch.bfloat16))
    with torch.no_grad():
        out_torch = conv(input_torch)              # [1, 768, 32, 32]
    out_torch = out_torch.squeeze(0).flatten(1).transpose(0, 1)  # [1024, 768]
    return out_torch.float().numpy().astype(np_bfloat16)


def vit_ref(num_layers, input, params: dict):
    """Vision ViT encoder on CPU."""
    ref_model = MiniVit().eval()
    ref_model.attn.q_proj.weight.data   = torch.tensor(params["Wq"].T.astype(np.float32))
    ref_model.attn.k_proj.weight.data   = torch.tensor(params["Wk"].T.astype(np.float32))
    ref_model.attn.v_proj.weight.data   = torch.tensor(params["Wv"].T.astype(np.float32))
    ref_model.attn.output_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    ref_model.ffn_up.weight.data        = torch.tensor(params["W_up"].T.astype(np.float32))
    ref_model.ffn_down.weight.data      = torch.tensor(params["W_down"].T.astype(np.float32))
    ref_model.ln_1.weight.data          = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref_model.ln_2.weight.data          = torch.tensor(params["W_norm_2"].astype(np.float32))
    ref_model.to(torch.bfloat16)
    # Convert input to torch if needed
    if isinstance(input, np.ndarray):
        input = torch.tensor(input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    else:
        input = input.unsqueeze(0).to(torch.bfloat16)
    for _ in range(num_layers):
        with torch.no_grad():
            input = ref_model(input)
    return input.squeeze(0).float().numpy().astype(np_bfloat16)


def con_ref(input: np.ndarray, params: dict):
    """Connector linear transformation on CPU."""
    input = input.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    ref = (input @ params["W"]).astype(np_bfloat16)
    return ref


def make_text_encoder_ref(params):
    """Create PyTorch text encoder block."""
    ref = TextEncoderBlock().eval()
    p = ref
    p.attn.q_proj.weight.data      = torch.tensor(params["Wq"].T.astype(np.float32))
    p.attn.k_proj.weight.data      = torch.tensor(params["Wk"].T.astype(np.float32))
    p.attn.v_proj.weight.data      = torch.tensor(params["Wv"].T.astype(np.float32))
    p.attn.output_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    p.gate_proj.weight.data        = torch.tensor(params["W_gate"].T.astype(np.float32))
    p.up_proj.weight.data          = torch.tensor(params["W_up"].T.astype(np.float32))
    p.down_proj.weight.data        = torch.tensor(params["W_down"].T.astype(np.float32))
    p.ln_1.weight.data             = torch.tensor(params["W_norm_1"].astype(np.float32))
    p.ln_2.weight.data             = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def make_exp_self_ref(params):
    """Create PyTorch action expert self-attention block."""
    ref = ActionExpertSelfBlock().eval()
    ref.q_proj.weight.data    = torch.tensor(params["Wq"].T.astype(np.float32))
    ref.k_proj.weight.data    = torch.tensor(params["Wk"].T.astype(np.float32))
    ref.v_proj.weight.data    = torch.tensor(params["Wv"].T.astype(np.float32))
    ref.o_proj.weight.data    = torch.tensor(params["Wo"].T.astype(np.float32))
    ref.gate_proj.weight.data = torch.tensor(params["W_gate"].T.astype(np.float32))
    ref.up_proj.weight.data   = torch.tensor(params["W_up"].T.astype(np.float32))
    ref.down_proj.weight.data = torch.tensor(params["W_down"].T.astype(np.float32))
    ref.ln_1.weight.data      = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref.ln_2.weight.data      = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def make_exp_cross_ref(params):
    """Create PyTorch action expert cross-attention block."""
    ref = ActionExpertCrossBlock().eval()
    ref.q_proj.weight.data    = torch.tensor(params["Wq"].T.astype(np.float32))
    ref.k_proj.weight.data    = torch.tensor(params["Wk_cross"].T.astype(np.float32))
    ref.v_proj.weight.data    = torch.tensor(params["Wv_cross"].T.astype(np.float32))
    ref.o_proj.weight.data    = torch.tensor(params["Wo"].T.astype(np.float32))
    ref.gate_proj.weight.data = torch.tensor(params["W_gate"].T.astype(np.float32))
    ref.up_proj.weight.data   = torch.tensor(params["W_up"].T.astype(np.float32))
    ref.down_proj.weight.data = torch.tensor(params["W_down"].T.astype(np.float32))
    ref.ln_1.weight.data      = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref.ln_2.weight.data      = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def action_expert_cross_forward(x, k, v):
    """Cross-attention forward with pre-computed k,v."""
    return x.forward(k, v)


def joint_transformer_ref(num_layers, vlm_input, action, params_vlm,
                          params_exp_self, params_exp_cross):
    """Joint text+action transformer on CPU."""
    vlm_ref = make_text_encoder_ref(params_vlm).to(torch.bfloat16)
    exp_self_ref = make_exp_self_ref(params_exp_self).to(torch.bfloat16)
    exp_cross_ref = make_exp_cross_ref(params_exp_cross).to(torch.bfloat16)

    vlm_t = torch.tensor(vlm_input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    act_t = torch.tensor(action.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)

    for i in range(num_layers):
        with torch.no_grad():
            vlm_t, text_k, text_v = vlm_ref(vlm_t)
            if i % SKIP == 0:
                act_t = exp_self_ref(act_t)
            else:
                # text_k, text_v have shape [1, seq, kv_dim], we pass them directly
                act_t = exp_cross_ref(act_t, text_k, text_v)

    act_out = act_t.squeeze(0).float().numpy().astype(np_bfloat16)
    return act_out


def postprocessing_ref(out, params_out):
    """Postprocessing: RMS norm + projection on CPU."""
    out_t = torch.tensor(out.astype(np.float32)).to(torch.bfloat16)
    rms = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
    rms.weight.data = torch.tensor(params_out["W_exp_norm"].astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        normed = rms(out_t)

    proj = nn.Linear(EMBD_EXP, MAX_STATE_DIM, bias=False)
    proj.weight.data = torch.tensor(params_out["W_action_out"].T.astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        v_t_ref = proj(normed)

    return v_t_ref.float().numpy().astype(np_bfloat16)
