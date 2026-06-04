"""
Pure PyTorch CPU reference implementations for VLA.
This module contains NO Allo imports, so it won't trigger any kernel builds.
"""

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import torch
import torch.nn as nn

# Import only the constants and PyTorch model classes (not Allo stuff)
from preprocessing_fused_bf16 import (
    CHANNELS as CH, PIX_LEN as PIX, KERNEL_H as KERNEL_DIM, EMBD_DIM as EMBD_P,
)

from connector_bf16 import TEXT

from text_encoder_bf16 import (
    EMBD as EMBD_TEXT,
    TextEncoderBlock,
)

from action_expert_bf16 import (
    EMBD as EMBD_EXP,
    ActionExpertSelfBlock, ActionExpertCrossBlock,
)

from vision_block_bf16 import MiniVit

# VLA Constants (from vla.py)
SEQ_T = 48
EMBD_S = TEXT  # 960
SEQ_S = 1
PADDING = 15
VIT_NUM_LAYERS = 12
LLAMA_NUM_LAYERS = 12
SKIP = 2
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
CHUNK_SIZE = 32  # EXP_SEQ

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
    return out_torch


def vit_ref(num_layers, input, params: dict):
    """Vision ViT encoder on CPU."""
    ref_model = MiniVit().eval()
    ref_model.attn.in_proj_weight.data = torch.tensor(
        np.concatenate([
            params["Wq"].T.astype(np.float32),
            params["Wk"].T.astype(np.float32),
            params["Wv"].T.astype(np.float32),
        ], axis=0)
    )
    ref_model.attn.out_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    ref_model.ffn_up.weight.data        = torch.tensor(params["W_up"].T.astype(np.float32))
    ref_model.ffn_down.weight.data      = torch.tensor(params["W_down"].T.astype(np.float32))
    ref_model.ln_1.weight.data          = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref_model.ln_1.bias.data            = torch.tensor(params["b_norm_1"].astype(np.float32))
    ref_model.ln_2.weight.data          = torch.tensor(params["W_norm_2"].astype(np.float32))
    ref_model.ln_2.bias.data            = torch.tensor(params["b_norm_2"].astype(np.float32))
    ref_model.to(torch.bfloat16)
    for _ in range(num_layers):
        with torch.no_grad():
            input = ref_model(input)
    return input.float().numpy().astype(np_bfloat16)


def con_ref(input: np.ndarray, params: dict):
    """Connector linear transformation on CPU."""
    input = input.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    ref = input @ params["W"]
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
                act_t = exp_cross_ref(act_t, text_k.unsqueeze(0), text_v.unsqueeze(0))

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
