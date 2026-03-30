# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLA Action Expert Layer (BFloat16)

Single action-expert transformer block for NPU.
Based on llama_block_rope_bf16.py and text_encoder_bf16.py.
  SEQ=32, EMBD=768, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2048

Two modes:
  - Self-attention (even layers): causal self-attention
  - Cross-attention (odd layers): Q from action, K/V from text encoder [128, 320]

Exported functions:
  - action_expert_self_forward(x, params) → output [32, 768]
  - action_expert_cross_forward(x, text_k, text_v, params) → output [32, 768]
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16 as Ty_bf16, int32
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from allo.library.aie.modules.gemm import GEMM

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

# ===============================================================================
# Model Configuration
# ===============================================================================
KERNEL_LIB_PATH = "../cc/float/"
KERNEL_BF16_PATH = "../cc/bf16/"
KERNEL_BF16_OLD_PATH = "../cc/bf16_old/"

BATCH = 1
SEQ = 32              # action sequence length
TEXT_SEQ = 128        # text encoder sequence length (for cross-attention K/V)
EMBD = 768            # action expert embedding dim
Q_H = 15
KV_H = 5
HEAD_DIM = 64
KV_DIM = KV_H * HEAD_DIM  # 320
FFN_HID = 2048

assert EMBD % 64 == 0
assert HEAD_DIM % 64 == 0

Ty = Ty_bf16
NP_DTYPE = np_bfloat16
LINEAR_TILE = 64
ATTN_TILE = 32

# ===============================================================================
# PyTorch Reference: Self-Attention Expert Block
# ===============================================================================
class ActionExpertSelfBlock(nn.Module):
    """Self-attention action expert (even layers)."""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.q_proj = nn.Linear(EMBD, Q_H * HEAD_DIM, bias=False)    # 768→960
        self.k_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)   # 768→320
        self.v_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)   # 768→320
        self.o_proj = nn.Linear(Q_H * HEAD_DIM, EMBD, bias=False)    # 960→768
        self.ln_2 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EMBD, FFN_HID, bias=False)        # 768→2048
        self.up_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.down_proj = nn.Linear(FFN_HID, EMBD, bias=False)        # 2048→768
        self.silu = nn.SiLU()
        self.max_wavelength = 10_000.0
        d_half = HEAD_DIM // 2
        freq_exponents = (2.0 / HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
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
        q = self.q_proj(x).view(B, L, Q_H, HEAD_DIM)
        k = self.k_proj(x).view(B, L, KV_H, HEAD_DIM)
        v = self.v_proj(x).view(B, L, KV_H, HEAD_DIM)
        q = self.apply_rope(q, L)
        k = self.apply_rope(k, L)
        kv_map = torch.div(torch.arange(Q_H, device=x.device) * KV_H, Q_H, rounding_mode='floor')
        k_sel = k.index_select(2, kv_map)
        v_sel = v.index_select(2, kv_map)
        q_h = q.transpose(1, 2).float()
        k_h = k_sel.transpose(1, 2).float()
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        scores.masked_fill_(torch.ones(L, L, device=x.device).triu(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, Q_H * HEAD_DIM)
        x = self.o_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


# ===============================================================================
# PyTorch Reference: Cross-Attention Expert Block
# ===============================================================================
class ActionExpertCrossBlock(nn.Module):
    """Cross-attention action expert (odd layers).
    Q from action input, K/V from text encoder output."""
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.q_proj = nn.Linear(EMBD, Q_H * HEAD_DIM, bias=False)    # 768→960
        self.k_proj = nn.Linear(KV_DIM, KV_DIM, bias=False)          # 320→320
        self.v_proj = nn.Linear(KV_DIM, KV_DIM, bias=False)          # 320→320
        self.o_proj = nn.Linear(Q_H * HEAD_DIM, EMBD, bias=False)    # 960→768
        self.ln_2 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.gate_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.up_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.down_proj = nn.Linear(FFN_HID, EMBD, bias=False)
        self.silu = nn.SiLU()
        self.max_wavelength = 10_000.0
        d_half = HEAD_DIM // 2
        freq_exponents = (2.0 / HEAD_DIM) * torch.arange(d_half, dtype=torch.float32)
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

    def forward(self, x, text_k, text_v):
        """
        x: [B, SEQ, EMBD] action input
        text_k: [B, TEXT_SEQ, KV_DIM] key from text encoder
        text_v: [B, TEXT_SEQ, KV_DIM] value from text encoder
        """
        residual = x
        x_norm = self.ln_1(x)
        B, L, _ = x_norm.shape
        _, Lc, _ = text_k.shape

        q = self.q_proj(x_norm).view(B, L, Q_H, HEAD_DIM)
        k = self.k_proj(text_k).view(B, Lc, KV_H, HEAD_DIM)
        v = self.v_proj(text_v).view(B, Lc, KV_H, HEAD_DIM)

        # RoPE on Q only for cross-attention
        q = self.apply_rope(q, L)

        kv_map = torch.div(torch.arange(Q_H, device=x.device) * KV_H, Q_H, rounding_mode='floor')
        k_sel = k.index_select(2, kv_map)
        v_sel = v.index_select(2, kv_map)

        q_h = q.transpose(1, 2).float()   # (B, Q_H, L, HEAD_DIM)
        k_h = k_sel.transpose(1, 2).float()  # (B, Q_H, Lc, HEAD_DIM)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        # No causal mask for cross-attention
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)

        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, Q_H * HEAD_DIM)

        x = self.o_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


# ===============================================================================
# Allo BF16 Version — External Modules
# ===============================================================================

# ----------------------------------------------------------------
# RMSNorm (bf16, width=768) — reuse existing kernel
# ----------------------------------------------------------------
norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path=KERNEL_BF16_OLD_PATH + "rms_norm_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
NORM_P0 = 4
NORM_SEQ_TILE = 16
norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def rms_norm_kernel(
    A: Ty[NORM_SEQ_TILE, EMBD],
    B: Ty[EMBD],
    C: Ty[NORM_SEQ_TILE, EMBD],
):
    @df.kernel(mapping=[NORM_P0], args=[A, B, C])
    def core(
        local_A: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_B: Ty[EMBD] @ norm_arg_layout,
        local_C: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        norm(local_A, local_B, local_C)

# ----------------------------------------------------------------
# GEMM (bf16) — all linear projections
# ----------------------------------------------------------------

# Q projection: 32 x 960 x 768 (Pm=1, Pn=15, Pk=12)
gemm_q_kernel, gemm_q_mp = GEMM(
    SEQ, Q_H * HEAD_DIM, EMBD,
    1, (Q_H * HEAD_DIM) // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# K/V projection (self-attn): 32 x 320 x 768 (Pm=1, Pn=5, Pk=12)
gemm_kv_self_kernel, gemm_kv_self_mp = GEMM(
    SEQ, KV_DIM, EMBD,
    1, KV_DIM // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# K/V projection (cross-attn): 128 x 320 x 320 (Pm=2, Pn=5, Pk=5)
gemm_kv_cross_kernel, gemm_kv_cross_mp = GEMM(
    TEXT_SEQ, KV_DIM, KV_DIM,
    TEXT_SEQ // LINEAR_TILE, KV_DIM // LINEAR_TILE, KV_DIM // LINEAR_TILE,
    Ty, Ty,
)

# Output projection: 32 x 768 x 960 (Pm=1, Pn=12, Pk=15)
gemm_out_kernel, gemm_out_mp = GEMM(
    SEQ, EMBD, Q_H * HEAD_DIM,
    1, EMBD // LINEAR_TILE, (Q_H * HEAD_DIM) // LINEAR_TILE,
    Ty, Ty,
)

# Self-attn score: 32 x 32 x 64 (Pm=1, Pn=1, Pk=2, tile=32)
# Q_scaled[32,64] × K^T[64,32] → score[32,32]
gemm_attn_self_score_kernel, gemm_attn_self_score_mp = GEMM(
    SEQ, SEQ, HEAD_DIM,
    SEQ // ATTN_TILE, SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE,
    Ty, Ty,
)

# Self-attn value: 32 x 64 x 32 (Pm=1, Pn=2, Pk=1, tile=32)
# weights[32,32] × V[32,64] → out[32,64]
gemm_attn_self_value_kernel, gemm_attn_self_value_mp = GEMM(
    SEQ, HEAD_DIM, SEQ,
    SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE, SEQ // ATTN_TILE,
    Ty, Ty,
)

# Cross-attn score: 32 x 128 x 64 (Pm=1, Pn=4, Pk=2, tile=32)
# Q_scaled[32,64] × K^T[64,128] → score[32,128]
gemm_attn_cross_score_kernel, gemm_attn_cross_score_mp = GEMM(
    SEQ, TEXT_SEQ, HEAD_DIM,
    SEQ // ATTN_TILE, TEXT_SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE,
    Ty, Ty,
)

# Cross-attn value: 32 x 64 x 128 (Pm=1, Pn=2, Pk=4, tile=32)
# weights[32,128] × V[128,64] → out[32,64]
gemm_attn_cross_value_kernel, gemm_attn_cross_value_mp = GEMM(
    SEQ, HEAD_DIM, TEXT_SEQ,
    SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE, TEXT_SEQ // ATTN_TILE,
    Ty, Ty,
)

# Gate/Up projection: 32 x 2048 x 768 (Pm=1, Pn=32, Pk=12)
gemm_ffn_up_kernel, gemm_ffn_up_mp = GEMM(
    SEQ, FFN_HID, EMBD,
    1, FFN_HID // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# FFN down: K=2048 too large, chunk as 8 x GEMM(32, 768, 256)
# 32 x 768 x 256 (Pm=1, Pn=12, Pk=4)
FFN_DOWN_K_CHUNK = 256
FFN_DOWN_K_CHUNKS = FFN_HID // FFN_DOWN_K_CHUNK  # 2048/256 = 8
gemm_ffn_down_kernel, gemm_ffn_down_mp = GEMM(
    SEQ, EMBD, FFN_DOWN_K_CHUNK,
    1, EMBD // LINEAR_TILE, FFN_DOWN_K_CHUNK // LINEAR_TILE,
    Ty, Ty,
)

# ----------------------------------------------------------------
# Unmasked Softmax (bf16) — for cross-attention [32, 128]
# ----------------------------------------------------------------
softmax_cross_ext = ExternalModule(
    top="softmax_bf16_32_128",
    impl_path=KERNEL_BF16_OLD_PATH + "v2_softmax_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)

@df.region()
def softmax_cross_kernel(
    input_x: Ty[ATTN_TILE, TEXT_SEQ],
    output_x: Ty[ATTN_TILE, TEXT_SEQ],
):
    @df.kernel(mapping=[1, 1], args=[input_x, output_x])
    def core(
        local_input_x: Ty[ATTN_TILE, TEXT_SEQ] @ [S(0), S(1)],
        local_output_x: Ty[ATTN_TILE, TEXT_SEQ] @ [S(0), S(1)],
    ):
        softmax_cross_ext(local_input_x, local_output_x)

# ----------------------------------------------------------------
# SiLU (bf16, FFN_HID=2048, per-core [4][256], P1=8)
# ----------------------------------------------------------------
silu_ext = ExternalModule(
    top="silu_256_bf16",
    impl_path=KERNEL_BF16_PATH + "silu_256_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)
SILU_P0 = 1
SILU_P1 = 8   # 2048 / 256 = 8 cores on feature dim
SILU_SEQ_TILE = 4
SILU_Ly = [S(0), S(1)]

@df.region()
def silu_kernel(
    input_x: Ty[SILU_SEQ_TILE, FFN_HID],
    output_x: Ty[SILU_SEQ_TILE, FFN_HID],
):
    @df.kernel(mapping=[SILU_P0, SILU_P1], args=[input_x, output_x])
    def core(
        local_input_x: Ty[SILU_SEQ_TILE, FFN_HID] @ SILU_Ly,
        local_output_x: Ty[SILU_SEQ_TILE, FFN_HID] @ SILU_Ly,
    ):
        silu_ext(local_input_x, local_output_x)

# ----------------------------------------------------------------
# RoPE (float32 — trig functions need precision)
# ----------------------------------------------------------------
HEAD_DIM_HALF = HEAD_DIM // 2
ROPE_TILE = 64  # RoPE kernels always process 64 rows; pad SEQ=32 → 64

VecLy = [S(0)]
MatLy = [S(1), S(0)]
OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"
SIN_IMPL = KERNEL_LIB_PATH + "sine.cc"
COS_IMPL = KERNEL_LIB_PATH + "cosine.cc"

Ty_rope = float32

radians_ext = ExternalModule(top="rope_make_radians_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
pack_ext    = ExternalModule(top="pack32to64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyL_ext   = ExternalModule(top="copy_left32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyR_ext   = ExternalModule(top="copy_right32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
join_ext    = ExternalModule(top="join32_to_64_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
mul32_ext   = ExternalModule(top="mul32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
add32_ext   = ExternalModule(top="add32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sub32_ext   = ExternalModule(top="sub32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sin_ext     = ExternalModule(top="sin_float32", impl_path=SIN_IMPL, input_idx=[0], output_idx=[1])
cos_ext     = ExternalModule(top="cos_float32", impl_path=COS_IMPL, input_idx=[0], output_idx=[1])

@df.region()
def radians_region(positions: Ty_rope[ROPE_TILE], inv_ts: Ty_rope[HEAD_DIM_HALF], radians32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[positions, inv_ts, radians32])
    def core(lp: Ty_rope[ROPE_TILE] @ VecLy, li: Ty_rope[HEAD_DIM_HALF] @ VecLy, lr: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        radians_ext(lp, li, lr)

@df.region()
def pack_region(r32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], r64: Ty_rope[ROPE_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[r32, r64])
    def core(lr32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lr64: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy):
        pack_ext(lr32, lr64)

@df.region()
def sin_region(i64: Ty_rope[ROPE_TILE, HEAD_DIM], o64: Ty_rope[ROPE_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[i64, o64])
    def core(li: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy):
        sin_ext(li, lo)

@df.region()
def cos_region(i64: Ty_rope[ROPE_TILE, HEAD_DIM], o64: Ty_rope[ROPE_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[i64, o64])
    def core(li: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy):
        cos_ext(li, lo)

@df.region()
def copy_left_region(i64: Ty_rope[ROPE_TILE, HEAD_DIM], o32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        copyL_ext(li, lo)

@df.region()
def copy_right_region(i64: Ty_rope[ROPE_TILE, HEAD_DIM], o32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        copyR_ext(li, lo)

@df.region()
def join_region(l32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], r32: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], o64: Ty_rope[ROPE_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[l32, r32, o64])
    def core(ll: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lr: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lo: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy):
        join_ext(ll, lr, lo)

@df.region()
def mul32_region(A: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        mul32_ext(la, lb, lc)

@df.region()
def add32_region(A: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        add32_ext(la, lb, lc)

@df.region()
def sub32_region(A: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_TILE, HEAD_DIM_HALF] @ MatLy):
        sub32_ext(la, lb, lc)


# ##############################################################
# BUILD
# ##############################################################
import sys
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

# Parse --mode flag: "self", "cross", or "all" (default)
_BUILD_MODE = "all"
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _BUILD_MODE = sys.argv[i + 1]

_BUILD_SELF = _BUILD_MODE in ("self", "all")
_BUILD_CROSS = _BUILD_MODE in ("cross", "all")

# Shared modules (needed by both modes)
rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="action_expert_bf16/rms_norm.prj")
gemm_q_mod = df.build(gemm_q_kernel, project="action_expert_bf16/gemm_q.prj", target="aie", mapping_primitives=gemm_q_mp)
gemm_out_mod = df.build(gemm_out_kernel, project="action_expert_bf16/gemm_out.prj", target="aie", mapping_primitives=gemm_out_mp)
gemm_ffn_up_mod = df.build(gemm_ffn_up_kernel, project="action_expert_bf16/gemm_ffn_up.prj", target="aie", mapping_primitives=gemm_ffn_up_mp)
gemm_ffn_down_mod = df.build(gemm_ffn_down_kernel, project="action_expert_bf16/gemm_ffn_down.prj", target="aie", mapping_primitives=gemm_ffn_down_mp)
silu_mod = df.build(silu_kernel, target="aie", project="action_expert_bf16/silu.prj")

# RoPE modules (shared)
radians_mod = df.build(radians_region, target="aie", project="action_expert_bf16/rope/radians.prj")
pack_mod    = df.build(pack_region, target="aie", project="action_expert_bf16/rope/pack.prj")
sin_mod     = df.build(sin_region, target="aie", project="action_expert_bf16/rope/sin.prj")
cos_mod     = df.build(cos_region, target="aie", project="action_expert_bf16/rope/cos.prj")
copyL_mod   = df.build(copy_left_region, target="aie", project="action_expert_bf16/rope/copyL.prj")
copyR_mod   = df.build(copy_right_region, target="aie", project="action_expert_bf16/rope/copyR.prj")
join_mod    = df.build(join_region, target="aie", project="action_expert_bf16/rope/join.prj")
mul32_mod   = df.build(mul32_region, target="aie", project="action_expert_bf16/rope/mul32.prj")
add32_mod   = df.build(add32_region, target="aie", project="action_expert_bf16/rope/add32.prj")
sub32_mod   = df.build(sub32_region, target="aie", project="action_expert_bf16/rope/sub32.prj")

# Self-attention specific
gemm_kv_self_mod = None
gemm_attn_self_score_mod = None
gemm_attn_self_value_mod = None
if _BUILD_SELF:
    gemm_kv_self_mod = df.build(gemm_kv_self_kernel, project="action_expert_bf16/gemm_kv_self.prj", target="aie", mapping_primitives=gemm_kv_self_mp)
    gemm_attn_self_score_mod = df.build(gemm_attn_self_score_kernel, project="action_expert_bf16/gemm_attn_self_score.prj", target="aie", mapping_primitives=gemm_attn_self_score_mp)
    gemm_attn_self_value_mod = df.build(gemm_attn_self_value_kernel, project="action_expert_bf16/gemm_attn_self_value.prj", target="aie", mapping_primitives=gemm_attn_self_value_mp)

# Cross-attention specific
gemm_kv_cross_mod = None
gemm_attn_cross_score_mod = None
gemm_attn_cross_value_mod = None
softmax_cross_mod = None
if _BUILD_CROSS:
    gemm_kv_cross_mod = df.build(gemm_kv_cross_kernel, project="action_expert_bf16/gemm_kv_cross.prj", target="aie", mapping_primitives=gemm_kv_cross_mp)
    gemm_attn_cross_score_mod = df.build(gemm_attn_cross_score_kernel, project="action_expert_bf16/gemm_attn_cross_score.prj", target="aie", mapping_primitives=gemm_attn_cross_score_mp)
    gemm_attn_cross_value_mod = df.build(gemm_attn_cross_value_kernel, project="action_expert_bf16/gemm_attn_cross_value.prj", target="aie", mapping_primitives=gemm_attn_cross_value_mp)
    softmax_cross_mod = df.build(softmax_cross_kernel, target="aie", project="action_expert_bf16/softmax_cross.prj")


# ##############################################################
# TOOL FUNCTIONS
# ##############################################################

def rmsnorm(input_x, weight, output_x, seq_len=SEQ):
    """RMSNorm over tiles of NORM_SEQ_TILE rows."""
    for i in range(seq_len // NORM_SEQ_TILE):
        rms_norm_mod(
            input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
            weight,
            output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
        )


ATTN_SCALE = NP_DTYPE(1.0 / (HEAD_DIM ** 0.5))


def masked_softmax_cpu(scores_bf16):
    """CPU causal masked softmax for [SEQ, SEQ] attention scores."""
    scores_f32 = scores_bf16.astype(np.float32)
    mask = np.triu(np.ones((SEQ, SEQ)), k=1).astype(bool)
    scores_f32[mask] = -np.inf
    # Numerically stable softmax
    row_max = scores_f32.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores_f32 - row_max)
    return (exp_scores / exp_scores.sum(axis=-1, keepdims=True)).astype(NP_DTYPE)


def unmasked_softmax_fn(score_per_head, weight_per_head):
    """NPU unmasked softmax for cross-attention [32, 128] per head."""
    softmax_cross_mod(score_per_head, weight_per_head)


def rope_apply_packed(packed_bf16, heads, head_dim=64, max_wavelength=10_000.0, pos_offset=0):
    """RoPE in float32. Converts bf16→float32, applies RoPE, returns bf16.
    Pads to ROPE_TILE=64 rows per tile."""
    packed = packed_bf16.astype(np.float32)
    seq_len, total_dim = packed.shape
    D = head_dim
    HALF = D // 2

    out = np.empty_like(packed, dtype=np.float32)
    k = np.arange(HALF, dtype=np.float32)
    inv_ts = (max_wavelength ** (-(2.0 / D) * k)).astype(np.float32)

    for t0 in range(0, seq_len, ROPE_TILE):
        rows = min(ROPE_TILE, seq_len - t0)
        pos32 = (pos_offset + np.arange(t0, t0 + rows, dtype=np.float32)).astype(np.float32)
        pos_pad = np.zeros(ROPE_TILE, dtype=np.float32)
        pos_pad[:rows] = pos32

        radians32 = np.zeros((ROPE_TILE, HALF), dtype=np.float32)
        radians_mod(pos_pad, inv_ts, radians32)
        radians64 = np.zeros((ROPE_TILE, D), dtype=np.float32)
        pack_mod(radians32, radians64)

        sin64 = np.zeros((ROPE_TILE, D), dtype=np.float32)
        cos64 = np.zeros((ROPE_TILE, D), dtype=np.float32)
        sin_mod(radians64, sin64)
        cos_mod(radians64, cos64)

        for h in range(heads):
            x_tile = np.zeros((ROPE_TILE, D), dtype=np.float32)
            x_tile[:rows, :] = packed[t0:t0 + rows, h * D:(h + 1) * D]

            xL = np.zeros((ROPE_TILE, HALF), dtype=np.float32)
            xR = np.zeros((ROPE_TILE, HALF), dtype=np.float32)
            s  = np.zeros((ROPE_TILE, HALF), dtype=np.float32)
            c  = np.zeros((ROPE_TILE, HALF), dtype=np.float32)

            copyL_mod(x_tile, xL)
            copyR_mod(x_tile, xR)
            copyL_mod(sin64, s)
            copyL_mod(cos64, c)

            tmp1 = np.zeros_like(xL); mul32_mod(xL, c, tmp1)
            tmp2 = np.zeros_like(xL); mul32_mod(xR, s, tmp2)
            yL   = np.zeros_like(xL); sub32_mod(tmp1, tmp2, yL)

            tmp3 = np.zeros_like(xL); mul32_mod(xR, c, tmp3)
            tmp4 = np.zeros_like(xL); mul32_mod(xL, s, tmp4)
            yR   = np.zeros_like(xL); add32_mod(tmp3, tmp4, yR)

            y64 = np.zeros((ROPE_TILE, D), dtype=np.float32)
            join_mod(yL, yR, y64)
            out[t0:t0 + rows, h * D:(h + 1) * D] = y64[:rows, :]

    return out.astype(NP_DTYPE)


def _check(name, arr):
    a = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    nans = np.isnan(a).sum()
    infs = np.isinf(a).sum()
    if nans or infs:
        print(f"  !! {name}: {nans} NaN, {infs} Inf, shape={arr.shape}")
    else:
        print(f"  OK {name}: max={np.abs(a).max():.4f}, shape={arr.shape}")


def _silu_with_fallback(gate_proj_x):
    """SiLU with CPU fallback for values outside bf16-safe range."""
    gate_f32 = gate_proj_x.astype(np.float32)
    safe_mask = (np.abs(gate_f32) <= 2.5)
    activated_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)

    # Run NPU SiLU on all values
    for i in range(SEQ // SILU_SEQ_TILE):
        silu_mod(
            gate_proj_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
            activated_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
        )

    if not safe_mask.all():
        # CPU fallback for unsafe values
        cpu_silu = (gate_f32 * (1.0 / (1.0 + np.exp(-np.clip(gate_f32, -20, 20))))).astype(NP_DTYPE)
        activated_x[~safe_mask] = cpu_silu[~safe_mask]
        # Patch remaining Inf/NaN
        npu_bad = np.isinf(activated_x.astype(np.float32)) | np.isnan(activated_x.astype(np.float32))
        if npu_bad.any():
            activated_x[npu_bad] = cpu_silu[npu_bad]

    return activated_x


def _ffn_block(x_norm, params):
    """Shared FFN block for both self-attention and cross-attention layers.
    gate_proj → SiLU, up_proj, hadamard, down_proj (K-chunked)."""
    # Gate + Up projections
    gate_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    up_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x_norm, params["W_gate"], gate_proj_x)
    gemm_ffn_up_mod(x_norm, params["W_up"], up_proj_x)
    _check("gate_proj", gate_proj_x)
    _check("up_proj", up_proj_x)

    # SiLU(gate) * up
    activated_x = _silu_with_fallback(gate_proj_x)
    _check("silu_gate", activated_x)
    activated_x *= up_proj_x  # hadamard in numpy bf16
    _check("activated", activated_x)

    # FFN down: chunk K=2048 as 8 x GEMM(32, 768, 256)
    x_out = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for chunk in range(FFN_DOWN_K_CHUNKS):
        chunk_A = np.ascontiguousarray(
            activated_x[:, chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK]
        )
        chunk_B = np.ascontiguousarray(
            params["W_down"][chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK, :]
        )
        partial = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
        gemm_ffn_down_mod(chunk_A, chunk_B, partial)
        x_out += partial

    _check("ffn_down", x_out)
    return x_out


# ##############################################################
# FORWARD PASSES
# ##############################################################

def action_expert_self_forward(x_np, params):
    """Self-attention action expert forward (even layers).

    Args:
        x_np: input [SEQ, EMBD] = [32, 768]
        params: dict with Wq, Wk, Wv, Wo, W_gate, W_up, W_down, W_norm_1, W_norm_2

    Returns:
        output [SEQ, EMBD] = [32, 768]
    """
    x = x_np.astype(NP_DTYPE)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    rmsnorm(residual, params["W_norm_1"], x)
    _check("rmsnorm1", x)

    # QKV projections
    query = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    key   = np.zeros((SEQ, KV_DIM), dtype=NP_DTYPE)
    value = np.zeros((SEQ, KV_DIM), dtype=NP_DTYPE)
    gemm_q_mod(x, params["Wq"], query)
    gemm_kv_self_mod(x, params["Wk"], key)
    gemm_kv_self_mod(x, params["Wv"], value)
    _check("query", query)
    _check("key", key)
    _check("value", value)

    # RoPE (float32 internally, returns bf16)
    query = rope_apply_packed(query, heads=Q_H, head_dim=HEAD_DIM)
    key   = rope_apply_packed(key, heads=KV_H, head_dim=HEAD_DIM)

    # Attention score: (Q * scale) @ K^T per head → [32, 32]
    query_scaled = (query.astype(np.float32) * float(ATTN_SCALE)).astype(NP_DTYPE)

    attn_weight = np.zeros((SEQ, Q_H * SEQ), dtype=NP_DTYPE)
    for h in range(Q_H):
        kv_idx = int(h * KV_H // Q_H)
        Q_head = np.ascontiguousarray(query_scaled[:, h * HEAD_DIM : (h + 1) * HEAD_DIM])
        K_head_T = np.ascontiguousarray(key[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM].T)  # [64, 32]
        score = np.zeros((SEQ, SEQ), dtype=NP_DTYPE)
        gemm_attn_self_score_mod(Q_head, K_head_T, score)

        # CPU causal masked softmax for [32, 32]
        weight = masked_softmax_cpu(score)
        attn_weight[:, h * SEQ : (h + 1) * SEQ] = weight

    _check("attn_weight", attn_weight)

    # Attention value: weights @ V per head → [32, 64]
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    for h in range(Q_H):
        kv_idx = int(h * KV_H // Q_H)
        head_weight = np.ascontiguousarray(attn_weight[:, h * SEQ : (h + 1) * SEQ])
        head_value  = np.ascontiguousarray(value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM])
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_attn_self_value_mod(head_weight, head_value, head_out)
        attn_value[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = head_out

    _check("attn_value", attn_value)

    # Output projection + residual
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_out_mod(attn_value, params["Wo"], x)
    _check("out_proj", x)
    residual += x
    _check("residual1", residual)

    # RMSNorm 2
    rmsnorm(residual, params["W_norm_2"], x)
    _check("rmsnorm2", x)

    # FFN block
    x = _ffn_block(x, params)
    residual += x
    _check("final", residual)

    return residual


def action_expert_cross_forward(x_np, text_k_np, text_v_np, params):
    """Cross-attention action expert forward (odd layers).

    Args:
        x_np:      input [SEQ, EMBD] = [32, 768]
        text_k_np: text encoder key [TEXT_SEQ, KV_DIM] = [128, 320]
        text_v_np: text encoder value [TEXT_SEQ, KV_DIM] = [128, 320]
        params: dict with Wq, Wk_cross, Wv_cross, Wo, W_gate, W_up, W_down, W_norm_1, W_norm_2

    Returns:
        output [SEQ, EMBD] = [32, 768]
    """
    x = x_np.astype(NP_DTYPE)
    text_k = text_k_np.astype(NP_DTYPE)
    text_v = text_v_np.astype(NP_DTYPE)

    residual = x.reshape(SEQ, EMBD)
    x_norm = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    rmsnorm(residual, params["W_norm_1"], x_norm)
    _check("rmsnorm1", x_norm)

    # Q from action input
    query = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    gemm_q_mod(x_norm, params["Wq"], query)
    _check("query", query)

    # K/V from text encoder through cross-attention projections (320→320)
    key   = np.zeros((TEXT_SEQ, KV_DIM), dtype=NP_DTYPE)
    value = np.zeros((TEXT_SEQ, KV_DIM), dtype=NP_DTYPE)
    gemm_kv_cross_mod(text_k, params["Wk_cross"], key)
    gemm_kv_cross_mod(text_v, params["Wv_cross"], value)
    _check("cross_key", key)
    _check("cross_value", value)

    # RoPE on Q only for cross-attention
    query = rope_apply_packed(query, heads=Q_H, head_dim=HEAD_DIM)

    # Attention score: (Q * scale) @ K^T per head → [32, 128]
    query_scaled = (query.astype(np.float32) * float(ATTN_SCALE)).astype(NP_DTYPE)

    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    for h in range(Q_H):
        kv_idx = int(h * KV_H // Q_H)
        Q_head = np.ascontiguousarray(query_scaled[:, h * HEAD_DIM : (h + 1) * HEAD_DIM])  # [32, 64]
        K_head_T = np.ascontiguousarray(
            key[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM].T  # [64, 128]
        )
        V_head = np.ascontiguousarray(value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM])  # [128, 64]

        # Score: [32, 64] × [64, 128] → [32, 128]
        score = np.zeros((SEQ, TEXT_SEQ), dtype=NP_DTYPE)
        gemm_attn_cross_score_mod(Q_head, K_head_T, score)

        # Unmasked softmax on NPU: [32, 128]
        weight = np.zeros((SEQ, TEXT_SEQ), dtype=NP_DTYPE)
        unmasked_softmax_fn(score, weight)

        # Value: [32, 128] × [128, 64] → [32, 64]
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_attn_cross_value_mod(weight, V_head, head_out)
        attn_value[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = head_out

    _check("attn_value", attn_value)

    # Output projection + residual
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_out_mod(attn_value, params["Wo"], x)
    _check("out_proj", x)
    residual += x
    _check("residual1", residual)

    # RMSNorm 2
    x_norm2 = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    rmsnorm(residual, params["W_norm_2"], x_norm2)
    _check("rmsnorm2", x_norm2)

    # FFN block
    x = _ffn_block(x_norm2, params)
    residual += x
    _check("final", residual)

    return residual


# ##############################################################
# TEST
# ##############################################################

if __name__ == "__main__":
  if _BUILD_SELF:
    print("=" * 60)
    print("Action Expert BF16 — Self-Attention Test")
    print("=" * 60)

    ref_self = ActionExpertSelfBlock().eval()
    p = {n: v.detach().numpy() for n, v in ref_self.named_parameters()}
    params_self = {
        "Wq":       p["q_proj.weight"].T.astype(NP_DTYPE),      # (768, 960)
        "Wk":       p["k_proj.weight"].T.astype(NP_DTYPE),      # (768, 320)
        "Wv":       p["v_proj.weight"].T.astype(NP_DTYPE),      # (768, 320)
        "Wo":       p["o_proj.weight"].T.astype(NP_DTYPE),      # (960, 768)
        "W_gate":   p["gate_proj.weight"].T.astype(NP_DTYPE),   # (768, 2048)
        "W_up":     p["up_proj.weight"].T.astype(NP_DTYPE),     # (768, 2048)
        "W_down":   p["down_proj.weight"].T.astype(NP_DTYPE),   # (2048, 768)
        "W_norm_1": p["ln_1.weight"].astype(NP_DTYPE),          # (768,)
        "W_norm_2": p["ln_2.weight"].astype(NP_DTYPE),          # (768,)
    }

    x_float = torch.randn(BATCH, SEQ, EMBD)

    # PyTorch bf16 reference
    with torch.no_grad():
        ref_self_bf16 = ref_self.to(torch.bfloat16)
        t0 = time.time()
        out_ref_self = ref_self_bf16(x_float.to(torch.bfloat16))
        t1 = time.time()
        print(f"PyTorch bf16 self-attn time: {t1 - t0:.6f} s")
        ref_out_self = out_ref_self[0, :, :].float().numpy()

    # Allo bf16
    x_input = x_float[0, :, :].numpy()
    a0 = time.time()
    allo_out_self = action_expert_self_forward(x_input, params_self)
    a1 = time.time()
    print(f"Allo bf16 self-attn time:   {a1 - a0:.6f} s")

    np.testing.assert_allclose(
        allo_out_self.astype(np.float32), ref_out_self, atol=1e-1, rtol=1e-1
    )
    max_err  = np.max(np.abs(allo_out_self.astype(np.float32) - ref_out_self))
    mean_err = np.mean(np.abs(allo_out_self.astype(np.float32) - ref_out_self))
    print(f"Self-attn PASSED — max_err={max_err:.4f}, mean_err={mean_err:.4f}")

  if _BUILD_CROSS:
    print()
    print("=" * 60)
    print("Action Expert BF16 — Cross-Attention Test")
    print("=" * 60)

    ref_cross = ActionExpertCrossBlock().eval()
    p2 = {n: v.detach().numpy() for n, v in ref_cross.named_parameters()}
    params_cross = {
        "Wq":        p2["q_proj.weight"].T.astype(NP_DTYPE),      # (768, 960)
        "Wk_cross":  p2["k_proj.weight"].T.astype(NP_DTYPE),      # (320, 320)
        "Wv_cross":  p2["v_proj.weight"].T.astype(NP_DTYPE),      # (320, 320)
        "Wo":        p2["o_proj.weight"].T.astype(NP_DTYPE),      # (960, 768)
        "W_gate":    p2["gate_proj.weight"].T.astype(NP_DTYPE),   # (768, 2048)
        "W_up":      p2["up_proj.weight"].T.astype(NP_DTYPE),     # (768, 2048)
        "W_down":    p2["down_proj.weight"].T.astype(NP_DTYPE),   # (2048, 768)
        "W_norm_1":  p2["ln_1.weight"].astype(NP_DTYPE),          # (768,)
        "W_norm_2":  p2["ln_2.weight"].astype(NP_DTYPE),          # (768,)
    }

    x_float2 = torch.randn(BATCH, SEQ, EMBD)
    text_k_float = torch.randn(BATCH, TEXT_SEQ, KV_DIM)
    text_v_float = torch.randn(BATCH, TEXT_SEQ, KV_DIM)

    # PyTorch bf16 reference
    with torch.no_grad():
        ref_cross_bf16 = ref_cross.to(torch.bfloat16)
        t0 = time.time()
        out_ref_cross = ref_cross_bf16(
            x_float2.to(torch.bfloat16),
            text_k_float.to(torch.bfloat16),
            text_v_float.to(torch.bfloat16),
        )
        t1 = time.time()
        print(f"PyTorch bf16 cross-attn time: {t1 - t0:.6f} s")
        ref_out_cross = out_ref_cross[0, :, :].float().numpy()

    # Allo bf16
    x_input2 = x_float2[0, :, :].numpy()
    text_k_input = text_k_float[0, :, :].numpy()
    text_v_input = text_v_float[0, :, :].numpy()
    a0 = time.time()
    allo_out_cross = action_expert_cross_forward(x_input2, text_k_input, text_v_input, params_cross)
    a1 = time.time()
    print(f"Allo bf16 cross-attn time:   {a1 - a0:.6f} s")

    np.testing.assert_allclose(
        allo_out_cross.astype(np.float32), ref_out_cross, atol=1e-1, rtol=1e-1
    )
    max_err  = np.max(np.abs(allo_out_cross.astype(np.float32) - ref_out_cross))
    mean_err = np.mean(np.abs(allo_out_cross.astype(np.float32) - ref_out_cross))
    print(f"Cross-attn PASSED — max_err={max_err:.4f}, mean_err={mean_err:.4f}")
