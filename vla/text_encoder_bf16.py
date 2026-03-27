# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLA Text Encoder Layer (BFloat16)

Single text-encoder transformer block for NPU.
Based on llama_block_rope_bf16.py with dimensions:
  SEQ=128, EMBD=960, Q_H=15, KV_H=5, HEAD_DIM=64, FFN_HID=2560

Returns (output, key, value) — key/value are needed by the action expert
for cross-attention.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import MultiHeadAttention
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16 as Ty_bf16, int32, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from allo.library.aie.modules.gemm import GEMM
from ml_dtypes import bfloat16 as np_bfloat16
import time

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

# ===============================================================================
# Model Configuration
# ===============================================================================
USE_ALL_NPU_KERNELS = True
KERNEL_LIB_PATH = "../cc/float/"
KERNEL_BF16_PATH = "../cc/bf16/"
KERNEL_BF16_OLD_PATH = "../cc/bf16_old/"
BATCH = 1
SEQ = 128
EMBD = 960
Q_H = 15
KV_H = 5
HEAD_DIM = 64
FFN_HID = 2560

assert EMBD % 64 == 0
assert HEAD_DIM % 64 == 0

Ty = Ty_bf16
NP_DTYPE = np_bfloat16
LINEAR_TILE = 64

# ===============================================================================
# Torch Reference
# ===============================================================================
class TextEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        q_proj = nn.Linear(EMBD, Q_H * HEAD_DIM, bias=False)
        k_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)
        v_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)
        o_proj = nn.Linear(Q_H * HEAD_DIM, EMBD, bias=False)
        self.attn = MultiHeadAttention(
            embed_dim=Q_H * HEAD_DIM, num_heads=Q_H, num_kv_heads=KV_H,
            head_dim=HEAD_DIM, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
            output_proj=o_proj, is_causal=True,
        )
        self.gate_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ln_1 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.up_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.down_proj = nn.Linear(FFN_HID, EMBD, bias=False)
        self.silu = nn.SiLU()
        self.ln_2 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.max_wavelength = 10_000.0
        self.head_dim = HEAD_DIM
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
        q = self.attn.q_proj(x).view(B, L, Q_H, D)
        k = self.attn.k_proj(x).view(B, L, KV_H, D)
        v = self.attn.v_proj(x).view(B, L, KV_H, D)
        pos = self._positions(L, x.device).expand(B, -1)
        q = self.apply_rope(q, pos)
        k = self.apply_rope(k, pos)
        kv_map = torch.div(torch.arange(Q_H, device=x.device) * KV_H, Q_H, rounding_mode='floor')
        k_sel = k.index_select(dim=2, index=kv_map)
        v_sel = v.index_select(dim=2, index=kv_map)
        q_h = q.transpose(1, 2)
        k_h = k_sel.transpose(1, 2)
        scores = torch.matmul(q_h.float(), k_h.float().transpose(-2, -1)) / (D ** 0.5)
        scores.masked_fill_(torch.ones(L, L, device=x.device).triu(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        v_h = v_sel.transpose(1, 2)
        ctx = torch.matmul(attn.float(), v_h.float()).to(torch.bfloat16).transpose(1, 2).contiguous().view(B, L, Q_H * D)
        x = self.attn.output_proj(ctx) + residual
        residual = x
        x = self.ln_2(x)
        act = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(act) + residual
        # Return key and value for cross-attention (pre-RoPE key for cross-attn k_proj)
        k_out = self.attn.k_proj(self.ln_1(residual)).view(B, L, KV_H, D)
        v_out = self.attn.v_proj(self.ln_1(residual)).view(B, L, KV_H, D)
        return x, k_out.squeeze(0).reshape(L, KV_H * D), v_out.squeeze(0).reshape(L, KV_H * D)


# ===============================================================================
# Allo BF16 Version
# ===============================================================================

# ----------------------------------------------------------------
# RMSNorm (bf16, width=960)
# ----------------------------------------------------------------
norm = ExternalModule(
    top="rms_norm_960_bf16",
    impl_path=KERNEL_BF16_PATH + "rms_norm_960_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0
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
# Q projection: SEQ x (Q_H*HEAD_DIM) x EMBD = 128 x 960 x 960
gemm_q_kernel, gemm_q_mp = GEMM(
    SEQ, Q_H * HEAD_DIM, EMBD,
    SEQ // LINEAR_TILE, (Q_H * HEAD_DIM) // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# K/V projection: SEQ x (KV_H*HEAD_DIM) x EMBD = 128 x 320 x 960
gemm_kv_kernel, gemm_kv_mp = GEMM(
    SEQ, KV_H * HEAD_DIM, EMBD,
    SEQ // LINEAR_TILE, (KV_H * HEAD_DIM) // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# Output projection: SEQ x EMBD x (Q_H*HEAD_DIM) = 128 x 960 x 960
gemm_out_kernel, gemm_out_mp = GEMM(
    SEQ, EMBD, Q_H * HEAD_DIM,
    SEQ // LINEAR_TILE, EMBD // LINEAR_TILE, (Q_H * HEAD_DIM) // LINEAR_TILE,
    Ty, Ty,
)

# Attention score: SEQ x SEQ x HEAD_DIM = 128 x 128 x 64
ATTN_TILE = 32
gemm_attn_score_kernel, gemm_attn_score_mp = GEMM(
    SEQ, SEQ, HEAD_DIM,
    SEQ // ATTN_TILE, SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE,
    Ty, Ty,
)

# Attention value: SEQ x HEAD_DIM x SEQ = 128 x 64 x 128
gemm_attn_value_kernel, gemm_attn_value_mp = GEMM(
    SEQ, HEAD_DIM, SEQ,
    SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE, SEQ // ATTN_TILE,
    Ty, Ty,
)

# Gate/Up projection: SEQ x FFN_HID x EMBD = 128 x 2560 x 960
gemm_ffn_up_kernel, gemm_ffn_up_mp = GEMM(
    SEQ, FFN_HID, EMBD,
    SEQ // LINEAR_TILE, FFN_HID // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# FFN down: SEQ x EMBD x FFN_HID = 128 x 960 x 2560
# K=2560 too large. Chunk as 8 x GEMM(128, 960, 320).
FFN_DOWN_K_CHUNK = 320
gemm_ffn_down_kernel, gemm_ffn_down_mp = GEMM(
    SEQ, EMBD, FFN_DOWN_K_CHUNK,
    SEQ // LINEAR_TILE, EMBD // LINEAR_TILE, FFN_DOWN_K_CHUNK // LINEAR_TILE,
    Ty, Ty,
)
FFN_DOWN_K_CHUNKS = FFN_HID // FFN_DOWN_K_CHUNK  # 2560 / 320 = 8

# ----------------------------------------------------------------
# Masked Softmax (bf16, 128-col causal)
# ----------------------------------------------------------------
Tint = int32
SOFTMAX_TILE_ROWS = 8  # kernel processes 8 rows at a time

masked_softmax_ext = ExternalModule(
    top="masked_softmax_128_bf16",
    impl_path=KERNEL_BF16_PATH + "masked_softmax_128_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

@df.region()
def masked_softmax_kernel(
    scores: Ty[SOFTMAX_TILE_ROWS, SEQ],
    row_start: Tint[1],
    weights: Ty[SOFTMAX_TILE_ROWS, SEQ],
):
    @df.kernel(mapping=[1, 1], args=[scores, row_start, weights])
    def core(
        local_scores: Ty[SOFTMAX_TILE_ROWS, SEQ] @ [S(0), S(1)],
        local_row: Tint[1] @ [R],
        local_weights: Ty[SOFTMAX_TILE_ROWS, SEQ] @ [S(0), S(1)],
    ):
        masked_softmax_ext(local_scores, local_row, local_weights)

# ----------------------------------------------------------------
# SiLU (bf16, FFN_HID=2560, per-core tile [4][160])
# ----------------------------------------------------------------
silu_ext = ExternalModule(
    top="silu_160_bf16",
    impl_path=KERNEL_BF16_PATH + "silu_160_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)
SILU_P0 = 1
SILU_P1 = 16  # 2560 / 160 = 16 cores on feature dim
SILU_SEQ_TILE = 4  # P0 * 4 = 4 rows per invocation
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
ROPE_TILE = 64  # RoPE kernels process 64 rows at a time
VecLy = [S(0)]
MatLy = [S(1), S(0)]
OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"
SIN_IMPL = KERNEL_LIB_PATH + "sine.cc"
COS_IMPL = KERNEL_LIB_PATH + "cosine.cc"

radians_ext = ExternalModule(top="rope_make_radians_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
pack_ext = ExternalModule(top="pack32to64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyL_ext = ExternalModule(top="copy_left32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyR_ext = ExternalModule(top="copy_right32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
join_ext = ExternalModule(top="join32_to_64_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
mul32_ext = ExternalModule(top="mul32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
add32_ext = ExternalModule(top="add32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sub32_ext = ExternalModule(top="sub32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sin_ext = ExternalModule(top="sin_float32", impl_path=SIN_IMPL, input_idx=[0], output_idx=[1])
cos_ext = ExternalModule(top="cos_float32", impl_path=COS_IMPL, input_idx=[0], output_idx=[1])

Ty_rope = float32  # RoPE stays float32

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
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="text_encoder_bf16/rms_norm.prj")

gemm_q_mod = df.build(gemm_q_kernel, project="text_encoder_bf16/gemm_q.prj", target="aie", mapping_primitives=gemm_q_mp)
gemm_kv_mod = df.build(gemm_kv_kernel, project="text_encoder_bf16/gemm_kv.prj", target="aie", mapping_primitives=gemm_kv_mp)
gemm_out_mod = df.build(gemm_out_kernel, project="text_encoder_bf16/gemm_out.prj", target="aie", mapping_primitives=gemm_out_mp)
gemm_attn_score_mod = df.build(gemm_attn_score_kernel, project="text_encoder_bf16/gemm_attn_score.prj", target="aie", mapping_primitives=gemm_attn_score_mp)
gemm_attn_value_mod = df.build(gemm_attn_value_kernel, project="text_encoder_bf16/gemm_attn_value.prj", target="aie", mapping_primitives=gemm_attn_value_mp)
gemm_ffn_up_mod = df.build(gemm_ffn_up_kernel, project="text_encoder_bf16/gemm_ffn_up.prj", target="aie", mapping_primitives=gemm_ffn_up_mp)
gemm_ffn_down_mod = df.build(gemm_ffn_down_kernel, project="text_encoder_bf16/gemm_ffn_down.prj", target="aie", mapping_primitives=gemm_ffn_down_mp)

masked_softmax_mod = df.build(masked_softmax_kernel, target="aie", project="text_encoder_bf16/masked_softmax.prj")
silu_mod = df.build(silu_kernel, target="aie", project="text_encoder_bf16/silu.prj")

radians_mod = df.build(radians_region, target="aie", project="text_encoder_bf16/rope/radians.prj")
pack_mod = df.build(pack_region, target="aie", project="text_encoder_bf16/rope/pack.prj")
sin_mod = df.build(sin_region, target="aie", project="text_encoder_bf16/rope/sin.prj")
cos_mod = df.build(cos_region, target="aie", project="text_encoder_bf16/rope/cos.prj")
copyL_mod = df.build(copy_left_region, target="aie", project="text_encoder_bf16/rope/copyL.prj")
copyR_mod = df.build(copy_right_region, target="aie", project="text_encoder_bf16/rope/copyR.prj")
join_mod = df.build(join_region, target="aie", project="text_encoder_bf16/rope/join.prj")
mul32_mod = df.build(mul32_region, target="aie", project="text_encoder_bf16/rope/mul32.prj")
add32_mod = df.build(add32_region, target="aie", project="text_encoder_bf16/rope/add32.prj")
sub32_mod = df.build(sub32_region, target="aie", project="text_encoder_bf16/rope/sub32.prj")


# ##############################################################
# TOOL FUNCTIONS
# ##############################################################
def rmsnorm(input_x, weight, output_x):
    for i in range(SEQ // NORM_SEQ_TILE):
        rms_norm_mod(
            input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
            weight,
            output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
        )


ATTN_SCALE = NP_DTYPE(1.0 / (HEAD_DIM ** 0.5))


def masked_softmax_fn(attention_score, attention_weight):
    """bf16 masked softmax. Processes [8, 128] tiles per head."""
    for h in range(Q_H):
        for tile_idx in range(SEQ // SOFTMAX_TILE_ROWS):
            row_start = tile_idx * SOFTMAX_TILE_ROWS
            row_start_np = np.array([row_start], dtype=np.int32)
            score_tile = np.ascontiguousarray(attention_score[row_start:row_start + SOFTMAX_TILE_ROWS, h, :])
            weight_tile = np.zeros((SOFTMAX_TILE_ROWS, SEQ), dtype=NP_DTYPE)
            masked_softmax_mod(score_tile, row_start_np, weight_tile)
            attention_weight[row_start:row_start + SOFTMAX_TILE_ROWS, h * SEQ:(h + 1) * SEQ] = weight_tile


def rope_apply_packed(packed_bf16, heads, head_dim=64, max_wavelength=10_000.0, pos_offset=0):
    """RoPE in float32. Converts bf16 input to float32, applies RoPE, returns bf16."""
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
            s = np.zeros((ROPE_TILE, HALF), dtype=np.float32)
            c = np.zeros((ROPE_TILE, HALF), dtype=np.float32)

            copyL_mod(x_tile, xL)
            copyR_mod(x_tile, xR)
            copyL_mod(sin64, s)
            copyL_mod(cos64, c)

            tmp1 = np.zeros_like(xL); mul32_mod(xL, c, tmp1)
            tmp2 = np.zeros_like(xL); mul32_mod(xR, s, tmp2)
            yL = np.zeros_like(xL); sub32_mod(tmp1, tmp2, yL)

            tmp3 = np.zeros_like(xL); mul32_mod(xR, c, tmp3)
            tmp4 = np.zeros_like(xL); mul32_mod(xL, s, tmp4)
            yR = np.zeros_like(xL); add32_mod(tmp3, tmp4, yR)

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


def text_encoder_forward(x_np, params):
    """Single text encoder layer forward pass.

    Returns: (output [SEQ, EMBD], key [SEQ, KV_H*HEAD_DIM], value [SEQ, KV_H*HEAD_DIM])
    """
    x = x_np.astype(NP_DTYPE)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    rmsnorm(residual, params["W_norm_1"], x)
    _check("rmsnorm1", x)

    # QKV projections
    query = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    key = np.zeros((SEQ, KV_H * HEAD_DIM), dtype=NP_DTYPE)
    value = np.zeros((SEQ, KV_H * HEAD_DIM), dtype=NP_DTYPE)
    gemm_q_mod(x, params["Wq"], query)
    gemm_kv_mod(x, params["Wk"], key)
    gemm_kv_mod(x, params["Wv"], value)
    _check("query", query)
    _check("key", key)
    _check("value", value)

    # RoPE (float32 internally, returns bf16)
    query = rope_apply_packed(query, heads=Q_H, head_dim=HEAD_DIM)
    key = rope_apply_packed(key, heads=KV_H, head_dim=HEAD_DIM)

    # Attention score: (Q * scale) @ K^T per head
    query_scaled = (query.astype(np.float32) * float(ATTN_SCALE)).astype(NP_DTYPE)
    attention_score = np.empty((SEQ, Q_H, SEQ), dtype=NP_DTYPE)
    for k_idx in range(Q_H):
        kv_idx = int(k_idx * KV_H // Q_H)
        Q_head = np.ascontiguousarray(query_scaled[:, k_idx * HEAD_DIM : (k_idx + 1) * HEAD_DIM])
        K_head_T = np.ascontiguousarray(key[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM].T)
        score = np.zeros((SEQ, SEQ), dtype=NP_DTYPE)
        gemm_attn_score_mod(Q_head, K_head_T, score)
        attention_score[:, k_idx, :] = score

    _check("attn_score", attention_score)

    # Masked softmax (bf16 — no float32 conversion needed)
    attn_weight = np.zeros((SEQ, Q_H * SEQ), dtype=NP_DTYPE)
    masked_softmax_fn(attention_score, attn_weight)
    _check("attn_weight", attn_weight)

    # Attention value: weights @ V per head
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    for k_idx in range(Q_H):
        kv_idx = int(k_idx * KV_H // Q_H)
        head_weight = np.ascontiguousarray(attn_weight[:, k_idx * SEQ : (k_idx + 1) * SEQ])
        head_value = np.ascontiguousarray(value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM])
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_attn_value_mod(head_weight, head_value, head_out)
        attn_value[:, k_idx * HEAD_DIM : (k_idx + 1) * HEAD_DIM] = head_out

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

    # Gate projection + SiLU
    gate_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_gate"], gate_proj_x)

    # Up projection
    up_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_up"], up_proj_x)

    _check("gate_proj", gate_proj_x)
    _check("up_proj", up_proj_x)

    # SiLU(gate) * up
    # bf16 Taylor series overflows for |x| > ~2.5 (x^12 exceeds bf16 max).
    # The kernel handles x > 2.5 with linear approx and x < -7 with zero,
    # but -7 < x < -2.5 still uses the Taylor series and can overflow.
    # Fix: compute SiLU on CPU for values where |x| > 2.5, NPU for the rest.
    gate_f32 = gate_proj_x.astype(np.float32)
    safe_mask = (np.abs(gate_f32) <= 2.5)
    activated_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)

    if safe_mask.all():
        # All values safe for NPU
        for i in range(SEQ // SILU_SEQ_TILE):
            silu_mod(
                gate_proj_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
                activated_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
            )
    else:
        # Mixed: run NPU on all, then patch unsafe values with CPU result
        for i in range(SEQ // SILU_SEQ_TILE):
            silu_mod(
                gate_proj_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
                activated_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
            )
        # CPU fallback for unsafe values
        cpu_silu = (gate_f32 * (1.0 / (1.0 + np.exp(-np.clip(gate_f32, -20, 20))))).astype(NP_DTYPE)
        activated_x[~safe_mask] = cpu_silu[~safe_mask]
        # Also patch any remaining Inf/NaN from the NPU kernel
        npu_bad = np.isinf(activated_x.astype(np.float32)) | np.isnan(activated_x.astype(np.float32))
        if npu_bad.any():
            print(f"  SiLU: {npu_bad.sum()} bad values after mask patch, patching with CPU")
            # Check what input values caused Inf
            bad_inputs = gate_f32[npu_bad]
            print(f"  Bad input range: [{bad_inputs.min():.4f}, {bad_inputs.max():.4f}]")
            activated_x[npu_bad] = cpu_silu[npu_bad]

    _check("silu_gate", activated_x)
    activated_x *= up_proj_x  # hadamard in numpy bf16
    _check("activated", activated_x)

    # FFN down: chunk K=2560 as 8 x GEMM(128, 960, 320)
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for chunk in range(FFN_DOWN_K_CHUNKS):
        chunk_A = np.ascontiguousarray(activated_x[:, chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK])
        chunk_B = np.ascontiguousarray(params["W_down"][chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK, :])
        partial = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
        gemm_ffn_down_mod(chunk_A, chunk_B, partial)
        x += partial

    _check("ffn_down", x)
    residual += x
    _check("final", residual)

    # Return output + key/value for cross-attention
    return residual, key, value


if __name__ == "__main__":
    ref_model = TextEncoderBlock().eval()
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params = {
        "Wq": p["attn.q_proj.weight"].T.astype(NP_DTYPE),
        "Wk": p["attn.k_proj.weight"].T.astype(NP_DTYPE),
        "Wv": p["attn.v_proj.weight"].T.astype(NP_DTYPE),
        "Wo": p["attn.output_proj.weight"].T.astype(NP_DTYPE),
        "W_gate": p["gate_proj.weight"].T.astype(NP_DTYPE),
        "W_up": p["up_proj.weight"].T.astype(NP_DTYPE),
        "W_down": p["down_proj.weight"].T.astype(NP_DTYPE),
        "W_norm_1": p["ln_1.weight"].astype(NP_DTYPE),
        "W_norm_2": p["ln_2.weight"].astype(NP_DTYPE),
    }

    x_float = torch.randn(BATCH, SEQ, EMBD)

    # PyTorch bf16 reference
    with torch.no_grad():
        ref_model_bf16 = ref_model.to(torch.bfloat16)
        t0 = time.time()
        out_ref, k_ref, v_ref = ref_model_bf16(x_float.to(torch.bfloat16))
        t1 = time.time()
        print(f"PyTorch bf16 forward time: {t1 - t0:.6f} s")
        ref_out = out_ref[0, :, :].float().numpy()

    x_input = x_float[0, :, :].numpy()
    a0 = time.time()
    allo_out, allo_key, allo_value = text_encoder_forward(x_input, params)
    a1 = time.time()
    print(f"Allo bf16 forward time:    {a1 - a0:.6f} s")

    np.testing.assert_allclose(
        allo_out.astype(np.float32), ref_out, atol=1e-1, rtol=1e-1
    )
    print("Text encoder bf16 matches PyTorch bf16 reference within tolerance")
