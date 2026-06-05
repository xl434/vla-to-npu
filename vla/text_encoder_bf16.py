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
KERNEL_BF16_PATH = "../cc/bf16_vla/"
KERNEL_BF16_OLD_PATH = "../cc/old_kernels/bf16_old/"
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
NORM_P0 = 8
NORM_SEQ_TILE = 32
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
SOFTMAX_TILE_ROWS = 8  # per-core kernel processes [8, SEQ]
SOFTMAX_NUM_TILES = SEQ // SOFTMAX_TILE_ROWS  # 16 tiles — all processed in parallel per head

softmax_ext = ExternalModule(
    top="softmax_128_bf16",
    impl_path=KERNEL_BF16_PATH + "softmax_128_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)

@df.region()
def softmax_kernel(
    scores: Ty[SEQ, SEQ],
    weights: Ty[SEQ, SEQ],
):
    @df.kernel(mapping=[SOFTMAX_NUM_TILES], args=[scores, weights])
    def core(
        local_scores: Ty[SEQ, SEQ] @ [S(0), R],
        local_weights: Ty[SEQ, SEQ] @ [S(0), R],
    ):
        softmax_ext(local_scores, local_weights)

# Pre-compute causal mask: upper-triangular True where col > row
_causal_mask = np.triu(np.ones((SEQ, SEQ), dtype=bool), k=1)  # [SEQ, SEQ]

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
SILU_SEQ_TILE = 16  # P0 * 16 = 16 rows per invocation
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
ROPE_TILE = 64  # legacy decomposed kernels process 64 rows at a time
ROPE_FUSED_TILE = 32  # rope_fused.cc is hard-coded to 32 rows per call
VecLy = [S(0)]
MatLy = [S(1), S(0)]
OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"
SIN_COS_IMPL = KERNEL_LIB_PATH + "sin_cos.cc"
ROPE_FUSED_IMPL = KERNEL_LIB_PATH + "rope_fused.cc"

radians_ext = ExternalModule(top="rope_make_radians_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
pack_ext = ExternalModule(top="pack32to64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyL_ext = ExternalModule(top="copy_left32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
copyR_ext = ExternalModule(top="copy_right32_from64_float32", impl_path=OPS_IMPL, input_idx=[0], output_idx=[1])
join_ext = ExternalModule(top="join32_to_64_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
mul32_ext = ExternalModule(top="mul32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
add32_ext = ExternalModule(top="add32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sub32_ext = ExternalModule(top="sub32_float32", impl_path=OPS_IMPL, input_idx=[0, 1], output_idx=[2])
sin_cos_ext = ExternalModule(top="sin_cos_float32", impl_path=SIN_COS_IMPL, input_idx=[0], output_idx=[1])

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
def sin_cos_region(i64: Ty_rope[ROPE_TILE // 2, HEAD_DIM], packed_o64: Ty_rope[ROPE_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[i64, packed_o64])
    def core(li: Ty_rope[ROPE_TILE // 2, HEAD_DIM] @ MatLy,
             lpo: Ty_rope[ROPE_TILE, HEAD_DIM] @ MatLy):
        sin_cos_ext(li, lpo)

ROPE_HALF_TILE = ROPE_TILE // 2  # decomposed split/join/elementwise kernels are hardcoded to 32 rows

@df.region()
def copy_left_region(i64: Ty_rope[ROPE_HALF_TILE, HEAD_DIM], o32: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[ROPE_HALF_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy):
        copyL_ext(li, lo)

@df.region()
def copy_right_region(i64: Ty_rope[ROPE_HALF_TILE, HEAD_DIM], o32: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[ROPE_HALF_TILE, HEAD_DIM] @ MatLy, lo: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy):
        copyR_ext(li, lo)

@df.region()
def join_region(l32: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], r32: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], o64: Ty_rope[ROPE_HALF_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[l32, r32, o64])
    def core(ll: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lr: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lo: Ty_rope[ROPE_HALF_TILE, HEAD_DIM] @ MatLy):
        join_ext(ll, lr, lo)

@df.region()
def mul32_region(A: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy):
        mul32_ext(la, lb, lc)

@df.region()
def add32_region(A: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy):
        add32_ext(la, lb, lc)

@df.region()
def sub32_region(A: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], B: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF], C: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[ROPE_HALF_TILE, HEAD_DIM_HALF] @ MatLy):
        sub32_ext(la, lb, lc)

# Fused RoPE: single kernel does split+mul+sub+add+join in one shot.
# Sin/cos are precomputed on host and packed into sin_cos[32][64].
rope_fused_ext = ExternalModule(
    top="rope_fused_float32", impl_path=ROPE_FUSED_IMPL, input_idx=[0, 1], output_idx=[2]
)

@df.region()
def rope_fused_region(x: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM],
                      sin_cos: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM],
                      out: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ MatLy,
             lsc: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ MatLy,
             lo: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ MatLy):
        rope_fused_ext(lx, lsc, lo)

# Parallel RoPE: process 5 heads per dispatch (mapping=[5,1], verified working).
# Q_H=15 = 3×5 → 3 calls/tile instead of 15; KV_H=5 → 1 call/tile instead of 5.
# Each of the 5 cores gets [ROPE_FUSED_TILE=32, HEAD_DIM=64] — one head's [32,64].
ROPE_CHUNK = 5  # heads per parallel dispatch
RopeLy = [S(0), S(1)]

@df.region()
def rope_fused_5h_region(x:       Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM],
                          sin_cos: Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM],
                          out:     Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[ROPE_CHUNK, 1], args=[x, sin_cos, out])
    def core(lx:  Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM] @ RopeLy,
             lsc: Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM] @ RopeLy,
             lo:  Ty_rope[ROPE_CHUNK * ROPE_FUSED_TILE, HEAD_DIM] @ RopeLy):
        rope_fused_ext(lx, lsc, lo)


# ##############################################################
# BUILD
# ##############################################################
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

# Build all modules
rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="text_encoder_bf16/rms_norm.prj")
gemm_q_mod = df.build(gemm_q_kernel, project="text_encoder_bf16/gemm_q.prj", target="aie", mapping_primitives=gemm_q_mp)
gemm_kv_mod = df.build(gemm_kv_kernel, project="text_encoder_bf16/gemm_kv.prj", target="aie", mapping_primitives=gemm_kv_mp)
gemm_out_mod = df.build(gemm_out_kernel, project="text_encoder_bf16/gemm_out.prj", target="aie", mapping_primitives=gemm_out_mp)
gemm_attn_score_mod = df.build(gemm_attn_score_kernel, project="text_encoder_bf16/gemm_attn_score.prj", target="aie", mapping_primitives=gemm_attn_score_mp)
gemm_attn_value_mod = df.build(gemm_attn_value_kernel, project="text_encoder_bf16/gemm_attn_value.prj", target="aie", mapping_primitives=gemm_attn_value_mp)
gemm_ffn_up_mod = df.build(gemm_ffn_up_kernel, project="text_encoder_bf16/gemm_ffn_up.prj", target="aie", mapping_primitives=gemm_ffn_up_mp)
gemm_ffn_down_mod = df.build(gemm_ffn_down_kernel, project="text_encoder_bf16/gemm_ffn_down.prj", target="aie", mapping_primitives=gemm_ffn_down_mp)
softmax_mod = df.build(softmax_kernel, target="aie", project="text_encoder_bf16/softmax.prj")
silu_mod = df.build(silu_kernel, target="aie", project="text_encoder_bf16/silu.prj")

# Legacy decomposed RoPE builds commented out, replaced by rope_fused_mod
# radians_mod = df.build(radians_region, target="aie", project="text_encoder_bf16/rope/radians.prj")
# pack_mod = df.build(pack_region, target="aie", project="text_encoder_bf16/rope/pack.prj")
# sin_cos_mod = df.build(sin_cos_region, target="aie", project="text_encoder_bf16/rope/sin_cos.prj")
# copyL_mod = df.build(copy_left_region, target="aie", project="text_encoder_bf16/rope/copyL.prj")
# copyR_mod = df.build(copy_right_region, target="aie", project="text_encoder_bf16/rope/copyR.prj")
# join_mod = df.build(join_region, target="aie", project="text_encoder_bf16/rope/join.prj")
# mul32_mod = df.build(mul32_region, target="aie", project="text_encoder_bf16/rope/mul32.prj")
# add32_mod = df.build(add32_region, target="aie", project="text_encoder_bf16/rope/add32.prj")
# sub32_mod = df.build(sub32_region, target="aie", project="text_encoder_bf16/rope/sub32.prj")
rope_fused_mod = df.build(rope_fused_region, target="aie", project="text_encoder_bf16/rope/fused.prj")
rope_fused_5h_mod = df.build(rope_fused_5h_region, target="aie", project="text_encoder_bf16/rope/fused_5h.prj")


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
    """bf16 causal softmax. Causal mask pre-applied; all 16 row-tiles run in parallel per head.
    15 calls/block (one per head) vs 240 before."""
    for h in range(Q_H):
        # Extract [SEQ, SEQ] score matrix for this head
        score_head = np.ascontiguousarray(attention_score[:, h, :].astype(np.float32))
        score_head[_causal_mask] = -np.inf  # apply causal mask: future positions → -inf
        score_head_bf16 = score_head.astype(NP_DTYPE)
        weight_head = np.zeros((SEQ, SEQ), dtype=NP_DTYPE)
        softmax_mod(score_head_bf16, weight_head)
        attention_weight[:, h * SEQ:(h + 1) * SEQ] = weight_head


def _precompute_sin_cos(tile_rows, head_dim, max_wavelength, pos_offset):
    HALF = head_dim // 2
    k = np.arange(HALF, dtype=np.float32)
    inv_ts = max_wavelength ** (-(2.0 / head_dim) * k)
    pos = (pos_offset + np.arange(tile_rows, dtype=np.float32))
    radians = pos[:, None] * inv_ts[None, :]
    sin_cos = np.zeros((tile_rows, head_dim), dtype=np.float32)
    sin_cos[:, :HALF] = np.sin(radians)
    sin_cos[:, HALF:] = np.cos(radians)
    return sin_cos


def rope_apply_packed(packed_bf16, heads, head_dim=64, max_wavelength=10_000.0, pos_offset=0):
    """RoPE via 5-head parallel dispatch (mapping=[5,1]).
    Q_H=15: 3 calls/tile (was 15); KV_H=5: 1 call/tile (was 5).
    Sin/cos tiled to [5*32, 64] so each of 5 cores gets its own [32,64] copy."""
    packed = packed_bf16.astype(np.float32)
    seq_len, total_dim = packed.shape
    D = head_dim
    tile_rows = ROPE_FUSED_TILE
    chunk = ROPE_CHUNK  # 5 heads per dispatch

    out = np.empty_like(packed, dtype=np.float32)

    for t0 in range(0, seq_len, tile_rows):
        rows = min(tile_rows, seq_len - t0)
        sin_cos = _precompute_sin_cos(tile_rows, D, max_wavelength, pos_offset + t0)
        sc_batch = np.tile(sin_cos, (chunk, 1))  # [5*32, 64]

        for h0 in range(0, heads, chunk):
            x_batch = np.zeros((chunk * tile_rows, D), dtype=np.float32)
            for i, h in enumerate(range(h0, h0 + chunk)):
                x_batch[i * tile_rows:i * tile_rows + rows, :] = packed[t0:t0 + rows, h * D:(h + 1) * D]
            out_batch = np.zeros_like(x_batch)
            rope_fused_5h_mod(x_batch, sc_batch, out_batch)
            for i, h in enumerate(range(h0, h0 + chunk)):
                out[t0:t0 + rows, h * D:(h + 1) * D] = out_batch[i * tile_rows:i * tile_rows + rows, :]

    return out.astype(NP_DTYPE)




def text_encoder_forward(x_np, params):
    """Single text encoder layer forward pass.

    Returns: (output [SEQ, EMBD], key [SEQ, KV_H*HEAD_DIM], value [SEQ, KV_H*HEAD_DIM])
    """
    x = x_np.astype(NP_DTYPE)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    rmsnorm(residual, params["W_norm_1"], x)

    # QKV projections
    query = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    key = np.zeros((SEQ, KV_H * HEAD_DIM), dtype=NP_DTYPE)
    value = np.zeros((SEQ, KV_H * HEAD_DIM), dtype=NP_DTYPE)
    gemm_q_mod(x, params["Wq"], query)
    gemm_kv_mod(x, params["Wk"], key)
    gemm_kv_mod(x, params["Wv"], value)

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


    # Masked softmax (bf16 — no float32 conversion needed)
    attn_weight = np.zeros((SEQ, Q_H * SEQ), dtype=NP_DTYPE)
    masked_softmax_fn(attention_score, attn_weight)

    # Attention value: weights @ V per head
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    for k_idx in range(Q_H):
        kv_idx = int(k_idx * KV_H // Q_H)
        head_weight = np.ascontiguousarray(attn_weight[:, k_idx * SEQ : (k_idx + 1) * SEQ])
        head_value = np.ascontiguousarray(value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM])
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_attn_value_mod(head_weight, head_value, head_out)
        attn_value[:, k_idx * HEAD_DIM : (k_idx + 1) * HEAD_DIM] = head_out


    # Output projection + residual
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_out_mod(attn_value, params["Wo"], x)
    residual += x  # residual is now the post-attention output

    # Snapshot post-attention residual; K/V for cross-attention are derived
    # from LN_1(post-attention) at the end of the layer (matches reference).
    post_attn = residual.copy()

    # RMSNorm 2
    rmsnorm(residual, params["W_norm_2"], x)

    # Gate projection + SiLU
    gate_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_gate"], gate_proj_x)

    # Up projection
    up_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_up"], up_proj_x)


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

    activated_x *= up_proj_x  # hadamard in numpy bf16

    # FFN down: chunk K=2560 as 8 x GEMM(128, 960, 320)
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for chunk in range(FFN_DOWN_K_CHUNKS):
        chunk_A = np.ascontiguousarray(activated_x[:, chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK])
        chunk_B = np.ascontiguousarray(params["W_down"][chunk * FFN_DOWN_K_CHUNK : (chunk + 1) * FFN_DOWN_K_CHUNK, :])
        partial = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
        gemm_ffn_down_mod(chunk_A, chunk_B, partial)
        x += partial

    residual += x

    # K/V exported for cross-attention: pre-RoPE, from LN_1(post-attn snapshot).
    # Computed on CPU to avoid disturbing AIE pipeline state.
    pa_f32 = post_attn.astype(np.float32)
    w_norm_f32 = params["W_norm_1"].astype(np.float32)
    rms = np.sqrt((pa_f32 ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    cross_norm = ((pa_f32 / rms) * w_norm_f32).astype(NP_DTYPE)
    key_out = (cross_norm.astype(np.float32) @ params["Wk"].astype(np.float32)).astype(NP_DTYPE)
    value_out = (cross_norm.astype(np.float32) @ params["Wv"].astype(np.float32)).astype(NP_DTYPE)

    return residual, key_out, value_out


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

    ref_k = k_ref.float().numpy()
    ref_v = v_ref.float().numpy()
    np.testing.assert_allclose(
        allo_key.astype(np.float32), ref_k, atol=1e-1, rtol=1e-1
    )
    np.testing.assert_allclose(
        allo_value.astype(np.float32), ref_v, atol=1e-1, rtol=1e-1
    )
    print("Returned K/V match reference within tolerance")
