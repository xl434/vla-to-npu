# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BFloat16 version of llama_block_rope.py
Uses bf16 kernels for RMSNorm, SiLU, and GEMM module for linear projections.
Masked softmax and RoPE remain float32.

  PyTorch bf16 forward time: 0.004492 s
  Allo bf16 forward time:    7.872250 s
  Allo bf16 llama block matches PyTorch bf16 reference within tolerance
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
KERNEL_BF16_PATH = "../cc/bf16_old/"
BATCH = 1
SEQ = 64
EMBD = 768
Q_H = 15
KV_H = 5
HEAD_DIM = 64
FFN_HID = EMBD * 4  # 3072

assert EMBD % 64 == 0
assert HEAD_DIM % 64 == 0
assert FFN_HID % EMBD == 0

Ty = Ty_bf16
NP_DTYPE = np_bfloat16
LINEAR_TILE = 64

# ===============================================================================
# Torch Reference
# ===============================================================================
class AttentionExpertBlock(nn.Module):
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
        return x


# ===============================================================================
# Allo BF16 Version
# ===============================================================================

# ----------------------------------------------------------------
# RMSNorm (bf16)
# ----------------------------------------------------------------
norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path=KERNEL_BF16_PATH + "rms_norm_bf16.cc",
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
# Q projection: SEQ x (Q_H*HEAD_DIM) x EMBD = 64 x 960 x 768
gemm_q_kernel, gemm_q_mp = GEMM(
    SEQ, Q_H * HEAD_DIM, EMBD,
    SEQ // LINEAR_TILE, (Q_H * HEAD_DIM) // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# K/V projection: SEQ x (KV_H*HEAD_DIM) x EMBD = 64 x 320 x 768
gemm_kv_kernel, gemm_kv_mp = GEMM(
    SEQ, KV_H * HEAD_DIM, EMBD,
    SEQ // LINEAR_TILE, (KV_H * HEAD_DIM) // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# Output projection: SEQ x EMBD x (Q_H*HEAD_DIM) = 64 x 768 x 960
gemm_out_kernel, gemm_out_mp = GEMM(
    SEQ, EMBD, Q_H * HEAD_DIM,
    SEQ // LINEAR_TILE, EMBD // LINEAR_TILE, (Q_H * HEAD_DIM) // LINEAR_TILE,
    Ty, Ty,
)

# Attn score & value: SEQ x SEQ x HEAD_DIM = 64 x 64 x 64
# Use tile=32 (tile=64 gives Pm=Pn=Pk=1 which GEMM module can't map)
ATTN_TILE = 32
gemm_attn_kernel, gemm_attn_mp = GEMM(
    SEQ, SEQ, HEAD_DIM,
    SEQ // ATTN_TILE, SEQ // ATTN_TILE, HEAD_DIM // ATTN_TILE,
    Ty, Ty,
)

# Gate/Up projection: SEQ x FFN_HID x EMBD = 64 x 3072 x 768
gemm_ffn_up_kernel, gemm_ffn_up_mp = GEMM(
    SEQ, FFN_HID, EMBD,
    SEQ // LINEAR_TILE, FFN_HID // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# FFN down: SEQ x EMBD x FFN_HID = 64 x 768 x 3072
# K=3072 with tile=64 overflows (Pk=48). Chunk as 4 x GEMM(64, 768, 768).
gemm_embd_kernel, gemm_embd_mp = GEMM(
    SEQ, EMBD, EMBD,
    SEQ // LINEAR_TILE, EMBD // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)
FFN_DOWN_K_CHUNKS = FFN_HID // EMBD  # 3072 / 768 = 4

# ----------------------------------------------------------------
# Masked Softmax (float32 — no bf16 version)
# ----------------------------------------------------------------
Ty_f32 = float32
Tint = int32

masked_softmax_ext = ExternalModule(
    top="masked_softmax_float32",
    impl_path=KERNEL_LIB_PATH + "masked_softmax.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
SOFTMAX_P0 = 2
SOFTMAX_P1 = 3
SOFTMAX_HEAD_TILE = SOFTMAX_P1
SOFTMAX_SEQ_TILE = SEQ // SOFTMAX_P0
SOFTMAX_Ly = [S(0), S(1)]
SOFTMAX_ROW_Ly = [S(0)]

@df.region()
def masked_softmax_kernel(
    input_x: Ty_f32[SEQ, SEQ * SOFTMAX_HEAD_TILE],
    row: Tint[SOFTMAX_P0],
    output_x: Ty_f32[SEQ, SEQ * SOFTMAX_HEAD_TILE],
):
    @df.kernel(mapping=[SOFTMAX_P0, SOFTMAX_P1], args=[input_x, row, output_x])
    def core(
        local_input_x: Ty_f32[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
        local_row: Tint[SOFTMAX_P0] @ SOFTMAX_ROW_Ly,
        local_output_x: Ty_f32[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
    ):
        masked_softmax_ext(local_input_x, local_row, local_output_x)

# ----------------------------------------------------------------
# SiLU (bf16)
# ----------------------------------------------------------------
silu_ext = ExternalModule(
    top="silu_bf16",
    impl_path=KERNEL_BF16_PATH + "silu_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)
SILU_P0 = 4
SILU_P1 = 4
SILU_SEQ_TILE = 16
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
def radians_region(positions: Ty_rope[SEQ], inv_ts: Ty_rope[HEAD_DIM_HALF], radians32: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[positions, inv_ts, radians32])
    def core(lp: Ty_rope[SEQ] @ VecLy, li: Ty_rope[HEAD_DIM_HALF] @ VecLy, lr: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        radians_ext(lp, li, lr)

@df.region()
def pack_region(r32: Ty_rope[SEQ, HEAD_DIM_HALF], r64: Ty_rope[SEQ, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[r32, r64])
    def core(lr32: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lr64: Ty_rope[SEQ, HEAD_DIM] @ MatLy):
        pack_ext(lr32, lr64)

@df.region()
def sin_region(i64: Ty_rope[SEQ, HEAD_DIM], o64: Ty_rope[SEQ, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[i64, o64])
    def core(li: Ty_rope[SEQ, HEAD_DIM] @ MatLy, lo: Ty_rope[SEQ, HEAD_DIM] @ MatLy):
        sin_ext(li, lo)

@df.region()
def cos_region(i64: Ty_rope[SEQ, HEAD_DIM], o64: Ty_rope[SEQ, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[i64, o64])
    def core(li: Ty_rope[SEQ, HEAD_DIM] @ MatLy, lo: Ty_rope[SEQ, HEAD_DIM] @ MatLy):
        cos_ext(li, lo)

@df.region()
def copy_left_region(i64: Ty_rope[SEQ, HEAD_DIM], o32: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[SEQ, HEAD_DIM] @ MatLy, lo: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        copyL_ext(li, lo)

@df.region()
def copy_right_region(i64: Ty_rope[SEQ, HEAD_DIM], o32: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[i64, o32])
    def core(li: Ty_rope[SEQ, HEAD_DIM] @ MatLy, lo: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        copyR_ext(li, lo)

@df.region()
def join_region(l32: Ty_rope[SEQ, HEAD_DIM_HALF], r32: Ty_rope[SEQ, HEAD_DIM_HALF], o64: Ty_rope[SEQ, HEAD_DIM]):
    @df.kernel(mapping=[1, 2], args=[l32, r32, o64])
    def core(ll: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lr: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lo: Ty_rope[SEQ, HEAD_DIM] @ MatLy):
        join_ext(ll, lr, lo)

@df.region()
def mul32_region(A: Ty_rope[SEQ, HEAD_DIM_HALF], B: Ty_rope[SEQ, HEAD_DIM_HALF], C: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        mul32_ext(la, lb, lc)

@df.region()
def add32_region(A: Ty_rope[SEQ, HEAD_DIM_HALF], B: Ty_rope[SEQ, HEAD_DIM_HALF], C: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        add32_ext(la, lb, lc)

@df.region()
def sub32_region(A: Ty_rope[SEQ, HEAD_DIM_HALF], B: Ty_rope[SEQ, HEAD_DIM_HALF], C: Ty_rope[SEQ, HEAD_DIM_HALF]):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(la: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lb: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy, lc: Ty_rope[SEQ, HEAD_DIM_HALF] @ MatLy):
        sub32_ext(la, lb, lc)


# ##############################################################
# BUILD
# ##############################################################
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="llama_bf16/rms_norm.prj")

gemm_q_mod = df.build(gemm_q_kernel, project="llama_bf16/gemm_q.prj", target="aie", mapping_primitives=gemm_q_mp)
gemm_kv_mod = df.build(gemm_kv_kernel, project="llama_bf16/gemm_kv.prj", target="aie", mapping_primitives=gemm_kv_mp)
gemm_out_mod = df.build(gemm_out_kernel, project="llama_bf16/gemm_out.prj", target="aie", mapping_primitives=gemm_out_mp)
gemm_attn_mod = df.build(gemm_attn_kernel, project="llama_bf16/gemm_attn.prj", target="aie", mapping_primitives=gemm_attn_mp)
gemm_ffn_up_mod = df.build(gemm_ffn_up_kernel, project="llama_bf16/gemm_ffn_up.prj", target="aie", mapping_primitives=gemm_ffn_up_mp)
gemm_embd_mod = df.build(gemm_embd_kernel, project="llama_bf16/gemm_embd.prj", target="aie", mapping_primitives=gemm_embd_mp)

masked_softmax_mod = df.build(masked_softmax_kernel, target="aie", project="llama_bf16/masked_softmax.prj")
silu_mod = df.build(silu_kernel, target="aie", project="llama_bf16/silu.prj")

radians_mod = df.build(radians_region, target="aie", project="llama_bf16/rope/radians.prj")
pack_mod = df.build(pack_region, target="aie", project="llama_bf16/rope/pack.prj")
sin_mod = df.build(sin_region, target="aie", project="llama_bf16/rope/sin.prj")
cos_mod = df.build(cos_region, target="aie", project="llama_bf16/rope/cos.prj")
copyL_mod = df.build(copy_left_region, target="aie", project="llama_bf16/rope/copyL.prj")
copyR_mod = df.build(copy_right_region, target="aie", project="llama_bf16/rope/copyR.prj")
join_mod = df.build(join_region, target="aie", project="llama_bf16/rope/join.prj")
mul32_mod = df.build(mul32_region, target="aie", project="llama_bf16/rope/mul32.prj")
add32_mod = df.build(add32_region, target="aie", project="llama_bf16/rope/add32.prj")
sub32_mod = df.build(sub32_region, target="aie", project="llama_bf16/rope/sub32.prj")


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
    """Runs in float32. Converts bf16 scores to float32, runs masked softmax, returns bf16."""
    row_idx = np.array(list(range(0, SEQ, SOFTMAX_SEQ_TILE))).astype(np.int32)
    for i in range(Q_H // SOFTMAX_HEAD_TILE):
        masked_softmax_mod(
            attention_score[:, i * SOFTMAX_HEAD_TILE : (i + 1) * SOFTMAX_HEAD_TILE, :],
            row_idx,
            attention_weight[:, i * (SOFTMAX_HEAD_TILE * SEQ) : (i + 1) * (SOFTMAX_HEAD_TILE * SEQ)],
        )


def rope_apply_packed(packed_bf16, heads, head_dim=64, max_wavelength=10_000.0, pos_offset=0):
    """RoPE in float32. Converts bf16 input to float32, applies RoPE, returns bf16."""
    packed = packed_bf16.astype(np.float32)
    seq_len, total_dim = packed.shape
    tile_rows = SEQ
    D = head_dim
    HALF = D // 2

    out = np.empty_like(packed, dtype=np.float32)
    k = np.arange(HALF, dtype=np.float32)
    inv_ts = (max_wavelength ** (-(2.0 / D) * k)).astype(np.float32)

    for t0 in range(0, seq_len, tile_rows):
        rows = min(tile_rows, seq_len - t0)
        pos32 = (pos_offset + np.arange(t0, t0 + rows, dtype=np.float32)).astype(np.float32)
        pos_pad = np.zeros(tile_rows, dtype=np.float32)
        pos_pad[:rows] = pos32

        radians32 = np.zeros((tile_rows, HALF), dtype=np.float32)
        radians_mod(pos_pad, inv_ts, radians32)
        radians64 = np.zeros((tile_rows, D), dtype=np.float32)
        pack_mod(radians32, radians64)

        sin64 = np.zeros((tile_rows, D), dtype=np.float32)
        cos64 = np.zeros((tile_rows, D), dtype=np.float32)
        sin_mod(radians64, sin64)
        cos_mod(radians64, cos64)

        for h in range(heads):
            x_tile = np.zeros((tile_rows, D), dtype=np.float32)
            x_tile[:rows, :] = packed[t0:t0 + rows, h * D:(h + 1) * D]

            xL = np.zeros((tile_rows, HALF), dtype=np.float32)
            xR = np.zeros((tile_rows, HALF), dtype=np.float32)
            s = np.zeros((tile_rows, HALF), dtype=np.float32)
            c = np.zeros((tile_rows, HALF), dtype=np.float32)

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

            y64 = np.zeros((tile_rows, D), dtype=np.float32)
            join_mod(yL, yR, y64)
            out[t0:t0 + rows, h * D:(h + 1) * D] = y64[:rows, :]

    return out.astype(NP_DTYPE)


def llama_block_rope(x_np, params):
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
    for k in range(Q_H):
        k_key_idx = int(k * KV_H // Q_H)
        Q_head = np.ascontiguousarray(query_scaled[:, k * HEAD_DIM : (k + 1) * HEAD_DIM])
        K_head_T = np.ascontiguousarray(key[:, k_key_idx * HEAD_DIM : (k_key_idx + 1) * HEAD_DIM].T)
        score = np.zeros((SEQ, SEQ), dtype=NP_DTYPE)
        gemm_attn_mod(Q_head, K_head_T, score)
        attention_score[:, k, :] = score

    # Masked softmax (float32)
    if USE_ALL_NPU_KERNELS:
        attn_score_f32 = attention_score.astype(np.float32)
        attn_weight_f32 = np.zeros((SEQ, Q_H * SEQ), dtype=np.float32)
        masked_softmax_fn(attn_score_f32, attn_weight_f32)
        attn_weight = attn_weight_f32.astype(NP_DTYPE)
    else:
        mask = torch.triu(torch.ones(SEQ, SEQ), 1).bool()
        mask = np.repeat(mask[:, np.newaxis, :], Q_H, axis=1)
        attn_score_f32 = attention_score.astype(np.float32)
        attn_score_f32[mask == 1] = -np.inf
        attn_weight = F.softmax(torch.from_numpy(attn_score_f32), dim=-1).numpy().astype(NP_DTYPE)

    # Attention value
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM), dtype=NP_DTYPE)
    for k in range(Q_H):
        kv_idx = int(k * KV_H // Q_H)
        head_weight = (
            attn_weight[:, k * SEQ : (k + 1) * SEQ]
            if USE_ALL_NPU_KERNELS
            else attn_weight[:, k, :]
        )
        head_value = np.ascontiguousarray(value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM])
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_attn_mod(head_weight, head_value, head_out)
        attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM] = head_out

    # Output projection
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_out_mod(attn_value, params["Wo"], x)
    residual += x

    # RMSNorm 2
    rmsnorm(residual, params["W_norm_2"], x)

    # Gate projection + SiLU
    gate_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_gate"], gate_proj_x)

    # Up projection
    up_proj_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_ffn_up_mod(x, params["W_up"], up_proj_x)

    # SiLU(gate) * up
    if USE_ALL_NPU_KERNELS:
        activated_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
        for i in range(SEQ // SILU_SEQ_TILE):
            silu_mod(
                gate_proj_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
                activated_x[i * SILU_SEQ_TILE : (i + 1) * SILU_SEQ_TILE, :],
            )
        activated_x *= up_proj_x  # hadamard in numpy bf16
    else:
        gate_t = torch.from_numpy(gate_proj_x.astype(np.float32))
        up_t = torch.from_numpy(up_proj_x.astype(np.float32))
        activated_x = (nn.SiLU()(gate_t) * up_t).numpy().astype(NP_DTYPE)

    # FFN down: chunk K=3072 as 4 x GEMM(64, 768, 768)
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for chunk in range(FFN_DOWN_K_CHUNKS):
        chunk_A = np.ascontiguousarray(activated_x[:, chunk * EMBD : (chunk + 1) * EMBD])
        chunk_B = np.ascontiguousarray(params["W_down"][chunk * EMBD : (chunk + 1) * EMBD, :])
        partial = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
        gemm_embd_mod(chunk_A, chunk_B, partial)
        x += partial

    residual += x
    return residual


if __name__ == "__main__":
    ref_model = AttentionExpertBlock().eval()
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
        sample = ref_model_bf16(x_float.to(torch.bfloat16))[0, :, :]
        t1 = time.time()
        print(f"PyTorch bf16 forward time: {t1 - t0:.6f} s")
        ref_out = sample.float().numpy()

    x_input = x_float[0, :, :].numpy()
    a0 = time.time()
    allo_out = llama_block_rope(x_input, params)
    a1 = time.time()
    print(f"Allo bf16 forward time:    {a1 - a0:.6f} s")

    np.testing.assert_allclose(
        allo_out.astype(np.float32), ref_out, atol=1e-1, rtol=1e-1
    )
    print("Allo bf16 llama block matches PyTorch bf16 reference within tolerance")
