# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BFloat16 version of vision_block.py
Uses bf16 kernels for LayerNorm, GELU, Attention Score, and GEMM module.
Softmax remains float32 due to bf16 program memory overflow on AIE.

  PyTorch bf16 forward time: 0.016832 s
  Allo bf16 forward time:    56.844355 s
  Allo bf16 block matches PyTorch bf16 reference within tolerance
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16 as Ty_bf16, Stream
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
KERNEL_LIB_PATH = "../cc/"
KERNEL_BF16_PATH = "../cc/bf16_old/"
BATCH = 1
SEQ = 1024
EMBD = 768
N_HEAD = 12
HEAD_DIM = EMBD // N_HEAD
FFN_HID = EMBD * 4  # 3072

assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"

Ty = Ty_bf16  # All tensors use bfloat16
NP_DTYPE = np_bfloat16

# ===============================================================================
# Torch Reference (bf16)
# ===============================================================================
class MiniVit(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBD, N_HEAD, batch_first=True)
        self.ln_1 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.ffn_up = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ffn_down = nn.Linear(FFN_HID, EMBD, bias=False)
        self.gelu = nn.GELU()
        self.ln_2 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.attn.in_proj_bias.data.zero_()
        self.attn.out_proj.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = attn_out + residual
        residual = x
        x = self.ln_2(x)
        activeated_x = self.gelu(self.ffn_up(x))
        x = self.ffn_down(activeated_x)
        x = residual + x
        return x


# ===============================================================================
# Allo BF16 Version
# ===============================================================================

# ----------------------------------------------------------------
# LayerNorm (bf16)
# ----------------------------------------------------------------
norm = ExternalModule(
    top="layer_norm_bf16",
    impl_path=KERNEL_BF16_PATH + "layer_norm_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0
norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def layer_norm_kernel(
    input_x: Ty[NORM_SEQ_TILE, EMBD],
    weight: Ty[EMBD],
    bias: Ty[EMBD],
    output_x: Ty[NORM_SEQ_TILE, EMBD],
):
    pipe: Stream[Ty[NORM_TILE, EMBD], 1][NORM_P0]

    @df.kernel(mapping=[NORM_P0], args=[input_x, weight])
    def norm_no_bias(
        local_input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_weight: Ty[EMBD] @ norm_arg_layout,
    ):
        pi = df.get_pid()
        tmp: Ty[NORM_TILE, EMBD] = 0
        norm(local_input_x, local_weight, tmp)
        pipe[pi].put(tmp)

    @df.kernel(mapping=[NORM_P0], args=[bias, output_x])
    def norm_add_bias(
        local_bias: Ty[EMBD] @ norm_arg_layout,
        local_output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        pi = df.get_pid()
        data = pipe[pi].get()
        local_output_x[:, :] = allo.add(data, local_bias)

# ----------------------------------------------------------------
# GEMM (bf16) using allo.library.aie.modules.gemm with 64x64 tiles
# ----------------------------------------------------------------
LINEAR_TILE = 64

# QKV + output projections: SEQ x EMBD x EMBD
gemm_embd_embd_kernel, gemm_embd_embd_mp = GEMM(
    SEQ, EMBD, EMBD,
    SEQ // LINEAR_TILE, EMBD // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# Attention value: SEQ x HEAD_DIM x SEQ
gemm_head_seq_kernel, gemm_head_seq_mp = GEMM(
    SEQ, HEAD_DIM, SEQ,
    SEQ // LINEAR_TILE, HEAD_DIM // LINEAR_TILE, SEQ // LINEAR_TILE,
    Ty, Ty,
)

# FFN up: SEQ x FFN_HID x EMBD
gemm_hid_embd_kernel, gemm_hid_embd_mp = GEMM(
    SEQ, FFN_HID, EMBD,
    SEQ // LINEAR_TILE, FFN_HID // LINEAR_TILE, EMBD // LINEAR_TILE,
    Ty, Ty,
)

# FFN down: SEQ x EMBD x FFN_HID (3072 = 768 * 4)
# Reuse gemm_embd_embd (1024x768x768) and call 4 times, accumulating partial results.
# K=3072 with tile=64 overflows program memory, K=128+ overflows data memory.
FFN_DOWN_K_CHUNKS = FFN_HID // EMBD  # 3072 / 768 = 4

# ----------------------------------------------------------------
# Attention Score: Q[SEQ,64] @ K[SEQ,64]^T * scale = [SEQ,SEQ]
# Use GEMM(SEQ, SEQ, HEAD_DIM) with Pk=1 (single K tile) — 12 calls vs 3072
# ----------------------------------------------------------------
ATTN_SCALE = NP_DTYPE(1.0 / (HEAD_DIM ** 0.5))  # 0.125 in bf16
gemm_score_kernel, gemm_score_mp = GEMM(
    SEQ, SEQ, HEAD_DIM,
    SEQ // LINEAR_TILE, SEQ // LINEAR_TILE, HEAD_DIM // LINEAR_TILE,
    Ty, Ty,
)

# ----------------------------------------------------------------
# Softmax (float32 - bf16 overflows AIE program memory)
# ----------------------------------------------------------------
Ty_f32 = float32

softmax_ext = ExternalModule(
    top="softmax_float32_seq1024",
    impl_path=KERNEL_LIB_PATH + "v1_softmax_float.cc",
    input_idx=[0],
    output_idx=[1],
)
SOFTMAX_PHYS_COLS = 512
SOFTMAX_PHYS_ROWS_PER_LOGICAL = SEQ // SOFTMAX_PHYS_COLS
SOFTMAX_NUM_CORES = 4
SOFTMAX_KERNEL_PHYS_ROWS = 4
SOFTMAX_BATCH_PHYS_ROWS = SOFTMAX_NUM_CORES * SOFTMAX_KERNEL_PHYS_ROWS
SOFTMAX_BATCH_LOGICAL_ROWS = SOFTMAX_BATCH_PHYS_ROWS // SOFTMAX_PHYS_ROWS_PER_LOGICAL
SOFTMAX_NUM_BATCHES = SEQ // SOFTMAX_BATCH_LOGICAL_ROWS
SOFTMAX_Ly = [S(0), S(1)]

@df.region()
def softmax_kernel(
    Input: Ty_f32[SOFTMAX_BATCH_PHYS_ROWS, SOFTMAX_PHYS_COLS],
    Output: Ty_f32[SOFTMAX_BATCH_PHYS_ROWS, SOFTMAX_PHYS_COLS],
):
    @df.kernel(mapping=[SOFTMAX_NUM_CORES, 1], args=[Input, Output])
    def core(
        local_Input: Ty_f32[SOFTMAX_BATCH_PHYS_ROWS, SOFTMAX_PHYS_COLS] @ SOFTMAX_Ly,
        local_Output: Ty_f32[SOFTMAX_BATCH_PHYS_ROWS, SOFTMAX_PHYS_COLS] @ SOFTMAX_Ly,
    ):
        softmax_ext(local_Input, local_Output)

# ----------------------------------------------------------------
# GELU (bf16)
# ----------------------------------------------------------------
gelu_ext = ExternalModule(
    top="gelu_bf16",
    impl_path=KERNEL_BF16_PATH + "gelu_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)
GELU_P0 = 4
GELU_P1 = 4
GELU_SEQ_TILE = 16
GELU_Ly = [S(0), S(1)]

@df.region()
def gelu_kernel(
    input_x: Ty[GELU_SEQ_TILE, FFN_HID],
    output_x: Ty[GELU_SEQ_TILE, FFN_HID],
):
    @df.kernel(mapping=[GELU_P0, GELU_P1], args=[input_x, output_x])
    def core(
        local_input_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
        local_output_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
    ):
        gelu_ext(local_input_x, local_output_x)


# ##############################################################
# BUILD
# ##############################################################
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm_bf16.prj")

gemm_embd_embd_mod = df.build(
    gemm_embd_embd_kernel,
    project="gemm_embd_embd_bf16.prj",
    target="aie",
    mapping_primitives=gemm_embd_embd_mp,
)
gemm_head_seq_mod = df.build(
    gemm_head_seq_kernel,
    project="gemm_head_seq_bf16.prj",
    target="aie",
    mapping_primitives=gemm_head_seq_mp,
)
gemm_hid_embd_mod = df.build(
    gemm_hid_embd_kernel,
    project="gemm_hid_embd_bf16.prj",
    target="aie",
    mapping_primitives=gemm_hid_embd_mp,
)
# gemm_embd_embd_mod is reused for FFN down (called 4 times with K-chunking)

gemm_score_mod = df.build(
    gemm_score_kernel,
    project="gemm_score_bf16.prj",
    target="aie",
    mapping_primitives=gemm_score_mp,
)
softmax_mod = df.build(
    softmax_kernel, target="aie", project="softmax_f32.prj"
)
gelu_mod = df.build(gelu_kernel, target="aie", project="gelu_bf16.prj")


# ##############################################################
# TOOL FUNCTIONS
# ##############################################################
def layernorm(input_x, weight, bias, output_x):
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_input = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        layer_norm_mod(
            tile_input,
            weight,
            bias,
            output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
        )


def softmax(attention_score, attention_weight):
    """Softmax runs in float32 (bf16 kernel overflows AIE program memory).
    Converts bf16 attention scores to float32, runs softmax, returns float32."""
    for k in range(N_HEAD):
        score_head = np.ascontiguousarray(
            attention_score[:, k, :].astype(np.float32)
        ).reshape(-1, SOFTMAX_PHYS_COLS)
        output_phys = np.zeros_like(score_head)
        for b in range(SOFTMAX_NUM_BATCHES):
            start = b * SOFTMAX_BATCH_PHYS_ROWS
            end = start + SOFTMAX_BATCH_PHYS_ROWS
            batch_in = np.ascontiguousarray(score_head[start:end])
            batch_out = np.zeros(
                (SOFTMAX_BATCH_PHYS_ROWS, SOFTMAX_PHYS_COLS), dtype=np.float32
            )
            softmax_mod(batch_in, batch_out)
            output_phys[start:end] = batch_out
        attention_weight[:, k * SEQ : (k + 1) * SEQ] = output_phys.reshape(SEQ, SEQ)


def add_residual(residual, x):
    """Element-wise add in numpy (bf16)."""
    residual += x


def vision_block(x_np: np.ndarray, params: dict):
    # ##############################################################
    # FORWARD (all bf16 except softmax which is float32)
    # ##############################################################
    x = x_np.astype(NP_DTYPE)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=NP_DTYPE)
    layernorm(residual, params["W_norm_1"], params["b_norm_1"], x)

    # QKV projections using GEMM module: SEQ x EMBD @ EMBD x EMBD
    query = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    key = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    value = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_embd_embd_mod(x, params["Wq"], query)
    gemm_embd_embd_mod(x, params["Wk"], key)
    gemm_embd_embd_mod(x, params["Wv"], value)

    # Attention score: (Q * scale) @ K^T per head (12 GEMM calls vs 3072 tiled calls)
    # Scale Q once before GEMM to fuse scale into computation
    query_scaled = (query.astype(np.float32) * float(ATTN_SCALE)).astype(NP_DTYPE)
    attention_score = np.empty((SEQ, N_HEAD, SEQ), dtype=NP_DTYPE)
    for k in range(N_HEAD):
        Q_head = np.ascontiguousarray(
            query_scaled[:, k * HEAD_DIM : (k + 1) * HEAD_DIM]
        )
        K_head_T = np.ascontiguousarray(
            key[:, k * HEAD_DIM : (k + 1) * HEAD_DIM].T
        )  # [HEAD_DIM, SEQ]
        score = np.zeros((SEQ, SEQ), dtype=NP_DTYPE)
        gemm_score_mod(Q_head, K_head_T, score)
        attention_score[:, k, :] = score

    # Softmax (float32, converts bf16 scores internally)
    if USE_ALL_NPU_KERNELS:
        # Softmax output stays float32, convert to bf16 for attn value
        attn_weight_f32 = np.zeros((SEQ, N_HEAD * SEQ), dtype=np.float32)
        softmax(attention_score, attn_weight_f32)
        attn_weight = attn_weight_f32.astype(NP_DTYPE)
    else:
        tensor_atten_score = torch.from_numpy(
            attention_score.astype(np.float32)
        )
        attn_weight = (
            F.softmax(tensor_atten_score, dim=-1).to(torch.bfloat16).float().numpy().astype(NP_DTYPE)
        )

    # Attention value using GEMM module: SEQ x HEAD_DIM x SEQ per head
    attn_value = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for k in range(N_HEAD):
        head_weight = (
            attn_weight[:, k * SEQ : (k + 1) * SEQ]
            if USE_ALL_NPU_KERNELS
            else attn_weight[:, k, :]
        )
        head_value = np.ascontiguousarray(
            value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM]
        )
        head_out = np.zeros((SEQ, HEAD_DIM), dtype=NP_DTYPE)
        gemm_head_seq_mod(head_weight, head_value, head_out)
        attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM] = head_out

    # Output projection: SEQ x EMBD x EMBD
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    gemm_embd_embd_mod(attn_value, params["Wo"], x)

    # Add residual
    add_residual(residual, x)

    # LayerNorm 2
    layernorm(residual, params["W_norm_2"], params["b_norm_2"], x)

    # FFN up: SEQ x FFN_HID x EMBD
    ffn_up_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
    gemm_hid_embd_mod(x, params["W_up"], ffn_up_x)

    # GELU (bf16)
    if USE_ALL_NPU_KERNELS:
        activeated_x = np.zeros((SEQ, FFN_HID), dtype=NP_DTYPE)
        for i in range(SEQ // GELU_SEQ_TILE):
            gelu_mod(
                ffn_up_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
                activeated_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
            )
    else:
        tensor_ffn_up_x = torch.from_numpy(ffn_up_x.astype(np.float32)).to(
            torch.bfloat16
        )
        gelu_func = nn.GELU()
        activeated_x = (
            gelu_func(tensor_ffn_up_x.float())
            .to(torch.bfloat16)
            .float()
            .numpy()
            .astype(NP_DTYPE)
        )

    # FFN down: SEQ x EMBD x FFN_HID
    # Split K=3072 into 4 chunks of 768, reuse gemm_embd_embd (1024x768x768)
    x = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
    for chunk in range(FFN_DOWN_K_CHUNKS):
        chunk_A = np.ascontiguousarray(
            activeated_x[:, chunk * EMBD : (chunk + 1) * EMBD]
        )
        chunk_B = np.ascontiguousarray(
            params["W_down"][chunk * EMBD : (chunk + 1) * EMBD, :]
        )
        partial = np.zeros((SEQ, EMBD), dtype=NP_DTYPE)
        gemm_embd_embd_mod(chunk_A, chunk_B, partial)
        x += partial

    # Add residual
    add_residual(residual, x)
    return residual


if __name__ == "__main__":
    ref_model = MiniVit().eval()

    # Extract weights and convert to bf16
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params = {
        "Wq": p["attn.in_proj_weight"][:EMBD, :].T.astype(NP_DTYPE),
        "Wk": p["attn.in_proj_weight"][EMBD : 2 * EMBD, :].T.astype(NP_DTYPE),
        "Wv": p["attn.in_proj_weight"][2 * EMBD :, :].T.astype(NP_DTYPE),
        "Wo": p["attn.out_proj.weight"].T.astype(NP_DTYPE),
        "W_up": p["ffn_up.weight"].T.astype(NP_DTYPE),
        "W_down": p["ffn_down.weight"].T.astype(NP_DTYPE),
        "W_norm_1": p["ln_1.weight"].astype(NP_DTYPE),
        "b_norm_1": p["ln_1.bias"].astype(NP_DTYPE),
        "W_norm_2": p["ln_2.weight"].astype(NP_DTYPE),
        "b_norm_2": p["ln_2.bias"].astype(NP_DTYPE),
    }

    # Fixed random input
    x_float = torch.randn(SEQ, EMBD)

    # PyTorch bf16 reference
    with torch.no_grad():
        ref_model_bf16 = ref_model.to(torch.bfloat16)
        t0 = time.time()
        sample = ref_model_bf16(x_float.to(torch.bfloat16))
        t1 = time.time()
        print(f"PyTorch bf16 forward time: {t1 - t0:.6f} s")
        ref_out = sample.float().numpy()

    # Allo bf16
    a0 = time.time()
    allo_out = vision_block(x_float.numpy(), params)
    a1 = time.time()
    print(f"Allo bf16 forward time:    {a1 - a0:.6f} s")

    # Correctness (bf16 tolerance)
    np.testing.assert_allclose(
        allo_out.astype(np.float32), ref_out, atol=1e-1, rtol=1e-1
    )
    print("Allo bf16 block matches PyTorch bf16 reference within tolerance")
