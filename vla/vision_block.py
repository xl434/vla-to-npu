# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
(vision_model): SmolVLMVisionTransformer(
            (embeddings): SmolVLMVisionEmbeddings(
              (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
              (position_embedding): Embedding(1024, 768)
            )
            (encoder): SmolVLMEncoder(
              (layers): ModuleList(
                (0-11): 12 x SmolVLMEncoderLayer(
                  (self_attn): SmolVLMVisionAttention(
                    (k_proj): Linear(in_features=768, out_features=768, bias=True)
                    (v_proj): Linear(in_features=768, out_features=768, bias=True)
                    (q_proj): Linear(in_features=768, out_features=768, bias=True)
                    (out_proj): Linear(in_features=768, out_features=768, bias=True)
                  )
                  (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                  (mlp): SmolVLMVisionMLP(
                    (activation_fn): PytorchGELUTanh()
                    (fc1): Linear(in_features=768, out_features=3072, bias=True)
                    (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  )
                  (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                )
              )
            )
            (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, int32, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule
import time


torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

# ===============================================================================
# Model Configuration
# ===============================================================================
USE_ALL_NPU_KERNELS = True  # if False, we will offload softmax and gelu to cpu
KERNEL_LIB_PATH = "../cc/"
BATCH = 1  # fixme: don't care for now
SEQ = 64
EMBD = 768  # 64 * 12
N_HEAD = 12
HEAD_DIM = EMBD // N_HEAD
FFN_HID = EMBD * 4

assert SEQ == 64, "SEQ must be 64 (to use masked softmax external kernel)"
assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"


# ===============================================================================
# Torch Version
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
        attn_out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
            # attn_mask=torch.triu(torch.ones(SEQ, SEQ), 1).bool(),
        )
        x = attn_out + residual
        residual = x
        x = self.ln_2(x)
        activeated_x = self.gelu(self.ffn_up(x))
        x = self.ffn_down(activeated_x)
        x = residual + x
        return x


# ===============================================================================
# Allo Version
# ===============================================================================


Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)
# ----------------------------------------------------------------
# LayerNorm
norm = ExternalModule(
    top="layer_norm",
    impl_path=KERNEL_LIB_PATH + "layer_norm.cc",
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
# Linear
# ----------------------------------------------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
linear_A_layout = [S(0), R]
linear_B_layout = [R, S(1)]
linear_C_layout = [S(0), S(1)]

@df.region()
def linear_matmul_kernel(A: Ty[LINEAR_M, LINEAR_K], B: Ty[LINEAR_K, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[4, 4], args=[A, B, C])
    def gemm(
        local_A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
        local_B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        local_C[:, :] = allo.matmul(local_A, local_B)

@df.region()
def linear_accumulate_kernel(A: Ty[LINEAR_M, LINEAR_N], B: Ty[LINEAR_M, LINEAR_N], C: Ty[LINEAR_M, LINEAR_N]):
    @df.kernel(mapping=[2, 4], args=[A, B, C])
    def core(
        local_A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        local_B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        local_C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        local_C[:, :] = allo.add(local_A, local_B)

# ----------------------------------------------------------------
# Attention Score
# ----------------------------------------------------------------
attn_score = ExternalModule(
    top="transpose_matmul_with_scale",
    impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
ATTN_P0 = 2
ATTN_P1 = 2
ATTN_SCORE_M_TILE = ATTN_P0 * 32
ATTN_SCORE_N_TILE = ATTN_P1 * 32
ATTN_SCORE_LyA = [S(0), R]
ATTN_SCORE_LyB = [S(1), R]
ATTN_SCORE_LyC = [S(0), S(1)]

@df.region()
def attn_score_kernel(A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM], B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM], C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE]):
    @df.kernel(mapping=[ATTN_P0, ATTN_P1], args=[A, B, C])
    def core(
        local_A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
        local_B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
        local_C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
    ):
        attn_score(local_A, local_B, local_C)

# ----------------------------------------------------------------
# Masked Softmax
# ----------------------------------------------------------------
softmax = ExternalModule(
    top="softmax_float32",
    impl_path=KERNEL_LIB_PATH + "v1_softmax_float.cc",
    input_idx=[0],
    output_idx=[1],
)
Tint = int32
SOFTMAX_P0 = 2
SOFTMAX_P1 = 3
SOFTMAX_HEAD_TILE = SOFTMAX_P1
SOFTMAX_SEQ_TILE = SEQ // SOFTMAX_P0
SOFTMAX_Ly = [S(0), S(1)]
SOFTMAX_ROW_Ly = [S(0)]

@df.region()
def softmax_kernel(input_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE], output_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE]):
    @df.kernel(mapping=[SOFTMAX_P0, SOFTMAX_P1], args=[input_x, output_x])
    def core(
        local_input_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
        local_output_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
    ):
        softmax(local_input_x, local_output_x)

# ----------------------------------------------------------------
# Gelu
# ----------------------------------------------------------------
gelu = ExternalModule(
    top="gelu_float32",
    impl_path=KERNEL_LIB_PATH + "gelu.cc",
    input_idx=[0],
    output_idx=[1],
)
GELU_P0 = 4
GELU_P1 = 4
GELU_SEQ_TILE = 16
GELU_Ly = [S(0), S(1)]

@df.region()
def gelu_kernel(input_x: Ty[GELU_SEQ_TILE, FFN_HID], output_x: Ty[GELU_SEQ_TILE, FFN_HID]):
    @df.kernel(mapping=[GELU_P0, GELU_P1], args=[input_x, output_x])
    def core(
        local_input_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
        local_output_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
    ):
        gelu(local_input_x, local_output_x)

# ##############################################################
# BUILD
# ##############################################################
layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm.prj")
linear_matmul_mod = df.build(
    linear_matmul_kernel, target="aie", project="linear_matmul.prj"
)
linear_accumulate_mod = df.build(
    linear_accumulate_kernel, target="aie", project="linear_accumulate.prj"
)
attn_score_mod = df.build(
    attn_score_kernel, target="aie", project="attn_score.prj"
)
softmax_mod = df.build(
    softmax_kernel, target="aie", project="softmax.prj"
)
gelu_mod = df.build(gelu_kernel, target="aie", project="gelu.prj")

# ##############################################################
# TOOL
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

def linear_projection(A, B, C, M, N, K):
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
            for k in range(K // LINEAR_K):
                tile_A = A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                ]
                tile_B = B[
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ]
                linear_matmul_mod(tile_A, tile_B, C_tmp)
                linear_accumulate_mod(
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    C_tmp,
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )

def add_residual(residual, x, M, N):
    """
    reuse 'linear_accumulate_mod' for residual
    residual = residual + x
    """
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            linear_accumulate_mod(
                residual[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                x[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                residual[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
            )

def softmax(attention_score, attention_weight):
    for i in range(N_HEAD // SOFTMAX_HEAD_TILE):
        softmax_mod(
            attention_score[
                :, i * SOFTMAX_HEAD_TILE : (i + 1) * SOFTMAX_HEAD_TILE, :
            ],
            attention_weight[
                :,
                i * (SOFTMAX_HEAD_TILE * SEQ) : (i + 1) * (SOFTMAX_HEAD_TILE * SEQ),
            ],
        )



def vision_block(x_fp32: np.ndarray, params: dict):
    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=np.float32)
    layernorm(residual, params["W_norm_1"], params["b_norm_1"], x)

    # qkv projections (M = SEQ, N = EMBD, K = EMBD)
    query = np.zeros((SEQ, EMBD)).astype(np.float32)
    key = np.zeros((SEQ, EMBD)).astype(np.float32)
    value = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(x, params["Wq"], query, SEQ, EMBD, EMBD)
    linear_projection(x, params["Wk"], key, SEQ, EMBD, EMBD)
    linear_projection(x, params["Wv"], value, SEQ, EMBD, EMBD)

    # attention score
    attention_score = np.empty((SEQ, N_HEAD, SEQ), dtype=np.float32)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            for k in range(N_HEAD):
                attn_score_mod(
                    query[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    key[
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    attention_score[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k,
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                    ],
                )

    # safe softmax
    if USE_ALL_NPU_KERNELS:
        attn_weight = np.zeros((SEQ, N_HEAD * SEQ)).astype(np.float32)
        softmax(attention_score, attn_weight)
    else:
        mask = torch.triu(torch.ones(SEQ, SEQ), 1).bool()
        mask = np.repeat(mask[:, np.newaxis, :], N_HEAD, axis=1)
        attention_score[mask == 1] = -np.inf
        tensor_atten_score = torch.from_numpy(attention_score)
        attn_weight = F.softmax(tensor_atten_score, dim=-1)
        attn_weight = attn_weight.numpy()

    # attention value
    attn_value = np.zeros((SEQ, EMBD)).astype(np.float32)
    for k in range(N_HEAD):
        linear_projection(
            (
                attn_weight[:, k * SEQ : (k + 1) * SEQ]
                if USE_ALL_NPU_KERNELS
                else attn_weight[:, k, :]
            ),
            value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            SEQ,
            HEAD_DIM,
            SEQ,
        )
    # output projection
    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(attn_value, params["Wo"], x, SEQ, EMBD, EMBD)
    # add residual
    add_residual(residual, x, SEQ, EMBD)
    # norm
    layernorm(residual, params["W_norm_2"], params["b_norm_2"], x)
    # up projection
    ffn_up_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_up"], ffn_up_x, SEQ, FFN_HID, EMBD)

    # if USE_ALL_NPU_KERNELS:
    if USE_ALL_NPU_KERNELS:
        activeated_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
        for i in range(SEQ // GELU_SEQ_TILE):
            gelu_mod(
                ffn_up_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
                activeated_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
            )
    else:
        tensor_ffn_up_x = torch.from_numpy(ffn_up_x)
        gelu_func = nn.GELU()
        activeated_x = gelu_func(tensor_ffn_up_x).numpy()

    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(activeated_x, params["W_down"], x, SEQ, EMBD, FFN_HID)
    add_residual(residual, x, SEQ, EMBD)
    return residual

if __name__ == "__main__":
    ref_model = MiniVit().eval()
    # reference weights (float32)
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params_fp32 = {
        "Wq": p["attn.in_proj_weight"][:EMBD, :].T,
        "Wk": p["attn.in_proj_weight"][EMBD : 2 * EMBD, :].T,
        "Wv": p["attn.in_proj_weight"][2 * EMBD :, :].T,
        "Wo": p["attn.out_proj.weight"].T,
        "W_up": p["ffn_up.weight"].T,
        "W_down": p["ffn_down.weight"].T,
        "W_norm_1": p["ln_1.weight"],
        "b_norm_1": p["ln_1.bias"],
        "W_norm_2": p["ln_2.weight"],
        "b_norm_2": p["ln_2.bias"],
    }

    params = {
        k: v.astype(np.float32) if isinstance(v, np.ndarray) else v
        for k, v in params_fp32.items()
    }

    # fixed random input
    x_float = torch.randn(SEQ, EMBD)

    # ---- timings first ----
    t0 = time.time()
    sample = ref_model(x_float)
    t1 = time.time()
    print(f"PyTorch forward time: {t1 - t0:.6f} s")

    a0 = time.time()
    allo_out = vision_block(x_float.numpy(), params)
    a1 = time.time()
    print(f"Allo forward time:    {a1 - a0:.6f} s")

    # ---- correctness after timings ----
    np.testing.assert_allclose(allo_out, sample.detach().numpy(), rtol=1e-1)
    print("Allo float32 block matches PyTorch float32 reference within tolerance ✔️")