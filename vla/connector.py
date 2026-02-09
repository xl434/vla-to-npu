"""
class Idefics3Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5) # height, width = sqrt(seq)
        x = x.view(bsz, height, width, embed_dim) # reshape seq dim -> H * W
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor) # reshape -> bsz, H, W/sf, EDIM * sf
        x = x.permute(0, 2, 1, 3) # reorder -> bsz, W/sf, H, EDIM * sf
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2)) # reshape -> bsz, W/sf, H/sf, EDIM * sf^2
        x = x.permute(0, 2, 1, 3) # reorder -> bsz, H/sf, W/sf, EDIM * sf^2
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2)) # reshape seq dim -> (H * W)/sf^2
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

class Idefics3SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)

vlm_with_expert.vlm.model.vision_model.post_layernorm                            | LayerNorm          | [(1, 1024, 768)] -> [(1, 1024, 768)]
vlm_with_expert.vlm.model.connector.modality_projection.proj                     | Linear             | [(1, 64, 12288)] -> [(1, 64, 960)]
"""

import os
import pytest
import torch
import torch.nn as nn
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

BATCH = 1
SEQ = 1024
EMBD = 768
SF = 4
NEW_SEQ = 64 # 1024 / 4 / 4
NEW_EMBD = 12288 # 768 * 4 * 4
TEXT = 960

# ===============================================================================
# Allo Version
# ===============================================================================

Ty = float32

linear_A_layout = [S(0), R]
linear_C_layout = [R, S(0)]

@df.region()
def copy(A: Ty[4, EMBD], C: Ty[1, EMBD*4]):
    @df.kernel(mapping=[4], args=[A,C])
    def mod(
        local_A: Ty[4, EMBD] @ linear_A_layout,
        local_C: Ty[1, EMBD*4] @ linear_C_layout,
    ):
        local_C[:,:] = local_A[:,:]

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

# ##############################################################
# BUILD
# ##############################################################
copy_mod = df.build(copy, target="aie", project="copy.proj")
linear_matmul_mod = df.build(linear_matmul_kernel, target="aie", project="linear_matmul.proj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie", project="linear_accumulate.proj")

# ##############################################################
# TOOL
# ##############################################################
def pixel_shuffle(A, C):
    for j in range(NEW_SEQ):
        i = j * 16
        for k in range(4):
            print(k * EMBD * 4, (k+1)*EMBD*4)
            tile_A = A[ 
                i + k * 4 : i + (k + 1) * 4,
                :
            ]
            tile_C = C[
                j: j + 1,
                k * EMBD * 4 : (k + 1) * (EMBD * 4)
            ]
            copy_mod(tile_A, tile_C)

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

# Weight matrix is size [12288, 960]
def connector_block(x_fp32: np.ndarray, params: dict):
    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)
    x = x.reshape(SEQ, EMBD)
    x_shuffled = np.zeros((NEW_SEQ, NEW_EMBD), dtype=np.float32)
    out = np.zeros((NEW_SEQ, TEXT), dtype=np.float32)
    pixel_shuffle(x, x_shuffled)
    linear_projection(x_shuffled, params["W"], out, NEW_SEQ, TEXT, NEW_EMBD)
    return out

# ##############################################################
# TEST
# ##############################################################
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np.float32)
    w = np.random.randn(EMBD*(SF**2), TEXT).astype(np.float32)
    dict = {"W": w}
    out = connector_block(x, dict)

    # test python
    x = x.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    expected = x @ w
    np.testing.assert_allclose(out, expected, rtol=1e-5)