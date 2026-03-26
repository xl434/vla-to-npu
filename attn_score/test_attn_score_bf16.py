# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import torch.nn.functional as F
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

KERNEL_LIB_PATH = "../cc/bf16_old/"

S = Layout.Shard
R = Layout.Replicate
Ty = bfloat16

HEAD_DIM = 64
SCALE = 1.0 / (HEAD_DIM ** 0.5)  # 0.125

ATTN_P0 = 2
ATTN_P1 = 2
ATTN_SCORE_M_TILE = ATTN_P0 * 32  # 64
ATTN_SCORE_N_TILE = ATTN_P1 * 32  # 64

ATTN_SCORE_LyA = [S(0), R]
ATTN_SCORE_LyB = [S(1), R]
ATTN_SCORE_LyC = [S(0), S(1)]


def _test_attn_score_bf16_single_tile():
    """Test a single tile: Q[64, 64] x K[64, 64]^T * scale -> [64, 64]"""
    attn_score = ExternalModule(
        top="transpose_matmul_with_scale_bf16",
        impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    @df.region()
    def top(
        A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM],
        B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM],
        C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE],
    ):
        @df.kernel(mapping=[ATTN_P0, ATTN_P1], args=[A, B, C])
        def core(
            local_A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
            local_B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
            local_C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
        ):
            attn_score(local_A, local_B, local_C)

    # Random bf16 input
    torch.manual_seed(0)
    Q_np = (np.random.randn(ATTN_SCORE_M_TILE, HEAD_DIM) * 0.1).astype(np_bfloat16)
    K_np = (np.random.randn(ATTN_SCORE_N_TILE, HEAD_DIM) * 0.1).astype(np_bfloat16)

    # PyTorch reference: (Q @ K^T) * scale
    Q_torch = torch.from_numpy(Q_np.astype(np.float32)).to(torch.bfloat16)
    K_torch = torch.from_numpy(K_np.astype(np.float32)).to(torch.bfloat16)
    ref = (Q_torch.float() @ K_torch.float().T * SCALE).to(torch.bfloat16)
    ref_np = ref.float().numpy()

    mod = df.build(top, target="aie")
    C_np = np.zeros((ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE), dtype=np_bfloat16)
    mod(Q_np, K_np, C_np)

    np.testing.assert_allclose(
        C_np.astype(np.float32), ref_np, atol=1e-1, rtol=1e-2
    )
    print("PASS! Attention score bf16 single tile matches PyTorch reference.")


def _test_attn_score_bf16_multi_tile():
    """Test with SEQ=128: 2x2 tiles of 64 each, single head."""
    SEQ = 128

    attn_score = ExternalModule(
        top="transpose_matmul_with_scale_bf16",
        impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale_bf16.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    @df.region()
    def top(
        A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM],
        B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM],
        C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE],
    ):
        @df.kernel(mapping=[ATTN_P0, ATTN_P1], args=[A, B, C])
        def core(
            local_A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
            local_B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
            local_C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
        ):
            attn_score(local_A, local_B, local_C)

    mod = df.build(top, target="aie")

    # Random bf16 input
    torch.manual_seed(42)
    Q_np = (np.random.randn(SEQ, HEAD_DIM) * 0.1).astype(np_bfloat16)
    K_np = (np.random.randn(SEQ, HEAD_DIM) * 0.1).astype(np_bfloat16)

    # Allo tiled computation
    C_np = np.empty((SEQ, SEQ), dtype=np_bfloat16)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            tile_Q = Q_np[i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE, :]
            tile_K = K_np[j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE, :]
            tile_C = np.zeros((ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE), dtype=np_bfloat16)
            mod(
                np.ascontiguousarray(tile_Q),
                np.ascontiguousarray(tile_K),
                tile_C,
            )
            C_np[
                i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
            ] = tile_C

    # PyTorch reference
    Q_torch = torch.from_numpy(Q_np.astype(np.float32)).to(torch.bfloat16)
    K_torch = torch.from_numpy(K_np.astype(np.float32)).to(torch.bfloat16)
    ref = (Q_torch.float() @ K_torch.float().T * SCALE).to(torch.bfloat16)
    ref_np = ref.float().numpy()

    np.testing.assert_allclose(
        C_np.astype(np.float32), ref_np, atol=1e-1, rtol=1e-2
    )
    print("PASS! Attention score bf16 multi-tile (128x128) matches PyTorch reference.")


if __name__ == "__main__":
    _test_attn_score_bf16_single_tile()
    _test_attn_score_bf16_multi_tile()
