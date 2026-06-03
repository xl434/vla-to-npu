# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# test_flash_attn_bf16.py — Flash Attention correctness prototype (BFloat16)
#
# OPT-C: This is a CORRECTNESS PROTOTYPE only.
# Not expected to be faster than current approach due to NPU stateless dispatch
# constraints (~28ms/call Python XRT, each hw_context open/close is expensive).
#
# Run from /home/xl434/vla-to-npu/vla/ directory:
#   cd /home/xl434/vla-to-npu/vla
#   python3 ../kernel_testing/flash_attn/test_flash_attn_bf16.py
#
# What this tests:
#   1. ExternalModule flash_attn_score_bf16: Q[32,64] @ K^T[64,64] → scores[32,64] * scale
#   2. Flash attention algorithm (online softmax) using GEMM NPU kernel for score+value
#   3. Correctness vs. standard attention reference (PyTorch float32)
#
# Timing note: Python XRT dispatch overhead ~28ms/call dominates.
# C++ dispatch with the same kernels would be ~4-12ms/call.

import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16 as np_bfloat16
import time

import allo
import allo.dataflow as df
from allo.ir.types import bfloat16 as Ty_bf16
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from allo.library.aie.modules.gemm import GEMM

np.random.seed(0)
torch.manual_seed(0)

S = Layout.Shard
R = Layout.Replicate

# ============================================================
# Configuration
# ============================================================
FLASH_Q_TILE  = 32   # Q tile rows for ExternalModule test
FLASH_KV_TILE = 64   # K/V tile size for flash attention
HEAD_DIM      = 64   # attention head dimension
SEQ           = 1024 # vision encoder sequence length

NP_DTYPE = np_bfloat16

# ============================================================
# GEMM module for flash attention score and value computation
# Both score [64,64]@[64,64]=[64,64] and value [64,64]@[64,64]=[64,64]
# use the same GEMM shape.
#
# FLASH_GEMM: GEMM(64, 64, 64, 2, 2, 2, bf16, bf16)
# ============================================================
TILE = FLASH_KV_TILE  # 64

flash_gemm_kernel, flash_gemm_mp = GEMM(
    TILE, TILE, TILE,
    2, 2, 2,      # Pm=2, Pn=2, Pk=2 → 8 cores (fits in 16-core NPU)
    Ty_bf16, Ty_bf16,
)

flash_gemm_mod = df.build(
    flash_gemm_kernel,
    project="flash_attn_test.prj",
    target="aie",
    mapping_primitives=flash_gemm_mp,
)

# ============================================================
# ExternalModule: flash_attn_score_bf16
# Q_tile[32][64] @ K_tile_T[64][64] → scores[32][64] * (1/sqrt(64))
# Allo convention: input_idx=[0,2], output_idx=[1]
# slot3=Q_tile, slot4=scores(out), slot5=K_tile_T
# ============================================================
KERNEL_BF16_PATH = "../cc/bf16_vla/"

flash_score_ext = ExternalModule(
    top="flash_attn_score_bf16",
    impl_path=KERNEL_BF16_PATH + "flash_attn_tile_bf16.cc",
    input_idx=[0, 2],
    output_idx=[1],
)

@df.region()
def flash_score_kernel(
    Q_tile:    Ty_bf16[FLASH_Q_TILE, HEAD_DIM],
    scores:    Ty_bf16[FLASH_Q_TILE, HEAD_DIM],
    K_tile_T:  Ty_bf16[HEAD_DIM, HEAD_DIM],
):
    @df.kernel(mapping=[1, 1], args=[Q_tile, scores, K_tile_T])
    def core(
        local_Q:   Ty_bf16[FLASH_Q_TILE, HEAD_DIM] @ [S(1), S(0)],
        local_out: Ty_bf16[FLASH_Q_TILE, HEAD_DIM] @ [S(1), S(0)],
        local_K_T: Ty_bf16[HEAD_DIM, HEAD_DIM]     @ [S(1), S(0)],
    ):
        flash_score_ext(local_Q, local_out, local_K_T)

flash_score_mod = df.build(
    flash_score_kernel,
    project="flash_score_test.prj",
    target="aie",
)

# ============================================================
# Online softmax flash attention (NumPy CPU + NPU GEMM)
# ============================================================
ATTN_SCALE = float(1.0 / (HEAD_DIM ** 0.5))  # 0.125


def flash_attention_single_head(Q, K, V):
    """
    Flash attention for a single head using online softmax.
    Q, K, V: [SEQ, HEAD_DIM] bf16
    Returns: output [SEQ, HEAD_DIM] bf16

    Uses flash_gemm_mod (NPU GEMM [64,64]@[64,64]) for both score and value GEMM.
    Online softmax accumulates in float32 on CPU.
    """
    n_tiles = SEQ // TILE
    scale = ATTN_SCALE

    out = np.zeros((SEQ, HEAD_DIM), dtype=np.float32)

    for q_t in range(n_tiles):
        q_slice = Q[q_t * TILE:(q_t + 1) * TILE, :]             # [TILE, HEAD_DIM]
        q_scaled = (q_slice.astype(np.float32) * scale).astype(NP_DTYPE)

        m = np.full(TILE, -np.inf, dtype=np.float32)              # running row max
        l = np.zeros(TILE, dtype=np.float32)                      # running sum
        O_acc = np.zeros((TILE, HEAD_DIM), dtype=np.float32)      # running output

        for kv_t in range(n_tiles):
            K_tile   = K[kv_t * TILE:(kv_t + 1) * TILE, :]       # [TILE, HEAD_DIM]
            K_tile_T = np.ascontiguousarray(K_tile.T)             # [HEAD_DIM, TILE]

            # NPU GEMM: q_scaled[TILE,HEAD_DIM] @ K_tile_T[HEAD_DIM,TILE] → scores[TILE,TILE]
            scores_tile = np.zeros((TILE, TILE), dtype=NP_DTYPE)
            flash_gemm_mod(q_scaled, scores_tile, K_tile_T)
            s = scores_tile.astype(np.float32)

            # Online softmax update
            m_new = np.maximum(m, s.max(axis=1))                  # [TILE]
            exp_s = np.exp(s - m_new[:, None])                    # [TILE, TILE]
            l_new = np.exp(m - m_new) * l + exp_s.sum(axis=1)    # [TILE]

            # Accumulate V contribution
            V_tile = V[kv_t * TILE:(kv_t + 1) * TILE, :]         # [TILE, HEAD_DIM]
            # attn_weights: row-normalized by l_new (unnormalized, we track l)
            attn_weights = (exp_s / l_new[:, None]).astype(NP_DTYPE)  # [TILE, TILE]

            # NPU GEMM: attn_weights[TILE,TILE] @ V_tile[TILE,HEAD_DIM] → v_contrib[TILE,HEAD_DIM]
            v_contrib = np.zeros((TILE, HEAD_DIM), dtype=NP_DTYPE)
            flash_gemm_mod(attn_weights, v_contrib, V_tile)

            # Update running output:
            # O_acc = (exp(m - m_new) * l / l_new)[:, None] * O_acc + v_contrib
            correction = (np.exp(m - m_new) * l / l_new)[:, None]  # [TILE, 1]
            O_acc = O_acc * correction + v_contrib.astype(np.float32)

            m, l = m_new, l_new

        out[q_t * TILE:(q_t + 1) * TILE, :] = O_acc

    return out.astype(NP_DTYPE)


# ============================================================
# Reference standard attention (PyTorch float32)
# ============================================================
def reference_attention_single_head(Q, K, V):
    """Standard scaled dot-product attention (float32 reference)."""
    Q_f = torch.from_numpy(Q.astype(np.float32))
    K_f = torch.from_numpy(K.astype(np.float32))
    V_f = torch.from_numpy(V.astype(np.float32))

    scores = (Q_f @ K_f.T) * ATTN_SCALE           # [SEQ, SEQ]
    weights = torch.softmax(scores, dim=-1)         # [SEQ, SEQ]
    out = (weights @ V_f)                           # [SEQ, HEAD_DIM]
    return out.numpy().astype(NP_DTYPE)


# ============================================================
# Test: ExternalModule flash_score_bf16 vs CPU reference
# ============================================================
def test_flash_score_external_module():
    print("=" * 60)
    print("Test 1: ExternalModule flash_attn_score_bf16")
    print("=" * 60)
    Q_tile = np.random.randn(FLASH_Q_TILE, HEAD_DIM).astype(NP_DTYPE)
    K_tile = np.random.randn(HEAD_DIM, HEAD_DIM).astype(NP_DTYPE)   # already transposed
    scores_npu = np.zeros((FLASH_Q_TILE, HEAD_DIM), dtype=NP_DTYPE)

    flash_score_mod(Q_tile, scores_npu, K_tile)

    # CPU reference: Q_tile @ K_tile * scale
    Q_f = Q_tile.astype(np.float32)
    K_f = K_tile.astype(np.float32)
    scores_ref = (Q_f @ K_f * ATTN_SCALE).astype(NP_DTYPE)

    max_err = np.max(np.abs(scores_npu.astype(np.float32) - scores_ref.astype(np.float32)))
    print(f"  Max abs error (NPU vs CPU): {max_err:.6f}")
    np.testing.assert_allclose(
        scores_npu.astype(np.float32),
        scores_ref.astype(np.float32),
        atol=1e-1, rtol=1e-1,
        err_msg="flash_attn_score_bf16 NPU vs CPU mismatch"
    )
    print("  PASSED")


# ============================================================
# Test: Flash attention vs standard attention
# ============================================================
def test_flash_attention_vs_standard():
    print("=" * 60)
    print("Test 2: Flash attention (online softmax + NPU GEMM) vs standard attention")
    print("=" * 60)
    print(f"  SEQ={SEQ}, HEAD_DIM={HEAD_DIM}, TILE={TILE}")
    print(f"  This uses {SEQ//TILE} Q-tiles × {SEQ//TILE} KV-tiles = {(SEQ//TILE)**2} GEMM calls")

    Q = np.random.randn(SEQ, HEAD_DIM).astype(NP_DTYPE)
    K = np.random.randn(SEQ, HEAD_DIM).astype(NP_DTYPE)
    V = np.random.randn(SEQ, HEAD_DIM).astype(NP_DTYPE)

    # Reference: standard attention
    ref_out = reference_attention_single_head(Q, K, V)

    # Flash attention with NPU GEMM
    t0 = time.time()
    flash_out = flash_attention_single_head(Q, K, V)
    t1 = time.time()
    print(f"  Flash attention time: {t1 - t0:.3f} s")
    print(f"  (Note: Python XRT dispatch ~28ms/call × {(SEQ//TILE)**2} calls dominates)")
    print(f"  (C++ dispatch: ~4-12ms/call × {(SEQ//TILE)**2} = est. {4*(SEQ//TILE)**2/1000:.1f}s)")

    max_err  = np.max(np.abs(flash_out.astype(np.float32) - ref_out.astype(np.float32)))
    mean_err = np.mean(np.abs(flash_out.astype(np.float32) - ref_out.astype(np.float32)))
    print(f"  Max abs error:  {max_err:.4f}")
    print(f"  Mean abs error: {mean_err:.6f}")

    np.testing.assert_allclose(
        flash_out.astype(np.float32),
        ref_out.astype(np.float32),
        atol=2e-1, rtol=1e-1,
        err_msg="Flash attention output does not match standard attention reference"
    )
    print("  PASSED")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Flash Attention BF16 Correctness Test (OPT-C)")
    print("=" * 60)
    print("Note: This is a CORRECTNESS PROTOTYPE only.")
    print("      Python XRT dispatch overhead dominates timing.")
    print()

    test_flash_score_external_module()
    print()
    test_flash_attention_vs_standard()
    print()
    print("All tests PASSED.")
