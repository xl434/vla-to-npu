# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused GEMM + SiLU kernel for SmolVLA action expert gate_proj FFN.

Fusion strategy: SiLU is applied inline in the last K-chunk GEMM core,
immediately after f32 accumulation and f32→bf16 conversion. This eliminates
one XRT dispatch (~28ms overhead) vs. calling GEMM and SiLU separately.

Dimensions (action expert gate_proj):
  M=32, N=2048, K=768
  Pm=8, Pn=8, Pk=12  →  per-core tile: Mt=4, Nt=256 (matches silu_256_bf16.cc)
  col_num=4, row_num=4  →  Pn%4=0 ✓, Pm%4=0 ✓
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16 as np_bfloat16

import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16 as Ty_bf16
from allo.ir.types import Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule
from allo.backend.aie import is_available

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cc.gemm import gen_gemm_mapping_primitive

S = Layout.Shard
R = Layout.Replicate

# ── Dimensions ────────────────────────────────────────────────────────────────
M, N, K   = 32, 2048, 768
Pm, Pn, Pk = 8, 8, 12       # → Mt=4, Nt=256, Kt=64
Mt, Nt    = M // Pm, N // Pn
col_num, row_num = 4, 4

SILU_CC   = os.path.join(os.path.dirname(__file__), "../../cc/bf16_vla/silu_256_bf16.cc")
NP_DTYPE  = np_bfloat16

assert M  % Pm == 0
assert N  % Pn == 0
assert K  % Pk == 0
assert Pn % col_num == 0
assert Pm % row_num == 0
assert Nt == 256 and Mt == 4, "tile size must match silu_256_bf16.cc"


# ── Fused GEMM + SiLU region ──────────────────────────────────────────────────
def build_gemm_silu():
    """
    Single df.region that computes C = SiLU(A @ B) in bf16.

    The f32 accumulator is converted to bf16 on-chip (in the pk==Pk-1 core)
    before applying SiLU, so no intermediate buffer leaves the AIE core.
    """
    TyI = Ty_bf16
    TyO = float32

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    silu = ExternalModule(
        top="silu_256_bf16",
        impl_path=SILU_CC,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyI[M, N]):
        pipe: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm_silu_core(
            local_A: TyI[M, K] @ LyA,
            local_B: TyI[K, N] @ LyB,
            local_C: TyI[M, N] @ LyC,
        ):
            pk, pm, pn = df.get_pid()

            # K-chain accumulation (same as standalone GEMM)
            C_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0

            matmul_result: TyI[Mt, Nt] = allo.matmul(local_A, local_B)
            matmul_f32: TyO[Mt, Nt]
            matmul_f32[:, :] = matmul_result[:, :]
            C_out: TyO[Mt, Nt] = allo.add(matmul_f32, C_in)

            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                # On-chip: f32 → bf16, then SiLU, then write output
                C_bf16: TyI[Mt, Nt]
                C_bf16[:, :] = C_out[:, :]  # implicit f32→bf16 cast
                C_act: TyI[Mt, Nt]
                silu(C_bf16, C_act)
                local_C[:, :] = C_act

    primitives = gen_gemm_mapping_primitive(Pm, Pn, Pk, col_num, row_num)
    return top, primitives


# ── Standalone GEMM region (bf16 output, for unfused baseline) ────────────────
def build_gemm_bf16_out():
    """
    GEMM that casts the f32 accumulator to bf16 before writing output.
    Used as the unfused baseline so that both paths produce bf16.
    """
    TyI = Ty_bf16
    TyO = float32

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyI[M, N]):
        pipe: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm_core(
            local_A: TyI[M, K] @ LyA,
            local_B: TyI[K, N] @ LyB,
            local_C: TyI[M, N] @ LyC,
        ):
            pk, pm, pn = df.get_pid()

            C_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0

            matmul_result: TyI[Mt, Nt] = allo.matmul(local_A, local_B)
            matmul_f32: TyO[Mt, Nt]
            matmul_f32[:, :] = matmul_result[:, :]
            C_out: TyO[Mt, Nt] = allo.add(matmul_f32, C_in)

            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C_bf16: TyI[Mt, Nt]
                C_bf16[:, :] = C_out[:, :]  # f32→bf16
                local_C[:, :] = C_bf16

    primitives = gen_gemm_mapping_primitive(Pm, Pn, Pk, col_num, row_num)
    return top, primitives


# ── Standalone SiLU region (for unfused baseline) ────────────────────────────
def build_silu():
    TyI = Ty_bf16
    Ly  = [S(0), S(1)]

    silu = ExternalModule(
        top="silu_256_bf16",
        impl_path=SILU_CC,
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top(X: TyI[M, N], Y: TyI[M, N]):
        @df.kernel(mapping=[Pm, Pn], args=[X, Y])
        def silu_core(
            local_X: TyI[M, N] @ Ly,
            local_Y: TyI[M, N] @ Ly,
        ):
            silu(local_X, local_Y)

    return top


# ── PyTorch reference ─────────────────────────────────────────────────────────
def pytorch_ref(A_np, B_np):
    A = torch.from_numpy(A_np.astype(np.float32)).to(torch.bfloat16)
    B = torch.from_numpy(B_np.astype(np.float32)).to(torch.bfloat16)
    C = torch.nn.functional.silu(A.float() @ B.float()).to(torch.bfloat16)
    return C.float().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not is_available():
        print("MLIR_AIE_INSTALL_DIR not set — skipping NPU test.")
        return

    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(NP_DTYPE)
    B_np = np.random.randn(K, N).astype(NP_DTYPE)
    ref  = pytorch_ref(A_np, B_np)

    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

    # ── Build fused module ────────────────────────────────────────────────────
    print("Building fused GEMM+SiLU...")
    fused_top, fused_primitives = build_gemm_silu()
    fused_mod = df.build(
        fused_top,
        project="gemm_silu_fused.prj",
        target="aie",
        mapping_primitives=fused_primitives,
    )

    C_fused = np.zeros((M, N), dtype=NP_DTYPE)
    fused_mod(A_np, B_np, C_fused)
    np.testing.assert_allclose(
        C_fused.astype(np.float32), ref, rtol=1e-1, atol=1e-1
    )
    print("  Correctness: PASS")

    # ── Build unfused modules ─────────────────────────────────────────────────
    print("Building unfused GEMM (bf16-out)...")
    gemm_top, gemm_primitives = build_gemm_bf16_out()
    gemm_mod = df.build(
        gemm_top,
        project="gemm_bf16_out.prj",
        target="aie",
        mapping_primitives=gemm_primitives,
    )

    print("Building standalone SiLU...")
    silu_top = build_silu()
    silu_mod = df.build(
        silu_top,
        project="silu_standalone.prj",
        target="aie",
    )

    C_gemm  = np.zeros((M, N), dtype=NP_DTYPE)
    C_unfused = np.zeros((M, N), dtype=NP_DTYPE)
    gemm_mod(A_np, B_np, C_gemm)
    silu_mod(C_gemm, C_unfused)
    np.testing.assert_allclose(
        C_unfused.astype(np.float32), ref, rtol=1e-1, atol=1e-1
    )
    print("  Correctness: PASS")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    WARMUP = 20
    ITERS  = 100

    print(f"\nBenchmark: {WARMUP} warmup + {ITERS} timed iterations")

    # Fused
    for _ in range(WARMUP):
        fused_mod(A_np, B_np, C_fused)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused_mod(A_np, B_np, C_fused)
    t_fused = (time.perf_counter() - t0) / ITERS * 1e3  # ms

    # Unfused
    for _ in range(WARMUP):
        gemm_mod(A_np, B_np, C_gemm)
        silu_mod(C_gemm, C_unfused)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        gemm_mod(A_np, B_np, C_gemm)
        silu_mod(C_gemm, C_unfused)
    t_unfused = (time.perf_counter() - t0) / ITERS * 1e3  # ms

    print(f"\n  Unfused  (GEMM + SiLU, 2 dispatches): {t_unfused:.2f} ms/iter")
    print(f"  Fused    (GEMM+SiLU,   1 dispatch):   {t_fused:.2f} ms/iter")
    print(f"  Savings: {t_unfused - t_fused:.2f} ms  ({(t_unfused-t_fused)/t_unfused*100:.1f}%)")

    del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]


if __name__ == "__main__":
    main()
