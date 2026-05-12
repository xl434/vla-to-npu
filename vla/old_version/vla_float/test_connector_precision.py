"""
Diagnostic test to pinpoint precision loss in connector_bf16.
Tests:
  1. Single GEMM tile precision (K=64 dot product)
  2. Full matmul with bf16 accumulation (original approach)
  3. Full matmul with float32 accumulation (improved approach)
"""
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as ml_bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from allo.memory import Layout

np.random.seed(0)

GEMM_K = 64
GEMM_M = 64
GEMM_N = 960
NEW_EMBD = 12288

Pm = 1
Pn = GEMM_N // 64
Pk = 1

gemm_top, gemm_mapping = GEMM(
    GEMM_M, GEMM_N, GEMM_K,
    Pm, Pn, Pk,
    bfloat16, bfloat16,
)

gemm_mod = df.build(
    gemm_top,
    target="aie",
    project="test_gemm.prj",
    mapping_primitives=gemm_mapping,
)

# Generate test data (full size)
A = np.random.randn(GEMM_M, NEW_EMBD).astype(ml_bfloat16)
B = np.random.randn(NEW_EMBD, GEMM_N).astype(ml_bfloat16)

# =============================================
# Test 1: Single GEMM tile (K=64)
# =============================================
A_tile = A[:, :GEMM_K]
B_tile = B[:GEMM_K, :]
C_single = np.zeros((GEMM_M, GEMM_N), dtype=ml_bfloat16)
gemm_mod(A_tile, B_tile, C_single)

expected_single = (A_tile.astype(np.float32) @ B_tile.astype(np.float32)).astype(ml_bfloat16)

# =============================================
# Test 2: Full matmul with bf16 accumulation
# =============================================
C_bf16 = np.zeros((GEMM_M, GEMM_N), dtype=ml_bfloat16)
C_tmp = np.zeros((GEMM_M, GEMM_N), dtype=ml_bfloat16)
for i in range(0, NEW_EMBD, GEMM_K):
    A_tile = A[:, i:i+GEMM_K]
    B_tile = B[i:i+GEMM_K, :]
    C_tmp[:] = 0
    gemm_mod(A_tile, B_tile, C_tmp)
    C_bf16 = (C_bf16.astype(np.float32) + C_tmp.astype(np.float32)).astype(ml_bfloat16)

expected_full = (A.astype(np.float32) @ B.astype(np.float32)).astype(ml_bfloat16)

# =============================================
# Test 3: Full matmul with float32 accumulation
# =============================================
C_fp32 = np.zeros((GEMM_M, GEMM_N), dtype=np.float32)
C_tmp = np.zeros((GEMM_M, GEMM_N), dtype=ml_bfloat16)
for i in range(0, NEW_EMBD, GEMM_K):
    A_tile = A[:, i:i+GEMM_K]
    B_tile = B[i:i+GEMM_K, :]
    C_tmp[:] = 0
    gemm_mod(A_tile, B_tile, C_tmp)
    C_fp32 += C_tmp.astype(np.float32)

C_fp32_as_bf16 = C_fp32.astype(ml_bfloat16)

# =============================================
# Print all results at the end
# =============================================
print()
print("=" * 60)
print("Test 1: Single GEMM tile (first K=64 slice)")
print("=" * 60)
diff1 = np.abs(C_single.astype(np.float32) - expected_single.astype(np.float32))
print(f"Max absolute difference: {diff1.max():.6f}")
print(f"Mean absolute difference: {diff1.mean():.6f}")
rel_mask1 = np.abs(expected_single.astype(np.float32)) > 1e-6
if rel_mask1.any():
    rel_diff1 = diff1[rel_mask1] / np.abs(expected_single.astype(np.float32))[rel_mask1]
    print(f"Max relative difference: {rel_diff1.max():.6f}")
    print(f"Mean relative difference: {rel_diff1.mean():.6f}")
try:
    np.testing.assert_allclose(C_single.astype(np.float32), expected_single.astype(np.float32), rtol=0.01)
    print("PASS: rtol=0.01")
except AssertionError:
    print("FAIL: rtol=0.01")
    try:
        np.testing.assert_allclose(C_single.astype(np.float32), expected_single.astype(np.float32), rtol=0.1)
        print("PASS: rtol=0.1")
    except AssertionError:
        print("FAIL: rtol=0.1")

print()
print("=" * 60)
print("Test 2: Full matmul with bf16 accumulation (192 tiles)")
print("=" * 60)
diff2 = np.abs(C_bf16.astype(np.float32) - expected_full.astype(np.float32))
print(f"Max absolute difference: {diff2.max():.6f}")
print(f"Mean absolute difference: {diff2.mean():.6f}")
rel_mask2 = np.abs(expected_full.astype(np.float32)) > 1e-6
if rel_mask2.any():
    rel_diff2 = diff2[rel_mask2] / np.abs(expected_full.astype(np.float32))[rel_mask2]
    print(f"Max relative difference: {rel_diff2.max():.6f}")
    print(f"Mean relative difference: {rel_diff2.mean():.6f}")
    num_fail_rtol1 = np.sum(rel_diff2 > 1.0)
    num_fail_rtol01 = np.sum(rel_diff2 > 0.1)
    num_fail_rtol001 = np.sum(rel_diff2 > 0.01)
    print(f"Elements failing rtol=1.0:  {num_fail_rtol1}/{rel_mask2.sum()} ({100*num_fail_rtol1/rel_mask2.sum():.2f}%)")
    print(f"Elements failing rtol=0.1:  {num_fail_rtol01}/{rel_mask2.sum()} ({100*num_fail_rtol01/rel_mask2.sum():.2f}%)")
    print(f"Elements failing rtol=0.01: {num_fail_rtol001}/{rel_mask2.sum()} ({100*num_fail_rtol001/rel_mask2.sum():.2f}%)")

print()
print("=" * 60)
print("Test 3: Full matmul with float32 accumulation (192 tiles)")
print("=" * 60)
diff3 = np.abs(C_fp32_as_bf16.astype(np.float32) - expected_full.astype(np.float32))
print(f"Max absolute difference: {diff3.max():.6f}")
print(f"Mean absolute difference: {diff3.mean():.6f}")
rel_mask3 = np.abs(expected_full.astype(np.float32)) > 1e-6
if rel_mask3.any():
    rel_diff3 = diff3[rel_mask3] / np.abs(expected_full.astype(np.float32))[rel_mask3]
    print(f"Max relative difference: {rel_diff3.max():.6f}")
    print(f"Mean relative difference: {rel_diff3.mean():.6f}")
    num_fail_rtol1 = np.sum(rel_diff3 > 1.0)
    num_fail_rtol01 = np.sum(rel_diff3 > 0.1)
    num_fail_rtol001 = np.sum(rel_diff3 > 0.01)
    print(f"Elements failing rtol=1.0:  {num_fail_rtol1}/{rel_mask3.sum()} ({100*num_fail_rtol1/rel_mask3.sum():.2f}%)")
    print(f"Elements failing rtol=0.1:  {num_fail_rtol01}/{rel_mask3.sum()} ({100*num_fail_rtol01/rel_mask3.sum():.2f}%)")
    print(f"Elements failing rtol=0.01: {num_fail_rtol001}/{rel_mask3.sum()} ({100*num_fail_rtol001/rel_mask3.sum():.2f}%)")

print()
print("=" * 60)
print("Test 4: float32 accum vs bf16 accum (how much does it help?)")
print("=" * 60)
print(f"bf16 accum - max abs diff:  {diff2.max():.6f}")
print(f"fp32 accum - max abs diff:  {diff3.max():.6f}")
print(f"bf16 accum - mean abs diff: {diff2.mean():.6f}")
print(f"fp32 accum - mean abs diff: {diff3.mean():.6f}")
improved = diff3 < diff2
print(f"Elements improved by fp32 accum: {improved.sum()}/{diff2.size} ({100*improved.sum()/diff2.size:.1f}%)")
