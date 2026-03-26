# Connector BF16 Precision Analysis (2026-03-20)

## Goal

Improve the precision of `connector_bf16.py` (bf16 GEMM-based connector block) to match or approximate the float32 `connector.py` version. The connector does: pixel shuffle → linear projection (matmul of 64×12288 @ 12288×960).

## Key Findings

### Single GEMM tile is precise
- A single GEMM call (K=64 dot product) passes `rtol=0.01` with max relative diff of 0.0078
- The AIE hardware uses a 48-bit accumulator internally (`accauto`), so per-tile precision is excellent
- The precision problem is NOT in individual GEMM tiles

### Accumulation of 192 tiles is the bottleneck
- The full matmul requires 12288/64 = 192 GEMM tiles along the K dimension
- Each tile outputs bf16 (truncated from 48-bit accumulator)
- Summing 192 bf16 partial results introduces significant error
- Max absolute difference: 6.0 (with float32 Python accumulation) vs 20.0 (bf16 accumulation)

### Float32 outer accumulation helps but isn't enough
| Metric | bf16 accum | float32 accum |
|--------|-----------|---------------|
| Max abs diff | 20.0 | 6.0 |
| Mean abs diff | 3.52 | 3.44 |
| Failing rtol=1.0 | 2.52% | 2.52% |
| Elements improved | — | 44.3% |

The rtol=1.0 failures are concentrated in near-zero elements where even small absolute errors cause huge relative errors.

### Larger K tiles make it worse
| K | Mismatch % | Max abs diff | Why |
|---|-----------|-------------|-----|
| 32 | 3.42% | — | More tiles, each less precise in aggregate |
| **64** | **2.44%** | **6.0** | **Sweet spot** |
| 1024 | 6.65% | 18.0 | 1024 bf16 multiply-adds inside GEMM = more rounding |

The GEMM accumulates internally in bf16 (TyO=bfloat16). Increasing K moves accumulation from float32 (outside) to bf16 (inside GEMM) — a worse tradeoff.

## Attempts to Enable Mixed-Precision GEMM (bf16 in → f32 out)

The `mm.cc` kernel library has `matmul_vectorized_4x8x4_bf16_f32()` which outputs float32. The GEMM module signature supports separate `TyI` and `TyO`. However, the allo framework has multiple blockers:

### Attempt 1: Library GEMM with TyO=float32
```python
GEMM(..., bfloat16, float32)
```
**Result:** `ValueError: Cannot overwrite type mapping T1 = bf16 by type f32`
**Cause:** Line 103 of gemm.py does `allo.add(allo.matmul(local_A, local_B), C_in)`. `allo.matmul` infers output type = bf16 from inputs. `allo.add` then tries to unify bf16 result with float32 `C_in`, causing a type variable conflict.

### Attempt 2: Explicit TyO annotation on matmul result
```python
matmul_result: TyO[Mt, Nt] = allo.matmul(local_A, local_B)
```
**Result:** MLIR compiler crash (SIGABRT) in `FoldEmptyCopy::matchAndRewrite`
**Cause:** Allo generates `memref.copy` from `memref<64x64xbf16>` (matmul output) to `memref<64x64xf32>` (annotated type). The MLIR canonicalizer's `FoldEmptyCopy` pattern assumes source and dest have the same element type and crashes on the type mismatch.

### Attempt 3: `float(tensor)` cast
```python
matmul_result: TyI[Mt, Nt] = allo.matmul(local_A, local_B)
C_out: TyO[Mt, Nt] = allo.add(float(matmul_result), C_in)
```
**Result:** `'arith.extf' op operand #0 must be floating-point-like, but got 'memref<64x64xbf16>'`
**Cause:** `float()` in allo works on scalars, not tensors/memrefs.

### Attempt 4: Implicit cast via array assignment
```python
matmul_result: TyI[Mt, Nt] = allo.matmul(local_A, local_B)
matmul_f32: TyO[Mt, Nt]
matmul_f32[:, :] = matmul_result[:, :]  # hoped for implicit cast
```
**Result:** Same MLIR crash as Attempt 2 (`FoldEmptyCopy` assertion failure)
**Cause:** `build_assign_stmt()` generates `memref.copy` for slice assignments, not element-wise `arith.extf`. Same underlying bug.

### Attempt 5: ExternalModule with mixed types
```python
matmul_bf16_f32 = ExternalModule(
    top="matmul_bf16_f32",
    impl_path="matmul_bf16_f32.cc",
    input_idx=[0, 1], output_idx=[2],
)
```
**Result:** `ValueError: Operand 0 of operation "memref.copy" must be a Value`
**Cause:** The dataflow framework copies data to/from kernel arguments via `memref.copy`, which can't handle mixed types.

### Attempt 6: Element-wise scalar cast in loop
```python
matmul_result: TyI[Mt, Nt] = allo.matmul(local_A, local_B)
C_out: TyO[Mt, Nt]
for i in allo.grid(Mt):
    for j in allo.grid(Nt):
        C_out[i, j] = float(matmul_result[i, j]) + C_in[i, j]
```
**Result:** `RuntimeError: Failed to compile — Data memory allocation failure`
**Cause:** The scalar loop + all the temporary buffers (matmul_result bf16, C_in f32, C_out f32, local_A, local_B, local_C) exceed the AIE tile's ~64KB data memory.

## Root Cause

All paths to bf16→f32 type conversion inside an allo dataflow kernel ultimately generate `memref.copy` between memrefs of different element types. The MLIR canonicalizer in allo's toolchain has a bug in `FoldEmptyCopy` that assumes matching types on both sides of the copy, causing a crash.

The `v2_test_mapping_large_gemm.py` in the allo test suite confirms this: `assert TyI == TyO or TyI is int4` (line 54), and bf16 tests are wrapped in try/except with `"bfloat16 have accuracy issue"` (line 188).

## Current Solution

**bf16/bf16 GEMM + float32 Python accumulation:**
- GEMM with K=64, Pm=1, Pn=15, Pk=1
- Each tile output (bf16) is cast to float32 in Python before accumulation
- Final result cast to bf16 once at the end
- Tolerance: `atol=8, rtol=0.1`

## Future Work

- Report the `FoldEmptyCopy` MLIR bug to the allo team — the hardware kernel (`matmul_vectorized_4x8x4_bf16_f32` in mm.cc) already supports bf16→f32, only the compiler pipeline is broken
- Once fixed, use `GEMM(..., bfloat16, float32)` with the implicit cast approach for true mixed-precision accumulation
- This would preserve the 48-bit accumulator precision through to float32 output, eliminating the per-tile bf16 truncation that causes ~2.5% of elements to fail rtol=1.0
