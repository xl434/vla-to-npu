
## Update: Division-Free Polynomial Attempts (2026-06-02 continued)

### Attempt 1: Piecewise Polynomial (3 regions)
- **Approach**: Use degree-13 poly for |x|<2, degree-9 for 2≤|x|<4.5, saturation else
- **Simulation Result**: Max GELU error = 0.01244 ✓ (meets < 0.03 target)
- **NPU Result**: Status 8 crash (kernel hung)
- **Root Cause**: Likely aie::select overhead or register pressure from region branching
- **Conclusion**: Mathematically sound but crashes on actual hardware

### Attempt 2: Single Polynomial (Degree-13)
- **Approach**: Single degree-13 polynomial across full range with saturation at ±4.5
- **Simulation Result**: Max GELU error = ???
- **NPU Result**: Kernel runs (~385µs) but output completely wrong (3360 instead of -0.0015)
- **Root Cause**: Horner's method implementation error in initial version
- **Correction Applied**: Fixed Horner accumulation order
- **After Correction**: Still wrong (same error persists)
- **Conclusion**: Either polynomial coefficients unsuitable for this range, or deeper hardware incompatibility with polynomial-based tanh

### Key Finding
Division-free approaches face fundamental challenges on AMD Phoenix AIE2:
1. **Piecewise logic** → register pressure → kernel crash
2. **High-degree polynomials** → instability or overflow in bfloat16 at extremes
3. **Single polynomial** → insufficient accuracy without piecewise

### Recommended Path Forward
Return to Padé rational approximation (known mathematically correct), but fix the division crash:
- Original crash: `aie::div(num, den)` where den comes from aie::add chain
- Solution: Use `aie::mul(num, aie::inv(den))` instead of aie::div
- Alternative: Compute intermediate results to avoid SRS-register sourcing
- Status: Not yet tested due to time constraints

### User Requirements vs. Findings
- User requested: "remove div? still want to make precision high"
- Mathematics shows: Division-free tanh requires either (a) lookup table, (b) piecewise polynomial, or (c) extreme polynomial degree
- Hardware reality: All approaches hit limitations on this architecture
- Realistic Solution: Hybrid approach with safe division + highest precision Padé form

## Attempt 3: Safe Padé Division via aie::inv (SUCCESSFUL)
- **Approach**: Use original Padé formula but compute `y = num * aie::inv(den)` instead of `aie::div(num, den)`
- **NPU Result**: ✓ Kernel runs without crash (~418µs)
- **Accuracy**: Mismatch rate 0.4781%, max error = 0.01562 (target < 0.03)
- **Root Cause of Original Crash**: The aie::div instruction on SRS-sourced registers (from aie::add) causes firmware hang
- **Why This Works**: aie::inv operates on the computed denominator value, then aie::mul combines with numerator safely
- **Conclusion**: Original Padé approximation with safe division method is the correct solution

### Summary of All Attempts
| Approach | Simulation | NPU Result | Max Error | Status |
|----------|-----------|-----------|-----------|--------|
| Original Padé (aie::div) | Good | Crash (status 8) | N/A | ✗ |
| Piecewise Polynomial | 0.01244 | Crash (status 8) | N/A | ✗ |
| Single Polynomial | ? | Runs | 3360 | ✗ |
| Padé (aie::inv) | Good | ✓ Works | 0.01562 | ✓ |

### Final Implementation
Both `/home/xl434/vla-to-npu/cc/bf16/gelu_bf16.cc` and `/home/xl434/vla-to-npu/cc/bf16/gelu_bf16_8rows.cc` now use:
```c
vec_t den_inv = aie::inv(den);
vec_t y = aie::mul(num, den_inv);
```

This provides:
- Full Padé precision (mathematically optimal rational approximation)
- No kernel crashes
- Max GELU error = 0.01562 < target 0.03
- Execution time ~418µs per 49152 elements

### Recommendation
Deploy current kernels. The division-free approach was mathematically interesting but hit architectural limits. The safe-division Padé formula is production-ready.

---

## Session 2: XRT Fix + GELU Kernel Status (2026-06-03)

### XRT 2.21.75 Fix Applied
- **Issue**: Test binaries were linking against old XRT 2.18.0 (`/opt/xilinx/xrt`)
- **Solution**: Applied PR #586 fix to `allo2/allo/CMakeLists.txt` to use system XRT 2.21.75
  - Changed `XRT_INC_DIR` from `/opt/xilinx/xrt/include` to `/usr/include`
  - Changed `XRT_LIB_DIR` from `/opt/xilinx/xrt/lib` to `/usr/lib/x86_64-linux-gnu`
- **Note**: `/opt/allo` (used by conda env) already had the correct paths

### GELU Kernel Status

#### Working Solution: Padé Rational + Safe Division
- **Kernel**: `cc/bf16/gelu_bf16.cc` (4-row variant, [4][768] per core)
- **Algorithm**: Padé rational approximation for tanh(x)
  - `tanh(x) ≈ x·(135135 + 17325·x² + 378·x⁴) / (135135 + 62370·x² + 3150·x⁴ + 28·x⁶)`
  - Safe division: `aie::inv(den); aie::mul(num, inv)` (avoids VRECIP crash)
  - Saturation at ±1 for |x| > 4

#### Precision (Verified with XRT 2.21.75)
- **Max absolute error**: 0.01563 (< 0.03 target ✅)
- **Mismatch rate**: 0.4781% (235/49152 elements)
- **Status**: Production ready

#### Performance Target (from 5-26 plan)
- **Current scalar LUT**: 474ms per layer (77% of ViT 12L time), ~14.8ms per dispatch, 32 dispatches
- **With Padé vectorized**: Estimate ~60ms per layer (~8× improvement), ~1ms per dispatch
  - Using [4,4] mapping → 64 dispatches (2× more than current)
  - But ~16× faster per dispatch (vectorized MAC vs scalar division)
  - Net: 2× more dispatches, 16× faster each = 8× total speedup
- **Build script**: `vla/vision_block/gelu_bf16_ffn.py` (P1=4, P0=4, seq=16, feature_dim=3072)
- **XClbin**: `vla/vision_block/gelu_bf16_ffn.prj/build/final.xclbin` ✅ Built

#### Attempted but Failed: 8-Row Polynomial Variants
- **Problem 1**: Polynomial (degree-13) outside |x| ≤ 2.5 produces NaN/Inf
- **Problem 2**: Comparisons (`aie::gt`, `aie::lt`) on SRS-sourced (chain-computed) registers give wrong results
- **Problem 3**: Saturation via `aie::select` with MAC-sourced first arg + load-sourced condition fails
- **Attempted fixes**:
  - Inner-clamping: `inner_safe = inner * w` where w = 1 outside range → massive errors
  - Scalar-extract reload: Force inner through scalar loop → still broken comparisons
  - Shift trick: `inner + 1000` to make comparisons positive → no effect
- **Root cause**: AIE2 vector float comparison appears fundamentally broken for SRS-sourced or chain-computed operands
- **Verdict**: Polynomial approach is a dead-end on this hardware

### Next Steps (from 5-26 Kernel Fusion Plan)
1. **Integrate Padé GELU** into vision_encoder.cpp C++ binary
   - Update `spec_gelu.preload()` to use new xclbin `gelu_bf16_ffn`
   - Update `GELU_SEQ_TILE` constant if needed
2. **Measure actual speedup** with `-v 2 -n 3` profiling
3. **Scale to 12-layer ViT** and verify timing improvements
4. **Phase 2b**: Causal masking for text encoder FA (deferred, requires kernel modification)
5. **Phase 3**: FFN fusion (after GELU integrated and measured)


---

## GELU Integration Complete (2026-06-03, Session 2 - Part 2)

### Integration Summary
- **Kernel**: Padé rational approximation (7/7 degree, safe division)
- **XClbin**: `vla/vision_block/gelu_bf16_ffn.prj/build/final.xclbin` ✅
- **C++ updates**: `vla/vision_block/unified.prj/vision_encoder.cpp`
  - Updated `GELU_SEQ_TILE: 32 → 16` (vectorized kernel uses 16-row tiles)
  - `SZ_FFN_TILE` auto-recalculated: 196KB → 98KB

### Measured Performance (Single-Layer Benchmark)
| Metric | Value |
|--------|-------|
| **GELU time** | 30.5 ms (avg of 3 runs) |
| **vs original** | 474 ms |
| **Speedup** | **15.5×** |
| **as % of layer** | ~5% (down from 77%) |

### Key Achievement
Vectorized Padé tanh eliminated 444ms per layer.
- **Per-ViT-layer savings**: 444ms
- **Extrapolated to 12-layer ViT**: 5.3s (from original ~6.5s for ViT alone)

### Next Steps (5-26 Kernel Fusion Plan)
1. ✅ Phase 0: XRT 2.21.75 fix (PR #586)
2. ✅ Phase 1: GELU vectorization (just completed)
3. Pending: Phase 2b: Causal FA for text encoder
4. Pending: Phase 3: FFN fusion (if needed for 5s target)
5. Pending: Phase 4: Scale to 12-layer ViT and full model timing

