"""Test script to probe parallel RoPE region compilation."""
import os
import numpy as np
import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie import ExternalModule
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

S = Layout.Shard
R = Layout.Replicate
Ty_rope = float32
ROPE_FUSED_TILE = 32
HEAD_DIM = 64
ROPE_FUSED_IMPL = "../cc/float/rope_fused.cc"

rope_fused_ext = ExternalModule(
    top="rope_fused_float32", impl_path=ROPE_FUSED_IMPL, input_idx=[0, 1], output_idx=[2]
)

# Baseline: existing single-core region (mapping=[1,1]) — expected to pass
@df.region()
def rope_single(x: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM],
                sin_cos: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM],
                out: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ [S(1), S(0)],
             lsc: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ [S(1), S(0)],
             lo: Ty_rope[ROPE_FUSED_TILE, HEAD_DIM] @ [S(1), S(0)]):
        rope_fused_ext(lx, lsc, lo)

# Test 1: mapping=[2, 1] — 2 heads, power of 2
@df.region()
def rope_2head(x: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM],
               sin_cos: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM],
               out: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[2, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[2 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

# Test 2: mapping=[4, 1] — 4 heads, power of 2
@df.region()
def rope_4head(x: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM],
               sin_cos: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM],
               out: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[4, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[4 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

# Test 3: mapping=[5, 1] — 5 heads, non-power-of-2
@df.region()
def rope_5head(x: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM],
               sin_cos: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM],
               out: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[5, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[5 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

print("Building baseline (mapping=[1,1])...")
try:
    mod = df.build(rope_single, target="aie", project="test_rope/single.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Building 2-head (mapping=[2,1])...")
try:
    mod = df.build(rope_2head, target="aie", project="test_rope/two.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Building 4-head (mapping=[4,1])...")
try:
    mod = df.build(rope_4head, target="aie", project="test_rope/four.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Building 5-head (mapping=[5,1])...")
try:
    mod = df.build(rope_5head, target="aie", project="test_rope/five.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: mapping=[8, 1] — 8 heads
@df.region()
def rope_8head(x: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM],
               sin_cos: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM],
               out: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[8, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[8 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

print("Building 8-head (mapping=[8,1])...")
try:
    mod = df.build(rope_8head, target="aie", project="test_rope/eight.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: mapping=[15, 1] — Q_H heads
@df.region()
def rope_15head(x: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM],
                sin_cos: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM],
                out: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[15, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[15 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

print("Building 15-head (mapping=[15,1])...")
try:
    mod = df.build(rope_15head, target="aie", project="test_rope/fifteen.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

@df.region()
def rope_9head(x: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM],
               sin_cos: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM],
               out: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[9, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[9 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

@df.region()
def rope_10head(x: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM],
                sin_cos: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM],
                out: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[10, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[10 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

@df.region()
def rope_12head(x: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM],
                sin_cos: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM],
                out: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[12, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[12 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

print("Building 9-head (mapping=[9,1])...")
try:
    mod = df.build(rope_9head, target="aie", project="test_rope/nine.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Building 10-head (mapping=[10,1])...")
try:
    mod = df.build(rope_10head, target="aie", project="test_rope/ten.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("Building 12-head (mapping=[12,1])...")
try:
    mod = df.build(rope_12head, target="aie", project="test_rope/twelve.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

@df.region()
def rope_11head(x: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM],
                sin_cos: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM],
                out: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM]):
    @df.kernel(mapping=[11, 1], args=[x, sin_cos, out])
    def core(lx: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lsc: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)],
             lo: Ty_rope[11 * ROPE_FUSED_TILE, HEAD_DIM] @ [S(0), S(1)]):
        rope_fused_ext(lx, lsc, lo)

print("Building 11-head (mapping=[11,1])...")
try:
    mod = df.build(rope_11head, target="aie", project="test_rope/eleven.prj")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
