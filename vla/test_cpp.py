"""
test_cpp.py — Modular correctness + timing tests for OPT-D C++ executables.

Run from /vla/:  python3 test_cpp.py [--stage all|vision|connector|text|action]

Each stage:
  1. Runs the Python NPU version (baseline)
  2. Runs the C++ NPU version (under test)
  3. Compares outputs
  4. Reports wall-clock time for both

Prerequisites:
  - All C++ executables built (see build instructions in vla_cpp.py docstring)
  - All .prj/build/final.xclbin files present
"""

import argparse
import time

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

import vla_cpp as cpp

# ============================================================
# Shared tolerance (matches the Python standalone tests)
# ============================================================
ATOL = 1.0
RTOL = 0.1


def _rng():
    return np.random.default_rng(0)


def _rand(rng, *shape):
    return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np_bfloat16)


def _ones(n):
    return np.ones((n,), dtype=np_bfloat16)


def _zeros(n):
    return np.zeros((n,), dtype=np_bfloat16)


def _timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed * 1e3:.1f} ms")
    return result, elapsed


def _check(name, py_out, cpp_out, atol=ATOL, rtol=RTOL):
    py_f  = py_out.astype(np.float32)
    cpp_f = cpp_out.astype(np.float32)
    max_err  = np.max(np.abs(py_f - cpp_f))
    mean_err = np.mean(np.abs(py_f - cpp_f))
    ok = np.allclose(py_f, cpp_f, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.4f}  mean_err={mean_err:.6f}")
    return ok


# ============================================================
# Stage 1 — Vision block
# ============================================================

def test_vision():
    print("\n=== Vision block (vision_block_bf16 vs C++ vision_encoder) ===")
    from vision_block_bf16 import vision_block as py_vision_block

    rng = _rng()
    x   = _rand(rng, 1024, 768)
    p   = dict(
        Wq=_rand(rng, 768, 768), Wk=_rand(rng, 768, 768),
        Wv=_rand(rng, 768, 768), Wo=_rand(rng, 768, 768),
        W_up=_rand(rng, 768, 3072), W_down=_rand(rng, 3072, 768),
        W_norm_1=_ones(768), b_norm_1=_zeros(768),
        W_norm_2=_ones(768), b_norm_2=_zeros(768),
    )

    py_out,  t_py  = _timed("Python NPU", py_vision_block, x, p)
    cpp_out, t_cpp = _timed("C++   NPU ", cpp.vision_block, x, p)
    ok = _check("vision_block", py_out, cpp_out)
    print(f"  Speed-up: {t_py / t_cpp:.2f}x  (py={t_py*1e3:.0f}ms  cpp={t_cpp*1e3:.0f}ms)")
    return ok


# ============================================================
# Stage 2 — Connector
# ============================================================

def test_connector():
    print("\n=== Connector (connector_bf16 vs C++ connector) ===")
    from connector_bf16 import connector_block as py_connector_block

    rng = _rng()
    x   = _rand(rng, 1024, 768)
    p   = dict(W=_rand(rng, 12288, 960))

    py_out,  t_py  = _timed("Python NPU", py_connector_block, x, p)
    cpp_out, t_cpp = _timed("C++   NPU ", cpp.connector_block, x, p)
    # connector does pixel-shuffle + matmul; expect same tolerance as Python test (atol=6)
    ok = _check("connector_block", py_out, cpp_out, atol=6.0, rtol=0.1)
    print(f"  Speed-up: {t_py / t_cpp:.2f}x  (py={t_py*1e3:.0f}ms  cpp={t_cpp*1e3:.0f}ms)")
    return ok


# ============================================================
# Stage 3 — Text encoder
# ============================================================

def test_text_encoder():
    print("\n=== Text encoder (text_encoder_bf16 vs C++ text_encoder) ===")
    from text_encoder_bf16 import text_encoder_forward as py_text_fwd

    rng = _rng()
    x   = _rand(rng, 128, 960)
    p   = dict(
        Wq=_rand(rng, 960, 960), Wk=_rand(rng, 960, 320),
        Wv=_rand(rng, 960, 320), Wo=_rand(rng, 960, 960),
        W_gate=_rand(rng, 960, 2560), W_up=_rand(rng, 960, 2560),
        W_down=_rand(rng, 2560, 960),
        W_norm_1=_ones(960), W_norm_2=_ones(960),
    )

    py_res,  t_py  = _timed("Python NPU", py_text_fwd, x, p)
    cpp_res, t_cpp = _timed("C++   NPU ", cpp.text_encoder_forward, x, p)

    py_out, py_k, py_v   = py_res
    cpp_out, cpp_k, cpp_v = cpp_res

    ok  = _check("text_encoder output", py_out, cpp_out)
    ok &= _check("text_encoder key",    py_k,   cpp_k)
    ok &= _check("text_encoder val",    py_v,   cpp_v)
    print(f"  Speed-up: {t_py / t_cpp:.2f}x  (py={t_py*1e3:.0f}ms  cpp={t_cpp*1e3:.0f}ms)")
    return ok


# ============================================================
# Stage 4 — Action expert (self)
# ============================================================

def test_action_self():
    print("\n=== Action expert self-attention (action_expert_bf16 vs C++) ===")
    from action_expert_bf16 import action_expert_self_forward as py_self_fwd

    rng    = _rng()
    action = _rand(rng, 32, 768)
    p      = dict(
        Wq=_rand(rng, 768, 960), Wk=_rand(rng, 768, 320),
        Wv=_rand(rng, 768, 320), Wo=_rand(rng, 960, 768),
        W_gate=_rand(rng, 768, 2048), W_up=_rand(rng, 768, 2048),
        W_down=_rand(rng, 2048, 768),
        W_norm_1=_ones(768), W_norm_2=_ones(768),
    )

    py_out,  t_py  = _timed("Python NPU", py_self_fwd, action, p)
    cpp_out, t_cpp = _timed("C++   NPU ", cpp.action_expert_self_forward, action, p)
    ok = _check("action_self output", py_out, cpp_out)
    print(f"  Speed-up: {t_py / t_cpp:.2f}x  (py={t_py*1e3:.0f}ms  cpp={t_cpp*1e3:.0f}ms)")
    return ok


# ============================================================
# Stage 5 — Action expert (cross)
# ============================================================

def test_action_cross():
    print("\n=== Action expert cross-attention (action_expert_bf16 vs C++) ===")
    from action_expert_bf16 import action_expert_cross_forward as py_cross_fwd

    rng    = _rng()
    action = _rand(rng, 32, 768)
    text_k = _rand(rng, 128, 320)
    text_v = _rand(rng, 128, 320)
    p      = dict(
        Wq=_rand(rng, 768, 960),
        Wk_cross=_rand(rng, 320, 320),
        Wv_cross=_rand(rng, 320, 320),
        Wo=_rand(rng, 960, 768),
        W_gate=_rand(rng, 768, 2048), W_up=_rand(rng, 768, 2048),
        W_down=_rand(rng, 2048, 768),
        W_norm_1=_ones(768), W_norm_2=_ones(768),
    )

    py_out,  t_py  = _timed("Python NPU", py_cross_fwd, action, text_k, text_v, p)
    cpp_out, t_cpp = _timed("C++   NPU ", cpp.action_expert_cross_forward,
                             action, text_k, text_v, p)
    ok = _check("action_cross output", py_out, cpp_out)
    print(f"  Speed-up: {t_py / t_cpp:.2f}x  (py={t_py*1e3:.0f}ms  cpp={t_cpp*1e3:.0f}ms)")
    return ok


# ============================================================
# Main
# ============================================================

STAGES = {
    "vision":    test_vision,
    "connector": test_connector,
    "text":      test_text_encoder,
    "self":      test_action_self,
    "cross":     test_action_cross,
}


def main():
    parser = argparse.ArgumentParser(description="Modular C++ pipeline tests")
    parser.add_argument(
        "--stage",
        default="all",
        choices=list(STAGES.keys()) + ["all", "action"],
        help="Which stage to test (default: all)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        stages = list(STAGES.keys())
    elif args.stage == "action":
        stages = ["self", "cross"]
    else:
        stages = [args.stage]

    results = {}
    for s in stages:
        try:
            results[s] = STAGES[s]()
        except Exception as e:
            print(f"\n[ERROR] Stage '{s}' raised an exception:\n  {e}")
            results[s] = False

    print("\n=== Summary ===")
    all_pass = True
    for s, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {s}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll stages passed.")
    else:
        print("\nSome stages failed — check C++ build and xclbin paths.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
