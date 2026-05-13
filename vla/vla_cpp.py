"""
vla_cpp.py — OPT-D subprocess wrappers for the C++ NPU pipeline executables.

Provides drop-in replacements for the Python forward functions in:
  vision_block_bf16.py, connector_bf16.py,
  text_encoder_bf16.py, action_expert_bf16.py
  preprocessing_fused_bf16.py  (via fused_unified)

Data is exchanged via temporary binary files (raw bf16 uint16 or float32).

Build the C++ executables first (from /vla/ directory):
  cd preprocessing/fused_unified && mkdir -p build && cd build && cmake .. && make -j4
  cd vision_block/unified.prj   && mkdir -p build && cd build && cmake .. && make -j4
  cd text_encoder_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4
  cd action_expert_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4
  cd connector/unified.prj      && mkdir -p build && cd build && cmake .. && make -j4

Test each stage with test_cpp.py before running the full pipeline.
"""

import os
import subprocess
import tempfile
import time

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

_VLA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------

def _write_bf16(path: str, arr: np.ndarray) -> None:
    arr.astype(np_bfloat16).view(np.uint16).tofile(path)


def _read_bf16(path: str, shape) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return raw.view(np_bfloat16).reshape(shape)


def _read_f32(path: str, shape) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def _run(cmd: list, cwd: str, label: str = "") -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] C++ executable failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}\n"
            f"  stdout: {result.stdout.strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Preprocessing  (fused im2col + GEMM, 2 hw_context opens instead of 320)
# ---------------------------------------------------------------------------

def preprocessing_block(image: np.ndarray, params: dict) -> np.ndarray:
    """Fused im2col + GEMM patch embedding via C++ (drop-in for preprocessing_fused_bf16.preprocessing_block).

    image:  [3, 512, 512] bf16
    params: kernel [768, 3, 16, 16] bf16
    returns [1024, 768] bf16

    Uses preprocessing/fused_unified/build/test — opens 2 XRT hw_contexts total
    (128 im2col dispatches + 192 GEMM dispatches) vs 320 subprocess spawns in Python.
    """
    exe = os.path.join(_VLA_DIR, "preprocessing/fused_unified/build/test")
    cwd = os.path.join(_VLA_DIR, "preprocessing/fused_unified")

    xclbin_im2col = os.path.join(_VLA_DIR, "preprocessing/im2col.prj/build/final.xclbin")
    instr_im2col  = os.path.join(_VLA_DIR, "preprocessing/im2col.prj/insts.txt")
    xclbin_gemm   = os.path.join(_VLA_DIR, "preprocessing/fused_conv_add.prj/build/final.xclbin")
    instr_gemm    = os.path.join(_VLA_DIR, "preprocessing/fused_conv_add.prj/insts.txt")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/image.data",  image)
        _write_bf16(f"{d}/kernel.data", params["kernel"])
        out = f"{d}/out.data"

        _run([
            exe,
            "--xclbin_im2col",   xclbin_im2col,
            "--instr_im2col",    instr_im2col,
            "--xclbin_gemm",     xclbin_gemm,
            "--instr_gemm",      instr_gemm,
            "--input_image",     f"{d}/image.data",
            "--input_kernel_raw", f"{d}/kernel.data",
            "--output_file",     out,
            "--verify",          "0",
        ], cwd, "preprocessing")

        return _read_bf16(out, (1024, 768))


# ---------------------------------------------------------------------------
# Vision block
# ---------------------------------------------------------------------------

def vision_block(x: np.ndarray, params: dict) -> np.ndarray:
    """Single vision transformer block via C++ (drop-in for vision_block_bf16.vision_block).

    x:      [1024, 768] bf16
    params: W_norm_1 [768], b_norm_1 [768], Wq/Wk/Wv/Wo [768,768],
            W_up [768,3072], W_norm_2 [768], b_norm_2 [768], W_down [3072,768]  — all bf16
    returns [1024, 768] bf16
    """
    exe = os.path.join(_VLA_DIR, "vision_block/unified.prj/build/vision_encoder")
    cwd = os.path.join(_VLA_DIR, "vision_block/unified.prj")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/x.data",     x)
        _write_bf16(f"{d}/w1.data",    params["W_norm_1"])
        _write_bf16(f"{d}/b1.data",    params["b_norm_1"])
        _write_bf16(f"{d}/wq.data",    params["Wq"])
        _write_bf16(f"{d}/wk.data",    params["Wk"])
        _write_bf16(f"{d}/wv.data",    params["Wv"])
        _write_bf16(f"{d}/wo.data",    params["Wo"])
        _write_bf16(f"{d}/wup.data",   params["W_up"])
        _write_bf16(f"{d}/w2.data",    params["W_norm_2"])
        _write_bf16(f"{d}/b2.data",    params["b_norm_2"])
        _write_bf16(f"{d}/wdown.data", params["W_down"])
        out = f"{d}/out.data"

        _run([
            exe,
            "--input",    f"{d}/x.data",
            "--W_norm_1", f"{d}/w1.data",
            "--b_norm_1", f"{d}/b1.data",
            "--Wq",       f"{d}/wq.data",
            "--Wk",       f"{d}/wk.data",
            "--Wv",       f"{d}/wv.data",
            "--Wo",       f"{d}/wo.data",
            "--W_up",     f"{d}/wup.data",
            "--W_norm_2", f"{d}/w2.data",
            "--b_norm_2", f"{d}/b2.data",
            "--W_down",   f"{d}/wdown.data",
            "--output",   out,
        ], cwd, "vision_block")

        return _read_bf16(out, (1024, 768))


def vision_encoder(num_layers: int, x: np.ndarray, params: dict) -> np.ndarray:
    """Run num_layers vision blocks via C++ (drop-in for vla.vision_encoder)."""
    for _ in range(num_layers):
        x = vision_block(x, params)
    return x


# ---------------------------------------------------------------------------
# Connector  (pixel shuffle + GEMM)
# ---------------------------------------------------------------------------

def connector_block(x: np.ndarray, params: dict) -> np.ndarray:
    """Pixel-shuffle + tiled-GEMM connector via C++ (drop-in for connector_bf16.connector_block).

    x:      [1024, 768] bf16
    params: W [12288, 960] bf16
    returns [64, 960] bf16

    C++ accumulates GEMM tiles in float32 internally; result is cast to bf16 here
    to match the Python API.
    """
    exe = os.path.join(_VLA_DIR, "connector/unified.prj/build/test")
    cwd = os.path.join(_VLA_DIR, "connector/unified.prj")

    xclbin_ps  = os.path.join(_VLA_DIR, "connector/pixel_shuffle.prj/build/final.xclbin")
    instr_ps   = os.path.join(_VLA_DIR, "connector/pixel_shuffle.prj/insts.txt")
    xclbin_g   = os.path.join(_VLA_DIR, "connector/gemm.prj/build/final.xclbin")
    instr_g    = os.path.join(_VLA_DIR, "connector/gemm.prj/insts.txt")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/A.data", x)
        _write_bf16(f"{d}/W.data", params["W"])
        out = f"{d}/out.data"

        _run([
            exe,
            "--xclbin_pixel_shuffle", xclbin_ps,
            "--instr_pixel_shuffle",  instr_ps,
            "--xclbin_gemm",          xclbin_g,
            "--instr_gemm",           instr_g,
            "--input_A",              f"{d}/A.data",
            "--input_W",              f"{d}/W.data",
            "--output_file",          out,
            "--verify",               "0",
        ], cwd, "connector")

        out_f32 = _read_f32(out, (64, 960))
        return out_f32.astype(np_bfloat16)


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

def text_encoder_forward(x: np.ndarray, params: dict):
    """Text encoder forward pass via C++ (drop-in for text_encoder_bf16.text_encoder_forward).

    x:      [128, 960] bf16
    params: W_norm_1 [960], W_norm_2 [960], Wq [960,960], Wk [960,320], Wv [960,320],
            Wo [960,960], W_gate [960,2560], W_up [960,2560], W_down [2560,960]  — all bf16
    returns (output [128,960], text_k [128,320], text_v [128,320])  — all bf16
    """
    exe = os.path.join(_VLA_DIR, "text_encoder_bf16/unified.prj/build/text_encoder")
    cwd = os.path.join(_VLA_DIR, "text_encoder_bf16/unified.prj")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/x.data",     x)
        _write_bf16(f"{d}/wn1.data",   params["W_norm_1"])
        _write_bf16(f"{d}/wn2.data",   params["W_norm_2"])
        _write_bf16(f"{d}/wq.data",    params["Wq"])
        _write_bf16(f"{d}/wk.data",    params["Wk"])
        _write_bf16(f"{d}/wv.data",    params["Wv"])
        _write_bf16(f"{d}/wo.data",    params["Wo"])
        _write_bf16(f"{d}/wgate.data", params["W_gate"])
        _write_bf16(f"{d}/wup.data",   params["W_up"])
        _write_bf16(f"{d}/wdown.data", params["W_down"])
        out_p = f"{d}/out.data"
        key_p = f"{d}/key.data"
        val_p = f"{d}/val.data"

        _run([
            exe,
            "--input",    f"{d}/x.data",
            "--W_norm_1", f"{d}/wn1.data",
            "--W_norm_2", f"{d}/wn2.data",
            "--Wq",       f"{d}/wq.data",
            "--Wk",       f"{d}/wk.data",
            "--Wv",       f"{d}/wv.data",
            "--Wo",       f"{d}/wo.data",
            "--W_gate",   f"{d}/wgate.data",
            "--W_up",     f"{d}/wup.data",
            "--W_down",   f"{d}/wdown.data",
            "--output",   out_p,
            "--key_out",  key_p,
            "--val_out",  val_p,
        ], cwd, "text_encoder")

        return (
            _read_bf16(out_p, (128, 960)),
            _read_bf16(key_p, (128, 320)),
            _read_bf16(val_p, (128, 320)),
        )


# ---------------------------------------------------------------------------
# Action expert — self-attention
# ---------------------------------------------------------------------------

def action_expert_self_forward(action: np.ndarray, params: dict) -> np.ndarray:
    """Action expert self-attention block via C++
    (drop-in for action_expert_bf16.action_expert_self_forward).

    action: [32, 768] bf16
    params: W_norm_1 [768], W_norm_2 [768], Wq [768,960], Wk [768,320], Wv [768,320],
            Wo [960,768], W_gate [768,2048], W_up [768,2048], W_down [2048,768]  — all bf16
    returns [32, 768] bf16
    """
    exe = os.path.join(_VLA_DIR, "action_expert_bf16/unified.prj/build/action_expert")
    cwd = os.path.join(_VLA_DIR, "action_expert_bf16/unified.prj")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/x.data",     action)
        _write_bf16(f"{d}/wn1.data",   params["W_norm_1"])
        _write_bf16(f"{d}/wn2.data",   params["W_norm_2"])
        _write_bf16(f"{d}/wq.data",    params["Wq"])
        _write_bf16(f"{d}/wk.data",    params["Wk"])
        _write_bf16(f"{d}/wv.data",    params["Wv"])
        _write_bf16(f"{d}/wo.data",    params["Wo"])
        _write_bf16(f"{d}/wgate.data", params["W_gate"])
        _write_bf16(f"{d}/wup.data",   params["W_up"])
        _write_bf16(f"{d}/wdown.data", params["W_down"])
        out = f"{d}/out.data"

        _run([
            exe,
            "--mode",     "self",
            "--input",    f"{d}/x.data",
            "--W_norm_1", f"{d}/wn1.data",
            "--W_norm_2", f"{d}/wn2.data",
            "--Wq",       f"{d}/wq.data",
            "--Wk",       f"{d}/wk.data",
            "--Wv",       f"{d}/wv.data",
            "--Wo",       f"{d}/wo.data",
            "--W_gate",   f"{d}/wgate.data",
            "--W_up",     f"{d}/wup.data",
            "--W_down",   f"{d}/wdown.data",
            "--output",   out,
        ], cwd, "action_expert_self")

        return _read_bf16(out, (32, 768))


# ---------------------------------------------------------------------------
# Action expert — cross-attention
# ---------------------------------------------------------------------------

def action_expert_cross_forward(
    action: np.ndarray,
    text_k: np.ndarray,
    text_v: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Action expert cross-attention block via C++
    (drop-in for action_expert_bf16.action_expert_cross_forward).

    action: [32, 768] bf16
    text_k: [128, 320] bf16  — from text_encoder_forward key output
    text_v: [128, 320] bf16  — from text_encoder_forward val output
    params: W_norm_1 [768], W_norm_2 [768], Wq [768,960],
            Wk_cross [320,320], Wv_cross [320,320], Wo [960,768],
            W_gate [768,2048], W_up [768,2048], W_down [2048,768]  — all bf16
    returns [32, 768] bf16
    """
    exe = os.path.join(_VLA_DIR, "action_expert_bf16/unified.prj/build/action_expert")
    cwd = os.path.join(_VLA_DIR, "action_expert_bf16/unified.prj")

    with tempfile.TemporaryDirectory() as d:
        _write_bf16(f"{d}/x.data",     action)
        _write_bf16(f"{d}/key.data",   text_k)
        _write_bf16(f"{d}/val.data",   text_v)
        _write_bf16(f"{d}/wn1.data",   params["W_norm_1"])
        _write_bf16(f"{d}/wn2.data",   params["W_norm_2"])
        _write_bf16(f"{d}/wq.data",    params["Wq"])
        _write_bf16(f"{d}/wkc.data",   params["Wk_cross"])
        _write_bf16(f"{d}/wvc.data",   params["Wv_cross"])
        _write_bf16(f"{d}/wo.data",    params["Wo"])
        _write_bf16(f"{d}/wgate.data", params["W_gate"])
        _write_bf16(f"{d}/wup.data",   params["W_up"])
        _write_bf16(f"{d}/wdown.data", params["W_down"])
        out = f"{d}/out.data"

        _run([
            exe,
            "--mode",     "cross",
            "--input",    f"{d}/x.data",
            "--text_k",   f"{d}/key.data",
            "--text_v",   f"{d}/val.data",
            "--W_norm_1", f"{d}/wn1.data",
            "--W_norm_2", f"{d}/wn2.data",
            "--Wq",       f"{d}/wq.data",
            "--Wk",       f"{d}/wkc.data",
            "--Wv",       f"{d}/wvc.data",
            "--Wo",       f"{d}/wo.data",
            "--W_gate",   f"{d}/wgate.data",
            "--W_up",     f"{d}/wup.data",
            "--W_down",   f"{d}/wdown.data",
            "--output",   out,
        ], cwd, "action_expert_cross")

        return _read_bf16(out, (32, 768))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    print(f"  {label}: {(t1 - t0) * 1e3:.1f} ms")
    return result
