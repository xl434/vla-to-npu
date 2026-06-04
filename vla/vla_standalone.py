"""
vla_standalone.py — VLA inference with zero build-time overhead.

Calls pre-built Allo ./build/top binaries directly (same file-I/O pattern
that Allo uses internally, but without df.build() recompiling everything).
Uses C++ subprocess wrappers (vla_cpp.py) for the major transformer stages.

No df.build() calls → no xclbin recompilation → ~1s startup instead of 7 minutes.

Run:  python3 vla_standalone.py
      (all .prj/build/ directories must already contain final.xclbin + insts.txt,
       all unified.prj/build/ executables must be compiled)
"""

import os
import subprocess
import sys
import time

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import torch
import torch.nn as nn

import vla_cpp as cpp      # C++ subprocess wrappers (no df.build())

_VLA = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# VLA constants  (copied from vla.py — no imports from NPU modules)
# ============================================================
SEQ_T          = 48
EMBD_S         = 960         # TEXT
SEQ_S          = 1
PADDING        = 15
VIT_NUM_LAYERS = 12
LLAMA_NUM_LAYERS = 12
SKIP           = 2
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM  = 32
CHUNK_SIZE     = 32          # EXP_SEQ

# Image / kernel dimensions (used in main() for random data generation)
_CH   = 3
_PIX  = 512
_KDIM = 16
_EMBD = 768

# ============================================================
# Helpers: call a pre-built Allo top binary (no df.build())
# ============================================================

def _write_bf16(path, arr):
    arr.astype(np_bfloat16).view(np.uint16).tofile(path)

def _read_bf16(path, shape):
    return np.fromfile(path, dtype=np.uint16).view(np_bfloat16).reshape(shape)

def _call_top(prj_dir, inputs: dict, out_slot: int, out_shape):
    """Write inputs, run ./build/top, return output array.

    inputs: {slot_index: np.ndarray}  — written to input{N}.data
    out_slot: slot index for output (read from output{N}.data)
    """
    for idx, arr in inputs.items():
        _write_bf16(f"{prj_dir}/input{idx}.data", arr)

    result = subprocess.run(
        "./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz 0",
        shell=True, cwd=prj_dir, capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"top binary failed in {prj_dir}:\n{result.stderr.decode()}"
        )
    return _read_bf16(f"{prj_dir}/output{out_slot}.data", out_shape)


# ============================================================
# Preprocessing  (fused C++ unified: 2 hw_context opens for 320 dispatches)
# ============================================================

def preprocessing_block(image: np.ndarray, params: dict) -> np.ndarray:
    """image [3,512,512] bf16, params['kernel'] [768,3,16,16] bf16 → [1024,768] bf16"""
    return cpp.preprocessing_block(image, params)


# ============================================================
# State embedding  (1 GEMM call)
# ============================================================

_STATE_GEMM_PRJ = os.path.join(_VLA, "state_emb/gemm.prj")

def create_state_emb(state_input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """state_input [1,32], weights [32,960] → [1,960] bf16"""
    A = np.ascontiguousarray(np.pad(state_input, ((0, 15), (0, 0))))  # [16,32]
    out = _call_top(_STATE_GEMM_PRJ, {0: A, 1: weights}, 2, (16, EMBD_S))
    return out[0:1, :]   # [1, 960]


# ============================================================
# Text embedding helper
# ============================================================

def create_text_emb(vocab_size, hidden_size, seq):
    embed = nn.Embedding(vocab_size, hidden_size)
    lang_tokens = torch.randint(0, vocab_size, (seq,))
    lang_emb = embed(lang_tokens) * np.sqrt(hidden_size)
    return lang_emb.detach().float().numpy().astype(np_bfloat16)


# ============================================================
# Vision encoder  (C++ subprocess)
# ============================================================

def vision_encoder(num_layers, x, params):
    return cpp.vision_encoder(num_layers, x, params)


# ============================================================
# Joint transformer  (C++ subprocess)
# ============================================================

def joint_transformer(num_layers, vlm_input, action, vlm_params,
                      exp_self_params, exp_cross_params):
    # Batch all text encoder layers in one subprocess (saves num_layers-1 XRT startups)
    _, kv_pairs = cpp.text_encoder_layers_forward(num_layers, vlm_input, vlm_params)
    # Batch all action expert layers in one subprocess (saves num_layers-1 XRT startups)
    action = cpp.action_expert_layers_forward(
        num_layers, SKIP, action, kv_pairs, exp_self_params, exp_cross_params
    )
    return action


# ============================================================
# Postprocessing  (2 Allo top calls: rms_norm + GEMM)
# ============================================================

_POST_NORM_PRJ = os.path.join(_VLA, "postprocessing/rms_norm.prj")
_POST_GEMM_PRJ = os.path.join(_VLA, "postprocessing/gemm.prj")

def postprocessing(out: np.ndarray, params: dict) -> np.ndarray:
    """out [32,768] bf16 → v_t [32,32] bf16"""
    normed = _call_top(_POST_NORM_PRJ,
                       {0: out, 1: params["W_exp_norm"]}, 2, (CHUNK_SIZE, 768))
    v_t = _call_top(_POST_GEMM_PRJ,
                    {0: normed, 1: params["W_action_out"]}, 2,
                    (CHUNK_SIZE, MAX_STATE_DIM))
    return v_t


# ============================================================
# Main
# ============================================================

def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    np.random.seed(0)

    def rand_mat(m, n): return (rng.standard_normal((m, n)) / np.sqrt(m)).astype(np_bfloat16)
    def ones_vec(n):    return np.ones((n,), dtype=np_bfloat16)
    def zeros_vec(n):   return np.zeros((n,), dtype=np_bfloat16)

    params_proc = dict(
        kernel=(rng.standard_normal((_EMBD, _CH, _KDIM, _KDIM))
                / np.sqrt(_CH * _KDIM * _KDIM)).astype(np_bfloat16)
    )
    params_con  = dict(W=rand_mat(12288, EMBD_S))
    params_vit  = dict(
        Wq=rand_mat(768, 768), Wk=rand_mat(768, 768), Wv=rand_mat(768, 768), Wo=rand_mat(768, 768),
        W_up=rand_mat(768, 3072), W_down=rand_mat(3072, 768),
        W_norm_1=ones_vec(768), b_norm_1=zeros_vec(768),
        W_norm_2=ones_vec(768), b_norm_2=zeros_vec(768),
    )
    params_vlm  = dict(
        Wq=rand_mat(960, 960), Wk=rand_mat(960, 320), Wv=rand_mat(960, 320), Wo=rand_mat(960, 960),
        W_gate=rand_mat(960, 2560), W_up=rand_mat(960, 2560), W_down=rand_mat(2560, 960),
        W_norm_1=ones_vec(960), W_norm_2=ones_vec(960),
    )
    params_exp_self = dict(
        Wq=rand_mat(768, 960), Wk=rand_mat(768, 320), Wv=rand_mat(768, 320), Wo=rand_mat(960, 768),
        W_gate=rand_mat(768, 2048), W_up=rand_mat(768, 2048), W_down=rand_mat(2048, 768),
        W_norm_1=ones_vec(768), W_norm_2=ones_vec(768),
    )
    params_exp_cross = dict(
        Wq=rand_mat(768, 960), Wk_cross=rand_mat(320, 320), Wv_cross=rand_mat(320, 320),
        Wo=rand_mat(960, 768),
        W_gate=rand_mat(768, 2048), W_up=rand_mat(768, 2048), W_down=rand_mat(2048, 768),
        W_norm_1=ones_vec(768), W_norm_2=ones_vec(768),
    )
    params_out  = dict(W_exp_norm=ones_vec(768), W_action_out=rand_mat(768, MAX_STATE_DIM))

    image_rgb   = rng.random((_CH, _PIX, _PIX)).astype(np_bfloat16)
    text_emb    = create_text_emb(TEXT_VOCAB_SIZE, EMBD_S, SEQ_T)
    state_input = rand_mat(SEQ_S, MAX_STATE_DIM)
    weights     = rand_mat(MAX_STATE_DIM, EMBD_S)
    state_emb   = create_state_emb(state_input, weights)
    zeros       = np.zeros((PADDING, EMBD_S), dtype=np_bfloat16)
    action      = rand_mat(CHUNK_SIZE, 768)

    print("Running standalone VLA pipeline (no df.build())...")
    t0 = time.perf_counter()
    conv_emb  = preprocessing_block(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb = vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)
    t2 = time.perf_counter()
    llama_emb  = cpp.connector_block(vision_emb, params_con)
    t3 = time.perf_counter()

    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)
    assert mm_seq.shape == (128, EMBD_S)

    out = joint_transformer(LLAMA_NUM_LAYERS, mm_seq, action,
                            params_vlm, params_exp_self, params_exp_cross)
    t4 = time.perf_counter()
    v_t = postprocessing(out, params_out)
    t5 = time.perf_counter()

    print("\n== Timings (standalone, no rebuild) ==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder ({VIT_NUM_LAYERS}L)    : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer ({LLAMA_NUM_LAYERS}L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")
    print(f"Total                   : {t5 - t0:.3f} s")
    print(f"Output shape: {v_t.shape}")

    # =====================================================================
    # Validation (optional): Compare with PyTorch CPU reference
    # =====================================================================
    if "--validate" in sys.argv:
        print("\n== Running PyTorch CPU reference for validation ==")
        from vla_ref import (
            preproc_ref, vit_ref, con_ref, joint_transformer_ref, postprocessing_ref
        )
        torch.set_default_dtype(torch.float32)

        t_ref_0 = time.perf_counter()
        state_emb_ref = state_input @ weights

        conv_emb_ref = preproc_ref(image_rgb, params_proc)
        t_ref_1 = time.perf_counter()
        vision_emb_ref = vit_ref(VIT_NUM_LAYERS, conv_emb_ref, params_vit)
        t_ref_2 = time.perf_counter()
        llama_emb_ref = con_ref(vision_emb_ref, params_con)
        t_ref_3 = time.perf_counter()
        mm_seq_ref = np.concatenate([llama_emb_ref, text_emb, state_emb_ref, zeros], axis=0)
        out_ref = joint_transformer_ref(
            LLAMA_NUM_LAYERS, mm_seq_ref, action, params_vlm, params_exp_self, params_exp_cross
        )
        t_ref_4 = time.perf_counter()
        v_t_ref = postprocessing_ref(out_ref, params_out)
        t_ref_5 = time.perf_counter()

        print(f"\nPreprocessing (PyTorch) : {t_ref_1 - t_ref_0:.3f} s")
        print(f"Vision encoder (PyTorch): {t_ref_2 - t_ref_1:.3f} s")
        print(f"Connector (PyTorch)     : {t_ref_3 - t_ref_2:.3f} s")
        print(f"Joint transformer (PyTorch) : {t_ref_4 - t_ref_3:.3f} s")
        print(f"Postprocessing (PyTorch): {t_ref_5 - t_ref_4:.3f} s")
        print(f"Total (PyTorch CPU)     : {t_ref_5 - t_ref_0:.3f} s")

        # Compare outputs
        try:
            np.testing.assert_allclose(
                v_t.astype(np.float32),
                v_t_ref.astype(np.float32),
                atol=1e-1, rtol=1e-1
            )
            max_err = np.max(np.abs(v_t.astype(np.float32) - v_t_ref.astype(np.float32)))
            print(f"\n✅ VALIDATION PASSED")
            print(f"Max error: {max_err:.6f}")
            print(f"Speedup: {(t_ref_5 - t_ref_0) / (t5 - t0):.2f}×")
        except AssertionError as e:
            print(f"\n❌ VALIDATION FAILED")
            print(f"Output mismatch: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
