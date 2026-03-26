import time
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import torch

"""
SmolVLA

[Paper]()

Designed by Hugging Face.
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐    │
│ |         │────► │      │    │
│ |         │ kv   │      │    │
│ |         │────► │Action│    │
│ |   VLM   │cache │Expert│    |
│ │         │────► |      │    │
│ │         │      │      │    │
│ └▲──▲───▲─┘      └───▲──┘    |
│  │  |   |            │       |
│  |  |   |          noise     │
│  │  │ state                  │
│  │ language tokens           │
│  image(s)                    │
└──────────────────────────────┘
"""

from preprocessing_bf16 import (
    CHANNELS as CH, PIX_LEN as PIX, KERNEL_DIM,
    preprocessing_block
)

from connector_bf16 import (
    NEW_SEQ as SEQ_C, NEW_EMBD as EMBD_C, TEXT, 
    connector_block
)

from llama_block_rope_bf16 import (
    SEQ as SEQ_MM, EMBD as EMBD_MM, Q_H, KV_H, HEAD_DIM,
    vlm_qkv_from_mm_seq,
    llama_q_from_emb_rope,          # pre-norm + Q (RoPE) for text
    llama_block_rope_cross, 
)

from vision_block_bf16 import (
    SEQ as SEQ_V, EMBD as EMBD_V,  # SEQ_V==1024, EMBD_V==768
    vision_block,
)

SEQ_T = 48
EMBD_T = EMBD_S = TEXT
SEQ_S = 1
PADDING = 15

assert SEQ_V == 1024 and SEQ_T == 48 and SEQ_S == 1, SEQ_MM == 128
assert EMBD_V == 768
assert EMBD_T == EMBD_S == EMBD_MM == 960

def vision_encoder(num_layers, x: np.ndarray, params: dict):
    for i in range(num_layers):
        x = vision_block(x, params)
    return x

def main():
    # 12 vision layers
    # 16 cross attn layers
    # expert hidden = 960 * 0.1 = 96
    # add helper functions
    rng = np.random.default_rng(0)


    def rand_mat(m, n): return rng.standard_normal((m, n)).astype(np_bfloat16)
    def rand_vec(n):    return rng.standard_normal((n,)).astype(np_bfloat16)

    params_proc = dict(
        kernel=rand_mat(KERNEL_DIM, KERNEL_DIM)
    )

    params_con = dict(
        W=rand_mat(EMBD_C, TEXT)
    )

    # Vision params (ViT-style)
    params_vit = dict(
        Wq=rand_mat(EMBD_V, EMBD_V), Wk=rand_mat(EMBD_V, EMBD_V), Wv=rand_mat(EMBD_V, EMBD_V),
        Wo=rand_mat(EMBD_V, EMBD_V),
        W_up=rand_mat(EMBD_V, 4*EMBD_V), W_down=rand_mat(4*EMBD_V, EMBD_V),
        W_norm_1=rand_vec(EMBD_V), b_norm_1=rand_vec(EMBD_V),
        W_norm_2=rand_vec(EMBD_V), b_norm_2=rand_vec(EMBD_V),
    )

    # VLM (SmolLM) layer params
    params_vlm = dict(
        Wq=rand_mat(EMBD_MM, Q_H*HEAD_DIM),
        Wk=rand_mat(EMBD_MM, KV_H*HEAD_DIM),
        Wv=rand_mat(EMBD_MM, KV_H*HEAD_DIM),
        Wo=rand_mat(Q_H*HEAD_DIM, EMBD_MM),
        W_up=rand_mat(EMBD_MM, 4*EMBD_MM),
        W_gate=rand_mat(EMBD_MM, 4*EMBD_MM),
        W_down=rand_mat(4*EMBD_MM, EMBD_MM),
        W_norm_1=rand_vec(EMBD_MM),
        W_norm_2=rand_vec(EMBD_MM),
    )

    # Expert layer params (text-only, SEQ_T=64)
    params_exp = dict(
        Wq=rand_mat(EMBD_T, Q_H*HEAD_DIM),
        Wk=rand_mat(EMBD_T, KV_H*HEAD_DIM),
        Wv=rand_mat(EMBD_T, KV_H*HEAD_DIM),
        Wo=rand_mat(Q_H*HEAD_DIM, EMBD_T),
        W_up=rand_mat(EMBD_T, 4*EMBD_T),
        W_gate=rand_mat(EMBD_T, 4*EMBD_T),
        W_down=rand_mat(4*EMBD_T, EMBD_T),
        W_norm_1=rand_vec(EMBD_T),
        W_norm_2=rand_vec(EMBD_T),
    )

    image_rgb = rng.random((CH, PIX, PIX)).astype(np_bfloat16)

    t0 = time.perf_counter()
    conv_emb = preprocessing_block(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb = vision_encoder(1, conv_emb, params_vit)
    t2 = time.perf_counter()
    llama_emb = connector_block(vision_emb, params_con)
    t3 = time.perf_counter()

    text_emb  = rng.standard_normal((SEQ_T, EMBD_T), dtype=np_bfloat16)
    state_emb = rng.standard_normal((SEQ_S, EMBD_S), dtype=np_bfloat16)
    zeros = np.zeros((PADDING, EMBD_S)).astype(np_bfloat16)

    # -----------------------------
    # 3) Multimodal concat for VLM layer (64 vision + 48 text + 1 state + 15 pad = 128)
    # -----------------------------
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)  # [128, 960]
    assert mm_seq.shape == (SEQ_MM, EMBD_MM)

    # -----------------------------
    # 4) VLM (SmolLM) layer (1 depth)
    #    - K_vlm, V_vlm from pre-norm (no RoPE) for expert cross-attn
    #    - Y_vlm: full layer output (self-attn w/ RoPE + MLP)
    # -----------------------------
    t2 = time.perf_counter()
    K_vlm, V_vlm, Y_vlm = vlm_qkv_from_mm_seq(mm_seq, params_vlm)
    t3 = time.perf_counter()

    # -----------------------------
    # 5) Expert layer (text-only, 1 depth)
    # -----------------------------
    action_noise   = rng.standard_normal((SEQ_MM, EMBD_T), dtype=np_bfloat16)
    residual_exp = action_noise.copy()  # [64, 768]

    t4 = time.perf_counter()
    Q_exp, X_pre_exp = llama_q_from_emb_rope(action_noise, params_exp)             # [64, 960], [64, 768]
    Y_exp = llama_block_rope_cross(Q_exp, K_vlm, V_vlm, residual_exp, params_exp)  # [64, 768]
    t5 = time.perf_counter()

    # -----------------------------
    # 6) Report shapes + timings
    # -----------------------------
    print("== Shapes ==")
    print("img_token :", image_token.shape)
    print("vision_emb :", vision_emb.shape)
    print("mm_seq     :", mm_seq.shape)
    print("K_vlm      :", K_vlm.shape, "  V_vlm:", V_vlm.shape, "  Y_vlm:", Y_vlm.shape)
    print("Q_exp      :", Q_exp.shape,  "  Y_exp:", Y_exp.shape)

    print("\n== Timings ==")
    print(f"Vision enc (1L)      : {t1 - t0:.3f} s")
    print(f"VLM layer (1L)       : {t3 - t2:.3f} s")
    print(f"Expert x-attn (1L)   : {t5 - t4:.3f} s")

if __name__ == "__main__":
    main()
