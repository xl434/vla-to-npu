import time
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import torch
import torch.nn as nn

from allo.ir.types import bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
from allo.backend.aie import ExternalModule
from allo.memory import Layout

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
    NEW_EMBD as EMBD_C, TEXT, 
    connector_block
)

from llama_block_rope_bf16 import (
    SEQ as SEQ_MM, EMBD as EMBD_MM, Q_H, KV_H, HEAD_DIM,
    llama_block_rope,
    llama_block_rope_cross
)

from vision_block_bf16 import (
    SEQ as SEQ_V, EMBD as EMBD_V,  
    vision_block,
)

SEQ_T = 48
EMBD_T = EMBD_S = TEXT
SEQ_S = 1
PADDING = 15
VIT_NUM_LAYERS = 12
LLAMA_NUM_LAYERS = 16
SKIP = 2

assert SEQ_V == 1024 and SEQ_T == 48 and SEQ_S == 1, SEQ_MM == 128
assert EMBD_V == 768
assert EMBD_T == EMBD_S == EMBD_MM == 960

TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
EXPERT_HIDDEN = TEXT * 0.1
CHUNK_SIZE = 32

def create_text_emb(vocab_size, hidden_size, seq):
    text_embed_tokens = nn.Embedding(vocab_size, hidden_size)
    lang_tokens = torch.randint(0, vocab_size, (seq,))
    lang_emb = text_embed_tokens(lang_tokens)  
    lang_emb = lang_emb * np.sqrt(hidden_size)
    return lang_emb.detach().float().numpy().astype(np_bfloat16)

MAX_STATE_DIM = 32

def create_state_emb(state_dim, hidden_size):
    state_input = np.random.randn(SEQ_S, state_dim).astype(np_bfloat16)
    weights = np.random.randn(state_dim, hidden_size).astype(np_bfloat16)

    top, mapping_primitives = GEMM(
        SEQ_S,
        hidden_size,
        state_dim,
        1,
        hidden_size // 64,
        1,
        bfloat16,
        bfloat16,
    )

    gemm_mod = df.build(
        top,
        target="aie",
        project="gemm.prj",
        mapping_primitives=mapping_primitives,
    )

    output = np.zeros((SEQ_S, hidden_size)).astype(np_bfloat16)
    gemm_mod(state_input, weights, output)
    return output

def vision_encoder(num_layers, x: np.ndarray, params: dict):
    for _ in range(num_layers):
        x = vision_block(x, params)
    return x

def joint_transformer(num_layers, vlm_input:np.ndarray, action:np.ndarray, vlm_params: dict, exp_params:dict):
    for i in range (num_layers):
        if (i % SKIP == 0):
            vlm_input = llama_block_rope(vlm_input, vlm_params)
            action = llama_block_rope(action, exp_params)
        else:
            vlm_input, keys, values = llama_block_rope(vlm_input, vlm_params)            
            action = llama_block_rope_cross(keys, values, action, exp_params)  

KERNEL_BF16_PATH = "../cc/bf16_old/"

norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path=KERNEL_BF16_PATH + "rms_norm_96_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

S = Layout.Shard
R = Layout.Replicate

EMBD = 96
NORM_P0 = 8
NORM_SEQ_TILE = 32
NORM_TILE = NORM_SEQ_TILE // NORM_P0
norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def rms_norm_kernel(
    A: Ty[NORM_SEQ_TILE, EMBD],
    B: Ty[EMBD],
    C: Ty[NORM_SEQ_TILE, EMBD],
):
    @df.kernel(mapping=[NORM_P0], args=[A, B, C])
    def core(
        local_A: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_B: Ty[EMBD] @ norm_arg_layout,
        local_C: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        norm(local_A, local_B, local_C)

rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="llama_bf16/rms_norm.prj")

def postprocessing(out, out_params: dict):
    rms_norm_mod(out, out_params["W_exp_norm"], out)   # (32, 96)

    top, mapping_primitives = GEMM(
        CHUNK_SIZE,
        MAX_STATE_DIM,
        EXPERT_HIDDEN,
        1,
        1,
        1,
        bfloat16,
        bfloat16,
    )

    gemm_mod = df.build(
        top,
        target="aie",
        project="gemm.prj",
        mapping_primitives=mapping_primitives,
    )

    v_t = np.zeros((CHUNK_SIZE, MAX_STATE_DIM)).astype(np_bfloat16)
    gemm_mod(out, out_params["W_action_out"], v_t)         # (32, 32)
    return v_t

def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    np.random.seed(0)

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

    params_out = dict(
        W_exp_norm=rand_mat(EXPERT_HIDDEN),
        W_action_out=rand_mat(EXPERT_HIDDEN, MAX_STATE_DIM),
    )

    # Expert layer params (text-only)
    params_exp = dict(
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

    image_rgb = rng.random((CH, PIX, PIX)).astype(np_bfloat16)

    t0 = time.perf_counter()
    conv_emb = preprocessing_block(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb = vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)
    t2 = time.perf_counter()
    llama_emb = connector_block(vision_emb, params_con)
    t3 = time.perf_counter()

    text_emb  = create_text_emb(TEXT_VOCAB_SIZE, EMBD_T, SEQ_T)
    state_emb = create_state_emb(MAX_STATE_DIM, EMBD_S)
    zeros = np.zeros((PADDING, EMBD_S)).astype(np_bfloat16)

    # -----------------------------
    # Multimodal concat for VLM layer (64 vision + 48 text + 1 state + 15 pad = 128)
    # -----------------------------
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)  # [128, 960]

    action = rand_mat(CHUNK_SIZE, EXPERT_HIDDEN)
    action = np.pad(action, (0, 4*CHUNK_SIZE), (0, 10*EXPERT_HIDDEN))
    out = joint_transformer(LLAMA_NUM_LAYERS, mm_seq, action, params_vlm, params_exp)
    t4 = time.perf_counter()

    v_t = postprocessing(out, params_out)
    t5 = time.perf_counter()

    # -----------------------------
    # Report timings and shapes
    # -----------------------------

    print("== Timings ==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder (12L).   : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer (16L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")
    print(v_t.shape)

if __name__ == "__main__":
    main()
