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
    CHANNELS as CH, PIX_LEN as PIX, KERNEL_DIM, EMBD_DIM as EMBD_P,
    preprocessing_block
)

from connector_bf16 import (
    NEW_EMBD as EMBD_C, TEXT, 
    connector_block
)

from llama_block_rope_bf16 import (
    EMBD as EMBD_MM, Q_H, KV_H, HEAD_DIM,
    llama_block_rope,
    llama_block_rope_cross,
    AttentionExpertBlock, CrossAttentionBlock
)

from vision_block_bf16 import (
    EMBD as EMBD_V,  
    vision_block, 
    MiniVit
)

SEQ_T = 48
EMBD_T = EMBD_S = TEXT
SEQ_S = 1
PADDING = 15
VIT_NUM_LAYERS = 12
LLAMA_NUM_LAYERS = 16
SKIP = 2
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
EXPERT_HIDDEN = TEXT // 10
CHUNK_SIZE = 32

top, mapping_primitives = GEMM(
    16,
    EMBD_S,
    MAX_STATE_DIM,
    1,
    EMBD_S // 64,
    1,
    bfloat16,
    bfloat16,
)

gemm_mod_state = df.build(
    top,
    target="aie",
    project="state_emb/gemm.prj",
    mapping_primitives=mapping_primitives,
)

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

gemm_mod_post = df.build(
    top,
    target="aie",
    project="postprocessing/gemm.prj",
    mapping_primitives=mapping_primitives,
)
Ty = bfloat16

norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path="../cc/bf16_old/rms_norm_96_bf16.cc",
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

rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="postprocessing/rms_norm.prj")

def create_text_emb(vocab_size, hidden_size, seq):
    text_embed_tokens = nn.Embedding(vocab_size, hidden_size)
    lang_tokens = torch.randint(0, vocab_size, (seq,))
    lang_emb = text_embed_tokens(lang_tokens)  
    lang_emb = lang_emb * np.sqrt(hidden_size)
    return lang_emb.detach().float().numpy().astype(np_bfloat16)

MAX_STATE_DIM = 32

def create_state_emb(state_input, weights):
    state_input = np.pad(state_input, ((0, 15), (0, 0)))
    output = np.zeros((SEQ_S, EMBD_S)).astype(np_bfloat16)
    output = np.pad(output, ((0, 15), (0, 0)))
    gemm_mod_state(state_input, weights, output)
    return output[0:1, :]

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

def postprocessing(out, out_params: dict):
    rms_norm_mod(out, out_params["W_exp_norm"], out)            # (32, 96)
    v_t = np.zeros((CHUNK_SIZE, MAX_STATE_DIM)).astype(np_bfloat16)
    gemm_mod_post(out, out_params["W_action_out"], v_t)         # (32, 32)
    return v_t

def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    np.random.seed(0)

    def rand_mat(m, n): return rng.standard_normal((m, n)).astype(np_bfloat16)
    def rand_vec(n):    return rng.standard_normal((n,)).astype(np_bfloat16)

    params_proc = dict(
        kernel=rng.standard_normal((EMBD_P, CH, KERNEL_DIM, KERNEL_DIM)).astype(np_bfloat16)
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
        W_exp_norm=rand_vec(EXPERT_HIDDEN),
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
    text_emb  = create_text_emb(TEXT_VOCAB_SIZE, EMBD_T, SEQ_T)
    state_input = rand_mat(SEQ_S, MAX_STATE_DIM)
    weights = rand_mat(MAX_STATE_DIM, EMBD_S)
    state_emb = create_state_emb(state_input, weights)
    zeros = np.zeros((PADDING, EMBD_S)).astype(np_bfloat16)
    state_emb_ref = state_input @ weights
    action = rand_mat(CHUNK_SIZE, EXPERT_HIDDEN)
    action = np.pad(action, ((0, 3*CHUNK_SIZE), (0, 9*EXPERT_HIDDEN)))
    
    t0 = time.perf_counter()
    conv_emb = preprocessing_block(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb = vision_encoder(1, conv_emb, params_vit)
    t2 = time.perf_counter()
    llama_emb = connector_block(vision_emb, params_con)
    t3 = time.perf_counter()
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)  # [128, 960]
    
    print(mm_seq.shape, mm_seq)
    
    return
    # -----------------------------
    # CPU reference
    # -----------------------------
    
    t0 = time.perf_counter()
    conv_emb_ref = preproc_ref(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb_ref = vit_ref(VIT_NUM_LAYERS, conv_emb_ref, params_vit)
    t2 = time.perf_counter()
    llama_emb_ref = con_ref(vision_emb_ref, params_con)
    t3 = time.perf_counter()
    mm_seq_ref = np.concatenate([llama_emb_ref, text_emb, state_emb_ref, zeros], axis=0)  # [128, 960]
    out_ref = joint_transformer_ref(
        LLAMA_NUM_LAYERS, mm_seq_ref, action, params_vlm, params_exp
    )
    t4 = time.perf_counter()
    v_t_ref = postprocessing_ref(out_ref, params_out)
    t5 = time.perf_counter()

    # -----------------------------
    # Report timings and shapes
    # -----------------------------

    print("== Timings for CPU==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder (12L)    : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer (16L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")
    print(v_t_ref.shape)
    
    # -----------------------------
    # NPU implementation
    # -----------------------------

    t0 = time.perf_counter()
    conv_emb = preprocessing_block(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb = vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)
    t2 = time.perf_counter()
    llama_emb = connector_block(vision_emb, params_con)
    t3 = time.perf_counter()
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)  # [128, 960]
    out = joint_transformer(LLAMA_NUM_LAYERS, mm_seq, action, params_vlm, params_exp)
    t4 = time.perf_counter()
    v_t = postprocessing(out, params_out)
    t5 = time.perf_counter()

    # -----------------------------
    # Report timings and shapes
    # -----------------------------

    print("== Timings for NPU==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder (12L)    : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer (16L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")
    print(v_t.shape)

    # compare
    np.testing.assert_allclose(
        v_t.astype(np.float32),
        v_t_ref.astype(np.float32),
        atol=1e-1, rtol=1e-1
    )
    print(f"postprocessing match — shape: {v_t.shape}")   # (32, 32)

    # print error stats
    max_err  = np.max(np.abs(v_t.astype(np.float32) - v_t_ref.astype(np.float32)))
    mean_err = np.mean(np.abs(v_t.astype(np.float32) - v_t_ref.astype(np.float32)))
    print(f"max error:  {max_err:.6f}")
    print(f"mean error: {mean_err:.6f}")
    

# -----------------------------
# CPU reference tools
# -----------------------------

def preproc_ref(input: np.ndarray, params: dict):
    input_torch = torch.tensor(input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)  # [1, 3, 512, 512]

    conv = nn.Conv2d(
        in_channels=CH,
        out_channels=EMBD_P,
        kernel_size=KERNEL_DIM,
        stride=KERNEL_DIM,
        padding=0,
    ).to(torch.bfloat16)

    with torch.no_grad():
        conv.weight = nn.Parameter(torch.tensor(params["kernel"].astype(np.float32)).to(torch.bfloat16))
        conv.bias   = nn.Parameter(torch.zeros(EMBD_P, dtype=torch.bfloat16))

    with torch.no_grad():
        out_torch = conv(input_torch)              # [1, 768, 32, 32] bfloat16

    out_torch = out_torch.squeeze(0)               # [768, 32, 32]
    out_torch = out_torch.flatten(1)               # [768, 1024]
    out_torch = out_torch.transpose(0, 1)          # [1024, 768]

    return out_torch

def vit_ref(num_layers, input, params: dict):
    ref_model = MiniVit().eval()
    
    # set weights using .data to bypass grad check
    ref_model.attn.in_proj_weight.data = torch.tensor(
        np.concatenate([
            params["Wq"].T.astype(np.float32),
            params["Wk"].T.astype(np.float32),
            params["Wv"].T.astype(np.float32),
        ], axis=0)
    )
    ref_model.attn.out_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    ref_model.ffn_up.weight.data        = torch.tensor(params["W_up"].T.astype(np.float32))
    ref_model.ffn_down.weight.data      = torch.tensor(params["W_down"].T.astype(np.float32))
    ref_model.ln_1.weight.data          = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref_model.ln_1.bias.data            = torch.tensor(params["b_norm_1"].astype(np.float32))
    ref_model.ln_2.weight.data          = torch.tensor(params["W_norm_2"].astype(np.float32))
    ref_model.ln_2.bias.data            = torch.tensor(params["b_norm_2"].astype(np.float32))
    
    ref_model.to(torch.bfloat16)
    
    for _ in range(num_layers):
        with torch.no_grad():
            input = ref_model(input)
    
    return input.float().numpy().astype(np_bfloat16)

def con_ref(input: np.ndarray, params: dict):
    input = input.reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    ref = input @ params["W"]
    return ref

def make_llama_ref(params):
    ref = AttentionExpertBlock().eval()
    ref.attn.q_proj.weight.data      = torch.tensor(params["Wq"].T.astype(np.float32))
    ref.attn.k_proj.weight.data      = torch.tensor(params["Wk"].T.astype(np.float32))
    ref.attn.v_proj.weight.data      = torch.tensor(params["Wv"].T.astype(np.float32))
    ref.attn.output_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    ref.gate_proj.weight.data        = torch.tensor(params["W_gate"].T.astype(np.float32))
    ref.up_proj.weight.data          = torch.tensor(params["W_up"].T.astype(np.float32))
    ref.down_proj.weight.data        = torch.tensor(params["W_down"].T.astype(np.float32))
    ref.ln_1.weight.data             = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref.ln_2.weight.data             = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref

def joint_transformer_ref(num_layers, vlm_input, action, params_vlm, params_exp):
    vlm_ref = make_llama_ref(params_vlm).to(torch.bfloat16)
    exp_self_ref  = make_llama_ref(params_exp).to(torch.bfloat16)
    exp_cross_ref = CrossAttentionBlock().to(torch.bfloat16)
    # inject same weights into cross-attn block
    exp_cross_ref.q_proj.weight.data    = torch.tensor(params_exp["Wq"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.k_proj.weight.data    = torch.tensor(params_exp["Wk"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.v_proj.weight.data    = torch.tensor(params_exp["Wv"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.o_proj.weight.data    = torch.tensor(params_exp["Wo"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.gate_proj.weight.data = torch.tensor(params_exp["W_gate"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.up_proj.weight.data   = torch.tensor(params_exp["W_up"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.down_proj.weight.data = torch.tensor(params_exp["W_down"].T.astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.ln_1.weight.data      = torch.tensor(params_exp["W_norm_1"].astype(np.float32)).to(torch.bfloat16)
    exp_cross_ref.ln_2.weight.data      = torch.tensor(params_exp["W_norm_2"].astype(np.float32)).to(torch.bfloat16)

    vlm_t = torch.tensor(vlm_input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    act_t = torch.tensor(action.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)

    for i in range(num_layers):
        with torch.no_grad():
            if i % SKIP == 0:
                vlm_t = vlm_ref(vlm_t)
                act_t = exp_self_ref(act_t)
            else:
                vlm_t = vlm_ref(vlm_t)
                act_t = exp_cross_ref(act_t, vlm_t)  # expert cross-attends into VLM

    act_out = act_t.squeeze(0).float().numpy().astype(np_bfloat16)
    return act_out

def postprocessing_ref(out, params_out):
    # 1. RMSNorm
    out_t = torch.tensor(out.astype(np.float32)).to(torch.bfloat16)
    rms = nn.RMSNorm(out.shape[-1], elementwise_affine=True)
    rms.weight.data = torch.tensor(params_out["W_exp_norm"].astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        normed = rms(out_t)

    # 2. linear projection (CHUNK_SIZE, EXPERT_HIDDEN) -> (CHUNK_SIZE, MAX_STATE_DIM)
    proj = nn.Linear(out.shape[-1], MAX_STATE_DIM, bias=False)
    proj.weight.data = torch.tensor(params_out["W_action_out"].T.astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        v_t_ref = proj(normed)

    return v_t_ref.float().numpy().astype(np_bfloat16)
    
if __name__ == "__main__":
    main()