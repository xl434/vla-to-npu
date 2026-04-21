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

from text_encoder_bf16 import (
    SEQ as TEXT_ENC_SEQ, EMBD as EMBD_TEXT, Q_H as TEXT_Q_H,
    KV_H as TEXT_KV_H, HEAD_DIM as TEXT_HEAD_DIM, FFN_HID as TEXT_FFN_HID,
    text_encoder_forward,
    TextEncoderBlock,
)

from action_expert_bf16 import (
    SEQ as EXP_SEQ, EMBD as EMBD_EXP, Q_H as EXP_Q_H,
    KV_H as EXP_KV_H, HEAD_DIM as EXP_HEAD_DIM, KV_DIM as EXP_KV_DIM,
    FFN_HID as EXP_FFN_HID,
    action_expert_self_forward,
    action_expert_cross_forward,
    ActionExpertSelfBlock, ActionExpertCrossBlock,
)

from vision_block_bf16 import (
    EMBD as EMBD_V,
    vision_block,
    MiniVit
)

# ===============================================================================
# VLA Constants
# ===============================================================================
SEQ_T = 48                          # text token sequence length
EMBD_T = EMBD_S = TEXT              # 960 — text/state embedding dim
SEQ_S = 1                           # state sequence length
PADDING = 15                        # padding to make mm_seq = 128
VIT_NUM_LAYERS = 1
LLAMA_NUM_LAYERS = 2
SKIP = 2                            # cross-attention every SKIP layers
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
CHUNK_SIZE = EXP_SEQ                # 32

# ===============================================================================
# NPU Modules — State Embedding
# ===============================================================================
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

# ===============================================================================
# NPU Modules — Postprocessing (action expert output → action dims)
# ===============================================================================
# Postprocessing GEMM: [32, 768] × [768, 32] → [32, 32]
top, mapping_primitives = GEMM(
    CHUNK_SIZE,
    MAX_STATE_DIM,
    EMBD_EXP,          # 768
    1,
    1,
    EMBD_EXP // 64,    # 12
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

# ===============================================================================
# NPU Modules — Postprocessing RMSNorm (width=768)
# ===============================================================================
norm = ExternalModule(
    top="rms_norm_bf16",
    impl_path="../cc/bf16_old/rms_norm_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

S = Layout.Shard
R = Layout.Replicate

POST_EMBD = EMBD_EXP          # 768
NORM_P0 = 8
NORM_SEQ_TILE = 32
NORM_TILE = NORM_SEQ_TILE // NORM_P0
norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def rms_norm_kernel(
    A: Ty[NORM_SEQ_TILE, POST_EMBD],
    B: Ty[POST_EMBD],
    C: Ty[NORM_SEQ_TILE, POST_EMBD],
):
    @df.kernel(mapping=[NORM_P0], args=[A, B, C])
    def core(
        local_A: Ty[NORM_SEQ_TILE, POST_EMBD] @ norm_io_layout,
        local_B: Ty[POST_EMBD] @ norm_arg_layout,
        local_C: Ty[NORM_SEQ_TILE, POST_EMBD] @ norm_io_layout,
    ):
        norm(local_A, local_B, local_C)

rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="postprocessing/rms_norm.prj")


# ===============================================================================
# Helper Functions
# ===============================================================================
def create_text_emb(vocab_size, hidden_size, seq):
    text_embed_tokens = nn.Embedding(vocab_size, hidden_size)
    lang_tokens = torch.randint(0, vocab_size, (seq,))
    lang_emb = text_embed_tokens(lang_tokens)
    lang_emb = lang_emb * np.sqrt(hidden_size)
    return lang_emb.detach().float().numpy().astype(np_bfloat16)


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


# ===============================================================================
# Joint Transformer (Text Encoder + Action Expert)
# ===============================================================================
def joint_transformer(num_layers, vlm_input: np.ndarray, action: np.ndarray,
                      vlm_params: dict, exp_self_params: dict, exp_cross_params: dict):
    """
    vlm_input:  [128, 960] — text encoder input
    action:     [32, 768]  — action expert input
    vlm_params: per-layer text encoder params
    exp_self_params:  action expert self-attention params
    exp_cross_params: action expert cross-attention params (has Wk_cross, Wv_cross)
    """
    for i in range(num_layers):
        if i % SKIP == 0:
            vlm_output, text_k, text_v = text_encoder_forward(vlm_input, vlm_params)
            vlm_input = vlm_output
            action = action_expert_self_forward(action, exp_self_params)
        else:
            vlm_output, text_k, text_v = text_encoder_forward(vlm_input, vlm_params)
            vlm_input = vlm_output
            action = action_expert_cross_forward(action, text_k, text_v, exp_cross_params)
    return action


# ===============================================================================
# Postprocessing
# ===============================================================================
def postprocessing(out, out_params: dict):
    rms_norm_mod(out, out_params["W_exp_norm"], out)            # (32, 768) → (32, 768)
    v_t = np.zeros((CHUNK_SIZE, MAX_STATE_DIM)).astype(np_bfloat16)
    gemm_mod_post(out, out_params["W_action_out"], v_t)         # (32, 768) × (768, 32) → (32, 32)
    return v_t


# ===============================================================================
# Main
# ===============================================================================
def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Xavier-style scaling: 1/sqrt(fan_in) prevents accumulation blowup in bf16
    # Without this, connector's 192-tile GEMM (K=12288) produces values in 1000s range,
    # causing SiLU overflow and NaN cascades in subsequent layers.
    def rand_mat(m, n): return (rng.standard_normal((m, n)) / np.sqrt(m)).astype(np_bfloat16)
    def rand_vec(n):    return rng.standard_normal((n,)).astype(np_bfloat16)

    # --- Preprocessing params ---
    params_proc = dict(
        kernel=(rng.standard_normal((EMBD_P, CH, KERNEL_DIM, KERNEL_DIM)) / np.sqrt(CH * KERNEL_DIM * KERNEL_DIM)).astype(np_bfloat16)
    )

    # --- Connector params ---
    params_con = dict(
        W=rand_mat(EMBD_C, TEXT)                                 # [12288, 960]
    )

    # --- Vision params (ViT-style) ---
    params_vit = dict(
        Wq=rand_mat(EMBD_V, EMBD_V), Wk=rand_mat(EMBD_V, EMBD_V), Wv=rand_mat(EMBD_V, EMBD_V),
        Wo=rand_mat(EMBD_V, EMBD_V),
        W_up=rand_mat(EMBD_V, 4*EMBD_V), W_down=rand_mat(4*EMBD_V, EMBD_V),
        W_norm_1=rand_vec(EMBD_V), b_norm_1=rand_vec(EMBD_V),
        W_norm_2=rand_vec(EMBD_V), b_norm_2=rand_vec(EMBD_V),
    )

    # --- Text Encoder (VLM) layer params: EMBD=960, FFN=2560 ---
    params_vlm = dict(
        Wq=rand_mat(EMBD_TEXT, TEXT_Q_H * TEXT_HEAD_DIM),        # [960, 960]
        Wk=rand_mat(EMBD_TEXT, TEXT_KV_H * TEXT_HEAD_DIM),       # [960, 320]
        Wv=rand_mat(EMBD_TEXT, TEXT_KV_H * TEXT_HEAD_DIM),       # [960, 320]
        Wo=rand_mat(TEXT_Q_H * TEXT_HEAD_DIM, EMBD_TEXT),        # [960, 960]
        W_gate=rand_mat(EMBD_TEXT, TEXT_FFN_HID),                # [960, 2560]
        W_up=rand_mat(EMBD_TEXT, TEXT_FFN_HID),                  # [960, 2560]
        W_down=rand_mat(TEXT_FFN_HID, EMBD_TEXT),                # [2560, 960]
        W_norm_1=rand_vec(EMBD_TEXT),                            # [960]
        W_norm_2=rand_vec(EMBD_TEXT),                            # [960]
    )

    # --- Action Expert self-attention params: EMBD=768, FFN=2048 ---
    params_exp_self = dict(
        Wq=rand_mat(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM),         # [768, 960]
        Wk=rand_mat(EMBD_EXP, EXP_KV_DIM),                     # [768, 320]
        Wv=rand_mat(EMBD_EXP, EXP_KV_DIM),                     # [768, 320]
        Wo=rand_mat(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP),         # [960, 768]
        W_gate=rand_mat(EMBD_EXP, EXP_FFN_HID),                # [768, 2048]
        W_up=rand_mat(EMBD_EXP, EXP_FFN_HID),                  # [768, 2048]
        W_down=rand_mat(EXP_FFN_HID, EMBD_EXP),                # [2048, 768]
        W_norm_1=rand_vec(EMBD_EXP),                            # [768]
        W_norm_2=rand_vec(EMBD_EXP),                            # [768]
    )

    # --- Action Expert cross-attention params: same as self + cross K/V projections ---
    params_exp_cross = dict(
        Wq=rand_mat(EMBD_EXP, EXP_Q_H * EXP_HEAD_DIM),         # [768, 960]
        Wk_cross=rand_mat(EXP_KV_DIM, EXP_KV_DIM),             # [320, 320]
        Wv_cross=rand_mat(EXP_KV_DIM, EXP_KV_DIM),             # [320, 320]
        Wo=rand_mat(EXP_Q_H * EXP_HEAD_DIM, EMBD_EXP),         # [960, 768]
        W_gate=rand_mat(EMBD_EXP, EXP_FFN_HID),                # [768, 2048]
        W_up=rand_mat(EMBD_EXP, EXP_FFN_HID),                  # [768, 2048]
        W_down=rand_mat(EXP_FFN_HID, EMBD_EXP),                # [2048, 768]
        W_norm_1=rand_vec(EMBD_EXP),                            # [768]
        W_norm_2=rand_vec(EMBD_EXP),                            # [768]
    )

    # --- Postprocessing params ---
    params_out = dict(
        W_exp_norm=rand_vec(EMBD_EXP),                          # [768]
        W_action_out=rand_mat(EMBD_EXP, MAX_STATE_DIM),         # [768, 32]
    )

    # =====================================================================
    # Inputs
    # =====================================================================
    image_rgb = rng.random((CH, PIX, PIX)).astype(np_bfloat16)
    text_emb  = create_text_emb(TEXT_VOCAB_SIZE, EMBD_T, SEQ_T)  # [48, 960]
    state_input = rand_mat(SEQ_S, MAX_STATE_DIM)                  # [1, 32]
    weights = rand_mat(MAX_STATE_DIM, EMBD_S)                     # [32, 960]
    state_emb = create_state_emb(state_input, weights)            # [1, 960]
    zeros = np.zeros((PADDING, EMBD_S)).astype(np_bfloat16)       # [15, 960]
    action = rand_mat(CHUNK_SIZE, EMBD_EXP)                       # [32, 768]

    # =====================================================================
    # NPU Pipeline
    # =====================================================================
    t0 = time.perf_counter()
    conv_emb = preprocessing_block(image_rgb, params_proc)         # [1024, 768]
    t1 = time.perf_counter()
    vision_emb = vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)  # [1024, 768]
    t2 = time.perf_counter()
    llama_emb = connector_block(vision_emb, params_con)            # [64, 960]
    t3 = time.perf_counter()

    # Assemble multimodal sequence: [64 + 48 + 1 + 15 = 128, 960]
    mm_seq = np.concatenate([llama_emb, text_emb, state_emb, zeros], axis=0)
    assert mm_seq.shape == (TEXT_ENC_SEQ, EMBD_TEXT), \
        f"mm_seq shape {mm_seq.shape} != ({TEXT_ENC_SEQ}, {EMBD_TEXT})"

    out = joint_transformer(
        LLAMA_NUM_LAYERS, mm_seq, action, params_vlm, params_exp_self, params_exp_cross
    )
    t4 = time.perf_counter()
    v_t = postprocessing(out, params_out)
    t5 = time.perf_counter()

    # =====================================================================
    # Report timings
    # =====================================================================
    print("== Timings for NPU ==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder ({VIT_NUM_LAYERS}L)    : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer ({LLAMA_NUM_LAYERS}L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")
    print(f"Total                   : {t5 - t0:.3f} s")
    print(f"Output shape: {v_t.shape}")  # (32, 32)

    # =====================================================================
    # CPU Reference
    # =====================================================================
    state_emb_ref = state_input @ weights

    t0 = time.perf_counter()
    conv_emb_ref = preproc_ref(image_rgb, params_proc)
    t1 = time.perf_counter()
    vision_emb_ref = vit_ref(VIT_NUM_LAYERS, conv_emb_ref, params_vit)
    t2 = time.perf_counter()
    llama_emb_ref = con_ref(vision_emb_ref, params_con)
    t3 = time.perf_counter()
    mm_seq_ref = np.concatenate([llama_emb_ref, text_emb, state_emb_ref, zeros], axis=0)
    out_ref = joint_transformer_ref(
        LLAMA_NUM_LAYERS, mm_seq_ref, action, params_vlm, params_exp_self, params_exp_cross
    )
    t4 = time.perf_counter()
    v_t_ref = postprocessing_ref(out_ref, params_out)
    t5 = time.perf_counter()

    print("\n== Timings for CPU ==")
    print(f"Preprocessing           : {t1 - t0:.3f} s")
    print(f"Vision encoder ({VIT_NUM_LAYERS}L)    : {t2 - t1:.3f} s")
    print(f"Connector               : {t3 - t2:.3f} s")
    print(f"Joint transformer ({LLAMA_NUM_LAYERS}L) : {t4 - t3:.3f} s")
    print(f"Postprocessing          : {t5 - t4:.3f} s")

    # =====================================================================
    # Compare
    # =====================================================================
    np.testing.assert_allclose(
        v_t.astype(np.float32),
        v_t_ref.astype(np.float32),
        atol=1e-1, rtol=1e-1
    )
    print(f"\npostprocessing match — shape: {v_t.shape}")   # (32, 32)

    max_err  = np.max(np.abs(v_t.astype(np.float32) - v_t_ref.astype(np.float32)))
    mean_err = np.mean(np.abs(v_t.astype(np.float32) - v_t_ref.astype(np.float32)))
    print(f"max error:  {max_err:.6f}")
    print(f"mean error: {mean_err:.6f}")


# ===============================================================================
# CPU Reference Functions
# ===============================================================================

def preproc_ref(input: np.ndarray, params: dict):
    input_torch = torch.tensor(input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    conv = nn.Conv2d(
        in_channels=CH, out_channels=EMBD_P,
        kernel_size=KERNEL_DIM, stride=KERNEL_DIM, padding=0,
    ).to(torch.bfloat16)
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.tensor(params["kernel"].astype(np.float32)).to(torch.bfloat16))
        conv.bias   = nn.Parameter(torch.zeros(EMBD_P, dtype=torch.bfloat16))
    with torch.no_grad():
        out_torch = conv(input_torch)              # [1, 768, 32, 32]
    out_torch = out_torch.squeeze(0).flatten(1).transpose(0, 1)  # [1024, 768]
    return out_torch


def vit_ref(num_layers, input, params: dict):
    ref_model = MiniVit().eval()
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


def make_text_encoder_ref(params):
    ref = TextEncoderBlock().eval()
    p = ref
    p.attn.q_proj.weight.data      = torch.tensor(params["Wq"].T.astype(np.float32))
    p.attn.k_proj.weight.data      = torch.tensor(params["Wk"].T.astype(np.float32))
    p.attn.v_proj.weight.data      = torch.tensor(params["Wv"].T.astype(np.float32))
    p.attn.output_proj.weight.data = torch.tensor(params["Wo"].T.astype(np.float32))
    p.gate_proj.weight.data        = torch.tensor(params["W_gate"].T.astype(np.float32))
    p.up_proj.weight.data          = torch.tensor(params["W_up"].T.astype(np.float32))
    p.down_proj.weight.data        = torch.tensor(params["W_down"].T.astype(np.float32))
    p.ln_1.weight.data             = torch.tensor(params["W_norm_1"].astype(np.float32))
    p.ln_2.weight.data             = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def make_exp_self_ref(params):
    ref = ActionExpertSelfBlock().eval()
    ref.q_proj.weight.data    = torch.tensor(params["Wq"].T.astype(np.float32))
    ref.k_proj.weight.data    = torch.tensor(params["Wk"].T.astype(np.float32))
    ref.v_proj.weight.data    = torch.tensor(params["Wv"].T.astype(np.float32))
    ref.o_proj.weight.data    = torch.tensor(params["Wo"].T.astype(np.float32))
    ref.gate_proj.weight.data = torch.tensor(params["W_gate"].T.astype(np.float32))
    ref.up_proj.weight.data   = torch.tensor(params["W_up"].T.astype(np.float32))
    ref.down_proj.weight.data = torch.tensor(params["W_down"].T.astype(np.float32))
    ref.ln_1.weight.data      = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref.ln_2.weight.data      = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def make_exp_cross_ref(params):
    ref = ActionExpertCrossBlock().eval()
    ref.q_proj.weight.data    = torch.tensor(params["Wq"].T.astype(np.float32))
    ref.k_proj.weight.data    = torch.tensor(params["Wk_cross"].T.astype(np.float32))
    ref.v_proj.weight.data    = torch.tensor(params["Wv_cross"].T.astype(np.float32))
    ref.o_proj.weight.data    = torch.tensor(params["Wo"].T.astype(np.float32))
    ref.gate_proj.weight.data = torch.tensor(params["W_gate"].T.astype(np.float32))
    ref.up_proj.weight.data   = torch.tensor(params["W_up"].T.astype(np.float32))
    ref.down_proj.weight.data = torch.tensor(params["W_down"].T.astype(np.float32))
    ref.ln_1.weight.data      = torch.tensor(params["W_norm_1"].astype(np.float32))
    ref.ln_2.weight.data      = torch.tensor(params["W_norm_2"].astype(np.float32))
    return ref


def joint_transformer_ref(num_layers, vlm_input, action, params_vlm,
                           params_exp_self, params_exp_cross):
    vlm_ref = make_text_encoder_ref(params_vlm).to(torch.bfloat16)
    exp_self_ref = make_exp_self_ref(params_exp_self).to(torch.bfloat16)
    exp_cross_ref = make_exp_cross_ref(params_exp_cross).to(torch.bfloat16)

    vlm_t = torch.tensor(vlm_input.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)
    act_t = torch.tensor(action.astype(np.float32)).unsqueeze(0).to(torch.bfloat16)

    for i in range(num_layers):
        with torch.no_grad():
            vlm_t, text_k, text_v = vlm_ref(vlm_t)
            if i % SKIP == 0:
                act_t = exp_self_ref(act_t)
            else:
                act_t = exp_cross_ref(act_t, text_k.unsqueeze(0), text_v.unsqueeze(0))

    act_out = act_t.squeeze(0).float().numpy().astype(np_bfloat16)
    return act_out


def postprocessing_ref(out, params_out):
    out_t = torch.tensor(out.astype(np.float32)).to(torch.bfloat16)
    rms = nn.RMSNorm(EMBD_EXP, elementwise_affine=True)
    rms.weight.data = torch.tensor(params_out["W_exp_norm"].astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        normed = rms(out_t)

    proj = nn.Linear(EMBD_EXP, MAX_STATE_DIM, bias=False)
    proj.weight.data = torch.tensor(params_out["W_action_out"].T.astype(np.float32)).to(torch.bfloat16)
    with torch.no_grad():
        v_t_ref = proj(normed)

    return v_t_ref.float().numpy().astype(np_bfloat16)


if __name__ == "__main__":
    main()
