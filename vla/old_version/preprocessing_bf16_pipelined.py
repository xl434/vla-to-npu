"""
preprocessing_bf16_pipelined.py — fused 3-channel conv2d on NPU.

Key changes from preprocessing_bf16.py:
  • conv3ch kernel: fuses all 3 channel convolutions + accumulation into one call.
    Eliminates add_mod (768×3=2304 calls) — accumulation done inside kernel.
  • Pre-tile A once: [3,512,512] → [4, 3, 256, 256] (no per-call CPU copy).
  • Pre-tile kernel3: [768,3,16,16] → [768,48,16] (3 kernels stacked).
  • copy_mod retained for NPU reshape: [8,32] → [1,256] (same as original).

Original call counts:
  768 × 3 × 4 conv   = 9216
  768 × 3     add    = 2304
  768 × 4     copy   = 3072
  Total             = 14592 NPU calls

Optimized call counts:
  768 × 4 conv3ch   = 3072
  768 × 4 copy      = 3072
  Total             = 6144 NPU calls  (58% reduction)
"""

import time
import torch
import torch.nn as nn
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

KERNEL_LIB_PATH = "../cc/bf16_old/"
INPUT_DIM   = 256
KERNEL_DIM  = 16
OUTPUT_DIM  = 16
SCALE_FACTOR = 4
CHANNELS    = 3
PIX_LEN     = 512
SEQ         = 1024
EMBD_DIM    = 768
N_SPATIAL   = 4          # 2×2 spatial tiles per image
KERNEL3_DIM = CHANNELS * KERNEL_DIM  # 48 — stacked kernels
PACKED_DIM  = CHANNELS * INPUT_DIM   # 768 — 3 channels stacked row-wise

Ty = bfloat16

# ---------------------------------------------------------------------------
# conv3ch — fused 3-channel convolution (accumulates channels in the kernel)
# Input packing: A_packed[CHANNELS*INPUT_DIM, INPUT_DIM] = 3 channels stacked.
# Reduces DMA ports from 5 (3 ch + kernel + out) to 3 (A_packed + kernel + out).
# ---------------------------------------------------------------------------
conv3ch = ExternalModule(
    top="conv3ch",
    impl_path=KERNEL_LIB_PATH + "conv3ch_bf16.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

split_format  = [S(0), S(1)]
kernel_format = [R, R]

@df.region()
def conv3ch_kernel(
    A_packed: Ty[PACKED_DIM, INPUT_DIM],
    kernel3:  Ty[KERNEL3_DIM, KERNEL_DIM],
    output:   Ty[OUTPUT_DIM, OUTPUT_DIM],
):
    @df.kernel(mapping=[SCALE_FACTOR, SCALE_FACTOR],
               args=[A_packed, kernel3, output])
    def core(
        local_A:      Ty[PACKED_DIM, INPUT_DIM]   @ split_format,
        local_kernel: Ty[KERNEL3_DIM, KERNEL_DIM] @ kernel_format,
        local_output: Ty[OUTPUT_DIM, OUTPUT_DIM]  @ split_format,
    ):
        conv3ch(local_A, local_kernel, local_output)

conv3ch_mod = df.build(conv3ch_kernel, target="aie", project="preprocessing/conv3ch.prj")

# ---------------------------------------------------------------------------
# copy — NPU reshape: [8, 32] → [1, 256]  (same as original preprocessing)
# Maps 8 rows × 32 cols into a flat 256-element row in the output.
# ---------------------------------------------------------------------------
linear_in_layout  = [S(0), R]
linear_out_layout = [R, S(0)]

@df.region()
def copy_region(A: Ty[8, 32], C: Ty[1, 256]):
    @df.kernel(mapping=[8], args=[A, C])
    def mod(
        local_A: Ty[8, 32] @ linear_in_layout,
        local_C: Ty[1, 256] @ linear_out_layout,
    ):
        local_C[:, :] = local_A[:, :]

copy_mod = df.build(copy_region, target="aie", project="preprocessing/copy.prj")


def conv2d_pipelined(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    t0 = time.time()

    # Pre-tile image once: [CHANNELS, 512, 512] → [N_SPATIAL, CHANNELS, INPUT_DIM, INPUT_DIM]
    A_tiles = np.ascontiguousarray(
        A.reshape(CHANNELS, 2, INPUT_DIM, 2, INPUT_DIM)
         .transpose(1, 3, 0, 2, 4)
    ).reshape(N_SPATIAL, CHANNELS, INPUT_DIM, INPUT_DIM)
    # A_tiles[t, j] = [INPUT_DIM, INPUT_DIM] contiguous; t = k*2+l for k,l ∈ {0,1}

    # Pack 3 channels interleaved per spatial row group: [4, 3, 256, 256] → [4, 768, 256]
    # With S(0) sharding over SCALE_FACTOR=4 cores, each core gets 192 consecutive rows.
    # Simple reshape gives rows [ch0_0..255, ch1_0..255, ch2_0..255] → core 0 gets
    # ch0 rows 0..191 only (wrong: misses ch1, ch2 entirely).
    # Fix: interleave so core r gets [ch0[r*64:(r+1)*64], ch1[r*64:(r+1)*64], ch2[r*64:(r+1)*64]].
    # reshape [4,3,256,256] → [4,3,4,64,256] → transpose [4,4,3,64,256] → reshape [4,768,256].
    A_packed = np.ascontiguousarray(
        A_tiles.reshape(N_SPATIAL, CHANNELS, SCALE_FACTOR, INPUT_DIM // SCALE_FACTOR, INPUT_DIM)
               .transpose(0, 2, 1, 3, 4)
               .reshape(N_SPATIAL, PACKED_DIM, INPUT_DIM)
    )  # [4, 768, 256]; A_packed[t, r*192:(r+1)*192, :] = [ch0[r*64], ch1[r*64], ch2[r*64]]

    # Pre-tile kernel: [EMBD_DIM, CHANNELS, KERNEL_DIM, KERNEL_DIM] → [EMBD_DIM, KERNEL3_DIM, KERNEL_DIM]
    B3 = np.ascontiguousarray(
        B.reshape(EMBD_DIM, KERNEL3_DIM, KERNEL_DIM)
    )  # [768, 48, 16] — 3 channel kernels stacked along dim-0

    # embd[32, 32, EMBD_DIM] — spatial-first layout matches original for copy_mod
    embd = np.zeros((32, 32, EMBD_DIM), dtype=np_bfloat16)
    # out[1, SEQ, EMBD_DIM] — intermediate for copy_mod output (matches original)
    out = np.empty((1, SEQ, EMBD_DIM), dtype=np_bfloat16)

    t1 = time.time()

    # Phase 1 — 3072 conv3ch_mod calls (768 embedding dims × 4 spatial tiles)
    for i in range(EMBD_DIM):
        B3_i = B3[i]   # [KERNEL3_DIM, KERNEL_DIM] contiguous view
        for t in range(N_SPATIAL):
            k, l = t // 2, t % 2
            conv3ch_mod(
                A_packed[t],   # [PACKED_DIM, INPUT_DIM] = [768, 256]
                B3_i,          # [48, 16]
                embd[k * OUTPUT_DIM:(k + 1) * OUTPUT_DIM,
                     l * OUTPUT_DIM:(l + 1) * OUTPUT_DIM, i],
            )

    t2 = time.time()

    # Phase 2 — 3072 copy_mod calls: flatten spatial [8,32] → [1,256] on NPU
    # Mirrors original: embd[k*8:(k+1)*8, :, i] → out[0, k*256:(k+1)*256, i]
    for i in range(EMBD_DIM):
        for k in range(4):
            copy_mod(
                embd[k * 8:(k + 1) * 8, :, i],
                out[:, k * 256:(k + 1) * 256, i],
            )

    C[:, :] = out[0]

    t3 = time.time()
    print(f"A/B pre-tile (CPU): {(t1 - t0) * 1e3:.2f} ms")
    print(f"conv3ch      (NPU): {(t2 - t1) * 1e3:.2f} ms  ({EMBD_DIM * N_SPATIAL} calls)")
    print(f"copy/reshape (NPU): {(t3 - t2) * 1e3:.2f} ms  ({EMBD_DIM * 4} calls)")
    print(f"total:              {(t3 - t0) * 1e3:.2f} ms")


def preprocessing_block(x_fp32: np.ndarray, params: dict) -> np.ndarray:
    x = x_fp32.astype(np_bfloat16)
    out = np.zeros((SEQ, EMBD_DIM), dtype=np_bfloat16)
    conv2d_pipelined(x, params["kernel"], out)
    return out


# ---------------------------------------------------------------------------
# Test / validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing pipelined conv2d (fused 3-ch NPU + NPU copy reshape) ===")

    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((CHANNELS, PIX_LEN, PIX_LEN)).astype(np_bfloat16)
    B_np = rng.standard_normal((EMBD_DIM, CHANNELS, KERNEL_DIM, KERNEL_DIM)).astype(np_bfloat16)
    C_np = np.zeros((SEQ, EMBD_DIM), dtype=np_bfloat16)

    conv2d_pipelined(A_np, B_np, C_np)

    # PyTorch reference
    A_t = torch.from_numpy(A_np.view(np.int16)).view(torch.bfloat16).unsqueeze(0).float()
    B_t = torch.from_numpy(B_np.view(np.int16)).view(torch.bfloat16).float()
    ref_conv = nn.Conv2d(CHANNELS, EMBD_DIM, KERNEL_DIM, stride=KERNEL_DIM, padding=0, bias=False)
    with torch.no_grad():
        ref_conv.weight.copy_(B_t)
        ref = ref_conv(A_t)                                              # [1, 768, 32, 32]
    ref = ref.squeeze(0).permute(1, 2, 0).reshape(SEQ, EMBD_DIM).numpy()  # [1024, 768]

    out_f32  = C_np.astype(np.float32)
    max_err  = np.max(np.abs(out_f32 - ref))
    mean_err = np.mean(np.abs(out_f32 - ref))
    print(f"Max absolute error:  {max_err:.4f}")
    print(f"Mean absolute error: {mean_err:.4f}")
    try:
        np.testing.assert_allclose(out_f32, ref, rtol=1e-1, atol=1.0)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
