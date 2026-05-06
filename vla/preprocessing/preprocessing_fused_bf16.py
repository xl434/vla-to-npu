"""
preprocessing_fused_bf16.py — Conv2d patch embedding fully on AMD NPU.

Pipeline (no CPU computation in the hot path):
  1. im2col   (NPU): image strips → patches tiles     32 calls, 1 context
  2. GEMM     (NPU): patches @ kernel_B → output     768 calls, 1 context

im2col kernel (mapping=[PW=32], one core per output patch):
  image_strip[3, 16, 512]  →  patches_row[32, 768]
  patches[pw, c*KH*KW + ky*KW + kx] = image_strip[c, ky, pw*KW + kx]
  Per core: [3, 16, 16] = 1536 bytes in, [1, 768] = 1536 bytes out ✓

GEMM kernel (Pk=4 K-chain, M=32, N=32, K=768):
  patches[32, 768] × kernel_B[768, 32] → output[32, 32]
  Per-core SRAM: A[32,192]×2 + B[192,32]×2 + C[32,32]×2 ≈ 54 KB < 64 KB ✓

One-time CPU setup (weights are static):
  kernel_B_tiles[24, 768, 32]  — pre-tiled, contiguous B tiles for GEMM

Per-inference CPU work:
  image strip rearrangement: one reshape+transpose+contiguous copy (~1.6 MB)
"""

import time
import torch
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
from allo.memory import Layout
from allo.library.aie.modules.gemm import GEMM

S = Layout.Shard
R = Layout.Replicate
Ty = bfloat16

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
SEQ      = 1024
EMBD_DIM = 768
CHANNELS = 3
KERNEL_H = 16
KERNEL_W = 16
PIX_LEN  = 512
PH       = PIX_LEN // KERNEL_H   # 32  — rows of patches
PW       = PIX_LEN // KERNEL_W   # 32  — patches per row
K_FULL   = CHANNELS * KERNEL_H * KERNEL_W  # 768

M_TILE   = PW        # 32  — patches per GEMM M-tile = one patch row
Pn       = 4         # N-parallel lanes (4 × N_TILE_LANE = N_TILE total)
N_TILE   = 32 * Pn   # 128 — output dims per GEMM call (Pn lanes × 32 each)
Pk       = 4         # K-chain depth
# Total cores: Pm×Pn×Pk = 1×4×4 = 16 (exactly the NPU physical limit)
# Per-core SRAM: A[32,192]×2 + B[192,32]×2 + C_pipe[32,32]×4 ≈ 57 KB < 64 KB ✓

# ---------------------------------------------------------------------------
# im2col NPU kernel
#
# Processes one-quarter of a patch row (8 patches) per call.
# Single-core (mapping=[1]), no layout sharding.
#
# With double-buffering: 2×input + 2×output must fit in 63KB:
#   Input  [C*KH=48, PIX_QT=128]  = 6144 elem = 12288 B → ×2 = 24576 B
#   Output [PW_QT=8,  K_FULL=768]  = 6144 elem = 12288 B → ×2 = 24576 B
#   Total: 49152 B + 1024 B stack = 50176 B < 65536 B ✓
#
# Total calls: PH=32 rows × 4 quarters = 128 NPU calls
# ---------------------------------------------------------------------------
IMG2D  = CHANNELS * KERNEL_H   # 48  — flattened C×KH
PIX_QT = PIX_LEN // 4           # 128 — columns per quarter-row
PW_QT  = PW // 4                # 8   — patches per quarter-row call
LyRR   = [R, R]                 # no sharding: single core gets full tile

@df.region()
def im2col_region(
    image_qt: Ty[IMG2D, PIX_QT],
    patches_qt: Ty[PW_QT, K_FULL],
):
    @df.kernel(mapping=[1], args=[image_qt, patches_qt])
    def core(
        local_img: Ty[IMG2D, PIX_QT] @ LyRR,
        local_pat: Ty[PW_QT, K_FULL] @ LyRR,
    ):
        for pw, c, ky, kx in allo.grid(PW_QT, CHANNELS, KERNEL_H, KERNEL_W):
            local_pat[pw, c * KERNEL_H * KERNEL_W + ky * KERNEL_W + kx] = \
                local_img[c * KERNEL_H + ky, pw * KERNEL_W + kx]

im2col_mod = df.build(
    im2col_region,
    target="aie",
    project="preprocessing/im2col.prj",
)

# ---------------------------------------------------------------------------
# GEMM NPU kernel  (Pn=4 N-parallel, Pk=4 K-chain)
#
# Computes patches[32, 768] @ kernel_B[768, 128] → out[32, 128]
# 16-core grid: 4 Pn-lanes × 4 Pk-stages.  Each core holds:
#   A[32,192]×2 + B[192,32]×2 + C_pipe[32,32]×4 ≈ 57 KB < 64 KB ✓
# Calls: 32 M-tiles × 6 N-tiles = 192  (vs 768 with Pn=1, N_TILE=32)
# ---------------------------------------------------------------------------
fused_top, mapping_primitives = GEMM(
    M_TILE, N_TILE, K_FULL,
    Pm=1, Pn=Pn, Pk=Pk,
    TyI=Ty, TyO=Ty,
)

fused_mod = df.build(
    fused_top,
    target="aie",
    project="preprocessing/fused_conv_add.prj",
    mapping_primitives=mapping_primitives,
)

# ---------------------------------------------------------------------------
# pre_tile_kernel — one-time CPU setup for static weights
#
# Stores kernel as kernel_B_tiles[N_TILES, K_FULL, N_TILE] so that each
# tile kernel_B_tiles[n] is contiguous in memory — no per-call CPU gather.
# ---------------------------------------------------------------------------
def pre_tile_kernel(kernel: np.ndarray) -> np.ndarray:
    """kernel[EMBD_DIM, C, KH, KW] → kernel_B_tiles[n_n, K_FULL, N_TILE]"""
    n_n = EMBD_DIM // N_TILE                            # 6  (was 24 with N_TILE=32)
    kernel_2d = kernel.reshape(EMBD_DIM, K_FULL)        # [768, 768] = [N, K]
    tiles = np.empty((n_n, K_FULL, N_TILE), dtype=np_bfloat16)
    for n in range(n_n):
        tiles[n] = kernel_2d[n * N_TILE:(n + 1) * N_TILE, :].T  # [K, N_TILE]
    return tiles  # [6, 768, 128], each tile contiguous

# ---------------------------------------------------------------------------
# conv2d_fused
#
# Hot-path inference function. CPU work is limited to:
#   - One reshape+transpose of the image into strips (mandatory for DMA)
#   - O(1) slice views per NPU call (no copies)
#
# NPU work:
#   - 128 im2col calls  (32 strip rows × 4 quarter-cols)
#   - 192 GEMM calls    (32 M-tiles × 6 N-tiles, each N-tile = 128 dims via Pn=4)
#
# Args:
#   image            : [C=3, 512, 512]    bfloat16
#   kernel_B_tiles   : [6, 768, 128]      bfloat16 (pre-tiled once at load time)
#   output           : [1024, 768]         bfloat16
# ---------------------------------------------------------------------------
def conv2d_fused(image: np.ndarray,
                 kernel_B_tiles: np.ndarray,
                 output: np.ndarray) -> None:
    t0 = time.time()

    # Rearrange image: [3,512,512] → [PH, C*KH, PIX_LEN] = [32, 48, 512]
    # Flatten C and KH so each strip is 2D (Allo layout only supports ≤2D tensors).
    image_strips = np.ascontiguousarray(
        image.reshape(CHANNELS, PH, KERNEL_H, PIX_LEN)
             .transpose(1, 0, 2, 3)
             .reshape(PH, IMG2D, PIX_LEN)
    )  # [32, 48, 512]

    patches = np.empty((SEQ, K_FULL), dtype=np_bfloat16)

    t1 = time.time()

    # Phase 1 — im2col on NPU (128 calls: PH=32 rows × 4 quarter-rows each)
    for ph in range(PH):
        for qt in range(4):
            col_start = qt * PIX_QT
            pat_start = ph * PW + qt * PW_QT
            im2col_mod(
                image_strips[ph, :, col_start:col_start + PIX_QT],   # [48, 128]
                patches[pat_start:pat_start + PW_QT, :],              # [8, 768]
            )

    t2 = time.time()

    # Phase 2 — GEMM on NPU (32 × 6 = 192 calls; Pn=4 handles 128 output dims/call)
    n_m = SEQ      // M_TILE   # 32
    n_n = EMBD_DIM // N_TILE   # 6

    for m in range(n_m):
        A_tile = patches[m * M_TILE:(m + 1) * M_TILE, :]       # [32, 768] contiguous
        for n in range(n_n):
            fused_mod(
                A_tile,                                           # [32, 768]
                kernel_B_tiles[n],                                # [768, 32] contiguous
                output[m * M_TILE:(m + 1) * M_TILE,
                       n * N_TILE:(n + 1) * N_TILE],             # [32, 32]
            )

    t3 = time.time()
    print(f"image strip rearrange (CPU): {(t1 - t0) * 1e3:.2f} ms")
    print(f"im2col (NPU):               {(t2 - t1) * 1e3:.2f} ms")
    print(f"GEMM   (NPU):               {(t3 - t2) * 1e3:.2f} ms")
    print(f"total:                      {(t3 - t0) * 1e3:.2f} ms")


def preprocessing_block(x_fp32: np.ndarray, params: dict) -> np.ndarray:
    image  = x_fp32.astype(np_bfloat16)
    kernel_B_tiles = pre_tile_kernel(params["kernel"])
    output = np.zeros((SEQ, EMBD_DIM), dtype=np_bfloat16)
    conv2d_fused(image, kernel_B_tiles, output)
    return output


# ---------------------------------------------------------------------------
# Test / validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing fused conv2d patch embedding (NPU im2col + GEMM) ===")

    rng = np.random.default_rng(0)
    image_np  = rng.standard_normal((CHANNELS, 512, 512)).astype(np_bfloat16)
    kernel_np = rng.standard_normal((EMBD_DIM, CHANNELS, KERNEL_H, KERNEL_W)).astype(np_bfloat16)
    output_np = np.zeros((SEQ, EMBD_DIM), dtype=np_bfloat16)

    kernel_B_tiles = pre_tile_kernel(kernel_np)
    conv2d_fused(image_np, kernel_B_tiles, output_np)

    # PyTorch reference
    image_t  = torch.from_numpy(image_np.view(np.int16)).view(torch.bfloat16).unsqueeze(0).float()
    kernel_t = torch.from_numpy(kernel_np.view(np.int16)).view(torch.bfloat16).float()

    conv_ref = torch.nn.Conv2d(CHANNELS, EMBD_DIM, KERNEL_H, stride=KERNEL_H, padding=0, bias=False)
    with torch.no_grad():
        conv_ref.weight.copy_(kernel_t)
    with torch.no_grad():
        ref = conv_ref(image_t)                                          # [1, 768, 32, 32]
    ref = ref.squeeze(0).permute(1, 2, 0).reshape(SEQ, EMBD_DIM).numpy()  # [1024, 768]

    out_f32  = output_np.astype(np.float32)
    max_err  = np.max(np.abs(out_f32 - ref))
    mean_err = np.mean(np.abs(out_f32 - ref))
    print(f"Max absolute error:  {max_err:.4f}")
    print(f"Mean absolute error: {mean_err:.4f}")
    try:
        np.testing.assert_allclose(out_f32, ref, rtol=1e-1, atol=1.0)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
