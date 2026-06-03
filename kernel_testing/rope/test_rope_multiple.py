import os
import numpy as np
import torch

import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie import ExternalModule

# ============================================================
# Shapes
# ============================================================

# AIE kernels are compiled for 32-row tiles.
# Full test can still use TEST_SEQ = 64 because rope_apply_packed
# loops over two 32-row tiles.
TILE_SEQ = 32
HEAD_DIM = 64
HEAD_DIM_HALF = HEAD_DIM // 2

# ============================================================
# Layouts / dtypes
# ============================================================

S = Layout.Shard
R = Layout.Replicate
Ty = float32

VecLy = [S(0)]
MatLy = [S(1), S(0)]

# ============================================================
# External kernels
# ============================================================

KERNEL_LIB_PATH = "../../cc/float/"

OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"
SIN_COS_IMPL = KERNEL_LIB_PATH + "sin_cos.cc"

# NOTE:
# We intentionally do NOT build these as AIE ExternalModules:
#
#   rope_make_radians_float32
#   pack32to64_float32
#
# because they triggered Allo lowering errors:
#
#   Operand 0 of operation "memref.copy" must be a Value
#
# Instead:
#   radians32 is computed on CPU:
#       radians32 = pos_pad[:, None] * inv_ts[None, :]
#
#   radians64 is packed on CPU:
#       radians64[:, :32] = radians32
#       radians64[:, 32:] = radians32

copyL_ext = ExternalModule(
    top="copy_left32_from64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0],
    output_idx=[1],
)

copyR_ext = ExternalModule(
    top="copy_right32_from64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0],
    output_idx=[1],
)

join_ext = ExternalModule(
    top="join32_to_64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],
    output_idx=[2],
)

mul32_ext = ExternalModule(
    top="mul32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],
    output_idx=[2],
)

add32_ext = ExternalModule(
    top="add32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],
    output_idx=[2],
)

sub32_ext = ExternalModule(
    top="sub32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],
    output_idx=[2],
)

# Packed-output sin/cos.
#
# Your C++ sin_cos.cc should export:
#
# extern "C" {
# void sin_cos_float32(
#     float in_mat[32][64],
#     float out_mat[64][64]
# ) {
#     sin_f32(in_mat, out_mat);
#     cos_kernel_32x64(in_mat, &out_mat[32]);
# }
# }
#
# Output layout:
#   packed_out64[0:32,  :] = sin
#   packed_out64[32:64, :] = cos
sin_cos_ext = ExternalModule(
    top="sin_cos_float32",
    impl_path=SIN_COS_IMPL,
    input_idx=[0],
    output_idx=[1],
)

# ============================================================
# AIE regions
# ============================================================

@df.region()
def sin_cos_region(
    in64: Ty[TILE_SEQ, HEAD_DIM],
    packed_out64: Ty[2 * TILE_SEQ, HEAD_DIM],
):
    @df.kernel(mapping=[1, 1], args=[in64, packed_out64])
    def core(
        local_in64: Ty[TILE_SEQ, HEAD_DIM] @ MatLy,
        local_packed_out64: Ty[2 * TILE_SEQ, HEAD_DIM] @ MatLy,
    ):
        sin_cos_ext(local_in64, local_packed_out64)


@df.region()
def copy_left_region(
    in64: Ty[TILE_SEQ, HEAD_DIM],
    out32: Ty[TILE_SEQ, HEAD_DIM_HALF],
):
    @df.kernel(mapping=[1, 1], args=[in64, out32])
    def core(
        local_in64: Ty[TILE_SEQ, HEAD_DIM] @ MatLy,
        local_out32: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
    ):
        copyL_ext(local_in64, local_out32)


@df.region()
def copy_right_region(
    in64: Ty[TILE_SEQ, HEAD_DIM],
    out32: Ty[TILE_SEQ, HEAD_DIM_HALF],
):
    @df.kernel(mapping=[1, 1], args=[in64, out32])
    def core(
        local_in64: Ty[TILE_SEQ, HEAD_DIM] @ MatLy,
        local_out32: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
    ):
        copyR_ext(local_in64, local_out32)


@df.region()
def join_region(
    left32: Ty[TILE_SEQ, HEAD_DIM_HALF],
    right32: Ty[TILE_SEQ, HEAD_DIM_HALF],
    out64: Ty[TILE_SEQ, HEAD_DIM],
):
    @df.kernel(mapping=[1, 1], args=[left32, right32, out64])
    def core(
        local_left32: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_right32: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_out64: Ty[TILE_SEQ, HEAD_DIM] @ MatLy,
    ):
        join_ext(local_left32, local_right32, local_out64)


@df.region()
def mul32_region(
    A: Ty[TILE_SEQ, HEAD_DIM_HALF],
    B: Ty[TILE_SEQ, HEAD_DIM_HALF],
    C: Ty[TILE_SEQ, HEAD_DIM_HALF],
):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(
        local_A: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_B: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_C: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
    ):
        mul32_ext(local_A, local_B, local_C)


@df.region()
def add32_region(
    A: Ty[TILE_SEQ, HEAD_DIM_HALF],
    B: Ty[TILE_SEQ, HEAD_DIM_HALF],
    C: Ty[TILE_SEQ, HEAD_DIM_HALF],
):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(
        local_A: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_B: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_C: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
    ):
        add32_ext(local_A, local_B, local_C)


@df.region()
def sub32_region(
    A: Ty[TILE_SEQ, HEAD_DIM_HALF],
    B: Ty[TILE_SEQ, HEAD_DIM_HALF],
    C: Ty[TILE_SEQ, HEAD_DIM_HALF],
):
    @df.kernel(mapping=[1, 1], args=[A, B, C])
    def core(
        local_A: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_B: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
        local_C: Ty[TILE_SEQ, HEAD_DIM_HALF] @ MatLy,
    ):
        sub32_ext(local_A, local_B, local_C)


# ============================================================
# Build AIE modules
# ============================================================

sin_cos_mod = df.build(
    sin_cos_region,
    target="aie",
    project="sin_cos.prj",
)

copyL_mod = df.build(
    copy_left_region,
    target="aie",
    project="copyL.prj",
)

copyR_mod = df.build(
    copy_right_region,
    target="aie",
    project="copyR.prj",
)

join_mod = df.build(
    join_region,
    target="aie",
    project="join.prj",
)

mul32_mod = df.build(
    mul32_region,
    target="aie",
    project="mul32.prj",
)

add32_mod = df.build(
    add32_region,
    target="aie",
    project="add32.prj",
)

sub32_mod = df.build(
    sub32_region,
    target="aie",
    project="sub32.prj",
)


# ============================================================
# Host helper: apply RoPE to packed multi-head tensor
# ============================================================

def rope_apply_packed(
    packed: np.ndarray,
    heads: int,
    head_dim: int = 64,
    max_wavelength: float = 10_000.0,
    pos_offset: int = 0,
) -> np.ndarray:
    """
    Apply RoPE to a packed tensor of shape:

        packed.shape = (seq_len, heads * head_dim)

    Returns:

        out.shape = same as packed

    This version uses:
      - CPU/NumPy for radians32
      - CPU/NumPy for packing radians32 -> radians64
      - AIE for sin_cos, copy, mul, add, sub, join

    AIE kernels are compiled for TILE_SEQ x HEAD_DIM = 32 x 64.
    The host loops over the full sequence in chunks of 32 rows.
    """

    seq_len, total_dim = packed.shape

    assert total_dim == heads * head_dim, (
        f"packed width must be heads * head_dim, got total_dim={total_dim}, "
        f"heads={heads}, head_dim={head_dim}"
    )

    D = head_dim
    HALF = D // 2

    assert D == HEAD_DIM, "this helper expects HEAD_DIM=64 kernels"
    assert HALF == HEAD_DIM_HALF, "head_dim must be 64, so half must be 32"

    tile_rows = TILE_SEQ

    out = np.empty_like(packed, dtype=np.float32)

    # inv_timescale[k] = max_wavelength^(-(2 / D) * k)
    # k = 0..31 for D=64
    k = np.arange(HALF, dtype=np.float32)
    inv_ts = (max_wavelength ** (-(2.0 / D) * k)).astype(np.float32)

    # Process sequence in 32-row AIE tiles.
    for t0 in range(0, seq_len, tile_rows):
        rows = min(tile_rows, seq_len - t0)

        # ----------------------------------------------------
        # 1. Build positions for this tile, padded to TILE_SEQ
        # ----------------------------------------------------
        pos32 = (
            pos_offset + np.arange(t0, t0 + rows, dtype=np.float32)
        ).astype(np.float32)

        pos_pad = np.zeros(tile_rows, dtype=np.float32)
        pos_pad[:rows] = pos32

        # ----------------------------------------------------
        # 2. Compute radians32 on CPU
        #
        # radians32[i, k] = position[i] * inv_timescale[k]
        # Shape: [32, 32]
        # ----------------------------------------------------
        radians32 = (pos_pad[:, None] * inv_ts[None, :]).astype(np.float32)

        # ----------------------------------------------------
        # 3. Pack radians32 [32,32] -> radians64 [32,64] on CPU
        #
        # Equivalent to pack32to64_float32:
        #   first half  = radians32
        #   second half = radians32
        # ----------------------------------------------------
        radians64 = np.zeros((tile_rows, D), dtype=np.float32)
        radians64[:, :HALF] = radians32
        radians64[:, HALF:] = radians32

        # ----------------------------------------------------
        # 4. Compute sin/cos using packed 2D output
        #
        # sin_cos_packed[0:32, :]  = sin
        # sin_cos_packed[32:64, :] = cos
        # ----------------------------------------------------
        sin_cos_packed = np.zeros((2 * tile_rows, D), dtype=np.float32)
        sin_cos_mod(radians64, sin_cos_packed)

        sin64 = sin_cos_packed[:tile_rows, :]
        cos64 = sin_cos_packed[tile_rows:, :]

        # ----------------------------------------------------
        # 5. Rotate each head
        # ----------------------------------------------------
        for h in range(heads):
            x_tile = np.zeros((tile_rows, D), dtype=np.float32)
            x_tile[:rows, :] = packed[t0:t0 + rows, h * D:(h + 1) * D]

            # Split x into left/right halves.
            xL = np.zeros((tile_rows, HALF), dtype=np.float32)
            xR = np.zeros((tile_rows, HALF), dtype=np.float32)

            # Split sin/cos to first 32 columns.
            s = np.zeros((tile_rows, HALF), dtype=np.float32)
            c = np.zeros((tile_rows, HALF), dtype=np.float32)

            copyL_mod(x_tile, xL)
            copyR_mod(x_tile, xR)

            copyL_mod(sin64, s)
            copyL_mod(cos64, c)

            # yL = xL * c - xR * s
            tmp1 = np.zeros_like(xL)
            tmp2 = np.zeros_like(xL)
            yL = np.zeros_like(xL)

            mul32_mod(xL, c, tmp1)
            mul32_mod(xR, s, tmp2)
            sub32_mod(tmp1, tmp2, yL)

            # yR = xR * c + xL * s
            tmp3 = np.zeros_like(xL)
            tmp4 = np.zeros_like(xL)
            yR = np.zeros_like(xL)

            mul32_mod(xR, c, tmp3)
            mul32_mod(xL, s, tmp4)
            add32_mod(tmp3, tmp4, yR)

            # Join [yL, yR] -> y64
            y64 = np.zeros((tile_rows, D), dtype=np.float32)
            join_mod(yL, yR, y64)

            out[t0:t0 + rows, h * D:(h + 1) * D] = y64[:rows, :]

    return out


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    TEST_SEQ = 64
    Q_H = 15
    D = 64

    q = torch.randn(TEST_SEQ, Q_H * D, dtype=torch.float32).numpy()

    # Allo/AIE path
    y_allo = rope_apply_packed(
        q,
        heads=Q_H,
        head_dim=D,
        max_wavelength=10_000.0,
        pos_offset=0,
    )

    # Torch reference
    x4 = torch.from_numpy(q).view(TEST_SEQ, Q_H, D)

    k = torch.arange(D // 2, dtype=torch.float32)
    inv = (10_000.0 ** (-(2.0 / D) * k)).view(1, 1, -1)

    pos = torch.arange(TEST_SEQ, dtype=torch.float32).view(TEST_SEQ, 1, 1)
    rad = pos * inv

    s = torch.sin(rad)
    c = torch.cos(rad)

    xL = x4[:, :, :D // 2]
    xR = x4[:, :, D // 2:]

    y_ref = torch.cat(
        [
            xL * c - xR * s,
            xR * c + xL * s,
        ],
        dim=-1,
    ).reshape(TEST_SEQ, Q_H * D).numpy()

    np.testing.assert_allclose(
        y_allo,
        y_ref,
        rtol=1e-2,
        atol=1e-3,
    )

    print("✅ multi-head RoPE matches PyTorch (TEST_SEQ=64, Q_H=15, D=64)")