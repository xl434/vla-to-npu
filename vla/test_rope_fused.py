import numpy as np
import torch
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie import ExternalModule

# Shapes (32-row tiles to fit AIE data memory with 3 double-buffered ports)
SEQ, HEAD_DIM = 32, 64
HEAD_DIM_HALF = HEAD_DIM // 2

# Layouts / dtypes
S = Layout.Shard
R = Layout.Replicate
Ty = float32
VecLy = [S(0)]
MatLy = [S(1), S(0)]

# Single fused kernel (3 ports: x, sin_cos, out)
KERNEL_LIB_PATH = "../cc/float/"

rope_fused_ext = ExternalModule(
    top="rope_fused_float32",
    impl_path=KERNEL_LIB_PATH + "rope_fused.cc",
    input_idx=[0, 1],   # x, sin_cos
    output_idx=[2],      # out
)

@df.region()
def rope_fused_region(x: Ty[SEQ, HEAD_DIM],
                      sin_cos: Ty[SEQ, HEAD_DIM],
                      out: Ty[SEQ, HEAD_DIM]):
    @df.kernel(mapping=[1, 1], args=[x, sin_cos, out])
    def core(local_x: Ty[SEQ, HEAD_DIM] @ MatLy,
             local_sc: Ty[SEQ, HEAD_DIM] @ MatLy,
             local_out: Ty[SEQ, HEAD_DIM] @ MatLy):
        rope_fused_ext(local_x, local_sc, local_out)

rope_mod = df.build(rope_fused_region, target="aie", project="rope/fused.prj")


# -------- Host helpers --------
def precompute_sin_cos(seq_len, head_dim, max_wavelength=10_000.0, pos_offset=0):
    """Precompute packed sin_cos[seq_len][head_dim] on host.
    Left half = sin(radians), right half = cos(radians)."""
    D = head_dim
    HALF = D // 2
    k = np.arange(HALF, dtype=np.float32)
    inv_ts = max_wavelength ** (-(2.0 / D) * k)  # [HALF]
    pos = (pos_offset + np.arange(seq_len, dtype=np.float32))  # [seq_len]
    radians = pos[:, None] * inv_ts[None, :]  # [seq_len, HALF]
    sin_cos = np.zeros((seq_len, D), dtype=np.float32)
    sin_cos[:, :HALF] = np.sin(radians)
    sin_cos[:, HALF:] = np.cos(radians)
    return sin_cos


def rope_apply_fused(packed, heads, head_dim=64, max_wavelength=10_000.0, pos_offset=0):
    """Apply RoPE to packed tensor of shape (SEQ, heads*head_dim).
    Uses a single fused AIE kernel per head tile."""
    seq_len, total_dim = packed.shape
    assert total_dim == heads * head_dim
    tile_rows = SEQ
    D = head_dim

    out = np.empty_like(packed, dtype=np.float32)

    for t0 in range(0, seq_len, tile_rows):
        rows = min(tile_rows, seq_len - t0)

        # Precompute sin/cos once per tile (shared across heads)
        sin_cos = precompute_sin_cos(tile_rows, D, max_wavelength, pos_offset + t0)

        for h in range(heads):
            x_tile = np.zeros((tile_rows, D), dtype=np.float32)
            x_tile[:rows, :] = packed[t0:t0 + rows, h * D:(h + 1) * D]

            out_tile = np.zeros((tile_rows, D), dtype=np.float32)
            sc_copy = sin_cos.copy()
            rope_mod(x_tile, sc_copy, out_tile)

            if np.isnan(out_tile).any():
                print(f"  NaN in out_tile! t0={t0} h={h}")
                print(f"    x_tile  has NaN: {np.isnan(x_tile).any()}")
                print(f"    sin_cos has NaN: {np.isnan(sin_cos).any()}")
                nan_r, nan_c = np.where(np.isnan(out_tile))
                print(f"    NaN positions: rows={np.unique(nan_r)}, cols={np.unique(nan_c)}")
                # Print row 0 and row 1 around col 14
                for row_idx in [0, 1]:
                    print(f"    out[{row_idx}][12:16] = {out_tile[row_idx, 12:16]}")
                    print(f"    out[{row_idx}][28:32] = {out_tile[row_idx, 28:32]}")
                    print(f"     sc[{row_idx}][12:16] = {sin_cos[row_idx, 12:16]}")
                    print(f"      x[{row_idx}][12:16] = {x_tile[row_idx, 12:16]}")
                    print(f"      x[{row_idx}][44:48] = {x_tile[row_idx, 44:48]}")

            out[t0:t0 + rows, h * D:(h + 1) * D] = out_tile[:rows, :]

    return out


# -------- CPU baseline --------
def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, SEQ_LEN, Q_H, D = 1, 64, 8, 64  # 2 tiles, 8 heads

    # Random input
    x_torch = torch.randn(B, SEQ_LEN, Q_H, D, dtype=torch.float32)
    positions = torch.arange(SEQ_LEN, dtype=torch.int32).unsqueeze(0)  # [1, L]

    # CPU baseline
    y_ref = apply_rope(x_torch, positions, max_wavelength=10_000)
    y_ref_np = y_ref.squeeze(0).reshape(SEQ_LEN, Q_H * D).numpy()

    # Fused AIE kernel
    q_packed = x_torch.squeeze(0).reshape(SEQ_LEN, Q_H * D).numpy()
    y_allo = rope_apply_fused(q_packed, heads=Q_H, head_dim=D,
                              max_wavelength=10_000.0, pos_offset=0)

    # Debug: check for NaN
    nan_mask = np.isnan(y_allo)
    if nan_mask.any():
        nan_rows, nan_cols = np.where(nan_mask)
        print(f"NaN count: {nan_mask.sum()} / {y_allo.size}")
        print(f"NaN rows (unique): {np.unique(nan_rows)}")
        print(f"NaN cols (unique): {np.unique(nan_cols)}")
        # Show first few NaN locations
        for i in range(min(5, len(nan_rows))):
            r, c = nan_rows[i], nan_cols[i]
            head = c // D
            dim = c % D
            print(f"  [{r},{c}] head={head} dim={dim}  ref={y_ref_np[r,c]:.6f}  allo={y_allo[r,c]}")
    else:
        np.testing.assert_allclose(y_allo, y_ref_np, rtol=1e-3, atol=1e-4)
        print(f"PASS: fused RoPE matches CPU baseline (B={B}, L={SEQ_LEN}, H={Q_H}, D={D})")
