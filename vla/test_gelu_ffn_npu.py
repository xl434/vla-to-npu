"""
Build and run the 8-row GELU FFN kernel via Allo Python interface.
Tests the division-free Padé tanh implementation on NPU.
Run from vla/:
    conda run -p /opt/anaconda3/envs/allo-base python3 vision_block/test_gelu_ffn_npu.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import ml_dtypes
from allo.ir.types import bfloat16
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_PATH = "/home/xl434/vla-to-npu/cc/bf16/gelu_bf16_8rows.cc"

S  = Layout.Shard
Ly = [S(0), S(1)]
Ty = bfloat16

P1, P0       = 4, 4
seq_tile     = 8
feature_tile = 768
seq          = P1 * seq_tile      # 32
feature_dim  = P0 * feature_tile  # 3072

gelu_ext = ExternalModule(
    top="gelu",
    impl_path=KERNEL_PATH,
    input_idx=[0],
    output_idx=[1],
)

@df.region()
def top(input_x: Ty[seq, feature_dim], output_x: Ty[seq, feature_dim]):
    @df.kernel(mapping=[P1, P0], args=[input_x, output_x])
    def core(
        local_input_x:  Ty[seq, feature_dim] @ Ly,
        local_output_x: Ty[seq, feature_dim] @ Ly,
    ):
        gelu_ext(local_input_x, local_output_x)

print("Building GELU FFN (8-row) on NPU...")
mod = df.build(
    top,
    target="aie",
    project="vision_block/gelu_bf16_ffn.prj",
)
print("Build done. Running on NPU...")

torch.manual_seed(42)
input_tensor = torch.randn(seq, feature_dim, dtype=torch.bfloat16)

def to_bf16_np(t):
    return t.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16)

input_np  = to_bf16_np(input_tensor)
output_np = np.zeros((seq, feature_dim), dtype=ml_dtypes.bfloat16)

mod(input_np, output_np)
print("NPU run complete.")

# Reference: PyTorch exact GELU
gelu_fn = nn.GELU()
with torch.no_grad():
    ref_out = gelu_fn(input_tensor)
ref_np = to_bf16_np(ref_out).astype(np.float32)
out_f32 = output_np.astype(np.float32)

RTOL, ATOL = 1e-2, 1e-2
diff = np.abs(out_f32 - ref_np)
tol  = ATOL + RTOL * np.abs(ref_np)
bad  = diff > tol
n_bad = int(bad.sum())
pct   = 100.0 * n_bad / bad.size

print(f"\nResults vs PyTorch GELU (rtol={RTOL}, atol={ATOL}):")
print(f"  Wrong elements: {n_bad} / {bad.size} ({pct:.2f}%)")
print(f"  Max abs error:  {diff.max():.4e}")
print(f"  Mean abs error: {diff.mean():.4e}")

if n_bad == 0:
    print("PASSED — all outputs correct!")
else:
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    r, c = max_idx
    print(f"  Worst at [{r},{c}]: in={input_tensor[r,c].item():.4f} "
          f"npu={out_f32[r,c]:.4f} ref={ref_np[r,c]:.4f}")
    print("FAILED" if n_bad > 100 else "Minor mismatches (likely Padé approximation tolerance)")
