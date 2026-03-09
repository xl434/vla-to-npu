import time
import torch
import torch.nn as nn
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available

torch.manual_seed(100)
np.random.seed(100)

S = Layout.Shard
R = Layout.Replicate

KERNEL_LIB_PATH = "../cc/"

Ty = float32

conv = ExternalModule(
    top="conv",
    impl_path=KERNEL_LIB_PATH + "conv.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

split_format = [S(0), R]
kernel_format = [R, R]

@df.region()
def conv_kernel(input: Ty[64, 16], kernel: Ty[16, 16], output: Ty[4, 1]):
    @df.kernel(mapping=[4, 1], args=[input, kernel, output])
    def core(
        local_input: Ty[64, 16] @ split_format,
        local_kernel: Ty[16, 16] @ kernel_format,
        local_output: Ty[4, 1] @ split_format,
    ):
        conv(local_input, local_kernel, local_output)

acc_format = [S(0), S(1)]

@df.region()
def linear_accumulate_kernel(A: Ty[32, 32], B: Ty[32, 32], C: Ty[32, 32]):
    @df.kernel(mapping=[2, 4], args=[A, B, C])
    def core(
        local_A: Ty[32, 32] @ acc_format,
        local_B: Ty[32, 32] @ acc_format,
        local_C: Ty[32, 32] @ acc_format,
    ):
        local_C[:, :] = allo.add(local_A, local_B)

linear_in_layout = [S(0), R]
linear_out_layout = [R, S(0)]

@df.region()
def copy(A: Ty[4, 4], C: Ty[1, 16]):
    @df.kernel(mapping=[4], args=[A,C])
    def mod(
        local_A: Ty[4, 4] @ linear_in_layout,
        local_C: Ty[1, 16] @ linear_out_layout,
    ):
        local_C[:,:] = local_A[:,:]

conv_mod = df.build(conv_kernel, target="aie", project="conv.prj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie", project="linear_accumulate.prj")
copy_mod = df.build(copy, target="aie", project="copy.prj")

def conv2d(A, B, C):
    embd = np.zeros((4, 1, 1)).astype(np.float32)
    out = np.zeros((1, 1, 1)).astype(np.float32)
    for i in range(1):
        A_tile = A[0, :, :]
        B_tile = B[i, 0, :, :]
        conv_mod(A_tile, B_tile, embd[:,:,i])
    C[:, :] = embd.reshape(4, 1)
    
if __name__ == "__main__":
    print("=== Testing conv2d_patch_embed ===")

    IN_CHANNELS  = 3
    IN_H         = 64
    IN_W         = 16
    PATCH_SIZE   = 16
    EMBED_DIM    = 1
    NUM_PATCHES  = (IN_H // PATCH_SIZE) * (IN_W // PATCH_SIZE)  

    input_np = np.random.randn(IN_CHANNELS, IN_H, IN_W).astype(np.float32)

    kernel_np = np.random.randn(EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE).astype(np.float32)

    output_np = np.zeros((NUM_PATCHES, EMBED_DIM), dtype=np.float32)

    print(input_np)
    print(kernel_np)

    for m in range(4):
        sum = 0
        for k in range(3):
            for i in range(16):
                for j in range(16):
                    sum += input_np[k,i+m*16,j] * kernel_np[0,k,i,j]
        print(sum)
    conv2d(input_np, kernel_np, output_np)

    # Reference: PyTorch Conv2d with same weights
    # PyTorch expects input as [batch, channels, H, W]
    input_torch = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0)

    conv = nn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=EMBED_DIM,
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        padding=0,
    )

    # Set conv weights to our kernel
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.tensor(kernel_np, dtype=torch.float32))
        conv.bias   = nn.Parameter(torch.zeros(EMBED_DIM))  # no bias

    # Run PyTorch conv
    with torch.no_grad():
        out_torch = conv(input_torch) 

    # Reshape PyTorch output to [1024, 768] to match AIE output
    # [1, 768, 32, 32] -> [1, 768, 1024] -> [1024, 768]
    out_torch = out_torch.squeeze(0)         # [768, 32, 32]
    out_torch = out_torch.flatten(1)         # [768, 1024]
    out_torch = out_torch.transpose(0, 1)    # [1024, 768]
    expected  = out_torch.numpy().astype(np.float32)

    # Check shapes
    assert output_np.shape == expected.shape, \
        f"Shape mismatch: got {output_np.shape}, expected {expected.shape}"
    print(f"Shape check passed: {output_np.shape}")

    # Check values
    try:
        np.testing.assert_allclose(output_np, expected, rtol=1e-5, atol=1e-5)
        print("Value check passed: outputs match within tolerance")
    except AssertionError as e:
        print(f"Value check FAILED: {e}")

    print(output_np, expected)
    # Error stats
    max_err  = np.max(np.abs(output_np - expected))
    mean_err = np.mean(np.abs(output_np - expected))
    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")

    print("=== All tests passed ===")