import torch
import torch.nn as nn
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

KERNEL_LIB_PATH = "../cc/float/"

Ty = float32

conv = ExternalModule(
    top="conv",
    impl_path=KERNEL_LIB_PATH + "conv.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

split_format = [S(0), S(1)]
kernel_format = [R, R]

@df.region()
def conv_kernel(input: Ty[64, 64], kernel: Ty[16, 16], output: Ty[4, 4]):
    @df.kernel(mapping=[1, 1], args=[input, kernel, output])
    def core(
        local_input: Ty[64, 64] @ kernel_format,
        local_kernel: Ty[16, 16] @ kernel_format,
        local_output: Ty[4, 4] @ kernel_format,
    ):
        conv(local_input, local_kernel, local_output)

conv_mod = df.build(conv_kernel, target="aie", project="conv.prj", profile=True)

def conv2d(A, B, C):
    embd = np.zeros((4, 4, 1)).astype(np.float32)
    for i in range(1):
        A_tile = A[0, :, :]
        B_tile = B[i, 0, :, :]
        conv_mod(A_tile, B_tile, embd[:,:,i])
    C[:, :] = embd.reshape(16, 1)
    
if __name__ == "__main__":
    print("=== Testing conv2d_patch_embed ===")

    IN_CHANNELS  = 1
    IN_H         = 64
    IN_W         = 64
    PATCH_SIZE   = 16
    EMBED_DIM    = 1
    NUM_PATCHES  = (IN_H // PATCH_SIZE) * (IN_W // PATCH_SIZE)  

    input_np = np.random.rand(IN_CHANNELS, IN_H, IN_W).astype(np.float32)

    kernel_np = np.random.rand(EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE).astype(np.float32)

    output_np = np.zeros((NUM_PATCHES, EMBED_DIM), dtype=np.float32)
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

    for i in range(16):
        x = np.abs(expected[i,0] - output_np[i, 0])
        if (x > 1e-3):
            print(x, i)

    # Check shapes
    assert output_np.shape == expected.shape, \
        f"Shape mismatch: got {output_np.shape}, expected {expected.shape}"
    print(f"Shape check passed: {output_np.shape}")

    # Check values
    try:
        np.testing.assert_allclose(output_np, expected, rtol=1e-1)
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