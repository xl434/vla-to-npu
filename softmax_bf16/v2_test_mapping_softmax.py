import os
import torch
import torch.nn.functional as F
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

np.random.seed(42)
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"



# ===============================================================================
# Model Configuration
# ===============================================================================

def test_mapping_softmax_4_1024():
    N = 1024
    P0_tile = 4 ## this match to the kernel row size 

    KERNEL_LIB_PATH = "../cc/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile  # 256
    SOFTMAX_Ly = Layout("S0R")

    softmax = ExternalModule(
        top="softmax_bf16",
        impl_path=KERNEL_LIB_PATH + "v2_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    def gen_softmax_primitives():
        SOFTMAX_ROW = 16    # number of physical tiles
        primitives = []
        for row in range(SOFTMAX_ROW):
            if SOFTMAX_P0 // SOFTMAX_ROW > 1:       # 16
                primitives.append(
                    (
                        "bundle",
                        [
                            f"core_{SOFTMAX_ROW*i_+row}"
                            for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                        ],
                    )
                )
        return primitives

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[N, N] @ SOFTMAX_Ly,
            output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    # ---------------------------------------------------------
    # Generate inputs
    # ---------------------------------------------------------
    input_tensor = np.random.randn(N, N).astype(np_bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    # Run Allo softmax
    softmax_mod(input_tensor, allo_out)

    # ---------------------------------------------------------
    # Torch reference
    # ---------------------------------------------------------
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    torch_out_fp32 = F.softmax(torch_in.float(), dim=-1)
    torch_out_np = torch_out_fp32.cpu().numpy()


    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        torch_out_np.astype(np.float32),
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")

def test_mapping_softmax_8_512():
    N = 512
    P0_tile = 8

    KERNEL_LIB_PATH = "../cc/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = Layout("S0R")

    softmax = ExternalModule(
        top="softmax_bf16_8_512",
        impl_path=KERNEL_LIB_PATH + "v2_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    def gen_softmax_primitives():
        SOFTMAX_ROW = 16
        primitives = []
        for row in range(SOFTMAX_ROW):
            if SOFTMAX_P0 // SOFTMAX_ROW > 1:
                primitives.append(
                    (
                        "bundle",
                        [
                            f"core_{SOFTMAX_ROW*i_+row}"
                            for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                        ],
                    )
                )
        return primitives

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[N, N] @ SOFTMAX_Ly,
            output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    # ---------------------------------------------------------
    # Generate inputs
    # ---------------------------------------------------------
    input_tensor = np.random.randn(N, N).astype(np_bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    # Run Allo softmax
    softmax_mod(input_tensor, allo_out)

    # ---------------------------------------------------------
    # Torch reference
    # ---------------------------------------------------------
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    torch_out_fp32 = F.softmax(torch_in.float(), dim=-1)
    torch_out_np = torch_out_fp32.cpu().numpy()


    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        torch_out_np.astype(np.float32),
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")

def test_mapping_softmax_16_256():
    N = 256
    P0_tile = 16

    KERNEL_LIB_PATH = "../cc/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = Layout("S0R")

    softmax = ExternalModule(
        top="softmax_bf16_16_256",
        impl_path=KERNEL_LIB_PATH + "v2_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    def gen_softmax_primitives():
        SOFTMAX_ROW = 16
        primitives = []
        for row in range(SOFTMAX_ROW):
            if SOFTMAX_P0 // SOFTMAX_ROW > 1:
                primitives.append(
                    (
                        "bundle",
                        [
                            f"core_{SOFTMAX_ROW*i_+row}"
                            for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                        ],
                    )
                )
        return primitives

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[N, N] @ SOFTMAX_Ly,
            output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    # ---------------------------------------------------------
    # Generate inputs
    # ---------------------------------------------------------
    input_tensor = np.random.randn(N, N).astype(np_bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    # Run Allo softmax
    softmax_mod(input_tensor, allo_out)

    # ---------------------------------------------------------
    # Torch reference
    # ---------------------------------------------------------
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    torch_out_fp32 = F.softmax(torch_in.float(), dim=-1)
    torch_out_np = torch_out_fp32.cpu().numpy()


    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        torch_out_np.astype(np.float32),
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")

def test_mapping_softmax_32_128():
    N = 128
    P0_tile = 32

    KERNEL_LIB_PATH = "../cc/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = Layout("S0R")

    softmax = ExternalModule(
        top="softmax_bf16_32_128",
        impl_path=KERNEL_LIB_PATH + "v2_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    def gen_softmax_primitives():
        SOFTMAX_ROW = 16
        primitives = []
        for row in range(SOFTMAX_ROW):
            if SOFTMAX_P0 // SOFTMAX_ROW > 1:
                primitives.append(
                    (
                        "bundle",
                        [
                            f"core_{SOFTMAX_ROW*i_+row}"
                            for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                        ],
                    )
                )
        return primitives

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[N, N] @ SOFTMAX_Ly,
            output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    # ---------------------------------------------------------
    # Generate inputs
    # ---------------------------------------------------------
    input_tensor = np.random.randn(N, N).astype(np_bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    # Run Allo softmax
    softmax_mod(input_tensor, allo_out)

    # ---------------------------------------------------------
    # Torch reference
    # ---------------------------------------------------------
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    torch_out_fp32 = F.softmax(torch_in.float(), dim=-1)
    torch_out_np = torch_out_fp32.cpu().numpy()


    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        torch_out_np.astype(np.float32),
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")

def test_mapping_softmax_64_64():
    N = 64


    KERNEL_LIB_PATH = "../cc/"
    Ty = bfloat16

    P0_tile = 64
    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = Layout("S0R")

    softmax = ExternalModule(
        top="softmax_bf16_64_64",
        impl_path=KERNEL_LIB_PATH + "v2_softmax_bf16.cc",
        input_idx=[0],
        output_idx=[1],
    )

    def gen_softmax_primitives():
        SOFTMAX_ROW = 1
        primitives = []
        for row in range(SOFTMAX_ROW):
            if SOFTMAX_P0 // SOFTMAX_ROW > 1:
                primitives.append(
                    (
                        "bundle",
                        [
                            f"core_{SOFTMAX_ROW*i_+row}"
                            for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                        ],
                    )
                )
        return primitives

    @df.region()
    def softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0])
        def core(
            input_x: Ty[N, N] @ SOFTMAX_Ly,
            output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(input_x, output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    # ---------------------------------------------------------
    # Generate inputs
    # ---------------------------------------------------------
    input_tensor = np.random.randn(N, N).astype(np_bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    # Run Allo softmax
    softmax_mod(input_tensor, allo_out)

    # ---------------------------------------------------------
    # Torch reference
    # ---------------------------------------------------------
    torch_in = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
    torch_out_fp32 = F.softmax(torch_in.float(), dim=-1)
    torch_out_np = torch_out_fp32.cpu().numpy()


    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        torch_out_np.astype(np.float32),
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


if __name__ == "__main__":
    test_mapping_softmax_4_1024()
    # test_mapping_softmax_8_512()
    # test_mapping_softmax_16_256()
    # test_mapping_softmax_32_128()
    # test_mapping_softmax_64_64()
