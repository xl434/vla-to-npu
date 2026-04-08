import os
import time
import torch
import torch.nn.functional as F
import ml_dtypes
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

np.random.seed(42)
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

S = Layout.Shard
R = Layout.Replicate


def test_mapping_softmax_4_768():
    N = 768
    P0_tile = 4

    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile 
    SOFTMAX_Ly = [S(0), R]

    softmax = ExternalModule(
        top="softmax_bf16_4_768",
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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")

def test_mapping_softmax_4_1024():
    N = 1024
    P0_tile = 4

    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile  # 256
    SOFTMAX_Ly = [S(0), R]

    softmax = ExternalModule(
        top="softmax_bf16_4_1024",
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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


def test_mapping_softmax_8_512():
    N = 512
    P0_tile = 8

    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = [S(0), R]

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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


def test_mapping_softmax_16_256():
    N = 256
    P0_tile = 16

    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = [S(0), R]

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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


def test_mapping_softmax_32_128():
    N = 128
    P0_tile = 32

    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = [S(0), R]

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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


def test_mapping_softmax_64_64():
    N = 64
    KERNEL_LIB_PATH = "../cc/bf16_old/"
    Ty = bfloat16

    P0_tile = 64
    SOFTMAX_P0 = N // P0_tile
    SOFTMAX_Ly = [S(0), R]

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
    def softmax_kernel(input_x: Ty[N, N], output_x: Ty[N, N]):
        @df.kernel(mapping=[SOFTMAX_P0], args=[input_x, output_x])
        def core(
            local_input_x: Ty[N, N] @ SOFTMAX_Ly,
            local_output_x: Ty[N, N] @ SOFTMAX_Ly,
        ):
            softmax(local_input_x, local_output_x)

    softmax_mod = df.build(
        softmax_kernel,
        target="aie",
        project="softmax.prj",
        mapping_primitives=gen_softmax_primitives(),
        profile=True,
        warmup=20,
        num_iters=1000,
    )

    input_tensor = torch.randn(N, N, dtype=torch.bfloat16)

    # CPU Execution Time
    with torch.no_grad():
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)   # input data prep
        output_ref = F.softmax(
            torch.from_numpy(input_numpy_cpu.view(np.int16)).view(torch.bfloat16).float(),
            dim=-1,
        )  # compute
        ref_numpy = output_ref.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16).astype(np.float32)  # output retrieval
        end = time.perf_counter()

    cpu_time_us = (end - start) * 1_000_000

    input_np = np.asarray(input_tensor.float().cpu().numpy(), dtype=ml_dtypes.bfloat16)
    allo_out = np.zeros((N, N), dtype=np_bfloat16)

    softmax_mod(input_np, allo_out)

    print(f"CPU execution time: {cpu_time_us:.2f} us")
    np.testing.assert_allclose(
        allo_out.astype(np.float32),
        ref_numpy,
        atol=1e-1, rtol=1e-2,
    )
    print("Softmax comparison PASSED!")


if __name__ == "__main__":
    # test_mapping_softmax_4_768()
    test_mapping_softmax_4_1024()
    # test_mapping_softmax_8_512()
    # test_mapping_softmax_16_256()
    # test_mapping_softmax_32_128()
    # test_mapping_softmax_64_64()