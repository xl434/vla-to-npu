# Test: SiLU bf16 tile [4][256] - diagnostic to check if 128 is the issue

import torch
import torch.nn as nn
import numpy as np
import ml_dtypes
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import bfloat16
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard

SEQ_TILE = 4
FEATURE_TILE = 256
KERNEL_PATH = "../../cc/bf16/silu_256_bf16.cc"


def test_silu_256_bf16():
    silu_ext = ExternalModule(
        top="silu_256_bf16",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16

    @df.region()
    def top(input_x: Ty[SEQ_TILE, FEATURE_TILE],
            output_x: Ty[SEQ_TILE, FEATURE_TILE]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(local_in: Ty[SEQ_TILE, FEATURE_TILE] @ [S(0), S(1)],
                 local_out: Ty[SEQ_TILE, FEATURE_TILE] @ [S(0), S(1)]):
            silu_ext(local_in, local_out)

    torch.manual_seed(42)
    silu_model = nn.SiLU()

    input_tensor = torch.randn(SEQ_TILE, FEATURE_TILE, dtype=torch.bfloat16)
    ref_out = silu_model(input_tensor)
    ref_np = ref_out.view(torch.int16).numpy().view(ml_dtypes.bfloat16).astype(np.float32)

    mod = df.build(top, target="aie", profile=True)
    input_np = input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    output_allo = np.zeros((SEQ_TILE, FEATURE_TILE), dtype=ml_dtypes.bfloat16)
    mod(input_np, output_allo)

    np.testing.assert_allclose(output_allo.astype(np.float32), ref_np, rtol=1e-1, atol=5e-2)
    print(f"PASSED: SiLU bf16 tile [{SEQ_TILE},{FEATURE_TILE}]")


if __name__ == "__main__":
    test_silu_256_bf16()
