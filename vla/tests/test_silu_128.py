# Test: SiLU float32 tile [4][128] (SmolVLA action expert FFN_HID=2048, 2048/16=128)

import torch
import torch.nn as nn
import numpy as np
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate

SEQ_TILE = 4
FEATURE_TILE = 128
KERNEL_PATH = "../../cc/float/silu_128.cc"


def test_silu_128_float32():
    silu_ext = ExternalModule(
        top="silu_128_float32",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32

    @df.region()
    def top(input_x: Ty[SEQ_TILE, FEATURE_TILE],
            output_x: Ty[SEQ_TILE, FEATURE_TILE]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(local_in: Ty[SEQ_TILE, FEATURE_TILE] @ [S(0), S(1)],
                 local_out: Ty[SEQ_TILE, FEATURE_TILE] @ [S(0), S(1)]):
            silu_ext(local_in, local_out)

    torch.manual_seed(42)
    silu_model = nn.SiLU()

    input_tensor = torch.randn(SEQ_TILE, FEATURE_TILE, dtype=torch.float32)
    ref_out = silu_model(input_tensor)

    mod = df.build(top, target="aie", profile=True)
    input_np = input_tensor.numpy().astype(np.float32)
    output_allo = np.zeros((SEQ_TILE, FEATURE_TILE), dtype=np.float32)
    mod(input_np, output_allo)

    np.testing.assert_allclose(output_allo, ref_out.detach().numpy(), rtol=1e-2, atol=1e-3)
    print(f"PASSED: SiLU float32 tile [{SEQ_TILE},{FEATURE_TILE}]")


if __name__ == "__main__":
    test_silu_128_float32()
