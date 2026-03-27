# Test: SiLU bf16 tile [4][128] (SmolVLA action expert FFN_HID=2048, 2048/16=128)
# Note: Kernel uses [4][256] buffer to avoid 1024-byte DMA alignment issue,
#       but only processes the first 128 columns.

import torch
import torch.nn as nn
import numpy as np
import ml_dtypes
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import bfloat16
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate

SEQ_TILE = 4
FEATURE_TILE = 128
PADDED_TILE = 256  # Padded to avoid 1024-byte (2^10) DMA edge case
KERNEL_PATH = "../../cc/bf16/silu_128_bf16.cc"


def test_silu_128_bf16():
    silu_ext = ExternalModule(
        top="silu_128_bf16",
        impl_path=KERNEL_PATH,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16

    @df.region()
    def top(input_x: Ty[SEQ_TILE, PADDED_TILE],
            output_x: Ty[SEQ_TILE, PADDED_TILE]):
        @df.kernel(mapping=[1, 1], args=[input_x, output_x])
        def core(local_in: Ty[SEQ_TILE, PADDED_TILE] @ [S(0), S(1)],
                 local_out: Ty[SEQ_TILE, PADDED_TILE] @ [S(0), S(1)]):
            silu_ext(local_in, local_out)

    torch.manual_seed(42)
    silu_model = nn.SiLU()

    input_tensor = torch.randn(SEQ_TILE, FEATURE_TILE, dtype=torch.bfloat16)
    ref_out = silu_model(input_tensor)
    ref_np = ref_out.view(torch.int16).numpy().view(ml_dtypes.bfloat16).astype(np.float32)

    mod = df.build(top, target="aie", profile=True)
    # Pad input to [4, 256], zeros in cols 128-255
    input_padded = torch.zeros(SEQ_TILE, PADDED_TILE, dtype=torch.bfloat16)
    input_padded[:, :FEATURE_TILE] = input_tensor
    input_np = input_padded.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    output_allo = np.zeros((SEQ_TILE, PADDED_TILE), dtype=ml_dtypes.bfloat16)
    mod(input_np, output_allo)

    # Only compare first 128 columns
    output_trimmed = output_allo[:, :FEATURE_TILE].astype(np.float32)
    np.testing.assert_allclose(output_trimmed, ref_np, rtol=1e-1, atol=5e-2)
    print(f"PASSED: SiLU bf16 tile [{SEQ_TILE},{FEATURE_TILE}] (padded to {PADDED_TILE})")


if __name__ == "__main__":
    test_silu_128_bf16()
