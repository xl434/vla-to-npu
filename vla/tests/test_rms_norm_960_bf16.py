# Test: RMSNorm bf16 with HIDDEN=960 (SmolVLA text encoder)

import time
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
HIDDEN = 960
KERNEL_PATH = "../../cc/bf16/rms_norm_960_bf16.cc"

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        norm = x.norm(dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * weight


def test_rms_norm_960_bf16():
    norm_ext = ExternalModule(
        top="rms_norm_960_bf16",
        impl_path=KERNEL_PATH,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    M, N = SEQ_TILE, HIDDEN

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M, N] @ [S(0), R],
                 local_B: Ty[N] @ [R],
                 local_C: Ty[M, N] @ [S(0), R]):
            norm_ext(local_A, local_B, local_C)

    torch.manual_seed(42)
    input_tensor = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)

    rms_norm = RMSNorm()
    ref_out = rms_norm(input_tensor.float(), weight.float()).to(torch.bfloat16)

    input_np = np.asarray(input_tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
    weight_np = np.asarray(weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
    output_allo = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    mod = df.build(top, target="aie", profile=True)
    mod(input_np, weight_np, output_allo)

    ref_np = ref_out.view(torch.int16).numpy().view(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(output_allo.astype(np.float32), ref_np, rtol=2e-2, atol=1e-1)
    print(f"PASSED: RMSNorm bf16 HIDDEN={HIDDEN} (tile [{M},{N}])")


if __name__ == "__main__":
    test_rms_norm_960_bf16()
