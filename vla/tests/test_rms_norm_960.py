# Test: RMSNorm float32 with HIDDEN=960 (SmolVLA text encoder)

import torch
import torch.nn as nn
import numpy as np
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate

SEQ_TILE = 2
HIDDEN = 960
KERNEL_PATH = "../../cc/float/rms_norm_960.cc"

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        norm = x.norm(dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * weight


def test_rms_norm_960_float32():
    norm_ext = ExternalModule(
        top="rms_norm_960",
        impl_path=KERNEL_PATH,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    M, N = SEQ_TILE, HIDDEN

    @df.region()
    def top(A: Ty[M, N], B: Ty[N], C: Ty[M, N]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M, N] @ [S(0), R],
                 local_B: Ty[N] @ [R],
                 local_C: Ty[M, N] @ [S(0), R]):
            norm_ext(local_A, local_B, local_C)

    torch.manual_seed(42)
    input_tensor = torch.randn(M, N, dtype=torch.float32)
    weight = torch.randn(N, dtype=torch.float32)

    rms_norm = RMSNorm()
    ref_out = rms_norm(input_tensor, weight)

    input_np = input_tensor.numpy().astype(np.float32)
    weight_np = weight.numpy().astype(np.float32)
    output_allo = np.zeros((M, N), dtype=np.float32)

    mod = df.build(top, target="aie", profile=True)
    mod(input_np, weight_np, output_allo)

    np.testing.assert_allclose(output_allo, ref_out.detach().numpy(), rtol=1e-3, atol=1e-4)
    print(f"PASSED: RMSNorm float32 HIDDEN={HIDDEN} (tile [{M},{N}])")


if __name__ == "__main__":
    test_rms_norm_960_float32()
