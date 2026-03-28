from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as ml_bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_LIB_PATH = "../cc/"
add = ExternalModule(
    top="add",
    impl_path=KERNEL_LIB_PATH + "add_32_32.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

S = Layout.Shard
R = Layout.Replicate
Ty = bfloat16
split = [R, R]

M = 32
N = 32
@df.region()
def add_kernel(x: Ty[M, N], y: Ty[M, N], out: Ty[M, N]):
    @df.kernel(mapping=[1, 1], args=[x, y, out])
    def core(
        local_x: Ty[M, N] @ split,
        local_y: Ty[M, N] @ split,
        local_output: Ty[M, N] @ split,
    ):
        add(local_x, local_y, local_output)

add_mod = df.build(add_kernel, target="aie", project="add.prj")

if __name__ == "__main__":
    x = np.random.randn(M, N).astype(ml_bfloat16)
    y = np.random.randn(M, N).astype(ml_bfloat16)
    z = np.zeros((M, N)).astype(ml_bfloat16)

    add_mod(x, y, z)
    exp = x + y

    np.testing.assert_allclose(z.astype(np.float32), exp.astype(np.float32), rtol=1e-1)
