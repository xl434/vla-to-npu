import time
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

KERNEL_LIB_PATH = "../cc/"

SEQ = 1024
EMBD = 768
SCALE_FACTOR = 4
NEW_SEQ = 64 # 1024 / 4 / 4
NEW_EMBD = 12288 # 768 * 4 * 4
TEXT = 960

# ===============================================================================
# Allo Version
# ===============================================================================

Ty = bfloat16

linear_in_layout = [S(0), R]
linear_out_layout = [R, S(0)]

@df.region()
def copy(A: Ty[4, EMBD], C: Ty[1, EMBD*4]):
    @df.kernel(mapping=[4], args=[A,C])
    def mod(
        local_A: Ty[4, EMBD] @ linear_in_layout,
        local_C: Ty[1, EMBD*4] @ linear_out_layout,
    ):
        local_C[:,:] = local_A[:,:]

K = 64

def build_gemm():

    M = NEW_SEQ
    N = TEXT

    Pm = 1
    Pn = N // 64
    Pk = 1

    top, mapping_primitives = GEMM(
        M,
        N,
        K,
        Pm,
        Pn,
        Pk,
        bfloat16,
        bfloat16,
    )

    mod = df.build(
        top,
        target="aie",
        project="connector_gemm.prj",
        mapping_primitives=mapping_primitives,
    )

    return mod

add = ExternalModule(
    top="add",
    impl_path=KERNEL_LIB_PATH + "add_64_64.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

linear = [R, R]

M = 64
N = 64
@df.region()
def add_kernel(x: Ty[M, N], y: Ty[M, N], out: Ty[M, N]):
    @df.kernel(mapping=[1, 1], args=[x, y, out])
    def core(
        local_x: Ty[M, N] @ linear,
        local_y: Ty[M, N] @ linear,
        local_output: Ty[M, N] @ linear,
    ):
        add(local_x, local_y, local_output)

add_mod = df.build(add_kernel, target="aie", project="add.prj")
copy_mod = df.build(copy, target="aie", project="copy.prj")

def fused_op(A, B, C):
    t0 = time.time()
    A_ = np.zeros((NEW_SEQ, NEW_EMBD), dtype=np_bfloat16)
    for i in range(NEW_SEQ):
        offset = (i // 8) * 128 + (i % 8) * 4
        for j in range(4):
            tile_A = A[ 
                offset + j * 32 : offset + j * 32 + 4,
                :
            ]
            tile_A_ = A_[
                i: i + 1,
                j * EMBD * 4 : (j + 1) * (EMBD * 4)
            ]
            copy_mod(tile_A, tile_A_) 
    
    t1 = time.time()
    gemm_mod = build_gemm()

    for i in range(0, NEW_EMBD, K):

        A_tile = A_[:, i:i+K]
        B_tile = B[i:i+K, :]

        C_tmp = np.zeros((NEW_SEQ, TEXT)).astype(np_bfloat16)

        gemm_mod(A_tile, B_tile, C_tmp)

        for j in range(0, TEXT, N):
            add_mod(C[:, j:j+N], C_tmp[:, j:j+N], C[:, j:j+N])


    t2 = time.time()

    print("pixel shuffle execution time: " + str(t1 - t0))
    print("matmul execution time: " + str(t2 - t1))
    print("total execution time: " + str(t2 - t0))

# Weight matrix is size [12288, 960]
def connector_block(x: np.ndarray, params: dict):
    out = np.zeros((NEW_SEQ, TEXT), dtype=np_bfloat16)
    fused_op(x, params["W"], out)
    return out

# ##############################################################
# TEST
# ##############################################################
if __name__ == "__main__":
    x = np.random.randn(SEQ, EMBD).astype(np_bfloat16)
    w = np.random.randn(EMBD*(SCALE_FACTOR**2), TEXT).astype(np_bfloat16)
    dict = {"W": w}
    out = connector_block(x, dict)

    # test python
    x = x.astype(np.float32).reshape(32, 32, 768).reshape(32, 8, 3072).transpose(1, 0, 2).reshape(8, 8, 12288).transpose(1, 0, 2).reshape(64, 12288)
    expected = x @ w.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), expected.astype(np.float32), rtol=1e-1)