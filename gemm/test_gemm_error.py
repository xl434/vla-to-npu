from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as ml_bfloat16
import allo.dataflow as df
from allo.library.aie.modules.gemm import GEMM
import numpy as np

np.random.seed(0)

SEQ = 64
TEXT = 960
K = 64

def build_gemm():

    M = SEQ
    N = TEXT

    Pm = M // 64
    Pn = N // 64
    Pk = K // 64

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

if __name__ == "__main__":
    factor = 192
    g = build_gemm()
    x = np.random.randn(SEQ, K*factor).astype(ml_bfloat16)
    y = np.random.randn(K*factor, TEXT).astype(ml_bfloat16)
    z = np.zeros((64, TEXT), dtype=ml_bfloat16)

    exp = x @ y
    for i in range(factor):
        tmp = np.zeros((64, TEXT), dtype=ml_bfloat16)
        g(x[:, i*K:(i+1)*K], y[i*K:(i+1)*K, :], tmp)
        z += tmp
    
    np.testing.assert_allclose(z.astype(np.float32), exp.astype(np.float32), rtol=1)