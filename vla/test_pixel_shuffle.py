import os
import pytest
import torch
import torch.nn as nn
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.backend.aie import is_available


torch.manual_seed(0)
np.random.seed(0)

S = Layout.Shard
R = Layout.Replicate

Ty=float32

linear_A_layout = [S(0), R]
linear_C_layout = [R, S(0)]

@df.region()
def copy(A: Ty[16, 768], C: Ty[4, 768*4]):
    @df.kernel(mapping=[4], args=[A, C])
    def mod(
        local_A: Ty[16, 768] @ linear_A_layout,
        local_C: Ty[4, 768*4] @ linear_C_layout,
    ):
        local_C[:,:] = local_A[:,:]

copy_mod = df.build(
    copy, target="aie", project="copy.prj"
)

A = np.random.rand(16, 768).astype(np.float32)
C = np.zeros((4, 768*4), dtype=np.float32)
copy_mod(A, C)
print("input:", A)
print("Copy output shape:", C.shape)
print("Copy output:", C)

# Verify correctness
# Interleaved reshape: row i of output = [A[i,:], A[i+4,:], A[i+8,:], A[i+12,:]]
expected = A.reshape(4, 4, 768).transpose(1, 0, 2).reshape(4, 768*4)
np.testing.assert_allclose(C, expected, rtol=1e-5)
print("Interleaved reshape verified correctly!")
