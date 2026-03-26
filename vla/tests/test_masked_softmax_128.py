# Test: Masked (causal) softmax float32 for 128-column rows (SmolVLA text encoder)

import torch
import torch.nn.functional as F
import numpy as np
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32, int32
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate

TILE_ROWS = 8
SEQ_COLS = 128
KERNEL_PATH = "../../cc/float/masked_softmax_128.cc"


def test_masked_softmax_128():
    softmax_ext = ExternalModule(
        top="masked_softmax_128_float32",
        impl_path=KERNEL_PATH,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Tint = int32

    @df.region()
    def top(scores: Ty[TILE_ROWS, SEQ_COLS],
            row_start: Tint[1],
            weights: Ty[TILE_ROWS, SEQ_COLS]):
        @df.kernel(mapping=[1, 1], args=[scores, row_start, weights])
        def core(local_scores: Ty[TILE_ROWS, SEQ_COLS] @ [S(0), S(1)],
                 local_row: Tint[1] @ [R],
                 local_weights: Ty[TILE_ROWS, SEQ_COLS] @ [S(0), S(1)]):
            softmax_ext(local_scores, local_row, local_weights)

    torch.manual_seed(42)

    # Test with row_start=0 (first 32 rows of a 128x128 attention matrix)
    row_start = 0
    input_scores = torch.randn(TILE_ROWS, SEQ_COLS, dtype=torch.float32)

    # PyTorch reference: apply causal mask then softmax
    ref_scores = input_scores.clone()
    for r in range(TILE_ROWS):
        global_row = row_start + r
        for c in range(SEQ_COLS):
            if c > global_row:
                ref_scores[r, c] = float('-inf')
    ref_weights = F.softmax(ref_scores, dim=-1)

    # Allo / AIE
    mod = df.build(top, target="aie", profile=True)
    scores_np = input_scores.numpy().copy()
    row_np = np.array([row_start], dtype=np.int32)
    weights_np = np.zeros((TILE_ROWS, SEQ_COLS), dtype=np.float32)
    mod(scores_np, row_np, weights_np)

    np.testing.assert_allclose(weights_np, ref_weights.numpy(), rtol=1e-3, atol=1e-4)
    print(f"PASSED: Masked softmax 128-col (row_start={row_start})")

    # Test with row_start=96 (last 32 rows: rows 96-127 of 128x128)
    row_start2 = 96
    input_scores2 = torch.randn(TILE_ROWS, SEQ_COLS, dtype=torch.float32)
    ref_scores2 = input_scores2.clone()
    for r in range(TILE_ROWS):
        global_row = row_start2 + r
        for c in range(SEQ_COLS):
            if c > global_row:
                ref_scores2[r, c] = float('-inf')
    ref_weights2 = F.softmax(ref_scores2, dim=-1)

    scores_np2 = input_scores2.numpy().copy()
    row_np2 = np.array([row_start2], dtype=np.int32)
    weights_np2 = np.zeros((TILE_ROWS, SEQ_COLS), dtype=np.float32)
    mod(scores_np2, row_np2, weights_np2)

    np.testing.assert_allclose(weights_np2, ref_weights2.numpy(), rtol=1e-3, atol=1e-4)
    print(f"PASSED: Masked softmax 128-col (row_start={row_start2})")


if __name__ == "__main__":
    test_masked_softmax_128()
