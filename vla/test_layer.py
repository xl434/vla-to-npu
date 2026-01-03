import os
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule

# ----------------------------
# Config
# ----------------------------
Ty = float32
SEQ = 64
EMBD = 768

# Use the SAME kernel path pattern as the GPT2 example
# KERNEL_LIB_PATH = os.path.join(
#     os.path.dirname(__file__), "../../allo2/allo/library/aie/kernels/"
# )
KERNEL_LIB_PATH="../../allo2/allo/allo/library/aie/"
# External module: layer norm (no bias inside kernel; bias added separately)
norm = ExternalModule(
    top="layer_norm",
    impl_path=KERNEL_LIB_PATH + "layer_norm.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0

norm_io_layout = Layout("S0R")
norm_arg_layout = Layout("R")

@df.region()
def layer_norm_kernel(
    input_x: Ty[NORM_SEQ_TILE, EMBD],
    weight: Ty[EMBD],
    bias: Ty[EMBD],
    output_x: Ty[NORM_SEQ_TILE, EMBD],
):
    pipe: Stream[Ty[NORM_TILE, EMBD], 1][NORM_P0]

    # NOTE: args=[input_x, weight] is how GPT2 code does it
    @df.kernel(mapping=[NORM_P0], args=[input_x, weight])
    def norm_no_bias(
        local_input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_weight: Ty[EMBD] @ norm_arg_layout,
    ):
        pi = df.get_pid()
        tmp: Ty[NORM_TILE, EMBD] = 0
        norm(local_input_x, local_weight, tmp)
        pipe[pi].put(tmp)

    # NOTE: args=[bias, output_x] is how GPT2 code does it
    @df.kernel(mapping=[NORM_P0], args=[bias, output_x])
    def norm_add_bias(
        local_bias: Ty[EMBD] @ norm_arg_layout,
        local_output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        pi = df.get_pid()
        data = pipe[pi].get()
        # This relies on broadcasting support: (NORM_TILE,EMBD) + (EMBD,)
        local_output_x[:, :] = allo.add(data, local_bias)

# Build once
layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm.prj")

def layernorm_full_seq(input_x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    input_x: (SEQ, EMBD) float32
    weight:  (EMBD,) float32
    bias:    (EMBD,) float32
    returns: (SEQ, EMBD) float32
    """
    assert input_x.shape == (SEQ, EMBD)
    assert weight.shape == (EMBD,)
    assert bias.shape == (EMBD,)

    out = np.empty((SEQ, EMBD), dtype=np.float32)
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_in = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        tile_out = out[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        layer_norm_mod(tile_in, weight, bias, tile_out)
    return out
