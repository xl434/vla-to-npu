# Build vectorized GELU for FFN hidden dim [32, 3072] per dispatch.
#
# Kernel:  cc/bf16_vla/gelu_bf16_8rows.cc — vectorized Padé tanh (aie::inv), [8][768] per core
# Mapping: [4, 4] = 16 cores (max on Phoenix NPU 4×4 grid)
#          → 4×8=32 rows, 4×768=3072 cols per dispatch
#          → 32 dispatches for [1024, 3072]
#
# Main speedup over scalar LUT: ~16× per dispatch (474ms → ~30ms for full [1024,3072])
#
# Run from vla/:
#   conda run -p /opt/anaconda3/envs/allo-base python3 vision_block/gelu_bf16_ffn.py

from allo.ir.types import bfloat16
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_PATH = "/home/xl434/vla-to-npu/cc/bf16/gelu_bf16.cc"

S  = Layout.Shard
Ly = [S(0), S(1)]
Ty = bfloat16

P1, P0         = 4, 4      # 16 cores (Phoenix NPU 4x4 grid max)
seq_tile       = 4          # rows per core (gelu_bf16.cc: [4][768] per core)
feature_tile   = 768        # cols per core

seq         = P1 * seq_tile      # 16
feature_dim = P0 * feature_tile  # 3072

gelu_ext = ExternalModule(
    top="gelu",
    impl_path=KERNEL_PATH,
    input_idx=[0],
    output_idx=[1],
)

@df.region()
def top(input_x: Ty[seq, feature_dim], output_x: Ty[seq, feature_dim]):
    @df.kernel(mapping=[P1, P0], args=[input_x, output_x])
    def core(
        local_input_x:  Ty[seq, feature_dim] @ Ly,
        local_output_x: Ty[seq, feature_dim] @ Ly,
    ):
        gelu_ext(local_input_x, local_output_x)

mod = df.build(
    top,
    target="aie",
    project="vision_block/gelu_bf16_ffn.prj",
)

print("Built gelu_bf16_ffn.prj successfully (P1=8, P0=4, 32 cores, tile [32,3072], reduced dispatches 64→32).")
