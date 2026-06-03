"""Build flash attention xclbin for ViT (SEQ=1024, HEAD_DIM=64).

Uses standard allo2 32x32 tile kernels (same as the working allo2 attention.py example).
Q_TILE=SEQ_LEN=1024: one FA kernel call per head (12 calls total vs 192 before).

Required env vars for MLIR-AIE to correctly fold the large logical core count
onto 4 physical AIE compute tiles (same env as allo2/examples/aie/attention.py).

Run from vla/ directory:
    conda run -p /opt/anaconda3/envs/allo-base python3 vision_block/flash_attn_vit.py

Produces: vision_block/flash_attn_vit.prj/build/final.xclbin + insts.txt
"""

import os
import sys
import allo.dataflow as df

# Standard 32x32 tile kernels from allo2 library (attn_out.cc, softmax_bf16.cc, mm.cc)
os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = "/home/xl434/allo2/allo/allo/library/aie/kernels/"

# Required MLIR-AIE compilation flags (from allo2/examples/aie/attention.py)
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
os.environ["COALESCE_MORE"] = "1"
os.environ["FORCE_UNROLL_INDEX"] = "0"

from allo.library.aie.modules.flash_attn import FA

SEQ_LEN  = 1024
HEAD_DIM = 64
Q_TILE   = SEQ_LEN  # process all 1024 Q rows per call → 12 calls total (one per head)
Q_CHUNK  = 32       # standard tile size matching attn_out.cc [32][64] signature
KV_CHUNK = 32       # standard tile size matching softmax_bf16.cc [32][32] signature

flash_attn_kernel, flash_attn_mp = FA(SEQ_LEN, HEAD_DIM, Q_TILE, Q_CHUNK, KV_CHUNK)

mod = df.build(
    flash_attn_kernel,
    target="aie",
    project="vision_block/flash_attn_vit.prj",
    mapping_primitives=flash_attn_mp,
)

print("flash_attn_vit build complete")
print(f"  xclbin: vision_block/flash_attn_vit.prj/build/final.xclbin")
