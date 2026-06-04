"""Build flash attention xclbin for action expert cross-attention.

Cross-attention: action expert Q (SEQ=32) attends to text encoder K/V (SEQ=128).

Parameters:
  SEQ_LEN   = 128  (KV side — text encoder)
  HEAD_DIM  = 64
  Q_tile    = 32   (all 32 action expert Q rows in one call → 1 call per head, 15 total)
  q_chunk   = 32   = Q_tile → iteration=1
  kv_chunk  = 32   → 4 KV positions (128/32=4)

No causal mask needed (cross-attention).

Run from vla/ directory:
    conda run -p /opt/anaconda3/envs/allo-base python3 action_expert_bf16/flash_attn_cross.py

Produces: action_expert_bf16/flash_attn_cross.prj/build/final.xclbin + insts.txt
"""

import os
import allo.dataflow as df

os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = "/home/xl434/allo2/allo/allo/library/aie/kernels/"
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
os.environ["COALESCE_MORE"] = "1"
os.environ["FORCE_UNROLL_INDEX"] = "0"

from allo.library.aie.modules.flash_attn import FA

SEQ_LEN  = 128
HEAD_DIM = 64
Q_TILE   = 32    # full Q in one call (action expert SEQ=32)
Q_CHUNK  = 32
KV_CHUNK = 32

flash_attn_kernel, flash_attn_mp = FA(SEQ_LEN, HEAD_DIM, Q_TILE, Q_CHUNK, KV_CHUNK)

mod = df.build(
    flash_attn_kernel,
    target="aie",
    project="action_expert_bf16/flash_attn_cross.prj",
    mapping_primitives=flash_attn_mp,
)

print("flash_attn_cross build complete")
print("  xclbin: action_expert_bf16/flash_attn_cross.prj/build/final.xclbin")
