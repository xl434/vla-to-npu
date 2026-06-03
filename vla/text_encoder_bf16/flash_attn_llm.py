"""Build Flash Attention xclbin for text encoder self-attention with causal masking.

Text encoder self-attention: Q_H=15 heads, each attending over SEQ=128.

Parameters:
  SEQ_LEN   = 128  (text encoder sequence)
  HEAD_DIM  = 64
  Q_tile    = 128  (full sequence at once → 1 FA call per head, 15 total)
  q_chunk   = 32   → 4 Q chunks (128/32=4) for streaming
  kv_chunk  = 32   → 4 KV chunks (128/32=4)

Causal masking strategy:
  - Build FA without causal mask (allo2 kernel doesn't support it)
  - Apply causal mask in C++ at score level (after FA, before softmax)
  - This avoids modifying allo2 kernel and keeps speedup benefits

Run from vla/ directory:
    conda run -p /opt/anaconda3/envs/allo-base python3 text_encoder_bf16/flash_attn_llm.py

Produces: text_encoder_bf16/flash_attn_llm.prj/build/final.xclbin + insts.txt
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
Q_TILE   = 128   # full sequence (process all 128 rows at once → 1 call per head)
Q_CHUNK  = 32    # 4 chunks for Q streaming
KV_CHUNK = 32    # 4 chunks for KV streaming

flash_attn_kernel, flash_attn_mp = FA(SEQ_LEN, HEAD_DIM, Q_TILE, Q_CHUNK, KV_CHUNK)

mod = df.build(
    flash_attn_kernel,
    target="aie",
    project="text_encoder_bf16/flash_attn_llm.prj",
    mapping_primitives=flash_attn_mp,
)

print("flash_attn_llm build complete")
print("  SEQ=128, HEAD_DIM=64, Q_tile=128 (1 call per head)")
print("  Causal mask applied in C++, not in kernel")
print("  xclbin: text_encoder_bf16/flash_attn_llm.prj/build/final.xclbin")
