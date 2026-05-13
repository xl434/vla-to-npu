"""
vla_opt_d.py — OPT-D unified pipeline: same as vla.py but using C++ NPU executables
for the major transformer stages to eliminate ~20ms Python dispatch overhead per call.

Run:  python3 vla_opt_d.py

What changes relative to vla.py:
  vision_block               → vla_cpp.vision_block              (C++ subprocess)
  connector_block            → vla_cpp.connector_block           (C++ subprocess)
  text_encoder_forward       → vla_cpp.text_encoder_forward      (C++ subprocess)
  action_expert_self_forward → vla_cpp.action_expert_self_forward (C++ subprocess)
  action_expert_cross_forward→ vla_cpp.action_expert_cross_forward (C++ subprocess)

Unchanged (still Python NPU):
  preprocessing_block  — uses im2col+fused_conv_add kernels, no standalone C++ binary
  create_state_emb     — small GEMM, overhead negligible
  postprocessing       — rms_norm + small GEMM, overhead negligible

Build C++ executables before running (from /vla/):
  cd vision_block/unified.prj      && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../../
  cd text_encoder_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../../
  cd action_expert_bf16/unified.prj&& mkdir -p build && cd build && cmake .. && make -j4 && cd ../../../
  cd connector/unified.prj         && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../../
"""

import vla          # Python baseline — also builds state_emb/postprocessing NPU modules
import vla_cpp      # C++ subprocess wrappers

# ---- Monkey-patch: replace Python NPU forward functions with C++ versions ----
vla.vision_block                   = vla_cpp.vision_block
vla.connector_block                = vla_cpp.connector_block
vla.text_encoder_forward           = vla_cpp.text_encoder_forward
vla.action_expert_self_forward     = vla_cpp.action_expert_self_forward
vla.action_expert_cross_forward    = vla_cpp.action_expert_cross_forward

if __name__ == "__main__":
    vla.main()
