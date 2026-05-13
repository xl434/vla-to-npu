# OPT-D: C++ Unified Pipeline + Zero-Rebuild Standalone Runner

**Date:** 2026-05-13

---

## Summary

Two problems were solved in this session:

1. **OPT-D** — Replace Python XRT dispatch wrappers with C++ unified executables to eliminate ~20ms Python→XRT overhead per NPU call.
2. **Zero-rebuild runner** — Replace `df.build()` (which unconditionally rebuilds all xclbins, ~7 min startup) with direct calls to pre-built binaries, reducing startup from ~7 minutes to ~1 second.

Combined result: **14.0s → 5.45s** total pipeline time, with instant startup.

---

## Background: Why Python Dispatch Is Slow

Every Python NPU call in the original `vla.py` opens a new XRT hw_context, transfers data, dispatches the kernel, and closes the context. This ~20ms overhead is paid per call, regardless of compute time.

For the transformer stages (vision, connector, text, action expert), a C++ executable can open the XRT context **once**, loop over all kernel dispatches internally, and close. This amortizes the context-open cost.

For preprocessing, Python was making **320 separate subprocess spawns** (128 im2col + 192 GEMM), each opening and closing XRT. David's `fused_unified` C++ class reduces this to **2 hw_context opens** (one for im2col phase, one for GEMM phase), regardless of dispatch count.

---

## Why `df.build()` Is Unavoidable in vla.py

Allo's `df.build()` (`/opt/allo/allo/backend/aie/__init__.py`, lines 718–720) always does:

```python
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)   # unconditional delete
os.makedirs(build_dir)
```

There is no caching. Every `import vla` triggers full xclbin recompilation for all ~33 kernels, taking ~7 minutes. This negates any runtime speedup.

**Solution:** Call `./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz 0` directly (the same command Allo uses internally at line 948), bypassing `df.build()` entirely.

---

## What Was Built

### New files

| File | Description |
|------|-------------|
| `vla/vla_cpp.py` | Python subprocess wrappers calling C++ executables. Drop-in for Python NPU forward functions. |
| `vla/vla_standalone.py` | Full pipeline runner: no `df.build()`, no xclbin rebuild, instant startup. |
| `vla/vla_opt_d.py` | Monkey-patches `vla.py` with C++ wrappers (still has 7-min startup from `import vla`). |
| `vla/test_cpp.py` | Modular correctness + timing tests for each C++ stage. |

### Modified files

| File | Change |
|------|--------|
| `vla/connector/unified.prj/test.cpp` | Renamed `--xclbin_copy`/`--instr_copy` CLI flags to `--xclbin_pixel_shuffle`/`--instr_pixel_shuffle` to match the OPT-B pixel_shuffle rename. |
| `vla/text_encoder_bf16/unified.prj/text_encoder.cpp` | Fixed rope xclbin path: `rope/fused_5h.prj/...` → `../rope/fused_5h.prj/...` (relative to `unified.prj/` CWD, not parent). |
| `vla/action_expert_bf16/unified.prj/action_expert.cpp` | Same rope path fix. |

### C++ executables (pre-existing, already built)

| Stage | Executable | Key property |
|-------|-----------|--------------|
| Preprocessing | `preprocessing/fused_unified/build/test` | 2 hw_context opens for 320 dispatches (David's implementation) |
| Vision encoder | `vision_block/unified.prj/build/vision_encoder` | 1 process per layer |
| Connector | `connector/unified.prj/build/test` | pixel_shuffle + tiled GEMM |
| Text encoder | `text_encoder_bf16/unified.prj/build/text_encoder` | Returns output + key + val |
| Action expert | `action_expert_bf16/unified.prj/build/action_expert` | `--mode self` or `--mode cross` |

---

## Timing Results

All measured on AMD XDNA NPU1, `vla_standalone.py`, VIT_NUM_LAYERS=1, LLAMA_NUM_LAYERS=2.

### Before OPT-D (Python NPU, `vla.py`, after 5-12 softmax + gelu work)

| Stage | Wall clock |
|-------|-----------|
| Preprocessing | 8.6 s (320 subprocess spawns) |
| Vision encoder (1L) | 17.8 s |
| Connector | 19.9 s |
| Joint transformer (2L) | 12.8 s |
| Postprocessing | 0.06 s |
| **Total** | **~59 s** |
| Startup (`df.build()`) | **~7 min** |

### After OPT-D + fused_unified preprocessing (`vla_standalone.py`)

| Stage | Wall clock | Notes |
|-------|-----------|-------|
| Preprocessing | **0.161 s** | 54× speedup from Python — 2 hw_context opens |
| Vision encoder (1L) | 3.464 s | Includes C++ internal warmup run (~2× actual) |
| Connector | 0.184 s | |
| Joint transformer (2L) | 1.588 s | |
| Postprocessing | 0.053 s | |
| **Total** | **5.450 s** | **11× speedup over Python** |
| Startup | **~1 s** | No xclbin rebuild |

### Preprocessing specifically: Python vs C++ unified

| Approach | Time | hw_context opens |
|----------|------|-----------------|
| Python (`vla.py`) | 8.6 s | 320 |
| C++ `fused_unified` | 0.161 s | 2 |
| **Speedup** | **54×** | |

> Note: Vision encoder 3.5s includes one internal warmup run by the C++ executable (~1.7s actual kernel). This is by design in the C++ binary; a future `--no-warmup` flag would bring it to ~1.7s.

---

## How to Run

### Prerequisites

All xclbins must already be compiled (i.e., `.prj/build/final.xclbin` exists for each stage).
All C++ unified executables must be built. From `/vla/`:

```bash
cd preprocessing/fused_unified    && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../..
cd vision_block/unified.prj       && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../..
cd text_encoder_bf16/unified.prj  && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../..
cd action_expert_bf16/unified.prj && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../..
cd connector/unified.prj          && mkdir -p build && cd build && cmake .. && make -j4 && cd ../../..
```

### Run the standalone pipeline (random weights)

```bash
cd /home/xl434/vla-to-npu/vla
python3 vla_standalone.py
```

Expected output (no xclbin rebuild, starts in ~1s):

```
Running standalone VLA pipeline (no df.build())...

== Timings (standalone, no rebuild) ==
Preprocessing           : 0.161 s
Vision encoder (1L)    : 3.464 s
Connector               : 0.172 s
Joint transformer (2L) : 1.591 s
Postprocessing          : 0.057 s
Total                   : 5.450 s
Output shape: (32, 32)
```

### Test individual C++ stages

```bash
python3 test_cpp.py                 # all stages
python3 test_cpp.py --stage vision
python3 test_cpp.py --stage connector
python3 test_cpp.py --stage text
python3 test_cpp.py --stage self    # action expert self-attention
python3 test_cpp.py --stage cross   # action expert cross-attention
```

Each stage prints Python NPU time, C++ time, max/mean error, and speedup.

---

## How to Load Real Weights

`vla_standalone.py` uses random weights in `main()` for benchmarking. To use real model weights, replace the `params_*` dictionaries with loaded numpy arrays.

### Weight shapes expected by each stage

| Stage | Parameter | Shape | dtype |
|-------|-----------|-------|-------|
| **Preprocessing** | `kernel` | `[768, 3, 16, 16]` | bf16 |
| **Vision encoder** (per layer) | `Wq`, `Wk`, `Wv`, `Wo` | `[768, 768]` | bf16 |
| | `W_up` | `[768, 3072]` | bf16 |
| | `W_down` | `[3072, 768]` | bf16 |
| | `W_norm_1`, `W_norm_2` | `[768]` | bf16 |
| | `b_norm_1`, `b_norm_2` | `[768]` | bf16 |
| **Connector** | `W` | `[12288, 960]` | bf16 |
| **Text encoder** (per layer) | `Wq` | `[960, 960]` | bf16 |
| | `Wk`, `Wv` | `[960, 320]` | bf16 |
| | `Wo` | `[960, 960]` | bf16 |
| | `W_gate`, `W_up` | `[960, 2560]` | bf16 |
| | `W_down` | `[2560, 960]` | bf16 |
| | `W_norm_1`, `W_norm_2` | `[960]` | bf16 |
| **Action expert self** (per layer) | `Wq` | `[768, 960]` | bf16 |
| | `Wk`, `Wv` | `[768, 320]` | bf16 |
| | `Wo` | `[960, 768]` | bf16 |
| | `W_gate`, `W_up` | `[768, 2048]` | bf16 |
| | `W_down` | `[2048, 768]` | bf16 |
| | `W_norm_1`, `W_norm_2` | `[768]` | bf16 |
| **Action expert cross** (per layer) | `Wq` | `[768, 960]` | bf16 |
| | `Wk_cross`, `Wv_cross` | `[320, 320]` | bf16 |
| | `Wo` | `[960, 768]` | bf16 |
| | `W_gate`, `W_up` | `[768, 2048]` | bf16 |
| | `W_down` | `[2048, 768]` | bf16 |
| | `W_norm_1`, `W_norm_2` | `[768]` | bf16 |
| **Postprocessing** | `W_exp_norm` | `[768]` | bf16 |
| | `W_action_out` | `[768, 32]` | bf16 |

### Loading from a PyTorch checkpoint

```python
from ml_dtypes import bfloat16 as np_bfloat16
import torch
import numpy as np

def to_bf16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).numpy().view(np_bfloat16)

ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")

# Example: load vision encoder weights (adjust key names to match your checkpoint)
params_vit = dict(
    Wq       = to_bf16(ckpt["vit.layers.0.attn.q_proj.weight"].T),  # [768, 768]
    Wk       = to_bf16(ckpt["vit.layers.0.attn.k_proj.weight"].T),
    Wv       = to_bf16(ckpt["vit.layers.0.attn.v_proj.weight"].T),
    Wo       = to_bf16(ckpt["vit.layers.0.attn.out_proj.weight"].T),
    W_up     = to_bf16(ckpt["vit.layers.0.mlp.fc1.weight"].T),      # [768, 3072]
    W_down   = to_bf16(ckpt["vit.layers.0.mlp.fc2.weight"].T),      # [3072, 768]
    W_norm_1 = to_bf16(ckpt["vit.layers.0.norm1.weight"]),
    b_norm_1 = to_bf16(ckpt["vit.layers.0.norm1.bias"]),
    W_norm_2 = to_bf16(ckpt["vit.layers.0.norm2.weight"]),
    b_norm_2 = to_bf16(ckpt["vit.layers.0.norm2.bias"]),
)

# Then call:
# conv_emb  = preprocessing_block(image_rgb, params_proc)
# vision_emb = vision_encoder(VIT_NUM_LAYERS, conv_emb, params_vit)
# ...
```

### Loading from raw numpy files

If weights are already saved as bf16 binary files (e.g., from a previous export):

```python
def load_bf16(path, shape):
    return np.fromfile(path, dtype=np.uint16).view(np_bfloat16).reshape(shape)

params_vit = dict(
    Wq = load_bf16("weights/vit_wq.bin", (768, 768)),
    # ...
)
```

---

## Architecture of vla_cpp.py

`vla_cpp.py` provides five drop-in wrappers:

```
preprocessing_block(image, params)          → [1024, 768] bf16
vision_block(x, params)                     → [1024, 768] bf16
connector_block(x, params)                  → [64, 960] bf16
text_encoder_forward(x, params)             → (out [128,960], key [128,320], val [128,320])
action_expert_self_forward(action, params)  → [32, 768] bf16
action_expert_cross_forward(action, text_k, text_v, params) → [32, 768] bf16
```

Each wrapper:
1. Writes input arrays to a `tempfile.TemporaryDirectory` as raw bf16 binary
2. Calls the C++ executable via `subprocess.run`
3. Reads the output binary and returns a numpy array

The C++ executables receive the data file paths as CLI arguments, run all NPU dispatches internally with a single XRT hw_context open, and write the result to the output path.

---

## Pipeline Architecture (`vla_standalone.py`)

```
Python (vla_standalone.py)
  │
  ├─ preprocessing_block()    → cpp.preprocessing_block()
  │    └─ preprocessing/fused_unified/build/test
  │         ├─ im2col hw_context  — 128 dispatches
  │         └─ GEMM hw_context   — 192 dispatches
  │
  ├─ vision_encoder()          → cpp.vision_block()  × num_layers
  │    └─ vision_block/unified.prj/build/vision_encoder
  │
  ├─ cpp.connector_block()
  │    └─ connector/unified.prj/build/test
  │         ├─ pixel_shuffle hw_context
  │         └─ GEMM hw_context
  │
  ├─ joint_transformer()
  │    ├─ cpp.text_encoder_forward()    × num_layers
  │    │    └─ text_encoder_bf16/unified.prj/build/text_encoder
  │    └─ cpp.action_expert_*_forward() × num_layers
  │         └─ action_expert_bf16/unified.prj/build/action_expert
  │
  └─ postprocessing()          → _call_top() × 2
       ├─ postprocessing/rms_norm.prj/build/top
       └─ postprocessing/gemm.prj/build/top
```

Postprocessing uses `_call_top()` (direct Allo binary) rather than a C++ unified executable — it is only 2 calls and contributes <0.06s.
