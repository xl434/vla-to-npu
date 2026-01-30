# vla_npu_kernel

## Project Structure

This repository contains NPU kernel implementations and tests for various operations:

- `cc/` - C++ kernel implementations (cosine.cc, gelu.cc, silu.cc, softmax, etc.)
- `cosine/`, `sine/`, `gelu/`, `silu/` - Activation function kernels and tests
- `softmax_bf16/`, `softmax_float/`, `masked_softmax/` - Softmax variants
- `gemm/`, `rope/`, `gpt2/` - Matrix operations and transformer components
- `vla/` - VLA layer implementations
- `tools/` - Utility scripts

## Running Kernel Tests

Each kernel directory contains test files that validate the implementation against PyTorch reference outputs.

**Run a test:**
```bash
cd <kernel_directory>
python test_*.py
```

**Example:**
```bash
cd cosine
python test_cosine.py
```

Tests will automatically compare kernel outputs against PyTorch reference implementations and report PASS/FAIL status.