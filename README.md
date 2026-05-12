# vla_npu_kernel

## Project Structure

```
vla-to-npu/
├── cc/                        # C++ AIE kernel implementations
│   ├── float/                 #   float32 kernels (rope, sin_cos, softmax, …)
│   ├── bf16/                  #   bfloat16 kernels (silu_160, silu_256, …)
│   ├── bf16_new/              #   updated bf16 kernels (gelu, silu, rms_norm, …)
│   └── bf16_old/              #   legacy bf16 kernels (kept for reference)
├── kernel_testing/            # Per-kernel test scripts
│   ├── add/                   #   elementwise add
│   ├── attn_score/            #   attention score (transpose matmul + scale)
│   ├── conv/                  #   conv2d
│   ├── cosine/                #   cosine kernel
│   ├── gelu/                  #   GELU activation
│   ├── gemm/                  #   GEMM variants
│   ├── gemm_silu/             #   fused GEMM + SiLU
│   ├── layer_norm/            #   layer norm
│   ├── masked_softmax/        #   masked softmax
│   ├── rms_norm/              #   RMS norm
│   ├── silu/                  #   SiLU activation
│   ├── sine/                  #   sine kernel
│   ├── softmax_bf16/          #   bf16 softmax
│   ├── softmax_float/         #   float32 softmax
│   └── run_tests.sh           #   runs all VLA component + e2e tests
├── vla/                       # VLA model implementation (NPU, bf16)
│   ├── vla.py                 #   full end-to-end VLA forward pass
│   ├── action_expert_bf16.py  #   action expert transformer (cross + self attn)
│   ├── text_encoder_bf16.py   #   text encoder (Llama-style)
│   ├── llama_block_rope_bf16.py  # Llama block with RoPE (bf16)
│   ├── vision_block_bf16.py   #   ViT vision block
│   ├── connector_bf16.py      #   pixel shuffle + linear connector
│   ├── preprocessing_fused_bf16.py  # conv2d patch embed (im2col + GEMM, NPU)
│   ├── test_rope_fused.py     #   test: fused RoPE kernel
│   ├── test_rope_multiple.py  #   test: multi-head RoPE tiling
│   ├── tests/                 #   additional kernel-level tests (sin_cos, rope_full)
│   ├── vla_float/             #   float32 reference implementations
│   └── old_version/           #   legacy preprocessing (bf16, pipelined)
├── analysis/                  # Profiling and performance analysis scripts
├── log/                       # Design notes and performance logs
└── tools/                     # Utility scripts
```

## Running Kernel Tests

Each subdirectory under `kernel_testing/` contains test scripts that validate the kernel against a PyTorch reference.

**Run a single kernel test:**
```bash
cd kernel_testing/cosine
python test_cosine.py
```

**Run all VLA component and e2e tests:**
```bash
bash kernel_testing/run_tests.sh
# Results written to test_results.log
```

Tests compare NPU kernel outputs against PyTorch reference with bf16-appropriate tolerances (`atol=0.1, rtol=0.1`) and report PASS/FAIL.