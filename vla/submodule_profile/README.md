# Submodule Profiling

This directory contains profiling scripts for individual submodules of the llama_block implementation.

## Linear Projection Profiling

The `linear_projection_profile.py` script profiles the NPU implementation of linear projections used throughout the llama_block.

### Features

- **Comprehensive Test Cases**: Profiles all linear projection operations used in llama_block:
  - Query projection (q_proj): 64×960×768
  - Key projection (k_proj): 64×320×768  
  - Value projection (v_proj): 64×320×768
  - Output projection (o_proj): 64×768×960
  - FFN up projection (up_proj): 64×3072×768
  - FFN gate projection (gate_proj): 64×3072×768
  - FFN down projection (down_proj): 64×768×3072

- **Detailed Kernel Profiling**: Measures individual NPU kernel performance:
  - `linear_matmul_mod` timing
  - `linear_accumulate_mod` timing
  - Per-tile performance analysis

- **Performance Metrics**:
  - Execution time comparison (NPU vs PyTorch)
  - Throughput in GFLOPS
  - Speedup calculation
  - Numerical accuracy verification

### Usage

#### Environment Setup
First, make sure your environment is set up [[memory:5502153]]:
```bash
conda activate py310
source ~/env_setup.sh
```

#### Run All Test Cases
```bash
cd vla/submodule_profile
python linear_projection_profile.py
```

#### Run Specific Test Case
```bash
python linear_projection_profile.py --test-case q_proj
python linear_projection_profile.py --test-case k_proj --num-runs 20
```

#### Quick Test
```bash
python test_linear_profile.py
```

### Available Test Cases

| Test Case | Description | Matrix Shapes (M×N×K) |
|-----------|-------------|----------------------|
| `q_proj` | Query projection | 64×960×768 |
| `k_proj` | Key projection | 64×320×768 |
| `v_proj` | Value projection | 64×320×768 |
| `o_proj` | Output projection | 64×768×960 |
| `up_proj` | FFN up projection | 64×3072×768 |
| `gate_proj` | FFN gate projection | 64×3072×768 |
| `down_proj` | FFN down projection | 64×768×3072 |

### Output Example

```
Profiling: q_proj
Description: Query projection: (64, 960, 768)
Matrix shapes: A(64, 768) @ B(768, 960) -> C(64, 960)
============================================================

Results:
  NPU Time:     12.345 ± 0.123 ms
  PyTorch Time: 8.901 ± 0.089 ms
  Speedup:      0.72x
  Max Error:    1.234567e-05
  Rel Error:    2.345678e-06

Kernel Breakdown:
  MatMul Time:     10.123 ms (192 calls)
  Accumulate Time: 2.222 ms (192 calls)
  Total Tiles:     192
  Avg MatMul/tile: 0.053 ms
  Avg Accum/tile:  0.012 ms

Throughput:
  NPU GFLOPS:     75.23
  PyTorch GFLOPS: 104.56
```

### Configuration

The profiling script uses the same configuration as `llama_block.py`:

- **Sequence Length**: 64
- **Embedding Dimension**: 768
- **Attention Heads**: Q_H=15, KV_H=5
- **Head Dimension**: 64
- **FFN Hidden Size**: 3072
- **Tile Size**: 64×64×64

### Adding New Test Cases

To add a new test case, modify the `test_cases` list in `linear_projection_profile.py`:

```python
test_cases.append(
    LinearProjectionTestCase("my_test", M, N, K, "Description")
)
```

### Performance Notes

- The NPU implementation uses tiled matrix multiplication with 64×64 tiles
- Each linear projection is decomposed into multiple matmul and accumulate operations
- Profiling includes both kernel execution time and data movement overhead
- Results may vary based on system load and NPU availability

### Troubleshooting

1. **Environment Issues**: Make sure you've activated the conda environment and sourced the setup script
2. **Kernel Build Errors**: Ensure the `cc/` directory contains the required kernel implementations
3. **Memory Issues**: Large matrix operations may require sufficient system memory
4. **NPU Availability**: Verify that the NPU is available and accessible

### Future Enhancements

- Memory usage profiling
- Power consumption measurement
- Multi-batch profiling
- Different precision comparisons (fp16, bf16)
- Cache performance analysis


## Commands for single_tile_matmul_profile.py
```bash
python single_tile_matmul_profile.py --dtype float32 --tile-m 64 --tile-n 128 --tile-k 128
```
