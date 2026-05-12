# GEMM Parameter Constraints

## Parameters

| Parameter | Description |
|-----------|-------------|
| `M, N, K` | Matrix dimensions: A(M×K) @ B(K×N) → C(M×N) |
| `m, n, k` | Tile sizes per AIE core |
| `col_num` | Number of column bundles for AIE mapping (default: 4) |
| `row_num` | Number of row bundles for AIE mapping (default: 4) |

## Derived Values

| Symbol | Formula | Description |
|--------|---------|-------------|
| `Pm`   | `M / m` | Number of partitions along M |
| `Pn`   | `N / n` | Number of partitions along N |
| `Pk`   | `K / k` | Number of partitions along K (chained) |

## Constraints

### 1. Tile Divisibility

Tile sizes must evenly divide their corresponding matrix dimensions:

```
M % m == 0
N % n == 0
K % k == 0
```

### 2. Bundling Divisibility

Partition counts along M and N must be divisible by `row_num` and `col_num`:

```
Pn % col_num == 0
Pm % row_num == 0
```

If this is not satisfied, some AIE cores will not be bundled and mapping will fail.

### 3. K-Dimension (No Bundling Constraint)

`Pk` is only used for **chaining** (sequential K-reduction across cores). There is no divisibility requirement — any `Pk >= 1` works.

## Example: Hidden Dimension = 960

With tile size 64: `960 / 64 = 15`

| Dimension | P value | col/row_num=4 | col/row_num=3 | col/row_num=5 |
|-----------|---------|---------------|---------------|---------------|
| N=960     | Pn=15   | 15%4≠0 **FAIL** | 15%3=0 PASS | 15%5=0 PASS |
| M=960     | Pm=15   | 15%4≠0 **FAIL** | 15%3=0 PASS | 15%5=0 PASS |
| K=960     | Pk=15   | no constraint | no constraint | no constraint |

## How to Choose col_num and row_num

Pick `col_num` as a factor of `Pn`, and `row_num` as a factor of `Pm`.

Common factor table for typical partition counts:

| Pn or Pm | Factors           | Recommended col/row_num |
|----------|-------------------|-------------------------|
| 4        | 1, 2, 4           | 4                       |
| 8        | 1, 2, 4, 8        | 4                       |
| 12       | 1, 2, 3, 4, 6, 12 | 4                       |
| 15       | 1, 3, 5, 15       | 3                       |
| 16       | 1, 2, 4, 8, 16    | 4                       |
| 32       | 1, 2, 4, 8, 16, 32| 4                       |

## Usage

```bash
# K=960 works with default col_num/row_num
python v2_test_mapping_large_gemm.py --M 64 --N 64 --K 960 --m 64 --n 64 --k 64 --dtype bf16

# N=960 requires col_num=3 (or 5)
python v2_test_mapping_large_gemm.py --M 64 --N 960 --K 64 --m 64 --n 64 --k 64 --col-num 3 --dtype bf16

# M=960 requires row_num=3 (or 5)
python v2_test_mapping_large_gemm.py --M 960 --N 64 --K 64 --m 64 --n 64 --k 64 --row-num 3 --dtype bf16
```
