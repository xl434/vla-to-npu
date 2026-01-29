#!/usr/bin/env python3
"""
run_profiles.py

Wrapper to run vla_profiling.py across multiple M,N,K configurations
for fixed M and varying N,K choices with tile-size constraints.
Results are saved into vla_profile.json automatically.
"""
import subprocess
import os

# Fixed M dimension
M_FIXED = 1024

# Specific (M, N, K) configurations to test
DESIRED_TESTS = [
    (1024, 2048, 512),
    (1024, 2048, 768),
    (1024, 2048, 1024),
    (1024, 2048, 2048),
    (1024, 2048, 3072),
    (1024, 3072, 32),
    (1024, 3072, 64),
    (1024, 3072, 128),
    (1024, 3072, 256),
    (1024, 3072, 512),
    (1024, 3072, 768),
    (1024, 3072, 1024),
    (1024, 3072, 2048),
    (1024, 3072, 3072)
]

# Candidate tile sizes to test
CANDIDATE_TILES = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (32, 32, 64),
    (64, 64, 32),
]

# Memory constraint constants
MAX_BYTES = 32256
MAX_DTYPE_SIZE = 2


def tile_is_valid(m: int, n: int, k: int, M: int, N: int, K: int) -> bool:
    # Bound on tile dimension
    if m >= 1024 or n >= 1024 or k >= 1024:
        return False
    # Divisibility and vectorization requirements
    if M % m != 0 or N % n != 0 or K % k != 0:
        return False
    if ((M // m) % 4 != 0 or (N // n) % 4 != 0 or (K // k) % 4 != 0) and \
       ((M // m) not in (1, 2) and (N // n) not in (1, 2) and (K // k) not in (1, 2)):
        return False
    # K/k ratio constraint
    if (K // k) > 32:
        return False
    # Memory footprint constraint
    footprint = m * n + n * k + k * m
    if footprint * MAX_DTYPE_SIZE > MAX_BYTES:
        return False
    return True


def main():
    base_dir = os.path.dirname(__file__)
    profiling_script = os.path.join(base_dir, "vla_profiling.py")

    for M, N, K in DESIRED_TESTS:
        combos = [
            (m, n, k)
            for (m, n, k) in CANDIDATE_TILES
            if tile_is_valid(m, n, k, M, N, K)
        ]
        if not combos:
            print(f"Skipping M={M}, N={N}, K={K}: no valid tile sizes.")
            continue

        for m, n, k in combos:
            print(f"Profiling M={M}, N={N}, K={K}, m={m}, n={n}, k={k}")
            subprocess.run(
                ["python3", profiling_script,
                 "--M", str(M),
                 "--N", str(N),
                 "--K", str(K),
                 "--m", str(m),
                 "--n", str(n),
                 "--k", str(k)],
                check=True
            )


if __name__ == "__main__":
    main()
