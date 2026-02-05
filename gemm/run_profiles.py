#!/usr/bin/env python3
"""
run_profiles.py

Wrapper to run vla_profiling.py across all combinations of M, N, K
and test all candidate tile sizes. Failures are logged by vla_profiling.py.
Results are saved into vla_profile.json automatically.
"""
import subprocess
import os

# Specific (M, N, K) configurations to test
M = [32, 64, 128, 256, 512, 1024, 2048]
N = [32, 64, 128, 256, 512, 768, 1024, 2048]
K = [32, 64, 128, 256, 512, 768, 1024, 2048]

# Candidate tile sizes to test
CANDIDATE_TILES = [
    (32,32,32),
    (64, 64, 64),
    (128, 128, 128),
    (32, 64, 32),
    (64, 128, 64),
]


def main():
    base_dir = os.path.dirname(__file__)
    profiling_script = os.path.join(base_dir, "vla_profiling.py")

    for M_val in M:
        for N_val in N:
            for K_val in K:
                for (m, n, k) in CANDIDATE_TILES:
                    print(f"Profiling M={M_val}, N={N_val}, K={K_val}, m={m}, n={n}, k={k}")
                    subprocess.run(
                        ["python3", profiling_script,
                         "--M", str(M_val),
                         "--N", str(N_val),
                         "--K", str(K_val),
                         "--m", str(m),
                         "--n", str(n),
                         "--k", str(k)],
                        check=False
                    )

if __name__ == "__main__":
    main()