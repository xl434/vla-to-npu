#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiling script for linear_projection submodule from llama_block.py

This script profiles the NPU linear_projection function with the same shapes
and parameters used in the main llama_block implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import allo
import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie import ExternalModule
import argparse
import os
import sys

# Add parent directory to path to import from llama_block
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Configuration (matching llama_block.py)
# ===============================================================================
KERNEL_LIB_PATH = "../../cc/"
BATCH = 1
SEQ = 64
EMBD = 768  # 64 * 12
Q_H = 15
KV_H = 5
HEAD_DIM = 64
FFN_HID = EMBD * 4

# Linear kernel configuration
LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
Ty = float32

# Layout configurations
linear_A_layout = Layout("S0R")
linear_B_layout = Layout("RS1")
linear_C_layout = Layout("S0S1")

# ===============================================================================
# NPU Kernel Definitions (from llama_block.py)
# ===============================================================================

@df.region()
def linear_matmul_kernel():
    @df.kernel(mapping=[4, 4])
    def gemm(
        A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
        B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
        C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        C[:, :] = allo.matmul(A, B)

@df.region()
def linear_accumulate_kernel():
    @df.kernel(mapping=[2, 4])
    def core(
        A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        C[:, :] = allo.add(A, B)

# Build kernels
print("Building NPU kernels...")
linear_matmul_mod = df.build(
    linear_matmul_kernel, target="aie", project="linear_matmul_profile.prj"
)
linear_accumulate_mod = df.build(
    linear_accumulate_kernel, target="aie", project="linear_accumulate_profile.prj"
)
print("NPU kernels built successfully!")

# ===============================================================================
# Linear Projection Implementation
# ===============================================================================

def linear_projection_npu(A, B, C, M, N, K):
    """NPU implementation of linear projection (from llama_block.py)"""
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
            for k in range(K // LINEAR_K):
                tile_A = A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                ]
                tile_B = B[
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ]
                linear_matmul_mod(tile_A, tile_B, C_tmp)
                linear_accumulate_mod(
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    C_tmp,
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )

def linear_projection_pytorch(A, B):
    """PyTorch reference implementation"""
    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    return torch.matmul(A_torch, B_torch).numpy()

# ===============================================================================
# Test Case Definitions
# ===============================================================================

class LinearProjectionTestCase:
    def __init__(self, name, M, N, K, description):
        self.name = name
        self.M = M
        self.N = N
        self.K = K
        self.description = description

# Define test cases matching llama_block usage patterns
test_cases = [
    LinearProjectionTestCase("q_proj", SEQ, Q_H * HEAD_DIM, EMBD, 
                           "Query projection: (64, 960, 768)"),
    LinearProjectionTestCase("k_proj", SEQ, KV_H * HEAD_DIM, EMBD, 
                           "Key projection: (64, 320, 768)"),
    LinearProjectionTestCase("v_proj", SEQ, KV_H * HEAD_DIM, EMBD, 
                           "Value projection: (64, 320, 768)"),
    LinearProjectionTestCase("o_proj", SEQ, EMBD, Q_H * HEAD_DIM, 
                           "Output projection: (64, 768, 960)"),
    LinearProjectionTestCase("up_proj", SEQ, FFN_HID, EMBD, 
                           "FFN up projection: (64, 3072, 768)"),
    LinearProjectionTestCase("gate_proj", SEQ, FFN_HID, EMBD, 
                           "FFN gate projection: (64, 3072, 768)"),
    LinearProjectionTestCase("down_proj", SEQ, EMBD, FFN_HID, 
                           "FFN down projection: (64, 768, 3072)"),
]

# ===============================================================================
# Profiling Functions
# ===============================================================================

def profile_npu_kernels(A, B, M, N, K, num_runs=10):
    """Profile individual NPU kernel calls"""
    C = np.zeros((M, N)).astype(np.float32)
    
    # Profile matmul kernel
    matmul_times = []
    accumulate_times = []
    total_matmul_calls = 0
    total_accumulate_calls = 0
    
    for run in range(num_runs):
        C.fill(0)  # Reset output
        run_matmul_time = 0
        run_accumulate_time = 0
        run_matmul_calls = 0
        run_accumulate_calls = 0
        
        for i in range(M // LINEAR_M):
            for j in range(N // LINEAR_N):
                C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
                for k in range(K // LINEAR_K):
                    tile_A = A[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        k * LINEAR_K : (k + 1) * LINEAR_K,
                    ]
                    tile_B = B[
                        k * LINEAR_K : (k + 1) * LINEAR_K,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ]
                    
                    # Time matmul kernel
                    start_time = time.perf_counter()
                    linear_matmul_mod(tile_A, tile_B, C_tmp)
                    run_matmul_time += time.perf_counter() - start_time
                    run_matmul_calls += 1
                    
                    # Time accumulate kernel
                    start_time = time.perf_counter()
                    linear_accumulate_mod(
                        C[
                            i * LINEAR_M : (i + 1) * LINEAR_M,
                            j * LINEAR_N : (j + 1) * LINEAR_N,
                        ],
                        C_tmp,
                        C[
                            i * LINEAR_M : (i + 1) * LINEAR_M,
                            j * LINEAR_N : (j + 1) * LINEAR_N,
                        ],
                    )
                    run_accumulate_time += time.perf_counter() - start_time
                    run_accumulate_calls += 1
        
        matmul_times.append(run_matmul_time)
        accumulate_times.append(run_accumulate_time)
        total_matmul_calls = run_matmul_calls
        total_accumulate_calls = run_accumulate_calls
    
    return {
        'matmul_times': matmul_times,
        'accumulate_times': accumulate_times,
        'total_matmul_calls': total_matmul_calls,
        'total_accumulate_calls': total_accumulate_calls,
        'output': C
    }

def profile_linear_projection(test_case, num_runs=10):
    """Profile a complete linear projection operation"""
    M, N, K = test_case.M, test_case.N, test_case.K
    
    # Generate random input data
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    print(f"\n{'='*60}")
    print(f"Profiling: {test_case.name}")
    print(f"Description: {test_case.description}")
    print(f"Matrix shapes: A({M}, {K}) @ B({K}, {N}) -> C({M}, {N})")
    print(f"{'='*60}")
    
    # Profile NPU implementation
    print("Profiling NPU implementation...")
    npu_times = []
    
    for run in range(num_runs):
        C_npu = np.zeros((M, N)).astype(np.float32)
        start_time = time.perf_counter()
        linear_projection_npu(A, B, C_npu, M, N, K)
        npu_times.append(time.perf_counter() - start_time)
    
    # Profile PyTorch reference
    print("Profiling PyTorch reference...")
    pytorch_times = []
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        C_pytorch = linear_projection_pytorch(A, B)
        pytorch_times.append(time.perf_counter() - start_time)
    
    # Profile individual NPU kernels
    print("Profiling individual NPU kernels...")
    kernel_profile = profile_npu_kernels(A, B, M, N, K, num_runs=5)
    
    # Verify correctness
    C_npu_final = np.zeros((M, N)).astype(np.float32)
    linear_projection_npu(A, B, C_npu_final, M, N, K)
    C_pytorch_final = linear_projection_pytorch(A, B)
    
    max_diff = np.max(np.abs(C_npu_final - C_pytorch_final))
    relative_error = max_diff / (np.max(np.abs(C_pytorch_final)) + 1e-8)
    
    # Calculate statistics
    npu_mean = np.mean(npu_times)
    npu_std = np.std(npu_times)
    pytorch_mean = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    
    matmul_mean = np.mean(kernel_profile['matmul_times'])
    accumulate_mean = np.mean(kernel_profile['accumulate_times'])
    
    # Calculate operations count
    total_ops = M * N * K * 2  # multiply-add operations
    tiles_M = M // LINEAR_M
    tiles_N = N // LINEAR_N
    tiles_K = K // LINEAR_K
    total_tiles = tiles_M * tiles_N * tiles_K
    
    print(f"\nResults:")
    print(f"  NPU Time:     {npu_mean*1000:.3f} ± {npu_std*1000:.3f} ms")
    print(f"  PyTorch Time: {pytorch_mean*1000:.3f} ± {pytorch_std*1000:.3f} ms")
    print(f"  Speedup:      {pytorch_mean/npu_mean:.2f}x")
    print(f"  Max Error:    {max_diff:.6e}")
    print(f"  Rel Error:    {relative_error:.6e}")
    
    print(f"\nKernel Breakdown:")
    print(f"  MatMul Time:     {matmul_mean*1000:.3f} ms ({kernel_profile['total_matmul_calls']} calls)")
    print(f"  Accumulate Time: {accumulate_mean*1000:.3f} ms ({kernel_profile['total_accumulate_calls']} calls)")
    print(f"  Total Tiles:     {total_tiles}")
    print(f"  Avg MatMul/tile: {matmul_mean*1000/kernel_profile['total_matmul_calls']:.3f} ms")
    print(f"  Avg Accum/tile:  {accumulate_mean*1000/kernel_profile['total_accumulate_calls']:.3f} ms")
    
    print(f"\nThroughput:")
    print(f"  NPU GFLOPS:     {total_ops / npu_mean / 1e9:.2f}")
    print(f"  PyTorch GFLOPS: {total_ops / pytorch_mean / 1e9:.2f}")
    
    return {
        'test_case': test_case,
        'npu_times': npu_times,
        'pytorch_times': pytorch_times,
        'kernel_profile': kernel_profile,
        'max_error': max_diff,
        'relative_error': relative_error,
        'speedup': pytorch_mean / npu_mean,
        'npu_gflops': total_ops / npu_mean / 1e9,
        'pytorch_gflops': total_ops / pytorch_mean / 1e9,
    }

def run_comprehensive_profile():
    """Run profiling for all test cases"""
    print("Linear Projection NPU Profiling")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  BATCH: {BATCH}, SEQ: {SEQ}, EMBD: {EMBD}")
    print(f"  Q_H: {Q_H}, KV_H: {KV_H}, HEAD_DIM: {HEAD_DIM}")
    print(f"  FFN_HID: {FFN_HID}")
    print(f"  Tile size: {LINEAR_M}x{LINEAR_N}x{LINEAR_K}")
    
    results = []
    
    for test_case in test_cases:
        try:
            result = profile_linear_projection(test_case)
            results.append(result)
        except Exception as e:
            print(f"Error profiling {test_case.name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Test Case':<15} {'NPU (ms)':<12} {'PyTorch (ms)':<15} {'Speedup':<10} {'NPU GFLOPS':<12} {'Max Error':<12}")
    print("-" * 80)
    
    for result in results:
        npu_mean = np.mean(result['npu_times']) * 1000
        pytorch_mean = np.mean(result['pytorch_times']) * 1000
        speedup = result['speedup']
        gflops = result['npu_gflops']
        error = result['max_error']
        
        print(f"{result['test_case'].name:<15} {npu_mean:<12.3f} {pytorch_mean:<15.3f} "
              f"{speedup:<10.2f} {gflops:<12.2f} {error:<12.3e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Profile linear_projection submodule')
    parser.add_argument('--test-case', type=str, choices=[tc.name for tc in test_cases] + ['all'], 
                       default='all', help='Specific test case to run')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for averaging')
    
    args = parser.parse_args()
    
    if args.test_case == 'all':
        run_comprehensive_profile()
    else:
        test_case = next(tc for tc in test_cases if tc.name == args.test_case)
        profile_linear_projection(test_case, args.num_runs)

if __name__ == "__main__":
    main()
