#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single tile matmul profiling script - profiles just the core NPU matmul kernel

This script profiles a single 64x64x64 tile matmul operation without any tiling overhead,
giving us the pure performance of the NPU matmul kernel.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16, int32
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie import ExternalModule
import argparse
import statistics

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Configuration
# ===============================================================================
KERNEL_LIB_PATH = "../../cc/"

# profiling times
WARMUP = 20
NUM_ITERS = 100

# Default tile dimensions (from llama_block.py)
DEFAULT_TILE_M, DEFAULT_TILE_N, DEFAULT_TILE_K = 64, 64, 64
DEFAULT_DTYPE = "float32"

# Global variables to be set by command line arguments
TILE_M, TILE_N, TILE_K = DEFAULT_TILE_M, DEFAULT_TILE_N, DEFAULT_TILE_K
Ty = float32
matmul_mod = None  # Will be built after parsing arguments

print(f"Single Tile Matmul Profiling")

# ===============================================================================
# NPU Kernel Definition
# ===============================================================================

def build_matmul_kernel(tile_m, tile_n, tile_k, dtype_str):
    """Build the NPU matmul kernel with specified dimensions and dtype"""
    # Set data type
    if dtype_str == "float32":
        Ty = float32
        np_dtype = np.float32
    elif dtype_str == "bfloat16":
        Ty = bfloat16
        np_dtype = np_bfloat16
    elif dtype_str == "int32":
        Ty = int32
        np_dtype = np.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # Layout configurations
    A_layout = Layout("S0R")
    B_layout = Layout("RS1") 
    C_layout = Layout("S0S1")
    
    @df.region()
    def single_tile_matmul_kernel():
        @df.kernel(mapping=[4, 4])
        def gemm(
            A: Ty[tile_m, tile_k] @ A_layout,
            B: Ty[tile_k, tile_n] @ B_layout,
            C: Ty[tile_m, tile_n] @ C_layout,
        ):
            C[:, :] = allo.matmul(A, B)
    
    print(f"Building NPU matmul kernel...")
    print(f"  Tile dimensions: {tile_m} x {tile_n} x {tile_k}")
    print(f"  Data type: {dtype_str}")
    
    mod = df.build(
        single_tile_matmul_kernel, 
        target="aie", 
        project=f"single_tile_matmul.prj",
        profile=True,
        warmup=WARMUP,
        num_iters=NUM_ITERS
    )
    print("NPU kernel built successfully!")
    
    return mod, Ty, np_dtype

# ===============================================================================
# Profiling Functions
# ===============================================================================

def profile_npu_matmul(A, B, matmul_mod, np_dtype):
    """Profile NPU matmul kernel for a single tile"""
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np_dtype)
    # Actual profiling run
    matmul_mod(A, B, C)
        
    return C.astype(np.float32)

def profile_pytorch_matmul(A, B):
    """Profile PyTorch matmul for comparison"""    
    # Warmup runs
    for _ in range(WARMUP):
        torch.matmul(A, B)
    
    # Actual profiling runs
    times = []
    for run in range(NUM_ITERS):
        start_time = time.perf_counter()
        C_torch = torch.matmul(A, B)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return times, C_torch.to(dtype=torch.float32).numpy()

def profile_numpy_matmul(A, B):
    """Profile NumPy matmul for comparison"""
    # Warmup runs
    for _ in range(WARMUP):
        np.matmul(A, B)
    
    # Actual profiling runs
    times = []
    for run in range(NUM_ITERS):
        start_time = time.perf_counter()
        C_numpy = np.matmul(A, B)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    return times, C_numpy.astype(np.float32)

def calculate_stats(times):
    """Calculate comprehensive statistics"""
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'p95': sorted(times)[int(0.95 * len(times))],
        'p99': sorted(times)[int(0.99 * len(times))]
    }

def verify_correctness(C_npu, C_ref):
    """Verify NPU output matches reference"""
    np.testing.assert_allclose(C_npu, C_ref, atol=1e-1, rtol=1e-2)
    print(f"Result matches reference within tolerance ✔️")
    return True

def main():
    parser = argparse.ArgumentParser(description='Profile single tile matmul kernel')
    parser.add_argument('--tile-m', type=int, default=DEFAULT_TILE_M,
                       help=f'Matrix M dimension (default: {DEFAULT_TILE_M})')
    parser.add_argument('--tile-n', type=int, default=DEFAULT_TILE_N,
                       help=f'Matrix N dimension (default: {DEFAULT_TILE_N})')
    parser.add_argument('--tile-k', type=int, default=DEFAULT_TILE_K,
                       help=f'Matrix K dimension (default: {DEFAULT_TILE_K})')
    parser.add_argument('--dtype', type=str, default=DEFAULT_DTYPE,
                       choices=['float32', 'bfloat16', 'int32'],
                       help=f'Data type (default: {DEFAULT_DTYPE})')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed timing statistics')
    
    args = parser.parse_args()
    
    # Update global variables
    global TILE_M, TILE_N, TILE_K
    TILE_M, TILE_N, TILE_K = args.tile_m, args.tile_n, args.tile_k
    
    print(f"\nProfiling Configuration:")
    print(f"  Tile size: {TILE_M}x{TILE_N}x{TILE_K}")
    print(f"  Data type: {args.dtype}")
    
    # Build the NPU kernel with specified parameters
    matmul_mod, Ty, np_dtype = build_matmul_kernel(TILE_M, TILE_N, TILE_K, args.dtype)
    
    A = np.random.randn(TILE_M, TILE_K)
    B = np.random.randn(TILE_K, TILE_N)


    
    print(f"\nInput statistics:")
    print(f"  A: shape={A.shape}")
    print(f"  B: shape={B.shape}")
    


    # Profile NPU implementation
    print(f"\nProfiling NPU matmul kernel...")
    A_npu = A.astype(np_dtype)
    B_npu = B.astype(np_dtype)
    C_npu = profile_npu_matmul(A_npu, B_npu, matmul_mod, np_dtype)

    # profile torch matmul
    print(f"Profiling PyTorch matmul...")
    A_torch = torch.from_numpy(A).to(dtype=torch.bfloat16)
    B_torch = torch.from_numpy(B).to(dtype=torch.bfloat16)
    pytorch_times, C_pytorch = profile_pytorch_matmul(A_torch, B_torch)
    pytorch_stats = calculate_stats(pytorch_times)
    
    # profile numpy matmul
    print(f"Profiling NumPy matmul...")
    A_numpy = A.astype(np_dtype)
    B_numpy = B.astype(np_dtype)
    numpy_times, C_numpy = profile_numpy_matmul(A_numpy, B_numpy)
    numpy_stats = calculate_stats(numpy_times)

    
    # Verify correctness
    npu_vs_pytorch = verify_correctness(C_npu, C_pytorch)
    npu_vs_numpy = verify_correctness(C_npu, C_numpy)
    



    # Calculate operations and throughput
    total_ops = TILE_M * TILE_N * TILE_K * 2  # multiply-add operations
    
    # Results
    print(f"\n{'='*80}")
    print(f"PROFILING RESULTS")
    print(f"{'='*80}")
    
    print(f"\nTiming Results (microseconds):")
    print(f"{'Implementation':<12} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Std':<8} {'P95':<8} {'P99':<8}")
    print(f"{'-'*80}")
    
    implementations = [
        ('PyTorch', pytorch_stats),
        ('NumPy', numpy_stats)
    ]
    
    for name, stats in implementations:
        print(f"{name:<12} {stats['mean']*1e6:<8.1f} {stats['median']*1e6:<8.1f} "
              f"{stats['min']*1e6:<8.1f} {stats['max']*1e6:<8.1f} {stats['stdev']*1e6:<8.1f} "
              f"{stats['p95']*1e6:<8.1f} {stats['p99']*1e6:<8.1f}")
    
    print(f"\nThroughput (GFLOPS):")
    pytorch_gflops = total_ops / pytorch_stats['mean'] / 1e9
    numpy_gflops = total_ops / numpy_stats['mean'] / 1e9
    
    print(f"  PyTorch: {pytorch_gflops:.2f}")
    print(f"  NumPy:   {numpy_gflops:.2f}")
    
    
    if args.verbose:
        print(f"\nDetailed Statistics:")
        print(f"\nPyTorch timing distribution (microseconds):")
        pytorch_times_us = [t * 1e6 for t in pytorch_times]
        print(f"  Raw times: {pytorch_times_us[:10]}... (showing first 10)")
        
        print(f"\nNumPy timing distribution (microseconds):")
        numpy_times_us = [t * 1e6 for t in numpy_times]
        print(f"  Raw times: {numpy_times_us[:10]}... (showing first 10)")
        
        print(f"\nOutput matrix statistics:")
        print(f"  NPU output:     mean={C_npu.mean():.3f}, std={C_npu.std():.3f}")
        print(f"  PyTorch output: mean={C_pytorch.mean():.3f}, std={C_pytorch.std():.3f}")
        print(f"  NumPy output:   mean={C_numpy.mean():.3f}, std={C_numpy.std():.3f}")
    
    print(f"\n{'='*80}")
    
    # Summary for easy parsing
    print(f"\nSUMMARY:")
    print(f"TILE_DIMENSIONS: {TILE_M}x{TILE_N}x{TILE_K}")
    print(f"DATA_TYPE: {args.dtype}")
    print(f"PYTORCH_MEAN_TIME_US: {pytorch_stats['mean']*1e6:.1f}")
    print(f"NUMPY_MEAN_TIME_US: {numpy_stats['mean']*1e6:.1f}")
    print(f"PYTORCH_GFLOPS: {pytorch_gflops:.2f}")
    print(f"NUMPY_GFLOPS: {numpy_gflops:.2f}")
    

if __name__ == "__main__":
    main()
