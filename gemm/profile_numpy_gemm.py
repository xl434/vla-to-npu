#!/usr/bin/env python3
# Profile numpy GEMM on the CPU

"""
Simple numpy GEMM profiling script
Profiles numpy matrix multiplication with configurable dimensions and data types

Example usage: 
python profile_numpy_gemm.py --M 64 --N 960 --K 768 --dtype bfloat16
"""

import numpy as np
import time
import argparse

# Try to import bfloat16, fallback to float32 if not available
try:
    from ml_dtypes import bfloat16 as np_bfloat16
    HAS_BFLOAT16 = True
except ImportError:
    print("Warning: ml_dtypes not available, bfloat16 will use float32")
    np_bfloat16 = np.float32
    HAS_BFLOAT16 = False

# Profiling configuration
WARMUP = 20
NUM_ITERS = 100

def profile_numpy_gemm(M, N, K, dtype_str):
    """
    Profile numpy GEMM: C = A @ B where A is MxK, B is KxN, C is MxN
    
    Args:
        M: Number of rows in A and C
        N: Number of columns in B and C  
        K: Number of columns in A and rows in B
        dtype_str: Data type string ('float32', 'bfloat16', 'float16', 'int32')
    
    Returns:
        Dictionary with timing results and metadata
    """
    
    # Set numpy data type
    if dtype_str == 'float32':
        np_dtype = np.float32
    elif dtype_str == 'bfloat16':
        if not HAS_BFLOAT16:
            print(f"  Warning: bfloat16 not available, using float32 instead")
        np_dtype = np_bfloat16
    elif dtype_str == 'float16':
        np_dtype = np.float16
    elif dtype_str == 'int32':
        np_dtype = np.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    print(f"Profiling numpy GEMM:")
    print(f"  Matrix dimensions: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")
    print(f"  Data type: {dtype_str}")
    print(f"  Warmup iterations: {WARMUP}")
    print(f"  Timing iterations: {NUM_ITERS}")
    
    # Generate random input matrices
    if dtype_str in ['float32', 'float16', 'bfloat16']:
        A = np.random.randn(M, K).astype(np_dtype)
        B = np.random.randn(K, N).astype(np_dtype)
    else:  # int32
        A = np.random.randint(-100, 100, (M, K), dtype=np_dtype)
        B = np.random.randint(-100, 100, (K, N), dtype=np_dtype)
    
    print(f"  Input A: shape={A.shape}, dtype={A.dtype}")
    print(f"  Input B: shape={B.shape}, dtype={B.dtype}")
    
    # Warmup phase
    print(f"\nRunning {WARMUP} warmup iterations...")
    for _ in range(WARMUP):
        C = np.matmul(A, B)
    
    # Timing phase
    print(f"Running {NUM_ITERS} timing iterations...")
    times = []
    
    for i in range(NUM_ITERS):
        start_time = time.perf_counter()
        C = np.matmul(A, B)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{NUM_ITERS} iterations")
    
    # Calculate statistics
    times_us = [t * 1e6 for t in times]  # Convert to microseconds
    
    results = {
        'times_seconds': times,
        'times_microseconds': times_us,
        'mean_us': np.mean(times_us),
        'median_us': np.median(times_us),
        'min_us': np.min(times_us),
        'max_us': np.max(times_us),
        'std_us': np.std(times_us),
        'p95_us': np.percentile(times_us, 95),
        'p99_us': np.percentile(times_us, 99),
        'matrix_shape': (M, N, K),
        'dtype': dtype_str,
        'output_shape': C.shape,
        'output_dtype': str(C.dtype)
    }
    
    # Calculate GFLOPS
    total_ops = M * N * K * 2  # multiply-add operations
    gflops = total_ops / (results['mean_us'] * 1e-6) / 1e9
    results['gflops'] = gflops
    
    return results

def print_results(results):
    """Print formatted profiling results"""
    
    print(f"\n{'='*60}")
    print("PROFILING RESULTS")
    print(f"{'='*60}")
    
    print(f"Matrix: A{results['matrix_shape'][0]}x{results['matrix_shape'][2]} @ "
          f"B{results['matrix_shape'][2]}x{results['matrix_shape'][1]} -> "
          f"C{results['matrix_shape'][0]}x{results['matrix_shape'][1]}")
    print(f"Data type: {results['dtype']}")
    print(f"Output shape: {results['output_shape']}, dtype: {results['output_dtype']}")
    
    print(f"\nTiming Statistics (microseconds):")
    print(f"  Mean:   {results['mean_us']:.1f}us")
    print(f"  Median: {results['median_us']:.1f}us")
    print(f"  Min:    {results['min_us']:.1f}us")
    print(f"  Max:    {results['max_us']:.1f}us")
    print(f"  Std:    {results['std_us']:.1f}us")
    print(f"  P95:    {results['p95_us']:.1f}us")
    print(f"  P99:    {results['p99_us']:.1f}us")
    
    print(f"\nPerformance:")
    print(f"  Throughput: {results['gflops']:.2f} GFLOPS")
    
    print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Profile numpy GEMM operations')
    parser.add_argument('--M', type=int, default=64, help='Matrix A rows (default: 64)')
    parser.add_argument('--N', type=int, default=64, help='Matrix B columns (default: 64)')
    parser.add_argument('--K', type=int, default=64, help='Inner dimension (default: 64)')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'bfloat16', 'float16', 'int32'],
                       help='Data type (default: float32)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show raw timing data')
    
    args = parser.parse_args()
    
    # Run profiling
    results = profile_numpy_gemm(args.M, args.N, args.K, args.dtype)
    
    # Print results
    print_results(results)
    
    if args.verbose:
        print(f"\nRaw timing data (first 10 values in microseconds):")
        print(results['times_microseconds'][:10])
        print(f"... and {len(results['times_microseconds']) - 10} more values")

def quick_benchmark(shapes_and_dtypes):
    """
    Quick benchmark for multiple configurations
    
    Args:
        shapes_and_dtypes: List of tuples [(M, N, K, dtype), ...]
    """
    print("Quick Benchmark Results:")
    print("=" * 80)
    print(f"{'Shape (MxNxK)':<20} {'Dtype':<10} {'Mean (us)':<12} {'GFLOPS':<10}")
    print("-" * 80)
    
    for M, N, K, dtype in shapes_and_dtypes:
        results = profile_numpy_gemm(M, N, K, dtype)
        shape_str = f"{M}x{N}x{K}"
        print(f"{shape_str:<20} {dtype:<10} {results['mean_us']:<12.1f} {results['gflops']:<10.2f}")
        print()  # Add blank line for readability

if __name__ == "__main__":
    main()
