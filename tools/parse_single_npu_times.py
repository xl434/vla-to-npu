#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to parse individual NPU execution times from test logs

This script parses output files containing individual NPU execution times
and extracts them assuming the pattern [matmul, accumulate, matmul, accumulate, ...]

Expected input format:
NPU execution time: 2859us
NPU execution time: 2687us
...

Example usage: 
python parse_single_npu_times.py ../gemm/tiled_gemm_output/tiled_gemm_64x960x768_bf16_noprofile.output --output ../gemm/tiled_gemm_output/tiled_gemm_64x960x768_bf16_noprofile.json
"""

import re
import argparse
import sys
import numpy as np
from typing import List
import json

def parse_single_npu_times(file_path: str) -> List[float]:
    """
    Parse individual NPU timing data from output file
    
    Args:
        file_path: Path to the output file containing NPU timing data
        
    Returns:
        List of execution times in microseconds
    """
    times = []
    
    # Regular expression to match timing lines
    time_pattern = r"NPU execution time:\s*([\d.]+)us"
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Check for execution time
                time_match = re.search(time_pattern, line)
                if time_match:
                    try:
                        exec_time = float(time_match.group(1))
                        times.append(exec_time)
                    except ValueError as e:
                        print(f"Warning: Could not parse execution time '{time_match.group(1)}' on line {line_num}: {e}")
                        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return []
    
    return times

def analyze_times(times: List[float]):
    """
    Analyze timing data assuming [matmul, accumulate, matmul, accumulate, ...] pattern
    
    Args:
        times: List of execution times in microseconds
        
    Returns:
        Dictionary with analysis results
    """
    if len(times) == 0:
        return {}
    
    # Separate matmul and accumulate times
    # Even indices (0, 2, 4, ...) are matmul times
    # Odd indices (1, 3, 5, ...) are accumulate times
    matmul_times = times[::2]
    accumulate_times = times[1::2]
    
    # Calculate statistics
    results = {
        # Raw data
        'all_times': times,
        'matmul_times': matmul_times,
        'accumulate_times': accumulate_times,
        
        # Counts
        'total_count': len(times),
        'matmul_count': len(matmul_times),
        'accumulate_count': len(accumulate_times),
        
        # Sums
        'sum_all_times': sum(times),
        'sum_matmul_times': sum(matmul_times),
        'sum_accumulate_times': sum(accumulate_times),
        
        # Averages
        'avg_all_times': np.mean(times),
        'avg_matmul_times': np.mean(matmul_times),
        'avg_accumulate_times': np.mean(accumulate_times),
        
        # Min/Max
        'min_all_times': np.min(times),
        'max_all_times': np.max(times),
        'min_matmul_times': np.min(matmul_times),
        'max_matmul_times': np.max(matmul_times),
        'min_accumulate_times': np.min(accumulate_times),
        'max_accumulate_times': np.max(accumulate_times),
        
        # Standard deviation
        'std_all_times': np.std(times),
        'std_matmul_times': np.std(matmul_times),
        'std_accumulate_times': np.std(accumulate_times),
        
        # Percentiles
        'p95_all_times': np.percentile(times, 95),
        'p99_all_times': np.percentile(times, 99),
        'p95_matmul_times': np.percentile(matmul_times, 95),
        'p99_matmul_times': np.percentile(matmul_times, 99),
        'p95_accumulate_times': np.percentile(accumulate_times, 95),
        'p99_accumulate_times': np.percentile(accumulate_times, 99),
    }
    
    return results

def print_analysis(results: dict):
    """Print formatted analysis results"""
    
    print(f"\n{'='*80}")
    print("NPU EXECUTION TIME ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nData Overview:")
    print(f"  Total measurements: {results['total_count']}")
    print(f"  Matmul operations: {results['matmul_count']}")
    print(f"  Accumulate operations: {results['accumulate_count']}")
    
    if results['matmul_count'] != results['accumulate_count']:
        print(f"  ⚠ Warning: Unequal matmul/accumulate counts!")
    
    print(f"\nTiming Sums (microseconds):")
    print(f"  Total execution time: {results['sum_all_times']:.1f}us")
    print(f"  Matmul total time: {results['sum_matmul_times']:.1f}us ({results['sum_matmul_times']/results['sum_all_times']*100:.1f}%)")
    print(f"  Accumulate total time: {results['sum_accumulate_times']:.1f}us ({results['sum_accumulate_times']/results['sum_all_times']*100:.1f}%)")
    
    print(f"\nAverage Times (microseconds):")
    print(f"  Overall average: {results['avg_all_times']:.1f}us")
    print(f"  Matmul average: {results['avg_matmul_times']:.1f}us")
    print(f"  Accumulate average: {results['avg_accumulate_times']:.1f}us")
    
    print(f"\nTime Ranges (microseconds):")
    print(f"  Overall: {results['min_all_times']:.1f} - {results['max_all_times']:.1f}us")
    print(f"  Matmul: {results['min_matmul_times']:.1f} - {results['max_matmul_times']:.1f}us")
    print(f"  Accumulate: {results['min_accumulate_times']:.1f} - {results['max_accumulate_times']:.1f}us")
    
    print(f"\nStandard Deviation:")
    print(f"  Overall: {results['std_all_times']:.1f}us")
    print(f"  Matmul: {results['std_matmul_times']:.1f}us")
    print(f"  Accumulate: {results['std_accumulate_times']:.1f}us")
    
    print(f"\nPercentiles (95th/99th):")
    print(f"  Overall: {results['p95_all_times']:.1f}us / {results['p99_all_times']:.1f}us")
    print(f"  Matmul: {results['p95_matmul_times']:.1f}us / {results['p99_matmul_times']:.1f}us")
    print(f"  Accumulate: {results['p95_accumulate_times']:.1f}us / {results['p99_accumulate_times']:.1f}us")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Parse individual NPU execution times from output files')
    parser.add_argument('input_file', type=str,
                       help='Input file containing NPU timing data')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSON file to save analysis results')
    parser.add_argument('--show-times', action='store_true',
                       help='Show first 20 individual timing values')
    
    args = parser.parse_args()
    
    print(f"Parsing NPU execution times from: {args.input_file}")
    
    # Parse the timing data
    times = parse_single_npu_times(args.input_file)
    
    if not times:
        print("No NPU execution time data found in the input file")
        sys.exit(1)
    
    # Analyze the data
    results = analyze_times(times)
    
    # Print analysis
    print_analysis(results)
    
    if args.show_times:
        print(f"\nFirst 20 timing values:")
        print(f"  Matmul times: {results['matmul_times'][:10]}")
        print(f"  Accumulate times: {results['accumulate_times'][:10]}")
        print(f"  Pattern verification (first 20): {times[:20]}")
    
    # Save to JSON if requested
    if args.output:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"\nAnalysis results saved to: {args.output}")

    
    # Summary for easy parsing
    print(f"\nQUICK SUMMARY:")
    print(f"TOTAL_TIME_US: {results['sum_all_times']:.1f}")
    print(f"MATMUL_TIME_US: {results['sum_matmul_times']:.1f}")
    print(f"ACCUMULATE_TIME_US: {results['sum_accumulate_times']:.1f}")
    print(f"AVG_MATMUL_US: {results['avg_matmul_times']:.1f}")
    print(f"AVG_ACCUMULATE_US: {results['avg_accumulate_times']:.1f}")
    print(f"TOTAL_OPERATIONS: {results['total_count']}")

if __name__ == "__main__":
    main()
