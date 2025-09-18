#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to parse NPU timing output from test logs

This script parses output files containing NPU execution times and extracts
average and minimum timing values into separate lists.

Expected input format:
Avg NPU execution time: 139.1us
Min NPU execution time: 107us

Example usage:
python parse_npu_times.py ../gemm/tiled_gemm_output/tiled_gemm_128x128x128_bf16.output --output ../gemm/tiled_gemm_output/tiled_gemm_128x128x128_bf16.json
"""

import re
import argparse
import sys
import numpy as np
from typing import List, Tuple
import json

def parse_npu_times(file_path: str) -> Tuple[List[float], List[float]]:
    """
    Parse NPU timing data from output file
    
    Args:
        file_path: Path to the output file containing NPU timing data
        
    Returns:
        Tuple of (avg_times, min_times) lists in microseconds
    """
    avg_times = []
    min_times = []
    
    # Regular expressions to match timing lines
    avg_pattern = r"Avg NPU execution time:\s*([\d.]+)us"
    min_pattern = r"Min NPU execution time:\s*([\d.]+)us"
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Check for average time
                avg_match = re.search(avg_pattern, line)
                if avg_match:
                    try:
                        avg_time = float(avg_match.group(1))
                        avg_times.append(avg_time)
                    except ValueError as e:
                        print(f"Warning: Could not parse average time '{avg_match.group(1)}' on line {line_num}: {e}")
                
                # Check for minimum time
                min_match = re.search(min_pattern, line)
                if min_match:
                    try:
                        min_time = float(min_match.group(1))
                        min_times.append(min_time)
                    except ValueError as e:
                        print(f"Warning: Could not parse minimum time '{min_match.group(1)}' on line {line_num}: {e}")
                        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return [], []
    except IOError as e:
        print(f"Error reading file '{file_path}': {e}")
        return [], []
    
    return avg_times, min_times


def main():
    parser = argparse.ArgumentParser(description='Parse NPU timing data from output files')
    parser.add_argument('input_file', type=str,
                       help='Input file containing NPU timing data')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file to save parsed data')
    
    args = parser.parse_args()
    
    print(f"Parsing NPU timing data from: {args.input_file}")
    
    # Parse the timing data
    avg_times, min_times = parse_npu_times(args.input_file)
    
    if not avg_times and not min_times:
        print("No NPU timing data found in the input file")
        sys.exit(1)
    
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Found total of {len(avg_times)} average time entries: {len(avg_times)/2} matmul time entries and {len(avg_times)/2} accumulate time entries")

    # avg_times array is: [matmul_time, accumulate_time, matmul_time, accumulate_time, ...]
    # min_times array is: [matmul_time, accumulate_time, matmul_time, accumulate_time, ...]
    # we want to calculate the average and minimum of the matmul_time and accumulate_time
    matmul_avg_times = avg_times[::2]
    accumulate_avg_times = avg_times[1::2]
    
    print(f"Sum of avg times: {sum(avg_times):.1f}us")

    print(f"Matmul average time: {np.mean(matmul_avg_times):.1f}us")
    print(f"Matmul range: {np.min(matmul_avg_times):.1f}us - {np.max(matmul_avg_times):.1f}us")
    print(f"Matmul times: {matmul_avg_times}")
    print(f"Sum of matmul times: {sum(matmul_avg_times):.1f}us")
    print("\n")
    print(f"Accumulate average time: {np.mean(accumulate_avg_times):.1f}us")
    print(f"Accumulate range: {np.min(accumulate_avg_times):.1f}us - {np.max(accumulate_avg_times):.1f}us")
    print(f"Accumulate times: {accumulate_avg_times}")
    print(f"Sum of accumulate times: {sum(accumulate_avg_times):.1f}us")

    # Save in json format about all the info
    with open(args.output, 'w') as f:
        json.dump({
            # sum
            'sum_avg_times': sum(avg_times),
            'sum_matmul_times': sum(matmul_avg_times),
            'sum_accumulate_times': sum(accumulate_avg_times),
            # size
            'size_avg_times': len(avg_times),
            'size_matmul_times': len(matmul_avg_times),
            'size_accumulate_times': len(accumulate_avg_times),
            # avg
            'avg_times': np.mean(avg_times),
            'avg_matmul_times': np.mean(matmul_avg_times),
            'avg_accumulate_times': np.mean(accumulate_avg_times),
            # range
            'min_avg_times': np.min(avg_times),
            'min_matmul_times': np.min(matmul_avg_times),
            'min_accumulate_times': np.min(accumulate_avg_times),
            # times
            'times_avg_times': avg_times,
            'times_matmul_times': matmul_avg_times,
            'times_accumulate_times': accumulate_avg_times,
        }, 
        f,
        indent=4)



if __name__ == "__main__":
    main()
