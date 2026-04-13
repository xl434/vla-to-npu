"""
Measure WHERE the ~24ms dispatch overhead comes from.
Breaks down into:
  1. Python file write (numpy → disk)
  2. Subprocess spawn + C++ execution
  3. Python file read (disk → numpy)

For the C++ side, we also time the binary directly to separate:
  - Process startup
  - XRT setup (device, xclbin, context, kernel, buffers)
  - DMA sync to device
  - Kernel launch + wait (NPU time)
  - DMA sync from device
  - File I/O in C++

Usage:
  cd vla && python ../analysis/measure_overhead_breakdown.py
"""

import os
import sys
import re
import json
import time
import subprocess
import tempfile
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

# Add vla directory to path
VLA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vla")
sys.path.insert(0, VLA_DIR)
os.chdir(VLA_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "overhead_breakdown_data.json")

# We need to import to trigger the build, but we'll also manually
# replicate the __call__ steps to time them individually
from text_encoder_bf16 import rms_norm_mod as text_rms_mod


def get_module_info(mod):
    """Extract project_dir and tensor info from an allo AIE module."""
    return {
        "project_dir": mod.project_dir,
        "global_tensors": mod.global_tensors,
    }


def measure_phases(mod, args, n_trials=10):
    """
    Manually replicate mod.__call__ but time each phase separately.
    """
    info = get_module_info(mod)
    proj_dir = info["project_dir"]
    tensors = info["global_tensors"]

    # Identify input/output tensors
    input_tensors = {}
    output_tensors = {}
    for idx, dt in tensors.items():
        if dt.is_input:
            input_tensors[idx] = dt
        else:
            output_tensors[idx] = dt

    # Build the command (same as allo does)
    trace_size = getattr(mod, "trace_size", 4096)
    cmd = f"cd {proj_dir} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz {trace_size}"

    # Warmup
    mod(*args)

    records = []
    for trial in range(n_trials):
        # Phase 1: Python writes input files
        t0 = time.perf_counter()
        for idx, dt in input_tensors.items():
            fpath = os.path.join(proj_dir, f"input{idx}.data")
            with open(fpath, "wb") as f:
                f.write(args[idx].tobytes())
        t1 = time.perf_counter()

        # Phase 2: Subprocess (C++ binary: XRT setup + file read + DMA + kernel + DMA + file write)
        t2 = time.perf_counter()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        t3 = time.perf_counter()

        # Parse NPU time from stdout
        npu_ms = None
        match = re.search(r"NPU execution time:\s*([\d.]+)us", result.stdout)
        if match:
            npu_ms = float(match.group(1)) / 1000.0

        # Phase 3: Python reads output files
        t4 = time.perf_counter()
        for idx, dt in output_tensors.items():
            fpath = os.path.join(proj_dir, f"output{idx}.data")
            data = np.fromfile(fpath, dtype=args[idx].dtype)
            args[idx].flat[:] = data
        t5 = time.perf_counter()

        file_write_ms = (t1 - t0) * 1000
        subprocess_ms = (t3 - t2) * 1000
        file_read_ms = (t5 - t4) * 1000
        total_ms = (t5 - t0) * 1000
        cpp_overhead_ms = subprocess_ms - (npu_ms if npu_ms else 0)

        rec = {
            "trial": trial + 1,
            "file_write_ms": round(file_write_ms, 3),
            "subprocess_ms": round(subprocess_ms, 3),
            "file_read_ms": round(file_read_ms, 3),
            "npu_ms": round(npu_ms, 3) if npu_ms else None,
            "cpp_overhead_ms": round(cpp_overhead_ms, 3) if npu_ms else None,
            "total_ms": round(total_ms, 3),
        }
        records.append(rec)
        print(f"  trial {trial+1:2d}: write={file_write_ms:.2f}  subprocess={subprocess_ms:.2f}  "
              f"read={file_read_ms:.2f}  npu={npu_ms:.2f}  cpp_oh={cpp_overhead_ms:.2f}  total={total_ms:.2f} ms")

    return records


def measure_subprocess_only(proj_dir, n_trials=10):
    """
    Time the C++ binary without any Python file I/O overhead.
    Input files already exist from previous calls.
    """
    trace_size = 4096
    cmd = f"cd {proj_dir} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE --trace_sz {trace_size}"

    records = []
    for trial in range(n_trials):
        t0 = time.perf_counter()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        t1 = time.perf_counter()

        npu_ms = None
        match = re.search(r"NPU execution time:\s*([\d.]+)us", result.stdout)
        if match:
            npu_ms = float(match.group(1)) / 1000.0

        wall = (t1 - t0) * 1000
        cpp_oh = wall - npu_ms if npu_ms else None
        records.append({
            "trial": trial + 1,
            "subprocess_ms": round(wall, 3),
            "npu_ms": round(npu_ms, 3) if npu_ms else None,
            "cpp_overhead_ms": round(cpp_oh, 3) if cpp_oh else None,
        })
    return records


def measure_empty_subprocess(n_trials=20):
    """Measure baseline cost of spawning a subprocess (no XRT)."""
    records = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        subprocess.run(["true"], capture_output=True)
        t1 = time.perf_counter()
        records.append((t1 - t0) * 1000)
    return records


def measure_file_io(sizes_bytes, n_trials=10):
    """Measure raw file write + read times for various sizes."""
    results = {}
    for size in sizes_bytes:
        data = np.random.bytes(size)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        write_times = []
        read_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            with open(tmp.name, "wb") as f:
                f.write(data)
            t1 = time.perf_counter()
            with open(tmp.name, "rb") as f:
                _ = f.read()
            t2 = time.perf_counter()
            write_times.append((t1 - t0) * 1000)
            read_times.append((t2 - t1) * 1000)

        os.unlink(tmp.name)
        results[size] = {
            "write_ms": round(np.median(write_times), 3),
            "read_ms": round(np.median(read_times), 3),
        }
    return results


def main():
    bf16 = np_bfloat16

    # Use RMSNorm as test kernel: 16x960 bf16 inputs
    norm_in = np.random.randn(16, 960).astype(bf16)
    norm_w = np.random.randn(960).astype(bf16)
    norm_out = np.zeros((16, 960), dtype=bf16)

    print("=" * 70)
    print("1. PHASE-BY-PHASE BREAKDOWN (RMSNorm 16x960)")
    print("=" * 70)
    phase_data = measure_phases(text_rms_mod, [norm_in, norm_w, norm_out], n_trials=10)

    print("\n" + "=" * 70)
    print("2. SUBPROCESS-ONLY (no Python file I/O, input files pre-existing)")
    print("=" * 70)
    proj_dir = text_rms_mod.project_dir
    sub_data = measure_subprocess_only(proj_dir, n_trials=10)
    for r in sub_data:
        print(f"  trial {r['trial']:2d}: subprocess={r['subprocess_ms']:.2f}  "
              f"npu={r['npu_ms']:.2f}  cpp_oh={r['cpp_overhead_ms']:.2f} ms")

    print("\n" + "=" * 70)
    print("3. EMPTY SUBPROCESS BASELINE (just `true`)")
    print("=" * 70)
    empty_times = measure_empty_subprocess(n_trials=20)
    print(f"  median: {np.median(empty_times):.3f} ms")
    print(f"  min:    {np.min(empty_times):.3f} ms")
    print(f"  max:    {np.max(empty_times):.3f} ms")

    print("\n" + "=" * 70)
    print("4. RAW FILE I/O BASELINE (various sizes)")
    print("=" * 70)
    sizes = [1024, 30720, 122880, 1966080, 3932160]  # 1KB, 30KB, 120KB, ~2MB, ~4MB
    io_data = measure_file_io(sizes, n_trials=20)
    for size, times in io_data.items():
        print(f"  {size:>8d} bytes: write={times['write_ms']:.3f}ms  read={times['read_ms']:.3f}ms")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg = lambda key: np.mean([r[key] for r in phase_data if r[key] is not None])
    print(f"\nAverage per-call breakdown (RMSNorm 16x960):")
    print(f"  Python file write:     {avg('file_write_ms'):6.2f} ms")
    print(f"  Subprocess total:      {avg('subprocess_ms'):6.2f} ms")
    print(f"    - C++ overhead:      {avg('cpp_overhead_ms'):6.2f} ms  (XRT setup + file I/O in C++ + DMA)")
    print(f"    - NPU time:          {avg('npu_ms'):6.2f} ms  (kernel launch + compute)")
    print(f"  Python file read:      {avg('file_read_ms'):6.2f} ms")
    print(f"  ────────────────────────────────")
    print(f"  Total wall-clock:      {avg('total_ms'):6.2f} ms")
    print(f"\nBaselines:")
    print(f"  Empty subprocess:      {np.median(empty_times):6.3f} ms")

    # C++ overhead further breakdown (estimated):
    sub_avg = np.mean([r["subprocess_ms"] for r in sub_data])
    npu_avg = np.mean([r["npu_ms"] for r in sub_data if r["npu_ms"]])
    cpp_oh = sub_avg - npu_avg
    proc_startup = np.median(empty_times)
    xrt_plus_io = cpp_oh - proc_startup

    print(f"\nC++ overhead breakdown (estimated):")
    print(f"  Process startup:       {proc_startup:6.2f} ms")
    print(f"  XRT setup + C++ I/O:   {xrt_plus_io:6.2f} ms  (device, xclbin, context, kernel, buffers, file read/write)")
    print(f"  NPU (launch+compute):  {npu_avg:6.2f} ms")

    # Save all data
    all_data = {
        "phase_breakdown": phase_data,
        "subprocess_only": sub_data,
        "empty_subprocess_ms": empty_times,
        "file_io_baseline": {str(k): v for k, v in io_data.items()},
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nData saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
