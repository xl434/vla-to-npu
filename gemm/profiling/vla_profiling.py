import subprocess
import re
import os
import json
import argparse
import sys
from pathlib import Path


def run_experiment(script_path, M, N, K, m, n, k):
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(script_path),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--m", str(m), "--n", str(n), "--k", str(k),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out_bytes, _ = proc.communicate()
    output = out_bytes.decode("utf-8", errors="ignore")
    print(output, end="")
    return output

def extract_timings(raw):
    matches = re.findall(r"Avg NPU execution time:\s*([\d\.]+)us", raw)
    avgs = [float(x) for x in matches]
    # pad to three entries (i8, i16, bf16) using None for missing values
    while len(avgs) < 3:
        avgs.append(None)
    return avgs[:3]

def update_json(profile_path, error_path, M, N, K, m, n, k, dtype, avg_time, error_message=None):
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with profile_path.open(encoding="utf-8") as pf:
            profile = json.load(pf) or {}
    except (FileNotFoundError, json.JSONDecodeError):
        profile = {}
    key = f"{M}_{N}_{K}"
    if key not in profile:
        profile[key] = {
            dt: {
                "Best m,n,k": {"size": None, "Best NPU average time": None},
                "Working m,n,k": {},
                "Failing m,n,k": []
            } for dt in ["i8", "i16", "bf16"]
        }
    sect = profile[key][dtype]
    size = [m, n, k]
    if error_message is not None:
        if size not in sect["Failing m,n,k"]:
            sect["Failing m,n,k"].append(size)
    else:
        ws = sect["Working m,n,k"]
        key_tile = f"{m}_{n}_{k}"
        if key_tile not in ws:
            ws[key_tile] = {"size": size, "avg": avg_time, "count": 1}
        else:
            e = ws[key_tile]
            e["count"] += 1
            e["avg"] = (e["avg"] * (e["count"] - 1) + avg_time) / e["count"]
        _, best_entry = min(ws.items(), key=lambda item: item[1]["avg"])
        sect["Best m,n,k"]["size"] = best_entry["size"]
        sect["Best m,n,k"]["Best NPU average time"] = best_entry["avg"]
    # write main profile
    with profile_path.open("w", encoding="utf-8") as pf:
        json.dump(profile, pf, indent=2)

    # log full errors separately
    if error_message is not None:
        try:
            with error_path.open(encoding="utf-8") as ef:
                errors = json.load(ef) or {}
        except (FileNotFoundError, json.JSONDecodeError):
            errors = {}
        dtype_map = errors.setdefault(key, {})
        dtype_entries = dtype_map.setdefault(dtype, {})
        dtype_entries[f"{m}_{n}_{k}"] = error_message.splitlines()
        with error_path.open("w", encoding="utf-8") as ef:
            json.dump(errors, ef, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run VLA profiling for a single GEMM configuration")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    gemm_base = base.parent
    data_dir = base / "data"
    profile_path = data_dir / "vla_profile.json"
    error_path = data_dir / "profile_errors.json"
    script = gemm_base / "scripts" / "v2_test_mapping_large_gemm.py"
    header = f"Profiling M={args.M}, N={args.N}, K={args.K}, m={args.m}, n={args.n}, k={args.k}\n"
    print(header, end="")
    raw_output = run_experiment(script, args.M, args.N, args.K, args.m, args.n, args.k)
    combined_output = header + raw_output

    avgs = extract_timings(combined_output)
    error_msg = None
    if any(avg is None for avg in avgs):
        error_msg = combined_output
        for marker in ["Traceback", "[AIE ERROR]", "terminate called", "RuntimeError:"]:
            idx = combined_output.find(marker)
            if idx != -1:
                error_msg = combined_output[idx:]
                break

    for dtype, avg in zip(["i8", "i16", "bf16"], avgs):
        dtype_error = error_msg if avg is None else None
        update_json(
            profile_path,
            error_path,
            args.M,
            args.N,
            args.K,
            args.m,
            args.n,
            args.k,
            dtype,
            avg,
            dtype_error,
        )

if __name__ == "__main__":
    main()
