#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DTYPES = ["i8", "i16", "bf16"]

# Simple in-file config for quick runs (used when CLI args are omitted)
DEFAULT_INPUTS = [
    (33,47,45), (63,63,63), (129,129,129), (257,257,257), (513,513,513), (769,769,769),
    (511,65,257), (1023,129,513), (1537,257,769), 
    (65,511,257), (129,1023,513), (257,1537,769),
    (129,129,1023), (257,257,1537), (513,513,1900),
    (512,512,33), (1024,768,47), (1536,1024,63)
]
DEFAULT_DTYPE = "all"
DEFAULT_PROFILE = "../profiling/data/vla_profile.json"
DEFAULT_OUT = "padding_sweep_profile.json"


def parse_inputs(raw):
    out = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        m_str, n_str, k_str = [x.strip() for x in part.split(",")]
        out.append((int(m_str), int(n_str), int(k_str)))
    if not out:
        raise ValueError("No valid --inputs parsed.")
    return out


def parse_profile_shapes(profile):
    shapes = []
    for key in profile.keys():
        m_str, n_str, k_str = key.split("_")
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return sorted(shapes)


def candidate_shapes(shapes, M0, N0, K0):
    return [(M, N, K) for (M, N, K) in shapes if M >= M0 and N >= N0 and K >= K0]


def run_case(
    script_path,
    M0,
    N0,
    K0,
    Mp,
    Np,
    Kp,
    tile,
    dtype,
    pad_impl,
    project_dir=None,
):
    m, n, k = tile
    cmd = [
        sys.executable,
        str(script_path),
        "--use-padding",
        "--M",
        str(M0),
        "--N",
        str(N0),
        "--K",
        str(K0),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--dtype",
        dtype,
        "--pad-M",
        str(Mp),
        "--pad-N",
        str(Np),
        "--pad-K",
        str(Kp),
        "--pad-impl",
        pad_impl,
    ]
    if project_dir is not None:
        cmd.extend(["--project-dir", str(project_dir)])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_bytes, _ = proc.communicate()
    output = out_bytes.decode("utf-8", errors="ignore")
    print(output, end="")
    return output, proc.returncode


def parse_output(raw, dtype):
    pad_match = re.search(rf"\[{dtype}\] Padding time:\s*([\d\.]+)us", raw)
    unpad_match = re.search(rf"\[{dtype}\] Unpadding time:\s*([\d\.]+)us", raw)
    npu_matches = re.findall(r"Avg NPU execution time:\s*([\d\.]+)us", raw)
    passed = "PASSED!" in raw
    return {
        "padding_us": float(pad_match.group(1)) if pad_match else None,
        "unpadding_us": float(unpad_match.group(1)) if unpad_match else None,
        "mapping_runtime_us": float(npu_matches[-1]) if npu_matches else None,
        "mapping_executed": bool(npu_matches),
        "correct": passed,
        "error": (not passed),
    }


def shape_key(M, N, K):
    return f"{M},{N},{K}"


def sort_key_for_dtype(entry):
    return (
        entry.get("padding_us") is None,
        float("inf") if entry.get("padding_us") is None else entry.get("padding_us"),
    )


def sort_dtype_entries(entries):
    sorted_items = sorted(entries.items(), key=lambda kv: sort_key_for_dtype(kv[1]))
    return {k: v for k, v in sorted_items}


def write_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep all profiled padded shapes >= input and measure padding overhead."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help='Semicolon-separated M0,N0,K0 triplets, e.g. "300,500,700;777,801,333"',
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="Path to vla_profile.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["all", "i8", "i16", "bf16"],
        default=DEFAULT_DTYPE,
        help="Run one dtype or all",
    )
    parser.add_argument(
        "--pad-impl",
        type=str,
        choices=["manual_copy", "torch_pad", "numpy_pad"],
        default="numpy_pad",
        help="Padding implementation passed through to v2_test_mapping_large_gemm.py",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Optional AIE project directory (use unique paths for async runs).",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    profile_path = Path(args.profile) if os.path.isabs(args.profile) else (base / args.profile)
    out_path = Path(args.out) if os.path.isabs(args.out) else (base / args.out)
    gemm_script = base.parent / "scripts" / "v2_test_mapping_large_gemm.py"
    dtypes = DTYPES if args.dtype == "all" else [args.dtype]
    project_dir = None if args.project_dir is None else Path(args.project_dir)

    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f) or {}

    shapes = parse_profile_shapes(profile)
    if args.inputs:
        inputs = parse_inputs(args.inputs)
    else:
        inputs = DEFAULT_INPUTS
    results = {"profile_path": str(profile_path), "inputs": {}}

    for M0, N0, K0 in inputs:
        input_key = shape_key(M0, N0, K0)
        results["inputs"][input_key] = {dtype: {} for dtype in dtypes}
        cands = candidate_shapes(shapes, M0, N0, K0)
        print(f"[INPUT] ({M0}, {N0}, {K0}) -> {len(cands)} candidate padded shapes")

        for Mp, Np, Kp in cands:
            cand_profile_key = f"{Mp}_{Np}_{Kp}"
            cand_out_key = shape_key(Mp, Np, Kp)
            shape_entry = profile.get(cand_profile_key, {})

            for dtype in dtypes:
                dtype_entry = shape_entry.get(dtype, {})
                tile = dtype_entry.get("Best m,n,k", {}).get("size")
                if not tile:
                    results["inputs"][input_key][dtype][cand_out_key] = {
                        "selected m,n,k": None,
                        "padding_us": None,
                        "unpadding_us": None,
                        "mapping_runtime_us": None,
                        "mapping_executed": False,
                        "correct": False,
                        "error": True,
                    }
                    write_results(out_path, results)
                    continue

                print(
                    f"  [CAND] {cand_out_key} dtype={dtype} "
                    f"best_tile={tile[0]},{tile[1]},{tile[2]}"
                )
                raw, code = run_case(
                    gemm_script,
                    M0,
                    N0,
                    K0,
                    Mp,
                    Np,
                    Kp,
                    tile,
                    dtype,
                    args.pad_impl,
                    project_dir=project_dir,
                )
                metrics = parse_output(raw, dtype)
                metrics["selected m,n,k"] = tile
                if code != 0:
                    metrics["error"] = True
                    metrics["correct"] = False
                results["inputs"][input_key][dtype][cand_out_key] = metrics
                print(
                    f"    parsed_npu_runtime_us={metrics['mapping_runtime_us']} "
                    f"mapping_executed={metrics['mapping_executed']} "
                    f"correct={metrics['correct']}"
                )
                write_results(out_path, results)

        for dtype in dtypes:
            results["inputs"][input_key][dtype] = sort_dtype_entries(
                results["inputs"][input_key][dtype]
            )
        write_results(out_path, results)

    write_results(out_path, results)
    print(f"[DONE] Saved sweep results to {out_path}")


if __name__ == "__main__":
    main()
