#!/usr/bin/env python3
"""Pure padding-only sweep runner (no Allo/NPU dependency)."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from utils import DEFAULT_INPUTS, DTYPES, dedupe_inputs, load_inputs_file, parse_inputs_arg, shape_key


Shape3D = Tuple[int, int, int]


def _parse_profile_shapes(profile_path: Path) -> List[Shape3D]:
    with profile_path.open("r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    out = []
    for key in raw.keys():
        parts = key.split("_")
        if len(parts) != 3:
            continue
        try:
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return sorted(set(out))


def _candidate_shapes(shapes: Sequence[Shape3D], inp: Shape3D) -> List[Shape3D]:
    m0, n0, k0 = inp
    return [(m, n, k) for (m, n, k) in shapes if m >= m0 and n >= n0 and k >= k0]


def _build_workload_estimate(
    inputs: Sequence[Shape3D],
    profile_shapes: Sequence[Shape3D],
    dtypes: Sequence[str],
    pad_impls: Sequence[str],
    warmup: int,
    repeats: int,
) -> dict:
    per_input = []
    total_candidates = 0
    for inp in inputs:
        cands = _candidate_shapes(profile_shapes, inp)
        count = len(cands)
        total_candidates += count
        per_input.append(
            {
                "input_shape": list(inp),
                "candidate_count": count,
            }
        )

    num_jobs = len(dtypes) * len(pad_impls)
    pad_calls_per_candidate = 2  # pad A + pad B
    total_pad_calls_per_job = total_candidates * (max(0, warmup) + max(1, repeats)) * pad_calls_per_candidate

    return {
        "num_inputs": len(inputs),
        "num_profile_shapes": len(profile_shapes),
        "num_jobs": num_jobs,
        "dtypes": list(dtypes),
        "pad_impls": list(pad_impls),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "per_input": per_input,
        "total_candidates_per_job": total_candidates,
        "total_pad_calls_per_job": total_pad_calls_per_job,
        "total_pad_calls_all_jobs": total_pad_calls_per_job * num_jobs,
    }


def _dtype_np(dtype: str):
    if dtype == "i8":
        return np.int8
    if dtype == "i16":
        return np.int16
    if dtype == "bf16":
        # Keep host-only and dependency-light: emulate via float32 data path.
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _make_inputs(m: int, n: int, k: int, dtype: str, rng: np.random.Generator):
    np_dtype = _dtype_np(dtype)
    if dtype == "bf16":
        a = (rng.random((m, k), dtype=np.float32) * np.float32(0.1)).astype(np_dtype, copy=False)
        b = (rng.random((k, n), dtype=np.float32) * np.float32(0.1)).astype(np_dtype, copy=False)
        return a, b
    if dtype == "i8":
        a = rng.integers(-4, 4, size=(m, k), dtype=np_dtype)
        b = rng.integers(-4, 4, size=(k, n), dtype=np_dtype)
        return a, b
    a = rng.integers(-8, 8, size=(m, k), dtype=np_dtype)
    b = rng.integers(-8, 8, size=(k, n), dtype=np_dtype)
    return a, b


def _pad_copy(arr: np.ndarray, add_rows: int, add_cols: int) -> np.ndarray:
    out = np.zeros((arr.shape[0] + add_rows, arr.shape[1] + add_cols), dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1]] = arr
    return out


def _pad_numpy(arr: np.ndarray, add_rows: int, add_cols: int) -> np.ndarray:
    return np.pad(arr, ((0, add_rows), (0, add_cols)), mode="constant", constant_values=0)


PAD_IMPLS = {
    "manual_copy": _pad_copy,
    "numpy_pad": _pad_numpy,
}


def _measure_candidate(
    pad_impl: str,
    dtype: str,
    input_shape: Shape3D,
    padded_shape: Shape3D,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    m0, n0, k0 = input_shape
    mp, np_, kp = padded_shape
    add_m = mp - m0
    add_n = np_ - n0
    add_k = kp - k0

    if add_m < 0 or add_n < 0 or add_k < 0:
        return {
            "padding_us": None,
            "unpadding_us": None,
            "padding_mean_us": None,
            "padding_std_us": None,
            "num_repeats": repeats,
            "error": True,
            "error_message": "Negative padding increment.",
        }

    rng = np.random.default_rng(seed)
    a, b = _make_inputs(m0, n0, k0, dtype=dtype, rng=rng)
    c_pad = np.zeros((mp, np_), dtype=_dtype_np(dtype))
    pad_fn = PAD_IMPLS[pad_impl]

    for _ in range(max(0, warmup)):
        _ = pad_fn(a, add_m, add_k)
        _ = pad_fn(b, add_k, add_n)
        _ = c_pad[:m0, :n0]

    pad_times = []
    unpad_times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter_ns()
        _ = pad_fn(a, add_m, add_k)
        _ = pad_fn(b, add_k, add_n)
        t1 = time.perf_counter_ns()
        _ = c_pad[:m0, :n0]
        t2 = time.perf_counter_ns()

        pad_times.append((t1 - t0) / 1000.0)
        unpad_times.append((t2 - t1) / 1000.0)

    return {
        "padding_us": float(statistics.median(pad_times)),
        "unpadding_us": float(statistics.median(unpad_times)),
        "padding_mean_us": float(statistics.fmean(pad_times)),
        "padding_std_us": float(statistics.pstdev(pad_times)) if len(pad_times) > 1 else 0.0,
        "num_repeats": int(max(1, repeats)),
        "mapping_runtime_us": None,
        "mapping_executed": False,
        "correct": True,
        "error": False,
    }


def _sort_entries(entries: Dict[str, dict]) -> Dict[str, dict]:
    sorted_items = sorted(
        entries.items(),
        key=lambda kv: (
            kv[1].get("padding_us") is None,
            float("inf") if kv[1].get("padding_us") is None else float(kv[1]["padding_us"]),
        ),
    )
    return {k: v for k, v in sorted_items}


def _run_worker(job: dict) -> dict:
    t0 = time.perf_counter()
    pad_impl = job["pad_impl"]
    dtype = job["dtype"]
    inputs: List[Shape3D] = [tuple(x) for x in job["inputs"]]
    shapes: List[Shape3D] = [tuple(x) for x in job["profile_shapes"]]
    warmup = int(job["warmup"])
    repeats = int(job["repeats"])
    seed = int(job["seed"])

    out = {"inputs": {}}
    total_candidates = 0
    for idx, inp in enumerate(inputs):
        input_key = shape_key(inp)
        out["inputs"][input_key] = {dtype: {}}
        cands = _candidate_shapes(shapes, inp)
        total_candidates += len(cands)
        for cand_idx, cand in enumerate(cands):
            padded_key = shape_key(cand)
            metrics = _measure_candidate(
                pad_impl=pad_impl,
                dtype=dtype,
                input_shape=inp,
                padded_shape=cand,
                warmup=warmup,
                repeats=repeats,
                seed=seed + (idx * 100003) + cand_idx,
            )
            out["inputs"][input_key][dtype][padded_key] = metrics
        out["inputs"][input_key][dtype] = _sort_entries(out["inputs"][input_key][dtype])
    return {
        "pad_impl": pad_impl,
        "dtype": dtype,
        "num_inputs": len(inputs),
        "num_candidates": total_candidates,
        "elapsed_s": time.perf_counter() - t0,
        "payload": out,
    }


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _generate_random_inputs(count: int, seed: int, min_dim: int, max_dim: int) -> List[Shape3D]:
    if count <= 0:
        return []
    if min_dim < 1 or max_dim < min_dim:
        raise ValueError("Invalid random min/max values")

    rng = random.Random(seed)
    lo = math.log2(float(min_dim))
    hi = math.log2(float(max_dim))
    out: List[Shape3D] = []
    for _ in range(count):
        dims = []
        for _ in range(3):
            d = int(round(2.0 ** rng.uniform(lo, hi)))
            d = max(min_dim, min(max_dim, d))
            dims.append(d)
        out.append((dims[0], dims[1], dims[2]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pure padding-only sweep for manual_copy/numpy_pad over all feasible padded shapes "
            "from vla_profile.json."
        )
    )
    parser.add_argument("--inputs", type=str, default=None, help='Semicolon-separated "M,N,K;..."')
    parser.add_argument("--inputs-file", type=Path, default=None, help="JSON file containing inputs")
    parser.add_argument("--include-default-inputs", action="store_true")
    parser.add_argument("--random-count", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--random-min", type=int, default=33)
    parser.add_argument("--random-max", type=int, default=1900)
    parser.add_argument(
        "--pad-impls",
        type=str,
        default="manual_copy,numpy_pad",
        help="Comma-separated list from {manual_copy,numpy_pad}",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="i8,i16,bf16",
        help='Comma-separated dtype list, e.g. "i8,i16,bf16"',
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("../../profiling/data/vla_profile.json"),
        help="Path to profile JSON (for feasible padded shapes).",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    profile_path = args.profile if args.profile.is_absolute() else (here / args.profile)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    inputs: List[Shape3D] = []
    if args.include_default_inputs:
        inputs.extend(DEFAULT_INPUTS)
    if args.inputs:
        inputs.extend(parse_inputs_arg(args.inputs))
    if args.inputs_file is not None:
        fp = args.inputs_file if args.inputs_file.is_absolute() else (here / args.inputs_file)
        inputs.extend(load_inputs_file(fp))
    inputs.extend(
        _generate_random_inputs(
            count=args.random_count,
            seed=args.random_seed,
            min_dim=args.random_min,
            max_dim=args.random_max,
        )
    )
    inputs = dedupe_inputs(inputs)
    if not inputs:
        raise ValueError("No inputs specified. Use --inputs/--inputs-file/--random-count.")

    dtypes = _parse_list(args.dtypes)
    for dtype in dtypes:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

    pad_impls = _parse_list(args.pad_impls)
    for pad_impl in pad_impls:
        if pad_impl not in PAD_IMPLS:
            raise ValueError(f"Unsupported pad impl: {pad_impl}")

    profile_shapes = _parse_profile_shapes(profile_path)
    if not profile_shapes:
        raise ValueError(f"No valid profile shapes found in {profile_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.out_dir
        if args.out_dir is not None
        else (here / "runs" / "padding" / f"padding_only_{timestamp}")
    )
    run_dir = run_dir if run_dir.is_absolute() else (here / run_dir)
    out_dir = run_dir / "padding_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save explicit input lists for quick inspection.
    (run_dir / "inputs.json").write_text(
        json.dumps({"inputs": [list(x) for x in inputs]}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "inputs.txt").write_text(
        "\n".join(shape_key(x) for x in inputs) + "\n",
        encoding="utf-8",
    )

    workload = _build_workload_estimate(
        inputs=inputs,
        profile_shapes=profile_shapes,
        dtypes=dtypes,
        pad_impls=pad_impls,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    (run_dir / "workload_estimate.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
    print(
        "[WORKLOAD] "
        f"inputs={workload['num_inputs']} candidates_per_job={workload['total_candidates_per_job']} "
        f"jobs={workload['num_jobs']} repeats={workload['repeats']} "
        f"total_pad_calls_all_jobs={workload['total_pad_calls_all_jobs']}"
    )

    jobs = []
    for pad_impl in pad_impls:
        for dtype in dtypes:
            jobs.append(
                {
                    "pad_impl": pad_impl,
                    "dtype": dtype,
                    "inputs": [list(x) for x in inputs],
                    "profile_shapes": [list(x) for x in profile_shapes],
                    "warmup": int(args.warmup),
                    "repeats": int(args.repeats),
                    "seed": int(args.random_seed),
                }
            )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "mode": "padding_only",
        "profile_path": str(profile_path),
        "num_inputs": len(inputs),
        "inputs": [list(x) for x in inputs],
        "pad_impls": pad_impls,
        "dtypes": dtypes,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "jobs": jobs,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[RUN] created {len(jobs)} jobs (pad_impl x dtype)")
    for job in jobs:
        print(f"[RUN] queued job pad_impl={job['pad_impl']} dtype={job['dtype']}")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [ex.submit(_run_worker, job) for job in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            print(
                "[JOB DONE] "
                f"pad_impl={row['pad_impl']} dtype={row['dtype']} "
                f"inputs={row['num_inputs']} candidates={row['num_candidates']} "
                f"elapsed_s={row['elapsed_s']:.2f}"
            )

    files_written = []
    for pad_impl in pad_impls:
        out_payload = {
            "mode": "padding_only",
            "profile_path": str(profile_path),
            "inputs": {},
        }
        impl_rows = [r for r in results if r["pad_impl"] == pad_impl]
        for row in impl_rows:
            dtype = row["dtype"]
            for in_key, dtype_block in row["payload"]["inputs"].items():
                out_payload["inputs"].setdefault(in_key, {})
                out_payload["inputs"][in_key].setdefault(dtype, {})
                out_payload["inputs"][in_key][dtype].update(dtype_block.get(dtype, {}))

        out_path = out_dir / f"padding_sweep_{pad_impl}.json"
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        files_written.append(str(out_path))

    summary = {
        "run_dir": str(run_dir),
        "mode": "padding_only",
        "jobs_total": len(jobs),
        "inputs_file_json": str(run_dir / "inputs.json"),
        "inputs_file_txt": str(run_dir / "inputs.txt"),
        "workload_estimate_file": str(run_dir / "workload_estimate.json"),
        "files_written": files_written,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] run_dir={run_dir}")
    for path in files_written:
        print(f"[DONE] wrote {path}")


if __name__ == "__main__":
    main()
