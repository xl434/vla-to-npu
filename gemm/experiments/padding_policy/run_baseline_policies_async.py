#!/usr/bin/env python3
"""Benchmark Option 1/2/3 padded-shape policies with async workers."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import (
    DEFAULT_INPUTS,
    DTYPES,
    ProfileDB,
    dedupe_inputs,
    input_feature,
    load_inputs_file,
    load_padding_lookup,
    parse_gemm_output,
    parse_inputs_arg,
    shape_key,
)

Shape3D = Tuple[int, int, int]


def _generate_random_inputs(count: int, seed: int, min_dim: int, max_dim: int) -> List[Shape3D]:
    if count <= 0:
        return []
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


def _fit_hybrid_threshold(rows: List[dict], option1_key: str, option2_key: str) -> dict:
    if not rows:
        return {"threshold": 0, "mean_selected_value": None, "num_threshold_candidates": 0}
    values = sorted({int(r["feature"]) for r in rows})
    candidates = [values[0] - 1] + values
    best = None
    for t in candidates:
        chosen_sum = 0.0
        for r in rows:
            if int(r["feature"]) <= t:
                chosen_sum += float(r[option2_key])
            else:
                chosen_sum += float(r[option1_key])
        avg = chosen_sum / len(rows)
        metric = (avg, t)
        if best is None or metric < best["metric"]:
            best = {"threshold": t, "avg_value": avg, "metric": metric}
    assert best is not None
    return {
        "threshold": int(best["threshold"]),
        "mean_selected_value": float(best["avg_value"]),
        "num_threshold_candidates": len(candidates),
    }


def _tree_leaf(rows: List[dict], option1_key: str, option2_key: str) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "type": "leaf",
            "pick_policy": "option1_closest",
            "mean_selected_value": 0.0,
            "mean_option1": None,
            "mean_option2": None,
            "num_samples": 0,
        }
    mean1 = sum(float(r[option1_key]) for r in rows) / float(n)
    mean2 = sum(float(r[option2_key]) for r in rows) / float(n)
    pick = "option2_fastest_exec" if mean2 <= mean1 else "option1_closest"
    selected = mean2 if pick == "option2_fastest_exec" else mean1
    return {
        "type": "leaf",
        "pick_policy": pick,
        "mean_selected_value": float(selected),
        "mean_option1": float(mean1),
        "mean_option2": float(mean2),
        "num_samples": int(n),
    }


def _fit_hybrid_tree(
    rows: List[dict],
    option1_key: str,
    option2_key: str,
    max_depth: int,
    min_leaf: int,
    depth: int = 0,
) -> dict:
    leaf = _tree_leaf(rows, option1_key, option2_key)
    n = len(rows)
    if n == 0:
        return leaf
    if depth >= max_depth or n < max(2, min_leaf * 2):
        return leaf

    best_split = None
    for feature in ("m", "n", "k", "ab_elements"):
        values = sorted({int(r[feature]) for r in rows})
        if len(values) <= 1:
            continue
        for t in values[:-1]:
            left = [r for r in rows if int(r[feature]) <= t]
            right = [r for r in rows if int(r[feature]) > t]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            left_leaf = _tree_leaf(left, option1_key, option2_key)
            right_leaf = _tree_leaf(right, option1_key, option2_key)
            score = (
                (len(left) * float(left_leaf["mean_selected_value"]))
                + (len(right) * float(right_leaf["mean_selected_value"]))
            ) / float(n)
            metric = (score, feature, int(t))
            if best_split is None or metric < best_split["metric"]:
                best_split = {
                    "feature": feature,
                    "threshold": int(t),
                    "left_rows": left,
                    "right_rows": right,
                    "metric": metric,
                    "score": float(score),
                }

    if best_split is None:
        return leaf
    if float(best_split["score"]) + 1e-12 >= float(leaf["mean_selected_value"]):
        return leaf

    left_node = _fit_hybrid_tree(
        rows=best_split["left_rows"],
        option1_key=option1_key,
        option2_key=option2_key,
        max_depth=max_depth,
        min_leaf=min_leaf,
        depth=depth + 1,
    )
    right_node = _fit_hybrid_tree(
        rows=best_split["right_rows"],
        option1_key=option1_key,
        option2_key=option2_key,
        max_depth=max_depth,
        min_leaf=min_leaf,
        depth=depth + 1,
    )

    selected = (
        (len(best_split["left_rows"]) * float(left_node["mean_selected_value"]))
        + (len(best_split["right_rows"]) * float(right_node["mean_selected_value"]))
    ) / float(n)
    return {
        "type": "split",
        "feature": best_split["feature"],
        "threshold": int(best_split["threshold"]),
        "left": left_node,
        "right": right_node,
        "mean_selected_value": float(selected),
        "num_samples": int(n),
        "depth": int(depth),
    }


def _predict_hybrid_tree(tree: dict, input_shape: Shape3D, feature_value: int) -> Tuple[str, List[dict]]:
    m, n, k = input_shape
    vals = {
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "ab_elements": int(feature_value),
    }
    node = tree
    path = []
    while isinstance(node, dict) and node.get("type") == "split":
        feat = str(node["feature"])
        threshold = int(node["threshold"])
        value = int(vals[feat])
        go_left = value <= threshold
        path.append(
            {
                "feature": feat,
                "value": value,
                "threshold": threshold,
                "decision": "left_le" if go_left else "right_gt",
            }
        )
        node = node["left"] if go_left else node["right"]
    if not isinstance(node, dict):
        return ("option1_closest", path)
    pick = str(node.get("pick_policy", "option1_closest"))
    if pick not in {"option1_closest", "option2_fastest_exec"}:
        pick = "option1_closest"
    return (pick, path)


def _load_transition_thresholds(path: Optional[Path], here: Path) -> dict:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else (here / path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _load_hybrid_thresholds_file(path: Optional[Path], here: Path) -> dict:
    if path is None:
        return {}
    resolved = path if path.is_absolute() else (here / path)
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    if "hybrid_thresholds" in raw and isinstance(raw["hybrid_thresholds"], dict):
        return {k: int(v) for k, v in raw["hybrid_thresholds"].items()}
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    return {}


def _select_pad_impl(
    input_shape: Shape3D,
    dtype: str,
    feature_name: str,
    transition: dict,
    default_pad_impl: str,
) -> str:
    if not transition:
        return default_pad_impl
    block = (transition.get("dtype_thresholds") or {}).get(dtype, {})
    if "threshold" not in block:
        return default_pad_impl
    threshold = int(block["threshold"])
    small = str(block.get("small_strategy", default_pad_impl))
    large = str(block.get("large_strategy", default_pad_impl))
    f = input_feature(input_shape, feature_name)
    return small if f <= threshold else large


def _estimate_padding_overhead_us(
    input_shape: Shape3D,
    padded_shape: Shape3D,
    dtype: str,
    pad_impl: str,
    manual_lookup: dict,
    numpy_lookup: dict,
) -> float:
    key = (input_shape, dtype, padded_shape)
    if pad_impl == "manual_copy":
        row = manual_lookup.get(key)
    elif pad_impl == "numpy_pad":
        row = numpy_lookup.get(key)
    else:
        row = None
    if row is not None:
        return float(row.get("overhead_us", 0.0))
    # Fallback heuristic if exact lookup is missing.
    m, n, k = input_shape
    mp, np_, kp = padded_shape
    added = max(0, mp * kp - m * k) + max(0, kp * np_ - k * n)
    return 0.0025 * float(added)


def _friendly_result_row(row: dict, cutoff_feature_name: str) -> dict:
    return {
        "baseline_policy": row.get("policy"),
        "data_type": row.get("dtype"),
        "input_shape_mnk": row.get("input_shape"),
        "selected_padded_shape_mnk": row.get("padded_shape"),
        "selected_tile_mnk": row.get("tile"),
        "selected_padding_strategy": row.get("pad_impl"),
        "cutoff_feature_name": cutoff_feature_name,
        "cutoff_feature_value": row.get("hybrid_feature_value", row.get("feature")),
        "cutoff_threshold_for_dtype": row.get("hybrid_threshold_value"),
        "cutoff_decision_branch": row.get("hybrid_branch"),
        "hybrid_selector_model": row.get("hybrid_selector_model"),
        "hybrid_selected_base_policy": row.get("hybrid_selected_base_policy"),
        "hybrid_tree_path": row.get("hybrid_tree_path"),
        "profiled_best_execution_us": row.get("profile_execution_us"),
        "measured_padding_us": row.get("padding_us"),
        "measured_unpadding_us": row.get("unpadding_us"),
        "measured_execution_us": row.get("execution_us"),
        "measured_total_us": row.get("total_us"),
        "run_status": row.get("status"),
        "process_return_code": row.get("return_code"),
        "mapping_executed": row.get("mapping_executed"),
        "passed_correctness": row.get("passed"),
        "log_path": row.get("log_path"),
    }


async def _worker(
    worker_id: int,
    queue: asyncio.Queue,
    results: List[dict],
    gemm_script: Path,
    run_dir: Path,
    dry_run: bool,
) -> None:
    project_dir = run_dir / "work" / f"top_worker_{worker_id}"
    project_dir.mkdir(parents=True, exist_ok=True)

    while True:
        case = await queue.get()
        if case is None:
            queue.task_done()
            break

        case_id = int(case["case_id"])
        dtype = case["dtype"]
        input_shape = case["input_shape"]
        padded_shape = case["padded_shape"]
        tile = case["tile"]
        pad_impl = case["pad_impl"]

        cmd = [
            sys.executable,
            str(gemm_script),
            "--use-padding",
            "--dtype",
            dtype,
            "--M",
            str(input_shape[0]),
            "--N",
            str(input_shape[1]),
            "--K",
            str(input_shape[2]),
            "--m",
            str(tile[0]),
            "--n",
            str(tile[1]),
            "--k",
            str(tile[2]),
            "--pad-M",
            str(padded_shape[0]),
            "--pad-N",
            str(padded_shape[1]),
            "--pad-K",
            str(padded_shape[2]),
            "--pad-impl",
            pad_impl,
            "--project-dir",
            str(project_dir),
        ]
        log_name = (
            f"case_{case_id:05d}_{case['policy']}_{dtype}_"
            f"{input_shape[0]}_{input_shape[1]}_{input_shape[2]}.log"
        )
        log_path = run_dir / "logs" / log_name

        if dry_run:
            out_row = dict(case)
            out_row.update(
                {
                    "status": "dry_run",
                    "return_code": None,
                    "padding_us": None,
                    "unpadding_us": None,
                    "execution_us": None,
                    "total_us": None,
                    "mapping_executed": None,
                    "passed": None,
                    "log_path": str(log_path),
                    "cmd": cmd,
                }
            )
            results.append(out_row)
            queue.task_done()
            continue

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out_bytes, _ = await proc.communicate()
        raw = out_bytes.decode("utf-8", errors="ignore")
        log_path.write_text(raw, encoding="utf-8")

        parsed = parse_gemm_output(raw, dtype=dtype)
        pad = parsed.get("padding_us")
        unpad = parsed.get("unpadding_us")
        exec_us = parsed.get("execution_us")
        total = None
        if pad is not None and exec_us is not None:
            total = float(pad) + float(unpad or 0.0) + float(exec_us)

        out_row = dict(case)
        out_row.update(
            {
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": proc.returncode,
                "padding_us": pad,
                "unpadding_us": unpad,
                "execution_us": exec_us,
                "total_us": total,
                "mapping_executed": bool(parsed.get("mapping_executed")),
                "passed": bool(parsed.get("passed")),
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )
        results.append(out_row)
        queue.task_done()


async def _execute_cases(
    cases: List[dict],
    jobs: int,
    gemm_script: Path,
    run_dir: Path,
    dry_run: bool,
) -> List[dict]:
    queue: asyncio.Queue = asyncio.Queue()
    for case in cases:
        queue.put_nowait(case)
    for _ in range(max(1, jobs)):
        queue.put_nowait(None)

    results: List[dict] = []
    workers = [
        asyncio.create_task(
            _worker(
                worker_id=i,
                queue=queue,
                results=results,
                gemm_script=gemm_script,
                run_dir=run_dir,
                dry_run=dry_run,
            )
        )
        for i in range(max(1, jobs))
    ]

    await queue.join()
    await asyncio.gather(*workers)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark three padded-shape policies: option1_closest, option2_fastest_exec, "
            "option3_hybrid (fixed-threshold split or auto decision-tree selector)."
        )
    )
    parser.add_argument("--inputs", type=str, default=None, help='Semicolon-separated "M,N,K;..."')
    parser.add_argument("--inputs-file", type=Path, default=None, help="JSON file containing inputs")
    parser.add_argument("--include-default-inputs", action="store_true")
    parser.add_argument("--random-count", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--random-min", type=int, default=33)
    parser.add_argument("--random-max", type=int, default=1900)
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
        help="Path to profile JSON (relative to this script if not absolute).",
    )
    parser.add_argument(
        "--transition-thresholds",
        type=Path,
        default=Path("runs/padding/analysis_primary/padding_transition_thresholds.json"),
        help=(
            "JSON from fit_transition_threshold.py to choose manual vs numpy padding. "
            "Defaults to runs/padding/analysis_primary/padding_transition_thresholds.json."
        ),
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["ab_elements", "touch_elements", "volume", "max_dim"],
        default="ab_elements",
        help="Feature used by transition and option3 thresholding.",
    )
    parser.add_argument(
        "--hybrid-threshold-mode",
        type=str,
        choices=["auto", "fixed"],
        default="auto",
        help=(
            "How to choose option3 rule per dtype. "
            "auto=decision-tree rule; fixed=single threshold split."
        ),
    )
    parser.add_argument(
        "--hybrid-thresholds-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON with per-dtype hybrid thresholds, e.g. "
            '{"i8": 123, "i16": 456, "bf16": 789}. '
            "When set, this overrides auto/fixed threshold fitting."
        ),
    )
    parser.add_argument(
        "--hybrid-threshold-objective",
        type=str,
        choices=["exec_only", "total_estimate"],
        default="total_estimate",
        help=(
            "Objective for auto hybrid threshold fitting: "
            "exec_only uses profile execution only; total_estimate uses "
            "profile execution + estimated padding overhead."
        ),
    )
    parser.add_argument(
        "--hybrid-threshold-fixed",
        type=int,
        default=0,
        help="Used when --hybrid-threshold-mode=fixed.",
    )
    parser.add_argument(
        "--hybrid-tree-max-depth",
        type=int,
        default=2,
        help="Decision-tree max depth for --hybrid-threshold-mode=auto.",
    )
    parser.add_argument(
        "--hybrid-tree-min-leaf",
        type=int,
        default=6,
        help="Decision-tree minimum samples per leaf for --hybrid-threshold-mode=auto.",
    )
    parser.add_argument(
        "--default-pad-impl",
        type=str,
        choices=["manual_copy", "numpy_pad", "torch_pad"],
        default="numpy_pad",
        help="Used if transition thresholds are missing for a dtype.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent worker count")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Do not run GEMM executable. Compute totals from profile execution_us and exact "
            "padding lookup from sweep JSON files."
        ),
    )
    parser.add_argument(
        "--manual-sweep",
        type=Path,
        default=Path(
            "runs/padding/padding_only_primary/padding_sweeps/padding_sweep_manual_copy.json"
        ),
        help="Manual copy sweep JSON (used by --estimate-only).",
    )
    parser.add_argument(
        "--numpy-sweep",
        type=Path,
        default=Path(
            "runs/padding/padding_only_primary/padding_sweeps/padding_sweep_numpy_pad.json"
        ),
        help="NumPy pad sweep JSON (used by --estimate-only).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    gemm_root = here.parent.parent
    gemm_script = gemm_root / "scripts" / "v2_test_mapping_large_gemm.py"

    profile_path = args.profile if args.profile.is_absolute() else (here / args.profile)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    inputs: List[Shape3D] = []
    if args.include_default_inputs:
        inputs.extend(DEFAULT_INPUTS)
    if args.inputs:
        inputs.extend(parse_inputs_arg(args.inputs))
    if args.inputs_file is not None:
        file_path = args.inputs_file if args.inputs_file.is_absolute() else (here / args.inputs_file)
        inputs.extend(load_inputs_file(file_path))
    inputs.extend(_generate_random_inputs(args.random_count, args.random_seed, args.random_min, args.random_max))
    inputs = dedupe_inputs(inputs)
    if not inputs:
        raise ValueError("No inputs specified. Provide --inputs/--inputs-file/--random-count.")

    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    for dtype in dtypes:
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.out_dir
        if args.out_dir is not None
        else (here / "runs" / "baseline" / "baseline3" / f"baseline3_{timestamp}")
    )
    run_dir = run_dir if run_dir.is_absolute() else (here / run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "work").mkdir(parents=True, exist_ok=True)

    db = ProfileDB(profile_path)
    transition = _load_transition_thresholds(args.transition_thresholds, here)
    transition_feature_name = str(transition.get("feature") or args.feature)
    fixed_hybrid_thresholds = _load_hybrid_thresholds_file(args.hybrid_thresholds_file, here)

    manual_sweep = args.manual_sweep if args.manual_sweep.is_absolute() else (here / args.manual_sweep)
    numpy_sweep = args.numpy_sweep if args.numpy_sweep.is_absolute() else (here / args.numpy_sweep)
    manual_lookup = load_padding_lookup(manual_sweep) if manual_sweep.exists() else {}
    numpy_lookup = load_padding_lookup(numpy_sweep) if numpy_sweep.exists() else {}

    # Fit per-dtype option3 selector on this input set.
    # - fixed threshold modes: keep threshold behavior for compatibility
    # - auto mode: use a small decision tree on (m,n,k,ab_elements)
    hybrid_thresholds: Dict[str, Optional[int]] = {}
    hybrid_threshold_fit: Dict[str, dict] = {}
    hybrid_tree_rules: Dict[str, dict] = {}
    for dtype in dtypes:
        if dtype in fixed_hybrid_thresholds:
            hybrid_thresholds[dtype] = int(fixed_hybrid_thresholds[dtype])
            hybrid_tree_rules[dtype] = {
                "type": "leaf",
                "pick_policy": None,
                "mode": "fixed_threshold_file",
            }
            hybrid_threshold_fit[dtype] = {
                "mode": "fixed_from_file",
                "threshold": int(fixed_hybrid_thresholds[dtype]),
                "selector_type": "threshold",
                "objective": None,
                "mean_selected_value": None,
                "num_samples": len(inputs),
                "num_threshold_candidates": None,
            }
            continue
        if args.hybrid_threshold_mode == "fixed":
            hybrid_thresholds[dtype] = int(args.hybrid_threshold_fixed)
            hybrid_tree_rules[dtype] = {
                "type": "leaf",
                "pick_policy": None,
                "mode": "fixed_threshold_single",
            }
            hybrid_threshold_fit[dtype] = {
                "mode": "fixed_single_value",
                "threshold": int(args.hybrid_threshold_fixed),
                "selector_type": "threshold",
                "objective": None,
                "mean_selected_value": None,
                "num_samples": len(inputs),
                "num_threshold_candidates": None,
            }
            continue
        rows = []
        for shp in inputs:
            c1 = db.closest_candidate(dtype, shp)
            c2 = db.fastest_candidate(dtype, shp)
            m, n, k = shp
            ab = input_feature(shp, args.feature)
            pad_impl = _select_pad_impl(
                input_shape=shp,
                dtype=dtype,
                feature_name=transition_feature_name,
                transition=transition,
                default_pad_impl=args.default_pad_impl,
            )
            row = {
                "feature": ab,
                "m": int(m),
                "n": int(n),
                "k": int(k),
                "ab_elements": int(m * k + k * n),
                "option1_exec_us": c1.execution_us,
                "option2_exec_us": c2.execution_us,
            }
            if args.hybrid_threshold_objective == "total_estimate":
                overhead1 = _estimate_padding_overhead_us(
                    input_shape=shp,
                    padded_shape=c1.shape,
                    dtype=dtype,
                    pad_impl=pad_impl,
                    manual_lookup=manual_lookup,
                    numpy_lookup=numpy_lookup,
                )
                overhead2 = _estimate_padding_overhead_us(
                    input_shape=shp,
                    padded_shape=c2.shape,
                    dtype=dtype,
                    pad_impl=pad_impl,
                    manual_lookup=manual_lookup,
                    numpy_lookup=numpy_lookup,
                )
                row["option1_total_est_us"] = float(c1.execution_us) + float(overhead1)
                row["option2_total_est_us"] = float(c2.execution_us) + float(overhead2)
            rows.append(row)

        option1_key = (
            "option1_total_est_us"
            if args.hybrid_threshold_objective == "total_estimate"
            else "option1_exec_us"
        )
        option2_key = (
            "option2_total_est_us"
            if args.hybrid_threshold_objective == "total_estimate"
            else "option2_exec_us"
        )

        # Old 1D cutoff implementation (disabled by request):
        # fit = _fit_hybrid_threshold(
        #     rows=rows,
        #     option1_key=option1_key,
        #     option2_key=option2_key,
        # )
        # hybrid_thresholds[dtype] = int(fit["threshold"])
        # hybrid_threshold_fit[dtype] = {
        #     "mode": "auto",
        #     "threshold": int(fit["threshold"]),
        #     "selector_type": "threshold",
        #     "objective": str(args.hybrid_threshold_objective),
        #     "mean_selected_value": fit["mean_selected_value"],
        #     "num_samples": len(rows),
        #     "num_threshold_candidates": fit["num_threshold_candidates"],
        # }

        tree = _fit_hybrid_tree(
            rows=rows,
            option1_key=option1_key,
            option2_key=option2_key,
            max_depth=max(0, int(args.hybrid_tree_max_depth)),
            min_leaf=max(1, int(args.hybrid_tree_min_leaf)),
        )
        hybrid_tree_rules[dtype] = tree
        root_threshold = None
        if tree.get("type") == "split" and str(tree.get("feature")) == str(args.feature):
            root_threshold = int(tree.get("threshold"))
        hybrid_thresholds[dtype] = root_threshold
        hybrid_threshold_fit[dtype] = {
            "mode": "auto",
            "selector_type": "decision_tree",
            "threshold": root_threshold,
            "objective": str(args.hybrid_threshold_objective),
            "mean_selected_value": float(tree.get("mean_selected_value", 0.0)),
            "num_samples": len(rows),
            "tree_max_depth": int(args.hybrid_tree_max_depth),
            "tree_min_leaf": int(args.hybrid_tree_min_leaf),
            "tree_root_feature": tree.get("feature") if tree.get("type") == "split" else None,
            "tree_root_threshold": tree.get("threshold") if tree.get("type") == "split" else None,
        }

    cases = []
    case_id = 0
    for dtype in dtypes:
        threshold = hybrid_thresholds.get(dtype)
        tree_rule = hybrid_tree_rules.get(dtype)
        for shp in inputs:
            feat = input_feature(shp, args.feature)
            c1 = db.closest_candidate(dtype, shp)
            c2 = db.fastest_candidate(dtype, shp)
            hybrid_base_policy = "option1_closest"
            hybrid_branch = "large_closest"
            hybrid_tree_path = None
            hybrid_selector_model = "threshold"
            if args.hybrid_threshold_mode == "auto" and dtype not in fixed_hybrid_thresholds:
                hybrid_selector_model = "decision_tree"
                pick, path = _predict_hybrid_tree(tree_rule or {}, shp, feat)
                hybrid_base_policy = pick
                hybrid_tree_path = path
                hybrid_branch = "tree_fastest_exec" if pick == "option2_fastest_exec" else "tree_closest"
                c3 = c2 if pick == "option2_fastest_exec" else c1
            else:
                # Threshold-based fallback for fixed modes and fixed-threshold files.
                threshold_val = int(threshold or 0)
                c3 = c2 if feat <= threshold_val else c1
                hybrid_base_policy = "option2_fastest_exec" if feat <= threshold_val else "option1_closest"
                hybrid_branch = "small_fastest_exec" if feat <= threshold_val else "large_closest"

            pad_impl = _select_pad_impl(
                input_shape=shp,
                dtype=dtype,
                feature_name=transition_feature_name,
                transition=transition,
                default_pad_impl=args.default_pad_impl,
            )

            for policy, cand in (
                ("option1_closest", c1),
                ("option2_fastest_exec", c2),
                ("option3_hybrid", c3),
            ):
                cases.append(
                    {
                        "case_id": case_id,
                        "policy": policy,
                        "dtype": dtype,
                        "feature": feat,
                        "input_shape": list(shp),
                        "padded_shape": list(cand.shape),
                        "tile": list(cand.tile),
                        "profile_execution_us": cand.execution_us,
                        "pad_impl": pad_impl,
                        "hybrid_feature_value": feat,
                        "hybrid_threshold_value": threshold,
                        "hybrid_branch": hybrid_branch,
                        "hybrid_selector_model": hybrid_selector_model,
                        "hybrid_selected_base_policy": hybrid_base_policy,
                        "hybrid_tree_path": hybrid_tree_path,
                    }
                )
                case_id += 1

    manifest = {
        "created_at": datetime.now().isoformat(),
        "profile_path": str(profile_path),
        "feature": args.feature,
        "transition_feature": transition_feature_name,
        "transition_thresholds_path": str(args.transition_thresholds) if args.transition_thresholds else None,
        "hybrid_threshold_mode": args.hybrid_threshold_mode,
        "hybrid_thresholds_file": str(args.hybrid_thresholds_file) if args.hybrid_thresholds_file else None,
        "hybrid_threshold_objective": args.hybrid_threshold_objective,
        "hybrid_tree_max_depth": int(args.hybrid_tree_max_depth),
        "hybrid_tree_min_leaf": int(args.hybrid_tree_min_leaf),
        "hybrid_thresholds": hybrid_thresholds,
        "hybrid_threshold_fit": hybrid_threshold_fit,
        "hybrid_tree_rules": hybrid_tree_rules,
        "num_inputs": len(inputs),
        "inputs": [list(s) for s in inputs],
        "num_cases": len(cases),
        "cases": cases,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.estimate_only:
        results = []
        for case in cases:
            dtype = str(case["dtype"])
            input_shape = tuple(int(x) for x in case["input_shape"])
            padded_shape = tuple(int(x) for x in case["padded_shape"])
            pad_impl = str(case["pad_impl"])
            key = (input_shape, dtype, padded_shape)

            if pad_impl == "manual_copy":
                pad_row = manual_lookup.get(key)
            elif pad_impl == "numpy_pad":
                pad_row = numpy_lookup.get(key)
            else:
                pad_row = None

            padding_us = None if pad_row is None else float(pad_row["padding_us"])
            unpadding_us = None if pad_row is None else float(pad_row["unpadding_us"])
            execution_us = float(case["profile_execution_us"])
            total_us = None
            if padding_us is not None:
                total_us = padding_us + float(unpadding_us or 0.0) + execution_us

            out_row = dict(case)
            out_row.update(
                {
                    "status": "estimated" if total_us is not None else "estimated_missing_padding",
                    "return_code": None,
                    "padding_us": padding_us,
                    "unpadding_us": unpadding_us,
                    "execution_us": execution_us,
                    "total_us": total_us,
                    "mapping_executed": None,
                    "passed": None,
                    "log_path": None,
                    "cmd": None,
                }
            )
            results.append(out_row)
    else:
        results = asyncio.run(
            _execute_cases(
                cases=cases,
                jobs=max(1, args.jobs),
                gemm_script=gemm_script,
                run_dir=run_dir,
                dry_run=args.dry_run,
            )
        )

    # Aggregate summary by policy/dtype.
    summary_by_group: Dict[str, dict] = {}
    for row in results:
        key = f"{row['policy']}::{row['dtype']}"
        blk = summary_by_group.setdefault(
            key,
            {
                "policy": row["policy"],
                "dtype": row["dtype"],
                "count": 0,
                "count_with_total": 0,
                "mean_total_us": 0.0,
            },
        )
        blk["count"] += 1
        if row.get("total_us") is not None:
            blk["count_with_total"] += 1
            blk["mean_total_us"] += float(row["total_us"])

    for blk in summary_by_group.values():
        n = int(blk["count_with_total"])
        blk["mean_total_us"] = (blk["mean_total_us"] / n) if n else None

    summary_friendly = []
    for blk in summary_by_group.values():
        summary_friendly.append(
            {
                "baseline_policy": blk["policy"],
                "data_type": blk["dtype"],
                "num_cases": blk["count"],
                "num_cases_with_total_time": blk["count_with_total"],
                "mean_total_time_us": blk["mean_total_us"],
            }
        )
    results_friendly = [_friendly_result_row(r, args.feature) for r in results]

    out = {
        "schema": "baseline_policies_v2",
        "run_dir": str(run_dir),
        "dry_run": bool(args.dry_run),
        "estimate_only": bool(args.estimate_only),
        "jobs": int(max(1, args.jobs)),
        "num_cases": len(cases),
        "num_results": len(results),
        "num_failed": sum(1 for r in results if r["status"] == "failed"),
        "summary_by_policy_dtype": summary_friendly,
        "results": results_friendly,
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] cases={len(cases)} failed={out['num_failed']}")


if __name__ == "__main__":
    main()
