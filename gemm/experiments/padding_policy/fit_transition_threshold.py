#!/usr/bin/env python3
"""Build manual-vs-numpy padding DB and fit per-dtype switch thresholds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    DTYPES,
    fit_threshold_1d,
    input_feature,
    load_best_padding_from_sweep,
    shape_key,
)


def _collect_files(values: List[Path], dirs: List[Path], patterns: List[str], here: Path) -> List[Path]:
    out: List[Path] = []
    for p in values:
        pp = p if p.is_absolute() else (here / p)
        if pp.exists():
            out.append(pp)
    for d in dirs:
        dd = d if d.is_absolute() else (here / d)
        if not dd.exists():
            continue
        for pattern in patterns:
            out.extend(sorted(dd.rglob(pattern)))
    uniq = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _merge_best(files: List[Path]) -> Dict[Tuple[Tuple[int, int, int], str], dict]:
    merged = {}
    for path in files:
        best = load_best_padding_from_sweep(path)
        for key, row in best.items():
            prev = merged.get(key)
            if prev is None:
                merged[key] = row
                continue
            prev_score = (float(prev["padding_us"]), float(prev.get("unpadding_us", 0.0)))
            curr_score = (float(row["padding_us"]), float(row.get("unpadding_us", 0.0)))
            if curr_score < prev_score:
                merged[key] = row
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge padding sweep JSON files (manual_copy + numpy_pad), then fit per-dtype "
            "transition threshold for switching padding implementation."
        )
    )
    parser.add_argument(
        "--manual-file",
        type=Path,
        action="append",
        default=[],
        help="Manual-copy sweep JSON (repeatable).",
    )
    parser.add_argument(
        "--numpy-file",
        type=Path,
        action="append",
        default=[],
        help="NumPy-pad sweep JSON (repeatable).",
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing manual sweep JSON files.",
    )
    parser.add_argument(
        "--numpy-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing numpy sweep JSON files.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["ab_elements", "touch_elements", "volume", "max_dim"],
        default="ab_elements",
        help="1D feature for regression threshold.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/padding/analysis_latest"),
        help="Output directory for DB and threshold JSON files.",
    )
    parser.add_argument(
        "--exclude-default-baseline",
        action="store_true",
        help="Use only files provided via --manual-file/--numpy-file/--manual-dir/--numpy-dir.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = args.out_dir if args.out_dir.is_absolute() else (here / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Include existing baseline 18-input files by default.
    default_manual = here.parent.parent / "padding" / "padding_sweep_copy.json"
    default_numpy = here.parent.parent / "padding" / "padding_sweep_numpy.json"
    base_manual_files = [] if args.exclude_default_baseline else [default_manual]
    base_numpy_files = [] if args.exclude_default_baseline else [default_numpy]

    manual_files = _collect_files(
        values=base_manual_files + list(args.manual_file),
        dirs=list(args.manual_dir),
        patterns=["padding_sweep_manual_copy*.json"],
        here=here,
    )
    numpy_files = _collect_files(
        values=base_numpy_files + list(args.numpy_file),
        dirs=list(args.numpy_dir),
        patterns=["padding_sweep_numpy_pad*.json"],
        here=here,
    )

    if not manual_files:
        raise ValueError("No manual files found. Provide --manual-file or --manual-dir.")
    if not numpy_files:
        raise ValueError("No numpy files found. Provide --numpy-file or --numpy-dir.")

    manual_best = _merge_best(manual_files)
    numpy_best = _merge_best(numpy_files)

    records = []
    per_dtype_rows: Dict[str, List[dict]] = {dtype: [] for dtype in DTYPES}
    for key, m_row in manual_best.items():
        if key not in numpy_best:
            continue
        input_shape, dtype = key
        n_row = numpy_best[key]
        feature_val = input_feature(input_shape, args.feature)
        manual_us = float(m_row["padding_us"])
        numpy_us = float(n_row["padding_us"])
        winner = "manual_copy" if manual_us <= numpy_us else "numpy_pad"

        rec = {
            "input_shape": list(input_shape),
            "input_key": shape_key(input_shape),
            "dtype": dtype,
            "feature": feature_val,
            "manual_copy_us": manual_us,
            "numpy_pad_us": numpy_us,
            "winner": winner,
            "manual_best_padded_shape": m_row.get("best_padded_shape"),
            "numpy_best_padded_shape": n_row.get("best_padded_shape"),
        }
        records.append(rec)
        per_dtype_rows[dtype].append(rec)

    thresholds = {
        "created_at": datetime.now().isoformat(),
        "feature": args.feature,
        "dtype_thresholds": {},
    }

    for dtype, rows in per_dtype_rows.items():
        if not rows:
            thresholds["dtype_thresholds"][dtype] = {
                "num_samples": 0,
                "error": "No overlapping manual/numpy records",
            }
            continue

        forward = fit_threshold_1d(
            rows=rows,
            small_strategy="manual_copy",
            large_strategy="numpy_pad",
        )
        reverse = fit_threshold_1d(
            rows=rows,
            small_strategy="numpy_pad",
            large_strategy="manual_copy",
        )

        if (forward["avg_chosen_us"], forward["avg_regret_us"]) <= (
            reverse["avg_chosen_us"],
            reverse["avg_regret_us"],
        ):
            chosen = {
                "small_strategy": "manual_copy",
                "large_strategy": "numpy_pad",
                **forward,
            }
        else:
            chosen = {
                "small_strategy": "numpy_pad",
                "large_strategy": "manual_copy",
                **reverse,
            }

        thresholds["dtype_thresholds"][dtype] = {
            "num_samples": len(rows),
            **chosen,
        }

    db = {
        "created_at": datetime.now().isoformat(),
        "feature": args.feature,
        "manual_files": [str(p) for p in manual_files],
        "numpy_files": [str(p) for p in numpy_files],
        "num_records": len(records),
        "records": records,
    }

    db_path = out_dir / "padding_transition_db.json"
    thr_path = out_dir / "padding_transition_thresholds.json"
    db_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    thr_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote DB: {db_path}")
    print(f"[DONE] Wrote thresholds: {thr_path}")


if __name__ == "__main__":
    main()
