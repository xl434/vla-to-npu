#!/usr/bin/env python3
"""Plot Baseline-1 timing chunks vs input matrix size feature (M*K + N*K)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

DTYPES = ["i8", "i16", "bf16"]
METRICS = [
    ("profile_execution_us", "NPU Execution Time"),
    ("select_padding_strategy_us", "Select Padding Strategy"),
    ("select_padded_shape_us", "Select Padded Shape"),
    ("select_tile_us", "Select Tile Size"),
    ("decision_total_us", "Decision Total"),
    ("padding_us", "Padding"),
    ("unpadding_us", "Unpadding"),
    ("total_component_us", "Total Profile Time"),
]


def _x_ab_elements(row: dict) -> int:
    m, n, k = [int(v) for v in row["input_shape"]]
    return m * k + n * k


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _collect_by_dtype(rows: List[dict], metric_key: str, scale: float) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dtype in DTYPES:
        xs = []
        ys = []
        for r in rows:
            if r.get("dtype") != dtype:
                continue
            y = _safe_float(r.get(metric_key))
            if y is None:
                continue
            xs.append(_x_ab_elements(r))
            ys.append(y * scale)
        if not xs:
            out[dtype] = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            continue
        order = np.argsort(np.asarray(xs))
        x_arr = np.asarray(xs, dtype=np.float64)[order]
        y_arr = np.asarray(ys, dtype=np.float64)[order]
        out[dtype] = (x_arr, y_arr)
    return out


def _plot_metric(rows: List[dict], metric_key: str, metric_label: str, y_unit: str, out_path: Path) -> None:
    scale = 1.0 if y_unit == "us" else 1.0 / 1000.0
    y_label = f"time ({y_unit})"

    data = _collect_by_dtype(rows, metric_key, scale)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"i8": "#1f77b4", "i16": "#d62728", "bf16": "#2ca02c"}

    for dtype in DTYPES:
        x_arr, y_arr = data[dtype]
        if x_arr.size == 0:
            continue
        ax.plot(
            x_arr,
            y_arr,
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            alpha=0.85,
            color=colors[dtype],
            label=f"{dtype} (n={x_arr.size})",
        )

    ax.set_title(f"{metric_label} vs A+B Input Size")
    ax.set_xlabel("A+B input size (M*K + N*K)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Baseline-1 timing chunk plots vs input size (M*K + N*K), "
            "with separate lines for i8/i16/bf16."
        )
    )
    parser.add_argument("--results", type=Path, required=True, help="Path to baseline1 results.json")
    parser.add_argument("--out-dir", type=Path, default=Path("plots"), help="Output directory for plots")
    parser.add_argument(
        "--y-unit",
        type=str,
        choices=["us", "ms"],
        default="us",
        help="Y-axis unit for time values.",
    )
    args = parser.parse_args()

    with args.results.open("r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    rows = payload.get("results", [])

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = args.results.resolve().parent / out_dir

    made = []
    for key, label in METRICS:
        out_path = out_dir / f"baseline1_{key}_{args.y_unit}.png"
        _plot_metric(rows, key, label, args.y_unit, out_path)
        if out_path.exists():
            made.append(out_path)

    print("[DONE] Generated plots:")
    for p in made:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
