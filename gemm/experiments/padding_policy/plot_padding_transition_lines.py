#!/usr/bin/env python3
"""Plot manual vs numpy padding times over feature for each dtype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DTYPES = ["i8", "i16", "bf16"]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return (0.0, float(y[0]) if y.size else 0.0)
    m, b = np.polyfit(x.astype(np.float64), y.astype(np.float64), deg=1)
    return (float(m), float(b))


def _intersection_x(m1: float, b1: float, m2: float, b2: float) -> float | None:
    denom = m1 - m2
    if abs(denom) < 1e-12:
        return None
    return (b2 - b1) / denom


def _plot_one_dtype(
    records: list[dict],
    dtype: str,
    feature_name: str,
    threshold_value: float | None,
    out_path: Path,
    connect_points: bool,
) -> None:
    rows = [r for r in records if r.get("dtype") == dtype]
    if not rows:
        return

    rows.sort(key=lambda r: float(r["feature"]))
    x = np.array([float(r["feature"]) for r in rows], dtype=np.float64)
    y_manual = np.array([float(r["manual_copy_us"]) for r in rows], dtype=np.float64)
    y_numpy = np.array([float(r["numpy_pad_us"]) for r in rows], dtype=np.float64)

    m_manual, b_manual = _fit_line(x, y_manual)
    m_numpy, b_numpy = _fit_line(x, y_numpy)
    x_reg = np.array([x.min(), x.max()], dtype=np.float64)
    y_manual_reg = m_manual * x_reg + b_manual
    y_numpy_reg = m_numpy * x_reg + b_numpy

    x_inter = _intersection_x(m_manual, b_manual, m_numpy, b_numpy)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Raw points: optionally connected or scatter-only.
    if connect_points:
        ax.plot(
            x,
            y_manual,
            color="#1f77b4",
            linewidth=1.0,
            alpha=0.55,
            label="manual_copy (connected points)",
        )
        ax.plot(
            x,
            y_numpy,
            color="#d62728",
            linewidth=1.0,
            alpha=0.55,
            label="numpy_pad (connected points)",
        )
    else:
        ax.scatter(
            x,
            y_manual,
            color="#1f77b4",
            s=10,
            alpha=0.40,
            label="manual_copy (points)",
        )
        ax.scatter(
            x,
            y_numpy,
            color="#d62728",
            s=10,
            alpha=0.40,
            label="numpy_pad (points)",
        )

    # Regression lines.
    ax.plot(
        x_reg,
        y_manual_reg,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.2,
        label=f"manual regression: y={m_manual:.3e}x+{b_manual:.2f}",
    )
    ax.plot(
        x_reg,
        y_numpy_reg,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=f"numpy regression: y={m_numpy:.3e}x+{b_numpy:.2f}",
    )

    if threshold_value is not None:
        ax.axvline(
            threshold_value,
            color="#2ca02c",
            linewidth=2.0,
            linestyle=":",
            label=f"threshold={threshold_value:.0f}",
        )

    if x_inter is not None and np.isfinite(x_inter):
        ax.axvline(
            x_inter,
            color="#9467bd",
            linewidth=1.8,
            linestyle="-.",
            label=f"regression intersection={x_inter:.0f}",
        )

    ax.set_title(f"{dtype}: padding time vs {feature_name} (n={len(rows)})")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("padding time (us)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-dtype plots for manual_copy vs numpy_pad padding time over feature, "
            "with connected lines and linear regression overlays."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to padding_transition_db.json",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        required=True,
        help="Path to padding_transition_thresholds.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plot images.",
    )
    parser.add_argument(
        "--connect-points",
        action="store_true",
        help="Connect raw points with lines (default: scatter-only).",
    )
    args = parser.parse_args()

    with args.db.open("r", encoding="utf-8") as f:
        db = json.load(f) or {}
    with args.thresholds.open("r", encoding="utf-8") as f:
        th = json.load(f) or {}

    records = db.get("records", [])
    feature_name = str(th.get("feature") or db.get("feature") or "feature")
    dtype_thresholds = th.get("dtype_thresholds", {}) or {}

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = args.db.resolve().parent / out_dir

    made = []
    for dtype in DTYPES:
        threshold_value = None
        block = dtype_thresholds.get(dtype, {})
        if isinstance(block, dict) and block.get("threshold") is not None:
            threshold_value = float(block["threshold"])
        out_path = out_dir / f"padding_transition_{dtype}.png"
        _plot_one_dtype(
            records=records,
            dtype=dtype,
            feature_name=feature_name,
            threshold_value=threshold_value,
            out_path=out_path,
            connect_points=bool(args.connect_points),
        )
        if out_path.exists():
            made.append(out_path)

    print("[DONE] Generated plots:")
    for p in made:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
