#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


STRATEGIES = {
    "manual_copy": {
        "label": "Manual Copy",
        "default_file": "padding_sweep_copy.json",
        "color": "#1f77b4",
    },
    "torch_zeropad2d": {
        "label": "Torch ZeroPad2d",
        "default_file": "padding_sweep_torch.json",
        "color": "#ff7f0e",
    },
    "numpy_pad": {
        "label": "NumPy pad",
        "default_file": "padding_sweep_numpy.json",
        "color": "#2ca02c",
    },
}

DTYPE_COLORS = {
    "i8": "#1f77b4",
    "i16": "#d62728",
    "bf16": "#2ca02c",
}

STRATEGY_MARKERS = {
    "Manual Copy": "o",
    "Torch ZeroPad2d": "s",
    "NumPy pad": "^",
}


plt.style.use("seaborn-v0_8-whitegrid")


def _parse_dims(key: str):
    parts = [p.strip() for p in key.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated dims, got: {key}")
    return tuple(int(x) for x in parts)


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def _load_strategy_file(path: Path, strategy_id: str) -> pd.DataFrame:
    strategy_label = STRATEGIES[strategy_id]["label"]
    if not path.exists():
        print(f"[WARN] Missing file for strategy '{strategy_label}': {path}")
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    inputs = data.get("inputs", {})
    for input_key, dtype_map in inputs.items():
        try:
            m0, n0, k0 = _parse_dims(input_key)
        except Exception:
            continue

        orig_vol = int(m0 * n0 * k0)

        for dtype, cand_map in dtype_map.items():
            for padded_key, metrics in cand_map.items():
                try:
                    mp, np_, kp = _parse_dims(padded_key)
                except Exception:
                    continue

                padding_us = _safe_float(metrics.get("padding_us"))
                if padding_us is None:
                    continue

                unpadding_us = _safe_float(metrics.get("unpadding_us"))
                execution_us = _safe_float(metrics.get("mapping_runtime_us"))

                padded_vol = int(mp * np_ * kp)
                added_zeros = int(padded_vol - orig_vol)

                records.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy": strategy_label,
                        "dtype": dtype,
                        "input_key": input_key,
                        "padded_key": padded_key,
                        "padding_us": padding_us,
                        "unpadding_us": unpadding_us,
                        "execution_us": execution_us,
                        "error": bool(metrics.get("error", False)),
                        "overhead_us": (
                            padding_us + unpadding_us
                            if unpadding_us is not None
                            else padding_us
                        ),
                        "orig_vol": orig_vol,
                        "padded_vol": padded_vol,
                        # Required by request: padding_vol = additional zeros.
                        "padding_vol": added_zeros,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print(f"[WARN] No padding records loaded for '{strategy_label}' from {path}")
    return df


def _intersect_comparable_cases(df: pd.DataFrame) -> pd.DataFrame:
    case_cols = ["input_key", "dtype", "padded_key"]
    strategies = sorted(df["strategy_id"].unique().tolist())
    if not strategies:
        return pd.DataFrame()

    dedup_cols = case_cols + ["strategy_id"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()

    counts = (
        df.groupby(case_cols, as_index=False)["strategy_id"]
        .nunique()
        .rename(columns={"strategy_id": "n_strategies"})
    )
    common = counts[counts["n_strategies"] == len(strategies)][case_cols]
    return df.merge(common, on=case_cols, how="inner")


def _save_summary_tables(df_common: pd.DataFrame, out_dir: Path, prefix: str):
    strategy_summary = (
        df_common.groupby("strategy")["padding_us"]
        .agg(
            count="count",
            median_us="median",
            mean_us="mean",
            p95_us=lambda s: s.quantile(0.95),
            p99_us=lambda s: s.quantile(0.99),
            max_us="max",
        )
        .sort_values("median_us")
    )
    strategy_summary_path = out_dir / f"{prefix}_padding_time_strategy_stats.csv"
    strategy_summary.to_csv(strategy_summary_path)

    dtype_summary = (
        df_common.groupby(["dtype", "strategy"])["padding_us"]
        .agg(
            count="count",
            median_us="median",
            p95_us=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values(["dtype", "median_us"])
    )
    dtype_summary_path = out_dir / f"{prefix}_padding_time_dtype_strategy_stats.csv"
    dtype_summary.to_csv(dtype_summary_path, index=False)

    exec_df = df_common.dropna(subset=["execution_us"]).copy()
    if not exec_df.empty:
        tradeoff_summary = (
            exec_df.groupby("strategy")[["padding_us", "execution_us", "overhead_us"]]
            .agg(
                padding_median_us=("padding_us", "median"),
                execution_median_us=("execution_us", "median"),
                overhead_median_us=("overhead_us", "median"),
                execution_p95_us=("execution_us", lambda s: s.quantile(0.95)),
                overhead_p95_us=("overhead_us", lambda s: s.quantile(0.95)),
                count=("execution_us", "count"),
            )
            .sort_values(["execution_median_us", "overhead_median_us"])
        )
        tradeoff_summary_path = out_dir / f"{prefix}_execution_time_tradeoff_stats.csv"
        tradeoff_summary.to_csv(tradeoff_summary_path)
    else:
        tradeoff_summary_path = None

    print("\n[SUMMARY] Strategy ranking by median padding_us:")
    print(strategy_summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n[SAVED] {strategy_summary_path}")
    print(f"[SAVED] {dtype_summary_path}")
    if tradeoff_summary_path is not None:
        print(f"[SAVED] {tradeoff_summary_path}")


def _strategy_palette(strategies):
    palette = {}
    for sid, meta in STRATEGIES.items():
        label = meta["label"]
        if label in strategies:
            palette[label] = meta["color"]
    for s in strategies:
        palette.setdefault(s, "#7f7f7f")
    return palette


def _dtype_palette(dtypes):
    palette = {}
    for d in dtypes:
        palette[d] = DTYPE_COLORS.get(d, "#7f7f7f")
    return palette


def _marker_for_strategy(strategy: str):
    return STRATEGY_MARKERS.get(strategy, "o")


def _add_strategy_dtype_legend(ax, strategies, dtypes, dtype_palette):
    strategy_handles = [
        Line2D(
            [0],
            [0],
            marker=_marker_for_strategy(s),
            color="black",
            linestyle="None",
            label=s,
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
        )
        for s in strategies
    ]
    dtype_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            linestyle="None",
            label=d,
            markerfacecolor=dtype_palette[d],
            markeredgecolor=dtype_palette[d],
            markersize=8,
        )
        for d in dtypes
    ]

    legend1 = ax.legend(handles=strategy_handles, title="Strategy", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=dtype_handles, title="Dtype", loc="lower right")


def _add_ecdf_legend(ax, strategies, dtypes, dtype_palette, strategy_linestyle):
    dtype_handles = [
        Line2D([0], [0], color=dtype_palette[d], linestyle="-", label=d, linewidth=2.0)
        for d in dtypes
    ]
    strategy_handles = [
        Line2D([0], [0], color="black", linestyle=strategy_linestyle[s], label=s, linewidth=2.0)
        for s in strategies
    ]
    legend1 = ax.legend(handles=dtype_handles, title="Dtype (color)", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=strategy_handles, title="Strategy (line)", loc="lower right")


def _plot_main_overview(df_common: pd.DataFrame, out_dir: Path, prefix: str):
    strategies = sorted(df_common["strategy"].unique().tolist())
    dtypes = sorted(df_common["dtype"].unique().tolist())
    dtype_palette = _dtype_palette(dtypes)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1) ECDF: color=dtype, line-style=strategy.
    ls_cycle = ["-", "--", ":", "-."]
    strategy_linestyle = {s: ls_cycle[i % len(ls_cycle)] for i, s in enumerate(strategies)}
    for strategy in strategies:
        for dtype in dtypes:
            sub = df_common[(df_common["strategy"] == strategy) & (df_common["dtype"] == dtype)]
            vals = np.sort(sub["padding_us"].values)
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax1.step(
                vals,
                y,
                where="post",
                color=dtype_palette[dtype],
                linestyle=strategy_linestyle[strategy],
                alpha=0.9,
            )
    ax1.set_title("ECDF of Padding Time (color=dtype, line-style=strategy)")
    ax1.set_xlabel("padding time (us)")
    ax1.set_ylabel("ECDF")
    ax1.grid(True, linestyle="--", alpha=0.4)
    _add_ecdf_legend(ax1, strategies, dtypes, dtype_palette, strategy_linestyle)

    # 2) Box plot by strategy + dtype combinations.
    combos = []
    combo_data = []
    combo_colors = []
    for strategy in strategies:
        for dtype in dtypes:
            label = f"{strategy}\n{dtype}"
            vals = df_common.loc[
                (df_common["strategy"] == strategy) & (df_common["dtype"] == dtype),
                "padding_us",
            ].values
            if len(vals) == 0:
                continue
            combos.append(label)
            combo_data.append(vals)
            combo_colors.append(dtype_palette[dtype])

    if combo_data:
        bp = ax2.boxplot(combo_data, tick_labels=combos, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], combo_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax2.set_yscale("log")
    ax2.set_title("Padding Distribution by Strategy and Dtype")
    ax2.set_xlabel("strategy / dtype")
    ax2.set_ylabel("padding time (us)")
    ax2.tick_params(axis="x", labelrotation=30)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 3) Requested x-axis semantics: padding_vol = additional zeros (new-old).
    for strategy in strategies:
        for dtype in dtypes:
            sub = df_common[(df_common["strategy"] == strategy) & (df_common["dtype"] == dtype)]
            if sub.empty:
                continue
            ax3.scatter(
                sub["padding_vol"],
                sub["padding_us"],
                color=dtype_palette[dtype],
                marker=_marker_for_strategy(strategy),
                edgecolors="black",
                linewidths=0.2,
                alpha=0.45,
                s=22,
            )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title("Padding Time vs Additional Zeros (new volume - old volume)")
    ax3.set_xlabel("padding_vol (additional zeros)")
    ax3.set_ylabel("padding time (us)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    _add_strategy_dtype_legend(ax3, strategies, dtypes, dtype_palette)

    # 4) Tradeoff view: padding time vs execution time.
    exec_df = df_common.dropna(subset=["execution_us"]).copy()
    if exec_df.empty:
        ax4.text(0.5, 0.5, "No execution_us data available", ha="center", va="center")
        ax4.set_title("Padding vs Execution Tradeoff")
        ax4.axis("off")
    else:
        for strategy in strategies:
            for dtype in dtypes:
                sub = exec_df[(exec_df["strategy"] == strategy) & (exec_df["dtype"] == dtype)]
                if sub.empty:
                    continue
                ax4.scatter(
                    sub["padding_us"],
                    sub["execution_us"],
                    color=dtype_palette[dtype],
                    marker=_marker_for_strategy(strategy),
                    edgecolors="black",
                    linewidths=0.2,
                    alpha=0.45,
                    s=22,
                )
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_title("Padding vs Execution Tradeoff")
        ax4.set_xlabel("padding time (us)")
        ax4.set_ylabel("execution time (us)")
        ax4.grid(True, linestyle="--", alpha=0.4)
        _add_strategy_dtype_legend(ax4, strategies, dtypes, dtype_palette)

    plt.tight_layout()
    out_path = out_dir / f"{prefix}_padding_execution_tradeoff_overview.jpg"
    fig.savefig(out_path, dpi=250)
    print(f"\n[SAVED] {out_path}")


def _plot_per_input_dtype_padding(df_common: pd.DataFrame, out_dir: Path, prefix: str):
    pad_df = df_common.dropna(subset=["padding_us"]).copy()
    if pad_df.empty:
        print("[WARN] No padding_us values available; skipping per-input padding plots.")
        return

    strategies = sorted(pad_df["strategy"].unique().tolist())
    palette = _strategy_palette(strategies)
    per_dir = out_dir / f"{prefix}_per_input_dtype_padding_plots"
    per_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    grouped = pad_df.groupby(["input_key", "dtype"], sort=True)
    for (input_key, dtype), sub in grouped:
        # Sort x-axis by added zeros, then by padded key for stable ordering.
        padded_order_df = (
            sub[["padded_key", "padding_vol"]]
            .drop_duplicates()
            .sort_values(["padding_vol", "padded_key"])
        )
        padded_order = padded_order_df["padded_key"].tolist()
        if not padded_order:
            continue

        padded_index = {k: i for i, k in enumerate(padded_order)}
        x_base = np.arange(len(padded_order), dtype=float)

        fig_w = max(12, 0.35 * len(padded_order) + 4)
        fig, ax = plt.subplots(figsize=(fig_w, 5.8))

        for strategy in strategies:
            ss = sub[sub["strategy"] == strategy]
            if ss.empty:
                continue
            xs = [padded_index[k] for k in ss["padded_key"].tolist()]
            ys = ss["padding_us"].tolist()
            ax.scatter(
                xs,
                ys,
                s=32,
                alpha=0.85,
                color=palette[strategy],
                marker=_marker_for_strategy(strategy),
                edgecolors="black",
                linewidths=0.2,
                label=strategy,
            )

        tick_labels = []
        for key in padded_order:
            pad_zeros = int(
                padded_order_df.loc[padded_order_df["padded_key"] == key, "padding_vol"].iloc[0]
            )
            tick_labels.append(f"{key}\n(+{pad_zeros})")

        ax.set_xticks(x_base)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"Padding Time vs Padding Size | input={input_key}, dtype={dtype}")
        ax.set_xlabel("padded size (M,N,K) with additional zeros")
        ax.set_ylabel("padding time (us)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Strategy", loc="best")

        plt.tight_layout()
        fname = (
            f"{prefix}__input_{_slug(input_key)}__dtype_{_slug(dtype)}"
            "__padding_time_vs_padding_size.jpg"
        )
        out_path = per_dir / fname
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        made += 1

    print(f"[SAVED] {made} per-input padding plots in {per_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare padding and execution tradeoffs across Manual Copy, "
            "Torch ZeroPad2d, and NumPy pad strategy sweeps."
        )
    )
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--manual-copy",
        type=Path,
        default=base / STRATEGIES["manual_copy"]["default_file"],
        help="Path to Manual Copy strategy JSON",
    )
    parser.add_argument(
        "--torch-zeropad2d",
        type=Path,
        default=base / STRATEGIES["torch_zeropad2d"]["default_file"],
        help="Path to Torch ZeroPad2d strategy JSON",
    )
    parser.add_argument(
        "--numpy-pad",
        type=Path,
        default=base / STRATEGIES["numpy_pad"]["default_file"],
        help="Path to NumPy pad strategy JSON",
    )
    # Backward-compatible aliases.
    parser.add_argument("--copy", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--torch", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--numpy", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base / "plots",
        help="Directory for output plots/tables",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="padding_tradeoff",
        help="Filename prefix for outputs",
    )
    args = parser.parse_args()

    strategy_paths = {
        "manual_copy": args.copy if args.copy is not None else args.manual_copy,
        "torch_zeropad2d": args.torch if args.torch is not None else args.torch_zeropad2d,
        "numpy_pad": args.numpy if args.numpy is not None else args.numpy_pad,
    }

    frames = []
    for strategy_id, path in strategy_paths.items():
        df_one = _load_strategy_file(path, strategy_id)
        if not df_one.empty:
            frames.append(df_one)

    if len(frames) < 2:
        print("[ERROR] Need at least 2 non-empty strategy files for comparison.")
        raise SystemExit(1)

    df_all = pd.concat(frames, ignore_index=True)
    df_common = _intersect_comparable_cases(df_all)
    if df_common.empty:
        print("[ERROR] No comparable cases exist across provided strategy files.")
        raise SystemExit(1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Loaded {len(df_all)} total rows, "
        f"{len(df_common)} comparable rows after intersection "
        f"across strategies={sorted(df_common['strategy'].unique().tolist())}"
    )

    _save_summary_tables(df_common, out_dir, args.prefix)
    _plot_main_overview(df_common, out_dir, args.prefix)
    _plot_per_input_dtype_padding(df_common, out_dir, args.prefix)


if __name__ == "__main__":
    main()
