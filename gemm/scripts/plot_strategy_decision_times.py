#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


STRATEGIES = ["hashmap_27", "sorted_first_fit", "preprocessed_map"]
DTYPES = ["i8", "i16", "bf16"]
COLORS = {
    "hashmap_27": "#4C78A8",
    "sorted_first_fit": "#F58518",
    "preprocessed_map": "#54A24B",
}


def load_strategy_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    if not isinstance(data, dict) or "inputs" not in data:
        raise ValueError(f"Invalid strategy JSON format: {path}")
    return data


def collect_decision_times(data_by_strategy: Dict[str, dict]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    first = data_by_strategy[STRATEGIES[0]]
    input_keys = list(first.get("inputs", {}).keys())
    times: Dict[str, Dict[str, Dict[str, float]]] = {}
    for strategy in STRATEGIES:
        strat_inputs = data_by_strategy[strategy].get("inputs", {})
        times[strategy] = {}
        for input_key in input_keys:
            dtype_map = strat_inputs.get(input_key, {})
            times[strategy][input_key] = {}
            for dtype in DTYPES:
                info = dtype_map.get(dtype, {})
                val = info.get("decision_us")
                times[strategy][input_key][dtype] = float(val) if val is not None else np.nan
    return input_keys, times


def plot_fastest_heatmap(input_keys: List[str], times: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    winner_idx = np.zeros((len(input_keys), len(DTYPES)), dtype=int)
    winner_labels = np.empty((len(input_keys), len(DTYPES)), dtype=object)
    for i, input_key in enumerate(input_keys):
        for j, dtype in enumerate(DTYPES):
            vals = [times[s][input_key][dtype] for s in STRATEGIES]
            best_idx = int(np.nanargmin(vals))
            winner_idx[i, j] = best_idx
            winner_labels[i, j] = STRATEGIES[best_idx]

    cmap = ListedColormap([COLORS[s] for s in STRATEGIES])
    fig_h = max(6, 0.35 * len(input_keys))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(winner_idx, cmap=cmap, aspect="auto", vmin=0, vmax=len(STRATEGIES) - 1)
    _ = im  # keep for readability; color is explained by legend.

    ax.set_xticks(np.arange(len(DTYPES)))
    ax.set_xticklabels(DTYPES)
    ax.set_yticks(np.arange(len(input_keys)))
    ax.set_yticklabels(input_keys)
    ax.set_xlabel("Dtype")
    ax.set_ylabel("Input Shape (M,N,K)")
    ax.set_title("Fastest Strategy by Input and Dtype (decision_us)")

    short = {"hashmap_27": "H", "sorted_first_fit": "S", "preprocessed_map": "P"}
    for i in range(len(input_keys)):
        for j in range(len(DTYPES)):
            label = short[winner_labels[i, j]]
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    legend_handles = [Patch(facecolor=COLORS[s], label=s) for s in STRATEGIES]
    ax.legend(handles=legend_handles, title="Winner", loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_summary_bars(input_keys: List[str], times: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path) -> None:
    # Mean decision time per strategy per dtype (averaged over inputs).
    means = {s: [] for s in STRATEGIES}
    for s in STRATEGIES:
        for dtype in DTYPES:
            vals = [times[s][k][dtype] for k in input_keys]
            means[s].append(float(np.nanmean(vals)))

    x = np.arange(len(DTYPES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, s in enumerate(STRATEGIES):
        ax.bar(x + (idx - 1) * width, means[s], width=width, label=s, color=COLORS[s])
    ax.set_xticks(x)
    ax.set_xticklabels(DTYPES)
    ax.set_ylabel("Mean decision_us across inputs")
    ax.set_title("Mean Decision Time by Strategy and Dtype")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_mean_decision_us_by_dtype.png", dpi=180)
    plt.close(fig)

    # Overall total decision time across all inputs and dtypes.
    totals = []
    for s in STRATEGIES:
        total = 0.0
        for input_key in input_keys:
            for dtype in DTYPES:
                total += times[s][input_key][dtype]
        totals.append(total)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(STRATEGIES, totals, color=[COLORS[s] for s in STRATEGIES])
    ax.set_ylabel("Total decision_us (all inputs x dtypes)")
    ax.set_title("Overall Decision Time by Strategy (Lower is Better)")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_total_decision_us.png", dpi=180)
    plt.close(fig)


def write_long_csv(input_keys: List[str], times: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "input_m", "input_n", "input_k", "dtype", "decision_us", "is_fastest"])
        for input_key in input_keys:
            m_str, n_str, k_str = input_key.split(",")
            for dtype in DTYPES:
                vals = {s: times[s][input_key][dtype] for s in STRATEGIES}
                best = min(vals, key=vals.get)
                for s in STRATEGIES:
                    writer.writerow([s, int(m_str), int(n_str), int(k_str), dtype, f"{vals[s]:.9f}", int(s == best)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick visualization for decision-time strategy JSON outputs."
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        default="../trials/strategy_decision_jsons",
        help="Directory containing hashmap_27.json, sorted_first_fit.json, preprocessed_map.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../trials/strategy_decision_jsons/plots",
        help="Directory to write plots and summary CSV",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    in_dir = Path(args.in_dir) if os.path.isabs(args.in_dir) else (base / args.in_dir)
    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else (base / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_by_strategy = {}
    for strategy in STRATEGIES:
        path = in_dir / f"{strategy}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data_by_strategy[strategy] = load_strategy_file(path)

    input_keys, times = collect_decision_times(data_by_strategy)
    plot_fastest_heatmap(input_keys, times, out_dir / "strategy_fastest_heatmap.png")
    plot_summary_bars(input_keys, times, out_dir)
    write_long_csv(input_keys, times, out_dir / "strategy_decision_us_long.csv")

    print(f"[DONE] Wrote plots to {out_dir}")
    print(f"  - {out_dir / 'strategy_fastest_heatmap.png'}")
    print(f"  - {out_dir / 'strategy_mean_decision_us_by_dtype.png'}")
    print(f"  - {out_dir / 'strategy_total_decision_us.png'}")
    print(f"  - {out_dir / 'strategy_decision_us_long.csv'}")


if __name__ == "__main__":
    main()
