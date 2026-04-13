"""
Plot repeated kernel calls from JSON data.
NPU time and Overhead shown as side-by-side bars for each trial.

Usage:
  python plot_repeated.py                              # uses default JSON
  python plot_repeated.py path/to/data.json            # custom JSON
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "repeated_calls_data.json")


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    with open(json_path) as f:
        data = json.load(f)

    # Clean up display names
    def clean_name(n):
        for remove in [", 2KB", ", f32", " 2KB", " f32"]:
            n = n.replace(remove, "")
        return n

    kernel_names = list(data.keys())
    display_names = [clean_name(n) for n in kernel_names]
    n_kernels = len(kernel_names)

    fig, axes = plt.subplots(n_kernels, 1, figsize=(12, 2.8 * n_kernels))
    if n_kernels == 1:
        axes = [axes]

    BAR_W = 0.25
    BLUE = "#2C7BE5"
    ORANGE = "#F59E0B"
    GRAY = "#9CA3AF"

    for ax, (name, dname) in zip(axes, zip(kernel_names, display_names)):
        trials = data[name]
        n = len(trials)
        x = np.arange(n)

        npus = [t["npu_ms"] if t["npu_ms"] is not None else 0 for t in trials]
        overheads = [t["overhead_ms"] if t["overhead_ms"] is not None else 0 for t in trials]
        walls = [t["wall_ms"] for t in trials]
        labels = [f"Trial {t['trial']}" for t in trials]

        # Three side-by-side bars
        ax.bar(x - BAR_W, walls, BAR_W, color=GRAY, label="Wall-clock total", zorder=3)
        ax.bar(x, overheads, BAR_W, color=ORANGE, label="Overhead (Context + Prep + DMA sync)", zorder=3)
        ax.bar(x + BAR_W, npus, BAR_W, color=BLUE, label="NPU time (Launch + Compute)", zorder=3)

        # Annotate values
        for i in range(n):
            ax.text(i - BAR_W, walls[i] + 0.3, f"{walls[i]:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="#555")
            ax.text(i, overheads[i] + 0.3, f"{overheads[i]:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="#B45309")
            ax.text(i + BAR_W, npus[i] + 0.3, f"{npus[i]:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color=BLUE)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Time (ms)")
        ax.set_title(dname, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, max(walls) * 1.25)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), frameon=True)

    plt.suptitle(
        "Repeated Calls to Same Kernel: NPU Time vs Dispatch Overhead\n"
        "Wall-clock = NPU time + Overhead",
        fontsize=13, fontweight="bold", y=1.05,
    )
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "repeated_calls.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
