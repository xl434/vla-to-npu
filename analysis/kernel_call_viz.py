"""
Kernel call analysis visualization for the VLA-to-NPU project.

4-panel figure (analysis/kernel_call_analysis.png) — every panel shows
"Before" (SiLU tile=4) vs "After" (SiLU tile=16) where applicable.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG: per-function call counts.
# Order:                  RMSNorm GEMM  RoPE  Score  Soft  Value  SiLU  FFN
# (only SiLU column changes between before and after)
# ─────────────────────────────────────────────────────────────────────────────
ops      = ["RMSNorm", "GEMM\n(proj)", "RoPE", "Attn Score", "Softmax", "Attn Value", "SiLU", "FFN Down"]

te_before   = np.array([16, 4, 412, 15, 240, 15, 32, 8])  # text encoder per layer (sums to 742, ~744 with chunk rounding)
te_after    = np.array([16, 4, 412, 15, 240, 15,  8, 8])

ae_s_before = np.array([ 4, 4, 206, 15,   0, 15,  8, 8])
ae_s_after  = np.array([ 4, 4, 206, 15,   0, 15,  2, 8])

ae_c_before = np.array([ 4, 4, 153, 15,  15, 15,  8, 8])
ae_c_after  = np.array([ 4, 4, 153, 15,  15, 15,  2, 8])

func_labels = ["Text Encoder\n(SEQ=128)", "AE Self-Attn\n(SEQ=32)", "AE Cross-Attn\n(SEQ=32)"]

# Joint transformer totals (2 layers: layer0 = te+ae_s, layer1 = te+ae_c)
total_before = 2 * te_before.sum() + ae_s_before.sum() + ae_c_before.sum()  # ~1970 (≈1974 with rounding)
total_after  = 2 * te_after.sum()  + ae_s_after.sum()  + ae_c_after.sum()   # ~1910 (≈1914 with rounding)

# ─────────────────────────────────────────────────────────────────────────────
# Pie chart values: pipeline category totals (2-layer)
# ─────────────────────────────────────────────────────────────────────────────
pie_categories = ["RoPE", "Masked Softmax", "Attn Score GEMMs",
                  "Attn Value GEMMs", "SiLU", "FFN Down", "Other"]
pie_before = [1183, 480, 60, 60, 80, 32, 79]   # sum = 1974
pie_after  = [1183, 480, 60, 60, 20, 32, 79]   # sum = 1914
pie_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#f39c12",
              "#2ecc71", "#3498db", "#95a5a6"]

# ─────────────────────────────────────────────────────────────────────────────
# Fusion roadmap (waterfall) — SiLU tile merge is now applied
# ─────────────────────────────────────────────────────────────────────────────
fusion_labels = [
    "Original\nbaseline",
    "SiLU tile=16\n(applied)",
    "RoPE\nfull fusion",
    "Softmax\nhead batch",
    "Attn GEMM\nbatch",
    "RMSNorm\ntile increase",
]
fusion_totals = [1974, 1914, 846, 398, 286, 258]
fusion_done   = [False, True, False, False, False, False]

# ─────────────────────────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("NPU Kernel Call Analysis — Before (SiLU tile=4) vs After (SiLU tile=16)",
             fontsize=15, fontweight="bold", y=0.99)
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)

# ─── Panel 1: total kernel calls per layer, grouped before/after ────────────
ax1 = fig.add_subplot(gs[0, 0])

x = np.arange(len(func_labels))
w = 0.36
totals_before = [te_before.sum(), ae_s_before.sum(), ae_c_before.sum()]
totals_after  = [te_after.sum(),  ae_s_after.sum(),  ae_c_after.sum()]

bars_b = ax1.bar(x - w/2, totals_before, w, label="Before (tile=4)",
                 color="#d97757", edgecolor="white")
bars_a = ax1.bar(x + w/2, totals_after,  w, label="After (tile=16)",
                 color="#5b8def", edgecolor="white")

for bar, val in zip(bars_b, totals_before):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 8, str(val),
             ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars_a, totals_after):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 8, str(val),
             ha="center", fontsize=9, fontweight="bold")

# delta annotation
for xi, (b, a) in enumerate(zip(totals_before, totals_after)):
    ax1.annotate(f"−{b - a}", xy=(xi + w/2, a), xytext=(xi + w/2 + 0.05, (b + a) / 2),
                 fontsize=8.5, color="#c0392b", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(func_labels, fontsize=9)
ax1.set_ylabel("Kernel calls per forward pass")
ax1.set_title("Total Calls per Forward Function", fontweight="bold", pad=8)
ax1.legend(fontsize=9, framealpha=0.9)
ax1.grid(axis="y", alpha=0.3)
ax1.set_axisbelow(True)

# ─── Panel 2: per-op stack, before vs after side-by-side ────────────────────
ax2 = fig.add_subplot(gs[0, 1])

op_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f",
             "#e67e22", "#9b59b6", "#1abc9c", "#34495e"]

# Pairs of bars: (function, state)  position
positions = []
for i in range(len(func_labels)):
    positions.append(i - 0.21)  # before
    positions.append(i + 0.21)  # after

bw = 0.36
all_calls = [te_before, te_after, ae_s_before, ae_s_after, ae_c_before, ae_c_after]

bottoms = np.zeros(len(positions))
for op_i, (op_name, op_color) in enumerate(zip(ops, op_colors)):
    vals = np.array([calls[op_i] for calls in all_calls])
    ax2.bar(positions, vals, width=bw, bottom=bottoms,
            color=op_color, edgecolor="white", linewidth=0.4,
            label=op_name.replace("\n", " ") if op_i < len(ops) else None)
    for px, v, b in zip(positions, vals, bottoms):
        if v >= 12:
            ax2.text(px, b + v/2, str(v), ha="center", va="center",
                     fontsize=6.8, color="white", fontweight="bold")
    bottoms += vals

# x ticks at function midpoints, with B/A labels under each pair
ax2.set_xticks([p for p in positions])
ax2.set_xticklabels(["B", "A"] * 3, fontsize=8)
# group labels above
for i, fl in enumerate(func_labels):
    ax2.text(i, -bottoms.max() * 0.085, fl, ha="center", va="top",
             fontsize=9, transform=ax2.get_xaxis_transform())

ax2.set_ylabel("Kernel calls (single layer)")
ax2.set_title("Per-Operation Call Breakdown — B = Before, A = After",
              fontweight="bold", pad=8)
ax2.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)

# total annotations
for px, total in zip(positions, bottoms):
    ax2.text(px, total + 6, str(int(total)), ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color="#333333")

# ─── Panel 3: side-by-side donut charts ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title("Operation Distribution — Before / After",
              fontweight="bold", pad=8)
ax3.axis("off")

# Two inset axes
left_ax  = fig.add_axes([0.06, 0.08, 0.21, 0.32])
right_ax = fig.add_axes([0.30, 0.08, 0.21, 0.32])

for axp, vals, title, total in [
    (left_ax,  pie_before, "Before  (1974)", sum(pie_before)),
    (right_ax, pie_after,  "After  (1914)",  sum(pie_after)),
]:
    wedges, texts, autotexts = axp.pie(
        vals, labels=None, colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=140, pctdistance=0.74,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.2),
    )
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_color("white")
        at.set_fontweight("bold")
    axp.set_title(title, fontsize=10, fontweight="bold")

# Shared legend below the two donuts
legend_handles = [mpatches.Patch(color=pie_colors[i], label=pie_categories[i])
                  for i in range(len(pie_categories))]
fig.legend(handles=legend_handles,
           loc="lower left", bbox_to_anchor=(0.06, 0.01),
           ncol=4, fontsize=8, framealpha=0.9)

# ─── Panel 4: fusion waterfall (with SiLU step now applied) ─────────────────
ax4 = fig.add_subplot(gs[1, 1])

bar_colors = ["#95a5a6", "#1abc9c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
x4 = np.arange(len(fusion_labels))
bars4 = ax4.bar(x4, fusion_totals, color=bar_colors, edgecolor="white",
                linewidth=0.5, width=0.6, zorder=3)

# mark the applied step
for i, done in enumerate(fusion_done):
    if done:
        ax4.text(i, fusion_totals[i] + 80, "✓ DONE",
                 ha="center", color="#16a085", fontsize=9, fontweight="bold")

for i in range(1, len(fusion_totals)):
    y_from = fusion_totals[i - 1]
    y_to   = fusion_totals[i]
    saved  = y_from - y_to
    ax4.annotate("", xy=(i, y_to + 5), xytext=(i, y_from - 5),
                 arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.5),
                 zorder=4)
    ax4.text(i + 0.25, (y_from + y_to) / 2,
             f"−{saved:,}", va="center", ha="left", fontsize=8,
             color="#c0392b", fontweight="bold")

for bar, total in zip(bars4, fusion_totals):
    ax4.text(bar.get_x() + bar.get_width() / 2, total + 20,
             f"{total:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax4.set_xticks(x4)
ax4.set_xticklabels(fusion_labels, fontsize=8.5)
ax4.set_ylabel("Total kernel calls (2-layer pipeline)")
ax4.set_title("Fusion Roadmap — SiLU step applied; remaining steps projected",
              fontweight="bold", pad=8)
ax4.set_ylim(0, 2300)
ax4.yaxis.grid(True, alpha=0.3, zorder=0)
ax4.set_axisbelow(True)

# annotations
ax4.text(0.0, 2200, "1974\n(~47 s @ 24 ms)", ha="center", va="top",
         fontsize=8.5, color="#7f8c8d")
ax4.text(1.0, 2200, "1914\n(meas 25.4 s)", ha="center", va="top",
         fontsize=8.5, color="#16a085", fontweight="bold")
ax4.text(5.0, 2200, "258\n(~6.2 s proj)", ha="center", va="top",
         fontsize=8.5, color="#27ae60", fontweight="bold")

out_path = "/home/ck795/vla-to-npu/analysis/kernel_call_analysis.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Saved → {out_path}")
print(f"Total before: {total_before}, after: {total_after}")
plt.close(fig)
