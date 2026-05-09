"""
Before/after comparison for the rope_apply_packed -> rope_fused swap.
Saves analysis/rope_fusion_compare.png.

Numbers come from analysis/kernel_call_analysis_fused.md.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- Constants ----------
MS_PER_CALL = 24.0  # measured subprocess dispatch overhead per NPU call

# ---------- RoPE-only per-block calls (joint transformer, 2 layers) ----------
blocks = ["TE Q+K\n(2 layers)", "AE self Q+K", "AE cross Q"]
rope_before = np.array([824, 206, 153])
rope_after  = np.array([160,  20,  15])

# ---------- Per-function totals ----------
funcs = ["text_encoder", "AE self", "AE cross"]
func_total_before = np.array([744, 262, 224])
func_total_after  = np.array([412,  76,  86])
func_rope_before  = np.array([412, 206, 153])  # rope share, single layer
func_rope_after   = np.array([ 80,  20,  15])

# ---------- Joint-transformer totals ----------
total_before = 1974
total_after  = 986
rope_total_before = 1183
rope_total_after  = 195

# ---------- Draw ----------
fig = plt.figure(figsize=(18, 11))
fig.suptitle("RoPE Fusion: rope_apply_packed (9 kernels)  ->  rope_fused (1 kernel)",
             fontsize=15, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

C_BEFORE = "#e74c3c"
C_AFTER  = "#27ae60"

# ============================================================
# Panel 1: per-block RoPE calls (before vs after)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(blocks))
w = 0.38
b1 = ax1.bar(x - w/2, rope_before, width=w, color=C_BEFORE,
             edgecolor="white", label="Packed (before)")
b2 = ax1.bar(x + w/2, rope_after,  width=w, color=C_AFTER,
             edgecolor="white", label="Fused (after)")
for bars, vals in [(b1, rope_before), (b2, rope_after)]:
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 12, str(v),
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(blocks, fontsize=9)
ax1.set_ylabel("RoPE kernel calls")
ax1.set_title("Per-block RoPE calls", fontweight="bold", pad=8)
ax1.set_ylim(0, max(rope_before) * 1.18)
ax1.legend(loc="upper right", fontsize=9)
ax1.yaxis.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# ============================================================
# Panel 2: per-function total calls (single layer), RoPE share highlighted
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
x2 = np.arange(len(funcs))
non_rope_before = func_total_before - func_rope_before
non_rope_after  = func_total_after  - func_rope_after
ax2.bar(x2 - w/2, non_rope_before, width=w, color="#bdc3c7",
        edgecolor="white", label="Other (before)")
ax2.bar(x2 - w/2, func_rope_before, width=w, bottom=non_rope_before,
        color=C_BEFORE, edgecolor="white", label="RoPE (before)")
ax2.bar(x2 + w/2, non_rope_after,  width=w, color="#ecf0f1",
        edgecolor="white", label="Other (after)")
ax2.bar(x2 + w/2, func_rope_after,  width=w, bottom=non_rope_after,
        color=C_AFTER, edgecolor="white", label="RoPE (after)")
for xi, (tb, ta) in enumerate(zip(func_total_before, func_total_after)):
    ax2.text(xi - w/2, tb + 8, str(tb), ha="center", va="bottom",
             fontsize=9, fontweight="bold")
    ax2.text(xi + w/2, ta + 8, str(ta), ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(funcs, fontsize=9)
ax2.set_ylabel("Kernel calls per single forward")
ax2.set_title("Per-function calls (single layer)\nRoPE share highlighted",
              fontweight="bold", pad=8)
ax2.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
ax2.set_ylim(0, max(func_total_before) * 1.18)
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# ============================================================
# Panel 3: joint-transformer totals, RoPE vs other
# ============================================================
ax3 = fig.add_subplot(gs[0, 2])
labels3 = ["Before\n(packed)", "After\n(fused)"]
rope_part  = [rope_total_before, rope_total_after]
other_part = [total_before - rope_total_before, total_after - rope_total_after]
x3 = np.arange(2)
ax3.bar(x3, other_part, width=0.55, color="#bdc3c7",
        edgecolor="white", label="Other ops")
ax3.bar(x3, rope_part, width=0.55, bottom=other_part,
        color=[C_BEFORE, C_AFTER], edgecolor="white", label="RoPE")
for i, (op, rp, total) in enumerate(zip(other_part, rope_part,
                                         [total_before, total_after])):
    ax3.text(i, op / 2, f"other\n{op}", ha="center", va="center",
             fontsize=8, color="#2c3e50")
    ax3.text(i, op + rp / 2, f"RoPE\n{rp}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    ax3.text(i, total + 30, str(total), ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax3.set_xticks(x3)
ax3.set_xticklabels(labels3, fontsize=10)
ax3.set_ylabel("Total kernel calls (2-layer joint transformer)")
ax3.set_title(f"Joint-transformer totals\nspeedup {total_before/total_after:.2f}x",
              fontweight="bold", pad=8)
ax3.set_ylim(0, total_before * 1.12)
ax3.legend(loc="upper right", fontsize=9)
ax3.yaxis.grid(True, alpha=0.3)
ax3.set_axisbelow(True)

# ============================================================
# Panel 4: latency comparison (24 ms / call)
# ============================================================
ax4 = fig.add_subplot(gs[1, 0])
ms_before = total_before * MS_PER_CALL / 1000.0
ms_after  = total_after  * MS_PER_CALL / 1000.0
rope_ms_before = rope_total_before * MS_PER_CALL / 1000.0
rope_ms_after  = rope_total_after  * MS_PER_CALL / 1000.0
other_ms_before = ms_before - rope_ms_before
other_ms_after  = ms_after  - rope_ms_after

ax4.bar(x3, [other_ms_before, other_ms_after], width=0.55,
        color="#bdc3c7", edgecolor="white", label="Other ops")
ax4.bar(x3, [rope_ms_before, rope_ms_after], width=0.55,
        bottom=[other_ms_before, other_ms_after],
        color=[C_BEFORE, C_AFTER], edgecolor="white", label="RoPE")
for i, (other_s, rope_s, total_s) in enumerate(zip(
        [other_ms_before, other_ms_after],
        [rope_ms_before, rope_ms_after],
        [ms_before, ms_after])):
    ax4.text(i, other_s / 2, f"{other_s:.1f}s", ha="center", va="center",
             fontsize=9, color="#2c3e50")
    ax4.text(i, other_s + rope_s / 2, f"{rope_s:.1f}s",
             ha="center", va="center", fontsize=9, color="white",
             fontweight="bold")
    ax4.text(i, total_s + 0.6, f"{total_s:.1f}s", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax4.set_xticks(x3)
ax4.set_xticklabels(labels3, fontsize=10)
ax4.set_ylabel(f"Wall time (s) at {MS_PER_CALL:.0f} ms/call")
ax4.set_title(f"Projected latency\n{ms_before:.1f}s -> {ms_after:.1f}s "
              f"(saves {ms_before - ms_after:.1f}s)",
              fontweight="bold", pad=8)
ax4.set_ylim(0, ms_before * 1.15)
ax4.legend(loc="upper right", fontsize=9)
ax4.yaxis.grid(True, alpha=0.3)
ax4.set_axisbelow(True)

# ============================================================
# Panel 5: per-tile RoPE call decomposition
# ============================================================
ax5 = fig.add_subplot(gs[1, 1])
# Show structure of RoPE calls per tile, head-by-head
# Before: 3 shared + per-head (10 calls each = copyL×3 + copyR + mul×4 + sub + add + join)
# After: 0 shared + per-head 1
heads_axis = np.array([1, 5, 15])
per_tile_before = 3 + heads_axis * 10
per_tile_after  = 0 + heads_axis * 1
ax5.plot(heads_axis, per_tile_before, marker="o", color=C_BEFORE, lw=2.0,
         markersize=9, label="Packed: 3 + 10*H")
ax5.plot(heads_axis, per_tile_after, marker="s", color=C_AFTER, lw=2.0,
         markersize=9, label="Fused: 1*H")
for h, pb, pa in zip(heads_axis, per_tile_before, per_tile_after):
    ax5.annotate(f"{pb}", xy=(h, pb), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9,
                 color=C_BEFORE, fontweight="bold")
    ax5.annotate(f"{pa}", xy=(h, pa), xytext=(0, -16),
                 textcoords="offset points", ha="center", fontsize=9,
                 color=C_AFTER, fontweight="bold")
ax5.set_xlabel("Heads in tile (KV_H=5, Q_H=15)")
ax5.set_ylabel("NPU calls per tile")
ax5.set_xticks(heads_axis)
ax5.set_title("RoPE calls per tile vs head count",
              fontweight="bold", pad=8)
ax5.legend(loc="upper left", fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_axisbelow(True)

# ============================================================
# Panel 6: summary table as text
# ============================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary_lines = [
    ("Metric", "Before", "After", "Delta"),
    ("RoPE calls", f"{rope_total_before}", f"{rope_total_after}",
     f"-{rope_total_before - rope_total_after} ({rope_total_before/rope_total_after:.1f}x)"),
    ("Total calls", f"{total_before}", f"{total_after}",
     f"-{total_before - total_after} ({total_before/total_after:.2f}x)"),
    ("RoPE wall time", f"{rope_ms_before:.1f} s", f"{rope_ms_after:.1f} s",
     f"-{rope_ms_before - rope_ms_after:.1f} s"),
    ("Total wall time", f"{ms_before:.1f} s", f"{ms_after:.1f} s",
     f"-{ms_before - ms_after:.1f} s"),
    ("RoPE share of calls",
     f"{rope_total_before/total_before*100:.0f}%",
     f"{rope_total_after/total_after*100:.0f}%", ""),
]
n_rows = len(summary_lines)
col_x = [0.02, 0.36, 0.55, 0.74]
y_top = 0.92
row_h = 0.13
for r, row in enumerate(summary_lines):
    y = y_top - r * row_h
    for c, val in enumerate(row):
        weight = "bold" if r == 0 else "normal"
        color  = "#2c3e50" if r == 0 else "#34495e"
        ax6.text(col_x[c], y, val, ha="left", va="center",
                 fontsize=10, fontweight=weight, color=color,
                 transform=ax6.transAxes)
    if r == 0:
        ax6.axhline(y - row_h * 0.5, xmin=0, xmax=1,
                    color="#7f8c8d", lw=1.0)
ax6.set_title("Summary", fontweight="bold", pad=8, loc="left")

# ============================================================
out_path = "/home/ck795/vla-to-npu/analysis/rope_fusion_compare.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close(fig)
