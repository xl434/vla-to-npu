import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis("off")

BLUE   = "#2C7BE5"
TEAL   = "#00A896"
GRAY   = "#6B7280"
LGRAY  = "#F3F4F6"
BLACK  = "#111827"
WHITE  = "white"
AMBER  = "#F59E0B"
PURPLE = "#7C3AED"

def rbox(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.18, zorder=3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)

def txt(ax, x, y, s, color=BLACK, size=9.5, bold=False, ha="center", va="center", zorder=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size,
            fontweight="bold" if bold else "normal",
            color=color, zorder=zorder)

def arrow_v(ax, x, y1, y2, color=GRAY, lw=1.4):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12), zorder=4)

def arrow_h(ax, x1, x2, y, color=GRAY, lw=1.4):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12), zorder=4)

# Title
txt(ax, 6, 8.6, "Agentic Tiling Exploration", color=BLACK, size=14, bold=True)

# Input
rbox(ax, 4.0, 7.5, 4.0, 0.7, LGRAY, GRAY)
txt(ax, 6.0, 7.85, "Unseen GEMM shape (M, N, K)", color=BLACK, size=10)
arrow_v(ax, 6.0, 7.5, 6.88)

# Decision diamond
diamond_cx, diamond_cy = 6.0, 6.5
dw, dh = 2.2, 0.75
diamond = plt.Polygon([
    [diamond_cx, diamond_cy + dh],
    [diamond_cx + dw, diamond_cy],
    [diamond_cx, diamond_cy - dh],
    [diamond_cx - dw, diamond_cy],
], closed=True, facecolor=LGRAY, edgecolor=GRAY, linewidth=1.3, zorder=3)
ax.add_patch(diamond)
txt(ax, diamond_cx, diamond_cy, "In profiling DB?", color=BLACK, size=9.5)

# YES → right → Retrieve best
arrow_h(ax, diamond_cx + dw, 9.5, diamond_cy, color=TEAL, lw=1.5)
txt(ax, 8.15, diamond_cy + 0.18, "Yes", color=TEAL, size=8.5, bold=True)
rbox(ax, 9.5, 6.1, 2.2, 0.8, "#E6F7F4", TEAL, lw=1.5)
txt(ax, 10.6, 6.5, "Retrieve best\n(m, n, k) directly", color=TEAL, size=8.8)

# NO → down
arrow_v(ax, diamond_cx, diamond_cy - dh, 4.88)
txt(ax, 6.3, 5.5, "No", color=GRAY, size=8.5, bold=True)

# Pad
rbox(ax, 4.0, 4.1, 4.0, 0.7, LGRAY, GRAY)
txt(ax, 6.0, 4.45, "Pad to nearest valid (M', N', K')", color=BLACK, size=10)
arrow_v(ax, 6.0, 4.1, 3.28)

# Agent
rbox(ax, 3.7, 2.0, 4.6, 1.2, "#EFF6FF", BLUE, lw=1.8, radius=0.2)
txt(ax, 6.0, 3.05, "AI Agent  (Claude Sonnet)", color=BLUE, size=10.5, bold=True)
txt(ax, 6.0, 2.62, "Reads: profiling DB · cost model · strategy insights", color=GRAY, size=8.5)
txt(ax, 6.0, 2.25, "Proposes best (m, n, k)  —  or drives sweep offline", color=BLACK, size=8.8)
arrow_v(ax, 6.0, 2.0, 1.18)

# Run on NPU
rbox(ax, 4.0, 0.4, 4.0, 0.7, LGRAY, GRAY)
txt(ax, 6.0, 0.75, "Run on NPU  →  record latency", color=BLACK, size=10)

# Feedback loop: NPU right → up → Retrieve best bottom
ax.plot([8.0, 10.6], [0.75, 0.75], color=PURPLE, lw=1.4, zorder=4)
ax.plot([10.6, 10.6], [0.75, 6.1], color=PURPLE, lw=1.4, zorder=4)
ax.annotate("", xy=(10.6, 6.1), xytext=(10.6, 6.15),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4, mutation_scale=12), zorder=5)
txt(ax, 11.3, 3.5, "Update\nDB", color=PURPLE, size=8.5, bold=True, ha="center")

# Return config (bottom of retrieve box → down)
arrow_v(ax, 10.6, 6.1, 1.2)
rbox(ax, 9.5, 0.4, 2.2, 0.7, "#E6F7F4", TEAL, lw=1.5)
txt(ax, 10.6, 0.75, "Return config\nto runtime", color=TEAL, size=8.5)

# Source inputs feeding agent
sources = [
    ("Profiling DB",      BLUE,   3.0),
    ("Cost Model",        TEAL,   2.4),
    ("Strategy Insights", PURPLE, 1.8),
]
for label, color, sy in sources:
    rbox(ax, 0.3, sy - 0.22, 2.6, 0.44, WHITE, color, lw=1.2, radius=0.12)
    txt(ax, 1.6, sy, label, color=color, size=8.5, bold=True)
    ax.annotate("", xy=(3.7, 2.6 - (3.0 - sy) * 0.35), xytext=(2.9, sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                mutation_scale=10,
                                connectionstyle="arc3,rad=0.0"), zorder=4)
txt(ax, 1.6, 3.55, "Reads from:", color=GRAY, size=8)

plt.tight_layout(pad=0.5)
plt.savefig("agentic_diagram.png", dpi=180, facecolor="white", bbox_inches="tight")