import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import colorsys
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(28, 42))
fig.patch.set_facecolor("white")

fig.text(0.5, 0.99, "Idefics3 Connector: Complete Tensor Transformation & AIE Implementation",
         fontsize=20, ha="center", fontweight="bold", va="top")

# Shared color helpers
def block_color(token_id, sat=0.45, val=0.92):
    return colorsys.hsv_to_rgb(token_id / 64.0, sat, val)

row_colors_16 = [colorsys.hsv_to_rgb(i / 16.0, 0.5, 0.9) for i in range(16)]
group_palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# ═══════════════════════════════════════════════════════════════
# SECTION 1: High-level Pipeline
# ═══════════════════════════════════════════════════════════════
sec1_y = 0.92
fig.text(0.5, sec1_y, "Section 1 — High-Level Pipeline",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#eaf2f8", edgecolor="#2980b9", pad=6))

ax1 = fig.add_axes([0.03, 0.855, 0.94, 0.055])
ax1.set_xlim(0, 40)
ax1.set_ylim(0, 4)
ax1.axis("off")

# Pipeline boxes
boxes = [
    (1,  6, "(1, 1024, 768)\nVision Encoder\nOutput", "#d4e6f1"),
    (9, 6,  "pixel_shuffle\nscale_factor=4\n(reshape only)", "#fdebd0"),
    (17, 6, "(1, 64, 12288)\nShuffled\nTokens", "#fadbd8"),
    (25, 6, "Linear\n(12288, 960)\nno bias", "#d5f5e3"),
    (33, 5, "(1, 64, 960)\nTo LLM", "#d4e6f1"),
]
for bx, bw, label, color in boxes:
    ax1.add_patch(mpatches.FancyBboxPatch((bx, 0.3), bw, 3.2,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor="black", lw=1.5))
    ax1.text(bx + bw / 2, 1.9, label, ha="center", va="center", fontsize=9, fontweight="bold")

for ax_x in [7.3, 15.3, 23.3, 31.3]:
    ax1.annotate("", xy=(ax_x + 1.2, 1.9), xytext=(ax_x, 1.9),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

# ═══════════════════════════════════════════════════════════════
# SECTION 2: Pixel Shuffle — 6 reshape/permute steps
# ═══════════════════════════════════════════════════════════════
sec2_y = 0.845
fig.text(0.5, sec2_y, "Section 2 — Pixel Shuffle: 6 Steps (view/permute/reshape)",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#fdebd0", edgecolor="#e67e22", pad=6))

ax2 = fig.add_axes([0.02, 0.76, 0.96, 0.075])
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 6)
ax2.axis("off")

steps = [
    ("(1,1024,768)\n[B, seq, E]", "#d4e6f1", "view\nseq→H×W"),
    ("(1,32,32,768)\n[B, H, W, E]", "#fef9e7", "view\ngroup W by 4"),
    ("(1,32,8,3072)\n[B, H, W/4, E×4]", "#fdebd0", "permute\nswap H↔W/4"),
    ("(1,8,32,3072)\n[B, W/4, H, E×4]", "#fadbd8", "reshape\ngroup H by 4"),
    ("(1,8,8,12288)\n[B, W/4, H/4, E×16]", "#e8daef", "permute\nswap back"),
    ("(1,8,8,12288)\n[B, H/4, W/4, E×16]", "#d5f5e3", "reshape\nflatten spatial"),
    ("(1,64,12288)\n[B, seq/16, E×16]", "#d4e6f1", None),
]
sx = 1
for i, (shape, color, op) in enumerate(steps):
    w = 5.8
    ax2.add_patch(mpatches.FancyBboxPatch((sx, 0.5), w, 4.5,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="black", lw=1.2))
    ax2.text(sx + w / 2, 2.7, shape, ha="center", va="center", fontsize=7.5, fontweight="bold")
    if op:
        ax2.annotate("", xy=(sx + w + 0.8, 2.7), xytext=(sx + w + 0.1, 2.7),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
        ax2.text(sx + w + 0.45, 4.7, op, ha="center", fontsize=6.5, color="#7f8c8d")
    sx += w + 1.1

# ═══════════════════════════════════════════════════════════════
# SECTION 3: Spatial intuition — 32×32 → 8×8 with 4×4 blocks
# ═══════════════════════════════════════════════════════════════
sec3_y = 0.75
fig.text(0.5, sec3_y, "Section 3 — Spatial Intuition: 32×32 Grid → 8×8 Grid (4×4 block pooling)",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#d5f5e3", edgecolor="#27ae60", pad=6))

# Left: 32x32 grid
ax3a = fig.add_axes([0.02, 0.58, 0.30, 0.16])
ax3a.set_title("32×32 input (1024 tokens)", fontsize=11, pad=6)
from matplotlib.colors import ListedColormap
colors_64 = [block_color(i) for i in range(64)]
cmap64 = ListedColormap(colors_64)
grid = np.zeros((32, 32))
for r in range(32):
    for c in range(32):
        grid[r, c] = (r // 4) * 8 + (c // 4)
ax3a.imshow(grid, cmap=cmap64, vmin=0, vmax=63, aspect="equal")
for i in range(0, 33, 4):
    ax3a.axhline(i - 0.5, color="black", lw=1.2)
    ax3a.axvline(i - 0.5, color="black", lw=1.2)
for br in range(8):
    for bc in range(8):
        if (br + bc) % 2 == 0:
            ax3a.text(bc * 4 + 1.5, br * 4 + 1.5, f"T{br*8+bc}",
                      ha="center", va="center", fontsize=5, fontweight="bold")
ax3a.set_xlabel("col (32)")
ax3a.set_ylabel("row (32)")
ax3a.set_xticks([0, 7, 15, 23, 31])
ax3a.set_yticks([0, 7, 15, 23, 31])

# Middle: zoom one 4x4 block
ax3b = fig.add_axes([0.36, 0.58, 0.25, 0.16])
ax3b.set_title("Zoom: one 4×4 block → 16 cells\neach cell has 768 embedding dims", fontsize=10, pad=6)
ax3b.set_xlim(0, 10)
ax3b.set_ylim(0, 10)
ax3b.axis("off")

cell_sz = 2.0
bx, by = 1, 3.5
for dr in range(4):
    for dc in range(4):
        ci = dr * 4 + dc
        x = bx + dc * cell_sz
        y = by + (3 - dr) * cell_sz
        ax3b.add_patch(mpatches.Rectangle((x, y), cell_sz, cell_sz,
            facecolor=row_colors_16[ci], edgecolor="black", lw=1))
        seq_idx = dr * 32 + dc
        ax3b.text(x + cell_sz / 2, y + cell_sz * 0.6, f"seq {seq_idx}",
                  ha="center", va="center", fontsize=7, fontweight="bold")
        ax3b.text(x + cell_sz / 2, y + cell_sz * 0.3, "768 dims",
                  ha="center", va="center", fontsize=5.5, color="gray")

ax3b.text(5, 2.5, "16 cells × 768 = 12,288 dims per token", ha="center", fontsize=9,
          fontweight="bold", color="#c0392b")

# Right: 8x8 grid
ax3c = fig.add_axes([0.66, 0.58, 0.30, 0.16])
ax3c.set_title("8×8 output (64 tokens, E=12288)", fontsize=11, pad=6)
for r in range(8):
    for c in range(8):
        tid = r * 8 + c
        rgb = block_color(tid)
        ax3c.add_patch(mpatches.Rectangle((c, 7 - r), 1, 1,
            facecolor=rgb, edgecolor="black", lw=1))
        ax3c.text(c + 0.5, 7 - r + 0.65, f"T{tid}", ha="center", va="center",
                  fontsize=7, fontweight="bold")
        ax3c.text(c + 0.5, 7 - r + 0.30, f"12288d", ha="center", va="center",
                  fontsize=5.5, color="dimgray")
ax3c.set_xlim(0, 8)
ax3c.set_ylim(0, 8)
ax3c.set_aspect("equal")
ax3c.set_xlabel("block col")
ax3c.set_ylabel("block row")
ax3c.set_xticks(range(9))
ax3c.set_yticks(range(9))

# ═══════════════════════════════════════════════════════════════
# SECTION 4: Exact row-level data movement
# ═══════════════════════════════════════════════════════════════
sec4_y = 0.57
fig.text(0.5, sec4_y, "Section 4 — Exact Data Movement: How 16 Rows Become 1 Row of 12,288",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#fadbd8", edgecolor="#e74c3c", pad=6))

ax4 = fig.add_axes([0.02, 0.385, 0.96, 0.175])
ax4.set_xlim(0, 42)
ax4.set_ylim(-1, 18)
ax4.axis("off")

# Left: Input column showing gathered rows
ax4.text(3.5, 17.5, "Input (1024, 768)", fontsize=11, ha="center", fontweight="bold")
ax4.text(3.5, 16.8, "16 scattered rows for Token 0", fontsize=9, ha="center", color="gray")

groups_info = [
    (0,   [0, 1, 2, 3],       "k=0"),
    (32,  [32, 33, 34, 35],   "k=1"),
    (64,  [64, 65, 66, 67],   "k=2"),
    (96,  [96, 97, 98, 99],   "k=3"),
]

y_pos = 15.5
for g_idx, (g_start, rows, g_label) in enumerate(groups_info):
    for local_i, row_idx in enumerate(rows):
        ci = g_idx * 4 + local_i
        ax4.add_patch(mpatches.Rectangle((0.5, y_pos), 5.5, 0.7,
            facecolor=row_colors_16[ci], edgecolor="black", lw=0.8))
        ax4.text(3.25, y_pos + 0.35, f"row {row_idx}  (768 values)",
                 ha="center", va="center", fontsize=7, fontweight="bold")
        y_pos -= 0.8
    # Group label
    bracket_top = y_pos + 0.8 * 4 + 0.7
    bracket_bot = y_pos + 0.8
    ax4.plot([6.3, 6.3], [bracket_bot, bracket_top], color=group_palette[g_idx], lw=2.5)
    ax4.plot([6.1, 6.5], [bracket_top, bracket_top], color=group_palette[g_idx], lw=2.5)
    ax4.plot([6.1, 6.5], [bracket_bot, bracket_bot], color=group_palette[g_idx], lw=2.5)
    ax4.text(7.0, (bracket_top + bracket_bot) / 2, g_label,
             ha="left", va="center", fontsize=9, fontweight="bold", color=group_palette[g_idx])
    y_pos -= 0.4

# Middle: Copy kernel arrows
ax4.text(12.5, 17.5, "Copy Kernel: (4, 768) → (1, 3072)", fontsize=11, ha="center", fontweight="bold")
ax4.text(12.5, 16.8, "Flatten 4 rows into 1 wide row", fontsize=9, ha="center", color="gray")

for g_idx in range(4):
    base_y = 14.5 - g_idx * 3.8
    # Arrow from left to middle
    ax4.annotate("", xy=(10, base_y), xytext=(8.0, base_y),
        arrowprops=dict(arrowstyle="-|>", color=group_palette[g_idx], lw=1.5))
    ax4.text(9, base_y + 0.5, f"flatten", ha="center", fontsize=7, style="italic")

    # Flattened bar
    seg_w = 1.5
    for local_i in range(4):
        ci = g_idx * 4 + local_i
        row_idx = groups_info[g_idx][1][local_i]
        sx = 10.2 + local_i * seg_w
        ax4.add_patch(mpatches.Rectangle((sx, base_y - 0.4), seg_w, 0.8,
            facecolor=row_colors_16[ci], edgecolor="black", lw=0.8))
        ax4.text(sx + seg_w / 2, base_y, f"r{row_idx}",
                 ha="center", va="center", fontsize=6.5, fontweight="bold")

    # Label
    k_start = g_idx * 3072
    k_end = (g_idx + 1) * 3072
    ax4.text(10.2 + 2 * seg_w, base_y - 0.8,
             f"cols [{k_start}:{k_end}]", ha="center", fontsize=7, color="gray")

# Right: Final concatenated output row
ax4.text(29, 17.5, "Output Row 0 of (64, 12288)", fontsize=11, ha="center", fontweight="bold")
ax4.text(29, 16.8, "16 segments × 768 = 12,288 columns", fontsize=9, ha="center", color="gray")

seg_h = 0.75
out_x = 19
out_y = 15.5

for g_idx in range(4):
    for local_i in range(4):
        ci = g_idx * 4 + local_i
        row_idx = groups_info[g_idx][1][local_i]
        col_start = ci * 768
        col_end = col_start + 768

        ax4.add_patch(mpatches.Rectangle((out_x, out_y), 18, seg_h,
            facecolor=row_colors_16[ci], edgecolor="black", lw=0.8))
        ax4.text(out_x + 9, out_y + seg_h / 2,
            f"row {row_idx} embeddings  →  output cols [{col_start}:{col_end}]",
            ha="center", va="center", fontsize=7, fontweight="bold")
        out_y -= seg_h

    # Group bracket
    bracket_top = out_y + seg_h * 4 + seg_h
    bracket_bot = out_y + seg_h
    ax4.plot([out_x - 0.4, out_x - 0.4], [bracket_bot, bracket_top],
             color=group_palette[g_idx], lw=3)
    ax4.text(out_x - 0.8, (bracket_top + bracket_bot) / 2,
             groups_info[g_idx][2], ha="center", va="center",
             fontsize=9, fontweight="bold", color=group_palette[g_idx], rotation=90)
    out_y -= 0.25

# Arrow from middle to right
ax4.annotate("", xy=(18.5, 8), xytext=(16.8, 8),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
ax4.text(17.6, 9, "concat", ha="center", fontsize=8, fontweight="bold")

# Address formula box
ax4.add_patch(mpatches.FancyBboxPatch((19, -0.5), 18, 2.0,
    boxstyle="round,pad=0.1", facecolor="#fef9e7", edgecolor="#d4ac0d", lw=1.5))
ax4.text(28, 1.1, "Address formula — token i, group k (0-3), sub-row dc (0-3), embed e (0-767):",
    ha="center", fontsize=8.5, fontweight="bold")
ax4.text(28, 0.3,
    "input_row = (i//8)×128 + (i%8)×4 + k×32 + dc    input_col = e    |    "
    "output_row = i    output_col = k×3072 + dc×768 + e",
    ha="center", fontsize=8, fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: Current AIE Implementation
# ═══════════════════════════════════════════════════════════════
sec5_y = 0.375
fig.text(0.5, sec5_y, "Section 5 — Current AIE Implementation: Two Separate Phases",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#fadbd8", edgecolor="#e74c3c", pad=6))

ax5 = fig.add_axes([0.02, 0.265, 0.96, 0.10])
ax5.set_xlim(0, 42)
ax5.set_ylim(0, 10)
ax5.axis("off")

# Phase 1
ax5.add_patch(mpatches.FancyBboxPatch((0.5, 4.5), 8, 5,
    boxstyle="round,pad=0.15", facecolor="#d4e6f1", edgecolor="black", lw=1.5))
ax5.text(4.5, 9, "Input A\n(1024, 768)", ha="center", fontsize=9, fontweight="bold")
ax5.text(4.5, 7.0, "Groups of 4 rows\nat stride 32", ha="center", fontsize=8, color="gray")
ax5.text(4.5, 5.5, "SEQ=1024, EMBD=768", ha="center", fontsize=7, fontfamily="monospace", color="gray")

ax5.annotate("", xy=(10.5, 7), xytext=(9, 7),
    arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2))
ax5.text(9.7, 8.0, "copy kernel\n×256 calls\nmapping=[4]", ha="center", fontsize=7.5,
         fontweight="bold", color="#e74c3c")

ax5.add_patch(mpatches.FancyBboxPatch((11, 4.5), 8, 5,
    boxstyle="round,pad=0.15", facecolor="#fdebd0", edgecolor="#e67e22", lw=2))
ax5.text(15, 9, "Intermediate Buffer\n(64, 12288) = 3 MB", ha="center", fontsize=9, fontweight="bold",
         color="#e67e22")
ax5.text(15, 7, "Must be fully\nmaterialized", ha="center", fontsize=8, color="#c0392b")
ax5.text(15, 5.5, "NEW_SEQ=64, NEW_EMBD=12288", ha="center", fontsize=7,
         fontfamily="monospace", color="gray")

# Phase 2
ax5.annotate("", xy=(21, 7), xytext=(19.5, 7),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

ax5.add_patch(mpatches.FancyBboxPatch((21.5, 4.5), 6, 5,
    boxstyle="round,pad=0.15", facecolor="#fdebd0", edgecolor="black", lw=1.5))
ax5.text(24.5, 9, "Shuffled A\n(64, 12288)", ha="center", fontsize=9, fontweight="bold")
ax5.text(24.5, 6.5, "M=1 tile\nK=192 tiles\nof 64", ha="center", fontsize=8, color="gray")

ax5.text(28.2, 7, "×", fontsize=16, ha="center", va="center", fontweight="bold")

ax5.add_patch(mpatches.FancyBboxPatch((29, 4.5), 4.5, 5,
    boxstyle="round,pad=0.15", facecolor="#d5f5e3", edgecolor="black", lw=1.5))
ax5.text(31.25, 9, "Weight W\n(12288, 960)", ha="center", fontsize=9, fontweight="bold")
ax5.text(31.25, 6.5, "N=15 tiles\nof 64", ha="center", fontsize=8, color="gray")

ax5.text(34.2, 7, "=", fontsize=16, ha="center", va="center", fontweight="bold")

ax5.add_patch(mpatches.FancyBboxPatch((35, 4.5), 3.5, 5,
    boxstyle="round,pad=0.15", facecolor="#d4e6f1", edgecolor="black", lw=1.5))
ax5.text(36.75, 9, "Output\n(64, 960)", ha="center", fontsize=9, fontweight="bold")
ax5.text(36.75, 6.5, "15 output\ntiles", ha="center", fontsize=8, color="gray")

# Summary
ax5.add_patch(mpatches.FancyBboxPatch((0.5, 0.5), 38, 3.5,
    boxstyle="round,pad=0.15", facecolor="#fef9e7", edgecolor="#c0392b", lw=1.5))
ax5.text(19.5, 3.3,
    "Phase 1: 64 tokens × 4 groups = 256 copy kernel calls   |   "
    "Phase 2: 1 × 15 × 192 = 2,880 matmul + 2,880 accumulate = 5,760 calls",
    ha="center", fontsize=9, fontweight="bold")
ax5.text(19.5, 2.0,
    "Total: 6,016 kernel calls  +  3 MB intermediate buffer  +  "
    "full buffer write/read round-trip before GEMM can start",
    ha="center", fontsize=9, color="#c0392b")
ax5.text(19.5, 0.9,
    "GEMM tile = 64×64, mapping=[4,4] → 16 cores, each core: A_local(16×64) + B_local(64×16) + C_local(16×16) ≈ 9 KB",
    ha="center", fontsize=8, color="gray")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: K-tile → Address Mapping (key insight)
# ═══════════════════════════════════════════════════════════════
sec6_y = 0.255
fig.text(0.5, sec6_y, "Section 6 — Key Insight: Each K-tile Maps to One Original Row",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#d5f5e3", edgecolor="#27ae60", pad=6))

ax6 = fig.add_axes([0.02, 0.175, 0.96, 0.07])
ax6.set_xlim(0, 42)
ax6.set_ylim(0, 10)
ax6.axis("off")

# Draw the 12288-dim bar
bar_y = 5
bar_h = 3
group_w = 9

for g in range(4):
    gx = 1 + g * (group_w + 0.5)
    ax6.add_patch(mpatches.Rectangle((gx, bar_y), group_w, bar_h,
        facecolor=group_palette[g], alpha=0.2, edgecolor=group_palette[g], lw=2))
    ax6.text(gx + group_w / 2, bar_y + bar_h + 0.3,
        f"group k={g}", ha="center", fontsize=9, fontweight="bold", color=group_palette[g])

    sub_w = group_w / 4
    for dc in range(4):
        sx = gx + dc * sub_w
        ax6.add_patch(mpatches.Rectangle((sx, bar_y), sub_w, bar_h,
            facecolor=group_palette[g], alpha=0.1 + dc * 0.08, edgecolor="gray", lw=0.5))
        ax6.text(sx + sub_w / 2, bar_y + bar_h / 2 + 0.5, f"dc={dc}", ha="center", fontsize=7)
        ax6.text(sx + sub_w / 2, bar_y + bar_h / 2 - 0.5, "768 cols", ha="center", fontsize=6, color="gray")

        # Show K-tile divisions (768/64 = 12 tiles per sub-row)
        for t in range(12):
            tx = sx + t * sub_w / 12
            ax6.plot([tx, tx], [bar_y, bar_y + 0.3], color="gray", lw=0.3)

    k_s, k_e = g * 3072, (g + 1) * 3072
    ax6.text(gx + group_w / 2, bar_y - 0.5, f"k ∈ [{k_s}, {k_e})", ha="center", fontsize=7, color="gray")

ax6.text(39, bar_y + bar_h / 2, "12,288\ntotal", ha="center", fontsize=10,
    fontweight="bold", bbox=dict(facecolor="lightyellow", edgecolor="black", pad=2))

# Explanation
ax6.add_patch(mpatches.FancyBboxPatch((1, 0.3), 37, 3.5,
    boxstyle="round,pad=0.15", facecolor="#d5f5e3", edgecolor="#27ae60", lw=1.5))
ax6.text(19.5, 3.2,
    "768 / 64 = 12  →  each K-tile of 64 lands ENTIRELY within one (group, sub-row) pair",
    ha="center", fontsize=9, fontweight="bold")
ax6.text(19.5, 2.1,
    "= contiguous 64 columns from ONE row in the original (1024, 768) input  →  simple DMA, no scatter/gather needed",
    ha="center", fontsize=9, color="#27ae60")
ax6.text(19.5, 1.0,
    "12 tiles/sub-row × 4 sub-rows × 4 groups = 192 K-tiles  ✓",
    ha="center", fontsize=8, color="gray")

# ═══════════════════════════════════════════════════════════════
# SECTION 7: Fused Approach
# ═══════════════════════════════════════════════════════════════
sec7_y = 0.165
fig.text(0.5, sec7_y, "Section 7 — Fused Approach: Shuffle-on-the-fly + GEMM",
         fontsize=16, ha="center", fontweight="bold",
         bbox=dict(facecolor="#d5f5e3", edgecolor="#27ae60", pad=6))

ax7 = fig.add_axes([0.02, 0.015, 0.96, 0.14])
ax7.set_xlim(0, 42)
ax7.set_ylim(0, 14)
ax7.axis("off")

# Original input
ax7.add_patch(mpatches.FancyBboxPatch((0.5, 4), 5, 8,
    boxstyle="round,pad=0.15", facecolor="#d4e6f1", edgecolor="black", lw=1.5))
ax7.text(3, 11.5, "Original A\n(1024, 768)", ha="center", fontsize=9, fontweight="bold")
ax7.text(3, 9, "Read directly\nfrom here —\nno intermediate\nbuffer!", ha="center", fontsize=8,
    color="#27ae60", fontweight="bold")

# Address computation
ax7.annotate("", xy=(7, 8), xytext=(6, 8),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

ax7.add_patch(mpatches.FancyBboxPatch((7.5, 4), 8, 8,
    boxstyle="round,pad=0.2", facecolor="#fef9e7", edgecolor="#f39c12", lw=2))
ax7.text(11.5, 11.5, "Address Computation\n(per K-tile)", ha="center", fontsize=10,
    fontweight="bold", color="#e67e22")
ax7.text(11.5, 9.5, "group = k_start // 3072", ha="center", fontsize=8, fontfamily="monospace")
ax7.text(11.5, 8.5, "dc = (k_start % 3072) // 768", ha="center", fontsize=8, fontfamily="monospace")
ax7.text(11.5, 7.5, "e = k_start % 768", ha="center", fontsize=8, fontfamily="monospace")
ax7.text(11.5, 6.2, "For each output token i:", ha="center", fontsize=8, fontweight="bold")
ax7.text(11.5, 5.2, "orig_row = (i//8)×128\n         + (i%8)×4\n         + group×32 + dc",
    ha="center", fontsize=7.5, fontfamily="monospace")

# Virtual A tile
ax7.annotate("", xy=(17.5, 8), xytext=(16, 8),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

ax7.add_patch(mpatches.FancyBboxPatch((18, 5), 4, 6,
    boxstyle="round,pad=0.15", facecolor="#fadbd8", edgecolor="#c0392b", lw=2, ls="--"))
ax7.text(20, 10.5, "A tile (64×64)", ha="center", fontsize=9, fontweight="bold", color="#c0392b")
ax7.text(20, 8.5, "64 rows from\n64 different\norig rows", ha="center", fontsize=8)
ax7.text(20, 6, "Same 64 cols\n[e : e+64]", ha="center", fontsize=8, color="gray")

# ×
ax7.text(23, 8, "×", fontsize=18, ha="center", va="center", fontweight="bold")

# W tile
ax7.add_patch(mpatches.FancyBboxPatch((24, 5.5), 3.5, 5,
    boxstyle="round,pad=0.15", facecolor="#d5f5e3", edgecolor="#27ae60", lw=1.5))
ax7.text(25.75, 10, "W tile\n(64×64)", ha="center", fontsize=9, fontweight="bold")

# +=
ax7.text(28.5, 8, "+=", fontsize=14, ha="center", va="center", fontweight="bold")

# Out tile
ax7.add_patch(mpatches.FancyBboxPatch((29.5, 5.5), 3, 5,
    boxstyle="round,pad=0.15", facecolor="#d4e6f1", edgecolor="#2980b9", lw=1.5))
ax7.text(31, 10, "Out tile\n(64×64)", ha="center", fontsize=9, fontweight="bold")

# Fused loop
ax7.add_patch(mpatches.FancyBboxPatch((33.5, 2), 8, 10.5,
    boxstyle="round,pad=0.2", facecolor="#eaf2f8", edgecolor="#2980b9", lw=2))
ax7.text(37.5, 11.8, "Fused Loop", ha="center", fontsize=11, fontweight="bold", color="#2980b9")
ax7.text(37.5, 10.2, "for n_tile in 0..14:", fontsize=8.5, ha="center", fontfamily="monospace")
ax7.text(37.5, 9.2, "  for k_tile in 0..191:", fontsize=8.5, ha="center", fontfamily="monospace")
ax7.text(37.5, 8.2, "    addr = shuffle_map(k)", fontsize=8.5, ha="center", fontfamily="monospace")
ax7.text(37.5, 7.2, "    load A[addr, e:e+64]", fontsize=8.5, ha="center",
    fontfamily="monospace", color="#c0392b", fontweight="bold")
ax7.text(37.5, 6.2, "    load W[k:k+64, n:n+64]", fontsize=8.5, ha="center", fontfamily="monospace")
ax7.text(37.5, 5.2, "    matmul + accum", fontsize=8.5, ha="center", fontfamily="monospace")

ax7.text(37.5, 3.5, "15 × 192 = 2,880 calls", fontsize=9, ha="center",
    fontweight="bold", color="#27ae60")
ax7.text(37.5, 2.5, "(was 6,016)", fontsize=8, ha="center", color="gray")

# Bottom savings bar
ax7.add_patch(mpatches.FancyBboxPatch((0.5, 0.2), 32, 2.5,
    boxstyle="round,pad=0.1", facecolor="#d5f5e3", edgecolor="#27ae60", lw=2))
ax7.text(16.5, 2.0, "Savings", fontsize=10, ha="center", fontweight="bold", color="#27ae60")
ax7.text(16.5, 0.8,
    "No 3 MB intermediate buffer    |    52% fewer kernel calls (2,880 vs 6,016)    |    "
    "No write-then-read latency — stream from input",
    ha="center", fontsize=9)

plt.savefig("/home/xl434/vla-to-npu/vla/connector_complete.pdf", dpi=150, bbox_inches="tight")
print("Saved to connector_complete.pdf")
