"""Compare SiLU kernel call counts across SILU_SEQ_TILE = 4, 8, 16."""
import matplotlib.pyplot as plt
import numpy as np

TEXT_SEQ = 128
EXP_SEQ = 32
LLAMA_LAYERS = 2

tiles = [4, 8, 16]
sites = [
    ("text_encoder MLP\n(per layer)", lambda t: TEXT_SEQ // t),
    ("action_expert MLP\n(per layer)", lambda t: EXP_SEQ // t),
    ("joint transformer\ntotal (2 layers)",
        lambda t: LLAMA_LAYERS * (TEXT_SEQ // t + EXP_SEQ // t)),
]

values = {tile: [fn(tile) for _, fn in sites] for tile in tiles}

x = np.arange(len(sites))
w = 0.26
colors = {4: "#d97757", 8: "#5b8def", 16: "#3aa17e"}

fig, ax = plt.subplots(figsize=(9.5, 5.2))
for i, tile in enumerate(tiles):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, values[tile], w,
                  label=f"SILU_SEQ_TILE = {tile}", color=colors[tile])
    for bar, val in zip(bars, values[tile]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.6, str(val),
                ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in sites])
ax.set_ylabel("SiLU kernel calls")
ax.set_title("SiLU kernel calls vs SILU_SEQ_TILE  (4 → 8 → 16)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = "/home/ck795/vla-to-npu/analysis/silu_tile_compare.png"
plt.savefig(out, dpi=120)
print("Wrote", out)
for tile in tiles:
    print(f"tile={tile:>2}: {dict(zip([s[0].replace(chr(10), ' ') for s in sites], values[tile]))}")
