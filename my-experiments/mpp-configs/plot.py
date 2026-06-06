#!/usr/bin/env python3
"""Plot mpp-configs sweep results as heatmaps: tile_shape × simdgroups, one per N."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / "results.csv"

TILES = ["32x16", "32x32", "64x32", "64x64", "128x32", "128x64", "128x128"]
N_VALUES = [512, 1024, 2048, 4096]
MAX_SG = 32

# data[n][tile][sg] = gflops  (sg is 0-indexed: sg-1)
data = {n: {t: [np.nan] * MAX_SG for t in TILES} for n in N_VALUES}

with open(CSV) as f:
    for row in csv.DictReader(f):
        if row["correct"] not in ("yes", "no"):
            continue
        tile = f"{row['m_tile']}x{row['n_tile']}"
        sg   = int(row["simdgroups"])
        n    = int(row["mat_n"])
        gf   = float(row["gflops"])
        if tile in TILES and n in N_VALUES and 1 <= sg <= MAX_SG:
            data[n][tile][sg - 1] = gf

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("MPP matmul GFLOPS: tile shape × simdgroups (M3 Pro, f16)", fontsize=14, fontweight="bold")

# Shared colour scale across all subplots
all_vals = [v for n in N_VALUES for t in TILES for v in data[n][t] if not np.isnan(v)]
vmin, vmax = min(all_vals), max(all_vals)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "plasma"

for idx, n in enumerate(N_VALUES):
    ax = axes[idx // 2][idx % 2]

    mat = np.array([data[n][t] for t in TILES])  # shape: (7, 32)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    ax.set_title(f"N = {n}", fontsize=12)
    ax.set_xlabel("simdgroups")
    ax.set_ylabel("tile shape (M×N)")

    ax.set_xticks(range(MAX_SG))
    ax.set_xticklabels(range(1, MAX_SG + 1), fontsize=7, rotation=45)
    ax.set_yticks(range(len(TILES)))
    ax.set_yticklabels(TILES, fontsize=9)

    # Annotate best cell per subplot
    flat_idx = np.nanargmax(mat)
    r, c = divmod(flat_idx, MAX_SG)
    best = mat[r, c]
    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                fill=False, edgecolor="cyan", linewidth=2))
    ax.text(c, r, f"{best:.0f}", ha="center", va="center",
            fontsize=7, color="black", fontweight="bold")

# Shared colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes, fraction=0.03, pad=0.04)
cbar.set_label("GFLOPS", fontsize=11)

plt.savefig(HERE / "results.png", dpi=150, bbox_inches="tight")
print("Saved results.png")
plt.show()
