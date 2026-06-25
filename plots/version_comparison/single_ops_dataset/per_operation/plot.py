#!/usr/bin/env python3
"""Plot grouped bar chart of geometric mean speedup by operation type."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(os.path.dirname(SCRIPT_DIR), "per_operation.csv")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "per_operation.png")

df = pd.read_csv(INPUT_CSV)

COLORS = {"V0": "#2563eb", "V4.9-S": "#16a34a", "V4.9-L": "#dc2626"}
LABELS = {"V0": "V0 (LSTM)", "V4.9-S": "V4.9-Small", "V4.9-L": "V4.9-Large"}

groups = sorted(df["group"].unique())
agents = sorted(df["agent"].unique())

fig, ax = plt.subplots(figsize=(18, 6))

width = 0.25
x = np.arange(len(groups))

for i, agent in enumerate(agents):
    agent_df = df[df["agent"] == agent].set_index("group")
    vals = [agent_df.loc[g, "geo_mean"] if g in agent_df.index else float("nan") for g in groups]
    ax.bar(x + (i - 1) * width, vals, width, color=COLORS[agent], label=LABELS[agent], zorder=3)

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2, label="Baseline 1.0x")

name_mapping = {
    "add": "Add", "batch_matmul": "Batch Matmul", "conv2d": "Conv2D",
    "matmul": "Matmul", "mul": "Mul", "pooling": "Pooling",
    "reduce_sum": "Reduce Sum", "relu": "ReLU", "sub": "Sub",
}
display_groups = [name_mapping.get(g, g) for g in groups]

ax.set_xticks(x)
ax.set_xticklabels(display_groups, rotation=0, ha="center", fontsize=14)

ax.set_xlabel("Operation Type", fontsize=16, fontweight="bold")
ax.set_ylabel("Geometric Mean Speedup", fontsize=16, fontweight="bold")
ax.set_title("Single Ops: Speedup by Operation Type (Best Checkpoint)", fontsize=18, fontweight="bold")

ax.set_xlim(-0.6, len(groups) - 0.4)

ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="Baseline 1.0x")
patches = [mpatches.Patch(color=COLORS[a], label=LABELS[a]) for a in agents]
ax.legend(handles=[baseline_line] + patches, fontsize=12, loc="upper right", frameon=True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
