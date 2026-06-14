import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths
INPUT_CSV  = "plots/overall_comparison/tayeb/graph1_performance.csv"
OUTPUT_PNG = "plots/overall_comparison/tayeb/graph1_performance.png"

# Load data
df = pd.read_csv(INPUT_CSV)

groups = list(df["op_type"])
v0_vals = list(df["V0_geo_mean"])
nw_vals = list(df["NoReward_geo_mean"])

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(18, 6))

ax.bar(x - width / 2, v0_vals, width, color="#D5373D", label="Prior Agent", zorder=3)
ax.bar(x + width / 2, nw_vals, width, color="#2EA42E", label="Our Agent",   zorder=3)

# Baseline at 1.0x
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2)

# Y-axis
ax.set_yticks([1.0, 2.0, 3.0, 6.0, 10.0, 11.0])
ax.set_yticklabels(["1.0x", "2.0x", "3.0x", "6.0x", "10.0x", "11.0x"], fontsize=13)

# X-axis
ax.set_xticks(x)

# Mapping names to be more readable
name_mapping = {
    "conv2d": "Conv2D",
    "generic": "ReLU",
    "matmul": "MatMul",
    "pooling": "Pooling"
}
display_groups = [name_mapping.get(g, g) for g in groups]
ax.set_xticklabels(display_groups, rotation=0, ha="center", fontsize=15)

# Labels and title
ax.set_xlabel("Operation Type", fontsize=18, fontweight='bold')
ax.set_ylabel("Geometric Mean Speedup", fontsize=18, fontweight='bold')
ax.set_title("Single-Operation Performance: Prior Agent vs Our Agent", fontsize=22, fontweight='bold')

ax.set_xlim(-0.6, len(groups) - 0.4)
ax.set_ylim(0, 12)

# Grid (y only)
ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="Baseline 1.0x")
v0_patch = mpatches.Patch(color="#D5373D", label="Prior Agent")
nw_patch = mpatches.Patch(color="#2EA42E", label="Our Agent")
ax.legend(handles=[baseline_line, v0_patch, nw_patch], fontsize=13, loc="upper right", frameon=True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
