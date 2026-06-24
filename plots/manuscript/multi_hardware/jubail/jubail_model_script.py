import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths
INPUT_CSV  = "plots/multi_hardware/jubail/graph2_jubail_model.csv"
OUTPUT_PNG = "plots/multi_hardware/jubail/graph2_jubail_model.png"


# Load data
df = pd.read_csv(INPUT_CSV)

hw    = df[df["agent"] == "No-Reward (HW)"].set_index("group")["geo_mean"]
no_hw = df[df["agent"] == "No-Reward (No-HW)"].set_index("group")["geo_mean"]

# Sorted model list (alphabetical, as in the figure)
models = sorted(df["group"].unique())

hw_vals    = [hw.get(m, float("nan"))    for m in models]
no_hw_vals = [no_hw.get(m, float("nan")) for m in models]

x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(18, 6))

ax.bar(x - width / 2, hw_vals,    width, color="#2ca02c", label="No-Reward (HW)",    zorder=3)
ax.bar(x + width / 2, no_hw_vals, width, color="#ff7f0e", label="No-Reward (No-HW)", zorder=3)

# Log scale y-axis
# ax.set_yscale("log")

# Baseline at 1.0x
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2)

# Y-axis tick labels: 1.0x and 10.0x (matching the figure)
ax.set_yticks([1.0, 10.0])
ax.set_yticklabels(["1.0x", "10.0x"], fontsize=15)

# X-axis
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=15)

# Labels and title
ax.set_xlabel("Models", fontsize=18, fontweight='bold')
ax.set_ylabel("Geometric Mean Speedup", fontsize=18, fontweight='bold')
ax.set_title("Multi-Hardware: Jubail \u2014 by Model", fontsize=24, fontweight='bold')

ax.set_xlim(-0.6, len(models) - 0.4)

# Grid (y only)
ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="Baseline 1.0x")
hw_patch    = mpatches.Patch(color="#2ca02c", label="Our Agent (with HW-F)")
no_hw_patch = mpatches.Patch(color="#ff7f0e", label="Our Agent (without HW-F)")
ax.legend(handles=[baseline_line, hw_patch, no_hw_patch], fontsize=13, loc="upper right", frameon=True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")