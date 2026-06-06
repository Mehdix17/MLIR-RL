import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
df = pd.read_csv("plots/full_model_comparison/graph3_fullmodel.csv")

# Sort models alphabetically (as shown in the figure)
df = df.sort_values("model").reset_index(drop=True)

models = df["model"].tolist()
prior_speedup = df["V0_total_speedup"].tolist()     # red bars (Prior Agent)
our_speedup   = df["No-Reward_total_speedup"].tolist()  # green bars (Our Agent)

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(32, 12))

# Draw bars; skip NaN for prior agent
for i, (p, o) in enumerate(zip(prior_speedup, our_speedup)):
    if not pd.isna(p):
        ax.bar(x[i] - width / 2, p, width, color="#d62728", zorder=3)
    if not pd.isna(o):
        ax.bar(x[i] + width / 2, o, width, color="#2ca02c", zorder=3)

# Baseline dashed line at y=1
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2)

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=20)
ax.set_xlabel('Models', fontsize=18, fontweight='bold')
ax.set_ylabel("Total Model Speedup", fontsize=18, fontweight='bold')
ax.set_title("Full Model Speedup (Prior Agent vs Our Agent)", fontsize=24, fontweight='bold')
ax.set_xlim(-0.6, len(models) - 0.4)
ax.set_ylim(0, None)

# Grid (y only, behind bars)
ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
baseline_line = plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0,
                            label="MLIR Baseline (1.0x)")
prior_patch   = mpatches.Patch(color="#d62728", label="Prior Agent")
our_patch     = mpatches.Patch(color="#2ca02c", label="Our Agent")
ax.legend(handles=[baseline_line, prior_patch, our_patch], fontsize=13, loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig("plots/full_model_comparison/graph3_fullmodel.png", dpi=150, bbox_inches="tight")
print("Saved: plots/full_model_comparison/graph3_fullmodel.png")