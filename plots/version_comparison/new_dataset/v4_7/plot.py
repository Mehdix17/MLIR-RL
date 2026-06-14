#!/usr/bin/env python3
"""Plot V4.7 geometric mean speedup evolution across checkpoints."""

import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "v4_7.csv")
PNG_PATH = os.path.join(SCRIPT_DIR, "v4_7.png")

df = pd.read_csv(CSV_PATH)
df = df.sort_values("checkpoint")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["checkpoint"], df["average_speedup"], marker="o", markersize=4, linewidth=1.5, color="#16a34a")
ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="MLIR Baseline (1.0x)")

ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")
ax.set_ylabel("Geometric Mean Speedup", fontsize=14, fontweight="bold")
ax.set_title("V4.7 - Geometric Mean Speedup Across Checkpoints", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")
ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {PNG_PATH}")
