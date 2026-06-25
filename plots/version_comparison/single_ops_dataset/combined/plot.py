#!/usr/bin/env python3
"""Plot all 3 agent eval evolution curves on one figure."""

import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

AGENTS = [
    ("V0 (LSTM)", os.path.join(BASE_DIR, "v0", "v0.csv"), "#2563eb"),
    ("V4.9-Small (Transformer)", os.path.join(BASE_DIR, "v4_9_small", "v4_9_small.csv"), "#16a34a"),
    ("V4.9-Large (Transformer)", os.path.join(BASE_DIR, "v4_9_large", "v4_9_large.csv"), "#dc2626"),
]

PNG_PATH = os.path.join(SCRIPT_DIR, "combined.png")

fig, ax = plt.subplots(figsize=(14, 7))

for label, csv_path, color in AGENTS:
    df = pd.read_csv(csv_path)
    df = df.sort_values("checkpoint")
    ax.plot(df["checkpoint"], df["average_speedup"], marker="o", markersize=3, linewidth=1.5,
            color=color, label=label)

ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="MLIR Baseline (1.0x)")

ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")
ax.set_ylabel("Geometric Mean Speedup", fontsize=14, fontweight="bold")
ax.set_title("Single Ops: Eval Evolution \u2014 V0 vs V4.9-Small vs V4.9-Large", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")
ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {PNG_PATH}")
