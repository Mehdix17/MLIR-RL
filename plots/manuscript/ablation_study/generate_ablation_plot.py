import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/scratch/mb10856/MLIR-RL/plots/ablation_study/comparison.csv')

# Remove single_bench (legacy)
df = df[df['model'] != 'single_bench (legacy)']

# Sort models alphabetically
df = df.sort_values('model').reset_index(drop=True)

# Extract the data
models = df['model'].values
v45_speedup = df['v45_speedup'].values
ntr_speedup = df['ntr_speedup'].values
nhw_speedup = df['nhw_speedup'].values
nrw_speedup = df['nrw_speedup'].values

# Set up the bar plot
fig, ax = plt.subplots(figsize=(16, 8))

# Bar positions
x = np.arange(len(models))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, v45_speedup, width, label='Our Agent', color='red', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, ntr_speedup, width, label='No-Transformer Agent', color='green', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, nhw_speedup, width, label='No-HW-Features Agent', color='#359CDB', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, nrw_speedup, width, label='No-Reward-Shaping Agent', color='orange', alpha=0.8)

# Customize the plot
ax.set_xlabel('Models', fontsize=18, fontweight='bold')
ax.set_ylabel('Geometric Mean Speedup', fontsize=18, fontweight='bold')
ax.set_title('Ablation Study - Per-Model Speedup Comparison', fontsize=22, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(fontsize=13, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal line at y=1 (baseline)
ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_path = '/scratch/mb10856/MLIR-RL/plots/ablation_study/ablation_study.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()
