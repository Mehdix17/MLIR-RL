# Enhanced Plotting and Neptune Integration

This document explains the new plotting and visualization system for MLIR-RL results.

## 🎨 What's New

### 1. **Comprehensive Plot Generation** (`plot_results.py`)
Generates comparison plots showing:
- **Speedup by operation type** (add, matmul, conv2d, pooling, relu)
- **Geometric mean speedup** for each operation type
- **Per-benchmark speedup** (horizontal bar charts)
- **Training metrics** (value loss, policy loss, entropy, cumulative reward)

### 2. **Enhanced Neptune Sync** (`sync_neptune_with_plots.py`)
- Automatically generates plots before syncing
- Uploads all metrics + plots to Neptune dashboard
- Single command to sync everything

## 📦 Installation

Matplotlib is already installed in the `mlir` conda environment. If you need to reinstall it:

```bash
conda activate mlir
pip install matplotlib
```

## 🚀 Usage

### Option 1: Generate Plots Locally

```bash
# Generate plots for a specific run
python analysis/plot_results.py results/run_9

# Plots will be saved to: results/run_9/plots/
```

This creates:
- `speedup_by_op_type.png` - Bar chart of mean speedup per operation type
- `geometric_mean_speedup.png` - Geometric mean speedup comparison
- `per_benchmark_speedup.png` - Detailed per-benchmark results
- `training_metrics.png` - Training curves (4 subplots)

### Option 2: Sync to Neptune with Plots

```bash
# Generate plots AND sync to Neptune in one command
python experiments/sync_neptune_with_plots.py results/run_9
```

This will:
1. Generate all plots (if matplotlib is available)
2. Upload metrics to Neptune
3. Upload plots to Neptune
4. Upload config and metadata

### Option 3: Continuous Neptune Sync (Original)

```bash
# For continuous monitoring during training
bash scripts/neptune-sync.sh
```

## 📊 Generated Plots

### 1. Speedup by Operation Type
Shows mean speedup (with error bars) for each operation type:
- **add**: Element-wise addition operations
- **matmul**: Matrix multiplication operations  
- **conv2d**: 2D convolution operations
- **pooling**: Pooling operations
- **relu**: ReLU activation operations

### 2. Geometric Mean Speedup
More robust metric for comparing speedups across different operation types.
Geometric mean is less affected by outliers than arithmetic mean.

### 3. Per-Benchmark Speedup
Horizontal bar chart showing speedup for each individual benchmark,
grouped by operation type. Helps identify which specific benchmarks
benefit most from the RL optimization.

### 4. Training Metrics
Four subplots showing:
- **(a) Value Loss**: How well the value network predicts returns
- **(b) Policy Loss**: PPO policy optimization loss
- **(c) Entropy**: Exploration metric (higher = more exploration)
- **(d) Cumulative Reward**: Total reward per evaluation

## 📈 Neptune Dashboard

After syncing, view your results at:
- **URL**: https://app.neptune.ai/mehdix/mlir-project

You'll see:
- All training/evaluation metrics as interactive plots
- Uploaded static plots in the "plots" section
- Configuration parameters
- Tags for filtering runs

## 🔧 Workflow Examples

### Example 1: Quick Test Run

```bash
# 1. Train on test dataset (3 iterations)
bash scripts/train.sh

# 2. Generate plots
python analysis/plot_results.py results/run_10

# 3. View plots locally
open results/run_10/plots/speedup_by_op_type.png
```

### Example 2: Full Training with Neptune

```bash
# Terminal 1: Start training
bash scripts/train.sh

# Terminal 2: Start continuous Neptune sync
bash scripts/neptune-sync.sh

# After training completes:
# Generate and upload plots
python sync_neptune_with_plots.py results/run_10
```

### Example 3: Comparing Multiple Runs

```bash
# Generate plots for multiple runs
for run in results/run_{9..12}; do
    python analysis/plot_results.py $run
done

# Sync all to Neptune
for run in results/run_{9..12}; do
    python experiments/sync_neptune_with_plots.py $run
done

# Now compare runs in Neptune dashboard by filtering tags
```

## 📋 Statistics Printed

When you run `plot_results.py`, it prints detailed statistics:

```
======================================================================
SPEEDUP STATISTICS BY OPERATION TYPE
======================================================================

ADD:
  Benchmarks: 2
  Mean speedup: 1.842x ± 0.524
  Geometric mean speedup: 1.807x
  Min speedup: 1.471x
  Max speedup: 2.064x

CONV2D:
  Benchmarks: 1
  Mean speedup: 3.264x ± 0.000
  Geometric mean speedup: 3.264x
  ...
```

## 🎯 Comparison with Baseline

The plots automatically compare against baseline (1x speedup = no improvement).
- Baseline execution times are loaded from your JSON files
- Red dashed line at y=1.0 indicates baseline performance
- Bars/points above the line show improvement
- Bars/points below the line show regression

## 🔍 Detailed Metrics Logged

### Training Metrics
- `train/reward` - Rewards during trajectory collection
- `train/entropy` - Policy entropy (exploration)
- `train/final_speedup` - Speedup achieved on training benchmarks

### PPO Metrics
- `train_ppo/policy_loss` - PPO policy gradient loss
- `train_ppo/value_loss` - Value function loss
- `train_ppo/clip_frac` - Fraction of clipped policy updates
- `train_ppo/approx_kl` - Approximate KL divergence
- `train_ppo/entropy_loss` - Entropy regularization loss

### Evaluation Metrics
- `eval/reward` - Rewards on evaluation benchmarks
- `eval/cumulative_reward` - Total reward per benchmark
- `eval/final_speedup` - Final speedup per benchmark
- `eval/average_speedup` - Average across all eval benchmarks
- `eval/speedup/<benchmark_name>` - Speedup for specific benchmark
- `eval/exec_time/<benchmark_name>` - Execution time for specific benchmark

## 🐛 Troubleshooting

### Matplotlib not installed
```bash
conda activate mlir
pip install matplotlib
```

### Neptune API token error
Make sure `.env` file has:
```
NEPTUNE_PROJECT=your-workspace/your-project
NEPTUNE_TOKEN=your-api-token
```

### No plots generated
Check that your run has evaluation metrics:
```bash
ls results/run_9/logs/eval/speedup/
```

If empty, run evaluation manually or train longer.

## 📚 Files Created

- `analysis/plot_results.py` - Main plotting script
- `experiments/sync_neptune_with_plots.py` - Enhanced Neptune sync with plots
- `experiments/test_neptune.py` - Simple Neptune connection test
- `docs/PLOTTING_README.md` - This documentation

## 🎓 Next Steps

1. **Install matplotlib**: `conda activate mlir && pip install matplotlib`
2. **Generate test plots**: `python analysis/plot_results.py results/run_9`
3. **Sync to Neptune**: `python experiments/sync_neptune_with_plots.py results/run_9`
4. **View dashboard**: https://app.neptune.ai/mehdix/mlir-project
5. **Run full training**: Configure `nb_iterations` and run `bash scripts/train.sh`

Happy plotting! 📊🚀
