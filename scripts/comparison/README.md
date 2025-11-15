# Comparison Scripts

Scripts for comparing RL-optimized MLIR vs PyTorch Default vs PyTorch JIT.

---

## 📋 Overview

This directory contains scripts to benchmark and compare three optimization approaches:

1. **RL-Optimized**: MLIR code optimized by trained RL agent (LSTM/DistilBERT)
2. **PyTorch Default**: Standard PyTorch execution (no JIT, no optimizations)
3. **PyTorch JIT**: TorchScript-compiled PyTorch models

---

## 🚀 Quick Start

### **Complete Workflow (Train → Compare)**

```bash
# All-in-one: Train LSTM + Run full comparison
sbatch scripts/comparison/train_and_compare.sh
```

This script will:
1. Train LSTM baseline model (~1 hour)
2. Create benchmark suite
3. Run RL-optimized benchmarks
4. Run PyTorch Default benchmarks
5. Run PyTorch JIT benchmarks
6. Generate comparison plots and summary

**Total time**: ~2-3 hours

### **Run Complete Comparison** (if model already trained)

```bash
# Make sure you have a trained model first
sbatch scripts/lstm/train_lstm_baseline.sh

# Then run comparison
sbatch scripts/comparison/compare_all.sh
```

### **View Results**

```bash
# Summary table
cat results/comparison_rl_vs_pytorch/comparison_summary.csv

# View plots
ls -lh results/comparison_rl_vs_pytorch/*.png
```

---

## 📂 Components

### **0. Complete Workflow** (`scripts/comparison/train_and_compare.sh`)

All-in-one script that:
- Trains LSTM baseline model
- Creates benchmark suite
- Runs all three executors (RL, PyTorch Default, PyTorch JIT)
- Generates comparison

**Run with SLURM:**
```bash
sbatch scripts/comparison/train_and_compare.sh
```

### **1. Benchmark Suite** (`benchmarks/benchmark_suite.py`)

Creates standardized benchmarks:
- Matrix multiplication (small, medium, large)
- 2D Convolution (small, medium, large)
- ResNet block
- Linear layers

**Run manually:**
```bash
python benchmarks/benchmark_suite.py
```

### **2. RL-Optimized Executor** (`evaluation/run_rl_optimized.py`)

Runs benchmarks through trained RL agent.

**Requirements:**
- Trained RL model in `results/lstm/run_0/models/` or `results/distilbert/run_0/models/`

**Run manually:**
```bash
python evaluation/run_rl_optimized.py
```

### **3. PyTorch Default Executor** (`evaluation/run_pytorch_default.py`)

Runs benchmarks with standard PyTorch (no optimizations).

**Run manually:**
```bash
python evaluation/run_pytorch_default.py
```

### **4. PyTorch JIT Executor** (`evaluation/run_pytorch_jit.py`)

Runs benchmarks with JIT-compiled PyTorch models.

**Run manually:**
```bash
python evaluation/run_pytorch_jit.py
```

### **5. Comparison Tool** (`evaluation/compare_all.py`)

Generates comparison plots and summary tables.

**Run manually:**
```bash
python evaluation/compare_all.py
```

---

## 📊 Output

### **Results Directory**

```
results/comparison_rl_vs_pytorch/
├── comparison_results.json       # Raw data (all methods)
├── comparison_summary.csv        # Summary table
├── comparison_bar_plot.png       # Execution time comparison
└── speedup_comparison.png        # Speedup relative to PyTorch Default
```

### **Intermediate Results**

```
evaluation/results/
├── rl_optimized_results.json     # RL agent results
├── pytorch_default_results.json  # PyTorch default results
└── pytorch_jit_results.json      # PyTorch JIT results
```

---

## 🎯 Understanding Results

### **Metrics**

- **Mean Time (ms)**: Average execution time over 100 runs
- **Std Dev (ms)**: Standard deviation of execution times
- **Speedup**: Ratio of PyTorch Default time / Method time
  - `> 1.0`: Method is faster than PyTorch Default
  - `< 1.0`: Method is slower than PyTorch Default

### **Expected Results**

Typical speedup patterns:
- **RL-Optimized**: 1.2× - 3.0× (depends on training quality)
- **PyTorch JIT**: 1.1× - 2.0× (standard optimization)

---

## 🔧 Customization

### **Add New Benchmarks**

Edit `benchmarks/benchmark_suite.py`:

```python
class BenchmarkSuite:
    def __init__(self):
        self.benchmarks = {
            # ... existing benchmarks ...
            'my_custom_benchmark': MyCustomBenchmark(),
        }
```

### **Change Number of Runs**

In each executor (`run_*.py`), modify:

```python
def execute_model(self, ..., num_runs: int = 100):  # Change 100 to desired value
```

### **Use Different RL Model**

Edit `evaluation/run_rl_optimized.py`:

```python
# Change model path
model_path = Path("results/distilbert/run_0/models")  # Use DistilBERT
```

---

## ⚠️ Troubleshooting

### **No trained model found**

```bash
# Train LSTM model
sbatch scripts/lstm/train_lstm_baseline.sh

# Or train DistilBERT
sbatch scripts/distilbert/train_distilbert.sh
```

### **Benchmark suite not created**

```bash
python benchmarks/benchmark_suite.py
```

### **Missing results**

Check that all executors completed:
```bash
ls -lh evaluation/results/
# Should show: rl_optimized_results.json, pytorch_default_results.json, pytorch_jit_results.json
```

---

## 📚 Related Documentation

- **Training Guide**: `scripts/TRAINING_GUIDE.md`
- **Config Guide**: `config/README.md`
- **Results Guide**: `results/README.md`

---

**Last Updated**: November 15, 2025
