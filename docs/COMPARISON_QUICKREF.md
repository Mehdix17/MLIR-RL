# Quick Reference: Comparison Framework

## 🚀 **Quick Start**

### **All-in-One: Train + Compare** (Recommended!)
```bash
sbatch scripts/comparison/train_and_compare.sh
```
This does everything:
- ✅ Trains LSTM baseline (~1 hour)
- ✅ Creates benchmark suite
- ✅ Runs all 3 comparisons (RL, PyTorch Default, PyTorch JIT)
- ✅ Generates plots and summary

**Total time**: ~2-3 hours

### **Test the Framework** (5 minutes)
```bash
sbatch scripts/comparison/test_comparison.sh
```

### **Full Comparison** (if model already trained)
```bash
# 1. Train model first (if not done)
sbatch scripts/lstm/train_lstm_baseline.sh

# 2. Run comparison
sbatch scripts/comparison/compare_all.sh

# 3. View results
cat results/comparison_rl_vs_pytorch/comparison_summary.csv
```

---

## 📊 **What Gets Compared**

1. **RL-Optimized**: Your trained agent optimizing MLIR
2. **PyTorch Default**: Standard PyTorch (no optimizations)
3. **PyTorch JIT**: TorchScript compiled models

---

## 📂 **Results Location**

```
results/comparison_rl_vs_pytorch/
├── comparison_results.json       # Raw data
├── comparison_summary.csv        # Summary table
├── comparison_bar_plot.png       # Execution times
└── speedup_comparison.png        # Speedup chart
```

---

## 🎯 **Understanding Speedup**

- **Speedup > 1.0**: Method is **faster** than PyTorch Default ✅
- **Speedup < 1.0**: Method is **slower** than PyTorch Default ❌
- **Speedup = 1.0**: Same performance as PyTorch Default ➖

### **Example**
```
Benchmark: matmul_large
  RL-Optimized: 10.5 ms → Speedup: 2.0× (2× faster!)
  PyTorch Default: 20.0 ms → Baseline
  PyTorch JIT: 15.0 ms → Speedup: 1.33×
```

---

## 🔧 **Manual Steps**

### **1. Create Benchmarks**
```bash
python benchmarks/benchmark_suite.py
```

### **2. Run Individual Executors**
```bash
python evaluation/run_rl_optimized.py     # RL agent
python evaluation/run_pytorch_default.py  # PyTorch default
python evaluation/run_pytorch_jit.py      # PyTorch JIT
```

### **3. Generate Comparison**
```bash
python evaluation/compare_all.py
```

---

## 📈 **View Results**

### **Summary Table**
```bash
cat results/comparison_rl_vs_pytorch/comparison_summary.csv
```

### **JSON Data**
```bash
cat results/comparison_rl_vs_pytorch/comparison_results.json | jq
```

### **Plots**
```bash
ls -lh results/comparison_rl_vs_pytorch/*.png
```

---

## ⚙️ **Customization**

### **Add Custom Benchmarks**

Edit `benchmarks/benchmark_suite.py`:
```python
class BenchmarkSuite:
    def __init__(self):
        self.benchmarks = {
            'my_benchmark': MyBenchmark(),
        }
```

### **Change Model Path**

Edit `evaluation/run_rl_optimized.py`:
```python
model_path = Path("results/distilbert/run_0/models")
```

### **Adjust Run Count**

In executor files:
```python
def execute_model(self, ..., num_runs: int = 100):  # Change this
```

---

## 🎓 **Typical Workflow**

```bash
# 1. Quick test (verify setup)
sbatch scripts/lstm/test_lstm.sh

# 2. Full training
sbatch scripts/lstm/train_lstm_baseline.sh

# 3. Run comparison
sbatch scripts/comparison/compare_all.sh

# 4. Analyze results
cat results/comparison_rl_vs_pytorch/comparison_summary.csv
```

---

## ⚠️ **Troubleshooting**

### **No trained model**
```bash
sbatch scripts/lstm/train_lstm_baseline.sh
```

### **Missing pandas/matplotlib**
```bash
pip install pandas matplotlib
```

### **No benchmarks created**
```bash
python benchmarks/benchmark_suite.py
```

---

## 📚 **Full Documentation**

See: `scripts/comparison/README.md`

---

**Last Updated**: November 15, 2025
