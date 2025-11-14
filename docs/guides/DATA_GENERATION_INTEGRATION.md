# Data Generation and Evaluation Integration Guide

This guide shows how to integrate the data generation and evaluation systems from the previous project into your MLIR-RL workflow.

## 🎯 Overview

The previous project (BouchamaDjad/MLIR-RL) provided two critical components:

1. **`data_utils/`** → Now integrated as **`data_generation/`**
   - Random MLIR code generation
   - Neural network to MLIR conversion
   - Benchmark creation

2. **`evaluation/`** → Now integrated as **`evaluation/`**
   - Single operation evaluation
   - Neural network evaluation
   - PyTorch baseline comparison

## 📁 New Project Structure

```
MLIR-RL/
├── data_generation/          # ✨ NEW: Generate training data
│   ├── __init__.py
│   ├── random_mlir_gen.py   # Generate random MLIR programs
│   ├── nn_to_mlir.py         # Convert PyTorch → MLIR
│   └── README.md
│
├── evaluation/               # ✨ NEW: Evaluate trained agents
│   ├── __init__.py
│   ├── single_op_eval.py    # Single operation evaluation
│   ├── nn_eval.py            # Neural network evaluation
│   ├── pytorch_baseline.py   # PyTorch baseline
│   └── README.md
│
├── benchmarks/               # ✨ NEW: Benchmark suites
│   ├── single_ops/          # Individual operations
│   ├── neural_nets/         # Complete neural networks
│   └── README.md
│
├── rl_autoschedular/         # Existing RL agent code
├── bin/                      # Existing training scripts
├── data/                     # Existing training data
└── results/                  # Existing results
```

## 🚀 Quick Start Workflow

### Step 1: Generate Training Data

```bash
# Generate 1000 random MLIR programs
python data_generation/random_mlir_gen.py
```

This creates:
- `data/generated/train/` (1000 training files)
- `data/generated/test/` (200 test files)

### Step 2: Convert Neural Networks to MLIR

```bash
# Convert ResNet-18 to MLIR
python data_generation/nn_to_mlir.py
```

This creates:
- `data/generated/neural_nets/resnet18.mlir`

### Step 3: Train RL Agent

```bash
# Train on generated data
CONFIG_FILE_PATH=config/config.json python bin/train.py
```

### Step 4: Evaluate Performance

```bash
# Evaluate on single operations
python -c "
from evaluation import SingleOperationEvaluator
from pathlib import Path

evaluator = SingleOperationEvaluator(
    agent_model_path=Path('results/best_model.pt'),
    benchmark_dir=Path('benchmarks/single_ops')
)

results = evaluator.evaluate_benchmark_suite(
    output_file=Path('results/eval_results.json')
)
"

# Evaluate on neural networks
python -c "
from evaluation import NeuralNetworkEvaluator
from pathlib import Path

evaluator = NeuralNetworkEvaluator(
    agent_model_path=Path('results/best_model.pt'),
    benchmark_dir=Path('benchmarks/neural_nets')
)

results = evaluator.evaluate_benchmark_suite(
    output_file=Path('results/nn_eval_results.json')
)
"
```

## 📝 Complete Integration Example

### Python Script: `run_full_pipeline.py`

```python
"""
Complete pipeline: Generate → Train → Evaluate
"""

from pathlib import Path
from data_generation import RandomMLIRGenerator, NeuralNetworkToMLIR
from evaluation import SingleOperationEvaluator, NeuralNetworkEvaluator, PyTorchBaseline
import subprocess


def step1_generate_data():
    """Generate training data"""
    print("="*60)
    print("STEP 1: Generating Training Data")
    print("="*60)
    
    # Generate random MLIR programs
    generator = RandomMLIRGenerator(seed=42)
    
    train_files = generator.generate_dataset(
        num_samples=1000,
        output_dir=Path("data/generated/train"),
        operation_types=['matmul', 'conv2d', 'pooling']
    )
    
    test_files = generator.generate_dataset(
        num_samples=200,
        output_dir=Path("data/generated/test"),
        operation_types=['matmul', 'conv2d', 'pooling']
    )
    
    print(f"\n✓ Generated {len(train_files)} training files")
    print(f"✓ Generated {len(test_files)} test files")


def step2_convert_neural_networks():
    """Convert neural networks to MLIR"""
    print("\n" + "="*60)
    print("STEP 2: Converting Neural Networks")
    print("="*60)
    
    try:
        from torchvision.models import resnet18
        
        converter = NeuralNetworkToMLIR(output_dir=Path("benchmarks/neural_nets"))
        
        # Convert ResNet-18
        model = resnet18(pretrained=False)
        model.eval()
        
        mlir_file = converter.convert_model(
            model=model,
            input_shape=(1, 3, 224, 224),
            model_name="resnet18",
            use_torch_mlir=False
        )
        
        print(f"\n✓ Converted ResNet-18 to {mlir_file}")
    except ImportError:
        print("⚠️  torchvision not installed, skipping neural network conversion")


def step3_train_agent():
    """Train RL agent"""
    print("\n" + "="*60)
    print("STEP 3: Training RL Agent")
    print("="*60)
    
    # Run training script
    cmd = ["python", "bin/train.py"]
    env = {"CONFIG_FILE_PATH": "config/config.json"}
    
    print("Training agent... (this may take a while)")
    print("Command:", " ".join(cmd))
    print("\nNote: Run this manually or uncomment the subprocess.run line below")
    # subprocess.run(cmd, env=env)


def step4_run_pytorch_baseline():
    """Run PyTorch baseline"""
    print("\n" + "="*60)
    print("STEP 4: Running PyTorch Baseline")
    print("="*60)
    
    baseline = PyTorchBaseline(device="cpu")
    results = baseline.run_benchmark_suite(
        output_file=Path("results/pytorch_baseline.json")
    )
    
    print("\n✓ PyTorch baseline complete")


def step5_evaluate_agent():
    """Evaluate trained agent"""
    print("\n" + "="*60)
    print("STEP 5: Evaluating Trained Agent")
    print("="*60)
    
    # Check if trained model exists
    model_path = Path("results/best_model.pt")
    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path}")
        print("Please train the agent first (Step 3)")
        return
    
    # Evaluate single operations
    print("\nEvaluating single operations...")
    single_eval = SingleOperationEvaluator(
        agent_model_path=model_path,
        benchmark_dir=Path("benchmarks/single_ops")
    )
    
    single_results = single_eval.evaluate_benchmark_suite(
        output_file=Path("results/single_op_eval.json")
    )
    
    # Evaluate neural networks
    print("\nEvaluating neural networks...")
    nn_eval = NeuralNetworkEvaluator(
        agent_model_path=model_path,
        benchmark_dir=Path("benchmarks/neural_nets")
    )
    
    nn_results = nn_eval.evaluate_benchmark_suite(
        output_file=Path("results/nn_eval.json")
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Single ops mean speedup: {single_results.get('mean_speedup', 0):.2f}x")
    print(f"Neural nets mean speedup: {nn_results.get('mean_speedup', 0):.2f}x")
    print("="*60)


def main():
    """Run full pipeline"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*10 + "MLIR-RL Complete Pipeline" + " "*23 + "║")
    print("╚" + "═"*58 + "╝")
    print()
    
    # Run all steps
    step1_generate_data()
    step2_convert_neural_networks()
    step3_train_agent()
    step4_run_pytorch_baseline()
    step5_evaluate_agent()
    
    print("\n" + "="*60)
    print("✓ Pipeline complete!")
    print("="*60)
    print("\nResults saved in:")
    print("  - results/pytorch_baseline.json")
    print("  - results/single_op_eval.json")
    print("  - results/nn_eval.json")


if __name__ == "__main__":
    main()
```

## 🔧 Configuration

### Update `config/config.json`

```json
{
  "model_type": "lstm",
  "data_dir": "data/generated/train",
  "test_dir": "data/generated/test",
  "benchmark_dir": "benchmarks",
  "results_dir": "results",
  
  "training": {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  
  "evaluation": {
    "num_runs": 10,
    "timeout": 60
  }
}
```

## 📊 Understanding the Results

### Single Operation Evaluation

```json
{
  "num_benchmarks": 100,
  "mean_speedup": 1.45,
  "median_speedup": 1.38,
  "individual_results": [
    {
      "file": "matmul_512_0001.mlir",
      "baseline_time": 0.0152,
      "agent_time": 0.0104,
      "speedup": 1.46,
      "improvement": 31.6
    }
  ]
}
```

**Key Metrics:**
- **speedup**: How much faster (e.g., 1.46x = 46% faster)
- **improvement**: Percentage improvement
- **mean/median**: Aggregate statistics

### Neural Network Evaluation

```json
{
  "networks": [
    {
      "network": "resnet18",
      "baseline_time": 0.245,
      "agent_time": 0.151,
      "pytorch_time": 0.189,
      "speedup_vs_baseline": 1.62,
      "speedup_vs_pytorch": 1.25
    }
  ]
}
```

**Key Metrics:**
- **speedup_vs_baseline**: Improvement over unoptimized MLIR
- **speedup_vs_pytorch**: Improvement over native PyTorch
- **agent_time**: Your RL agent's optimized time

## 🎨 Visualization

### Plot Evaluation Results

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('results/single_op_eval.json') as f:
    data = json.load(f)

# Extract speedups
speedups = [r['speedup'] for r in data['individual_results']]

# Plot
plt.figure(figsize=(10, 6))
plt.hist(speedups, bins=20, edgecolor='black')
plt.xlabel('Speedup', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('RL Agent Speedup Distribution', fontsize=14)
plt.axvline(x=1.0, color='r', linestyle='--', label='No improvement')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('results/speedup_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved to results/speedup_histogram.png")
```

## 🔄 Differences from Previous Project

| Previous Project | Current Integration | Notes |
|-----------------|-------------------|-------|
| `data_utils/` | `data_generation/` | Renamed for clarity |
| `evaluation/` | `evaluation/` | Same name, enhanced |
| Manual scripts | Python modules | Easier to import/use |
| Hardcoded paths | Configurable paths | More flexible |
| Limited docs | Comprehensive READMEs | Better documentation |

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Add to PYTHONPATH
export PYTHONPATH=/scratch/mb10856/MLIR-RL:$PYTHONPATH
```

### Issue: mlir-opt not found
```python
# Specify path explicitly
evaluator = SingleOperationEvaluator(
    agent_model_path="results/best_model.pt",
    mlir_opt_path="/path/to/llvm-project/build/bin/mlir-opt"
)
```

### Issue: Out of memory during generation
```python
# Generate in smaller batches
for i in range(10):
    generator.generate_dataset(
        num_samples=100,
        output_dir=f"data/train/batch_{i}"
    )
```

## 📚 Related Documentation

- [Data Generation README](../data_generation/README.md)
- [Evaluation README](../evaluation/README.md)
- [Benchmarks README](../benchmarks/README.md)
- [Training Guide](../docs/guides/SLURM_GUIDE.md)

## ✅ Next Steps

1. **Generate data**: Run `python data_generation/random_mlir_gen.py`
2. **Convert models**: Run `python data_generation/nn_to_mlir.py`
3. **Train agent**: Run `CONFIG_FILE_PATH=config/config.json python bin/train.py`
4. **Evaluate**: Run evaluation scripts
5. **Analyze**: Review results in `results/` directory
6. **Iterate**: Adjust configuration and retrain based on results

## 🎓 Understanding the Pipeline

```
┌─────────────────┐
│ Generate Data   │  ← data_generation/random_mlir_gen.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Convert Models  │  ← data_generation/nn_to_mlir.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train RL Agent  │  ← bin/train.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate Agent  │  ← evaluation/single_op_eval.py
└────────┬────────┘         evaluation/nn_eval.py
         │
         ▼
┌─────────────────┐
│ Analyze Results │  ← Plotting, metrics, comparison
└─────────────────┘
```

---

**Created**: November 14, 2025  
**Integration Status**: ✅ Complete  
**Previous Project**: [BouchamaDjad/MLIR-RL](https://github.com/BouchamaDjad/MLIR-RL/tree/Experimental)
