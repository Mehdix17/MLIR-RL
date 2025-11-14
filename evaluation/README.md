# Evaluation System for MLIR-RL

This module provides comprehensive evaluation tools for the RL agent's optimization performance.

## Overview

The evaluation system measures how well the RL agent optimizes MLIR programs compared to:
1. **Baseline** (no optimization)
2. **PyTorch** (native performance)
3. **Standard passes** (LLVM default optimizations)

## Components

### 1. Single Operation Evaluator (`single_op_eval.py`)

Evaluates agent on individual operations.

**Usage:**
```python
from evaluation import SingleOperationEvaluator

evaluator = SingleOperationEvaluator(
    agent_model_path="results/best_model.pt",
    benchmark_dir="benchmarks/single_ops"
)

# Evaluate single file
result = evaluator.evaluate_operation("benchmarks/single_ops/matmul_512_0001.mlir")
print(f"Speedup: {result['speedup']:.2f}x")

# Evaluate entire benchmark suite
results = evaluator.evaluate_benchmark_suite(
    output_file="results/evaluation_results.json"
)
```

**Metrics:**
- Execution time (baseline vs optimized)
- Speedup factor
- Improvement percentage

### 2. Neural Network Evaluator (`nn_eval.py`)

Evaluates agent on complete neural networks.

**Usage:**
```python
from evaluation import NeuralNetworkEvaluator

evaluator = NeuralNetworkEvaluator(
    agent_model_path="results/best_model.pt",
    benchmark_dir="benchmarks/neural_nets"
)

# Evaluate single network
result = evaluator.evaluate_neural_network("benchmarks/neural_nets/resnet18.mlir")
print(f"Speedup: {result['speedup_vs_baseline']:.2f}x")

# Evaluate all networks
results = evaluator.evaluate_benchmark_suite(
    output_file="results/nn_evaluation_results.json"
)
```

**Metrics:**
- Inference time (baseline, optimized, PyTorch)
- Speedup vs baseline
- Speedup vs PyTorch
- Memory usage

### 3. PyTorch Baseline (`pytorch_baseline.py`)

Measures PyTorch's native performance for comparison.

**Usage:**
```python
from evaluation import PyTorchBaseline

# CPU baseline
baseline = PyTorchBaseline(device="cpu")

# Single operation
result = baseline.benchmark_matmul(512, 512, 512)
print(f"Time: {result['median_time']*1000:.2f}ms")

# Full benchmark suite
results = baseline.run_benchmark_suite(
    output_file="results/pytorch_baseline.json"
)
```

**Benchmarks:**
- Matrix multiplication (various sizes)
- Convolutions (2D/3D)
- Pooling operations
- Complete neural networks

## Quick Start

### 1. Evaluate Single Operations

```bash
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
```

### 2. Evaluate Neural Networks

```bash
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

### 3. Run PyTorch Baseline

```bash
python evaluation/pytorch_baseline.py
```

## Installation

### Requirements
```bash
pip install torch numpy
```

### MLIR Setup
Ensure `mlir-opt` is available:
```bash
export PATH=/path/to/llvm-project/build/bin:$PATH
```

Or specify path in code:
```python
evaluator = SingleOperationEvaluator(
    agent_model_path="results/best_model.pt",
    mlir_opt_path="/path/to/mlir-opt"
)
```

## Evaluation Workflow

### Complete Evaluation Pipeline

```python
from evaluation import (
    SingleOperationEvaluator,
    NeuralNetworkEvaluator,
    PyTorchBaseline
)
from pathlib import Path

# 1. Run PyTorch baseline
print("Running PyTorch baseline...")
baseline = PyTorchBaseline(device="cpu")
pytorch_results = baseline.run_benchmark_suite(
    output_file=Path("results/pytorch_baseline.json")
)

# 2. Evaluate single operations
print("\nEvaluating single operations...")
single_eval = SingleOperationEvaluator(
    agent_model_path=Path("results/best_model.pt"),
    benchmark_dir=Path("benchmarks/single_ops")
)
single_results = single_eval.evaluate_benchmark_suite(
    output_file=Path("results/single_op_eval.json")
)

# 3. Evaluate neural networks
print("\nEvaluating neural networks...")
nn_eval = NeuralNetworkEvaluator(
    agent_model_path=Path("results/best_model.pt"),
    benchmark_dir=Path("benchmarks/neural_nets")
)
nn_results = nn_eval.evaluate_benchmark_suite(
    output_file=Path("results/nn_eval.json")
)

# 4. Print summary
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Single ops mean speedup: {single_results['mean_speedup']:.2f}x")
print(f"Neural nets mean speedup: {nn_results['mean_speedup']:.2f}x")
print("="*60)
```

## Results Format

### Single Operation Results
```json
{
  "num_benchmarks": 100,
  "mean_speedup": 1.45,
  "median_speedup": 1.38,
  "max_speedup": 2.87,
  "min_speedup": 0.95,
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

### Neural Network Results
```json
{
  "num_networks": 5,
  "mean_speedup": 1.62,
  "median_speedup": 1.55,
  "networks": [
    {
      "network": "resnet18",
      "baseline_time": 0.245,
      "agent_time": 0.151,
      "pytorch_time": 0.189,
      "speedup_vs_baseline": 1.62,
      "speedup_vs_pytorch": 1.25,
      "improvement_pct": 38.4
    }
  ]
}
```

### PyTorch Baseline Results
```json
{
  "matmul": [
    {
      "operation": "matmul",
      "shape": "512x512 @ 512x512",
      "mean_time": 0.0142,
      "median_time": 0.0138,
      "std_time": 0.0012
    }
  ],
  "conv2d": [...],
  "pooling": [...]
}
```

## Customization

### Add Custom Evaluation Metrics

Extend the evaluators:

```python
from evaluation import SingleOperationEvaluator

class CustomEvaluator(SingleOperationEvaluator):
    def evaluate_operation(self, mlir_file, num_runs=10):
        result = super().evaluate_operation(mlir_file, num_runs)
        
        # Add custom metrics
        result['memory_usage'] = self._measure_memory(mlir_file)
        result['compilation_time'] = self._measure_compilation(mlir_file)
        
        return result
    
    def _measure_memory(self, mlir_file):
        # Custom memory measurement
        return 0.0
    
    def _measure_compilation(self, mlir_file):
        # Custom compilation time measurement
        return 0.0
```

### Custom Optimization Passes

Override `_apply_agent_optimization`:

```python
def _apply_agent_optimization(self, mlir_file):
    # Use your agent to select passes
    passes = self.agent.select_optimization_passes(mlir_file)
    
    cmd = [str(self.mlir_opt_path)] + passes + [str(mlir_file)]
    subprocess.run(cmd, check=True)
    
    return optimized_file
```

## Benchmarking Best Practices

### 1. Warmup Runs
Always include warmup runs to avoid cold-start effects:
```python
result = evaluator.evaluate_operation(mlir_file, num_runs=10)
# First 2-3 runs are typically slower
```

### 2. Multiple Runs
Use median time (more robust than mean):
```python
times = []
for _ in range(num_runs):
    times.append(measure_time())
median_time = np.median(times)
```

### 3. Isolate Execution
Minimize background processes during evaluation:
```bash
# Set CPU affinity
taskset -c 0-3 python evaluation/single_op_eval.py
```

### 4. GPU Synchronization
For GPU benchmarks, always synchronize:
```python
if device.type == 'cuda':
    torch.cuda.synchronize()
```

## Troubleshooting

### Issue: mlir-opt not found
**Solution:** Specify path explicitly:
```python
evaluator = SingleOperationEvaluator(
    agent_model_path="results/best_model.pt",
    mlir_opt_path="/path/to/llvm-project/build/bin/mlir-opt"
)
```

### Issue: Compilation failures
**Solution:** Check MLIR dialect compatibility:
```bash
mlir-opt --verify-diagnostics your_file.mlir
```

### Issue: Timeout during execution
**Solution:** Increase timeout or skip problematic files:
```python
try:
    result = subprocess.run(cmd, timeout=60)
except subprocess.TimeoutExpired:
    print(f"Skipping {mlir_file} (timeout)")
```

### Issue: Inconsistent timings
**Solution:** 
- Run more iterations
- Use CPU pinning
- Disable frequency scaling:
```bash
sudo cpupower frequency-set --governor performance
```

## Integration with Training

Monitor evaluation during training:

```python
# In training loop
if epoch % 10 == 0:
    evaluator = SingleOperationEvaluator(
        agent_model_path=f"results/checkpoint_epoch_{epoch}.pt"
    )
    results = evaluator.evaluate_benchmark_suite()
    
    # Log to Neptune/TensorBoard
    logger.log_metric("mean_speedup", results['mean_speedup'])
```

## Visualization

Generate plots from evaluation results:

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('results/eval_results.json') as f:
    data = json.load(f)

# Plot speedup distribution
speedups = [r['speedup'] for r in data['individual_results']]
plt.hist(speedups, bins=20)
plt.xlabel('Speedup')
plt.ylabel('Count')
plt.title('RL Agent Speedup Distribution')
plt.savefig('results/speedup_distribution.png')
```

## Next Steps

1. **Run baseline**: Establish PyTorch performance baseline
2. **Train agent**: Train RL agent on generated data
3. **Evaluate**: Use evaluation system to measure improvements
4. **Iterate**: Adjust agent based on evaluation results

## Related Documentation

- [Data Generation](../data_generation/README.md)
- [Benchmarks](../benchmarks/README.md)
- [Training Guide](../docs/guides/SLURM_GUIDE.md)
- [Plotting Results](../docs/guides/PLOTTING_README.md)
