# Benchmarking Guide

## Structure

Each trained model has a standardized benchmarking directory:

```
results/
├── lstm/
│   └── run_1/
│       ├── models/          # Trained model checkpoints
│       ├── logs/            # Training logs
│       └── benchmarks/      # Benchmark outputs (NEW)
│           ├── agent_output.json          # RL agent optimized execution
│           ├── pytorch_output.json        # PyTorch default execution
│           ├── pytorch_jit_output.json    # PyTorch JIT execution
│           ├── comparison_results.json    # Combined results
│           ├── comparison_summary.csv     # Summary table
│           ├── comparison_bar_plot.png    # Execution time comparison
│           └── speedup_comparison.png     # Speedup chart
├── distilbert/
│   └── run_0/
│       └── benchmarks/
│           ├── agent_output.json
│           ├── pytorch_output.json
│           └── pytorch_jit_output.json
├── gpt-2/
│   └── run_0/
│       └── benchmarks/
│           └── ...
└── convnext/
    └── run_0/
        └── benchmarks/
            └── ...
```

## Workflow

### 1. Train a Model (Optional - use existing model)

```bash
# Train LSTM baseline
sbatch scripts/lstm/train_lstm_baseline.sh

# Or train DistilBERT
sbatch scripts/distilbert/train_distilbert.sh
```

### 2. Generate Benchmark Outputs (3 separate steps)

```bash
# Step 1: Run RL Agent optimization
python evaluation/run_rl_optimized.py
# → Saves to: results/{model}/run_X/benchmarks/agent_output.json

# Step 2: Run PyTorch Default
python evaluation/run_pytorch_default.py
# → Saves to: results/{model}/run_X/benchmarks/pytorch_output.json

# Step 3: Run PyTorch JIT
python evaluation/run_pytorch_jit.py
# → Saves to: results/{model}/run_X/benchmarks/pytorch_jit_output.json
```

### 3. Compare Results

```bash
# Compare the 3 outputs and generate plots
python evaluation/compare_all.py
# → Generates comparison plots and summary in benchmarks/ directory
```

### All-in-One Script

```bash
# Run all steps automatically
scripts/submit_and_monitor.sh compare_all.sh
```

## Output Files

### Benchmark Outputs (Inputs to comparison)
- **agent_output.json**: RL agent optimized execution times
- **pytorch_output.json**: PyTorch default (eager mode) execution times  
- **pytorch_jit_output.json**: PyTorch JIT compiled execution times

### Comparison Results (Generated from inputs)
- **comparison_results.json**: Combined results from all 3 methods
- **comparison_summary.csv**: Summary table with speedups
- **comparison_bar_plot.png**: Execution time comparison chart
- **speedup_comparison.png**: Speedup relative to PyTorch Default

## Viewing Results

```bash
# View summary table
cat results/lstm/run_1/benchmarks/comparison_summary.csv

# View plots (copy to local machine or use image viewer)
# comparison_bar_plot.png - Shows execution times
# speedup_comparison.png - Shows speedup relative to PyTorch Default
```

## Key Features

✅ **Model Agnostic**: Works with any model (LSTM, DistilBERT, GPT-2, ConvNeXt, etc.)
✅ **Automatic Detection**: Finds the latest trained model automatically
✅ **Standardized Names**: All models use same filename convention
✅ **Separation of Concerns**: Execution separate from comparison
✅ **Reproducible**: Benchmark outputs are saved and can be re-compared anytime

## Benchmarks Tested

- **matmul_small**: 256×256 matrix multiplication
- **matmul_medium**: 512×512 matrix multiplication
- **matmul_large**: 1024×1024 matrix multiplication
- **conv2d_small**: 3x3 convolution, 32 channels
- **conv2d_medium**: 3x3 convolution, 64 channels
- **conv2d_large**: 3x3 convolution, 128 channels
- **resnet_block**: ResNet basic block with 64 channels
- **linear_small**: Linear layer 128→64
- **linear_large**: Linear layer 1024→512

Each benchmark is run 100 times for statistical reliability.
