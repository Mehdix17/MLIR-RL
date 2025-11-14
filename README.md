# MLIR-RL: Reinforcement Learning for MLIR Compiler Optimization

Automated MLIR compiler optimization using Proximal Policy Optimization (PPO) reinforcement learning. This project trains neural network agents to discover optimal sequences of compiler transformations for MLIR programs.

## 🎯 Overview

MLIR-RL learns to optimize MLIR code by:
- **Training an RL agent** to select compiler transformations (tiling, interchange, vectorization)
- **Measuring real execution speedups** using MLIR's execution engine
- **Generalizing across benchmarks** through neural network policies

**Key Results:**
- 156x speedup on matrix multiplication
- 11x speedup on 2D convolution  
- 6-8x speedup on element-wise operations

## 📁 Project Structure

See [docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md) for detailed organization.

```
MLIR-RL/
├── bin/                  # Main executable scripts
├── rl_autoschedular/    # Core RL implementation (PPO, environment, models)
│   └── models/          # Modular neural network architectures (LSTM, DistilBERT)
├── data_generation/     # ✨ Generate training data & convert models
├── evaluation/          # ✨ Evaluate trained agents vs baselines
├── benchmarks/          # ✨ Benchmark suites (single ops, neural nets)
├── utils/               # Utility modules (config, logging, Dask)
├── analysis/            # Plotting and analysis tools
├── experiments/         # Neptune sync and experiment utilities
├── notebooks/           # Jupyter notebooks for exploration
├── docs/                # Documentation (organized by topic)
├── scripts/             # SLURM job submission scripts
├── config/              # Training configuration files
├── data/                # Benchmark datasets
└── tools/               # MLIR analysis tools (AST dumper, vectorizer)
```

✨ **New**: Integrated data generation and evaluation systems from previous project!

## 🚀 Quick Start

### Prerequisites

- **CMake** ≥ 3.20
- **Ninja** build system
- **GCC/G++** 13.2
- **Python** ≥ 3.11
- **LLD** linker
- SLURM cluster (for distributed training)

### 1. Build LLVM/MLIR

```bash
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake -S ../llvm -B . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build . --target check-mlir
```

See [docs/MLIR_Python_Setup_Steps.md](docs/MLIR_Python_Setup_Steps.md) for detailed setup.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy and edit `.env.example`:

```bash
cp .env.example .env
# Edit .env with your paths and credentials
```

Required variables:
```bash
LLVM_BUILD_PATH=./llvm-project/build
AST_DUMPER_BIN_PATH=./tools/ast_dumper/build/bin/AstDumper
NEPTUNE_PROJECT=your-workspace/your-project  # Optional: for tracking
NEPTUNE_TOKEN=your-api-token                  # Optional: for tracking
```

### 4. Generate Training Data (Optional)

If you want to create custom training data:

```bash
# Generate random MLIR programs
python data_generation/random_mlir_gen.py

# Convert neural networks to MLIR
python data_generation/nn_to_mlir.py
```

See [Data Generation Guide](docs/guides/DATA_GENERATION_INTEGRATION.md) for details.

### 5. Run Training

**On SLURM cluster:**
```bash
sbatch scripts/train.sh
```

**Interactive mode (for testing):**
```bash
bash scripts/train.sh
```

Training will automatically:
- ✅ Collect trajectories using PPO
- ✅ Save model checkpoints every 5 iterations
- ✅ Evaluate every 100 iterations
- ✅ Sync results to Neptune (if configured)

## 📊 Monitoring and Visualization

### Check Training Progress

```bash
# View job status
squeue -u $USER

# Monitor output
tail -f logs/p-original-c_<JOB_ID>.out
```

### Generate Plots

```bash
python analysis/plot_results.py results/run_X
```

Creates 4 plots:
- Speedup by operation type
- Geometric mean comparison
- Per-benchmark speedup
- Training metrics (loss, entropy, rewards)

### Neptune Dashboard

If Neptune is configured, view results at:
```
https://app.neptune.ai/<your-workspace>/<your-project>
```

See [docs/guides/NEPTUNE_AUTO_SYNC.md](docs/guides/NEPTUNE_AUTO_SYNC.md) for auto-sync details.

## 🧪 Evaluation

### Evaluate Trained Agent

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
    output_file=Path('results/nn_eval.json')
)
"

# Run PyTorch baseline
python evaluation/pytorch_baseline.py
```

See [Evaluation Guide](docs/guides/DATA_GENERATION_INTEGRATION.md) for details.

## 🔧 Configuration

Edit `config/config.json` to customize:

```json
{
  "benchmarks_folder_path": "data/test/code_files",
  "json_file": "data/test/execution_times_train.json",
  "nb_iterations": 1000,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "clip_epsilon": 0.2,
  "value_epochs": 5,
  "ppo_epochs": 4
}
```

See [config/example.json](config/example.json) for all options.

## 📚 Documentation

### Quick Links
- **[📖 Documentation Index](docs/README.md)** - Complete documentation guide

### Architecture & Design
- **[Project Structure](docs/architecture/PROJECT_STRUCTURE.md)** - Complete directory guide
- **[Model Architecture Comparison](docs/architecture/MODEL_ARCHITECTURE_COMPARISON.py)** - LSTM vs DistilBERT
- **[Refactoring Guide](docs/architecture/REFACTORING_PHASE1_COMPLETE.md)** - Modular architecture

### Models
- **[DistilBERT Model](docs/models/DISTILBERT_MODEL.md)** - Transformer-based embedding
- **[DistilBERT Implementation](docs/models/DISTILBERT_IMPLEMENTATION.md)** - Technical details
- **[DistilBERT Data Integration](docs/models/DISTILBERT_DATA_INTEGRATION.md)** - Data preprocessing

### Guides
- **[Data Generation Integration](docs/guides/DATA_GENERATION_INTEGRATION.md)** - Generate training data & evaluate
- **[SLURM Guide](docs/guides/SLURM_GUIDE.md)** - Running on clusters
- **[Plotting Guide](docs/guides/PLOTTING_README.md)** - Visualization
- **[Neptune Auto-Sync](docs/guides/NEPTUNE_AUTO_SYNC.md)** - Experiment tracking
- **[GitHub Checklist](docs/guides/GITHUB_CHECKLIST.md)** - Pre-push checklist

### Setup
- **[MLIR Setup](docs/setup/MLIR_Python_Setup_Steps.md)** - Detailed LLVM/MLIR build
- **[Quick Reference](docs/setup/quick_reference.sh)** - Common commands

## 🎓 How It Works

### RL Environment

**State**: AST-based representation of MLIR code (loop nest structure, tensor dimensions, access patterns)

**Actions**: Compiler transformations
- Tiling (with learned tile sizes)
- Loop interchange
- Vectorization
- Caching strategies

**Reward**: Execution speedup vs. baseline

### Training Algorithm

Uses **Proximal Policy Optimization (PPO)**:
1. Collect trajectories by applying transformations to benchmarks
2. Execute optimized code and measure speedup
3. Update policy network to maximize expected speedup
4. Update value network to predict future rewards
5. Repeat for thousands of iterations

### Model Architecture

Hierarchical neural network:
- **State encoder**: Processes loop nest features
- **Action decoder**: Outputs transformation probabilities
- **Value head**: Estimates state value

## 🧪 Datasets

- **`data/test/`** - Small balanced dataset (17 benchmarks) for quick testing
- **`data/all/`** - Full dataset with diverse operation types

Benchmarks include:
- Matrix multiplication (`matmul`)
- 2D convolution (`conv2d`)
- Element-wise addition (`add`)
- Pooling operations (`pooling`)
- ReLU activation (`relu`)

## 🤝 Contributing

This is a research project. Feel free to:
1. Open issues for bugs or questions
2. Submit pull requests with improvements
3. Share your results and benchmarks

## 📄 License

MIT License

Copyright (c) 2025 Mehdi BENCHIKH & Tayeb BESSAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🙏 Acknowledgments

- Built on [LLVM/MLIR](https://mlir.llvm.org/)
- Uses [Neptune.ai](https://neptune.ai/) for experiment tracking
- Distributed computing via [Dask](https://dask.org/)

## 📧 Contact

mehdi.benchikh1@gmail.com \
bessai.tayeb2003@gmail.com
