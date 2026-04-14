# MLIR-RL Project Overview

MLIR-RL is a Reinforcement Learning (RL)-based autoscheduler for MLIR. It uses the Proximal Policy Optimization (PPO) algorithm to automatically explore and apply high-level code transformations (such as loop tiling, fusion, and vectorization) to MLIR operations, aiming to optimize performance on target hardware.

## Core Technologies
- **MLIR/LLVM**: Used for intermediate representation and applying transformations.
- **Python (PyTorch)**: Implements the RL agent and training logic (PPO).
- **C++**: Custom tools for AST analysis and specialized transformations.
- **Streamlit**: Provides a dashboard for visualizing results and agent performance.
- **Neptune**: Used for experiment tracking and logging.

## Project Structure
- `rl_autoschedular/`: Core RL implementation (Environment, PPO, Models, States).
- `tools/`: C++ components:
  - `ast_dumper`: Extracts features from MLIR for RL observations.
  - `vectorizer`: Specialized vectorization tool.
- `scripts/`: Entry points for common tasks:
  - `train.py`: Main training script.
  - `eval.py`: Evaluation script for trained models.
  - `get_base.py`: Script to generate baseline execution times.
- `config/`: JSON configuration files for different experiments.
- `data/`: Benchmarks (MLIR files) and pre-extracted features.
- `dashboard/`: Streamlit dashboard for monitoring and analysis.
- `utils/`: Shared utility functions (logging, configuration parsing).

## Building and Setup

### Prerequisites
- CMake (>= 3.20), Ninja, GCC 13.2, Python 3.11+
- LLVM project (cloned from `llvm/llvm-project`)

### MLIR Build
Build MLIR with Python bindings enabled:
```bash
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON
cmake --build . --target check-mlir
```

### Environment Configuration
1. Install Python dependencies: `pip install -r requirements.txt`
2. Configure `.env` based on `.env.example`:
   - `LLVM_BUILD_PATH`: Path to your MLIR build directory.
   - `MLIR_SHARED_LIBS`: Paths to required MLIR shared libraries.
   - `AST_DUMPER_BIN_PATH`: Path to the compiled AstDumper binary.
   - `VECTORIZER_BIN_PATH`: Path to the compiled Vectorizer binary.

## Common Workflows

### Training
To start training a new agent:
```bash
python scripts/train.py --config config/your_config.json
```

### Evaluation
To evaluate a trained model:
```bash
python scripts/eval.py --config config/your_config.json --model results/your_model.pt
```

### Generating Baselines
Before training, you often need base execution times:
```bash
python scripts/get_base.py --benchmarks-dir data/your_benchmarks --output data/base_times.json
```

## Development Conventions
- **Configuration**: Always use JSON configuration files in the `config/` directory for experiments.
- **Environment Variables**: Use the `.env` file for local path configurations and API keys.
- **Logging**: Training metrics are automatically logged to Neptune if `logging` is enabled in the config.
- **C++ Tools**: Any changes to `tools/` require a rebuild of the respective binaries.
