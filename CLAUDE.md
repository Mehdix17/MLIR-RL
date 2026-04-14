# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Getting Started

### Prerequisites
- CMake ≥3.20, Ninja, GCC 13.2, Python 3.11+
- LLVM project cloned from llvm/llvm-project repository

### Setup Process
1. **Build MLIR**:
   ```bash
   git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
   mkdir llvm-project/build
   cd llvm-project/build
   cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
     -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
     -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
     -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON
   cmake --build . --target check-mlir
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** (in ~/.bashrc or .env):
   ```env
   NEPTUNE_PROJECT=<NEPTUNE_PROJECT_URL>
   NEPTUNE_TOKEN=<NEPTUNE_API_TOKEN>
   LLVM_BUILD_PATH=llvm-project/build
   MLIR_SHARED_LIBS=llvm-project/build/lib/libomp.so,llvm-project/build/lib/libmlir_c_runner_utils.so,llvm-project/build/lib/libmlir_runner_utils.so
   ```

## Project Architecture

This repository implements a reinforcement learning system for optimizing MLIR code through program transformations.

### Core Components

#### 1. RL Autoschedular (`rl_autoschedular/` directory)
- **Model (`model.py`)**: Hierarchical PPO model with policy and value networks using LSTM embeddings
- **Environment (`env.py`)**: Manages benchmarks, state transitions, and reward calculation
- **Observation (`observation.py`)**: Defines observation space including operation features, producer features, action history, and masks
- **Actions (`actions/` directory)**: Transformation actions (Tiling, Vectorization, Interchange, Fusion)
- **State (`state.py`)**: Tracks operation state, benchmark features, and transformation history
- **Execution (`execution.py`)**: Handles MLIR code compilation and execution timing
- **Benchmarks (`benchmarks.py`)**: Manages benchmark datasets and execution time collection

#### 2. Configuration System (`utils/config.py`)
- JSON-based configuration files in `config/` directory
- Parameters control:
  - Search space dimensions (`max_num_stores_loads`, `max_num_loops`, `max_num_load_store_dim`)
  - PPO hyperparameters (`lr`, `entropy_coef`, `nb_iterations`, `ppo_batch_size`, `ppo_epochs`)
  - Model architecture (`len_trajectory`, `truncate`)
  - Execution settings (`use_bindings`, `use_vectorizer`, `data_format`)

#### 3. Tooling (`tools/` directory)
- **AST Dumper (`tools/ast_dumper`)**: Extracts features from MLIR code
- **Vectorizer (`tools/vectorizer`)**: Vectorization utilities using LLVM passes

## Common Development Tasks

### Building & Testing
```bash
# Build MLIR (from llvm-project/build)
cmake --build . --target check-mlir

# Run specific test
ctest -VV --test <test_name>

# Run all tests
make check
```

### Working with Benchmarks
```bash
# Generate base execution times for training
python scripts/get_base.py --benchmarks-dir data/nn/code_files --output data/nn/base_exec_times.json

# Configuration example (see config/example.json)
# Adjust parameters in JSON files to control search space and training behavior
```

### Training Workflow
1. Prepare benchmark data using `scripts/get_base.py`
2. Configure training parameters in `config/` directory JSON files
3. Run training script (typically `train.py` or similar entry point)
4. Monitor results via Neptune logging (if enabled) or local logs

### Environment Troubleshooting

#### Fixing Broken MLIR Python Bindings
If MLIR Python bindings are broken (symlinks pointing to wrong paths):
```bash
# Restore source files from git
cd llvm-project
git checkout HEAD -- mlir/python/mlir/

# Fix symlinks to point to your user directory
find build/tools/mlir/python_packages/mlir_core -type l | while read link; do
    target=$(readlink "$link")
    if echo "$target" | grep -q "OTHER_USER"; then
        new_target=$(echo "$target" | sed 's|/scratch/OTHER_USER/|/scratch/YOUR_USER/|g')
        rm "$link" && ln -s "$new_target" "$link"
    fi
done
```

#### Environment Variable Issues
Add to virtualenv activate script:
```bash
export PYTHONPATH=/scratch/YOUR_USER/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=~/envs/mlir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Key Files Reference

- **Configuration**: `config/example.json` (template), `utils/config.py` (parser)
- **Data**: `data/` directory contains benchmarks, execution times, and training data
- **Documentation**: `README.md` contains detailed setup and troubleshooting guides
- **Main Entry Points**: Look for training scripts in root directory or `scripts/` folder

## Development Guidelines

1. **Configuration Changes**: Modify JSON files in `config/` directory for experimentation
2. **Action Development**: Add new transformations in `rl_autoschedular/actions/` following the base Action pattern
3. **Observation Space**: Update `observation.py` when adding new features to the state representation
4. **Testing**: Use existing test patterns in llvm-project for MLIR-specific functionality

## Evaluation Dashboard

A Streamlit-based dashboard visualizes evaluation results from the MLIR-RL system. Key aspects:

- **Data Sources**: Reads directly from existing files:
  - `results/<run_i>/exec_times/base_eval.json` (MLIR baseline execution times)
  - `results/<run_i>/exec_times/pytorch.json` (PyTorch eager/compile/JIT baselines)
  - `results/<run_i>/logs/eval/speedup/<bench_name>` (RL speedup ratios, one float per checkpoint)
- **Benchmark Categorization**: Identified from naming patterns:
  - **Model Family**: Prefix before first underscore (e.g., `gpt-2`, `convnext`, `albert`)
  - **Operation Type**: Middle segment matching known ops (`conv_2d_nchw_fchw`, `matmul`, `pooling_nchw_max`, etc.)
- **Primary Visual Focus**: Aggregations per model family (not raw tables) due to >3000 benchmarks
- **Core Visualizations**:
  - Average speedup distribution bar chart
  - Comparison against MLIR baseline and PyTorch baselines
  - Execution time distributions with interactive histograms
  - Speedup variance shown via box/violin plots
  - Checkpoint progress line chart
  - Failure analysis pie chart with drill-down capability
  - Statistical summary table (count, mean, median, std‑dev)
- **Interactive Filters**:
  - Select training run (`run_i`)
  - Choose model family and operation type
  - Filter by speedup range
  - Select checkpoint range (when progress view is enabled)
- **Export Capabilities**:
  - **CSV Export**: Download filtered view data
  - **JSON Export**: Export raw evaluation results
  - **Image Export**: Save charts as PNG/SVG
- **User Workflow**:
  1. Launch with `streamlit run dashboard.py`
  2. Select a training run from dropdown
  3. Apply filters to update visualizations
  4. Drill into failure analysis or statistical tables
  5. Click “Download CSV” to export current view
  6. Press “Refresh Data” after new evaluation results appear (manual refresh only)
- **Implementation Notes**:
  - Data loading follows the pattern from `scripts/compare_baselines.py`
  - Aggregations computed per model family to handle >3000 benchmarks efficiently
  - Performance optimized through lazy loading and aggregation‑first approach
  - Required dependencies: `streamlit`, `pandas`, `plotly`
- **File Location**: The specification is documented in `docs/dashboard.md`

---
**End of Document**