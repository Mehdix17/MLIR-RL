# Onboarding: Project Setup & LLVM Build

> **Module 2**: Setting up the development environment, compiling LLVM/MLIR from source, building custom C++ utilities, and configuring variables.

---

## 1. Environment Prerequisites

Our reinforcement learning auto-scheduler relies on a customized LLVM compiler toolchain, C++ utilities, and a PyTorch-based RL agent. Before starting, ensure your system (or HPC user node) has the following:

- **OS**: Linux (Slurm HPC clusters supported and recommended)
- **Compilers**: GCC/G++ version 13.2+ or Clang equivalent
- **Build Tools**: CMake 3.20+, Ninja, Make
- **Python**: Version 3.11+
- **GPU (optional)**: CUDA (only needed if training models on GPU; compilation/JIT runs on CPU)

---

## 2. Step 1: Building LLVM & MLIR from Source

We compile a specific branch of LLVM from source to enable MLIR and Python Bindings.

```bash
# 1. Clone LLVM project (19.x release branch is recommended)
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git

# 2. Create build directory
mkdir -p llvm-project/build
cd llvm-project/build

# 3. Configure the build with CMake and Ninja
cmake -S ../llvm -B . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# 4. Compile MLIR (this may take 30-60 minutes depending on your CPU cores)
cmake --build . --target check-mlir
```

---

## 3. Step 2: Building Custom C++ Tools

We use two C++ helper tools to analyze MLIR code and extract hardware-specific properties:
1. **`ast_dumper`**: Generates a JSON representation of an MLIR operation dependency graph.
2. **`vectorizer`**: Checks if an operation can be vectorized using MLIR's target vector registers.

To build these tools, point them to your compiled LLVM and MLIR CMake configurations:

```bash
# Set environment variables for the build
export LLVM_BUILD_PATH=$(pwd)/llvm-project/build
export CMake_LLVM_DIR=$LLVM_BUILD_PATH/lib/cmake/llvm
export CMake_MLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir

# 1. Build ast_dumper
cd tools/ast_dumper
mkdir -p build && cd build
cmake -DLLVM_DIR=$CMake_LLVM_DIR -DMLIR_DIR=$CMake_MLIR_DIR ..
make -j$(nproc)
cd ../../..

# 2. Build vectorizer
cd tools/vectorizer
mkdir -p build && cd build
cmake -DLLVM_DIR=$CMake_LLVM_DIR -DMLIR_DIR=$CMake_MLIR_DIR ..
make -j$(nproc)
cd ../../..
```

---

## 4. Step 3: Conda Environment Setup

Set up a Python 3.11 virtual environment. You can use the provided `environment.yml` or install the packages manually:

```bash
# Create conda env from definition file
conda env create -f environment.yml -p ~/envs/mlir

# Activate the environment
conda activate ~/envs/mlir

# If installing manually, the core dependencies are:
# conda install python=3.11 pytorch=2.2 cpuonly -c pytorch
# pip install neptune numpy dask python-dotenv
```

---

## 5. Step 4: Environment Variables (`.env`)

Create a `.env` file in the root of the workspace. This file is parsed by `python-dotenv` at startup.

```ini
# --- Compiler Paths ---
LLVM_BUILD_PATH=/absolute/path/to/llvm-project/build
MLIR_SHARED_LIBS=/absolute/path/to/llvm-project/build/lib/libomp.so,/absolute/path/to/llvm-project/build/lib/libmlir_c_runner_utils.so,/absolute/path/to/llvm-project/build/lib/libmlir_runner_utils.so

# --- Custom C++ Binaries ---
AST_DUMPER_BIN_PATH=/absolute/path/to/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=/absolute/path/to/MLIR-RL/tools/vectorizer/build/bin/Vectorizer

# --- Tracker & HPC settings ---
NEPTUNE_PROJECT=your_neptune_workspace/mlir-rl
NEPTUNE_TOKEN=your_long_api_token_here
CONDA_ENV=/absolute/path/to/conda/envs/mlir
```

> [!WARNING]
> Keep your `NEPTUNE_TOKEN` secret and never commit `.env` files to git. A sample file is provided as `.env.example`.

---

## 6. Verification: Interactive Use

For interactive development and testing, you must load the environment variables and libraries correctly. Source them as follows:

```bash
# 1. Activate conda
source ~/envs/mlir/bin/activate

# 2. Load .env variables into current shell
set -a && source .env && set +a

# 3. Add GCC library paths to LD_LIBRARY_PATH (critical on HPC clusters to use GCC-14)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/gcc-14/lib64

# 4. Set target config file
export CONFIG_FILE_PATH=config/new_dataset/train/v4_9_large.json

# 5. Verify python compilation and import
python -c "import rl_autoschedular_v4_9.model; print('Imports work!')"
```

---

## 7. LLVM Build Gotchas (Shared Build Fix)

If your team is sharing an `llvm-project` directory compiled by another user or in another directory, the python package bindings will contain broken absolute symlinks pointing to the creator's user home directory. 

Run this recovery script inside `llvm-project/` to clean and re-create symlinks matching *your* paths:

```bash
cd llvm-project

# 1. Checkout python files to clear broken bindings
git checkout HEAD -- $(git ls-files mlir/python/mlir/ | tr '\n' ' ')

# 2. Find and rebuild symlinks targeting other users
find build/tools/mlir/python_packages/mlir_core -type l | while read link; do
    target=$(readlink "$link")
    if echo "$target" | grep -q "/scratch/OTHER_USER/"; then
        # Replace creator's username with yours
        new_target=$(echo "$target" | sed "s|/scratch/OTHER_USER/|/scratch/$USER/|g")
        rm "$link" && ln -s "$new_target" "$link"
    fi
done
```

> [!TIP]
> Use `rm` and `ln -s` individually instead of `ln -sf` because `ln -sf` will attempt to write *into* the broken symlink target, which is likely inside a folder you do not have permission to write to!

In the next module, we will review the overall software architecture of this project.
