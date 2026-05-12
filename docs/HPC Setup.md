# New Installation (HPC)

# Local Installation

- Notion Guide : [https://yielding-sheet-a7e.notion.site/Installation-Running-MLIR-RL-Agent-On-Ubuntu-24-24dc22bb1876802e94dedbf2c0223777](https://www.notion.so/24dc22bb1876802e94dedbf2c0223777?pvs=21)

# Conda Environment Setup

### Create conda environment

Install needed packages

```bash
conda create -n mlir
conda activate mlir
```

### HPC GCC

```bash
conda install -c -y conda-forge python=3.11 cmake ninja numpy PyYAML pybind11
which gcc
which g++
which c++
```

### GCC (did not work, faced recurring linking issues)

```bash
conda install -c -y conda-forge python=3.11 gxx_linux-64=13.2.0 gcc_linux-64=13.2.0 libstdcxx-ng libgcc-devel_linux-64 cmake ninja numpy PyYAML pybind11
```

```bash
# set these environment variables BEFORE running cmake
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export CXXFLAGS="-I$CONDA_PREFIX/include"
export CFLAGS="-I$CONDA_PREFIX/include"

# most importantly - force atomic library
export LDFLAGS="$LDFLAGS -latomic"
```

### Create `gcc` and `g++` symbolic links

```bash
cd ~/.conda/envs/mlir/bin/
ln -s x86_64-conda-linux-gnu-gcc gcc
ln -s x86_64-conda-linux-gnu-g++ g++
ln -s x86_64-conda-linux-gnu-ar ar
```

### Verify the symbolic links and compilation

```bash
# verify
which gcc
which g++
gcc --version
g++ --version

# test compilation
echo '#include <cstddef>' | g++ -x c++ -c - -o /dev/null
echo '#include <cstdlib>' | g++ -x c++ -c - -o /dev/null
```

### Clang

```bash
conda install -c -y conda-forge python=3.11 clang clangxx lld cmake ninja numpy PyYAML pybind11
cmake --version
clang --version
clang++ --version
```

# LLVM Installation

### Clone MLIR-RL repo

```bash
cd $SCRATCH
git clone https://github.com/Modern-Compilers-Lab/MLIR-RL.git
```

### Clone LLVM repo inside MLIR-RL

```bash
cd MLIR-RL
git clone --depth 1 -b release/19.x [https://github.com/llvm/llvm-project.git](https://github.com/llvm/llvm-project.git)
```

### Build LLVM with MLIR and Python Bindings

```bash
mkdir llvm-project/build
cd llvm-project/build
```

### A catch, `cmake` fails to link to conda's `libstdc++` , it is predefined to look into the system's `libstdc++` . A workaround is to force the linking with :

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34 cmake --version
```

### Doing this command can also help build LLVM with MLIR and Python Bindings :

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/scratch/**<NetID>**/MLIR-RL/llvm-project/build/lib:$LD_LIBRARY_PATH
```

### Clang version (if gcc versions failed)

Test first without setting these flags, it may work out of the box

```bash
export CXXFLAGS="-nostdinc++ \
-I$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/15.2.0/include/c++ \
-I$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/15.2.0/include/c++/x86_64-conda-linux-gnu \
-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=X86 \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python) \
   -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34 cmake --build . --target check-mlir
```

![](https://t90121153285.p.clickup-attachments.com/t90121153285/2d6b8f3e-297f-4db4-a7b0-2483a9bbd7c9/image.png)

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34 cmake --build . --target check-mlir-python
```

![](https://t90121153285.p.clickup-attachments.com/t90121153285/1d9c6879-8512-4366-8905-632e102f3fa9/image.png)

```bash
cmake --build . --target check-openmp # This will build OpenMP related ones / didn't work, yet the execution works fine
```

### Or testing with ninja

```bash
ninja -j28 check-mlir
ninja -j28 check-mlir-python
ninja -j28 check-openmp # didn't work, yet the execution works fine
```

### Add MLIR binaries and Python bindings to `PATH`

```bash
export PATH=/scratch/**<NetID>**/MLIR-RL/llvm-project/build/bin:$PATH
export PYTHONPATH=/scratch/**<NetID>**/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
```

### Test MLIR and Python bindings

```bash
mlir-opt --version
python -c "import mlir; print('MLIR Python bindings loaded successfully')"
```

```python
python << 'EOF'
from mlir.ir import Context, Module

with Context() as ctx:
    module = Module.parse("""
    module {
      func.func @add(%arg0: i32, %arg1: i32) -> i32 {
        %0 = arith.addi %arg0, %arg1 : i32
        return %0 : i32
      }
    }
    """)
    print(module)
    print("\nMLIR working correctly!")
EOF
```

### To make these paths permanent, add to your `~/.bashrc` :

```bash
cat >> ~/.bashrc << 'EOF'

# MLIR paths (only when mlir-build env is active)
if [[ "$CONDA_DEFAULT_ENV" == "mlir-build" ]]; then
    export PATH=/scratch/**<NetID>**/MLIR-RL/llvm-project/build/bin:$PATH
    export PYTHONPATH=/scratch/**<NetID>**/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
fi
EOF
```

```bash
source ~/.bashrc

# reactivate the environment
conda activate mlir
```

# 8. Tools Installation

Within **MLIR-RL** repository execute : 

```bash
cd $SCRATCH/MLIR-RL
PROJECT_ROOT=$(pwd)
```

## AST Dumper

```bash
cd tools/ast_dumper
mkdir build
cd build
```

### Adjust the paths accordingly for LLLVM and MLIR

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/mlir \
  ..
  
cmake --build .
```

## Vectorizer

```bash
cd $SCRATCH/MLIR-RL
cd tools/vectorizer
mkdir build
cd build
```

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/mlir \
  ..
  
cmake --build .
```

# 9. Install Python Packages

Make sure your Conda environment is activated

```bash
cd $SCRATCH/MLIR-RL
python -m pip install -r requirements.txt
```

# 10. Create a Neptune Project

[neptune.ai | Experiment tracker purpose-built for foundation models](https://neptune.ai/)

# 11. Setup Environment Variables

Set `.env` (looking like the following)

```bash
CONDA_ENV=/home/**<NetID>**/.conda/envs/mlir-build
NEPTUNE_PROJECT=**<Neptune Project>**
NEPTUNE_TOKEN=**<Neptune Token>**
LLVM_BUILD_PATH=/scratch/**<NetID>**/MLIR-RL/llvm-project/build
MLIR_SHARED_LIBS=/scratch/**<NetID>**/MLIR-RL/llvm-project/build/lib/libomp.so,/scratch/**<NetID>**/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/**<NetID>**/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/**<NetID>**/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=/scratch/**<NetID>**/MLIR-RL/tools/vectorizer/build/bin/Vectorizer
```

### Mehdi .env :

```bash
CONDA_ENV=/home/**mb10856**/.conda/envs/mlir-build
NEPTUNE_PROJECT=mehdix/mlir-project
NEPTUNE_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NDQ5ZDFhNy0xMGQ5LTQxZDAtYjYzMC1hZTViYmE5OWIxMmUifQ==
LLVM_BUILD_PATH=/scratch/**mb10856**/MLIR-RL/llvm-project/build
MLIR_SHARED_LIBS=/scratch/**mb10856**/MLIR-RL/llvm-project/build/lib/libomp.so,/scratch/**mb10856**/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/**mb10856**/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/**<mb10856**/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=/scratch/**<mb10856**/MLIR-RL/tools/vectorizer/build/bin/Vectorizer
```

# 12. Change the config file

In config/config.json put this :

```json
{
    "max_num_stores_loads": 7,
    "max_num_loops": 12,
    "max_num_load_store_dim": 12,
    "num_tile_sizes": 7,
    "vect_size_limit": 512,
    "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
    "interchange_mode": "pointers",
    "exploration": ["entropy"],
    "init_epsilon": 0.5,
    "new_architecture": false,
    "normalize_bounds": "max",
    "normalize_adv": "standard",
    "sparse_reward": true,
    "split_ops": true,
    "reuse_experience": "none",
    "activation": "relu",
    "benchmarks_folder_path": "data/all/code_files",
    "bench_count": 64,
    "replay_count": 10,
    "nb_iterations": 10000,
    "ppo_epochs": 4,
    "ppo_batch_size": 32,
    "value_epochs": 0,
    "value_batch_size": 32,
    "value_coef": 0.5,
    "value_clip": false,
    "entropy_coef": 0.01,
    "lr": 0.001,
    "truncate": 5,
    "json_file": "data/all/execution_times_train.json",
    "eval_json_file": "data/all/execution_times_eval.json",
    "tags": ["all", "pointers", "IPFT~V", "entropy", "standard", "dask", "new-v"],
    "debug": false,
    "main_exec_data_file": "",
    "results_dir": "results"
}
```

### GCC HPC version

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=X86 \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=gcc \
   -DCMAKE_CXX_COMPILER=g++ \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python) \
   -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

### GCC version

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=X86 \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=$CC \
   -DCMAKE_CXX_COMPILER=$CXX \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=$(which python) \
   -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```