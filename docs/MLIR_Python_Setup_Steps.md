# MLIR Python Bindings Setup and Troubleshooting

This document summarizes the steps taken to build, install, and enable the MLIR Python bindings in a custom LLVM/MLIR build, including troubleshooting and environment configuration for a conda-based workflow.

---

## 1. Building LLVM/MLIR with Python Bindings

- Configured CMake with:
  - `-DLLVM_ENABLE_PROJECTS="mlir;clang;openmp"`
  - `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`
  - `-DPython3_EXECUTABLE=$(which python)` (using conda Python 3.11)
  - `-Dpybind11_DIR=$(python -m pybind11 --cmakedir)`
  - Used the correct source root: `-S ../llvm` from the `llvm-project/build` directory
- Ensured the conda toolchain and libstdc++ were recent enough for LLVM's CMake checks
- Built the project with `ninja` or `cmake --build .`

## 2. Locating the Built Python Package

- Located the built MLIR Python package at:
  - `/scratch/mb10856/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core`
- Verified the presence of compiled extension modules (e.g., `_mlir.cpython-311-x86_64-linux-gnu.so`)

## 3. Installing the Package into the Conda Environment

- Copied the built package into the conda environment's site-packages:
  ```bash
  cp -r /scratch/mb10856/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core \
      /home/mb10856/.conda/envs/mlir/lib/python3.11/site-packages/
  ```
- Uninstalled any pip-installed `mlir` package that could shadow the built package:
  ```bash
  /home/mb10856/.conda/envs/mlir/bin/pip uninstall -y mlir
  ```

## 4. Setting Up Runtime Library Paths Automatically

- Created conda activation/deactivation scripts to set `LD_LIBRARY_PATH` so the MLIR extensions can find their shared libraries:
  - **Activation script:**
    - `/home/mb10856/.conda/envs/mlir/etc/conda/activate.d/mlir_env.sh`
    - Prepends MLIR build and conda lib directories to `LD_LIBRARY_PATH`.
  - **Deactivation script:**
    - `/home/mb10856/.conda/envs/mlir/etc/conda/deactivate.d/mlir_env.sh`
    - Restores the previous `LD_LIBRARY_PATH`.

## 5. Verifying the Installation

- After activating the environment (`conda activate mlir`), tested with:
  ```python
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
  ```
- Output confirmed MLIR Python API is working.

## 6. Troubleshooting Notes

- If you see `ModuleNotFoundError: No module named 'mlir.ir'`, ensure:
  - The package is copied to the correct `site-packages` for the Python version used.
  - No pip-installed `mlir` package is shadowing the built one.
- If you see `ImportError` about missing symbols or version mismatch:
  - Make sure you are using the same Python version (e.g., 3.11) as was used for the build.
  - Ensure `LD_LIBRARY_PATH` includes both the MLIR build lib dir and the conda lib dir.

## 7. How to Use Going Forward

- Always activate your conda environment before running MLIR Python code:
  ```bash
  conda activate mlir
  ```
- The activation script will set up `LD_LIBRARY_PATH` automatically.
- Run your Python scripts as usual with the conda Python:
  ```bash
  python my_mlir_script.py
  ```

---

If you need to update the MLIR Python bindings, repeat steps 1–3 and ensure the activation scripts are still present.
