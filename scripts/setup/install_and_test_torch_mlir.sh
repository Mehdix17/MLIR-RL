#!/usr/bin/env bash
# install_and_test_torch_mlir.sh
# Installs PyTorch + torch-mlir (pip wheels) in the active conda environment
# then runs basic import and compile tests. Designed for HPC/conda env usage.

set -euo pipefail

# Defaults
CUDA="auto"   # detect automatically
AUTO=false
PYTORCH_INDEX=""

usage() {
    cat <<EOF
Usage: $0 [--cuda <cpu|cu118|cu121|auto>] [--auto]

Options:
  --cuda <cpu|cu118|cu121|auto>  Force install to use CPU or a specific CUDA toolkit version. Default: auto (automatic detection of GPU)
  --auto                        Run non-interactively (assume defaults)

This script installs PyTorch and torch-mlir in your current conda environment using pip.
It then runs tests to verify import and compile functionality.

Examples:
  $0 --cuda cpu                  # Install CPU-only PyTorch and torch-mlir
  $0 --cuda cu118 --auto         # Install CUDA 11.8 compatible PyTorch and torch-mlir non-interactively
EOF
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA="$2"; shift 2;;
    --auto)
      AUTO=true; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"; usage;;
  esac
done

# Check conda env
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: No active conda environment detected. Please activate your conda env (e.g., 'conda activate mlir') and re-run."
  exit 2
fi

echo "Active conda env: $CONDA_PREFIX"

echo "Detecting GPU presence..."
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=true
else
  HAS_GPU=false
fi

if [[ "$CUDA" == "auto" ]]; then
  if [[ "$HAS_GPU" == "true" ]]; then
    echo "GPU detected. Please select target CUDA version for PyTorch and torch-mlir (example choices: cu118, cu121)"
    if [[ "$AUTO" == "true" ]]; then
      echo "Non-interactive mode: defaulting to cu118"
      CUDA=cu118
    else
      read -p "CUDA version to install (cu118/cu121/cu126 or type 'cpu'): " CUDA
      CUDA=${CUDA:-cu118}
    fi
  else
    echo "No GPU detected: installing CPU-only PyTorch"
    CUDA=cpu
  fi
fi

if [[ "$CUDA" == "cpu" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
fi

echo "Installing for CUDA target: $CUDA"

# Ensure pip up to date
python -m pip install --upgrade pip setuptools wheel packaging || true

# Choose common PyTorch versions; use the PyTorch wheels provided at PyTorch index
# We avoid pinning exact versions here so we pull the latest wheel available for requested CUDA target.

echo "Installing PyTorch and torchvision from PyTorch index: ${PYTORCH_INDEX}"
python -m pip install --upgrade "torch" "torchvision" --extra-index-url ${PYTORCH_INDEX}

# Install torch-mlir from PyPI (stable). If you want nightly, switch to 'torch-mlir-nightly'.
# In some environments, it's better to use the 'torch-mlir-nightly' package if the PyTorch wheel is bleeding edge.

echo "Installing torch-mlir (stable)"
python -m pip install --upgrade torch-mlir

# Additional utils for tooling (optional)
python -m pip install --upgrade numpy pybind11

# Summarize versions
echo "== Installed package versions =="
python - <<'PY'
import importlib, pkgutil
import sys
print('python', sys.version.splitlines()[0])
try:
  import torch
  print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
except Exception as e:
  print('torch import failed:', e)
try:
  import torch_mlir
  print('torch-mlir:', getattr(torch_mlir, '__version__', 'N/A'))
  print('torch-mlir module path:', getattr(torch_mlir, '__file__', 'N/A'))
except Exception as e:
  print('torch-mlir import failed:', e)
PY

# Run tests
if [[ ! -x scripts/test_torch_mlir.sh ]]; then
  echo "Making test scripts executable"
  chmod +x scripts/test_torch_mlir.sh scripts/test_torch_mlir_compile.py
fi

echo "Running tests (import + compile)..."
if ./scripts/test_torch_mlir.sh; then
  echo "Tests passed. Installation appears functional."
  exit 0
else
  echo "Tests failed. Review output above for possible issues."
  exit 3
fi
