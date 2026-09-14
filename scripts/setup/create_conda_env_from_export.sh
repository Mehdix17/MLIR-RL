#!/usr/bin/env bash
# Create a conda environment from exported YAML and optional pip requirements
# Usage:
#   ./scripts/create_conda_env_from_export.sh env_export/environment_mlir.yml
#   ./scripts/create_conda_env_from_export.sh env_export/environment_mlir.yml pip_requirements_mlir.txt
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <environment.yml> [pip_requirements.txt]" >&2
    exit 1
fi

YML_FILE="$1"
PIP_FILE="${2:-}"

if [[ ! -f "$YML_FILE" ]]; then
    echo "YAML file not found: $YML_FILE" >&2
    exit 2
fi

# Extract env name from yaml (the name key) or default to 'mlir_export'
ENV_NAME=$(grep '^name: ' "$YML_FILE" | awk '{print $2}')
if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME="mlir_export"
fi

# Create the env using mamba if available; otherwise conda
if command -v mamba >/dev/null 2>&1; then
    echo "Using mamba to create environment $ENV_NAME"
    mamba env create -f "$YML_FILE" --force
else
    echo "Using conda to create environment $ENV_NAME"
    conda env create -f "$YML_FILE" --force
fi

# Activate and install pip extras if provided
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

if [[ -n "$PIP_FILE" && -f "$PIP_FILE" ]]; then
    echo "Installing pip packages from $PIP_FILE"
    python -m pip install -r "$PIP_FILE"
fi

# Print versions checks
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
except Exception as e:
    print('torch import failed', e)
try:
    import torch_mlir
    print('torch-mlir', getattr(torch_mlir, '__version__','N/A'))
except Exception as e:
    print('torch-mlir import failed', e)
PY

echo "Environment $ENV_NAME created and packages installed. Activate with: conda activate $ENV_NAME"
