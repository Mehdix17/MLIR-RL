#!/bin/bash
#SBATCH --job-name=mlir-train-generated
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train_generated_%j.out
#SBATCH --error=logs/train_generated_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
	set -a
	source "$PROJECT_ROOT/.env"
	set +a
fi

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate "${CONDA_ENV:-mlir}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

CONFIG_PATH="$PROJECT_ROOT/config/generated-benchmarks.json"

python train.py --config "$CONFIG_PATH"
