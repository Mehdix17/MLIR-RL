#!/bin/bash
#SBATCH --job-name=mlir-eval-generated
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_generated_%j.out
#SBATCH --error=logs/eval_generated_%j.err

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

cd "$PROJECT_ROOT/evaluation"

# Run RL evaluation only (uses checkpoint default in orchestrator)
python orchestrator.py --checkpoint "$PROJECT_ROOT/models/ppo_model_MLIR-185-1.pt" --output "$PROJECT_ROOT/results/eval_combined.csv"
