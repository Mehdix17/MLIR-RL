#!/bin/bash
#SBATCH --job-name=mlir-final-eval
#SBATCH --partition=compute
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --constraint=bergamo
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/final_eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/final_eval_%j.err

set -e

# 1: final-eval config (config/.../v5_distributed_final_eval.json)
FINAL_CFG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
python scripts/eval/final_eval.py "$FINAL_CFG"