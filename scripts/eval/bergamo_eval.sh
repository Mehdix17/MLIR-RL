#!/bin/bash
#SBATCH --job-name=mlir-eval-bergamo
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=128
#SBATCH -C bergamo
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_bergamo_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_bergamo_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL

set -e

# Usage:
#   sbatch scripts/bergamo_eval.sh config/ablation/v45_no_hw_blocks.json
#   EVAL_START=400 FORCE_RUN_ID=1 sbatch scripts/bergamo_eval.sh config/ablation/v45_no_hw_blocks.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Restore standard utilities path
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Environment Setup
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

# Config Resolution
CONFIG="${1:-$PROJECT_ROOT/config/ablation/v45_no_hw_blocks.json}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

# Determine Implementation from Config
CONFIG_IMPL=$(python3 - <<PY
import json
try:
    with open("$CONFIG", "r") as f:
        print((json.load(f).get("implementation") or "").strip())
except Exception:
    print("")
PY
)

IMPLEMENTATION="${CONFIG_IMPL:-rl_autoschedular_v4_5}"
export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
export CONFIG_FILE_PATH="$CONFIG"

# Set Dask/Multiprocessing to use the full Bergamo node resources
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Evaluate only the last checkpoint
export EVAL_LAST_ONLY=1
export EVAL_DIR=results/experiment3/v4_5_agent/run_0/models

echo "=========================================="
echo "Bergamo Evaluation started at $(date)"
echo "Implementation: $IMPLEMENTATION"
echo "Config:         $CONFIG_FILE_PATH"
echo "Resources:      128 CPUs, 300GB RAM"
echo "Range:          Start=${EVAL_START:-0}, End=${EVAL_END:-Max}"
echo "Stride:         100"
echo "Logging Run ID: ${FORCE_RUN_ID:-Default}"
echo "Node:           $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"

# Run evaluation
python scripts/eval/eval.py

echo "Evaluation completed at $(date)"
