#!/bin/bash
#SBATCH --job-name=mlir-eval-dalma
#SBATCH --partition=compute
#SBATCH --constraint=dalma
#SBATCH --mem=32G
#SBATCH --cpus-per-task=28
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_dalma_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_dalma_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL

set -e

# Usage:
#   sbatch scripts/dalma_eval.sh config/train/v4_5.json
#   EVAL_LAST_ONLY=1 sbatch scripts/dalma_eval.sh config/train/v4_5.json

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
CONFIG="${1:-$PROJECT_ROOT/config/train/v4_5.json}"
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

# Evaluate only the last checkpoint (model_1999.pt)
export EVAL_LAST_ONLY=1
export EVAL_DIR=results/experiment3/v4_5_agent/run_0/models

echo "=========================================="
echo "Dalma Evaluation started at $(date)"
echo "Implementation: $IMPLEMENTATION"
echo "Config:         $CONFIG_FILE_PATH"
echo "Resources:      28 CPUs, 32GB RAM"
echo "Constraint:     dalma"
echo "Checkpoint:     last only (EVAL_LAST_ONLY=1)"
echo "Node:           $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"

# Run evaluation
python scripts/eval/eval.py

echo "Evaluation completed at $(date)"
