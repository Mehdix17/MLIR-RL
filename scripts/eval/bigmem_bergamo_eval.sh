#!/bin/bash
#SBATCH --job-name=mlir-eval-bigmem
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=900G
#SBATCH --exclusive
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_bigmem_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_bigmem_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL

set -e

# Usage:
#   EVAL_START=400 EVAL_END=1100 FORCE_RUN_ID=1 sbatch scripts/bigmem_bergamo_eval.sh config/ablation/v45_no_reward_blocks.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-$PROJECT_ROOT/config/ablation/v45_no_reward_blocks.json}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

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

# Optimize for 256 logical cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=========================================="
echo "Bigmem Bergamo Evaluation started at $(date)"
echo "Implementation: $IMPLEMENTATION"
echo "Config:         $CONFIG_FILE_PATH"
echo "Resources:      256 CPUs, 900GB RAM (EXCLUSIVE)"
echo "Range:          Start=${EVAL_START:-0}, End=${EVAL_END:-Max}"
echo "Stride:         100"
echo "Logging Run ID: ${FORCE_RUN_ID:-Default}"
echo "Node:           $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/eval/eval.py
echo "Evaluation completed at $(date)"
