#!/bin/bash
#SBATCH --job-name=mlir-ablation-eval
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=128
#SBATCH -C bergamo
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/ablation_eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/ablation_eval_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL

set -e

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

CONFIG="${1}"
if [[ -z "$CONFIG" ]]; then
    echo "Usage: sbatch scripts/ablation_bergamo_eval.sh <config_path>"
    exit 1
fi

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

IMPLEMENTATION="${CONFIG_IMPL}"
export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
export CONFIG_FILE_PATH="$CONFIG"

# Support for FORCE_RUN_ID for separate ablation logging
if [[ -n "${FORCE_RUN_ID:-}" ]]; then
    export FORCE_RUN_ID="$FORCE_RUN_ID"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=========================================="
echo "Ablation Evaluation started at $(date)"
echo "Implementation: $IMPLEMENTATION"
echo "Config:         $CONFIG_FILE_PATH"
echo "Run ID:         ${FORCE_RUN_ID:-Default}"
echo "Node:           $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/eval/ablation_eval.py
echo "Ablation Evaluation completed at $(date)"
