#!/bin/bash
#SBATCH --job-name=mlir-eval-batch
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=bergamo
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_batch_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_batch_%j.err

set -e

CONFIG="$1"
START="${2:-100}"
END="${3:-1000}"
STEP="${4:-100}"

if [[ -z "$CONFIG" ]]; then
    echo "Usage: sbatch scripts/eval/eval_batch.sh <config_file_path> [start] [end] [step]"
    exit 1
fi

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
export MIN_EXEC_TIMEOUT=${MIN_EXEC_TIMEOUT:-300}

CONFIG_IMPL=$(python3 -c "import json; print(json.load(open('$CONFIG'))['implementation'])")
export AUTOSCHEDULER_IMPL="$CONFIG_IMPL"
export CONFIG_FILE_PATH="$CONFIG"

RESULTS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['results_dir'])")
export EVAL_DIR="$PROJECT_ROOT/$RESULTS_DIR/models"

echo "Evaluating checkpoints from $START to $END with step $STEP..."

for ckpt in $(seq "$START" "$STEP" "$END"); do
    echo "=========================================="
    echo "Evaluating checkpoint $ckpt..."
    echo "=========================================="
    export EVAL_CHECKPOINT="$ckpt"
    export EVAL_START="$ckpt"
    export EVAL_END="$ckpt"
    export EVAL_STRIDE=1
    export FORCE_RUN_ID="ckpt_${ckpt}"
    
    python scripts/eval/eval.py
done
