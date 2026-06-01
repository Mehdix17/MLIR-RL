#!/bin/bash
#SBATCH --job-name=v45-eval
#SBATCH --partition=compute
#SBATCH -C dalma
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-08:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/v45_eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/v45_eval_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/eval/eval_v4_5.sh 700
#   sbatch scripts/eval/eval_v4_5.sh 800
#   ...
#   sbatch scripts/eval/eval_v4_5.sh 1999
#
# Single-checkpoint V4.5 evaluation on base_eval.json (Dalma hardware).
# Output: results/V4_5_agent/v4_5_agent/run_<CKPT>/logs/eval/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CKPT_NUM="${1:-${CKPT:-}}"
[[ -z "$CKPT_NUM" ]] && { echo "Usage: sbatch scripts/eval/eval_v4_5.sh <checkpoint_number>"; exit 1; }

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$PROJECT_ROOT/config/old_dataset/eval/v4_5_eval.json}"
export EVAL_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"
export EVAL_START="$CKPT_NUM"
export EVAL_END="$CKPT_NUM"
export FORCE_RUN_ID="$CKPT_NUM"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "V4.5 Eval (base_eval.json) — $(date)"
echo "Cluster:  dalma"
echo "Checkpoint: model_${CKPT_NUM}.pt"
echo "Benchmarks: 3014 (base_eval.json)"
echo "Output:    results/V4_5_agent/"
echo "Node:      $(hostname)"
echo "=========================================="

python scripts/eval/eval.py

echo ""
echo "Completed at $(date)"
