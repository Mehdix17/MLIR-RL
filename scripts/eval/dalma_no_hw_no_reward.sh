#!/bin/bash
#SBATCH --job-name=mlir-eval-nohw-norw
#SBATCH --partition=compute
#SBATCH --constraint=dalma
#SBATCH --mem=64G
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_nohw_norw_da_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_nohw_norw_da_%j.err
#SBATCH --mail-type=END,FAIL

set -e

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

echo "=========================================="
echo "No-HW-No-Reward Dalma Evaluation"
echo "Checkpoint: 1100"
echo "Cluster: Dalma (Intel Xeon E5-2680 v4, 28c)"
echo "Benchmarks: eval_base.json (2163)"
echo "=========================================="

export EVAL_LABEL=dalma

cd "$PROJECT_ROOT"
CONFIG="$PROJECT_ROOT/config/new_dataset/eval/v45_no_hw_no_reward_eval.json"
bash "$PROJECT_ROOT/scripts/eval/eval.sh" "$CONFIG" --checkpoint 1100

echo "Evaluation completed at $(date)"
