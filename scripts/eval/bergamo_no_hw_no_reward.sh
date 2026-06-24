#!/bin/bash
#SBATCH --job-name=mlir-eval-nohw-norw
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_nohw_norw_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_nohw_norw_%j.err
#SBATCH --mail-type=END,FAIL

set -e

# Usage:
#   sbatch scripts/eval/bergamo_no_hw_no_reward.sh
#   Evaluates No-HW-No-Reward ckpt 1100 on Bergamo (eval_base, 2163 benchmarks)

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
echo "No-HW-No-Reward Bergamo Evaluation"
echo "Checkpoint: 1100"
echo "Cluster: Bergamo (train cluster, baseline)"
echo "Benchmarks: eval_base.json (2163)"
echo "=========================================="

export EVAL_LABEL=bergamo

cd "$PROJECT_ROOT"
CONFIG="$PROJECT_ROOT/config/new_dataset/eval/v45_no_hw_no_reward_eval.json"
bash "$PROJECT_ROOT/scripts/eval/eval.sh" "$CONFIG" --checkpoint 1100

echo "Evaluation completed at $(date)"
