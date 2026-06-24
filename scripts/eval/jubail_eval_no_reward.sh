#!/bin/bash
#SBATCH --job-name=mlir-eval-norew-jub
#SBATCH --partition=compute
#SBATCH -C jubail
#SBATCH --mem=64G
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_norew_jubail_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_norew_jubail_%j.err
#SBATCH --mail-type=END,FAIL

set -e

# Usage:
#   sbatch scripts/eval/jubail_eval_no_reward.sh
#   Evaluates No-Reward checkpoint 1100 on Jubail (AMD EPYC 7742)

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

export MIN_EXEC_TIMEOUT=60

export CONFIG_FILE_PATH="$PROJECT_ROOT/config/new_dataset/eval/v45_no_shaped_reward_eval_multi_hw.json"
export AUTOSCHEDULER_IMPL=rl_autoschedular_v45_no_shaped_reward
export EVAL_START=1100
export EVAL_END=1100
export EVAL_STRIDE=1
export FORCE_RUN_ID=jubail

echo "=========================================="
echo "No-Reward Jubail Evaluation started at $(date)"
echo "Implementation: $AUTOSCHEDULER_IMPL"
echo "Config:         $CONFIG_FILE_PATH"
echo "Checkpoint:     1100"
echo "Cluster:        Jubail (AMD EPYC 7742, 128 cores)"
echo "Benchmarks:     eval_base.json (2163 benchmarks)"
echo "Resources:      28 CPUs, 32GB RAM"
echo "Node:           $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"
python scripts/eval/eval.py

echo "Evaluation completed at $(date)"
