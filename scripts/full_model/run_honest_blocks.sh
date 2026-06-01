#!/bin/bash
#SBATCH --job-name=honest-blk
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/honest_blk_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/honest_blk_%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

export CONFIG_FILE_PATH="$PROJECT_ROOT/config/old_dataset/full_model/full_model_optim.json"
export AUTOSCHEDULER_IMPL="rl_autoschedular_v4_5"
export EXEC_TIMEOUT=300
export EXEC_TIMEOUT_CMD=7200

cd "$PROJECT_ROOT"

echo "Honest block speedup — $(date)"
echo "Models: albert bart bert convnext_tiny deberta densenet121 distilbert gat gpt2 vit_b_16"

python scripts/data/honest_blocks_speedup.py \
    --models albert bart bert convnext_tiny deberta densenet121 distilbert gat gpt2 vit_b_16 \
    --output results/full_model/honest_blocks.json

echo "Done at $(date)"
