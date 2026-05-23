#!/bin/bash
#SBATCH --job-name=blk-v10
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/blk_v10_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/blk_v10_%A_%a.err

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

MODEL="$1"
CHECKPOINT_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"

export CONFIG_FILE_PATH="$PROJECT_ROOT/config/full_model_optim.json"
export AUTOSCHEDULER_IMPL="rl_autoschedular_v4_5"

echo "=========================================="
echo "Block v10 optimization started at $(date)"
echo "Model:       $MODEL"
echo "Checkpoint: 1999"
echo "Node:        $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"

OUTPUT="results/full_model/blocks_v10_${MODEL}.json"
python scripts/optimize_model_via_blocks.py \
    --checkpoints \
        "$CHECKPOINT_DIR/model_1999.pt" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --window-size 5 \
    --stride 3 \
    --skip-extraction

echo "Completed at $(date)"
echo "Results → $OUTPUT"
