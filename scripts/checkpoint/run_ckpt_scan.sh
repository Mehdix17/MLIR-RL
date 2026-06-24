#!/bin/bash
#SBATCH --job-name=ckpt-scan
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/ckpt_scan_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/ckpt_scan_%j.err
#
# Checkpoint scan: block-based optimization across checkpoints 700-1999 (every 100).
# Usage: sbatch scripts/run_ckpt_scan.sh <model_name>

set -e

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Usage: sbatch $0 <model_name>"
    exit 1
fi

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

CKPT_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"

CKPTS=()
for n in $(seq 700 100 1900) 1999; do
    CKPTS+=("$CKPT_DIR/model_${n}.pt")
done

OUTPUT="$PROJECT_ROOT/results/full_model/scan/${MODEL}.json"
mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================="
echo "Checkpoint scan started at $(date)"
echo "Model:       $MODEL"
echo "Checkpoints: ${#CKPTS[@]} (700-1999 every 100)"
echo "Node:        $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"

python scripts/full_model/optimize_model_via_blocks.py \
    --checkpoints "${CKPTS[@]}" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --window-size 5 \
    --stride 3 \
    --skip-extraction

echo "Completed at $(date)"
echo "Results → $OUTPUT"
