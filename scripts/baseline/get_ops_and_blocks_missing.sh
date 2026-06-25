#!/bin/bash
#SBATCH --job-name=mlir-base-obs
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/mlir_base_obs_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/mlir_base_obs_%A_%a.err
#
# Usage: sbatch --array=0-9 scripts/baseline/get_ops_and_blocks_missing.sh
# 10 chunks, ~74 files each

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

source ~/envs/mlir/bin/activate
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_CHUNKS=10

BLOCKS_DIR="$PROJECT_ROOT/data/ops_and_blocks/missing_baselines"
OUTPUT="$PROJECT_ROOT/results/ops_and_blocks_results/baselines/mlir/missing_chunk${TASK_ID}.json"

mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================="
echo "Missing baselines — $(date)"
echo "Chunk:    $TASK_ID / $NUM_CHUNKS"
echo "Input:    $BLOCKS_DIR"
echo "Output:   $OUTPUT"
echo "=========================================="

cd "$PROJECT_ROOT"

python scripts/baseline/get_base.py \
    --benchmarks-dir "$BLOCKS_DIR" \
    --output "$OUTPUT" \
    --implementation rl_autoschedular_v4_5 \
    --timeout 15 \
    --chunk-index "$TASK_ID" \
    --num-chunks "$NUM_CHUNKS"

echo "Completed at $(date)"
