#!/bin/bash
#SBATCH --job-name=hpo-baseline
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpo/baseline_%A_%a.out
#SBATCH --error=logs/hpo/baseline_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-4%5
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

export PATH="${CONDA_ENV:-$HOME/.conda/envs/mlir}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_ENV:-$HOME/.conda/envs/mlir}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
mkdir -p logs/hpo baselines/temp

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=5

echo "=========================================="
echo "Baseline computation started at $(date)"
echo "Chunk: $((CHUNK_IDX + 1))/$NUM_CHUNKS"
echo "Node: $(hostname)"
echo "=========================================="

python scripts/get_base.py \
    --benchmarks-dir data/all/temp \
    --output baselines/temp/base.json \
    --implementation rl_autoschedular \
    --timeout 15 \
    --chunk-index $CHUNK_IDX \
    --num-chunks $NUM_CHUNKS

echo "Baseline computation completed at $(date)"
