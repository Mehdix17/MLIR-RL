#!/bin/bash
#SBATCH --job-name=newds-pt-eval
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/newds_pt_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/newds_pt_%A_%a.err
#SBATCH --array=0-0
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/baseline/get_new_dataset_pytorch.sh data/new_dataset/all/eval eval_pytorch
#   sbatch scripts/baseline/get_new_dataset_pytorch.sh data/new_dataset/all/eval_full eval_full_pytorch
#   sbatch scripts/baseline/get_new_dataset_pytorch.sh data/new_dataset/all/train train_pytorch
#
# First arg: path relative to PROJECT_ROOT containing .mlir files
# Second arg: output filename prefix (output: results/new_dataset_results/baselines/pytorch/<prefix>.json)

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

cd "$PROJECT_ROOT"

BENCHMARKS_DIR="${1:?Usage: sbatch scripts/baseline/get_new_dataset_pytorch.sh <benchmarks-dir> <output-name>}"
OUTPUT_NAME="${2:?Usage: sbatch scripts/baseline/get_new_dataset_pytorch.sh <benchmarks-dir> <output-name>}"

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-5}
OUTPUT_DIR="$PROJECT_ROOT/results/new_dataset_results/baselines/pytorch"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "PyTorch baseline — ${OUTPUT_NAME}"
echo "Started at:      $(date)"
echo "Node:            $(hostname)"
echo "Chunk:           $((CHUNK_IDX + 1)) / $NUM_CHUNKS"
echo "Benchmarks dir:  $BENCHMARKS_DIR"
echo "Output:          $OUTPUT_DIR/${OUTPUT_NAME}.json"
echo "=========================================="

python scripts/baseline/get_pytorch_times.py \
    --benchmarks-dir "$PROJECT_ROOT/$BENCHMARKS_DIR" \
    --output "$OUTPUT_DIR/${OUTPUT_NAME}.json" \
    --chunk-index $CHUNK_IDX \
    --num-chunks $NUM_CHUNKS

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "To merge chunks after all array tasks finish:"
echo "  python3 -c \"import json,glob; m={}; [m.update(json.load(open(f))) for f in sorted(glob.glob('${OUTPUT_DIR}/${OUTPUT_NAME}_chunk*.json'))]; print(f'Merged {len(m)} benchmarks'); json.dump(m, open('${OUTPUT_DIR}/${OUTPUT_NAME}.json','w'), indent=2)\""
