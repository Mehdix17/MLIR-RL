#!/bin/bash
#SBATCH --job-name=newds-base-train
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/newds_base_train_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/newds_base_train_%A_%a.err
#SBATCH --array=0-19
#
# Usage:
#   sbatch scripts/baseline/get_new_dataset_base.sh
#
# Processes 9,407 files from data/new_dataset/all/train/ across 20 array tasks.
# Each chunk writes train_base_chunk{N}.json.
# Merge separately after all chunks complete:
#   python3 -c "import json,glob; ..." (see end of script)

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

export CONFIG_FILE_PATH="$PROJECT_ROOT/config/new_dataset/train/v4_5.json"
export AUTOSCHEDULER_IMPL="rl_autoschedular_v4_5"

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-20}

echo "=========================================="
echo "New dataset baseline — TRAIN"
echo "Started at:      $(date)"
echo "Node:            $(hostname)"
echo "Chunk:           $((CHUNK_IDX + 1)) / $NUM_CHUNKS"
echo "Config:          $CONFIG_FILE_PATH"
echo "=========================================="

python scripts/baseline/get_base.py \
    --config "$CONFIG_FILE_PATH" \
    --output "$PROJECT_ROOT/results/new_dataset_results/baselines/exec_times/train_base.json" \
    --implementation "$AUTOSCHEDULER_IMPL" \
    --timeout 15 \
    --chunk-index $CHUNK_IDX \
    --num-chunks $NUM_CHUNKS

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "To merge chunks after all array tasks finish, run:"
echo "  python3 -c \"import json,glob; m={}; [m.update(json.load(open(f))) for f in sorted(glob.glob('results/new_dataset_results/baselines/exec_times/train_base_chunk*.json'))]; print(f'Merged {len(m)} benchmarks'); json.dump(m, open('results/new_dataset_results/baselines/exec_times/train_base.json','w'), indent=2)\""
