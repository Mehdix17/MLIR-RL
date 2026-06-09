#!/bin/bash
#SBATCH --job-name=baseline_gen
#SBATCH --partition=compute
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_gen_%j.out
#SBATCH --error=logs/baseline_gen_%j.err

source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export PYTHONPATH=/scratch/mb10856/MLIR-RL:/scratch/mb10856/MLIR-RL/rl_autoschedular:$PYTHONPATH

mkdir -p logs results/single_ops_dataset_results/baselines/mlir

CHUNK_INDEX=${1:-0}
NUM_CHUNKS=${2:-1}

python scripts/baseline/get_base.py \
    --benchmarks-dir data/single_ops_dataset/all \
    --output results/single_ops_dataset_results/baselines/mlir/base.json \
    --implementation rl_autoschedular_v0 \
    --timeout 30 \
    --chunk-index "$CHUNK_INDEX" \
    --num-chunks "$NUM_CHUNKS"
