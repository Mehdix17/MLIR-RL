#!/bin/bash
#SBATCH --job-name=mlir-base
#SBATCH --partition=compute
#SBATCH -C dalma
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/mlir_base_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/mlir_base_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Usage: sbatch --array=0-18 scripts/baseline/get_blocks_baseline.sh
# 19 models: all in data/nn/blocks/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

MODELS=(
    "albert"          "bart"            "bert"
    "convnext_tiny"   "deberta"         "densenet121"
    "distilbert"      "efficientnet_b0" "gat"
    "gcn"             "gpt2-large"
    "lstm"            "mobilenet_v3_small" "resnet18"
    "resnet50"        "resnext50"       "t5"
    "vgg11"           "vit_b_16"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"
BLOCKS_DIR="$PROJECT_ROOT/data/nn/blocks/${MODEL}"
OUTPUT_DIR="$PROJECT_ROOT/results/full_model_dalma/baselines"
OUTPUT="${OUTPUT_DIR}/blocks_${MODEL}.json"
mkdir -p "$OUTPUT_DIR"

BLOCK_COUNT=$(ls "$BLOCKS_DIR"/*.mlir 2>/dev/null | wc -l)

echo "=========================================="
echo "MLIR Baseline — $(date)"
echo "Cluster:  jubail"
echo "Task:     $TASK_ID / ${#MODELS[@]}"
echo "Model:    $MODEL  ($BLOCK_COUNT blocks)"
echo "Output:   $OUTPUT"
echo "Node:     $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"

python scripts/baseline/get_base.py \
    --benchmarks-dir "$BLOCKS_DIR" \
    --output "$OUTPUT" \
    --implementation rl_autoschedular_v4_5 \
    --timeout 15

python3 - <<PY
import json
with open("$OUTPUT") as f:
    d = json.load(f)
valid = sum(1 for v in d.values() if isinstance(v, (int,float)) and v > 0)
failed = sum(1 for v in d.values() if isinstance(v, (int,float)) and v <= 0)
total = sum(v for v in d.values() if isinstance(v, (int,float)) and v > 0)
print(f"Result: {valid} valid / {len(d)} total  ({failed} failed)")
print(f"Sum: {total:,} ns ({total/1e6:.2f} ms)")
PY

echo "Completed at $(date)"
