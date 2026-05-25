#!/bin/bash
#SBATCH --job-name=blocks-fresh
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#
# Called by: sbatch --mem=X --cpus-per-task=N --time=H:M:S scripts/baseline/get_blocks_baseline_fresh.sh <model>
#
# Outputs: results/full_model/baselines/blocks_<model>_fresh.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

MODEL="$1"
[[ -z "$MODEL" ]] && { echo "Usage: sbatch ... get_blocks_baseline_fresh.sh <model>"; exit 1; }

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

BLOCKS_DIR="$PROJECT_ROOT/data/nn/blocks/${MODEL}"
OUTPUT_DIR="$PROJECT_ROOT/results/full_model/baselines"
OUTPUT="${OUTPUT_DIR}/blocks_${MODEL}_fresh.json"
mkdir -p "$OUTPUT_DIR"

BLOCK_COUNT=$(ls "$BLOCKS_DIR"/*.mlir 2>/dev/null | wc -l)

echo "=========================================="
echo "Block baseline (fresh) — $(date)"
echo "Model:   $MODEL  ($BLOCK_COUNT blocks)"
echo "Output:  $OUTPUT"
echo "Node:    $(hostname)"
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
total_baseline = sum(v for v in d.values() if isinstance(v, (int,float)) and v > 0)
print(f"Result: {valid} valid / {len(d)} total  ({failed} failed)")
print(f"Baseline sum: {total_baseline:,} ns ({total_baseline/1e6:.2f} ms)")
PY

echo "Completed at $(date)"
