#!/bin/bash
#SBATCH --job-name=mlir-get-base
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/get_base_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/get_base_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/get_base.sh <config>
#   sbatch scripts/get_base.sh <config> [implementation]
#
# Examples:
#   sbatch scripts/get_base.sh config/train1.json
#
# All paths (benchmarks dir, output JSON) are derived from the config file.
# Override individual paths with --benchmarks-dir / --output if needed.

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
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

CONFIG="${1:?Usage: sbatch scripts/get_base.sh <config>}"
IMPLEMENTATION="${2:-${AUTOSCHEDULER_IMPL:-rl_autoschedular}}"
export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi
export CONFIG_FILE_PATH="$CONFIG"

echo "=========================================="
echo "get_base.py started at $(date)"
echo "Config:  $CONFIG"
echo "Implementation: $AUTOSCHEDULER_IMPL"
echo "Node:    $(hostname)"
echo "=========================================="

python scripts/get_base.py --config "$CONFIG" --implementation "$AUTOSCHEDULER_IMPL"

echo "=========================================="
echo "get_base.py completed at $(date)"
echo "=========================================="
