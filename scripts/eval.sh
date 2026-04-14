#!/bin/bash
#SBATCH --job-name=mlir-eval
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/eval.sh <config>
#   sbatch scripts/eval.sh <config> [implementation]
#   scripts/submit_and_monitor.sh scripts/eval.sh config/train1.json
#
# Arguments:
#   $1  CONFIG  - path to config JSON; results_dir inside it determines which
#                <old_agent|new_agent>/run_N/models folder to evaluate
#                (latest run with checkpoints)
#   $2  implementation - autoscheduler package name (default: rl_autoschedular)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLURM_SUBMIT_DIR is set by Slurm to the directory where sbatch was called.
# Fall back to deriving from BASH_SOURCE only for local (non-Slurm) runs.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Slurm nodes start with a stripped PATH; restore standard utilities before activating the venv
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:?Usage: $0 <config>}"
IMPLEMENTATION="${2:-${AUTOSCHEDULER_IMPL:-rl_autoschedular}}"
export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
export CONFIG_FILE_PATH="$CONFIG"

# Derive EVAL_DIR: latest run_N/models inside results_dir that contains .pt files
RESULTS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['results_dir'])")
case "$IMPLEMENTATION" in
    rl_autoschedular)
        AGENT_SUBDIR="old_agent"
        ;;
    new_rl_autoschedular)
        AGENT_SUBDIR="new_agent"
        ;;
    *)
        echo "Unsupported AUTOSCHEDULER_IMPL: $IMPLEMENTATION"
        exit 1
        ;;
esac
AGENT_ROOT="$RESULTS_DIR/$AGENT_SUBDIR"
LATEST_MODELS=$(python3 -c "
import os
r = '$AGENT_ROOT'
if not os.path.isdir(r):
    raise SystemExit(f'Agent run directory not found: {r}')
runs = sorted([d for d in os.listdir(r) if d.startswith('run_') and d.split('_')[-1].isdigit()], key=lambda x: int(x.split('_')[1]))
candidates = [os.path.join(r, d, 'models') for d in runs]
candidates = [p for p in candidates if os.path.isdir(p) and any(f.endswith('.pt') for f in os.listdir(p))]
print(candidates[-1]) if candidates else exit(1)
")
export EVAL_DIR="$LATEST_MODELS"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Evaluation started at $(date)"
echo "Eval dir:  $EVAL_DIR"
echo "Config:   $CONFIG_FILE_PATH"
echo "Implementation: $AUTOSCHEDULER_IMPL"
echo "Node:     $(hostname)"
echo "=========================================="

python scripts/eval.py

echo "Evaluation completed at $(date)"
