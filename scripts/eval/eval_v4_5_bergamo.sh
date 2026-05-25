#!/bin/bash
#SBATCH --job-name=v45-bergamo
#SBATCH --partition=compute
#SBATCH -C bergamo
#SBATCH --mem=300G
#SBATCH --cpus-per-task=128
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/v45_bergamo_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/v45_bergamo_%j.err
#SBATCH --mail-type=END,FAIL
#
# V4.5 evaluation on Bergamo (all 14 checkpoints: 700-1999)
# 128 threads → ~2 min per checkpoint → ~30 min total

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$PROJECT_ROOT/config/eval_bergamo/v4_5_eval_bergamo.json}"
export EVAL_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"
export EVAL_START="${EVAL_START:-0}"
export EVAL_END="${EVAL_END:-999999}"
export FORCE_RUN_ID="${FORCE_RUN_ID:-bergamo}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_ROOT"

echo "=========================================="
echo "V4.5 Bergamo Eval — $(date)"
echo "Cluster:   bergamo (AMD Zen 4, 256 cores, 1 TB)"
echo "CPUs:      128"
echo "Checkpoints: 700-1999 (stride 100)"
echo "Benchmarks: 3014 (base_eval.json)"
echo "Output:    results/V4_5_agent_bergamo/"
echo "Node:      $(hostname)"
echo "=========================================="

python scripts/eval/eval.py

echo ""
echo "Completed at $(date)"
