#!/bin/bash
#SBATCH --job-name=hpo-eval
#SBATCH --partition=compute
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --constraint=bergamo
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpo/eval_%x_%j.out
#SBATCH --error=logs/hpo/eval_%x_%j.err
#SBATCH --mail-type=END,FAIL
set -e
trap 'echo "HPO EVALUATION FAILED"' ERR
#
# Usage:
#   TRIAL_ID=0 sbatch scripts/hpo/eval_trial.sh
#
# Requires TRIAL_ID env var. Config must exist at
# scripts/hpo/trials/trial_${TRIAL_ID}/config.json
#
# Evaluates only the last checkpoint (EVAL_LAST_ONLY=1).

if [[ -z "${TRIAL_ID:-}" ]]; then
    echo "ERROR: TRIAL_ID environment variable is not set"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CONFIG="$PROJECT_ROOT/scripts/hpo/trials/trial_${TRIAL_ID}/config.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Patch benchmarks_folder_path to point to eval data
python3 -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['benchmarks_folder_path'] = 'data/all/eval'
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
"

TRIAL_DIR="$PROJECT_ROOT/results/hpo/trial_${TRIAL_ID}"
export EVAL_DIR="$TRIAL_DIR/rl_autoschedular_paper_transformer_agent/run_0/models"
export EVAL_STRIDE=${EVAL_STRIDE:-100}
export FORCE_RUN_ID="trial_${TRIAL_ID}"

if [[ ! -d "$EVAL_DIR" ]] || ! ls "$EVAL_DIR"/model_*.pt 1>/dev/null 2>&1; then
    echo "ERROR: No model checkpoints found in $EVAL_DIR"
    exit 1
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

export PATH="${CONDA_ENV:-$HOME/.conda/envs/mlir}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_ENV:-$HOME/.conda/envs/mlir}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export MIN_EXEC_TIMEOUT=${MIN_EXEC_TIMEOUT:-300}

CONFIG_IMPL=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('implementation', ''))")
export AUTOSCHEDULER_IMPL="${CONFIG_IMPL:-rl_autoschedular_paper_transformer}"
export CONFIG_FILE_PATH="$CONFIG"

cd "$PROJECT_ROOT"
mkdir -p logs/hpo

echo "=========================================="
echo "HPO Evaluation started at $(date)"
echo "Trial:     $TRIAL_ID"
echo "Config:    $CONFIG_FILE_PATH"
echo "Impl:      $AUTOSCHEDULER_IMPL"
echo "EVAL_DIR:  $EVAL_DIR"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-interactive}"
echo "=========================================="

python scripts/eval/eval.py

echo "HPO Evaluation completed at $(date)"
