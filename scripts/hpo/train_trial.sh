#!/bin/bash
#SBATCH --job-name=hpo-train
#SBATCH --partition=compute
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --constraint=bergamo
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/hpo/train_%x_%j.out
#SBATCH --error=logs/hpo/train_%x_%j.err
#SBATCH --mail-type=END,FAIL
set -e
trap 'echo "HPO TRAINING FAILED"' ERR
#
# Usage:
#   TRIAL_ID=0 sbatch scripts/hpo/train_trial.sh
#
# Requires TRIAL_ID env var. Config must exist at
# scripts/hpo/trials/trial_${TRIAL_ID}/config.json
#
# Auto-resumes if the trial directory already has model checkpoints.

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

# Patch benchmarks_folder_path to point to train data
python3 -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['benchmarks_folder_path'] = 'data/all/train'
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
"

TRIAL_DIR="$PROJECT_ROOT/results/hpo/trial_${TRIAL_ID}"

# Auto-resume: if models already exist, resume from latest checkpoint
if [[ -d "$TRIAL_DIR/models" ]] && ls "$TRIAL_DIR"/models/model_*.pt 1>/dev/null 2>&1; then
    export RESUME_FROM="$TRIAL_DIR"
    echo "Auto-resuming from: $RESUME_FROM"
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

CONFIG_IMPL=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('implementation', ''))")
export AUTOSCHEDULER_IMPL="${CONFIG_IMPL:-rl_autoschedular_paper_transformer}"
export CONFIG_FILE_PATH="$CONFIG"

cd "$PROJECT_ROOT"
mkdir -p logs/hpo

echo "=========================================="
echo "HPO Training started at $(date)"
echo "Trial:     $TRIAL_ID"
echo "Config:    $CONFIG_FILE_PATH"
echo "Impl:      $AUTOSCHEDULER_IMPL"
[[ -n "${RESUME_FROM:-}" ]] && echo "Resume:    $RESUME_FROM"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-interactive}"
echo "=========================================="

python scripts/train/train.py

echo "HPO Training completed at $(date)"
