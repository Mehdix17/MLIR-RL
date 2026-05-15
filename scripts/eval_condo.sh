#!/bin/bash
#SBATCH -q c2
#SBATCH --job-name=mlir-eval
#SBATCH --partition=condo
#SBATCH --mem=32G
#SBATCH --cpus-per-task=28
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/eval.sh                           # array mode: auto-picks version
#   sbatch scripts/eval.sh config/baseline.json       # single version from config
#   sbatch scripts/eval.sh config/baseline.json v1    # explicit version
#   sbatch scripts/eval.sh config/baseline.json v1 v2 v3  # space-separated versions
#
# Array mode: sbatch --array=0-2 scripts/eval.sh     # v1, v2, v3

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

# Version resolution: array task > positional args > config file
ALL_VERSIONS=(baseline v1 v2 v3 v4)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    VERSIONS=("${ALL_VERSIONS[$SLURM_ARRAY_TASK_ID]}")
elif [[ $# -ge 2 ]]; then
    CONFIG_FILE="$1"
    shift
    VERSIONS=("$@")
else
    CONFIG_FILE="${1:-$PROJECT_ROOT/config/baseline.json}"
    VERSIONS=()
fi

CONFIG="${CONFIG_FILE:-$PROJECT_ROOT/config/baseline.json}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

if [[ ${#VERSIONS[@]} -eq 0 ]]; then
    CONFIG_IMPL=$(python3 - <<PY
import json
try:
    with open("$CONFIG", "r") as f:
        print((json.load(f).get("implementation") or "").strip())
except Exception:
    print("")
PY
    )
    IMPL="${CONFIG_IMPL:-rl_autoschedular}"
    VERSIONS=("${IMPL#rl_autoschedular_}")
    [[ "$VERSIONS" == "rl_autoschedular" ]] && VERSIONS=("baseline")
fi

for VERSION in "${VERSIONS[@]}"; do
    IMPLEMENTATION="rl_autoschedular_${VERSION}"
    [[ "$VERSION" == "baseline" ]] && IMPLEMENTATION="rl_autoschedular"
    export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
    export CONFIG_FILE_PATH="$CONFIG"

    echo "=========================================="
    echo "Evaluation started at $(date)"
    echo "Version:  $VERSION ($IMPLEMENTATION)"
    echo "Config:   $CONFIG_FILE_PATH"
    echo "Node:     $(hostname)"
    echo "=========================================="

    # Derive EVAL_DIR: latest run_N/models that contains .pt files
    mapfile -t EVAL_META < <(python3 - <<PY
import json
from utils.implementation import get_agent_subdir
cfg = json.load(open("$CONFIG"))
print(cfg["results_dir"])
print(get_agent_subdir("$IMPLEMENTATION"))
PY
)
    RESULTS_DIR="${EVAL_META[0]}"
    AGENT_SUBDIR="${EVAL_META[1]}"
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
    export EVAL_LAST_ONLY=1
    python scripts/eval.py

    echo "Evaluation completed at $(date)"
    echo ""
done
