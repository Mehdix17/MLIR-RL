#!/usr/bin/env bash
# =============================================================================
# MLIR-RL Full Pipeline Orchestration
# =============================================================================
# Runs the complete end-to-end pipeline for a set of RL implementations:
#   1. Base MLIR execution times (once per dataset)     → Slurm
#   2. PyTorch baseline times (once per dataset)         → local
#   3. Stratified train/eval split (once per dataset)    → local
#   4. Train each implementation (using its own config)  → Slurm
#   5. Evaluate each implementation                      → Slurm
#
# Shared steps (1-3) use the first config argument.
# Per-version steps (4-5) auto-derive config/old_dataset/train/<version>.json from each
# implementation name.  Example: rl_autoschedular_v1 → config/old_dataset/train/v1.json.
#
# Failures in one implementation do NOT stop the pipeline — remaining
# versions continue.  Progress is logged to a timestamped file in
# pipeline_logs/.
#
# Usage:
#   bash scripts/pipeline.sh config/old_dataset/train/baseline.json
#   bash scripts/pipeline.sh config/old_dataset/train/baseline.json "rl_autoschedular_v0,rl_autoschedular_v1,rl_autoschedular_v3"
#   bash scripts/pipeline.sh config/old_dataset/train/baseline.json "rl_autoschedular_v3"
# =============================================================================

BASE_CONFIG="${1:?Usage: pipeline.sh <base_config_path> [implementations]}"
IMPL_FILTER="${2:-}"

# Make base_config absolute
if [[ "$BASE_CONFIG" != /* ]]; then
    BASE_CONFIG="$PWD/$BASE_CONFIG"
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ~/envs/mlir/bin/activate
set -a && source .env && set +a

# .env may replace PATH — ensure conda python + slurm + standard utilities
export PATH="$HOME/envs/mlir/bin:/opt/slurm/default/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p pipeline_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="pipeline_logs/pipeline_${TIMESTAMP}.log"
STATUSFILE="pipeline_logs/pipeline_${TIMESTAMP}.status"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== MLIR-RL Pipeline ==="
echo "Started:      $(date)"
echo "Base config:  $BASE_CONFIG"
echo "Log:          $LOGFILE"
echo ""

# ---------------------------------------------------------------------------
# Config helper — derive per-version config path from implementation name
# ---------------------------------------------------------------------------
resolve_impl_config() {
    local impl="$1"
    # Strip rl_autoschedular_ prefix → v0, v1, v2, v3, ...
    # rl_autoschedular_v0 uses config/old_dataset/train/baseline.json (or BASE_CONFIG fallback)
    local token
    if [[ "$impl" == "rl_autoschedular" ]]; then
        echo "WARNING: 'rl_autoschedular' renamed to 'rl_autoschedular_v0'" >&2
        token="baseline"
    elif [[ "$impl" == "rl_autoschedular_v0" ]]; then
        token="baseline"
    elif [[ "$impl" == new_rl_autoschedular ]]; then
        token="new"
    else
        token="${impl#rl_autoschedular_}"
    fi

    local candidate="config/old_dataset/train/${token}.json"
    if [[ -z "$token" ]] || [[ ! -f "$candidate" ]]; then
        echo "  [resolve] $impl → fallback to $BASE_CONFIG" | tee -a "$LOGFILE"
        echo "$BASE_CONFIG"
    else
        echo "  [resolve] $impl → $candidate" | tee -a "$LOGFILE"
        echo "$PROJECT_ROOT/$candidate"
    fi
}

# ---------------------------------------------------------------------------
# Determine which implementations to run
# ---------------------------------------------------------------------------
if [[ -z "$IMPL_FILTER" ]]; then
    # Read from the "implementations" list in the config
    IMPL_LIST=$(python -c "
import json
with open('$BASE_CONFIG') as f:
    cfg = json.load(f)
impls = cfg.get('implementations', cfg.get('implementation', 'rl_autoschedular_v0'))
if isinstance(impls, str):
    impls = [impls]
print(','.join(impls))
" 2>/dev/null || echo "rl_autoschedular_v0")
else
    IMPL_LIST="$IMPL_FILTER"
fi

IFS=',' read -ra IMPLS <<< "$IMPL_LIST"

echo "Implementations to process:"
for impl in "${IMPLS[@]}"; do
    IMPL_CONFIG=$(resolve_impl_config "$impl")
    echo "  $impl → $(basename "$IMPL_CONFIG")"
done
echo ""

# ---------------------------------------------------------------------------
# Status tracking helpers
# ---------------------------------------------------------------------------
init_status() {
    echo "{}" > "$STATUSFILE"
}

mark_status() {
    local key="$1" value="$2"
    python3 -c "
import json, sys
try:
    with open('$STATUSFILE') as f:
        d = json.load(f)
except Exception:
    d = {}
d['$key'] = '$value'
with open('$STATUSFILE', 'w') as f:
    json.dump(d, f, indent=2)
" 2>/dev/null || true
}

init_status
mark_status "started" "$TIMESTAMP"

# ===========================================================================
# STEP 1: MLIR baseline (once, shared)
# ===========================================================================
echo "──────────── Step 1: MLIR baseline ────────────"

# Derive results_dir from the base config to check for existing baseline
RESULTS_DIR=$(python -c "import json; print(json.load(open('$BASE_CONFIG'))['results_dir'])" 2>/dev/null || echo "")
BASE_JSON="${RESULTS_DIR:-results}/exec_times/base.json"

if [[ -f "$BASE_JSON" ]]; then
    echo "  ✓ Already exists: $BASE_JSON"
    mark_status "step1_mlir_base" "skipped_exists"
else
    echo "  Submitting get_base job..."
    BASE_JOB=$(sbatch --job-name="base_${TIMESTAMP}" \
                      --parsable \
                      --output="logs/base_${TIMESTAMP}_%j.out" \
                      --error="logs/base_${TIMESTAMP}_%j.err" \
                      scripts/baseline/get_base.sh "$BASE_CONFIG" 2>&1) || true
    if [[ "$BASE_JOB" =~ ^[0-9]+$ ]]; then
        echo "  ✓ Submitted: Job $BASE_JOB"
        mark_status "step1_mlir_base" "submitted_${BASE_JOB}"
    else
        echo "  ✗ Failed to submit. Error: $BASE_JOB"
        mark_status "step1_mlir_base" "failed"
    fi
fi

# ===========================================================================
# STEP 2: PyTorch baselines (once, local)
# ===========================================================================
echo ""
echo "──────────── Step 2: PyTorch baselines ────────────"
PYTORCH_JSON="${RESULTS_DIR:-results}/exec_times/pytorch.json"

if [[ -f "$PYTORCH_JSON" ]]; then
    echo "  ✓ Already exists: $PYTORCH_JSON"
    mark_status "step2_pytorch" "skipped_exists"
else
    echo "  Running get_pytorch_times.py..."
    if python scripts/baseline/get_pytorch_times.py --config "$BASE_CONFIG"; then
        echo "  ✓ Done"
        mark_status "step2_pytorch" "completed"
    else
        echo "  ✗ Failed — you can rerun after get_base completes:"
        echo "    python scripts/baseline/get_pytorch_times.py --config $BASE_CONFIG"
        mark_status "step2_pytorch" "failed"
    fi
fi

# ===========================================================================
# STEP 3: Train/eval split (once)
# ===========================================================================
echo ""
echo "──────────── Step 3: Train/eval split ────────────"
SPLIT_INPUT="${RESULTS_DIR:-results}/exec_times/base.json"
SPLIT_JSON="${RESULTS_DIR:-results}/exec_times/base_train.json"

if [[ -f "$SPLIT_JSON" ]]; then
    echo "  ✓ Already exists: $SPLIT_JSON"
    mark_status "step3_split" "skipped_exists"
elif [[ -f "$SPLIT_INPUT" ]]; then
    echo "  Running split_json.py..."
    if python scripts/data/split_json.py --config "$BASE_CONFIG"; then
        echo "  ✓ Done"
        mark_status "step3_split" "completed"
    else
        echo "  ✗ Failed"
        mark_status "step3_split" "failed"
    fi
else
    echo "  ⚠ base.json not ready yet (get_base still running)."
    echo "    Run this after the get_base Slurm job finishes:"
    echo "      python scripts/data/split_json.py --config $BASE_CONFIG"
    mark_status "step3_split" "waiting_on_get_base"
fi

# ===========================================================================
# STEPS 4-5: Train + eval each implementation
# ===========================================================================
TOTAL=${#IMPLS[@]}
VERSION_INDEX=0

for impl in "${IMPLS[@]}"; do
    VERSION_INDEX=$((VERSION_INDEX + 1))
    IMPL_CONFIG=$(resolve_impl_config "$impl")
    IMPL_CONFIG_NAME=$(basename "$IMPL_CONFIG")

    echo ""
    echo "──────────── Step 4: Train $impl [$VERSION_INDEX/$TOTAL] ────────────"
    echo "  Config: $IMPL_CONFIG_NAME"

    TRAIN_JOB=$(sbatch --job-name="train_${impl}" \
                       --parsable \
                       --output="logs/train_${impl}_%j.out" \
                       --error="logs/train_${impl}_%j.err" \
                       scripts/train/train.sh "$IMPL_CONFIG" "$impl" 2>&1) || true

    if [[ "$TRAIN_JOB" =~ ^[0-9]+$ ]]; then
        echo "  ✓ Submitted: Job $TRAIN_JOB"
        mark_status "train_${impl}" "submitted_${TRAIN_JOB}"

        # Submit eval dependent on train
        echo "──────────── Step 5: Eval $impl [$VERSION_INDEX/$TOTAL] ────────────"
        EVAL_JOB=$(sbatch --job-name="eval_${impl}" \
                          --parsable \
                          --dependency=afterany:"$TRAIN_JOB" \
                          --output="logs/eval_${impl}_%j.out" \
                          --error="logs/eval_${impl}_%j.err" \
                          scripts/eval/eval.sh "$IMPL_CONFIG" "$impl" 2>&1) || true

        if [[ "$EVAL_JOB" =~ ^[0-9]+$ ]]; then
            echo "  ✓ Submitted: Job $EVAL_JOB (waits for $TRAIN_JOB)"
            mark_status "eval_${impl}" "submitted_${EVAL_JOB}"
        else
            echo "  ⚠ Eval submission failed for $impl — will evaluate manually later"
            echo "    Error: $EVAL_JOB"
            mark_status "eval_${impl}" "submit_failed"
        fi
    else
        echo "  ✗ Train submission failed for $impl — skipping eval"
        echo "    Error: $TRAIN_JOB"
        mark_status "train_${impl}" "submit_failed"
        mark_status "eval_${impl}" "skipped_train_failed"
    fi
done

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo "=== Pipeline submitted ==="
echo "Log:      $LOGFILE"
echo "Status:   $STATUSFILE"
echo ""
echo "Monitor jobs:     squeue -u \$USER"
echo "Monitor pipeline: cat $STATUSFILE | python3 -m json.tool"
echo "Dashboard:        cd dashboard && streamlit run dashboard.py"
echo ""
echo "Completed at $(date)"
mark_status "completed_at" "$(date -Iseconds)"

# Print status summary
echo ""
echo "=== Submission summary ==="
python3 -c "
import json
with open('$STATUSFILE') as f:
    status = json.load(f)
for k, v in sorted(status.items()):
    print(f'  {k}: {v}')
" 2>/dev/null || echo "  (status file: $STATUSFILE)"
