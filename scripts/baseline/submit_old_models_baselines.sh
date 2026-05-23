#!/usr/bin/env bash
# =============================================================================
# Runs get_base.py for the 10 old models to get fresh block baselines.
# Outputs to blocks_<model>_fresh.json (does NOT overwrite old blocks_*.json).
#
# Usage:  bash scripts/baseline/submit_old_models_baselines.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resources by model block count (fast: ~1.5s per block)
echo "=== Submitting get_base.py for 10 old models ==="
echo ""

submit_model() {
    local model="$1"
    local mem="$2"
    local cpus="$3"
    local walltime="$4"
    echo "  $model → sbatch --mem=${mem} --cpus-per-task=${cpus} --time=${walltime}"
    sbatch \
        --mem="${mem}" --cpus-per-task="${cpus}" --time="${walltime}" \
        --output="/scratch/mb10856/MLIR-RL/logs/blocks_fresh_%j.out" \
        --error="/scratch/mb10856/MLIR-RL/logs/blocks_fresh_%j.err" \
        "$SCRIPT_DIR/get_blocks_baseline_fresh.sh" "$model"
}

# Tiny (<30 blocks)
submit_model "densenet121"  "4G" 1 "00:10:00"
submit_model "gat"          "4G" 1 "00:10:00"

# Medium (300-600 blocks)
submit_model "convnext_tiny" "16G" 4 "01:00:00"
submit_model "bart"          "16G" 4 "01:00:00"
submit_model "distilbert"    "16G" 4 "01:00:00"
submit_model "gpt2"          "16G" 4 "01:00:00"

# Large (900-1300 blocks)
submit_model "vit_b_16"      "16G" 4 "01:30:00"
submit_model "bert"          "16G" 4 "01:30:00"
submit_model "deberta"       "16G" 4 "01:30:00"
submit_model "albert"        "16G" 4 "02:00:00"

echo ""
echo "All submitted. Monitor: squeue -u \$USER"
echo "Output → results/full_model/baselines/blocks_<model>_fresh.json"
