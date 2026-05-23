#!/usr/bin/env bash
# =============================================================================
# Submits block baseline jobs for all 9 models missing blocks_*.json.
# Resources are auto-selected per model based on block count.
#
# Usage:  bash scripts/baseline/submit_blocks_baselines.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Tier 1: large (300+ blocks)  → 16G / 4 CPUs / 1h
TIER1_MODELS=("efficientnet_b0" "t5" "resnet50" "mobilenet_v3_small" "resnext50")
TIER1_ARGS="--mem=16G --cpus-per-task=4 --time=01:00:00"

# Tier 2: medium (30-300 blocks) → 8G / 2 CPUs / 30m
TIER2_MODELS=("resnet18")
TIER2_ARGS="--mem=8G --cpus-per-task=2 --time=00:30:00"

# Tier 3: small (<30 blocks)     → 4G / 1 CPU / 10m
TIER3_MODELS=("lstm" "gcn" "vgg11")
TIER3_ARGS="--mem=4G --cpus-per-task=1 --time=00:10:00"

echo "=== Submitting block baseline jobs ==="
echo ""

submit_tier() {
    local tier_args="$1"
    shift
    local models=("$@")
    for model in "${models[@]}"; do
        echo "  $model → sbatch $tier_args scripts/baseline/get_blocks_baseline.sh $model"
        sbatch $tier_args "$SCRIPT_DIR/get_blocks_baseline.sh" "$model"
    done
}

echo "--- Tier 1 (large: 300+ blocks) ---"
submit_tier "$TIER1_ARGS" "${TIER1_MODELS[@]}"

echo ""
echo "--- Tier 2 (medium: 30-300 blocks) ---"
submit_tier "$TIER2_ARGS" "${TIER2_MODELS[@]}"

echo ""
echo "--- Tier 3 (small: <30 blocks) ---"
submit_tier "$TIER3_ARGS" "${TIER3_MODELS[@]}"

echo ""
echo "All submitted. Monitor: squeue -u \$USER"
echo "Results → results/full_model/blocks/blocks_<model>.json"
