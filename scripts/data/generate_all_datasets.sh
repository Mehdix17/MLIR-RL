#!/usr/bin/env bash
# generate_all_datasets.sh
# -------------------------
# Run the full complementary benchmark generation pipeline.
#
# Part 1: Synthetic benchmarks (continuing legacy numbering)
# Part 2: NN model extraction (single ops + bench blocks)
#
# Usage:
#   source .env && bash scripts/generate_all_datasets.sh [--synthetic-only] [--nn-only]
#   or use set -a; source .env; set +a before running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODE="${1:-all}"

# --- Helper: check env vars ---
check_env() {
    local missing=()
    for var in LLVM_BUILD_PATH MLIR_SHARED_LIBS AST_DUMPER_BIN_PATH CONFIG_FILE_PATH; do
        if [ -z "${!var:-}" ]; then
            missing+=("$var")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing env vars: ${missing[*]}"
        echo "Run: set -a; source .env; set +a"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Part 1: Synthetic benchmarks
# ---------------------------------------------------------------------------
generate_synthetic() {
    local num_single="${1:-500}"
    local num_bench="${2:-500}"
    local seed="${3:-42}"

    echo "========================================"
    echo "Part 1: Synthetic Benchmarks"
    echo "  Singles: $num_single"
    echo "  Benches: $num_bench"
    echo "  Seed:    $seed"
    echo "========================================"

    python data_utils/generate_synthetic.py \
        --num-single "$num_single" \
        --num-bench "$num_bench" \
        --seed "$seed"
}

# ---------------------------------------------------------------------------
# Part 2: NN model extraction
# ---------------------------------------------------------------------------
extract_nn_models() {
    local raw_dir="${PROJECT_ROOT}/data/nn/raw_bench"
    local single_out="${PROJECT_ROOT}/data/nn/code_files/single_bench"
    local bench_out="${PROJECT_ROOT}/data/nn/code_files/bench"

    echo "========================================"
    echo "Part 2: NN Model Extraction"
    echo "  Raw dir:       $raw_dir"
    echo "  Single output: $single_out"
    echo "  Bench output:  $bench_out"
    echo "========================================"

    # Ensure output dirs exist
    mkdir -p "$single_out" "$bench_out"

    local models=()
    local count=0

    # Collect model names from _linalg.mlir files
    for f in "$raw_dir"/*_linalg.mlir; do
        if [ -f "$f" ]; then
            local model_name
            model_name="$(basename "$f" _linalg.mlir)"
            models+=("$model_name")
        fi
    done

    if [ ${#models[@]} -eq 0 ]; then
        echo "No _linalg.mlir files found in $raw_dir"
        return 1
    fi

    echo "Models found: ${models[*]}"
    echo ""

    for model in "${models[@]}"; do
        echo "--- Processing: $model ---"

        # Extract single ops
        python -m data_utils.orchestrate extract \
            --input "${raw_dir}/${model}_linalg.mlir" \
            --output-dir "$single_out" \
            --model-name "$model" \
            --batch-sizes 1 \
            --dynamic-shape-policy static \
            --min-parallel-loops 2 \
            2>&1 | tail -1

        # Extract bench blocks
        python -m data_utils.orchestrate extract-blocks \
            --input "${raw_dir}/${model}_linalg.mlir" \
            --output-dir "$bench_out" \
            --model-name "$model" \
            --window-size 5 \
            --stride 3 \
            --max-depth 64 \
            --max-paths 5000 \
            --batch-policy heuristic \
            2>&1 | tail -1

        echo ""
    done

    echo "========================================"
    echo "Extraction complete!"
    echo "  Singles: $(ls "$single_out" 2>/dev/null | wc -l) files"
    echo "  Benches: $(ls "$bench_out" 2>/dev/null | wc -l) files"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
check_env

case "$MODE" in
    all)
        generate_synthetic 500 500 42
        extract_nn_models
        ;;
    synthetic-only)
        generate_synthetic 500 500 42
        ;;
    nn-only)
        extract_nn_models
        ;;
    *)
        echo "Usage: $0 [all|synthetic-only|nn-only]"
        exit 1
        ;;
esac

echo ""
echo "All done!"
