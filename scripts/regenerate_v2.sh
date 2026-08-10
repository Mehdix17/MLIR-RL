#!/usr/bin/env bash
# regenerate_v2.sh
# ----------------
# Regenerate NN single-op benchmarks with relaxed filters,
# plus more synthetic benchmarks, and harvest tmp/ files.
#
# Usage: source .env && bash scripts/regenerate_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RAW_DIR="$PROJECT_ROOT/data/nn/raw_bench"
SINGLE_OUT="$PROJECT_ROOT/data/nn/code_files/single_bench"
BENCH_OUT="$PROJECT_ROOT/data/nn/code_files/bench"

echo "============================================="
echo "Phase 1: Clean existing NN single-op files"
echo "============================================="
rm -f "$SINGLE_OUT"/*.mlir
echo "Cleaned $SINGLE_OUT/"

echo ""
echo "============================================="
echo "Phase 2: NN single-op extraction (relaxed filters)"
echo "============================================="

for f in "$RAW_DIR"/*_linalg.mlir; do
    model="$(basename "$f" _linalg.mlir)"
    echo "--- $model ---"
    python -m data_utils.orchestrate extract \
        --input "$f" \
        --output-dir "$SINGLE_OUT" \
        --model-name "$model" \
        --batch-sizes 1 \
        --dynamic-shape-policy static \
        2>&1 | grep -E "Written|Skipped|Op breakdown"
done

echo ""
echo "Phase 2 complete: $(ls "$SINGLE_OUT" 2>/dev/null | wc -l) single-op files"

echo ""
echo "============================================="
echo "Phase 3: Harvest tmp/ model files"
echo "============================================="
# Copy model subdirectories (not block_smoke which are test files)
for d in "$PROJECT_ROOT/tmp"/*/; do
    dname="$(basename "$d")"
    case "$dname" in
        block_smoke|block_smoke_manifest|onnx-importer-temp) continue ;;
    esac
    count=$(ls "$d"*.mlir 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        cp "$d"*.mlir "$SINGLE_OUT"/ 2>/dev/null || true
        echo "  $dname: $count files → $SINGLE_OUT/"
    fi
done

TOTAL_SINGLES=$(ls "$SINGLE_OUT" 2>/dev/null | wc -l)
echo "Phase 3 complete: $TOTAL_SINGLES total single-op files"

echo ""
echo "============================================="
echo "Phase 4: Synthetic generation (10k singles, 10k benches)"
echo "============================================="

python data_utils/generate_synthetic.py \
    --num-single 10000 \
    --num-bench 10000 \
    --seed 12345

echo ""
echo "============================================="
echo "ALL DONE"
echo "============================================="
echo "  NN singles:   $(ls "$SINGLE_OUT" 2>/dev/null | wc -l) files"
echo "  NN benches:   $(ls "$BENCH_OUT" 2>/dev/null | wc -l) files"
echo "  Syn singles:  $(ls "$PROJECT_ROOT/data/all/code_files/single_bench/" | wc -l) files"
echo "  Syn benches:  $(ls "$PROJECT_ROOT/data/all/code_files/bench/" | wc -l) files"
