#!/bin/bash
#
# Quick test of comparison framework (uses test data)
# Tests with LSTM model using test config
#

#SBATCH -J test-comparison
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mb10856@nyu.edu

# Setup environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate mlir

# Change to project root
cd "${SLURM_SUBMIT_DIR}" || exit 1
PROJECT_ROOT="$(pwd)"

echo "================================================================================"
echo "Quick Test: Comparison Framework"
echo "================================================================================"
echo ""

# Configuration
MODEL_TYPE="lstm"
export CONFIG_FILE="${PROJECT_ROOT}/config/test.json"

# Find the latest test run or create new one
if [ -d "results/${MODEL_TYPE}" ]; then
    LATEST_RUN=$(find "results/${MODEL_TYPE}" -maxdepth 1 -type d -name "run_*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_RUN" ]; then
        MODEL_DIR="$LATEST_RUN"
        RUN_NUMBER=$(basename "$MODEL_DIR" | sed 's/run_//')
        echo "Using existing model: $MODEL_DIR"
    else
        RUN_NUMBER=0
        MODEL_DIR="results/${MODEL_TYPE}/run_${RUN_NUMBER}"
        echo "No existing runs found, will use: $MODEL_DIR"
    fi
else
    RUN_NUMBER=0
    MODEL_DIR="results/${MODEL_TYPE}/run_${RUN_NUMBER}"
    echo "No model directory found, will use: $MODEL_DIR"
fi

BENCHMARK_OUTPUT_DIR="${MODEL_DIR}/benchmarks"
mkdir -p "$BENCHMARK_OUTPUT_DIR"

echo "Config: ${CONFIG_FILE}"
echo "Output: ${BENCHMARK_OUTPUT_DIR}"
echo ""

# Step 1: Create benchmark suite
echo "[1/4] Creating benchmark suite..."
echo "--------------------------------------------------------------------------------"
python benchmarks/benchmark_suite.py
echo ""

# Step 2: Run PyTorch benchmarks (these are fast)
echo "[2/4] Running PyTorch Default benchmark..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_pytorch_default.py "$BENCHMARK_OUTPUT_DIR"
echo ""

echo "[3/4] Running PyTorch JIT benchmark..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_pytorch_jit.py "$BENCHMARK_OUTPUT_DIR"
echo ""

# Step 3: Check if RL model exists and run benchmark
echo "[4/4] Checking for RL model..."
echo "--------------------------------------------------------------------------------"
if [ -d "${MODEL_DIR}/models" ]; then
    echo "✓ Found RL model, running RL benchmarks..."
    python evaluation/run_rl_optimized.py "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR" || echo "⚠️  RL benchmarks failed"
else
    echo "⚠️  No RL model found at ${MODEL_DIR}/models"
    echo "   Run 'sbatch scripts/lstm/test_lstm.sh' first to train a test model"
    echo "   Or comparison will only show PyTorch Default vs JIT"
fi
echo ""

# Step 5: Generate comparison if we have outputs
echo "[5/5] Generating comparison..."
echo "--------------------------------------------------------------------------------"
if [ -f "${BENCHMARK_OUTPUT_DIR}/pytorch_output.json" ]; then
    python evaluation/compare_all.py "$BENCHMARK_OUTPUT_DIR"
    echo ""
    echo "================================================================================"
    echo "✓ TEST COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "Model: ${MODEL_TYPE} (run_${RUN_NUMBER})"
    echo "Benchmarks directory: ${BENCHMARK_OUTPUT_DIR}"
    echo ""
    echo "View results:"
    echo "  cat ${BENCHMARK_OUTPUT_DIR}/comparison_summary.csv"
    echo "  ls ${BENCHMARK_OUTPUT_DIR}/*.png"
    echo ""
    echo "To run full comparison on trained models:"
    echo "  sbatch scripts/comparison/compare_all.sh"
    echo "  sbatch scripts/comparison/compare_all.sh ${MODEL_TYPE} ${RUN_NUMBER}"
    echo ""
else
    echo "⚠️  No benchmark outputs found, comparison skipped"
fi

echo "================================================================================"
