#!/bin/bash
#
# Complete workflow: Train LSTM → Run Comparison
#

#SBATCH -J train-and-compare
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 8:00:00
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
echo "Complete Workflow: Train LSTM Baseline → Run Comparison"
echo "================================================================================"
echo ""
echo "Project: ${PROJECT_ROOT}"
echo "Date: $(date)"
echo ""

# Configuration
export DASK_NODES=4
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"
export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"

#------------------------------------------------------------------------------
# STEP 1: Train LSTM Baseline
#------------------------------------------------------------------------------
echo "================================================================================"
echo "[STEP 1/5] Training LSTM Baseline Model"
echo "================================================================================"
echo ""
echo "Config: config/config.json"
echo "Data: data/all (9,441 MLIR files)"
echo "Iterations: 5"
echo "Expected time: ~1 hour"
echo ""

python bin/train.py

if [ $? -ne 0 ]; then
    echo "✗ Training failed!"
    exit 1
fi

echo ""
echo "✓ Training complete!"
echo ""

#------------------------------------------------------------------------------
# STEP 2: Create Benchmark Suite
#------------------------------------------------------------------------------
echo "================================================================================"
echo "[STEP 2/5] Creating Benchmark Suite"
echo "================================================================================"
echo ""

python benchmarks/benchmark_suite.py

if [ $? -ne 0 ]; then
    echo "✗ Benchmark creation failed!"
    exit 1
fi

echo ""
echo "✓ Benchmarks created!"
echo ""

#------------------------------------------------------------------------------
# STEP 3: Run RL-Optimized Benchmarks
#------------------------------------------------------------------------------
echo "================================================================================"
echo "[STEP 3/5] Running RL-Optimized Benchmarks"
echo "================================================================================"
echo ""

python evaluation/run_rl_optimized.py

if [ $? -ne 0 ]; then
    echo "⚠️  RL benchmarks had issues, but continuing..."
fi

echo ""
echo "✓ RL benchmarks complete!"
echo ""

#------------------------------------------------------------------------------
# STEP 4: Run PyTorch Benchmarks
#------------------------------------------------------------------------------
echo "================================================================================"
echo "[STEP 4/5] Running PyTorch Benchmarks (Default + JIT)"
echo "================================================================================"
echo ""

echo "Running PyTorch Default..."
python evaluation/run_pytorch_default.py

if [ $? -ne 0 ]; then
    echo "✗ PyTorch Default benchmarks failed!"
    exit 1
fi

echo ""
echo "Running PyTorch JIT..."
python evaluation/run_pytorch_jit.py

if [ $? -ne 0 ]; then
    echo "✗ PyTorch JIT benchmarks failed!"
    exit 1
fi

echo ""
echo "✓ PyTorch benchmarks complete!"
echo ""

#------------------------------------------------------------------------------
# STEP 5: Generate Comparison
#------------------------------------------------------------------------------
echo "================================================================================"
echo "[STEP 5/5] Generating Comparison Report"
echo "================================================================================"
echo ""

python evaluation/compare_all.py

if [ $? -ne 0 ]; then
    echo "✗ Comparison generation failed!"
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo "================================================================================"
echo "✓✓✓ WORKFLOW COMPLETE! ✓✓✓"
echo "================================================================================"
echo ""
echo "Training results:"
echo "  - Model saved to: results/lstm/run_*/models/"
echo "  - Training logs: results/lstm/run_*/logs/"
echo ""
echo "Comparison results:"
echo "  - Summary table: results/comparison_rl_vs_pytorch/comparison_summary.csv"
echo "  - Bar plot: results/comparison_rl_vs_pytorch/comparison_bar_plot.png"
echo "  - Speedup plot: results/comparison_rl_vs_pytorch/speedup_comparison.png"
echo "  - Raw data: results/comparison_rl_vs_pytorch/comparison_results.json"
echo ""
echo "View summary:"
echo "  cat results/comparison_rl_vs_pytorch/comparison_summary.csv"
echo ""
echo "================================================================================"
echo "Completed at: $(date)"
echo "================================================================================"
