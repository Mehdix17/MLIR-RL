#!/bin/bash
#
# Compare RL-Optimized vs PyTorch Default vs PyTorch JIT
#

#SBATCH -J compare-all
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 4:00:00
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
echo "3-Way Comparison: RL-Optimized vs PyTorch Default vs PyTorch JIT"
echo "================================================================================"
echo ""
echo "Project: ${PROJECT_ROOT}"
echo "Date: $(date)"
echo ""

# Step 1: Create benchmark suite
echo "[1/5] Creating benchmark suite..."
echo "--------------------------------------------------------------------------------"
python benchmarks/benchmark_suite.py
if [ $? -ne 0 ]; then
    echo "✗ Error creating benchmark suite"
    exit 1
fi
echo ""

# Step 2: Check for trained RL model
echo "[2/5] Checking for trained RL model..."
echo "--------------------------------------------------------------------------------"
if [ -d "results/lstm/run_0/models" ] && [ -n "$(ls -A results/lstm/run_0/models/*.pt 2>/dev/null)" ]; then
    echo "✓ Found trained RL model"
else
    echo "✗ No trained RL model found."
    echo ""
    echo "Please train a model first:"
    echo "  sbatch scripts/lstm/train_lstm_baseline.sh"
    echo ""
    echo "Or if you just want to test PyTorch Default vs JIT:"
    echo "  python evaluation/run_pytorch_default.py"
    echo "  python evaluation/run_pytorch_jit.py"
    echo "  python evaluation/compare_all.py"
    exit 1
fi
echo ""

# Step 3: Run RL-optimized benchmarks
echo "[3/5] Running RL-optimized benchmarks..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_rl_optimized.py
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: RL-optimized benchmarks failed, continuing anyway..."
fi
echo ""

# Step 4: Run PyTorch benchmarks
echo "[4/5] Running PyTorch benchmarks..."
echo "--------------------------------------------------------------------------------"
echo "Running PyTorch Default..."
python evaluation/run_pytorch_default.py
if [ $? -ne 0 ]; then
    echo "✗ Error running PyTorch Default benchmarks"
    exit 1
fi
echo ""

echo "Running PyTorch JIT..."
python evaluation/run_pytorch_jit.py
if [ $? -ne 0 ]; then
    echo "✗ Error running PyTorch JIT benchmarks"
    exit 1
fi
echo ""

# Step 5: Generate comparison
echo "[5/5] Generating comparison report..."
echo "--------------------------------------------------------------------------------"
python evaluation/compare_all.py
if [ $? -ne 0 ]; then
    echo "✗ Error generating comparison"
    exit 1
fi
echo ""

echo "================================================================================"
echo "✓ COMPARISON COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/comparison_rl_vs_pytorch/comparison_results.json"
echo "  - results/comparison_rl_vs_pytorch/comparison_summary.csv"
echo "  - results/comparison_rl_vs_pytorch/comparison_bar_plot.png"
echo "  - results/comparison_rl_vs_pytorch/speedup_comparison.png"
echo ""
echo "To view the summary:"
echo "  cat results/comparison_rl_vs_pytorch/comparison_summary.csv"
echo ""
echo "================================================================================"
