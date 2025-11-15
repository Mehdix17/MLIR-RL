#!/bin/bash
#
# Quick test of comparison framework (uses test data)
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

# Step 1: Create benchmark suite
echo "[1/4] Creating benchmark suite..."
echo "--------------------------------------------------------------------------------"
python benchmarks/benchmark_suite.py
echo ""

# Step 2: Run PyTorch benchmarks (these are fast)
echo "[2/4] Running PyTorch benchmarks..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_pytorch_default.py
python evaluation/run_pytorch_jit.py
echo ""

# Step 3: Check if RL model exists
echo "[3/4] Checking for RL model..."
echo "--------------------------------------------------------------------------------"
if [ -d "results/lstm/run_0/models" ]; then
    echo "✓ Found RL model, running RL benchmarks..."
    python evaluation/run_rl_optimized.py || echo "⚠️  RL benchmarks skipped"
else
    echo "⚠️  No RL model found, skipping RL benchmarks"
    echo "   (Comparison will only show PyTorch Default vs JIT)"
fi
echo ""

# Step 4: Generate comparison
echo "[4/4] Generating comparison..."
echo "--------------------------------------------------------------------------------"
python evaluation/compare_all.py
echo ""

echo "================================================================================"
echo "✓ TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "View results:"
echo "  cat results/comparison_rl_vs_pytorch/comparison_summary.csv"
echo ""
echo "To run full comparison after training:"
echo "  sbatch scripts/comparison/compare_all.sh"
echo ""
echo "================================================================================"
