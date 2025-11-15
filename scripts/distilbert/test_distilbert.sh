#!/bin/bash
#
# Quick test with DistilBERT (minimal iterations)
# Uses: config/test_distilbert.json
# Data: data/test (17 MLIR files only)
# Expected time: ~10-20 minutes
#

#SBATCH -J test-distilbert
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 1-00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mb10856@nyu.edu

# Setup environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate mlir

# Change to project root (SLURM_SUBMIT_DIR is where sbatch was called)
cd "${SLURM_SUBMIT_DIR}" || exit 1
PROJECT_ROOT="$(pwd)"

# Configuration
export DASK_NODES=4
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/test_distilbert.json"
export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"

echo "================================================"
echo "Quick Test: DistilBERT (Minimal)"
echo "================================================"
echo "Config: config/test_distilbert.json"
echo "Data: data/test (17 files only)"
echo "Model: 6-layer DistilBERT (768 hidden, 12 heads)"
echo "Iterations: 3 (quick validation)"
echo "Learning rate: 0.0001"
echo "Batch size: 16"
echo "Purpose: Verify DistilBERT setup and pipeline"
echo "================================================"
echo ""

# Train model
python bin/train.py

echo ""
echo "================================================"
echo "Test complete! Check results/ directory"
echo "If this succeeded, you're ready for full training"
echo "================================================"
