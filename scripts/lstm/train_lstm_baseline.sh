#!/bin/bash
#
# Train LSTM baseline model
# Uses: config/config.json
# Data: data/all (9,441 MLIR files)
# Expected time: ~30-60 minutes
#

#SBATCH -J train-lstm-baseline
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 1-00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mb10856@nyu.edu

# Setup environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate mlir

# Change to project root (SLURM_SUBMIT_DIR is where sbatch was called)
cd "${SLURM_SUBMIT_DIR}" || exit 1
PROJECT_ROOT="$(pwd)"

# Configuration
export DASK_NODES=16
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"
export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"

echo "================================================"
echo "Training LSTM Baseline Model"
echo "================================================"
echo "Config: config/config.json"
echo "Data: data/all (9,441 files)"
echo "Iterations: 5"
echo "Learning rate: 0.001"
echo "Batch size: 32"
echo "================================================"
echo ""

# Train model
python bin/train.py

echo ""
echo "================================================"
echo "Training complete! Check results/ directory"
echo "================================================"
