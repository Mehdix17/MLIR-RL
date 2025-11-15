#!/bin/bash
#
# Train LSTM with augmented dataset
# Uses: config/config_augmented.json
# Data: data/all + data/generated (9,941 MLIR files)
# Expected time: ~10-15 hours (1000 iterations)
#

#SBATCH -J train-lstm-aug
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 3-00
#SBATCH -o ../logs/%x_%j.out
#SBATCH -e ../logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

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
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config_augmented.json"
export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"

echo "================================================"
echo "Training LSTM with Augmented Dataset"
echo "================================================"
echo "Config: config/config_augmented.json"
echo "Data: data/all + data/generated (9,941 files)"
echo "Iterations: 1000"
echo "Learning rate: 0.0001"
echo "Batch size: 32"
echo "Augmentation ratio: 30%"
echo "================================================"
echo ""

# Train model
python bin/train.py

echo ""
echo "================================================"
echo "Training complete! Check results/ directory"
echo "================================================"
