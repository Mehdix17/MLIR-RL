#!/bin/bash
#
# Evaluate DistilBERT models
# Evaluates all saved model checkpoints from a DistilBERT training run
#

#SBATCH -J eval-distilbert
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
export DASK_NODES=0
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config_distilbert.json"

# Default to latest DistilBERT run if not specified
if [ -z "$EVAL_DIR" ]; then
    # Find latest DistilBERT run
    LATEST_RUN=$(ls -dt ${PROJECT_ROOT}/results/distilbert/run_* 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "❌ No DistilBERT runs found in results/distilbert/"
        exit 1
    fi
    export EVAL_DIR="$LATEST_RUN"
fi

echo "================================================"
echo "Evaluating DistilBERT Model"
echo "================================================"
echo "Evaluation directory: $EVAL_DIR"
echo "Config: $CONFIG_FILE_PATH"
echo "================================================"
echo ""

# Run evaluation
python bin/evaluate.py

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "Results saved in: $EVAL_DIR"
echo "================================================"
