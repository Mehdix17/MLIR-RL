#!/bin/bash

# SLURM settings for full-scale training on generated benchmarks
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/tb3654/MLIR-RL/logs/train_fullscale_%j.out
#SBATCH -e /scratch/tb3654/MLIR-RL/logs/train_fullscale_%j.err
#SBATCH -J mlir-train-fullscale

# Load environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
	set -a
	source "$PROJECT_ROOT/.env"
	set +a
fi

conda activate "${CONDA_ENV:-mlir}"

# Set environment variables
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="$PROJECT_ROOT/config/generated-benchmarks.json"

# Navigate to project root
cd "$PROJECT_ROOT"

# Print configuration info
echo "=========================================="
echo "Full-scale training on generated benchmarks"
echo "=========================================="
echo "Config file: $CONFIG_FILE_PATH"
echo "Benchmarks: 7 (albert, bert, densenet121, distilbert, efficientnet_b0, mobilenet_v3_small, resnet18)"
echo "Iterations: 5000"
echo "Checkpoints: Every 5 iterations (1000 total)"
echo "Wall time: 12 hours"
echo "=========================================="

# Launch training
python train.py

# Print completion status
echo "Training completed at $(date)"
echo "Checkpoints saved to: results/run_*/models/"
