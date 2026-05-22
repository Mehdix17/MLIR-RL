#!/bin/bash
#SBATCH --job-name=pytorch-times
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH -C bergamo
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/pytorch_times_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/pytorch_times_%j.err

set -e
cd /scratch/mb10856/MLIR-RL
source ~/envs/mlir/bin/activate
set -a && source .env && set +a

python scripts/measure_full_model_baselines.py
