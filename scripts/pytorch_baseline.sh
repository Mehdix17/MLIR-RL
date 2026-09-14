#!/bin/bash
#SBATCH --job-name=pt-base
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7
#SBATCH --output=/scratch/tb3654/MLIR-RL/logs/pt_%A_%a.out
#SBATCH --error=/scratch/tb3654/MLIR-RL/logs/pt_%A_%a.err

PROJECT_ROOT=/scratch/tb3654/MLIR-RL
[[ -f "$PROJECT_ROOT/.env" ]] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }
export PATH="/home/tb3654/.conda/envs/mlir/bin:$PATH"
export LD_LIBRARY_PATH="/home/tb3654/.conda/envs/mlir/lib:$LLVM_BUILD_PATH/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PyTorch baseline — chunk ${SLURM_ARRAY_TASK_ID}/8"
echo "Started at $(date) on $(hostname)"
echo "=========================================="

python -u scripts/get_pytorch_times.py \
    --benchmarks-dir data/all/code_files \
    --output results/all/exec_times/pytorch.json \
    --chunk-index ${SLURM_ARRAY_TASK_ID} \
    --num-chunks 8

echo "=========================================="
echo "Chunk ${SLURM_ARRAY_TASK_ID} completed at $(date)"
echo "=========================================="
