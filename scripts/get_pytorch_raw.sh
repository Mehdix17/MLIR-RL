#!/bin/bash
#SBATCH --job-name=pt-raw
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/tb3654/MLIR-RL/logs/pt_raw_%j.out
#SBATCH --error=/scratch/tb3654/MLIR-RL/logs/pt_raw_%j.err

PROJECT_ROOT=/scratch/tb3654/MLIR-RL
[[ -f "$PROJECT_ROOT/.env" ]] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }
export PATH="/home/tb3654/.conda/envs/mlir/bin:$PATH"
export LD_LIBRARY_PATH="/home/tb3654/.conda/envs/mlir/lib:$LLVM_BUILD_PATH/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

echo "=== PyTorch baseline on raw models ==="
python -u scripts/get_pytorch_times.py --benchmarks-dir data/nn/raw_bench --output results/raw_models/exec_times/pytorch_base.json
echo "=== Done ==="
