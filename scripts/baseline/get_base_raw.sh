#!/bin/bash
#SBATCH --job-name=mlir-raw
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/tb3654/MLIR-RL/logs/mlir_raw_%j.out
#SBATCH --error=/scratch/tb3654/MLIR-RL/logs/mlir_raw_%j.err

PROJECT_ROOT=/scratch/tb3654/MLIR-RL
[[ -f "$PROJECT_ROOT/.env" ]] && { set -a; source "$PROJECT_ROOT/.env"; set +a; }
export PATH="/home/tb3654/.conda/envs/mlir/bin:$PATH"
export LD_LIBRARY_PATH="/home/tb3654/.conda/envs/mlir/lib:$LLVM_BUILD_PATH/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=== MLIR baseline on raw models ==="
python -u scripts/get_base.py --benchmarks-dir data/nn/raw_bench --output results/raw_models/exec_times/mlir_base.json --timeout 120
echo "=== Done ==="
