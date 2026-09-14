#!/bin/bash
#SBATCH --job-name=v0-train
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/tb3654/MLIR-RL/logs/v0_full_%j.out
#SBATCH --error=/scratch/tb3654/MLIR-RL/logs/v0_full_%j.err

export PATH="/home/tb3654/.conda/envs/mlir/bin:$PATH"
export LD_LIBRARY_PATH="/home/tb3654/.conda/envs/mlir/lib:/scratch/tb3654/MLIR-RL/llvm-project/build/lib"
export PYTHONPATH="/scratch/tb3654/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:/scratch/tb3654/MLIR-RL"
export AST_DUMPER_BIN_PATH=/scratch/tb3654/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
export AUTOSCHEDULER_IMPL=rl_autoschedular
export CONFIG_FILE_PATH=/scratch/tb3654/MLIR-RL/config/v0.json
cd /scratch/tb3654/MLIR-RL
python -u scripts/train.py
