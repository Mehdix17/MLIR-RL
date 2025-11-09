#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J p-original-c
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 7-00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mb10856@nyu.edu

# Resource requiremenmt commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate mlir

# Change to project root directory
cd "$(dirname "$0")/.." && PROJECT_ROOT="$(pwd)"

# Execute the code
export DASK_NODES=16
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"
export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"

# Note: Training will automatically sync to Neptune after completion
# if NEPTUNE_PROJECT and NEPTUNE_TOKEN are set in .env
python bin/train.py
