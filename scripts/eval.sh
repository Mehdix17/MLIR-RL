#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J eval-p-original
#SBATCH -p compute
#SBATCH --reservation=scomputer-dalma
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 7-00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mb10856@nyu.edu

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate mlir

# Change to project root directory
cd "$(dirname "$0")/.." && PROJECT_ROOT="$(pwd)"

# Execute the code
export EVAL_DIR=results/run
export DASK_NODES=0
export OMP_NUM_THREADS=12
export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"
python bin/evaluate.py

