#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07-00
#SBATCH -o logs/neptune/%j.out
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mb10856@nyu.edu

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate mlir

# Change to project root directory
cd /scratch/mb10856/MLIR-RL

# Execute the code
python experiments/neptune_sync.py

