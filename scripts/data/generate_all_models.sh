#!/bin/bash
#SBATCH --job-name=mlir-gen
#SBATCH --partition=compute
#SBATCH --array=0-10
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

# Model lists (removed gpt2, bart, t5 due to ONNX/export compatibility issues)
TRANSFORMER_MODELS=("bert" "distilbert" "roberta" "albert")
VISION_MODELS=("resnet18" "resnet50" "efficientnet_b0" "mobilenet_v3_small" "densenet121" "vit_b_16" "convnext_tiny")

# Combine into one array
ALL_MODELS=("${TRANSFORMER_MODELS[@]}" "${VISION_MODELS[@]}")

# Get model for this task
MODEL=${ALL_MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Activate conda environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate "${CONDA_ENV:-mlir}"

# Fix GLIBCXX library path - use conda's newer libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Navigate to data_utils directory
cd "$PROJECT_ROOT/data_utils"

# Determine which script to use
if [[ " ${TRANSFORMER_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Generating transformer model: $MODEL"
    python models-to-onnx.py \
        --model "$MODEL" \
        --output-dir "../data/generated/code files" \
        --strip-weights
else
    echo "Generating vision model: $MODEL"
    python vision-to-mlir.py \
        --model "$MODEL" \
        --output-dir "../data/generated/code files" \
        --strip-weights
fi

EXIT_CODE=$?

echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# List generated files
if [ $EXIT_CODE -eq 0 ]; then
    echo "Generated files:"
    ls -lh "../data/generated/code files"/${MODEL}* 2>/dev/null || echo "No files found"
fi

exit $EXIT_CODE
