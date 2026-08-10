#!/bin/bash
#SBATCH --job-name=mlir-pipeline
#SBATCH --partition=compute
#SBATCH --array=0-10
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

set -euo pipefail

# Model matrix: family:model
MODELS=(
  "vision:resnet18"
  "vision:resnet50"
  "vision:efficientnet_b0"
  "vision:mobilenet_v3_small"
  "vision:densenet121"
  "vision:vit_b_16"
  "vision:convnext_tiny"
  "transformer:bert"
  "transformer:distilbert"
  "transformer:roberta"
  "transformer:albert"
)

ENTRY="${MODELS[$SLURM_ARRAY_TASK_ID]}"
FAMILY="${ENTRY%%:*}"
MODEL="${ENTRY##*:}"

# In sbatch execution BASH_SOURCE can point to a spool copy; prefer submit dir.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Family : $FAMILY"
echo "Model  : $MODEL"
echo "Host   : $(hostname)"
echo "Start  : $(date)"
echo "=========================================="

mkdir -p "$PROJECT_ROOT/logs"

# Conda setup
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate mlir

# Runtime + project env
export GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
export LD_LIBRARY_PATH="$GCC14_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="/home/tb3654/.conda/envs/mlir/bin/python"

RAW_DIR="data/nn/raw_batch"
CODE_DIR="data/nn/code_files_batch"
REAL_DIR="data/nn/real_batch"
DYN_DIR="data/nn/synthetic/dynamic_batch"
MANIFEST_DIR="data/nn/manifests_batch"

cd "$PROJECT_ROOT"

if [[ "$FAMILY" == "vision" ]]; then
  "$PYTHON_BIN" -m data_utils.orchestrate pipeline \
    --vision-models "$MODEL" \
    --transformer-models none \
    --raw-dir "$RAW_DIR" \
    --code-dir "$CODE_DIR" \
    --real-output-dir "$REAL_DIR" \
    --dynamic-output-dir "$DYN_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --batch-sizes 1 16 64 \
    --dynamic-shape-policy smart \
    --dynamic-max-variants 2 \
    --vision-backend onnx \
    --clean-extract
else
  "$PYTHON_BIN" -m data_utils.orchestrate pipeline \
    --vision-models none \
    --transformer-models "$MODEL" \
    --raw-dir "$RAW_DIR" \
    --code-dir "$CODE_DIR" \
    --real-output-dir "$REAL_DIR" \
    --dynamic-output-dir "$DYN_DIR" \
    --manifest-dir "$MANIFEST_DIR" \
    --batch-sizes 1 16 64 \
    --dynamic-shape-policy smart \
    --dynamic-max-variants 2 \
    --transformer-backend onnx \
    --seq-lens 16 \
    --transformer-batch-size 1 \
    --transformer-input-mode synthetic \
    --clean-extract
fi

STATUS=$?

echo "=========================================="
echo "Exit   : $STATUS"
echo "End    : $(date)"
echo "=========================================="

exit $STATUS
