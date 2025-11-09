#!/bin/bash
# Script to prepare the project for GitHub
# This makes all hardcoded paths relative and portable

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "  MLIR-RL GitHub Preparation Script"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Update train.sh
echo "✓ Updating scripts/train.sh..."
sed -i 's|cd /scratch/mb10856/MLIR-RL|cd "$(dirname "$0")/.." \&\& PROJECT_ROOT="$(pwd)"|g' scripts/train.sh
sed -i 's|export CONFIG_FILE_PATH=/scratch/mb10856/MLIR-RL/config/config.json|export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"|g' scripts/train.sh
sed -i 's|export AST_DUMPER_BIN_PATH=/scratch/mb10856/MLIR-RL/tools/ast_dumper/build/bin/AstDumper|export AST_DUMPER_BIN_PATH="${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper"|g' scripts/train.sh

# 2. Update eval.sh
echo "✓ Updating scripts/eval.sh..."
sed -i 's|cd /scratch/mb10856/MLIR-RL|cd "$(dirname "$0")/.." \&\& PROJECT_ROOT="$(pwd)"|g' scripts/eval.sh
sed -i 's|export CONFIG_FILE_PATH=/scratch/mb10856/MLIR-RL/config/config.json|export CONFIG_FILE_PATH="${PROJECT_ROOT}/config/config.json"|g' scripts/eval.sh

# 3. Update neptune-sync.sh
echo "✓ Updating scripts/neptune-sync.sh..."
sed -i 's|cd /scratch/mb10856/MLIR-RL|cd "$(dirname "$0")/.." \&\& PROJECT_ROOT="$(pwd)"|g' scripts/neptune-sync.sh

# 4. Create .env.example without secrets
echo "✓ Creating .env.example..."
cat > .env.example << 'EOF'
# Environment Configuration
# Copy this file to .env and fill in your values

# Conda environment path
CONDA_ENV=/path/to/your/conda/envs/mlir-build

# Neptune.ai credentials (get from https://app.neptune.ai)
NEPTUNE_PROJECT=your-workspace/your-project
NEPTUNE_TOKEN=your-neptune-api-token-here

# LLVM/MLIR paths (relative to project root)
LLVM_BUILD_PATH=./llvm-project/build
MLIR_SHARED_LIBS=./llvm-project/build/lib/libomp.so,./llvm-project/build/lib/libmlir_c_runner_utils.so,./llvm-project/build/lib/libmlir_runner_utils.so

# Tool paths (relative to project root)
AST_DUMPER_BIN_PATH=./tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=./tools/vectorizer/build/bin/Vectorizer

# SLURM job tracking (set automatically, no need to modify)
SLURM_JOB_ID=interactive
SLURM_JOB_NAME=interactive
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Preparation Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Update your .env with project-specific paths"
echo "3. Update SLURM email in scripts/*.sh"
echo "4. Commit and push to GitHub"
echo ""
