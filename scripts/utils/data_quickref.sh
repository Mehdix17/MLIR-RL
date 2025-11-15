#!/bin/bash
# Quick reference for working with organized data structure

# ============================================================
# Data Organization Quick Reference
# ============================================================

echo "╔════════════════════════════════════════════════════════╗"
echo "║     MLIR-RL Data Organization Quick Reference         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# 1. VIEW CURRENT DATA STRUCTURE
echo "1️⃣  View Data Structure"
echo "   tree data -L 2 -d"
echo

# 2. TRAIN WITH EXISTING DATA (NO CHANGES)
echo "2️⃣  Train with Existing Data (Original Workflow)"
echo "   CONFIG_FILE_PATH=config/config.json python bin/train.py"
echo

# 3. TRAIN WITH AUGMENTED DATA
echo "3️⃣  Train with Augmented Data (New!)"
echo "   CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py"
echo

# 4. GENERATE MORE AUGMENTATION
echo "4️⃣  Generate More Augmentation Data"
echo "   python scripts/augment_dataset.py"
echo "   # Generates 500 files matching data/all/ format"
echo

# 5. CONVERT NEURAL NETWORKS
echo "5️⃣  Convert Neural Networks to MLIR"
echo "   python data_generation/nn_to_mlir.py"
echo "   # Converts ResNet-18 to data/neural_nets/"
echo

# 6. GENERATE RANDOM OPERATIONS
echo "6️⃣  Generate Random Operations (for testing)"
echo "   python data_generation/random_mlir_gen.py"
echo "   # Generates matmul, conv2d, pooling in data/generated/train/"
echo

# 7. CHECK STATISTICS
echo "7️⃣  Check Data Statistics"
echo "   echo 'Existing training data:'"
echo "   find data/all/code_files -name '*.mlir' | wc -l"
echo "   echo 'Augmented data:'"
echo "   find data/generated/code_files -name '*.mlir' | wc -l"
echo

# 8. EVALUATE TRAINED AGENT
echo "8️⃣  Evaluate Trained Agent"
echo "   python -c 'from evaluation import SingleOperationEvaluator; from pathlib import Path; evaluator = SingleOperationEvaluator(agent_model_path=Path(\"results/best_model.pt\"), benchmark_dir=Path(\"data/benchmarks/single_ops\")); evaluator.evaluate_benchmark_suite()'"
echo

# 9. CLEAN GENERATED DATA
echo "9️⃣  Clean Generated Data (can be regenerated)"
echo "   rm -rf data/generated/code_files/*.mlir"
echo "   rm -rf data/generated/train/*.mlir"
echo "   # Then regenerate with: python scripts/augment_dataset.py"
echo

# 10. REORGANIZE DATA (if needed)
echo "🔟  Reorganize Data Structure"
echo "   python scripts/organize_data.py"
echo

echo "════════════════════════════════════════════════════════"
echo "📊 Current Statistics:"
echo "════════════════════════════════════════════════════════"

# Count files
if [ -d "data/all/code_files" ]; then
    existing_count=$(find data/all/code_files -name "*.mlir" 2>/dev/null | wc -l)
    echo "   Existing training data: $existing_count files"
fi

if [ -d "data/generated/code_files" ]; then
    augmented_count=$(find data/generated/code_files -name "*.mlir" 2>/dev/null | wc -l)
    echo "   Augmented data:         $augmented_count files"
fi

if [ -d "data/test/code_files" ]; then
    test_count=$(find data/test/code_files -name "*.mlir" 2>/dev/null | wc -l)
    echo "   Test data:              $test_count files"
fi

echo
echo "════════════════════════════════════════════════════════"
echo "📁 Directory Structure:"
echo "════════════════════════════════════════════════════════"
echo "   data/all/           → Original training data (9441 files)"
echo "   data/test/          → Original test data (17 files)"
echo "   data/generated/     → Augmentation data (500 files)"
echo "   data/neural_nets/   → Converted neural networks"
echo "   data/benchmarks/    → Evaluation benchmarks"
echo
echo "════════════════════════════════════════════════════════"
echo "✨ Quick Start:"
echo "════════════════════════════════════════════════════════"
echo "   # Train with augmentation (recommended)"
echo "   CONFIG_FILE_PATH=config/config_augmented.json python bin/train.py"
echo
echo "════════════════════════════════════════════════════════"
