#!/bin/bash
#
# Run all 3 benchmark evaluations on a trained model
# Usage: run_benchmarks.sh [model_type] [run_number]
# Example: run_benchmarks.sh lstm 1
#

#SBATCH -J run-benchmarks
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mb10856@nyu.edu

# Setup environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate mlir

# Change to project root
if [ -f "benchmarks/benchmark_suite.py" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -f "../benchmarks/benchmark_suite.py" ]; then
    cd ..
    PROJECT_ROOT="$(pwd)"
else
    PROJECT_ROOT="/scratch/mb10856/MLIR-RL"
    cd "${PROJECT_ROOT}" || exit 1
fi

echo "================================================================================"
echo "Benchmark Evaluation Suite"
echo "================================================================================"
echo ""
echo "Project: ${PROJECT_ROOT}"
echo "Date: $(date)"
echo ""

# Parse arguments or auto-detect
if [ -n "$1" ] && [ -n "$2" ]; then
    # Manual specification
    MODEL_TYPE="$1"
    RUN_NUMBER="$2"
    RUN_DIR="results/${MODEL_TYPE}/run_${RUN_NUMBER}"
    
    if [ ! -d "$RUN_DIR/models" ]; then
        echo "✗ Error: Model directory not found: $RUN_DIR/models"
        exit 1
    fi
    
    echo "Using specified model: $MODEL_TYPE, run: $RUN_NUMBER"
else
    # Auto-detect latest trained model
    echo "Auto-detecting latest trained model..."
    
    LATEST_RUN=""
    LATEST_TIME=0
    
    if [ -d "results" ]; then
        for model_dir in results/*; do
            if [ -d "$model_dir" ]; then
                for run_dir in $model_dir/run_*; do
                    if [ -d "$run_dir/models" ] && [ -n "$(ls -A $run_dir/models/*.pt 2>/dev/null)" ]; then
                        MOD_TIME=$(stat -c %Y "$run_dir/models" 2>/dev/null || stat -f %m "$run_dir/models" 2>/dev/null)
                        if [ "$MOD_TIME" -gt "$LATEST_TIME" ]; then
                            LATEST_TIME=$MOD_TIME
                            LATEST_RUN="$run_dir"
                        fi
                    fi
                done
            fi
        done
    fi
    
    if [ -z "$LATEST_RUN" ]; then
        echo "✗ Error: No trained model found"
        echo ""
        echo "Usage: $0 [model_type] [run_number]"
        echo "Example: $0 lstm 1"
        echo ""
        echo "Or train a model first:"
        echo "  sbatch scripts/lstm/train_lstm_baseline.sh"
        echo "  sbatch scripts/distilbert/train_distilbert.sh"
        exit 1
    fi
    
    RUN_DIR="$LATEST_RUN"
    MODEL_TYPE="$(basename $(dirname $RUN_DIR))"
    RUN_NUMBER="$(basename $RUN_DIR | sed 's/run_//')"
    
    echo "✓ Found: $MODEL_TYPE, run: $RUN_NUMBER"
fi

echo ""
echo "Model Type: $MODEL_TYPE"
echo "Run Directory: $RUN_DIR"
echo "Output Directory: ${RUN_DIR}/benchmarks"
echo ""

# Create benchmarks directory
mkdir -p "${RUN_DIR}/benchmarks"

# Export environment variable for evaluation scripts
export BENCHMARK_OUTPUT_DIR="${RUN_DIR}/benchmarks"
export MODEL_DIR="${RUN_DIR}/models"

# Set config file based on model type
case "$MODEL_TYPE" in
    lstm)
        export CONFIG_FILE="config/config.json"
        ;;
    distilbert)
        export CONFIG_FILE="config/config_distilbert.json"
        ;;
    gpt2|gpt-2)
        export CONFIG_FILE="config/config_gpt2.json"
        ;;
    convnext)
        export CONFIG_FILE="config/config_convnext.json"
        ;;
    *)
        export CONFIG_FILE="config/config.json"
        echo "⚠️  Unknown model type: $MODEL_TYPE, using default config"
        ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo ""

# Step 1: Create benchmark suite
echo "[1/4] Creating benchmark suite..."
echo "--------------------------------------------------------------------------------"
python benchmarks/benchmark_suite.py
if [ $? -ne 0 ]; then
    echo "✗ Error creating benchmark suite"
    exit 1
fi
echo ""

# Step 2: Run RL Agent optimization
echo "[2/4] Running RL Agent benchmarks..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_rl_optimized.py "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "⚠️  RL Agent benchmarks failed, continuing..."
fi
echo ""

# Step 3: Run PyTorch Default
echo "[3/4] Running PyTorch Default benchmarks..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_pytorch_default.py "$BENCHMARK_OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "✗ Error running PyTorch Default benchmarks"
    exit 1
fi
echo ""

# Step 4: Run PyTorch JIT
echo "[4/4] Running PyTorch JIT benchmarks..."
echo "--------------------------------------------------------------------------------"
python evaluation/run_pytorch_jit.py "$BENCHMARK_OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "✗ Error running PyTorch JIT benchmarks"
    exit 1
fi
echo ""

echo "================================================================================"
echo "✓ BENCHMARK EVALUATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Model: $MODEL_TYPE (run_${RUN_NUMBER})"
echo "Output directory: ${RUN_DIR}/benchmarks"
echo ""
echo "Generated files:"
echo "  ✓ agent_output.json"
echo "  ✓ pytorch_output.json"
echo "  ✓ pytorch_jit_output.json"
echo ""
echo "Next step - Run comparison:"
echo "  sbatch scripts/comparison/compare_all.sh ${RUN_DIR}/benchmarks"
echo ""
echo "Or directly:"
echo "  python evaluation/compare_all.py ${RUN_DIR}/benchmarks"
echo ""
echo "================================================================================"
