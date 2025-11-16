#!/bin/bash
#
# Compare RL-Optimized vs PyTorch Default vs PyTorch JIT
# Usage: compare_all.sh [model_type] [run_number]
#        compare_all.sh [benchmarks_directory]
# Example: compare_all.sh lstm 1
#          compare_all.sh results/lstm/run_1/benchmarks
#

#SBATCH -J compare-all
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mb10856@nyu.edu

# Setup environment
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate mlir

# Change to project root - handle both submission from project root or scripts dir
if [ -f "data/benchmarks/benchmark_suite.py" ]; then
    # Already in project root
    PROJECT_ROOT="$(pwd)"
elif [ -f "../data/benchmarks/benchmark_suite.py" ]; then
    # In scripts/ directory
    cd ..
    PROJECT_ROOT="$(pwd)"
else
    # Try to find project root
    PROJECT_ROOT="/scratch/mb10856/MLIR-RL"
    cd "${PROJECT_ROOT}" || exit 1
fi

echo "================================================================================"
echo "3-Way Comparison: RL-Optimized vs PyTorch Default vs PyTorch JIT"
echo "================================================================================"
echo ""
echo "Project: ${PROJECT_ROOT}"
echo "Date: $(date)"
echo ""

# Parse arguments: can be (model_type run_number) or (benchmarks_directory)
if [ -n "$1" ]; then
    if [ -d "$1" ]; then
        # Argument is a directory path
        BENCHMARKS_DIR="$1"
        RUN_DIR="$(dirname "$BENCHMARKS_DIR")"
        MODEL_TYPE="$(basename $(dirname "$RUN_DIR"))"
        RUN_NUMBER="$(basename "$RUN_DIR" | sed 's/run_//')"
        echo "Using specified benchmarks directory: $BENCHMARKS_DIR"
    elif [ -n "$2" ]; then
        # Two arguments: model_type and run_number
        MODEL_TYPE="$1"
        RUN_NUMBER="$2"
        RUN_DIR="results/${MODEL_TYPE}/run_${RUN_NUMBER}"
        BENCHMARKS_DIR="${RUN_DIR}/benchmarks"
        
        if [ ! -d "$BENCHMARKS_DIR" ]; then
            echo "✗ Error: Benchmarks directory not found: $BENCHMARKS_DIR"
            echo ""
            echo "Generate benchmarks first:"
            echo "  sbatch scripts/run_benchmarks.sh $MODEL_TYPE $RUN_NUMBER"
            exit 1
        fi
        
        echo "Using specified model: $MODEL_TYPE, run: $RUN_NUMBER"
    else
        echo "✗ Error: Invalid arguments"
        echo ""
        echo "Usage:"
        echo "  $0                                    # Auto-detect latest"
        echo "  $0 lstm 1                             # Specify model and run"
        echo "  $0 results/lstm/run_1/benchmarks      # Specify directory"
        exit 1
    fi
else
    # Auto-detect latest benchmarks directory with all 3 output files
    echo "Auto-detecting benchmarks directory..."
    BENCHMARKS_DIR=""
    if [ -d "results" ]; then
        LATEST_TIME=0
        for model_dir in results/*; do
            if [ -d "$model_dir" ]; then
                for run_dir in $model_dir/run_*; do
                    bench_dir="$run_dir/benchmarks"
                    if [ -d "$bench_dir" ] && \
                       [ -f "$bench_dir/agent_output.json" ] && \
                       [ -f "$bench_dir/pytorch_output.json" ] && \
                       [ -f "$bench_dir/pytorch_jit_output.json" ]; then
                        MOD_TIME=$(stat -c %Y "$bench_dir" 2>/dev/null || stat -f %m "$bench_dir" 2>/dev/null)
                        if [ "$MOD_TIME" -gt "$LATEST_TIME" ]; then
                            LATEST_TIME=$MOD_TIME
                            BENCHMARKS_DIR="$bench_dir"
                            MODEL_TYPE="$(basename $model_dir)"
                            RUN_NUMBER="$(basename $run_dir | sed 's/run_//')"
                        fi
                    fi
                done
            fi
        done
    fi
    
    if [ -z "$BENCHMARKS_DIR" ]; then
        echo "✗ Error: No benchmarks directory with all outputs found"
        echo ""
        echo "Generate benchmarks first:"
        echo "  sbatch scripts/run_benchmarks.sh"
        exit 1
    fi
    
    echo "✓ Found latest: $MODEL_TYPE, run: $RUN_NUMBER"
fi

# Extract model info if not already set
if [ -z "$MODEL_TYPE" ]; then
    RUN_DIR="$(dirname "$BENCHMARKS_DIR")"
    MODEL_TYPE="$(basename $(dirname "$RUN_DIR"))"
    RUN_NUMBER="$(basename "$RUN_DIR" | sed 's/run_//')"
fi

echo ""
echo "Model Type: $MODEL_TYPE"
echo "Run Number: $RUN_NUMBER"
echo "Benchmarks Directory: $BENCHMARKS_DIR"
echo ""

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

if [ -f "$CONFIG_FILE" ]; then
    echo "Config file: $CONFIG_FILE"
else
    echo "⚠️  Config file not found: $CONFIG_FILE"
fi
echo ""

# Validate benchmarks directory
if [ ! -d "$BENCHMARKS_DIR" ]; then
    echo "✗ Error: Benchmarks directory not found: $BENCHMARKS_DIR"
    echo ""
    echo "Usage:"
    echo "  $0                                    # Auto-detect latest"
    echo "  $0 lstm 1                             # Specify model and run"
    echo "  $0 results/lstm/run_1/benchmarks      # Specify directory"
    echo ""
    echo "Generate benchmarks first:"
    echo "  sbatch scripts/run_benchmarks.sh"
    exit 1
fi

# Check required files exist
MISSING_FILES=0
for file in agent_output.json pytorch_output.json pytorch_jit_output.json; do
    if [ ! -f "$BENCHMARKS_DIR/$file" ]; then
        echo "✗ Missing required file: $BENCHMARKS_DIR/$file"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "✗ Error: Missing required benchmark output files"
    echo ""
    echo "Required files in benchmarks directory:"
    echo "  - agent_output.json"
    echo "  - pytorch_output.json"
    echo "  - pytorch_jit_output.json"
    echo ""
    echo "Generate them by running:"
    echo "  python evaluation/run_rl_optimized.py"
    echo "  python evaluation/run_pytorch_default.py"
    echo "  python evaluation/run_pytorch_jit.py"
    exit 1
fi

echo "✓ All required benchmark files found"
echo ""

# Run comparison
echo "Generating comparison report..."
echo "--------------------------------------------------------------------------------"
python evaluation/compare_all.py "$BENCHMARKS_DIR"
if [ $? -ne 0 ]; then
    echo "✗ Error generating comparison"
    exit 1
fi
echo ""

echo "================================================================================"
echo "✓ COMPARISON COMPLETE!"
echo "================================================================================"
echo ""
echo "Model: $MODEL_TYPE (run_${RUN_NUMBER})"
echo "Benchmarks directory: $BENCHMARKS_DIR"
echo ""
echo "Input files:"
echo "  - ${BENCHMARKS_DIR}/agent_output.json"
echo "  - ${BENCHMARKS_DIR}/pytorch_output.json"
echo "  - ${BENCHMARKS_DIR}/pytorch_jit_output.json"
echo ""
echo "Output files:"
echo "  - ${BENCHMARKS_DIR}/comparison_results.json"
echo "  - ${BENCHMARKS_DIR}/comparison_summary.csv"
echo "  - ${BENCHMARKS_DIR}/comparison_bar_plot.png"
echo "  - ${BENCHMARKS_DIR}/speedup_comparison.png"
echo ""
echo "To view the summary:"
echo "  cat ${BENCHMARKS_DIR}/comparison_summary.csv"
echo ""
echo "================================================================================"
