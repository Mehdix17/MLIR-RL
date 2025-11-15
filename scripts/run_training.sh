#!/bin/bash
#
# Master training script launcher
# Shows all available training options and helps you choose
#

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         MLIR-RL Training Script Launcher                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Available Training Scripts:"
echo ""
echo "🧪 QUICK TESTS (Start here!)"
echo "  1) test_lstm.sh          - LSTM quick test (15 min, 17 files)"
echo "  2) test_distilbert.sh    - DistilBERT quick test (20 min, 17 files)"
echo ""
echo "🎯 FULL TRAINING"
echo "  3) train_lstm_baseline.sh   - LSTM baseline (1 hour, 9,441 files)"
echo "  4) train_lstm_augmented.sh  - LSTM augmented (12 hours, 9,941 files)"
echo "  5) train_distilbert.sh      - DistilBERT (3 hours, 9,441 files)"
echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "📊 Script Details:"
echo ""
echo "┌─────────────────────────────┬─────────┬────────┬──────────┐"
echo "│ Script                      │ Time    │ Data   │ Config   │"
echo "├─────────────────────────────┼─────────┼────────┼──────────┤"
echo "│ test_lstm.sh                │  15 min │  17    │ test     │"
echo "│ test_distilbert.sh          │  20 min │  17    │ test_db  │"
echo "│ train_lstm_baseline.sh      │  1 hour │ 9,441  │ baseline │"
echo "│ train_lstm_augmented.sh     │ 12 hour │ 9,941  │ augment  │"
echo "│ train_distilbert.sh         │  3 hour │ 9,441  │ distilb  │"
echo "└─────────────────────────────┴─────────┴────────┴──────────┘"
echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# If argument provided, run that script
if [ $# -eq 1 ]; then
    case "$1" in
        1) SCRIPT="lstm/test_lstm.sh" ;;
        2) SCRIPT="distilbert/test_distilbert.sh" ;;
        3) SCRIPT="lstm/train_lstm_baseline.sh" ;;
        4) SCRIPT="lstm/train_lstm_augmented.sh" ;;
        5) SCRIPT="distilbert/train_distilbert.sh" ;;
        test-lstm) SCRIPT="lstm/test_lstm.sh" ;;
        test-distilbert) SCRIPT="distilbert/test_distilbert.sh" ;;
        lstm) SCRIPT="lstm/train_lstm_baseline.sh" ;;
        lstm-aug) SCRIPT="lstm/train_lstm_augmented.sh" ;;
        distilbert) SCRIPT="distilbert/train_distilbert.sh" ;;
        *)
            echo "❌ Unknown option: $1"
            echo ""
            echo "Usage:"
            echo "  bash scripts/run_training.sh [option]"
            echo ""
            echo "Options: 1-5, test-lstm, test-distilbert, lstm, lstm-aug, distilbert"
            exit 1
            ;;
    esac
    
    echo "▶️  Launching: $SCRIPT"
    echo ""
    
    # Check if script exists
    if [ ! -f "scripts/$SCRIPT" ]; then
        echo "❌ Script not found: scripts/$SCRIPT"
        exit 1
    fi
    
    # Check if running on SLURM
    if command -v sbatch &> /dev/null; then
        echo "🚀 Submitting to SLURM..."
        sbatch "scripts/$SCRIPT"
        echo ""
        echo "✅ Job submitted! Check status with: squeue -u \$USER"
        echo "📋 View logs in: logs/"
    else
        echo "💻 Running locally (no SLURM detected)..."
        bash "scripts/$SCRIPT"
    fi
    
else
    # Interactive mode
    echo "Choose an option (1-5) or press Ctrl+C to cancel:"
    read -p "Your choice: " choice
    
    echo ""
    # Recursively call with the choice
    bash "$0" "$choice"
fi
