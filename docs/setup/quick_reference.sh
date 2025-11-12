#!/bin/bash
# Quick reference for MLIR-RL plotting and Neptune commands

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════╗
║              MLIR-RL PLOTTING & NEPTUNE QUICK REFERENCE           ║
╚═══════════════════════════════════════════════════════════════════╝

📦 REQUIREMENTS
──────────────────────────────────────────────────────────────────────
  ✓ Matplotlib is already installed in the mlir environment

📊 GENERATE PLOTS LOCALLY
──────────────────────────────────────────────────────────────────────
  python analysis/plot_results.py results/run_9

  Output: results/run_9/plots/
    • speedup_by_op_type.png       (bar chart by operation type)
    • geometric_mean_speedup.png   (geometric mean comparison)
    • per_benchmark_speedup.png    (detailed horizontal bars)
    • training_metrics.png         (4-panel training curves)

🌊 NEPTUNE SYNC (3 OPTIONS)
──────────────────────────────────────────────────────────────────────
  Option 1: Quick test (single run with plots)
    python experiments/sync_neptune_with_plots.py results/run_9

  Option 2: Simple test (without plots)
    python experiments/test_neptune.py

  Option 3: Continuous sync during training
    bash scripts/neptune-sync.sh   # Run in separate terminal

🚀 FULL TRAINING WORKFLOW
──────────────────────────────────────────────────────────────────────
  # Simple: Auto-sync enabled (recommended)
  bash scripts/train.sh
  # → Training will automatically sync to Neptune when complete!

  # Advanced: Real-time sync during training
  # Terminal 1: Training
  bash scripts/train.sh

  # Terminal 2: Real-time Neptune sync
  bash scripts/neptune-sync.sh

  # Manual sync after training (if auto-sync disabled)
  python experiments/sync_neptune_with_plots.py results/run_X

🔍 VIEW RESULTS
──────────────────────────────────────────────────────────────────────
  Local:   results/run_9/plots/*.png
  Neptune: https://app.neptune.ai/mehdix/mlir-project

📈 METRICS LOGGED
──────────────────────────────────────────────────────────────────────
  Training:
    • train/reward, train/entropy, train/final_speedup

  PPO:
    • train_ppo/policy_loss, train_ppo/value_loss
    • train_ppo/clip_frac, train_ppo/approx_kl

  Evaluation:
    • eval/average_speedup
    • eval/speedup/<benchmark_name> (per-benchmark)
    • eval/exec_time/<benchmark_name>
    • eval/reward, eval/cumulative_reward

  Plots:
    • plots/speedup_by_op_type.png
    • plots/geometric_mean_speedup.png
    • plots/per_benchmark_speedup.png
    • plots/training_metrics.png

📝 CONFIGURATION
──────────────────────────────────────────────────────────────────────
  Training config:  config/config.json
  Environment vars: .env
  Test dataset:     data/test/
    • execution_times_train.json (11 benchmarks)
    • execution_times_eval.json (6 benchmarks)

🔧 USEFUL COMMANDS
──────────────────────────────────────────────────────────────────────
  # List all runs
  ls -lht results/

  # Generate plots for multiple runs
  for run in results/run_{9..12}; do
      python analysis/plot_results.py $run
  done

  # Sync multiple runs to Neptune
  for run in results/run_{9..12}; do
      python experiments/sync_neptune_with_plots.py $run
  done

  # Check training progress
  tail -f logs/interactive_*.debug

  # View latest results
  cat results/run_9/logs/eval/average_speedup

🆘 TROUBLESHOOTING
──────────────────────────────────────────────────────────────────────
  Problem: "ModuleNotFoundError: No module named 'matplotlib'"
  Solution: conda activate mlir && pip install matplotlib

  Problem: "NeptuneMissingApiTokenException"
  Solution: Check .env file has NEPTUNE_PROJECT and NEPTUNE_TOKEN

  Problem: No plots generated
  Solution: Make sure run has evaluation metrics in logs/eval/speedup/

  Problem: Training fails with SLURM_JOB_ID error
  Solution: Check .env has SLURM_JOB_ID and SLURM_JOB_NAME defined

📚 DOCUMENTATION
──────────────────────────────────────────────────────────────────────
  Detailed guide:  PLOTTING_README.md
  Project README:  README.md

EOF

