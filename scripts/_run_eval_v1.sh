#!/bin/bash
source ~/envs/mlir/bin/activate
set -a && source /scratch/mb10856/MLIR-RL/.env && set +a
export AUTOSCHEDULER_IMPL=rl_autoschedular_v1
export CONFIG_FILE_PATH=/scratch/mb10856/MLIR-RL/config/v1.json
export EVAL_LAST_ONLY=1
export EVAL_DIR=/scratch/mb10856/MLIR-RL/results/new_experiment/v1_agent/run_0/models
cd /scratch/mb10856/MLIR-RL
rm -f results/new_experiment/v1_agent/run_0/logs/eval/final_speedup
rm -f results/new_experiment/v1_agent/run_0/logs/eval/_eval_checkpoint.txt
python scripts/eval.py
echo "DONE_V1" > /scratch/mb10856/MLIR-RL/logs/eval_v1_shell_done.txt
