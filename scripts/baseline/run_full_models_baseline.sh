#!/bin/bash
#SBATCH --job-name=paper_full_base
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/paper_full_baseline_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/paper_full_baseline_%j.err
set -e
set -x
PROJECT_ROOT="/scratch/mb10856/MLIR-RL"
cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then set -a; source "$PROJECT_ROOT/.env"; set +a; fi
source "$HOME/envs/mlir/bin/activate"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
echo "=== $(date) on $(hostname) ==="
# Time just the 3 full models (use filtered dir)
FILTERED_DIR="/tmp/paper_full_filtered_$$"
export FILTERED_DIR
mkdir -p "$FILTERED_DIR"
for f in model_mobile_net_v2 model_res_net model_vgg; do
  ln -sf "$PROJECT_ROOT/data/mlir_rl_v1_paper/${f}.mlir" "$FILTERED_DIR/${f}.mlir"
done
ls "$FILTERED_DIR"
OUTPUT_TMP="results/mlir_rl_v1_paper_results/baselines/mlir/.tmp_full_${SLURM_JOB_ID}.json"
export OUTPUT_TMP
python scripts/baseline/get_base.py --benchmarks-dir "$FILTERED_DIR" --output "$OUTPUT_TMP" --implementation rl_autoschedular_v5 --timeout 60
echo "=== Raw ==="
python3 -c "import json,os; d=json.load(open(os.environ['OUTPUT_TMP'])); print(d)"
# Merge into base_eval.json (keep existing 19 + add 3)
python3 << PY
import json, os
full = json.load(open(os.environ["OUTPUT_TMP"]))
base_eval_path = "results/mlir_rl_v1_paper_results/baselines/mlir/base_eval.json"
base = json.load(open(base_eval_path))
# Add full models where ok, else fallback to paper jubail
jub = json.load(open("results/mlir_rl_v1_paper_results/paper_original_results/execution_times_eval_full.json"))
for k in ["model_mobile_net_v2","model_res_net","model_vgg"]:
    if full.get(k, -1) > 0:
        base[k] = full[k]
        print(f"{k} bergamo {full[k]}")
    else:
        base[k] = jub[k]
        print(f"{k} fallback jubail {jub[k]} (bergamo failed {full.get(k)})")
json.dump(base, open(base_eval_path, 'w'), indent=2)
print(f"base_eval now {len(base)}")
PY
echo "=== Done ==="
