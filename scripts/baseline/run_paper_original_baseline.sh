#!/bin/bash
#SBATCH --job-name=paper_orig_base
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/paper_original_baseline_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/paper_original_baseline_%j.err

set -e
set -x
PROJECT_ROOT="/scratch/mb10856/MLIR-RL"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

mkdir -p results/mlir_rl_v1_paper_results/baselines/mlir/archive_jubail
cp results/mlir_rl_v1_paper_results/baselines/mlir/base_train.json results/mlir_rl_v1_paper_results/baselines/mlir/archive_jubail/base_train.jubail.json 2>/dev/null || true
cp results/mlir_rl_v1_paper_results/baselines/mlir/base_eval.json results/mlir_rl_v1_paper_results/baselines/mlir/archive_jubail/base_eval.jubail.json 2>/dev/null || true

source "$HOME/envs/mlir/bin/activate"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Started $(date) on $(hostname) ==="

# Build filtered dir with only the 1269 paper benches (not all 1354)
FILTERED_DIR="/tmp/paper_original_filtered_$$"
export FILTERED_DIR
mkdir -p "$FILTERED_DIR"
python3 << PY
import json, pathlib
train = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/base_train.json')).keys())
eval_ = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/base_eval.json')).keys())
all_names = train | eval_
src_dir = pathlib.Path('data/mlir_rl_v1_paper')
for n in all_names:
    src = src_dir / f"{n}.mlir"
    dst = pathlib.Path(f"$FILTERED_DIR") / f"{n}.mlir"
    # Use FILTERED_DIR from shell via env
    import os
    dst = pathlib.Path(os.environ.get("FILTERED_DIR", "/tmp/paper_original_filtered")) / f"{n}.mlir"
    if src.exists() and not dst.exists():
        dst.symlink_to(src.resolve())
print(f"filtered: {len(list(pathlib.Path(os.environ.get('FILTERED_DIR','/tmp/paper_original_filtered')).glob('*.mlir')))}")
PY

# Simpler: recreate cleanly in bash
rm -rf "$FILTERED_DIR"
mkdir -p "$FILTERED_DIR"
python3 << PY2
import json, pathlib, os
filtered = os.environ["FILTERED_DIR"]
train = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/base_train.json')).keys())
eval_ = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/base_eval.json')).keys())
all_names = train | eval_
for n in all_names:
    src = pathlib.Path('data/mlir_rl_v1_paper') / f'{n}.mlir'
    dst = pathlib.Path(filtered) / f'{n}.mlir'
    if src.exists():
        try: dst.symlink_to(src.resolve())
        except: pass
print(len(list(pathlib.Path(filtered).glob('*.mlir'))))
PY2

# Now run get_base.py on filtered dir
OUTPUT_TMP="results/mlir_rl_v1_paper_results/baselines/mlir/.tmp_baseline_${SLURM_JOB_ID}.json"
export OUTPUT_TMP
python scripts/baseline/get_base.py --benchmarks-dir "$FILTERED_DIR" --output "$OUTPUT_TMP" --implementation rl_autoschedular_v5 --timeout 15

echo "=== Raw output ==="
python3 -c "import json,os; d=json.load(open(os.environ.get("OUTPUT_TMP"))); print(f'total {len(d)}, ok {sum(1 for v in d.values() if v>0)}, failed {sum(1 for v in d.values() if v<=0)}'); print([k for k,v in d.items() if v<=0][:10])"

# Split back into train/eval using paper lists
python3 << PY
import json, os
all_data = json.load(open(os.environ["OUTPUT_TMP"]))
train_names = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/archive_jubail/base_train.jubail.json')).keys())
eval_names = set(json.load(open('results/mlir_rl_v1_paper_results/baselines/mlir/archive_jubail/base_eval.jubail.json')).keys())
# Build new train/eval from fresh timings, keeping only paper names
train_new = {k: all_data[k] for k in train_names if k in all_data}
eval_new = {k: all_data[k] for k in eval_names if k in all_data}
print(f"train_new {len(train_new)} (failed {sum(1 for v in train_new.values() if v<=0)})")
print(f"eval_new {len(eval_new)} (failed {sum(1 for v in eval_new.values() if v<=0)})")
# Check failures - keep jubail fallback for those?
failed_train = [k for k,v in train_new.items() if v<=0]
failed_eval = [k for k,v in eval_new.items() if v<=0]
if failed_train: print(f"failed train: {failed_train[:5]}")
if failed_eval: print(f"failed eval: {failed_eval[:5]}")
# Write new baselines (filter to >0 only)
train_ok = {k:v for k,v in train_new.items() if v>0}
eval_ok = {k:v for k,v in eval_new.items() if v>0}
json.dump(train_ok, open('results/mlir_rl_v1_paper_results/baselines/mlir/base_train.json','w'), indent=2)
json.dump(eval_ok, open('results/mlir_rl_v1_paper_results/baselines/mlir/base_eval.json','w'), indent=2)
print(f"wrote train {len(train_ok)} eval {len(eval_ok)}")
PY

echo "=== Done $(date) ==="
