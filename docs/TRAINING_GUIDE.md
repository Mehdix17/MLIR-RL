# MLIR-RL Training Guide

All steps use the same config file. Start by copying `config/example.json`:

```json
{
  "benchmarks_folder_path": "data/all",
  "results_dir":            "results/my_run",
  "json_file":              "results/my_run/exec_times/old_base_train.json",
  "eval_json_file":         "results/my_run/exec_times/old_base_eval.json",
  "main_exec_data_file":    "",
  "bench_count":            32,
  "nb_iterations":          2000,
  ...
}
```

> For all available fields see `utils/config.py`.

For hardware-aware versions (for example V1+), keep the config simple on a single HPC machine:
- Set `"hardware_auto_detect": true`.
- Do not set manual `hardware_*` override fields unless you intentionally run a cross-hardware experiment.
- Hardware detection runs once per process and does not directly change execution-time measurement.

---

## Setup

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

### Select Autoscheduler Implementation

The pipeline supports two implementations:

- `rl_autoschedular` (current default)
- `new_rl_autoschedular` (enhanced implementation, when available)

Set it once per shell:

```bash
export AUTOSCHEDULER_IMPL=rl_autoschedular
# or
export AUTOSCHEDULER_IMPL=new_rl_autoschedular
```

For Slurm wrappers, you can also pass implementation as the second argument:

```bash
sbatch scripts/train.sh config/my_config.json new_rl_autoschedular
sbatch scripts/eval.sh config/my_config.json new_rl_autoschedular
sbatch scripts/get_base.sh config/my_config.json new_rl_autoschedular
```

---

## 1. Get Baseline Execution Times

**MLIR baseline** — runs each `.mlir` file unoptimized and records execution time:

```bash
python scripts/get_base.py --config config/my_config.json
# → results/my_run/exec_times/{old|new}_base.json
```

**PyTorch baselines** — measures eager, `torch.compile`, and `torch.jit` for each benchmark:

```bash
python scripts/get_pytorch_times.py --config config/my_config.json
# → results/my_run/exec_times/pytorch.json
```

On the cluster (MLIR baseline only — PyTorch is fast enough to run locally):

```bash
sbatch scripts/get_base.sh config/my_config.json
```

---

## 2. Split Train / Eval

```bash
python scripts/split_json.py config/my_config.json $AUTOSCHEDULER_IMPL
# reads:  results/my_run/exec_times/{old|new}_base.json
# writes: results/my_run/exec_times/{old|new}_base_train.json
#         results/my_run/exec_times/{old|new}_base_eval.json
```

Update `json_file` and `eval_json_file` in your config to point to these files (already set if you followed the template above).

---

## 3. Train

```bash
sbatch scripts/train.sh config/my_config.json
```

Or locally:

```bash
export CONFIG_FILE_PATH=config/my_config.json
python scripts/train.py
```

Checkpoints are saved every 5 iterations to:

- `results_dir/old_agent/run_N/models/model_<step>.pt` when `AUTOSCHEDULER_IMPL=rl_autoschedular`
- `results_dir/new_agent/run_N/models/model_<step>.pt` when `AUTOSCHEDULER_IMPL=new_rl_autoschedular`

---

## 4. Evaluate

```bash
sbatch scripts/eval.sh config/my_config.json
```

Or locally:

```bash
export CONFIG_FILE_PATH=config/my_config.json
# optional: choose a specific checkpoint directory
# export EVAL_DIR=results/my_run/old_agent/run_1/models
python scripts/eval.py
```

Results are written under the implementation run directory:

- `results_dir/old_agent/run_N/logs/eval/`
- `results_dir/new_agent/run_N/logs/eval/`

---

## 5. Compare Baselines

Compare RL agent speedup against MLIR baseline, PyTorch eager, compile, and JIT:

```bash
python scripts/compare_baselines.py --config config/my_config.json
# auto-detects latest run_N under old_agent/new_agent based on AUTOSCHEDULER_IMPL
# → results/my_run/comparison.csv
```
