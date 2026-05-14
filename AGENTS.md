# MLIR-RL — Agent Quick-Start

Reinforcement-learning auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC cluster, Conda env at `~/envs/mlir`.

## Must-Do Environment Setup

**Interactive / local use** needs these three steps **in this order**:

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

**Slurm scripts** (`train.sh`, `eval.sh`, `get_base.sh`, `get_pytorch_times.sh`) already handle this internally — they source `.env` and activate the conda env. You only need the steps above for running Python scripts directly or using `pipeline.sh`.

The `.env` file sets:
- `PYTHONPATH` to the in-tree LLVM build's MLIR Python bindings
- `LD_LIBRARY_PATH` to the Conda env **and** a GCC-14 `libstdc++` (required on this cluster)
- `LLVM_BUILD_PATH`, `MLIR_SHARED_LIBS`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`
- Neptune credentials (`NEPTUNE_PROJECT`, `NEPTUNE_API_TOKEN`)

**Critical:** `LD_LIBRARY_PATH` must include both `~/envs/mlir/lib` and the GCC-14 path. Missing either causes `ImportError` or `GLIBCXX_3.4.29` errors.

## CONFIG_FILE_PATH — Set Before Any Python

`utils.config.Config` is a **singleton** that reads `CONFIG_FILE_PATH` **at first import**. Any Python script that touches `Config` (directly or transitively) will fail if `CONFIG_FILE_PATH` is not set before the import.

```bash
export CONFIG_FILE_PATH=config/train1.json
```

The Slurm scripts set this automatically. For custom scripts, always import `dotenv` and load `.env` first, then set `CONFIG_FILE_PATH`.

## LLVM Build Gotchas

`llvm-project/` is a full LLVM/MLIR build (release/19.x). If it was compiled by another user, the MLIR Python bindings are broken symlinks. Fix:

```bash
cd llvm-project
git checkout HEAD -- $(git ls-files mlir/python/mlir/ | tr '\n' ' ')
find build/tools/mlir/python_packages/mlir_core -type l | while read link; do
    target=$(readlink "$link")
    if echo "$target" | grep -q "OTHER_USER"; then
        new_target=$(echo "$target" | sed 's|/scratch/OTHER_USER/|/scratch/YOUR_USER/|g')
        rm "$link" && ln -s "$new_target" "$link"
    fi
done
```

Use `rm + ln -s` (not `ln -sf`) — `-f` can fail with "Permission denied" on broken symlinks to inaccessible paths.

## Implementation Packages (Versioned Agents)

| Package | Purpose |
|---|---|
| `rl_autoschedular` | Baseline — **must remain untouched** |
| `rl_autoschedular_v1` | Hardware-aware observation |
| `rl_autoschedular_v2` | Shaped reward |
| `rl_autoschedular_v3` | Transformer loop-nest encoder |
| `rl_autoschedular_v4`+ | Future novelties (one per version) |

Each `vN` is a **full standalone copy** of the baseline with internal imports redirected to itself. Do **not** mix imports between packages.

## Architecture & Entrypoints

- **`scripts/train.py`** — Main training loop (PPO). Imports the selected implementation dynamically.
- **`scripts/eval.py`** — Evaluates all `.pt` checkpoints in the latest `run_N/models/` for the selected implementation.
- **`scripts/get_base.py`** — Measures unoptimized MLIR baseline execution times.
- **`scripts/get_pytorch_times.py`** — Measures PyTorch eager / compile / JIT baselines.
- **`scripts/split_json.py`** — Splits a baseline JSON into train/eval sets (accepts config path positional or `--config`).
- **`scripts/pipeline.sh`** — One-shot full pipeline orchestration (runs locally, submits Slurm jobs for train/eval).
- **`scripts/submit_and_monitor.sh`** — Submits a Slurm job and tails its output live.
- **`dashboard/dashboard.py`** — Streamlit evaluation dashboard (multi-implementation comparison).

## Config-Driven Workflow

All scripts read `CONFIG_FILE_PATH` (env var) or accept a config path as `$1`. The config JSON selects the implementation:

```json
{
  "implementation": "rl_autoschedular_v3",
  "results_dir": "results/my_experiment",
  ...
}
```

**Implementation resolution order** (from `utils/implementation.py`):
1. `AUTOSCHEDULER_IMPL` env var
2. `--implementation` flag (where supported)
3. Config file `"implementation"` field
4. Default: `rl_autoschedular`

Train/eval Slurm scripts derive the implementation from the config. Override with a second positional arg:

```bash
sbatch scripts/train.sh config/my_config.json rl_autoschedular_v2
```

### Slurm Array Job Mode

`train.sh` and `eval.sh` support Slurm array jobs to run multiple versions:

```bash
sbatch --array=0-3 scripts/train.sh    # task 0→baseline, 1→v1, 2→v2, 3→v3
sbatch --array=0-2 scripts/eval.sh     # same mapping
```

When `SLURM_ARRAY_TASK_ID` is set and no config is provided, the script auto-selects `config/<version>.json`.

### Standard Pipeline Order

```bash
# 1. MLIR baseline (Slurm)
sbatch scripts/get_base.sh config/my_config.json

# 2. PyTorch baseline (Slurm or local)
sbatch scripts/get_pytorch_times.sh config/my_config.json

# 3. Split into train / eval
python scripts/split_json.py config/my_config.json

# 4. Train (Slurm)
sbatch scripts/train.sh config/my_config.json

# 5. Evaluate checkpoints (Slurm)
sbatch scripts/eval.sh config/my_config.json

# 6. Compare in dashboard
cd dashboard && streamlit run dashboard.py --server.fileWatcherType none
```

Or use the one-shot pipeline:
```bash
bash scripts/pipeline.sh config/baseline.json
bash scripts/pipeline.sh config/baseline.json "rl_autoschedular_v1,rl_autoschedular_v3"
```

## Results Directory Layout

```
results/<experiment>/
  exec_times/
    <prefix>_base.json          # unoptimized MLIR baseline
    <prefix>_base_train.json    # train split
    <prefix>_base_eval.json     # eval split
    pytorch.json                # PyTorch baselines
  <impl_agent>/                 # e.g. old_agent, v1_agent, v2_agent
    run_0/
      models/                   # model_4.pt, model_9.pt, ...
      logs/
        train/
        eval/
      exec_data.json
      eval/
        markers/                # crash-resilience markers
        eval_exec_times.json    # {bench_name: optimized_time_ns}
```

`<prefix>` is resolved by `utils/implementation.py`:
- `rl_autoschedular` → `old` (agent dir: `old_agent`)
- `new_rl_autoschedular` → `new` (agent dir: `new_agent`)
- `rl_autoschedular_vN` → `vN` (agent dir: `vN_agent`)

The config fields `json_file` and `eval_json_file` default to `""` — when empty, paths are auto-derived from `results_dir` and the current implementation via `get_split_file_path()`.

## Singletons & Import Order

`utils.config.Config` and `utils.dask_manager.DaskManager` are **singletons**. `Config` reads `CONFIG_FILE_PATH` at first import. Importing `utils.config` before setting `CONFIG_FILE_PATH` will fail.

`scripts/train.py` and `scripts/eval.py` load `.env` and `.env.debug` via `python-dotenv` at the very top. `.env.debug` is optional (typically absent). Custom scripts should follow the same pattern.

## Operational Gotchas

- **DaskManager is disabled by default** (`ENABLED = False` in `utils/dask_manager.py`). Distributed execution across Slurm workers only activates if you set `ENABLED = True` and `DASK_NODES` env var. All execution runs single-process on the Slurm node otherwise.
- **SIGABRT handler** — `train.py` and `eval.py` install a signal handler that catches native MLIR crashes (`SIGABRT`) and converts them to Python exceptions so training can continue past a bad schedule.
- **`submit_and_monitor.sh`** — Submits a Slurm script, waits for the job to start, then tails its output (`scripts/submit_and_monitor.sh scripts/train.sh config/train1.json`).

### Critical: BindingsProcess.ENABLED Must Stay False

`utils/bindings_process.py` has `ENABLED = False`. **Do NOT set it to True.** Reasons:

1. **Fork corrupts MLIR C++ state** — Linux `fork()` duplicates MLIR C++ globals inconsistently. Transform subprocesses fail with `exit code: 1` (12M+ errors per eval run), producing speedup=1.0 for all benchmarks.
2. **Dill-based spawn is too slow** — A dill/subprocess replacement was attempted but each subprocess launch takes 1-2s. With ~24K operations per eval, this is impractical.
3. **In-process (False) works** — Transforms + execution run correctly in-process. The SIGABRT handler catches most MLIR crashes. Some `convnext_tiny_block_*` benchmarks crash with `CollectDiagnosticsToStringScope` assertion — a reproducible MLIR binding bug in certain operation patterns.

If a benchmark crashes with SIGABRT during eval, the process dies and any in-progress eval results are lost. See **Eval Crash Resilience** below for the mitigation.

### Eval Crash Resilience

The eval saves per-benchmark results incrementally via **marker files** at `<agent>/run_0/eval/markers/<bench_name>`. Each marker is a JSON containing `rewards`, `speedup`, `exec_time`, `cache_miss`.

**How it works:**
1. Before evaluating a benchmark, `__execute_states` checks if a marker file exists → skips with "Skipped (cached)" if found
2. After successful evaluation, writes marker atomically via temp file + rename
3. If SIGABRT kills the process mid-eval, only the current benchmark is lost

**To resume a crashed eval:**
```bash
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
export AUTOSCHEDULER_IMPL=rl_autoschedular_v1 CONFIG_FILE_PATH=config/v1.json
export EVAL_LAST_ONLY=1 EVAL_DIR=results/new_experiment/v1_agent/run_0/models
# Markers survive crashes — restart auto-skips completed benchmarks
python scripts/eval.py
```

**To launch eval safely** (survives terminal closure):
```python
import subprocess, os
p = subprocess.Popen(['python', 'scripts/eval.py'],
    stdout=open('logs/eval_myrun.out','w'), stderr=open('logs/eval_myrun.err','w'),
    env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    preexec_fn=os.setpgrp)
```

**To clear markers and start fresh:** delete `<agent>/run_0/eval/markers/` directory.

### Evaluation Results

Eval saves results in multiple formats:
- `eval_exec_times.json` — `{bench_name: optimized_time_ns}` (same format as `base.json`). Saved incrementally after each benchmark.
- `eval/final_speedup` — one speedup per benchmark per checkpoint
- `eval/exec_time/<bench_name>` — per-benchmark optimized times
- `eval/markers/<bench_name>` — crash-resilience marker files

### EVAL_LAST_ONLY — Fast Single-Checkpoint Eval

Env var `EVAL_LAST_ONLY=1` evaluates only the last model checkpoint (e.g., `model_1999.pt`). For 41-checkpoint eval, omit this var.

### Slurm CPU Limits

When submitting eval jobs, use low CPU count to avoid `AssocGrpCpuLimit`:
```bash
sbatch --cpus-per-task=4 --mem=16G --time=02:00:00 ...
```

The eval script auto-discovers `EVAL_DIR` from `results_dir` + implementation. When running outside Slurm, set `EVAL_DIR` explicitly or let the script auto-discover.

### Interactive Session — NEVER Cancel

The user's Slurm interactive session (`squeue` shows as "interactive") is the session running OpenCode. **Never cancel it** — it will kill this agent's connection.

## C++ Tools

- `tools/ast_dumper/` — CMake project; build artifacts in `tools/ast_dumper/build/`
- `tools/vectorizer/` — CMake project; build artifacts in `tools/vectorizer/build/`
- `tools/pre_vec/` — Pre-vectorizer (no build dir checked in)

Referenced by env vars in `.env`.

## Testing & Validation

There is **no pytest suite**. Validation is done via:
- Python compile checks (`python -m py_compile <file>`)
- Import smoke tests (e.g. `python -c "import rl_autoschedular_v3.model"`)
- End-to-end sanity runs on a single benchmark
- Slurm job logs in `logs/`

The only test scripts are `scripts/test_torch_mlir.sh` and `scripts/test_torch_mlir_compile.py` (torch-mlir sanity checks).

## Code Style

- `.flake8` ignores `E501` (line length) and `E402` (module-level import not at top).
- No auto-formatter or type-checker configured.
- Use `from utils.log import print_info, print_success` for consistent colored CLI output.

## Quick Commands

See `quick_commands.txt` for copy-paste workflow commands. Key verification:

```bash
# Verify MLIR Python bindings work
python -c "from mlir.ir import Context; print('OK')"
```
