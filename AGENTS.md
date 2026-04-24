# MLIR-RL — Agent Quick-Start

Reinforcement-learning auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC cluster, Conda env at `~/envs/mlir`.

## Must-Do Environment Setup

Every session needs these three steps **in this order**:

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

The `.env` file sets:
- `PYTHONPATH` to the in-tree LLVM build’s MLIR Python bindings
- `LD_LIBRARY_PATH` to the Conda env **and** a GCC-14 `libstdc++` (required on this cluster)
- `LLVM_BUILD_PATH`, `MLIR_SHARED_LIBS`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`
- Neptune credentials (`NEPTUNE_PROJECT`, `NEPTUNE_API_TOKEN`)

**Critical:** `LD_LIBRARY_PATH` must include both `~/envs/mlir/lib` and the GCC-14 path. Missing either causes `ImportError` or `GLIBCXX_3.4.29` errors.

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

Use `rm + ln -s` (not `ln -sf`) — `-f` can fail with “Permission denied” on broken symlinks to inaccessible paths.

## Architecture & Entrypoints

- **`scripts/train.py`** — Main training loop (PPO). Imports the selected implementation dynamically.
- **`scripts/eval.py`** — Evaluates all `.pt` checkpoints in the latest `run_N/models/` for the selected implementation.
- **`scripts/get_base.py`** — Measures unoptimized MLIR baseline execution times.
- **`scripts/get_pytorch_times.py`** — Measures PyTorch eager / compile / JIT baselines.
- **`scripts/split_json.py`** — Splits a baseline JSON into train/eval sets.
- **`scripts/compare_baselines.py`** — Compares RL vs MLIR vs PyTorch.
- **`dashboard/dashboard.py`** — Streamlit evaluation dashboard.

### Implementation Packages (Versioned Agents)

| Package | Purpose |
|---|---|
| `rl_autoschedular` | Baseline — **must remain untouched** |
| `rl_autoschedular_v1` | Hardware-aware observation |
| `rl_autoschedular_v2` | Shaped reward |
| `rl_autoschedular_v3` | Transformer loop-nest encoder |
| `rl_autoschedular_v4`+ | Future novelties (one per version) |

Each `vN` is a **full standalone copy** of the baseline with internal imports redirected to itself. Do **not** mix imports between packages.

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
2. Config file `"implementation"` field
3. Default: `rl_autoschedular`

### Standard Pipeline Order

```bash
# 1. MLIR baseline (Slurm)
sbatch scripts/get_base.sh config/my_config.json

# 2. PyTorch baseline (local — fast enough)
python scripts/get_pytorch_times.py --config config/my_config.json

# 3. Split into train / eval
python scripts/split_json.py config/my_config.json

# 4. Train (Slurm)
sbatch scripts/train.sh config/my_config.json

# 5. Evaluate checkpoints (Slurm)
sbatch scripts/eval.sh config/my_config.json

# 6. Compare
python scripts/compare_baselines.py --config config/my_config.json
```

Train/eval Slurm scripts automatically derive the implementation from the config. You can override with a second positional arg:

```bash
sbatch scripts/train.sh config/my_config.json rl_autoschedular_v2
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
```

`<prefix>` is resolved by `utils/implementation.py`:
- `rl_autoschedular` → `old`
- `new_rl_autoschedular` → `new`
- `rl_autoschedular_vN` → `vN`

## Singletons & Import Order

`utils.config.Config` and `utils.dask_manager.DaskManager` are **singletons**. `Config` reads `CONFIG_FILE_PATH` at first import. Importing `utils.config` before setting `CONFIG_FILE_PATH` will fail or load the wrong config.

`scripts/train.py` and `scripts/eval.py` load `.env` and `.env.debug` via `python-dotenv` at the very top. Custom scripts should do the same.

## C++ Tools

- `tools/ast_dumper/` — CMake project; build artifacts in `tools/ast_dumper/build/`
- `tools/vectorizer/` — CMake project; build artifacts in `tools/vectorizer/build/`
- `tools/pre_vec/` — Pre-vectorizer (no build dir checked in)

These are referenced by env vars in `.env`.

## Testing & Validation

There is **no pytest suite**. Validation is done via:
- Python compile checks (`python -m py_compile`)
- Import smoke tests (e.g. `python -c "import rl_autoschedular_v3.model"`)
- End-to-end sanity runs on a single benchmark
- Slurm job logs in `logs/`

The only existing test scripts are `scripts/test_torch_mlir.sh` and `scripts/test_torch_mlir_compile.py` (torch-mlir sanity checks).

## Code Style

- `.flake8` ignores `E501` (line length) and `E402` (module-level import not at top).
- No auto-formatter or type-checker configured.
- Use `from utils.log import print_info, print_success` for consistent colored CLI output.

## Dashboard

```bash
cd dashboard
streamlit run dashboard.py --server.fileWatcherType none
```

The dashboard auto-discovers experiment folders under `results/` and supports multi-implementation comparison.

## Quick Commands Cheat-Sheet

See `quick_commands.txt` for copy-paste commands. Key ones:

```bash
# Submit and monitor a job
scripts/submit_and_monitor.sh scripts/train.sh config/train1.json

# Verify MLIR Python bindings work
python -c "from mlir.ir import Context; print('OK')"
```
