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

- `PYTHONPATH` to the in-tree LLVM build’s MLIR Python bindings
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

| Package                | Purpose                              |
| ---------------------- | ------------------------------------ |
| `rl_autoschedular`     | Baseline — **must remain untouched** |
| `rl_autoschedular_v1`  | Hardware-aware observation           |
| `rl_autoschedular_v2`  | Shaped reward                        |
| `rl_autoschedular_v3`  | Transformer loop-nest encoder        |
| `rl_autoschedular_v4`  | Integrated V1 + V2 + V3              |
| `rl_autoschedular_v4_5`| **Robust Integrated** (Integration + Hardened Reliability) |
| `rl_autoschedular_v5`+ | Future novelties (one per version)   |

Each `vN` is a **full standalone copy** of the baseline with internal imports redirected to itself. Do **not** mix imports between packages.

## Architecture: AST & Feature Pipeline

The RL agent does **not** parse MLIR directly. A **C++ AST dumper** (`tools/ast_dumper/`, env var `AST_DUMPER_BIN_PATH`) serves as the bridge between raw MLIR and the Python observation tensor:

```
.mlir file  →  C++ AST dumper  →  structured text (operations, loops, graph)  →  state.py  →  BenchmarkFeatures  →  observation.py  →  torch.Tensor
```

**What the AST dumper outputs** (parsed by `state.py::__extract_bench_features_from_ast_result()`):
- `#START_OPERATION` blocks — each linalg op with its tag, name, type
- `#START_NESTED_LOOPS` — loop bounds, steps, iterator types (parallel/reduction)
- `#START_LOAD_DATA` / `#START_STORE_DATA` — memory access patterns per loop dimension
- `#START_OP_COUNT` — arithmetic op counts (mul, add, etc.)
- `#BEGIN_GRAPH` — producer→consumer edges between operations

**What the observation tensor contains** (built by `observation.py::Observation.from_state()`):
1. **OpFeatures** — operation type one-hot, loop upper bounds, iterator types, load/store access matrices, arithmetic counts
2. **ProducerOpFeatures** — same features for the producing operation (or zeros)
3. **ActionHistory** — multi-hot encoding of previously applied transformations
4. **NumLoops** — current loop nest depth
5. **ActionMask** — bitmask of currently legal actions

**Key files:** `rl_autoschedular/state.py` (feature extraction + `BenchmarkFeatures` dataclass), `rl_autoschedular/observation.py` (tensor construction), `rl_autoschedular/benchmarks.py` (bulk loading).

## Safety & Hardened Reliability (V4.5+)

Starting with V4.5, the system includes proactive safeguards against MLIR instability:

1. **Process Isolation:** All JIT transformations and execution profiling run in independent subprocesses. If a transformation sequence causes a native crash (e.g., LLVM assertion failure), the worker process dies, but the main RL training/evaluation loop remains unaffected.
2. **Success-Contingent Rewards:** To prevent the agent from learning "risky" behavior (e.g., aggressive tiling that causes timeouts), shaped rewards are only granted if the final code executes successfully. If it fails, all intermediate rewards are zeroed, and a total failure penalty is applied.
3. **Stability Rails:** The action space now masks out transformations that are statistically correlated with instability, such as vectorizing deeply nested loops (>6) or exceeding a transformation complexity threshold (>4 steps).
4. **Resilient Markers:** Progress is saved incrementally in `results/experiment3/global_markers/`. If a job is interrupted, it will automatically resume from the last completed benchmark.

## Documentation References

For more detailed guides and architectural decisions, refer to the following documents:

- [Main README config properties](README.md) — Exhaustive list of `CONFIG_FILE_PATH` fields.
- [HPC Setup Guide](docs/HPC%20Setup.md) — Slurm cluster specific instructions.
- [Training Guide](docs/TRAINING_GUIDE.md) — Comprehensive guide on training the RL agent.
- [RL Agent Tutorial](docs/RL_AGENT_TUTORIAL.md) — Walkthrough of the RL framework and logic.
- [Pipeline Orchestration](docs/PIPELINE.md) — Full lifecycle of MLIR baseline up to evaluation.
- [Dashboard Guide](docs/dashboard.md) — Streamlit evaluation instructions.
- [Data Utils](data_utils/README.md) — Tools for generating synthetic MLIR datasets and extraction operations.
- [Novelties](docs/NOVELTIES.md) and [Versions](docs/VERSIONS.md) — Changelog and upcoming version plans.

## Manuscript & Paper Citations

**CRITICAL REMINDER:** When writing or editing our current manuscript, **always cite the published or officially archived papers**. Do not cite the raw master's thesis manuscripts directly in the text.

- **Baseline Environment (2024 Focus)**
  - **Manuscript Path:** `manuscript/references/Bendib 2024.md`
  - **Always Cite the Paper:** Use the citation `bendib2024reinforcement` (Master's thesis).
- **Extended System / Our Contributions (2025 Focus)**
  - **Manuscripts Paths:** `manuscript/references/Rafik & Djad 2025.md` and `manuscript/references/Nassim & Mohamed 2025.md`
  - **Always Cite the Paper:** These converged into the arXiv preprint. Use the citation `tirichine2025reinforcement`.

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

### Benchmark Format Requirements

The RL agent works on **extracted operation blocks**, NOT full model `.mlir` files. Every `.mlir` file in `benchmarks_folder_path` must have:

- **`{tag = "operation_NNN"}`** annotations on each linalg op — required by the AST dumper for feature extraction
- **`@nanoTime()` timing calls** — measures block execution time
- **Weights passed as function arguments** (not inlined constants) — keeps files small and self-contained
- **A `func.func @main`** entry point returning `(output_tensor, i64)` (tensor + execution delta)

Example block-ready format (`data/all/albert_block_301.mlir`, 29 lines):
```
func.func @main(%arg_0_in_0: tensor<...>, ..., %arg_0_out_0: tensor<...>, ...) -> (tensor<...>, i64) {
  %t0 = call @nanoTime() : () -> i64
  %v0 = linalg.batch_matmul {tag = "operation_211"} ins(...) outs(...) -> ...
  %v1 = linalg.generic {tag = "operation_212"} ... -> ...
  %t1 = call @nanoTime() : () -> i64
  %delta = arith.subi %t1, %t0 : i64
  return %v4, %delta : tensor<...>, i64
}
```

**Full model `.mlir` files are NOT directly usable** — including those in `data/nn/raw_bench/` (e.g. `albert_linalg.mlir`, `resnet18_torch.mlir`). These lack operation tags, timing wrappers, and contain thousands of lines with inlined weight constants. They must be processed into blocks first (see below).

The `json_file` and `eval_json_file` config fields map benchmark **names** to baseline execution times (ns). Each name must match a `.mlir` file in `benchmarks_folder_path` (e.g. `"albert_block_0": 12345` → loads `benchmarks_folder_path/albert_block_0.mlir`).

### Data Pipeline: Raw Model → RL-Ready Blocks

To convert full model `.mlir` files into block-level benchmarks:

**Step 1 — Extract blocks from a linalg model file:**
```bash
# From data/nn/raw_bench/ (linalg dialect, already lowered)
python data_utils/extract_blocks.py \
    --input data/nn/raw_bench/albert_linalg.mlir \
    --output-dir data/nn/extracted_blocks/albert/

# Batch-process all linalg models
for f in data/nn/raw_bench/*_linalg.mlir; do
  python data_utils/extract_blocks.py --input "$f" --output-dir data/nn/extracted_blocks/
done
```

`extract_blocks.py` uses the AST dumper to build an operation graph, then windows consumer→producer paths into fixed-size blocks (default: 5 ops, stride 3). Output: `{model}_block_{i}.mlir` files ready for RL.

Alternative: `extract_ops.py` extracts single operations (one op per file) instead of multi-op sequences.

**Step 2 — Get baseline execution times for the new blocks:**
```bash
python scripts/get_base.py \
    --config config/v1.json \
    --benchmarks-dir data/nn/extracted_blocks/ \
    --output results/experiment3/exec_times/custom_base.json
```

**Step 3 — Split into train/eval (or use 100% for eval-only):**
```bash
python scripts/split_json.py \
    --input results/experiment3/exec_times/custom_base.json \
    --eval-ratio 1.0   # all → eval for inference-only use
```

**Step 4 — Update config to point at the new data:**
```json
{
  "benchmarks_folder_path": "data/nn/extracted_blocks",
  "eval_json_file": "results/experiment3/exec_times/custom_base_eval.json"
}
```

Alternatively, to benchmark a full model end-to-end (no RL scheduling), use `wrap_mlir.py` to add a timed `@main` wrapper, then run the resulting `.mlir` through the `Execution` class directly. There is no built-in script for full-model RL evaluation since the agent schedules per-operation-block.

### Evaluating a Trained Model on New Benchmarks

To evaluate an already-trained agent on blocks extracted from a different model set:

1. Extract blocks and get base times (steps 1-3 above)
2. Point `eval_json_file` in config to the new eval JSON, or set the file explicitly:
   ```bash
   export AUTOSCHEDULER_IMPL=rl_autoschedular_v1
   export CONFIG_FILE_PATH=config/v1.json
   export EVAL_DIR=results/experiment3/v1_agent/run_0/models
   ```
   Then run `python scripts/eval.py`. The eval script loads `Benchmarks(is_training=False)`, which uses `eval_json_file` from the config. If the config's `eval_json_file` is empty, it auto-derives from `results_dir` + implementation. Override by setting it in the config or by passing a custom data path.

3. If you only want the final checkpoint's results: `EVAL_LAST_ONLY=1 python scripts/eval.py`

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

- `tools/ast_dumper/` — **Critical bridge**: parses `.mlir` files into structured text consumed by `state.py`. Without it, the RL agent cannot extract features from MLIR. Build artifacts in `tools/ast_dumper/build/`. Env var: `AST_DUMPER_BIN_PATH`.
- `tools/vectorizer/` — Auto-vectorization analysis tool. Env var: `VECTORIZER_BIN_PATH`.
- `tools/pre_vec/` — Pre-vectorizer analysis (no build dir checked in).

Referenced by env vars in `.env`. These C++ tools are called as subprocesses from Python (not linked as libraries).

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
