# MLIR-RL — Agent Quick-Start

RL auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC, Conda env at `~/envs/mlir`.

## Setup (interactive use only)

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/new_dataset/train/v4_7.json   # adjust per task
```

Slurm scripts (`train.sh`, `eval.sh`) handle `.env` and conda internally.

## Hard Rules

- **NEVER delete files without explicit permission.**
- **`data/` is the irreplaceable, untracked dataset** (`data/` is gitignored via `*`, so nothing under it is recoverable from git or GitHub). **Never** `git clean`, `rm -rf data/`, or include `data/` in any cleanup sweep — even when asked to "clean up".
- **"Cleanup <branch>" means `origin/<branch>` (the remote), not the local working tree.** When asked to clean a branch, operate on the remote's tracked content; never remove untracked local files (dataset, results, logs) as part of it. If scope is ambiguous, ask.
- **NEVER cancel the interactive Slurm session** — it runs the CLI coding tool.
- **Never mix imports between packages** — each `rl_autoschedular_vN` is fully standalone.
- `utils.config.Config` is a singleton — reads `CONFIG_FILE_PATH` at first import.
- Load `.env` BEFORE any config imports in custom scripts.
- Verify code with `python -m py_compile <file>` — no pytest suite.
- `json_file` / `eval_json_file` auto-derive from `results_dir` + implementation when empty.

## Behavioral Guidelines

**Think before coding:** State assumptions. If multiple interpretations exist, present them. If something is unclear, stop and ask.

**Simplicity first:** Minimum code that solves the problem. No speculative features, no abstractions for single-use code. If 200 lines could be 50, rewrite.

**Ponytail (always-on when writing code):** The ponytail plugin (`/ponytail`, version 4.9.0, installed via `hermes plugins install DietrichGebert/ponytail --enable`) is the enforcement of "lazy senior dev" minimalism. When writing or reviewing code, follow its ruleset: reach for the simplest existing primitive (native `<input type="date">`, an existing util, the standard library) before adding a dependency or a wrapper component; delete dead code; prefer the 1-line fix. Levels: `/ponytail lite|full|ultra|off`. Use `/ponytail-review`, `/ponytail-audit`, `/ponytail-debt`, `/ponytail-gain`, `/ponytail-help` for reviews/metrics.

**rtk (always-on when running commands):** Route output-heavy shell commands through the RTK token-optimizing proxy — `rtk git status`, `rtk ls`, `rtk tree`, `rtk err <cmd>`, `rtk json < file`, `rtk test` — to compress command output before it enters context. Binary: `~/.local/bin/rtk` (v0.44.2). `rtk proxy <cmd>` runs the raw unfiltered command when output looks wrong; `rtk gain` shows token savings. See `.agents/rules/rtk.md`. Note: `read_file` / `search_files` remain the primary file-reading tools.

**Surgical changes:** Touch only what you must. Match existing style. Don't refactor things that aren't broken. Every changed line should trace to the user's request.

**Goal-driven:** Define success criteria. For multi-step tasks, state a brief plan with verification per step. Loop until verified.

---

## Datasets

| Dataset | Files | Dtype | Purpose |
|---------|-------|-------|---------|
| `new_dataset/all/` | 12K+ | f32 | Primary training/eval (24 NN models) |
| `single_ops_dataset/all/` | ~1,569 | f32 | Paper single-op benchmarks (18 models) |
| `ops_and_blocks/all/` | ~8,962 | f32 | Merged single-ops + multi-op blocks |
| `lqcd/` | 155 | f64 | Lattice QCD kernels + full models |

Pipeline: `raw model → MLIR → extract blocks → baseline timing → train/eval split`. Key scripts: `data_utils/orchestrate.py`, `data_utils/extract/extract_blocks.py`, `scripts/baseline/get_base.py`, `scripts/data/split_json.py`.

**MLIR file requirements:** `{tag = "operation_NNN"}` on linalg ops, `@nanoTime()` wrapper, weights as function args, `@main` returning `(tensor, i64)`.

## Packages

All under `rl_autoschedular/`. Each is fully standalone (no cross-package imports).

| Package | Encoder | HW | Shaped Reward | Notes |
|---------|---------|-----|:---:|-------|
| `v0` | LSTM | ❌ | ❌ | Baseline |
| `v4_5` | Transformer | ✅ | ✅ | Integrated + robust isolation |
| `v4_9` | Transformer | ✅ | ❌ | Entropy collapse fix |
| `paper` | LSTM | ❌ | ❌ | Paper artifact |
| `paper_transformer` | Transformer | ❌ | ❌ | Paper ablation |
| `v5` | Transformer | ❌ | ❌ | V5 platform: CPU-only (no GPUOccupier), no eval-in-training; base for V5.1/V5.2 |

V4.6/V4.7/V4.8 use `v4_5` with different configs. V1–V4 are legacy. Ablations: `v45_no_hw`, `v45_no_shaped_reward`, `v45_no_transformer`.

Paper packages: `interchange_mode="pointers"`, no HW features, no shaped reward, process-isolated. `paper` uses `LSTMEmbedding`, `paper_transformer` uses `TransformerEmbedding` (self-attention, CLS pooling).

## Commands

```bash
# Train
sbatch scripts/train/train.sh config/<dataset>/train/<config>.json
sbatch scripts/train/train.sh <config> --resume results/.../run_0   # resume
FORCE_NEW=1 sbatch scripts/train/train.sh <config>                   # fresh

# Train (V5+ unified config: one JSON drives train AND eval)
sbatch scripts/train/train.sh config/v5/v5_small.json

# Eval
sbatch --cpus-per-task=12 --mem=16G scripts/eval/eval.sh <eval_config> --checkpoint 500
sbatch scripts/eval/eval.sh config/v5/v5_small.json --checkpoint 500   # unified config (64c/100G defaults)
python scripts/eval/submit_eval.py paper_transformer_small 7300 10200 100 --time 3-00:00:00
python scripts/eval/sync_progress.py

# Reporting
python scripts/utils/fast_report.py -d ops_and_blocks               # unified (0.1s)
python scripts/utils/report_training.py -v v4_6 v4_7 v4_8            # training progress
python scripts/utils/report_eval.py --best                          # best per agent
```

`eval.sh` auto-discovers latest `run_N` from `results_dir/run_N/models/`.

## Key Gotchas

**Entropy collapse:** Shaped reward + Transformer → policy collapses to zero entropy. Fix: disable shaped reward (V4.9) or `entropy_coef ≥ 0.05`.

**Failed benchmarks:** Timeout → `speedup = 0.0`. Excluded from speedup means. RL reward = flat `-20.0` penalty.

**Reward shaping:** Must be ≤10% of terminal reward. Correct: `scale=0.05, clip=0.1, vectorization_bonus=0.0`.

**`BindingsProcess.ENABLED` must stay `False`** — fork corrupts MLIR C++ state.

**DaskManager disabled** — `ThreadPoolExecutor` fallback uses `SLURM_CPUS_PER_TASK` workers. Set `--cpus-per-task` to match the node (128 on Jubail, 64-128 on C2 GPU nodes). See [V5 Training Acceleration](docs/design/done/v5_training_acceleration.md).

**Lustre:** `/scratch` has 500K file soft limit. Check `lfs quota -u $USER /scratch` before large eval batches.

## Results Layout

```
results/<experiment>/<agent_dir>/run_N/
├── train/        results.json, checkpoint_100.json
├── eval/         checkpoint_100.json ({bench: exec_time_ns})
├── logs/         exec_data.json, train/, train_ppo/, eval/
└── models/       model_50.pt (every 50 iters)
```

`FORCE_RUN_ID=N` → `run_N/`. `FORCE_RUN_ID=ckpt_N` → temp dir.

## HPC Hardware

Training runs on **Jubail standard nodes** (`compute` partition, 128 CPU cores, 480GB RAM, no GPU).
GPU + CPU jobs run on **C2 QOS** (`nvidia` partition, A100/H100/H200 + 128 CPU cores on same node).
See [HPC Hardware](docs/hpc/HPC_HARDWARE.md) and [C2 Guide](docs/hpc/Guide%20to%20Using%20C2%20Machines.md).

## Key Docs

**Pipeline:** [Training & Eval](docs/pipeline/TRAINING_AND_EVALUATION.md) · [Training Manual](docs/pipeline/TRAINING_MANUAL.md) · [Pipeline](docs/pipeline/PIPELINE.md)
**Results:** [Architecture](docs/results/RESULTS_ARCHITECTURE.md) · [Results](docs/results/RESULTS.md) · [Dashboard](docs/results/DASHBOARD.md) · [Eval Tracker](docs/results/eval_progress.md)
**Design:** [Versions](docs/design/VERSIONS.md) · [CONFIG](docs/design/CONFIG.md) · [NOVELTIES](docs/design/NOVELTIES.md)
**Investigations:** [Entropy Collapse](docs/investigations/ENTROPY_COLLAPSE_INVESTIGATION.md) · [Paper Eval Pipeline](docs/paper/EVAL_PIPELINE_ANALYSIS.md) · [Train Failures](docs/archive/TRAIN_FAILURES_2026_06_24.md)
**HPC:** [Hardware](docs/hpc/HPC_HARDWARE.md) · [C2 Guide](docs/hpc/Guide%20to%20Using%20C2%20Machines.md)

### Design Docs

`docs/design/done/` — completed features. `docs/design/todo/` — planned features (V5 generation: V5 platform → V5.1 full-model eval → V5.2 action space):
- [V5 Training Acceleration](docs/design/done/v5_training_acceleration.md) — V5 platform: GPU-ready pipeline, CPU parallelism, resource allocation, GPUOccupier wiring
- [V5.1 Full-Model Evaluation](docs/design/todo/v5_1_full_model_eval.md) — block-trained policy → full `.mlir` eval (reuses `scripts/checkpoint/ckpt_scan*`)
- [V5.2 Expanded Action Space](docs/design/todo/v5_2_expanded_action_space.md) — padding, unrolling, packing, LICM, fusion
- [HPO Plan](docs/design/todo/HPO_PLAN.md) — hyperparameter tuning (candidate V5.3)

New feature design → `docs/design/todo/<feature>.md`. Move to `done/` when implemented.

---

## Custom AI Skills

Skills are stored in `.agents/skills/`. All skills can call other skills (e.g. `/graphify`) at any point.

### Feature Pipeline (3 Phases)

```
Phase 1: feature-brainstorm    →  docs/design/todo/<feature>-brainstorm.md
Phase 2: feature-architect     →  docs/design/todo/<feature>.md
Phase 3: feature-develop       →  implementation + move doc to docs/design/done/
```

- **`feature-brainstorm`** (Phase 1): Interview the user. Adaptive — broad if they have nothing, structuring if they have a concept. Uses `graphify query` to explore codebase. Outputs a brainstorm doc.
- **`feature-architect`** (Phase 2): Read brainstorm doc, investigate deeper via `graphify query`, resolve open questions, produce design doc with task breakdown. MLIR-RL conventions: package isolation, config singleton, Slurm resources, `py_compile`.
- **`feature-develop`** (Phase 3): Read design doc, implement tasks in order, verify with `py_compile` + smoke test + `--resume`, check Lustre quota, move doc to `done/`.

### Utility Skills

- **`/report-progress`**: Active Slurm jobs, training, evals, Lustre quota in one call (`scripts/utils/fast_report.py`).
- **`/commit`**: Conventional git helper with branch safety check.
- **`/graphify`**: Query `graphify-out/graph.json` for codebase/architecture questions. Use before reading files.
- **`plot-experimentation-results`**: Generate line evolution charts and comparison plots from training/eval results.
- **`research-paper-writing`**: Scientific writing assistant for the MLIR-RL paper — structure, prose, figures, and LaTeX.