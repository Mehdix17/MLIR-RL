# MLIR-RL — Agent Quick-Start

Reinforcement-learning auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC cluster, Conda env at `~/envs/mlir`.

## Must-Do Setup (interactive use only)

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/new_dataset/train/v4_7.json   # adjust per task
```

`.env` sets `PYTHONPATH`, `LD_LIBRARY_PATH` (conda + GCC-14 libstdc++), `LLVM_BUILD_PATH`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`, Neptune credentials.

Slurm scripts (`train.sh`, `eval.sh`) handle `.env` and conda activation internally. You only need the steps above for running Python directly.

## Key Rules

- **NEVER delete files without explicit user permission.**
- **NEVER cancel the interactive Slurm session** — it runs OpenCode.
- `utils.config.Config` is a singleton — reads `CONFIG_FILE_PATH` at first import.
- Import `dotenv` and load `.env` BEFORE any config imports in custom scripts.
- Use `python -m py_compile <file>` to verify — no pytest suite exists.
- `json_file` / `eval_json_file` auto-derive from `results_dir` + implementation when empty in config.
- **Never mix imports between packages** — each `rl_autoschedular_vN` is fully standalone.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Datasets

Four datasets exist under `data/`:

| Dataset | Files | Dtype | Format | Purpose |
|---------|-------|-------|--------|---------|
| `new_dataset/all/` | 12K+ | f32 | `{model}_{op}_{idx}.mlir` | Primary training/eval (24 NN models) |
| `single_ops_dataset/all/` | ~1,569 | f32 | `{model}_{op}_{idx}.mlir` | Paper single-op benchmarks (18 models) |
| `ops_and_blocks/all/` | ~8,962 | f32 | mixed | Merged single-ops + multi-op blocks |
| `lqcd/` | 155 | f64 | `{op}_{dims}.mlir` | Lattice QCD kernels + full models |

`single_ops_dataset/old_paper_dataset/` (1,202 f64 files, legacy) is no longer actively used.

### Creating / Extending Datasets

Pipeline: `raw model → MLIR → extract blocks → baseline timing → train/eval split`

```bash
# 1. Convert model to MLIR (vision example)
python data_utils/convert/vision2mlir.py --model resnet18 --output-dir data/new_dataset/nn/raw_bench/

# 2. Extract operation blocks
python data_utils/extract/extract_blocks.py \
  --input data/new_dataset/nn/raw_bench/resnet18_linalg.mlir \
  --output-dir data/new_dataset/nn/code_files/bench_train/ \
  --window 5 --stride 3

# 3. Generate baseline timings
python scripts/baseline/get_base.py --benchmarks-dir data/new_dataset/all/train \
  --output results/new_dataset_results/baselines/mlir/train_base.json

# 4. Split into train/eval
python scripts/data/split_json.py config/new_dataset/train/v4_7.json
```

Key scripts: `data_utils/orchestrate.py` (unified CLI), `data_utils/extract/extract_blocks.py`, `data_utils/extract/extract_ops.py`, `scripts/baseline/get_base.py`.

**MLIR file requirements:** Must have `{tag = "operation_NNN"}` on linalg ops, `@nanoTime()` wrapper, weights as function args, `@main` returning `(tensor, i64)`.

---

## Implementation Packages

All packages live under `rl_autoschedular/`:

| Package | Encoder | HW | Shaped Reward | Notes |
|---------|---------|-----|--------------|-------|
| `rl_autoschedular_v0` | LSTM | ❌ | ❌ | Baseline |
| `rl_autoschedular_v1` | LSTM | ✅ | ❌ | Legacy ablation |
| `rl_autoschedular_v2` | LSTM | ❌ | ✅ | Legacy ablation |
| `rl_autoschedular_v2_5` | LSTM | ❌ | ✅ | Hardened V2 (fair baseline) |
| `rl_autoschedular_v3` | Transformer | ❌ | ❌ | Legacy ablation |
| `rl_autoschedular_v4` | Transformer | ✅ | ✅ | Legacy (high failure rate) |
| `rl_autoschedular_v4_5` | Transformer | ✅ | ✅ | Integrated + robust isolation |
| `rl_autoschedular_v4_9` | Transformer | ✅ | ❌ | No shaped reward (entropy collapse fix) |
| `rl_autoschedular_v45_no_hw` | Transformer | ❌ | ✅ | Ablation: HW disabled |
| `rl_autoschedular_v45_no_shaped_reward` | Transformer | ✅ | ❌ | Ablation: no reward shaping |
| `rl_autoschedular_v45_no_transformer` | LSTM | ✅ | ✅ | Ablation: LSTM instead of Transformer |
| `rl_autoschedular_paper` | LSTM | ❌ | ❌ | Paper artifact (process-isolated) |
| `rl_autoschedular_paper_transformer` | Transformer | ❌ | ❌ | Paper ablation (Transformer encoder) |

**V4.6/V4.7/V4.8** all use `rl_autoschedular_v4_5` with different configs (corrected reward shaping).
**V4.9** is its own standalone package — shaped reward hardcoded to 0.0.
**V1–V4** are legacy; do not actively use or modify unless asked.

### Paper Packages

`rl_autoschedular_paper` and `rl_autoschedular_paper_transformer` are identical except for the encoder:
- **paper**: `LSTMEmbedding` (2-layer LSTM over consumer+producer)
- **paper_transformer**: `TransformerEmbedding` (self-attention over loop tokens, CLS pooling)

Both use `interchange_mode = "pointers"`, no HW features, no shaped reward, process-isolated execution.

Paper packages write directly to `results_dir/run_N/` (same `run_N` structure as all other packages — no special flat layout).

---

## Config Structure

Configs live under `config/<dataset>/<train|eval>/`:

| Dataset | Train configs | Eval configs |
|---------|--------------|-------------|
| `new_dataset` | `config/new_dataset/train/v0.json`, `v4_6.json`, `v4_7.json`, `v4_8.json`, `v4_9.json`, ablation variants | `config/new_dataset/eval/` |
| `single_ops_dataset` | `config/single_ops_dataset/train/v0.json`, `v4_9_small.json`, `v4_9_large.json`, `paper.json` | `config/single_ops_dataset/eval/` |
| `ops_and_blocks` | `config/ops_and_blocks/train/paper_original.json`, `paper_transformer_small.json`, `paper_transformer_large.json`, `v0.json`, `v4_9_small.json`, `v4_9_large.json` | `config/ops_and_blocks/eval/` |
| `paper` (single_ops) | `config/paper/single_ops_dataset/paper_original_train.json`, `paper_transformer_{small,large}_train.json` | `config/paper/single_ops_dataset/*_eval.json` |

---

## Results Directory Architecture

### Main packages (v0, v4_5, v4_9, ablations, paper, paper_transformer)

```
results/<experiment>/<agent_dir>/run_N/
├── train/
│   ├── results.json              # Cumulative {bench: {rewards, speedup, exec_time, cache_miss}}
│   └── checkpoint_100.json       # Snapshot every 100 iters
├── eval/
│   └── checkpoint_100.json       # {bench: exec_time_ns} per eval checkpoint
├── logs/
│   ├── exec_data.json            # Execution time cache
│   ├── tags
│   ├── train/                    # entropy, reward, final_speedup
│   ├── train_ppo/                # policy_loss, value_loss, approx_kl, clip_frac
│   └── eval/                     # eval_exec_times.json + per-benchmark files
└── models/
    └── model_50.pt               # Saved every 50 iterations (not every iter)
```

`FORCE_RUN_ID` env var controls run directory: `FORCE_RUN_ID=5` → `run_5/` (reuse), `FORCE_RUN_ID=ckpt_100` → temp dir.

---

## Reporting Scripts

```bash
# Training and Evaluation progress scripts
python scripts/utils/report_training.py -d ops_and_blocks         # training progress table
python scripts/utils/report_training.py -v v4_6 v4_7 v4_8 v0_v2   # training progress (v4/v0 configs)
python scripts/utils/report_training.py -w 300                      # watch mode
python scripts/utils/report_eval.py                                 # all eval checkpoints
python scripts/utils/report_eval.py --best                          # best per agent
python scripts/utils/fast_report.py -d ops_and_blocks               # accelerated unified report (0.1s)

# Unified report-progress skill (AI slash command)
# Inside the Antigravity TUI, type: /report-progress
# Runs fast_report.py to show active Slurm jobs, training, evaluations, and Lustre quota.
```

---

## Training & Eval Commands

```bash
# Train from scratch (auto-resumes if models/ exist)
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json

# Resume training from a run directory
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json \
  --resume results/new_dataset_results/v4_7_agent/run_0

# Submit a large evaluation batch job for a single agent version (30 checkpoints, time limit 3 days)
python scripts/eval/submit_eval.py paper_transformer_small 7300 10200 100 --time 3-00:00:00

# Synchronize evaluation progress from Slurm and actual outputs
python scripts/eval/sync_progress.py

# Eval a single checkpoint manually
sbatch --cpus-per-task=12 --mem=16G --time=04:00:00 \
  scripts/eval/eval.sh config/new_dataset/eval/v4_7_eval.json --checkpoint 500

# Force fresh training (overwrite existing results)
FORCE_NEW=1 sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json

# Paper packages (ops_and_blocks dataset)
sbatch scripts/train/train.sh config/ops_and_blocks/train/paper_original.json
sbatch scripts/train/train.sh config/ops_and_blocks/train/paper_transformer_small.json
sbatch scripts/train/train.sh config/ops_and_blocks/train/paper_transformer_large.json
```

`eval.sh` uses `EVAL_DIR=<results_dir>/run_N/models/` — auto-discovers latest `run_N`.
`--cpus-per-task=12` targets ~1h per eval for ~1,600–2,163 benchmarks.

---

## Lustre Quota Awareness

`/scratch` has **500K file soft limit**, **1M hard limit**. Training + evals can consume files quickly:
- Each model checkpoint = 1 file (~45MB)
- `train/results.json` accumulates ~8K entries across training

Before submitting many eval jobs, check `lfs quota -u $USER /scratch`. If near limit, notify user.

---

## Speedup & Reward Gotchas

**Entropy collapse** (V4.x lesson): Shaped reward + Transformer causes policy to collapse to zero entropy mid-training. Once entropy = 0, PPO gradient vanishes — no recovery. V0 (LSTM, no shaped reward) does not collapse. **Fix**: disable shaped reward (V4.9) or increase `entropy_coef` to 0.05+.

**Shaped reward misleads the agent**: When intermediate reward dominates terminal speedup, agent optimizes static heuristics (parallelism ratio, vectorizability) instead of actual execution time. The no-shaped-reward ablation outperformed all shaped-reward variants.

**Failed benchmarks:** Execution timeout → `speedup = 0.0` (not 1.0). Failed benchmarks are excluded from speedup means in reporting scripts. RL reward unaffected — it uses a flat -20.0 penalty regardless of speedup.

**Reward shaping scale**: Must be ≤10% of terminal reward magnitude. Correct values: `reward_shaping_scale=0.05`, `reward_shaping_clip=0.1`, `reward_shaping_vectorization_bonus=0.0`.

---

## Execution Safety Mechanisms

All safety features from V4.9 are now ported to the paper packages. Current status:

| Mechanism | V4.9 | paper | paper_transformer |
|-----------|:----:|:-----:|:-----------------:|
| SIGABRT handler (train entry point) | ✅ `scripts/train/train.py` | ✅ shared | ✅ shared |
| SIGABRT handler (eval entry point) | ✅ `scripts/eval/eval.py` | ✅ `evaluate.py` | ✅ `evaluate.py` |
| Process-isolated MLIR execution | ✅ | ✅ ported | ✅ ported |
| Dynamic timeout (`root_exec_time × 5`) | ✅ | ✅ ported | ✅ ported |
| mlir-cpu-runner subprocess fallback | ✅ | ✅ ported | ✅ ported |
| SIGABRT guard in `Benchmarks.__init__` | ✅ | ✅ | ✅ |
| TiledFusion constant dim skip (`continue`) | ✅ | ✅ | ✅ |

**`BindingsProcess.ENABLED` must stay `False`** — fork corrupts MLIR C++ state.
**DaskManager is disabled** (`ENABLED = False`). All execution runs on a single compute node. We enabled a local `ThreadPoolExecutor` parallel fallback in the paper packages (matching other versions) to run benchmark executions in parallel. To prevent Out-Of-Memory (OOM) failures during parallel compilation/execution, jobs must be submitted requesting 12 CPUs and 32 GB memory (`--cpus-per-task=12 --mem=32G`).

---

## LLVM Build Gotchas

If `llvm-project/` was compiled by another user, MLIR Python bindings are broken symlinks:

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

Use `rm + ln -s` (not `ln -sf`) — `-f` fails on broken symlinks to inaccessible paths.

---

## Key Docs

- [Training & Evaluation Guide](docs/pipeline/TRAINING_AND_EVALUATION.md) — end-to-end training and evaluation workflow, resources, and commands
- [Results Architecture](docs/results/RESULTS_ARCHITECTURE.md) — full `run_N/` structure, FileLogger, crash resilience
- [Training Guide](docs/pipeline/TRAINING_MANUAL.md) — comprehensive training walkthrough
- [Pipeline](docs/pipeline/PIPELINE.md) — full lifecycle: baseline → split → train → eval
- [Versions](docs/design/VERSIONS.md) — version-by-version changelog and validation notes
- [Dashboard](docs/results/DASHBOARD.md) — Streamlit comparison dashboard
- [Evaluation Tracker](docs/results/eval_progress.md) — Live evaluation progress tracker for Slurm jobs and checkpoints
- [Entropy Collapse Investigation](docs/investigations/ENTROPY_COLLAPSE_INVESTIGATION.md) — root cause, timeline, recommended fixes
- [Results](docs/results/RESULTS.md) — experimental results (single_ops_dataset + ops_and_blocks)
- [Paper Eval Pipeline Analysis](docs/paper/EVAL_PIPELINE_ANALYSIS.md) — SIGABRT safety mechanisms, paper vs V4.9 comparison
- [Paper Train Failures 2026-06-24](docs/archive/TRAIN_FAILURES_2026_06_24.md) — ops_and_blocks bugs fixed (TiledFusion, dead code, Benchmarks guard)

---

## Custom AI Skills

Custom slash commands designed to accelerate AI agent workflow inside the Antigravity TUI:

* **`/report-progress`**: Unified and accelerated progress reporting (runs [scripts/utils/fast_report.py](file:///scratch/mb10856/MLIR-RL/scripts/utils/fast_report.py) concurrently in `~0.1s` via threads). Queries active Slurm jobs, training checkpoints, evaluation summaries, and Lustre storage quota in one call.
* **`/commit`**: Conventional git helper that stages changes, performs branch safety check (aborts on `main` or `master`), prompts for target branch, commits with conventional formatting, pushes to remote, and suggests PR details.
