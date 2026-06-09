# MLIR-RL — Agent Quick-Start

Reinforcement-learning auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC cluster, Conda env at `~/envs/mlir`.

## Must-Do Setup (interactive use only)

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/train/v4_7.json   # always needed
```

`.env` sets `PYTHONPATH`, `LD_LIBRARY_PATH` (conda + GCC-14 libstdc++), `LLVM_BUILD_PATH`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`, Neptune credentials.

Slurm scripts (`train.sh`, `eval.sh`) handle `.env` and conda activation internally. You only need the steps above for running Python directly.

## Key Rules

- **NEVER delete files without explicit user permission.**
- **NEVER cancel the interactive Slurm session** — it runs OpenCode.
- `utils.config.Config` is a singleton — reads `CONFIG_FILE_PATH` at first import.
- Import `dotenv` and load `.env` BEFORE any config imports in custom scripts.
- Use `python -m py_compile <file>` to verify — no pytest suite exists.

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

## Datasets

Two datasets exist under `data/`:

| Dataset | Files | Dtype | Format | Purpose |
|---------|-------|-------|--------|---------|
| `new_dataset/all/` | 12K+ | f32 | `{model}_{op}_{idx}.mlir` | Primary training/eval (24 NN models) |
| `paper_dataset/new_single_ops/` | 405 | f32 | `{model}_{op}_{idx}.mlir` | Paper single-op benchmarks (18 models) |
| `paper_dataset/old_paper_dataset/` | 1,202 | f64 | `{op}_{dims}.mlir` | Legacy paper data (matmul, add, conv, pooling, relu) |
| `data/lqcd/` | 155 | f64 | `{op}_{dims}.mlir` | Lattice QCD kernels + full models (moved from old_paper_dataset) |

**`new_dataset`** is the active dataset for training. Splits: `all/train/` (9,443), `all/eval/` (2,372), `all/eval_full/` (948).

**`paper_dataset`** is for paper-specific experiments. No baseline JSONs exist yet — must generate before training.

### Creating / Extending Datasets

Pipeline: `raw model → MLIR → extract blocks → baseline timing → train/eval split`

```bash
# 1. Convert model to MLIR (vision example)
python data_utils/vision2mlir.py --model resnet18 --output-dir data/new_dataset/nn/raw_bench/

# 2. Extract operation blocks
python data_utils/extract_blocks.py \
  --input data/new_dataset/nn/raw_bench/resnet18_linalg.mlir \
  --output-dir data/new_dataset/nn/code_files/bench_train/ \
  --window 5 --stride 3

# 3. Generate baseline timings
python scripts/baseline/get_base.py --benchmarks-dir data/new_dataset/all/train \
  --output results/new_dataset_results/baselines/mlir/train_base.json

# 4. Split into train/eval
python scripts/data/split_json.py config/new_dataset/train/v4_7.json
```

Key scripts: `data_utils/orchestrate.py` (unified CLI), `data_utils/extract_blocks.py`, `data_utils/extract_ops.py`, `scripts/baseline/get_base.py`.

**MLIR file requirements:** Must have `{tag = "operation_NNN"}` on linalg ops, `@nanoTime()` wrapper, weights as function args, `@main` returning `(tensor, i64)`.

## Results Directory Architecture (flat)

```
results/new_dataset_results/<agent_dir>/
├── train/
│   ├── results.json              # Cumulative {bench: {rewards, speedup, exec_time, cache_miss}}
│   └── checkpoint_100.json       # Snapshot every 100 iters
├── eval/
│   ├── checkpoint_100.json       # {bench: exec_time_ns} per eval checkpoint
│   └── markers/<ckpt>/           # Eval crash-resilience markers
├── logs/
│   ├── exec_data.json            # Execution time cache
│   ├── tags
│   ├── train/                    # entropy, reward, final_speedup
│   ├── train_ppo/                # policy_loss, value_loss, approx_kl, clip_frac
│   └── eval/                     # eval_exec_times.json + per-benchmark files
└── models/
    └── model_50.pt               # Saved every 50 iterations (not every iter)
```

## Reporting Scripts

```bash
python scripts/utils/report_training.py -v v4_6 v4_7 v4_8 v0_v2   # training progress
python scripts/utils/report_training.py -w 300                      # watch mode
python scripts/utils/report_eval.py                                 # all eval checkpoints
python scripts/utils/report_eval.py --best                          # best per agent
python scripts/utils/report_eval.py --missing                       # models without eval
```

## Training & Eval Commands

```bash
# Train from scratch (auto-resumes if models/ exist)
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json

# Resume training from an experiment directory
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json \
  --resume results/new_dataset_results/v4_7_agent

# Eval a single checkpoint
sbatch --cpus-per-task=12 --mem=16G --time=04:00:00 \
  scripts/eval/eval.sh config/new_dataset/eval/v4_7_eval.json --checkpoint 500

# Force fresh training (overwrite existing results)
FORCE_NEW=1 sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json
```

`eval.sh` uses `EVAL_DIR=<results_dir>/models/` directly — no run_N auto-discovery.
`--cpus-per-task=12` targets ~1h per eval for 2163 benchmarks.

## Lustre Quota Awareness

`/scratch` has **500K file soft limit**, **1M hard limit**. Training + evals can consume files quickly:
- Each model checkpoint = 1 file (~45MB)
- `train/results.json` accumulates ~8K entries across training
- Eval markers create 2K+ files per checkpoint eval

Before submitting many eval jobs, check `lfs quota -u $USER /scratch`. If near limit, clean with `scripts/utils/cleanup_quota.py` (thins model checkpoints, removes obsolete markers).

## Implementation Packages

| Package | Purpose |
|---------|---------|
| `rl_autoschedular_v0` | Baseline (LSTM, no HW features) |
| `rl_autoschedular_v4_5` | Integrated (Transformer + HW + shaped reward) |
| `rl_autoschedular_v45_no_hw` | Ablation: HW disabled |
| `rl_autoschedular_v45_no_shaped_reward` | Ablation: no reward shaping |
| `rl_autoschedular_v45_no_transformer` | Ablation: LSTM instead of Transformer |
| `rl_autoschedular_v1` … `v4` | Legacy versions |

V4.6/V4.7/V4.8 all use `rl_autoschedular_v4_5` impl with different configs. Each `vN` is a standalone package — never mix imports between them.

Configs live in `config/new_dataset/train/` and `config/new_dataset/eval/`.

## Speedup & Reward Gotchas

**Shaped reward misleads the agent** (V4.5 lesson): When intermediate reward dominates terminal speedup, agent optimizes static heuristics instead of execution time. The no-reward ablation outperformed all shaped-reward variants.

**Failed benchmarks:** Execution timeout → `speedup = 0.0` (not 1.0). Failed benchmarks are excluded from speedup means in reporting scripts. RL reward unaffected — it uses a flat -20.0 penalty regardless of speedup.

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

## Operational Gotchas

- **DaskManager is disabled** (`ENABLED = False`). All execution runs single-process.
- **SIGABRT handler** catches MLIR crashes so training continues past bad schedules.
- **BindingsProcess.ENABLED must stay False** — fork corrupts MLIR C++ state.
- **Model checkpoints saved every 50 iterations** (not every iteration) to limit disk usage.
- **Eval crash resilience**: markers in `run_N/eval/markers/<ckpt>/`. Restarting eval auto-skips completed benchmarks.

## Key Docs

- [Results Architecture](docs/RESULTS_ARCHITECTURE.md) — full run_i/ structure details
- [Training Guide](docs/TRAINING_GUIDE.md) — comprehensive training walkthrough
- [Pipeline](docs/PIPELINE.md) — full lifecycle: baseline → split → train → eval
- [Full Model](docs/FULL_MODEL.md) — end-to-end model optimization architecture
- [Dashboard](docs/DASHBOARD.md) — Streamlit comparison dashboard
- [HPC Setup](docs/HPC%20Setup.md) — cluster-specific Slurm instructions
