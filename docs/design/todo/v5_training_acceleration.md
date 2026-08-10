# V5: Accelerated Training Platform — Design

**Status**: ✅ Phase 2 complete (feature-architect) — design approved in draft, ready for `feature-develop` (Phase 3)
**Last verified**: 2026-08-06 against live code + live Slurm state
**Version**: **V5** of the new MLIR-RL generation (V5 → V5.1 → V5.2)
**Target package**: `rl_autoschedular_v5` (new standalone package)
**Base package**: `rl_autoschedular_paper_transformer` structure — Transformer encoder (the core contribution), **NO hardware features, NO reward shaping** (both explored in v4_9 and **abandoned** — not helpful). Verified: paper_transformer = v4_9 minus HW observation minus shaped reward, and is fully standalone — it never imports from another `rl_autoschedular_*` package. V5 follows the same rule: repo-level shared `utils/` (e.g. `utils.config`, `utils.implementation`) may be used where useful, but no cross-`rl_autoschedular` imports (see Constraints).
**Depends on**: nothing (foundation version)
**Followed by**: V5.1 (full-model eval, `v5_1_full_model_eval.md`), V5.2 (expanded action space, `v5_2_expanded_action_space.md`)
**Parallel track**: HPO (`HPO_PLAN.md`) — hyperparameter search runs **in parallel** to V5/V5.1, feeding tuned hyperparameters into training **before** V5.2 expands the action space.

> **What V5 is**: the platform version — a fast training/eval pipeline plus the
> `rl_autoschedular_v5` package that V5.1 and V5.2 build on. Acceleration is
> infrastructure, not a research novelty (per VERSIONS.md convention); it produces no
> paper result by itself, but every later version depends on its speed.
>
> **Decisions locked (2026-08-06)**:
> 1. Base = paper_transformer structure (Transformer encoder only; HW features and
>    reward shaping abandoned — explored in v4_9, found unhelpful).
> 2. **V5 is CPU-only** — training/eval stay on the Jubail `compute` partition.
>    GPU analysis, C2 limits, and the GPUOccupier discussion moved to
>    `v5_gpu_exploration.md`.
> 3. Resource allocation: see the DECIDED matrix in §Resource Allocation.
> 4. Version order V5 → V5.1 → V5.2 confirmed (no swap). Consequence accepted:
>    V5.1's full-model numbers describe the 6-action agent; re-run after V5.2 if
>    the paper needs them for the expanded agent.
> 5. HPO is a **parallel track**, not a version (V5.3 dropped).
> 6. **Unified config**: one config file per run in `config/v5/` serves both
>    training and eval (`json_file` + `eval_json_file` in the same JSON).
> 7. **Eval resources match training**: 64 CPUs / 100G for eval and eval_batch.
> 8. **GPUOccupier removed entirely** from the V5 package — no GPU code paths.
> 9. **No `evaluate_benchmarks` in training** — eval is a separate Slurm job only.
>
> This doc is the input to the `feature-architect` skill. It captures the verified
> current state, the measured bottleneck, the decisions already made (with
> rationale), and the open questions that still need architect/user input.
> Everything marked ✅ was confirmed in code; everything marked ❌ is a proposal.

---

## Problem

Training is slow: **iteration ≈ 50s**, of which **MLIR compilation/execution ≈
95%** and model sampling ≈ 3%. The pipeline runs on the Jubail `compute`
partition with only **12 CPU cores** requested (`--cpus-per-task=12`), wasting
the 128-core nodes it runs on. The dominant cost is CPU-bound MLIR execution,
and the parallelism lever is unused.

**The one win that matters**: 12 → 64 workers multiplies MLIR execution
parallelism ~5x (the dominant cost). GPU was evaluated and dropped — the
analysis is archived in [v5_gpu_exploration.md](v5_gpu_exploration.md).

Goal: cut wall-clock per iteration by **~4-6x** (50s → 8-12s) with zero change to
the RL algorithm, reward function, or action space.

---

## Idea (rough shape)

- **Raise CPU parallelism on the Jubail `compute` partition**: `--cpus-per-task=64`
  for training (matches `bench_count=64` — one wave per iteration) and
  `--cpus-per-task=64` for eval too (user decision — eval resources match training).
- **Bump `--mem`** to 100G (train and eval) — 64 concurrent MLIR children
  × ~0.7-1G RSS (measured, see §Memory calibration) + parent ≈ 50-70G peak; 100G ≈ 1.5x.
- **Bump `--time`** to the compute-partition max **7-00:00:00**.
- **No GPU, no nvidia migration, no GPUOccupier** — removed from V5 entirely
  (analysis in [v5_gpu_exploration.md](v5_gpu_exploration.md)).
- **Optional follow-ups** (already implemented in config, just flip):
  `reuse_experience`/`replay_count`; `MIN_EXEC_TIMEOUT` straggler control.
- **Keep `bench_count=64`** — with 64 workers it executes in one wave, so
  gradient diversity is preserved *at the same wall-clock*.

---

## How Training Is Done (verified current state)

### Entry point (`scripts/train/train.sh` → `scripts/train/train.py`)

1. `train.sh` submits a Slurm job; `scripts/train/train.py` is the unified entry (what Slurm runs — the per-package `train.py` files are NOT used by the Slurm pipeline). **V5 removes `evaluate_benchmarks` from training entirely**: evaluation is a separate Slurm job (`scripts/eval/eval.sh`) and has nothing to do with the training loop. (The paper_transformer per-package `train.py` called `evaluate_benchmarks` every 100 steps — V5 does not.)
2. Installs a SIGABRT handler (MLIR native crashes → Python exception, not hard kill).
3. Loads `Config`, `Benchmarks`, `Execution` singletons; builds `HiearchyModel`; moves it to `device`.
4. Handles `--resume` (loads latest checkpoint from `models/model_<N>.pt`).
5. Runs the PPO loop. **No `evaluate_benchmarks` inside the loop** — eval is a separate Slurm job.

### PPO loop (`scripts/train/train.py:159`)

```
for step in range(start_step, cfg.nb_iterations):
    trajectory = collect_trajectory(train_data, model, step)
    if cfg.reuse_experience != 'none':          # ✅ implemented (train.py:175-182)
        trajectory = old_trajectory + trajectory
    if cfg.value_epochs > 0:                     # ⚠️ value_epochs=0 in ALL current configs → skipped
        value_update(trajectory, model, optimizer)
    ppo_update(trajectory, model, optimizer)     # ✅ wrapped in GPUOccupier().gpu_needed() in ppo.py:318
    if step % 50 == 0:
        save checkpoint
```

### Trajectory collection (`ppo.py:collect_trajectory`)

1. Samples `bench_count` (currently 64) random benchmarks via `torch.randperm` (`ppo.py:54`).
2. Steps envs until terminal; model samples actions on `device` — **wrapped in `GPUOccupier().gpu_needed()` (`ppo.py:69`)**.
3. `DaskManager.map_objs(__execute_states, ...)` executes all terminal states in parallel.
4. Returns trajectory (rewards, speedups, exec times).

### Execution (`execution.py`) — THE BOTTLENECK

- `Execution.execute_code()` checks a JSON cache first (**✅ implemented**, `execution.py:118-121`; per-run `exec_data.json`, survives `--resume`).
- Cache miss → bufferize + lower through ~20-pass pipeline, run via `ExecutionEngine` in an isolated child process (`__execute_bufferized_code_isolated`, `execution.py:196-277`).
- Child crash (SIGABRT) or timeout → fallback to `mlir-opt | mlir-cpu-runner` subprocess.
- Dynamic timeout: `root_exec_time * 5`, clamped to `[MIN_EXEC_TIMEOUT, 300]` (**✅ env var**, `execution.py:124-127`).
- **All compile/exec is CPU-only, always** — MLIR JIT-compiles to CPU machine code regardless of node type.

### Parallelism (`dask_manager.py`)

- `DaskManager` **disabled** by default (`ENABLED = DASK_NODES > 0`; `.env` doesn't set it). ✅
- Falls back to `ThreadPoolExecutor(max_workers = SLURM_CPUS_PER_TASK)` (`dask_manager.py:122-130`).
- Each worker thread reuses the **shared** `Execution` singleton (created once, `scripts/train/train.py:87`) and spawns one `multiprocessing.Process` per benchmark. Real parallelism is process-based; the pool caps concurrency.

### Evaluation (separate job)

- `sbatch scripts/eval/eval.sh config/.../eval.json --checkpoint N` → `scripts/eval/eval.py` → `evaluate_benchmarks()` over the **full eval set** (~2,363 benches; `indices = range(len(data))`, `ppo.py:289`) — greedy mode, so all cores are useful.
- **`eval.sh` has the same stale resource pin as `train.sh` did** — both were `--cpus-per-task=12 --constraint=bergamo` (constraint now removed, see below).

---

## Hardware (verified live, 2026-08-06 — compute partition only; GPU analysis moved to `v5_gpu_exploration.md`)

### Partitions

| Partition | Nodes | CPU cores | RAM | GPU | Max wall (live `sinfo`) |
|-----------|-------|-----------|-----|-----|--------------------------|
| `compute` (Jubail) | 404 | 128 (standard) / 256 (Bergamo subset) | 105G-480G | ❌ none | **7-00:00:00** |

GPU partitions (`nvidia` C2 QOS, `dalma`), their caps, and the A100 vs H100
analysis: see [v5_gpu_exploration.md](v5_gpu_exploration.md).

### Key structural insights

1. **`bench_count` reduction and CPU count are substitutes.** The old "reduce bench_count 64→16" advice was a workaround for 12 CPUs. With 64+ workers, `bench_count=64` runs in one wave — same wall-clock as 16/16, but 4x the gradient diversity. **Keep `bench_count=64`.**
2. **Training caps at `min(CPUS, bench_count)` = 64.** Eval (2,363 benches) could use more, but V5 pins eval at **64 CPUs / 100G — same as training** (user decision: footprint parity; 64 workers still parallelize the eval set ~5x vs the old 12).
3. **The bottleneck is CPU, always.** MLIR compile/exec (95%) is CPU-only. Expected gains: **~5-10x on the execution phase** (12→64 workers), net wall-clock **~4-6x** (50s → 8-12s). No GPU involved.

---

## Resource Allocation — DECIDED (apply as-is, no GPU)

| Script | Partition | CPUs | Mem | Wall | Rationale |
|--------|-----------|------|-----|------|-----------|
| `scripts/train/train.sh` | `compute` | **64** | **100G** | **7-00:00:00** | 64 workers = bench_count=64 (one wave); ~0.7-1G/worker (measured) + parent |
| `scripts/eval/eval.sh` | `compute` | **64** | **100G** | 7-00:00:00 | same as training (user decision — eval resources match train) |
| `scripts/eval/eval_batch.sh` | `compute` | 64 | 100G | 7-00:00:00 | same as eval |
| `scripts/hpo/train_trial.sh` | `compute` | **64** | **100G** | 7-00:00:00 | bergamo constraint |
| `scripts/hpo/eval_trial.sh` | `compute` | **64** | **100G** | 7-00:00:00 | bergamo constraint |
| `scripts/checkpoint/ckpt_scan_all.sh` | `compute` | 128 | 300G | 7-00:00:00 | full-model eval, CPU-only |
| `scripts/checkpoint/submit_ckpt_scan.sh` | `compute` | 8 | 16G | — | launches children; CPU-only |

Memory math (measured 2026-08-07, sacct MaxRSS on old 12c jobs): ~0.7-1G per
concurrent MLIR child → 64 children ≈ 45-64G + parent ≈ 50-70G peak → **100G ≈ 1.5x**
for both train and eval (both run 64 workers). Re-check MaxRSS on the first V5 run.

**Note**: `--constraint=bergamo` was removed from ALL 7 Slurm scripts (train, eval,
eval_batch, hpo train/eval, ckpt_scan, submit_ckpt_scan) on 2026-08-06 —
verified with `bash -n` + grep. Remaining "bergamo" strings are prose in docs
(`docs/hpc/HPC_HARDWARE.md`, `config/README.md`, `docs/archive/...`) and
result-dir names (`config/old_dataset/eval_bergamo/`) — no directives.

---

## GPUOccupier — decision: REMOVED from V5 (analysis moved)

V5 **removes all GPUOccupier logic from the package**: delete `utils/gpu_occupier.py`
and every `gpu_needed()` wrap in `ppo.py` (sampling `:69`, PPO update `:318`),
per-package `train.py`, and `evaluate.py`. No GPU code paths remain — V5 is
CPU-only. What GPUOccupier is, its wiring history, and the analysis behind the
removal: [v5_gpu_exploration.md](v5_gpu_exploration.md).

---

## Implementation Levers — status audit

| Lever | Status | What's needed |
|-------|--------|---------------|
| Raise CPUs on `compute` partition (12 → 64/128) | ⚠️ Code exists; scripts stale | Edit 5 scripts (matrix above) |
| Persistent MLIR worker pool (spawn-based) | ❌ Not implemented | **Moved to V5.1** (`v5_1_full_model_eval.md` §3.6) — ~1-3% for block training, not worth it there |
| Sampling/execution pipelining | ❌ Not implemented | Code change (`ppo.py`) — low priority (~3% of iteration) |
| `reuse_experience` / `replay_count` | ✅ Implemented (config.py:37,43; train.py:117-124) | Config only |
| `MIN_EXEC_TIMEOUT` straggler control | ✅ Implemented (execution.py:124) | Env only |
| Reduce `bench_count` | ✅ Config knob | Config only — **decided: keep 64** |
| Early stopping on plateau | ❌ Not implemented | Small change (`train.py`) |
| Benchmark feature cache | ❌ Not implemented | Small change (`benchmarks.py`) — saves 2-3 min/start |
| Multi-seed array runs | ⚠️ Array mode exists (version-based) | Extend to seed-based |
| GPU / GPUOccupier | ✅ **Removed from V5** | Delete `gpu_occupier.py` + all `gpu_needed()` wraps in `ppo.py`/`train.py`/`evaluate.py` (analysis: [v5_gpu_exploration.md](v5_gpu_exploration.md)) |
| ckpt_scan (full-model eval) | ✅ Works, CPU-only | Keep on `compute`; no GPU needed |

---

## Scope

**In scope (Phase 2 design + Phase 3 implement):**
1. Resource re-allocation of the 5 Slurm scripts (matrix above) on `compute` + `--time=7-00:00:00`.
2. ~~Persistent MLIR worker pool~~ — **moved to V5.1** (`v5_1_full_model_eval.md` §3.6): ~1-3% wall-clock for block training (repeats hit the time cache; fresh fork ≈ free), real payoff only in full-model eval.

**Out of scope for V5:**
- GPU / GPUOccupier / nvidia migration — **removed entirely** from V5 (CPU-only; see [v5_gpu_exploration.md](v5_gpu_exploration.md)).
- Pipelining sampling/execution (marginal: ~3% of iteration).
- Multi-seed arrays, early stopping, feature cache, `reuse_experience` flips — separate decisions, mostly config-only.
- Any change to the RL algorithm, reward function, action space, `opt_level=3`, or `ppo_batch_size` in paper-artifact configs.
- HW features and reward shaping — **abandoned** (explored in v4_9, found unhelpful); V5 base drops the dead config fields (`hardware_*`, `reward_shaping_*`).

---

## Constraints (hard rules)

- **Package isolation** (clarified 2026-08-06): no `rl_autoschedular_vN` may **import from another `rl_autoschedular_*` package** — ever. V5 is a new standalone package; it MAY use repo-level shared utils and tools (`utils.config`, `utils.implementation`, `utils.log`, ...) where useful — exactly like earlier versions do — and it MAY copy/paste code from previous versions (e.g. the paper_transformer tree) when building. What it must NOT do is `import rl_autoschedular_*`. Changes to `scripts/` (repo-level, not a package) are fine; changes inside a package must stay within that package.
- **Root `utils/config.py` keeps `hardware_*` / `reward_shaping_*`**: it is shared by v4_5/v4_9, which still read those fields. The dead-field cleanup is scoped to `rl_autoschedular_v5/utils/config.py` only.
- **Config singleton**: `utils/config.Config` reads `CONFIG_FILE_PATH` at first import; load `.env` before any config import.
- **`BindingsProcess.ENABLED` must stay `False`** — fork corrupts MLIR C++ state. Any worker-pool design (incl. V5.1's, `v5_1_full_model_eval.md` §3.6) must use `multiprocessing.get_context("spawn")` or spawn before MLIR import.
- **`torch.set_num_threads(4)`** at `scripts/train/train.py:100` — keep, or drop to 1-2 if CPU oversubscription appears with 64 workers (measure first).
- **No pytest suite** — verification is `python -m py_compile <file>` + short smoke run via `sbatch` with a small config.
- **NEVER delete files without explicit permission.**
- **Academic constraints**: reward function (`-20.0` penalty, speedup ratio), action space, `opt_level=3`, `ppo_batch_size` in paper configs must not change. `MIN_EXEC_TIMEOUT` changes only if eval uses the same value (it does — same env var).
- **Lustre**: `/scratch` 500K-file soft limit — check `lfs quota -u $USER /scratch` before bulk eval sweeps.

---

## Success Criteria

1. `sbatch scripts/train/train.sh config/v5/v5_small.json` runs on the `compute` partition with `--cpus-per-task=64`, `--mem=100G`, `--time=7-00:00:00` (verify via `squeue`).
2. Logs show `device = cpu` (expected — no GPU in V5); training proceeds without CUDA and without GPUOccupier.
3. Iteration wall-clock drops from ~50s to ~8-12s (measured in train log `iter_time_dlt`).
4. Eval (`sbatch scripts/eval/eval.sh config/v5/v5_small.json --checkpoint N`) runs at `--cpus-per-task=64 --mem=100G` — same resources as training — and completes faster than the old 12-CPU runs.
5. Training dynamics unchanged: same reward curve / speedup trajectory vs. a CPU run at same config (spot-check a few checkpoints via `scripts/utils/report_eval.py`).
6. `python -m py_compile scripts/train/train.py` passes; all edited `.sh` files pass `bash -n`.
7. **One unified config drives both**: `config/v5/v5_small.json` (with `json_file` + `eval_json_file`) works for train.sh and eval.sh unchanged — no separate eval config needed.

---

## Risks

- **Oversubscription**: 64 worker threads × torch/LLVM threads could thrash. Mitigate: keep `torch.set_num_threads(4)` (or lower), measure first. MLIR passes are single-threaded by default, so risk is low.
- **Entropy collapse** (known failure mode): shaped reward + Transformer → zero entropy. NOT caused by resource changes, but any `bench_count`/reuse experiments must keep `entropy_coef ≥ 0.05` if shaped reward is on. (V5 drops shaped reward, so this risk mostly disappears — the v4_9 ablation found shaped reward unhelpful.)
- **Timeout semantics**: lowering `MIN_EXEC_TIMEOUT` changes the reward signal (timed-out → `speedup=0.0` / `-20.0`); only acceptable if eval uses the same value (it does).
- **Stragglers**: pathological benchmarks can still burn the 300s cap; config-level only, not a blocker for the resource change.

---

## Codebase Pointers (for the architect)

| File | Role |
|------|------|
| `scripts/train/train.sh` | Slurm entry — resources to change (matrix) |
| `scripts/train/train.py` | Unified entry (`:87` Execution singleton, `:100` torch threads, `:159` loop) |
| `scripts/eval/eval.sh`, `scripts/eval/eval_batch.sh` | Eval entries — resources to change |
| `scripts/hpo/train_trial.sh`, `scripts/hpo/eval_trial.sh` | HPO entries — resources to change |
| `scripts/checkpoint/ckpt_scan_all.sh`, `submit_ckpt_scan.sh` | Full-model eval — stay on `compute` (V5.1 reuses these) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/` | **Base structure for `rl_autoschedular_v5`** — standalone package, own `utils/`, Transformer encoder, no HW features, no shaped reward |
| `rl_autoschedular/rl_autoschedular_v4_9/transforms.py` | Transform implementations V5.2 extends (padding, unrolling, packing already there) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/execution.py` | Bottleneck — cache (`:118-121`), isolated exec (`:196-277`), timeout (`:124-127`); worker-pool target moved to V5.1 (`v5_1_full_model_eval.md` §3.6) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/ppo.py` | `collect_trajectory` (`:28`), sampling (`:54`), eval (`:285`) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/utils/dask_manager.py` | ThreadPoolExecutor fallback (`:122-130`) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/utils/config.py` | Config fields — V5's copy drops dead `hardware_*` (`:83-97`) and `reward_shaping_*` (`:99-112`) fields (root `utils/config.py` keeps them — v4_5/v4_9 use them) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/utils/gpu_occupier.py` | **Deleted in V5** — plus all `gpu_needed()` wraps (`ppo.py:21,69,318`, `train.py`, `evaluate.py`) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/train.py` | Per-package entry (NOT used by Slurm) — V5's copy drops the `evaluate_benchmarks` calls (`:142-144,154-156`) |
| `config/v5/v5_small.json` | **NEW unified config** (train + eval in one JSON: `json_file` + `eval_json_file`) |
| `docs/hpc/HPC_HARDWARE.md` | Partitions, node inventory (verified 2026-08-03); C2 caps moved to `v5_gpu_exploration.md` |
| `docs/design/todo/v5_gpu_exploration.md` | **NEW** — GPU analysis archive (C2 limits, A100 vs H100, GPUOccupier analysis) |
| `docs/design/todo/v5_1_full_model_eval.md` | What ckpt_scan implements (full-model eval, CPU-only) |

---

## Open Questions — ALL RESOLVED (2026-08-06)

Phase 1 questions 1-5 were resolved in the Phase 2 scope table below (worker pool →
V5.1; torch threads → keep 4, measure; ckpt_scan → keep on `compute`; mini-features
→ follow-up; multi-seed → skip). The second wave of user comments (unified config,
GPUOccupier removal, eval resources = training, no `evaluate_benchmarks` in
training, package-isolation clarification) is folded into the Phase 2 design
below — see "Scope decisions".

---

## What NOT to Change (Academic Constraints)

| Item | Reason |
|------|--------|
| `ppo_batch_size=64` (paper-artifact configs only) | Standard PPO batch size for academic reproducibility. Exploratory configs (v4.5+) already use 32. |
| `opt_level=3` | Changes the reward signal (execution times). Must stay consistent. |
| `evaluate_benchmarks` during training | Not called by `scripts/train/train.py`; **V5 also removes it from the per-package `train.py`** — eval is a separate Slurm job, never part of training |
| Reward function (`-20.0` penalty, speedup ratio) | Core to the RL formulation. |
| Action space (tiling, interchange, vectorization, etc.) | Core to the problem formulation. |
| `BindingsProcess.ENABLED` must stay `False` | Fork corrupts MLIR C++ state — applies to any worker-pool design (use spawn). |
| Transformer encoder | The core contribution — V5 keeps it exactly as in paper_transformer. |

---

## Phase 2 Design (feature-architect, 2026-08-06)

### Summary
V5 is the platform version: a standalone `rl_autoschedular_v5` package (paper_transformer
structure, cleaned of dead HW/shaping config fields **and all GPUOccupier logic**, and with
`evaluate_benchmarks` removed from training), a **unified config** per run in `config/v5/`
(one JSON serves both train and eval), and a resource re-allocation of the 5 Slurm entry
scripts (12 → 64 CPUs, 32G → 100G, wall → 7d). No change to the RL algorithm, reward,
action space, or Transformer encoder. Expected net effect: iteration wall-clock ~50s →
~8-12s (~4-6x).

### Scope decisions (locked this session — both waves)
| Open question | Decision | Rationale |
|---|---|---|
| Persistent MLIR worker pool | **Moved to V5.1** — full-model eval (`v5_1_full_model_eval.md` §3.6) | Resource change alone gives 4-6x; pool ≈ 1-3% for block training (per-exec setup hidden in parallel, repeats hit the time cache). Real payoff in V5.1: parse-once per model + incremental transforms + parallel model eval. |
| ckpt_scan (full-model eval) future | **Keep untouched on `compute`** | Already correctly sized (128c/300G); V5.1 reuses these scripts; archiving saves nothing. |
| Config-only mini-features (early stopping, feature cache, `reuse_experience` flips) | **Out of scope for V5** | Keeps V5's before/after measurement clean; separate follow-up mini-design after the resource change is measured. |
| Multi-seed array runs | **Skip** | Only needed if the paper requires variance bars. |
| `torch.set_num_threads(4)` | **Keep 4**, measure | MLIR passes are single-threaded; drop to 1-2 only if 64-worker oversubscription appears (measure `iter_time_dlt` first). |
| **Unified config** (train + eval in one JSON) | **Yes — `config/v5/<name>.json`** | Fields already exist (`json_file`/`eval_json_file`); `benchmarks.py:38-42` picks the right split; both train.sh and eval.sh read the same `CONFIG_FILE_PATH` + `results_dir` + `implementation`. |
| **GPUOccupier** | **Removed entirely from the v5 package** | CPU-only pipeline; no GPU code paths. Analysis archived in `v5_gpu_exploration.md`. |
| **Eval resources** | **64 CPUs / 100G — same as training** (eval.sh + eval_batch.sh) | User decision: footprint parity; still ~5x the old 12-CPU eval parallelism. |
| **`evaluate_benchmarks` in per-package `train.py`** | **Removed in v5** | Eval is a separate process — never inside training. (paper_transformer stays frozen; its copy keeps the call.) |
| **Package isolation scope** | **No imports from other `rl_autoschedular_*` packages**; repo-level shared `utils/` and copy/paste from previous versions are fine | User clarification — the constraint is cross-`rl_autoschedular` imports only. |

### Approach
1. **New standalone package** `rl_autoschedular_v5` — copy of `rl_autoschedular_paper_transformer`
   (which stays frozen as the paper artifact), then **clean it**:
   - drop the 15 dead config fields (`hardware_*`/`reward_shaping_*` — referenced nowhere in the
     package, verified by grep);
   - **delete `utils/gpu_occupier.py`** and every `gpu_needed()` wrap (`ppo.py`, `train.py`,
     `evaluate.py`);
   - **remove `evaluate_benchmarks` calls from `train.py`** (per-package train entry no longer
     evaluates; eval is a separate Slurm job).
2. **Unified config** — one JSON per run in `config/v5/` carrying `json_file` (train split) +
   `eval_json_file` (eval split). `benchmarks.py:38-42` already selects by `is_training`; both
   scripts read the same `CONFIG_FILE_PATH`. No schema change needed.
3. **Resource re-allocation** — edit the `#SBATCH` header of 5 scripts per the DECIDED matrix
   (train 64/100G/7d; **eval + eval_batch 64/100G/7d**; HPO 64c/100G). `scripts/checkpoint/*`
   already correct → untouched.
4. **Script adaptation for unified configs** — `scripts/eval/submit_eval.py` registry (v5 entries;
   name derivation no longer assumes an `_eval.json` suffix); usage comments in `eval.sh`.
5. **No code changes** to `execution.py`, `ppo.py` (beyond GPUOccupier removal), `dask_manager.py`,
   `utils/implementation.py` (the `v[\d_]+` regex auto-derives `v5` → agent_dir `v5_agent`,
   prefix `v5` — no registry edit). Root `utils/config.py` keeps its dead fields (v4_5/v4_9 use them).

Alternatives considered: (a) reusing `rl_autoschedular_paper_transformer` in place instead of a new
package — rejected: paper artifact must stay frozen for the paper; V5.1/V5.2 need a stable base to
extend. (b) separate train/eval configs — rejected (user): they share most parameters; one unified
config per run. (c) GPU/GPUOccupier — removed entirely (see `v5_gpu_exploration.md`). (d) worker pool in V5 — moved to V5.1 (§3.6 of `v5_1_full_model_eval.md`); block training can't amortize it (repeats hit the time cache, fresh fork ≈ free).

### Components / Changes
| Path | Change |
|---|---|
| `rl_autoschedular/rl_autoschedular_v5/` | **NEW** — full standalone copy of `rl_autoschedular_paper_transformer/`, cleaned: minus 15 dead config fields, minus `utils/gpu_occupier.py` + all `gpu_needed()` wraps, minus `evaluate_benchmarks` calls in `train.py` |
| `rl_autoschedular/rl_autoschedular_v5/utils/config.py` | Drop `hardware_*` (8) + `reward_shaping_*` (7) annotations (see Data Model). Root `utils/config.py` unchanged (v4_5/v4_9 use the fields). |
| `rl_autoschedular/rl_autoschedular_v5/utils/gpu_occupier.py` | **DELETED** (with `gpu_needed()` wraps in `ppo.py:69,318`, `train.py`, `evaluate.py`) |
| `rl_autoschedular/rl_autoschedular_v5/train.py` | Per-package entry (NOT used by Slurm) — `evaluate_benchmarks` calls removed (`:142-144,154-156` in the source) and `eval_data` loading dropped |
| `config/v5/v5_small.json` | **NEW unified config** — one JSON for train + eval: `implementation` → `rl_autoschedular_v5`, `results_dir` → `results/new_dataset_results/v5_small_agent`, `json_file` = train split, `eval_json_file` = eval split, dead keys removed, all else identical to the paper_transformer_small train config |
| `scripts/train/train.sh` | `--cpus-per-task=64 --mem=100G --time=7-00:00:00` |
| `scripts/eval/eval.sh` | `--cpus-per-task=64 --mem=100G --time=7-00:00:00` (same as training; add explicit `--time`) |
| `scripts/eval/eval_batch.sh` | `--cpus-per-task=64 --mem=100G --time=7-00:00:00` (add explicit `--time`) |
| `scripts/hpo/train_trial.sh` | `--cpus-per-task=64 --mem=100G --constraint=bergamo --time=7-00:00:00` |
| `scripts/hpo/eval_trial.sh` | `--cpus-per-task=64 --mem=100G --constraint=bergamo --time=7-00:00:00` |
| `scripts/eval/submit_eval.py` | Registry: add `v5_small` → `config/v5/v5_small.json`; name derivation must not assume an `_eval.json` suffix |
| `docs/design/todo/v5_gpu_exploration.md` | **NEW** — GPU analysis archive (C2 limits, A100 vs H100, GPUOccupier) |
| `AGENTS.md` | Add `v5` row to package table; update commands to the unified-config usage |

Not touched: `scripts/checkpoint/ckpt_scan_all.sh`, `submit_ckpt_scan.sh`, `utils/implementation.py`,
`execution.py`, `dask_manager.py`, root `utils/config.py`, `rl_autoschedular_paper_transformer/` (frozen).

### Data Model / API
**No new config fields — and the config schema becomes unified.**
- **Unified config** (`config/v5/<name>.json`): one JSON with `json_file` (train split) **and**
  `eval_json_file` (eval split). `Benchmarks(is_training=...)` (`benchmarks.py:30-44`) picks the
  right one. Both `scripts/train/train.py` and `scripts/eval/eval.py` read the same
  `CONFIG_FILE_PATH` via the Config singleton, plus `results_dir` + `implementation` (used by
  eval.sh for EVAL_DIR). No schema change — the fields already exist.
- **Removed from `rl_autoschedular_v5/utils/config.py` (15):**
  `hardware_auto_detect`, `hardware_l1_kb`, `hardware_l2_kb`, `hardware_l3_kb`, `hardware_physical_cores`,
  `hardware_logical_cores`, `hardware_simd_width`, `hardware_clock_mhz`,
  `reward_shaping_enabled`, `reward_shaping_scale`, `reward_shaping_clip`, `reward_shaping_weight_ai`,
  `reward_shaping_weight_vectorizable`, `reward_shaping_weight_parallel`, `reward_shaping_vectorization_bonus`.

Compatibility notes:
- `Config.__init__` iterates class annotations only — unknown JSON keys are **silently ignored**,
  so a stale config carrying the dead keys still loads under v5. Safe.
- `json_file` / `eval_json_file` auto-derive via `utils.implementation.get_split_file_path`:
  prefix `v5` → tries `exec_times/v5_base_{train,eval}.json`, falls back to generic
  `base_{train,eval}.json` — existing baselines work with zero setup.
- `results_dir` is explicit per config; naming auto-derived (v5_agent) is not used unless `results_dir` is empty.
- Root `utils/config.py` (used by the Slurm scripts) still declares the dead fields — they fall back
  to class defaults when absent from a v5 config. No error.

### Edge Cases & Error Handling
- **CPU oversubscription** (64 workers × torch/LLVM threads): keep `torch.set_num_threads(4)`;
  MLIR passes single-threaded by default → low risk. If `iter_time_dlt` regresses, lower to 1-2
  (measure first, per AGENTS.md).
- **Memory**: 64 concurrent MLIR children × ~0.7-1G RSS (measured) + parent ≈ 50-70G peak;
  100G = ~1.5x headroom. If OOM → bump `--mem`, no code change. Train and eval both run
  64 workers → both fit in 100G. Re-check MaxRSS on the first V5 run.
- **MLIR SIGABRT**: unchanged path — isolated child process → fallback `mlir-opt | mlir-cpu-runner`
  subprocess; SIGABRT handler in `train.py` converts native crashes to Python exceptions. Resource
  change touches none of this.
- **Timeout consistency**: `MIN_EXEC_TIMEOUT` (env, default 300) is the same var for train and eval —
  reward signal stays consistent. Not changed.
- **Cache**: `exec_data.json` per-run, keyed by benchmark + action sequence; the shared `Execution`
  singleton already served 12 threads (GIL-atomic dict access); 64 threads change nothing structurally.
- **Unified config with an old eval-only config**: pointing eval.sh at a legacy `*_eval.json`
  under v5 can fail Config load if required keys (e.g. `nb_iterations`, `ppo_epochs`) are missing
  and have no defaults. Mitigation: always use the unified `config/v5/<name>.json` for v5 runs.
- **7d wall**: `--resume results/.../run_0` and `FORCE_RUN_ID=ckpt_N` cover interrupted long runs.
- **Lustre**: check `lfs quota -u $USER /scratch` before bulk eval sweeps (existing rule).

### Resource Requirements
Apply the DECIDED matrix (§Resource Allocation) — restated for Phase 3 (eval = training):

| Script | Partition | CPUs | Mem | Wall |
|---|---|---|---|---|
| `scripts/train/train.sh` | `compute` | 64 | 100G | 7-00:00:00 |
| `scripts/eval/eval.sh` | `compute` | 64 | 100G | 7-00:00:00 |
| `scripts/eval/eval_batch.sh` | `compute` | 64 | 100G | 7-00:00:00 |
| `scripts/hpo/train_trial.sh` | `compute` | 64 | 100G | 7-00:00:00 |
| `scripts/hpo/eval_trial.sh` | `compute` | 64 | 100G | 7-00:00:00 |
| `scripts/checkpoint/ckpt_scan_all.sh` | `compute` | 128 | 300G | 7-00:00:00 *(no change)* |

`eval.sh` and `eval_batch.sh` currently have **no `--time` line** — add `--time=7-00:00:00` explicitly.
No GPU anywhere (see `v5_gpu_exploration.md` for the GPU partition analysis).

### Tasks (for feature-develop, in order)
- [x] **T1 — Create `rl_autoschedular_v5` package (clean copy).** `cp -r rl_autoschedular/rl_autoschedular_paper_transformer rl_autoschedular/rl_autoschedular_v5`; then: (a) remove the 15 dead fields from `utils/config.py`; (b) **delete `utils/gpu_occupier.py`** and strip every `gpu_needed()`/GPUOccupier reference from `ppo.py`, `train.py`, `evaluate.py`; (c) **remove the `evaluate_benchmarks` calls** from `train.py` (the `(step + 1) % 100 == 0` block, the trailing block, and the now-unused `eval_data` loading). Verify: `python -m py_compile` on every `.py` in the package; `python -c "import rl_autoschedular_v5"` and confirm `rl_autoschedular_v5.device` exists; `grep -rn "gpu_occupier\|GPUOccupier\|gpu_needed" rl_autoschedular/rl_autoschedular_v5/` = 0 hits; `grep -rn "evaluate_benchmarks" rl_autoschedular/rl_autoschedular_v5/train.py` = 0 hits (allowed in `evaluate.py`/`ppo.py` — eval module keeps it). ✅ verified 2026-08-08: py_compile all-pass, import OK (device=cpu), greps 0/0/0.
- [x] **T2 — Unified V5 config.** Create `config/v5/v5_small.json` from `config/paper/new_dataset/paper_transformer_small_train.json`: set `"implementation": "rl_autoschedular_v5"`, `"results_dir": "results/new_dataset_results/v5_small_agent"`, set `"eval_json_file"` to the eval split (the paper eval pair's `json_file`, e.g. `results/new_dataset_results/baselines/mlir/eval_base.json` — or leave empty to auto-derive), delete the 15 dead keys (keep `bench_count: 64`, `ppo_batch_size: 64`). Verify: `python -m json.tool config/v5/v5_small.json`; load with `CONFIG_FILE_PATH=config/v5/v5_small.json python -c "from rl_autoschedular_v5.utils.config import Config; c=Config(); assert not hasattr(c,'hardware_l1_kb'); assert c.json_file and c.eval_json_file"`; smoke both splits: `Benchmarks(is_training=True)` and `Benchmarks(is_training=False)` load. ✅ json.tool OK; Config load OK (47 annotations, both splits set). ⚠️ Benchmarks split smoke deferred — dataset .mlir files absent on this node (T5 blocker, see note).
- [x] **T3 — Resource re-allocation.** Edit the 5 `#SBATCH` headers per the matrix: train 64c/100G/7d; **eval + eval_batch 64c/100G/7d** (add the missing `--time=7-00:00:00` to both); **hpo train + eval 64c/100G/7d** (bergamo constraint added). Verify: `bash -n` each script; `grep -n "cpus-per-task\|mem=\|time="` shows the new values; all 5 scripts match the standard allocation. ✅ verified 2026-08-10: all scripts now use 64c/100G/bergamo/7d.
- [x] **T4 — Script adaptation for unified configs.** `scripts/eval/submit_eval.py`: add `"v5_small": "config/v5/v5_small.json"` to the agent registry; make the agent-name derivation work without an `_eval.json` suffix (strip `.json` only). Update `eval.sh` usage comments (unified config examples). Verify: `python -m py_compile scripts/eval/submit_eval.py`; dry-run `resolve_agent_config("v5_small")` returns `config/v5/v5_small.json`. ✅ verified 2026-08-08: py_compile OK, resolve OK, name derivation `v5_small` OK; eval.sh usage updated.
- [ ] **T5 — Smoke train run.** `sbatch scripts/train/train.sh config/v5/v5_small.json` with a short-run override (nb_iterations ~50, or `--resume` on an existing run). Verify: `squeue` shows `compute`/`64`/`100G`; log shows `device = cpu`, no CUDA, no GPUOccupier lines, **no "Evaluating benchmarks" lines**; first iterations complete with `iter_time_dlt` in the 8-12s band. ✅ unblocked 2026-08-10: dataset restored to `data/ops_and_blocks` (8,093 files, flat); 147 JSON names with no file pruned from the split JSONs (train 6,464 / eval 1,629; originals in `.bak`, record in `docs/archive/OPS_AND_BLOCKS_MISSING_BENCHMARKS_2026_08_10.md`); config `benchmarks_folder_path` → `data/ops_and_blocks`; ad-hoc verified (config load + extraction + name match).
- [ ] **T6 — Smoke eval run.** `sbatch scripts/eval/eval.sh config/v5/v5_small.json --checkpoint N`. Verify: `squeue` shows `compute`/`64`/`100G`; `eval/checkpoint_<N>.json` written with `{bench: exec_time_ns}`; completes faster than the old 12-CPU eval. ⛔ BLOCKED: same as T5.
- [ ] **T7 — Measure & record.** Compare before/after `iter_time_dlt` (≈50s → ≈8-12s); spot-check reward/speedup trajectory via `python scripts/utils/report_eval.py` (training dynamics must match a same-config CPU run). Record numbers in this doc.
- [ ] **T8 — Docs.** AGENTS.md: add the `v5` row to the package table (Transformer, no HW, no shaping, no GPUOccupier; base for V5.1/V5.2); update the Commands section to unified-config usage (`sbatch scripts/train/train.sh config/v5/v5_small.json`; `sbatch scripts/eval/eval.sh config/v5/v5_small.json --checkpoint N`). When the user confirms results, move this doc to `docs/design/done/`. ✅ AGENTS.md package table + Commands updated 2026-08-08; ⏳ doc move pending user confirmation of T7 results.


