# V5: Accelerated Training Platform — Design Input

**Status**: Phase 1 complete (brainstorm + verification) — ready for `feature-architect` (Phase 2)
**Last verified**: 2026-08-06 against live code + live Slurm state
**Version**: **V5** of the new MLIR-RL generation (V5 → V5.1 → V5.2)
**Target package**: `rl_autoschedular_v5` (new standalone package)
**Base package**: `rl_autoschedular_paper_transformer` structure — Transformer encoder (the core contribution), **NO hardware features, NO reward shaping** (both explored in v4_9 and **abandoned** — not helpful). Verified: paper_transformer = v4_9 minus HW observation minus shaped reward, and is fully standalone (own `utils/config.py`), unlike v4_9 which imports shared root `utils.config`.
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
>    reward shaping abandoned — explored in v4.9, found unhelpful).
> 2. **No GPU in V5.** Training/eval stay on the Jubail `compute` partition. GPU
>    accelerates only the ~3% sampling/PPO fraction (~1.05-1.1x) while C2 `nvidia`
>    is a scarce shared resource (team cap ~5 GPUs); CPU parallelism 12→64/128
>    accelerates the 95% MLIR-exec fraction (5-10x) on 404 available compute
>    nodes. Revisit GPU only if a future version scales the model.
> 3. Resource allocation: see the DECIDED matrix in §Resource Allocation.
> 4. Version order V5 → V5.1 → V5.2 confirmed (no swap). Consequence accepted:
>    V5.1's full-model numbers describe the 6-action agent; re-run after V5.2 if
>    the paper needs them for the expanded agent.
> 5. HPO is a **parallel track**, not a version (V5.3 dropped).
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

**The one win that matters**: 12 → 64/128 workers multiplies MLIR execution
parallelism ~5-10x (the dominant cost). GPU was evaluated and **rejected** — it
accelerates only the ~3% sampling/PPO fraction (~1.05-1.1x) while C2 `nvidia`
is a scarce shared resource (team cap ~5 GPUs). See "Why V5 runs WITHOUT GPU".

Goal: cut wall-clock per iteration by **~4-6x** (50s → 8-12s) with zero change to
the RL algorithm, reward function, or action space.

---

## Idea (rough shape)

- **Raise CPU parallelism on the Jubail `compute` partition**: `--cpus-per-task=64`
  for training (matches `bench_count=64` — one wave per iteration) and
  `--cpus-per-task=128` for eval (matches the ~2,363-benchmark eval set).
- **Bump `--mem`** to 128G (train) / 256G (eval) — 64 concurrent MLIR children
  × ~2G RSS ≈ 128G + parent/features.
- **Bump `--time`** to the compute-partition max **7-00:00:00**.
- **No GPU, no nvidia migration, no GPUOccupier wiring** — rejected for V5
  (rationale in the hardware section).
- **Optional follow-ups** (already implemented in config, just flip):
  `reuse_experience`/`replay_count`; `MIN_EXEC_TIMEOUT` straggler control.
- **Keep `bench_count=64`** — with 64 workers it executes in one wave, so
  gradient diversity is preserved *at the same wall-clock*.

---

## How Training Is Done (verified current state)

### Entry point (`scripts/train/train.sh` → `scripts/train/train.py`)

1. `train.sh` submits a Slurm job; `scripts/train/train.py` is the unified entry (what Slurm runs — the per-package `train.py` files are NOT used by the Slurm pipeline, though they exist and do call `evaluate_benchmarks` every 100 steps).
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

## Hardware (verified live, 2026-08-06)

### Partitions

| Partition | Nodes | CPU cores | RAM | GPU | Max wall (live `sinfo`) |
|-----------|-------|-----------|-----|-----|--------------------------|
| `compute` (Jubail) | 404 | 128 (standard) / 256 (Bergamo subset) | 105G-480G | ❌ none | **7-00:00:00** |
| `nvidia` (C2 QOS) | 48 | **128 on A100 nodes**, **64 on H100/H200 nodes** | 365-491G | A100/H100/H200 | **4-00:00:00** |
| `dalma` | — | 40 | 105G-1T | V100 (legacy) | — |

### C2 QOS live limits (`sacctmgr show qos c2`)

- `GrpTRES: cpu=384, gres+` → **~384 concurrent CPUs, ~5 concurrent GPUs** across the whole team.
- **No explicit MaxWall at QOS level** (blank); the `nvidia` partition TIMELIMIT is **4-00:00:00** — that's the max wall for any job there.

### A100 vs H100/H200 — analysis (recorded; GPU now out of scope for V5)

| | A100 nodes (cn001-268) | H100/H200 nodes (cn270-276) |
|---|---|---|
| CPU cores | **128** | **64** |
| GPU | 1-4× A100 80GB | 2-8× H100/H200 (faster) |

H100's raw GPU advantage (FP8, more FLOPs) is **irrelevant** here: the model is
tiny (`d_model` 64/256) and uses the GPU ~3% of the time. But H100/H200 nodes
have **half the CPU cores**, and CPU cores drive MLIR execution (the 95%
bottleneck). **Conclusion (if GPU is ever revisited): pin `--gres=gpu:a100:1`,
never bare `--gres=gpu:1`** — a bare request could land on a 64-core H100 node
and *slow training down*.

### Why V5 runs WITHOUT GPU (decision, 2026-08-06)

- GPU accelerates only the **~3% sampling/PPO fraction** → **~1.05-1.1x** at best.
- CPU parallelism 12 → 64/128 accelerates the **~95% MLIR-exec fraction** →
  **5-10x** — and it's free on the `compute` partition's 128-core nodes
  (404 nodes available).
- C2 `nvidia` is a **scarce shared resource** (team cap ~5 GPUs / 384 CPUs).
  Spending quota for ~1.05x while compute nodes sit idle is a bad trade.
- V5.1 (full-model eval) is CPU-bound anyway; V5.2 (more actions) makes
  trajectories longer — both need CPU headroom, not GPU.
- **Revisit GPU only if** a future version scales the model (full-model PPO,
  GNN encoder, larger d_model) — at which point the A100-pin rule above applies.

### Key structural insights

1. **`bench_count` reduction and CPU count are substitutes.** The old "reduce bench_count 64→16" advice was a workaround for 12 CPUs. With 64+ workers, `bench_count=64` runs in one wave — same wall-clock as 16/16, but 4x the gradient diversity. **Keep `bench_count=64`.**
2. **CPU ceiling is bounded by workload, not node size.** Training caps at `min(CPUS, bench_count)` = 64; eval (2,363 benches) can use all 128. Giving training 128 CPUs wastes half; eval wants 128.
3. **The bottleneck is CPU, always.** MLIR compile/exec (95%) is CPU-only. Expected gains: **~5-10x on the execution phase** (12→64/128 workers), net wall-clock **~4-6x** (50s → 8-12s). No GPU involved.

---

## Resource Allocation — DECIDED (apply as-is, no GPU)

| Script | Partition | CPUs | Mem | Wall | Rationale |
|--------|-----------|------|-----|------|-----------|
| `scripts/train/train.sh` | `compute` | **64** | **128G** | **7-00:00:00** | 64 workers = bench_count=64 (one wave); ~2G/worker + parent |
| `scripts/eval/eval.sh` | `compute` | **128** | **256G** | 7-00:00:00 | eval set = 2,363 benches — all cores useful |
| `scripts/eval/eval_batch.sh` | `compute` | 128 | 256G | 7-00:00:00 | same as eval |
| `scripts/hpo/train_trial.sh` | `compute` | 32 | 64G | 7-00:00:00 | HPO runs many trials; keep each lean |
| `scripts/hpo/eval_trial.sh` | `compute` | 64 | 128G | 7-00:00:00 | mid-size eval |
| `scripts/checkpoint/ckpt_scan_all.sh` | `compute` | 128 | 300G | 7-00:00:00 | full-model eval, CPU-only |
| `scripts/checkpoint/submit_ckpt_scan.sh` | `compute` | 8 | 16G | — | launches children; CPU-only |

**Budget check** (compute partition: 404 nodes, 128 cores each — no shared caps):
- 1 train (64c) + 1 eval (128c) + 2 HPO train (32c) + 2 HPO eval (64c) =
  352 CPUs ≈ **2.75 nodes** — trivially fits; no GPU quota involved.
- Compute partition TIMELIMIT is **7-00:00:00** (vs 4-00:00:00 on nvidia) — a
  bonus: longer max wall for the same work.

Memory math: 64 concurrent MLIR children × ~2G RSS ≈ 128G + parent/features → 128G comfortable for train; 256G for the 128-worker eval.

**Note**: `--constraint=bergamo` was removed from ALL 7 Slurm scripts (train, eval,
eval_batch, hpo train/eval, ckpt_scan, submit_ckpt_scan) on 2026-08-06 —
verified with `bash -n` + grep. Remaining "bergamo" strings are prose in docs
(`docs/hpc/HPC_HARDWARE.md`, `config/README.md`, `docs/archive/...`) and
result-dir names (`config/old_dataset/eval_bergamo/`) — no directives.

---

## GPUOccupier — analysis & decision: REJECTED (no GPU in V5)

### What it is
A singleton that spawns a background process running a dummy `torch.matmul(64×64)`
loop on the GPU **whenever the model isn't using it** (`gpu_needed()` events gate
it). Purpose: keep the GPU "hot" (avoid downclocking between sampling bursts) and
guard against idle-GPU policies. See `rl_autoschedular_paper_transformer/utils/gpu_occupier.py`.

### Verified wiring
- Per-package `train.py` (not used by Slurm): ✅ fully wired — `go.start(device)` (line 55), `gpu_needed()` wraps eval + PPO (lines 89, 128, 132), `go.stop()` (line 165).
- `ppo.py`: sampling wrapped (`:69`), PPO update wrapped (`:318`).
- **Unified `scripts/train/train.py` (what Slurm actually runs): ❌ NEVER calls `start()`** → today it's a silent no-op (events set/cleared, nobody reads them).

### Decision — REJECTED for V5
**Do NOT wire GPUOccupier; do not use the GPU at all in V5** (see "Why V5 runs
WITHOUT GPU"). The occupier is moot on a CPU-only pipeline: `device` is CPU,
`start()` would raise `ValueError` (it requires CUDA), and the `gpu_needed()`
wraps in `ppo.py` become harmless no-ops. Leave the code as-is (it does nothing
on CPU); do not add GPU paths. If a future version scales the model and re-enables
GPU, wire `GPUOccupier().start(device)` into `scripts/train/train.py` then.

---

## Implementation Levers — status audit

| Lever | Status | What's needed |
|-------|--------|---------------|
| Raise CPUs on `compute` partition (12 → 64/128) | ⚠️ Code exists; scripts stale | Edit 5 scripts (matrix above) |
| Persistent MLIR worker pool (spawn-based) | ❌ Not implemented | Code change (`execution.py`) |
| Sampling/execution pipelining | ❌ Not implemented | Code change (`ppo.py`) — low priority (~3% of iteration) |
| `reuse_experience` / `replay_count` | ✅ Implemented (config.py:37,43; train.py:117-124) | Config only |
| `MIN_EXEC_TIMEOUT` straggler control | ✅ Implemented (execution.py:124) | Env only |
| Reduce `bench_count` | ✅ Config knob | Config only — **decided: keep 64** |
| Early stopping on plateau | ❌ Not implemented | Small change (`train.py`) |
| Benchmark feature cache | ❌ Not implemented | Small change (`benchmarks.py`) — saves 2-3 min/start |
| Multi-seed array runs | ⚠️ Array mode exists (version-based) | Extend to seed-based |
| GPU / GPUOccupier | ❌ **Rejected for V5** | None — CPU-only pipeline (see above) |
| ckpt_scan (full-model eval) | ✅ Works, CPU-only | Keep on `compute`; no GPU needed |

---

## Scope

**In scope (Phase 2 design + Phase 3 implement):**
1. Resource re-allocation of the 5 Slurm scripts (matrix above) on `compute` + `--time=7-00:00:00`.
2. *(Optional, architect's call)* persistent MLIR worker pool — the biggest remaining code-side win; must be **spawn-based, never fork** (see Constraints).

**Out of scope for V5:**
- GPU / GPUOccupier / nvidia migration — **rejected** (see decision above).
- Pipelining sampling/execution (marginal: ~3% of iteration).
- Multi-seed arrays, early stopping, feature cache, `reuse_experience` flips — separate decisions, mostly config-only.
- Any change to the RL algorithm, reward function, action space, `opt_level=3`, or `ppo_batch_size` in paper-artifact configs.
- HW features and reward shaping — **abandoned** (explored in v4_9, found unhelpful); V5 base drops the dead config fields (`hardware_*`, `reward_shaping_*`).

---

## Constraints (hard rules)

- **Package isolation**: every `rl_autoschedular_vN` is fully standalone — no cross-package imports. V5 is a new standalone package structured like `paper_transformer` (own `utils/`), NOT like v4_9 (which imports shared root `utils.config`). Changes to `scripts/train/train.py` (repo-level, not a package) are fine; changes inside packages must stay within one package.
- **Config singleton**: `utils/config.Config` reads `CONFIG_FILE_PATH` at first import; load `.env` before any config import.
- **`BindingsProcess.ENABLED` must stay `False`** — fork corrupts MLIR C++ state. Any worker-pool design must use `multiprocessing.get_context("spawn")` or spawn before MLIR import.
- **`torch.set_num_threads(4)`** at `scripts/train/train.py:100` — keep, or drop to 1-2 if CPU oversubscription appears with 64 workers (measure first).
- **No pytest suite** — verification is `python -m py_compile <file>` + short smoke run via `sbatch` with a small config.
- **NEVER delete files without explicit permission.**
- **Academic constraints**: reward function (`-20.0` penalty, speedup ratio), action space, `opt_level=3`, `ppo_batch_size` in paper configs must not change. `MIN_EXEC_TIMEOUT` changes only if eval uses the same value (it does — same env var).
- **Lustre**: `/scratch` 500K-file soft limit — check `lfs quota -u $USER /scratch` before bulk eval sweeps.

---

## Success Criteria

1. `sbatch scripts/train/train.sh <paper_transformer_small config>` runs on the `compute` partition with `--cpus-per-task=64`, `--mem=128G`, `--time=7-00:00:00` (verify via `squeue`).
2. Logs show `device = cpu` (expected — no GPU in V5); training proceeds without CUDA.
3. Iteration wall-clock drops from ~50s to ~8-12s (measured in train log `iter_time_dlt`).
4. Eval (`sbatch scripts/eval/eval.sh ... --checkpoint N`) runs at `--cpus-per-task=128` and completes faster than before.
5. Training dynamics unchanged: same reward curve / speedup trajectory vs. a CPU run at same config (spot-check a few checkpoints via `scripts/utils/report_eval.py`).
6. `python -m py_compile scripts/train/train.py` passes; all edited `.sh` files pass `bash -n`.

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
| `rl_autoschedular/rl_autoschedular_paper_transformer/execution.py` | Bottleneck — cache (`:118-121`), isolated exec (`:196-277`), timeout (`:124-127`); persistent-pool target if in scope |
| `rl_autoschedular/rl_autoschedular_paper_transformer/ppo.py` | `collect_trajectory` (`:28`), sampling (`:54`), eval (`:285`) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/utils/dask_manager.py` | ThreadPoolExecutor fallback (`:122-130`) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/utils/config.py` | Config fields — V5 drops dead `hardware_*` (`:83-97`) and `reward_shaping_*` (`:99-112`) fields |
| `docs/hpc/HPC_HARDWARE.md` | Partitions, C2 caps, node inventory (verified 2026-08-03) |
| `docs/design/todo/v5_1_full_model_eval.md` | What ckpt_scan implements (full-model eval, CPU-only) |

---

## Open Questions (for the architect to resolve with the user)

1. **Persistent MLIR worker pool in scope for V5?** It's the biggest remaining code-side win (2-3x on execution phase, compounds with CPU raise) but adds real complexity (spawn-only, crash isolation, cache interaction). Recommend: **defer to V5.x later** — the resource change alone gives 4-6x. Architect's call.
2. **`torch.set_num_threads(4)`** — keep or lower to 1-2 with 64 workers? Measure after the resource change; default: keep.
3. **ckpt_scan future**: is the full-model eval still needed for the paper? If yes, keep scripts on `compute` (V5.1 reuses them). If cancelled, archive them. (User leaning: "training + eval jobs are sufficient" — confirm.)
4. **Early stopping / feature cache / reuse_experience flips** — worth a follow-up mini-design after the resource change is measured, or bundle into V5? (All are config-only or tiny.)
5. **Multi-seed arrays** — extend `train.sh` array mode to seed-based for variance analysis? Only if the paper needs variance bars.

---

## What NOT to Change (Academic Constraints)

| Item | Reason |
|------|--------|
| `ppo_batch_size=64` (paper-artifact configs only) | Standard PPO batch size for academic reproducibility. Exploratory configs (v4.5+) already use 32. |
| `opt_level=3` | Changes the reward signal (execution times). Must stay consistent. |
| `evaluate_benchmarks` during training | Already not called by `scripts/train/train.py`. Eval is separate. |
| Reward function (`-20.0` penalty, speedup ratio) | Core to the RL formulation. |
| Action space (tiling, interchange, vectorization, etc.) | Core to the problem formulation. |
| `BindingsProcess.ENABLED` must stay `False` | Fork corrupts MLIR C++ state — applies to any worker-pool design (use spawn). |
| Transformer encoder | The core contribution — V5 keeps it exactly as in paper_transformer. |

---

## Recommended Execution Order (for Phase 3)

1. **Create `rl_autoschedular_v5` package** — standalone copy of `paper_transformer` structure; drop dead `hardware_*`/`reward_shaping_*` config fields; update `utils/implementation.py` mapping. Verify: `python -m py_compile` all files.
2. **Resource re-allocation** — edit 5 scripts per the matrix (config-only, ~10 min, immediate 4-6x). Verify: `bash -n` each; submit a short smoke run.
3. **Measure** — compare `iter_time_dlt` before/after on the same config; spot-check reward trajectory via `scripts/utils/report_eval.py`.
4. *(V5.x, if approved)* **Persistent MLIR worker pool** — spawn-based pool in `execution.py`.
