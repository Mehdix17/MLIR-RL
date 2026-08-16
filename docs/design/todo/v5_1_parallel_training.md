# V5.1: Distributed PPO Training (Dask) — Design

**Status**: Draft
**Version**: **V5.1** of the new MLIR-RL generation (V5 → V5.1 → V5.2 → V5.3)
**Target package**: `rl_autoschedular_v5` (in-place extension — the single-node path stays byte-identical; the distributed path is additive)
**Base**: V5 (`v5_training_acceleration.md`, done) + `config/v5/v5_single_node.json`
**Depends on**: V5 — the accelerated pipeline and unified config
**Precedes**: V5.2 (full-model eval, `v5_2_full_model_eval.md`), V5.3 (expanded action space, `v5_3_expanded_action_space.md`)

> **Decisions locked (2026-08-13, user)**:
> 1. V5.1 = **distributed PPO**: ONE shared brain trained by a driver + 16 Dask
>    worker nodes. 64 benchmarks per iteration, 4 per node, executed
>    **4-parallel-within-node** (4 threads on a 128-core node; user decision).
> 2. Transport = **Dask** (existing `DaskManager`, spawn-based, crash recovery).
>    torch.distributed rejected (all-new machinery, one dead rank kills the job).
> 3. **The single-node system stays** as the default path, and V5.1's experiment
>    is a **head-to-head comparison**: single-node vs distributed, same config,
>    same budget, same eval split.
> 4. Replaces the earlier independent-runs design (16 separate brains) — that
>    interpretation was corrected by the user; this doc supersedes it.
> 5. Doc reorg (V5.2/V5.3 renames + cross-links) already done.
> 6. Knobs resolved (2026-08-13): worker topology = **lean** (16c/16G, no
>    `--exclusive`, 16 workers pack onto ~2 physical nodes); seed policy =
>    **matched seeds** (both runs share the same seed — the only difference is
>    the architecture); pipelining = **out of v1** (tracked in
>    `v5_future_ideas.md`). SIZING REVISED 2026-08-14 from live measurement
>    (ps on bn* nodes): workers ~0.7GB RSS, ~4 cores during exec bursts →
>    worker sizing now lives in the JSON config (`dask_worker_cores=8`,
>    `dask_worker_mem=3GB`, `dask_worker_exclusive=false`,
>    `dask_node_count=16` in `config/v5/v5_distributed.json`); precedence
>    env > config > default. No launch env vars needed for a standard
>    distributed run: `sbatch scripts/train/train.sh config/v5/v5_distributed.json`.

---

## 1. Summary

One policy (the brain) lives on a **driver** node. Each PPO iteration, the driver
samples 64 benchmarks (same `randperm` as today), partitions them into 16 groups
of 4, and dispatches one group per Dask worker node. Each worker steps its 4
envs **in parallel within the node** (env loop + MLIR execution), using a local
replica of the current weights, and returns its trajectory segment. The driver
concatenates all 16 segments into the full 64-benchmark trajectory and runs the
**same `ppo_update` as today** (one update per iteration, identical PPO
semantics). The single-node path (`DASK_NODES` unset) is untouched and remains
the default; the comparison run measures wall-clock, training dynamics, and
final eval of the two systems.

Honest expectation (from V5's measurements): at `bench_count=64` the distributed
system will be **comparable to, not faster than**, the single node (64 exec
children either way; sync overhead added). Its wins: the driver is freed from
collection/execution, the collection phase (17.8s on one node) parallelizes
across workers, and the path to scale (`bench_count` up, more workers) is open.
The comparison quantifies the real gap. Pipelining (overlap exec with update) is
a follow-up, deliberately out of v1 to keep the comparison clean.

---

## 2. Scope

**In scope:**
1. Env-var-ize Dask worker job sizing in the v5 package copy of `dask_manager.py` (28c/100G/`--exclusive` hardcoded → configurable).
2. Refactor `ppo.py`: extract the per-benchmark rollout (env loop) into a reusable function — single-node `collect_trajectory` keeps identical behavior.
3. NEW `rl_autoschedular_v5/distributed.py`: worker-side `rollout_group` (4-parallel rollouts with local model replica) + driver-side `collect_distributed_trajectory` (partition, dispatch 16 group tasks, gather, concatenate, failure handling).
4. `scripts/train/train.py`: branch to the distributed collector when `DASK_NODES>0`; single-node path unchanged.
5. Comparison experiment: paired single-node vs distributed runs (same config, same seed policy, same budget) + comparison report.
6. Checkpoint compatibility: same model architecture on both paths → checkpoints interchangeable (resume distributed from single-node checkpoint and vice versa).

**Explicitly out of scope:**
- Pipelining (overlap exec/update) — **out of v1** (changes update cadence and muddies the comparison); tracked in `v5_future_ideas.md`.
- Any change to the RL algorithm, reward, action space, model, `opt_level`, or `ppo_batch_size` — both paths use the identical `ppo_update`.
- Distributed **eval** — eval stays a separate single-node job (V5 stance), run identically for both systems' checkpoints.
- `torch.distributed`, mpi4py, or any new transport.
- Changes to the root `utils/dask_manager.py` (hardcoded off — do not touch; the package copy is the one that matters).

---

## 3. Architecture (verified against code, 2026-08-13)

### 3.1 Per-iteration protocol (synchronous, one round-trip)

```
driver (main Slurm job)                   16 Dask worker nodes (spawned via SLURMCluster)
────────────────────────                  ────────────────────────────────────────────
1. randperm(train split)[:64]             
2. partition → 16 groups of 4            
3. weights = model.state_dict() (~0.5MB)  
4. dispatch 16 group tasks (one per worker, pinned via workers=)
                                          each worker: rollout_group(group_i, weights):
                                             build local HiearchyModel ← weights
                                             4 rollouts IN PARALLEL (ThreadPoolExecutor(4)):
                                               env loop (step until terminal, sample locally)
                                               execute terminal states (isolated children)
                                             return TrajectoryCollector segment
5. gather 16 segments (concat via TrajectoryCollector sum — same pattern as ppo.py:132)
6. ppo_update(full_64_bench_trajectory)   ← same function as single-node
7. checkpoint every 50 iters (driver)
```

- **Brain update**: `ppo_update` runs ONLY on the driver, once per iteration, on
  the full 64-benchmark trajectory — exactly the same PPO semantics as the
  single-node loop. Workers never compute gradients.
- **Weight freshness**: synchronous protocol → workers always step with `W_t`,
  the same weights that collected the trajectory; no staleness (unlike
  async/IMPALA-style).
- **Collection parallelizes**: the env loop (17.8s of today's iteration) splits
  into 16 workers × 4 envs (~1-3s each). The driver only samples indices,
  dispatches, gathers, and updates.

### 3.2 Why Dask fits

- `DaskManager.map_objs` already implements "dispatch N objs across M workers,
  one at a time per worker" — the group-task design is a one-shot variant:
  16 tasks pinned to 16 workers (existing `__submit_obj(..., workers=)`).
- Spawn-based (MLIR-safe; `BindingsProcess.ENABLED` stays `False`).
- Worker crash recovery + persistent registrations (`load_train_data`,
  `load_main_exec_data`) already exist.
- **Non-issue confirmed**: root `utils/dask_manager.py` has `ENABLED = False`
  hardcoded (`:19`) — only the package copy (`rl_autoschedular_v5/utils/
  dask_manager.py:22`, `ENABLED = DASK_NODES > 0`) can spin a cluster. No
  double-cluster risk. **Caveat**: worker processes also have `ENABLED=True`
  (same env) — the worker rollout path must NOT instantiate `DaskManager`
  (it doesn't: it uses raw `Env` + `Execution`), and `DaskManager.__init__`
  only runs when called.

### 3.3 Single-node path (baseline, untouched)

`DASK_NODES` unset → `ENABLED=False` → `collect_trajectory` runs exactly as
today (ThreadPoolExecutor fallback). The rollout refactor (T2) must preserve
this byte-for-byte behavior — verified by rerunning the v5_small smoke profile
(entropy/reward/timing signature).

---

## 4. Components / Changes

| Path | Change |
|---|---|
| `rl_autoschedular_v5/utils/dask_manager.py` | Env-var-ize worker job: `DASK_WORKER_CORES` (default 28), `DASK_WORKER_MEM` (default 100GB), keep `--nodes=1`/`--exclusive` behind `DASK_WORKER_EXCLUSIVE` (default on, to preserve current behavior). Keep `single_task_slot=1` (harmless: 1 group task per worker). |
| `rl_autoschedular_v5/ppo.py` | Extract the per-benchmark env-loop body of `collect_trajectory` (`:60-104`) into `rollout_benchmark(env, bench_idx, model, cfg) -> TrajectoryCollector`. `collect_trajectory` calls it in a loop — behavior identical. |
| `rl_autoschedular_v5/distributed.py` | **NEW** — `rollout_group(group_indices, weights, ...)`: build model from weights, run 4 `rollout_benchmark`s in a `ThreadPoolExecutor(4)`, execute terminal states (4-parallel, isolated children), return the summed `TrajectoryCollector`. `collect_distributed_trajectory(data, model, step)`: randperm(64) → 16 groups → dispatch one task per worker → gather → sum collectors → failure handling (None → `-20` penalty path, same as today's `failed_seq`). |
| `scripts/train/train.py` | Branch: `if DASK_NODES>0: trajectory = collect_distributed_trajectory(...) else: collect_trajectory(...)`. Everything else (loop, checkpointing, resume, SIGABRT handler) unchanged. |
| `scripts/utils/report_parallel.py` | **NEW** — comparison report: per-system iter wall-clock (median/mean), entropy/reward/speedup curves, final eval (mean/geomean speedup on shared eval split). |
| `config/v5/v5_single_node.json` | Unchanged (bench_count=64, ppo_batch_size=64). Distributed is env-driven, not config-driven. |
| `docs/design/todo/v5_1_parallel_training.md` | This doc. |

Not touched: root `utils/dask_manager.py`, `execution.py`, `benchmarks.py`,
`model.py`, `observation.py`, `env.py`, `eval.py`, `scripts/eval/*`,
`scripts/train/train.sh` (driver resources via sbatch CLI, see §7).

---

## 5. Data Model / API

**No new Config fields** — the distributed switch and worker sizing are env vars
(consistent with how `DASK_NODES` already works):

| Env var | Default | Meaning |
|---|---|---|
| `DASK_NODES` | unset | >0 → distributed path (driver + N workers) |
| `DASK_WORKER_CORES` | 28 (current hardcode) | cores per worker job |
| `DASK_WORKER_MEM` | 100GB (current hardcode) | mem per worker job |
| `DASK_WORKER_EXCLUSIVE` | 1 (current behavior) | add `--exclusive` to worker jobs |

**Worker task payload** (per iteration): `(group_indices: list[int] (4), weights: dict[str, Tensor] (~0.5MB))`. Cloudpickle handles torch tensors; 16 × 0.5MB ≈ 8MB/iteration — negligible on Dask's intra-cluster transport.

**Worker task return**: `TrajectoryCollector` (transitions for 4 benches) — serialized ≈ 1-2MB/iteration total (64 benches × ~10-30 transitions × obs ~300 dims). Concatenated on the driver via the existing `TrajectoryCollector` sum (`ppo.py:132` pattern).

**Checkpoints**: driver-only, identical format (`{'model', 'optimizer', 'step'}`) → `--resume` works across paths.

**Results layout**: flat, v4.9-style — `results/.../v5_single_node_agent/` for single-node, `results/.../v5_distributed_agent/` for the distributed run (distinct `results_dir` in the config copy used for the comparison).

---

## 6. Edge Cases & Error Handling

- **Worker dies mid-iteration** (SIGABRT, OOM): Dask detects the dead worker, restarts it and renews persistents (existing `DaskManager` machinery); the failed group's tasks return None → driver applies the failed-bench path (rewards `-20.0`, speedup 0, excluded from means — same as today's `failed_seq`), trajectory = 60 benches that iteration. No crash, no hang.
- **Group timeout**: `map_objs` uses `batch_timeout=300` (training). A group runs 4 rollouts in parallel, each capped by `MIN_EXEC_TIMEOUT` (default 300) — worst case ≈ 300s + env loop, so 300 can false-trigger on stragglers. The distributed dispatcher must use a higher per-task timeout (e.g. `MIN_EXEC_TIMEOUT * 2` or a `DASK_GROUP_TIMEOUT` env, default 600).
- **SIGABRT in worker**: the worker rollout function installs the same SIGABRT handler as `train.py` (native MLIR crash → Python exception → task returns failure, not hard kill).
- **Workers must not instantiate DaskManager** (`ENABLED=True` on workers): the rollout path uses raw `Env`/`Execution` only. Noted in code review checklist for T3.
- **Benchmark assignment changes every iteration** (randperm): workers load the FULL train split once (persistent `load_train_data`, existing pattern) and index locally — no per-iteration payload of benchmark data.
- **Worker startup cost**: 16 workers × feature extraction of the full split (~1-2 min each, parallel) — paid once, same as the driver pays today.
- **Determinism for the comparison**: **matched seeds** — identical
  `torch.manual_seed` in both runs (bench sampling + weight init); worker RNG
  streams differ by design (acceptable — each run is reproducible as a system).
- **Checkpoint compat**: same architecture → cross-load verified in the comparison (resume distributed from single-node checkpoint, spot-check dynamics).
- **Memory**: worker = Benchmarks object (~1-2G) + 4 exec children (~4G) + torch ≈ 6-8G → 16G per worker is ~2x headroom. Driver = sampling + ppo_update only (no exec) → 16G is ample.

---

## 7. Resource Requirements

| Job | Partition | CPUs | Mem | Wall | Notes |
|---|---|---|---|---|---|
| Driver (`train.sh` + `DASK_NODES=16`) | `compute` | 16 | 16G | 7-00:00:00 | no execution on driver; torch(4) + dispatch + update |
| 16 Dask workers | `compute` | 16 (`DASK_WORKER_CORES`) | 16G (`DASK_WORKER_MEM`) | 7-00 (SLURMCluster walltime) | 4 exec children + env loop; ~8G real → 16G headroom |
| Eval (both systems) | `compute` | 64 | 16G | 7-00:00:00 | existing `eval.sh` header, unchanged |

**Packing — DECIDED (lean)**: 16 workers × 16c pack ~8 per 128-core node →
**~2 physical nodes** for the comparison run (`DASK_WORKER_EXCLUSIVE=0`,
`DASK_WORKER_CORES=16`, `DASK_WORKER_MEM=16G`). Training dynamics are identical
to a spread topology — workers are independent processes using ~4-8 cores each;
the remaining ~120 cores per node sit unused either way. The literal
"one worker per physical node" topology (`--exclusive`) remains available as a
paper-framing variation (see `v5_future_ideas.md`), not used for the
comparison.

---

## 8. Comparison Experiment (the V5.1 deliverable)

1. **Paired runs**: single-node (`results/.../v5_single_node_agent/`, config
   `v5_single_node.json` unchanged) vs distributed (`results/.../v5_distributed_agent/`,
   same config with `results_dir` swapped + `DASK_NODES=16`), same
   `nb_iterations`, **matched seeds** (identical seed for bench sampling +
   weight init in both runs — the only difference between the two systems is
   the architecture). Both CPU-only, `compute` partition.
2. **Metrics**: median/mean `iter_time_dlt`; entropy/reward/speedup curves from
   train logs; MaxRSS (`sacct`) for both systems.
3. **Eval**: `eval.sh` on the shared eval split for both checkpoints (same
   checkpoint steps); `report_parallel.py` builds the comparison table.
4. **Cross-check**: resume distributed from the single-node checkpoint for a
   few hundred iterations — verifies checkpoint compat and dynamics continuity.
5. **Expected result framing**: distributed ≈ single-node wall at
   `bench_count=64` (with sync overhead), collection phase parallelized, driver
   idle during exec; the scaling argument (bench_count↑ / workers↑) is the
   paper-relevant outcome, measured here for the first point on the curve.

---

## 9. Tasks (for feature-develop, in order)

- [x] **T0 — Doc reorg** (done with the version reorder): renames
  `v5_1_full_model_eval.md` → `v5_2_full_model_eval.md`,
  `v5_2_expanded_action_space.md` → `v5_3_expanded_action_space.md`, all
  cross-links fixed (AGENTS.md, V5 done-doc, HPO_PLAN.md). Verify:
  `grep -rn "v5_1_full_model_eval\|v5_2_expanded_action_space" --include="*.md" .`
  = 0 hits outside `graphify-out/`.
- [x] **T1 — Env-var-ize worker sizing** (`rl_autoschedular_v5/utils/dask_manager.py`): `DASK_WORKER_CORES`, `DASK_WORKER_MEM`, `DASK_WORKER_EXCLUSIVE` (defaults preserve current 28c/100G/exclusive). Verify: `py_compile`; `python -c` import with env vars set → SLURMCluster constructed with the overrides. ✅ 2026-08-13: py_compile OK; defaults preserve current behavior.
- [x] **T2 — Rollout refactor** (`ppo.py`): extract the env-loop body (`:60-104`) into `rollout_benchmarks(...)`; `collect_trajectory` becomes a thin loop over it. ✅ 2026-08-13: extracted (returns envs/states/tcs/entropies); entropy logging hoisted to the caller (same values/order); py_compile OK. Single-node parity to be confirmed by the smoke.
- [x] **T3 — `distributed.py`**: `rollout_group` (build model from weights, 4-parallel rollouts via `ThreadPoolExecutor(4)`, execute, return collector) + `collect_distributed_trajectory` (randperm → 16 groups → one task per worker → gather → sum → failure handling → group timeout ≥ 600s). ✅ 2026-08-13: implemented + import-tested in the conda env. API notes: `Client.gather` has **no `timeout`** in distributed 2025.10.0 → used `distributed.wait(futures, timeout=GROUP_TIMEOUT)`; all-groups-failed raises a clear error instead of training on empty data. Weights payload measured **~21MB** (not 0.5MB — full HierarchicalModel state_dict); 16×21MB/iter over Dask TCP is fine.
- [x] **T4 — `train.py` branch**: `if int(os.getenv('DASK_NODES', 0)) > 0: collect_distributed_trajectory else collect_trajectory`. ✅ 2026-08-13: branch added (imports `distributed` only when enabled); py_compile OK; `DASK_NODES` unset path byte-identical (no code path change).
- [x] **T5 — Distributed smoke**: `DASK_NODES=4` short run (~50 iters): 4 workers, 16 benches/iter, 4-parallel within node. Verify: `squeue` shows driver + 4 worker jobs; log shows "Collecting 64 benchmarks using 4 workers" (16 groups of 4 → still 64 total), no SIGABRT, iteration completes; worker logs show 4-parallel exec. ✅ **2026-08-13** (job 17226164): **passed** — 4 workers up and stable, "Collecting 64 benchmarks on 4 worker nodes (4 per node)", first collection 1624 transitions in 6:15 (worker startup + per-worker feature extraction), then **~15s/collection, ~29s/iter** for 50/50 iterations; job COMPLETED cleanly ("Training completed", 38:35), checkpoint saved. **Cluster-infrastructure fixes discovered** (all in `dask_manager.py` + `train.sh`): (1) `sbatch`/`scancel` must resolve to absolute paths (bare exec ENOENT on some nodes); (2) worker job script needs `shebang='#!/bin/bash'` (bare PATH breaks `env bash`); (3) Slurm batch jobs here do NOT inherit the driver env — worker PATH/PYTHONPATH must be built deterministically (the driver's env carries literal `$PYTHONPATH`/`$PATH` from `.env` self-references); (4) `Client(cluster)` races SpecCluster state-correction and closes fresh workers — connect by address after awaiting the cluster; (5) worker acceptance via `scheduler_info()` (squeue unreliable from driver); (6) no `signal.signal()` at `distributed.py` module level (illegal in Dask worker threads — nanny recovers dead workers instead).
- [x] **T6 — Paired comparison runs** (LAUNCHED 2026-08-13, relaunched after fix): single-node `v5_small.json` (job 17227579, bn002) and distributed `v5_distributed.json` (job **17228493**, bn003) with `DASK_NODES=16`, matched seed 42, paper-matching hyperparameters (`max_num_loops`/`max_num_load_store_dim` 12→7 to match `paper_original.json`; all other shared keys verified equal by `tests/test_v5_parallel.py::test_paper_hyperparameter_parity`). Workers: `--constraint=bergamo` → all 16 on `bn*` nodes, 7-day wall. **Resume verified with distributed** (job 17227202): "Resumed model + optimizer", `start=51, end=55`, healthy collections, `model_55.pt` chained, COMPLETED. **Critical bug found during launch: `Client.scheduler_info()` defaults to `n_workers=5`** — it returns info for only 5 workers, so the driver silently "saw" 5 of 16 workers; the old acceptance poll even scancel'd 11 healthy workers as "never connected". Fixed: `scheduler_info(n_workers=-1)` in `workers_names` + acceptance replaced with `wait_for_workers` (live `identity` RPC). Verified live: default → 5, `-1` → 16. **After fix: "Collecting 64 benchmarks on 16 worker nodes (4 per node)", collection ~6s, iter ~16s** (≈2× faster than single-node ~32.6s). Single-node leg will exceed the 7-day wall (~9d ETA) → one `--resume` chunk needed. Verify on completion: `report_parallel.py` table (iter wall median, collection, PPO fit, MaxRSS via `--*-job` ids).
- [ ] **T7 — Eval + cross-check**: `eval.sh` both checkpoints on the shared eval split; `--resume` distributed from single-node checkpoint for a few hundred iters. Verify: eval JSONs comparable; dynamics continuous across the resume.
- [ ] **T8 — Docs**: AGENTS.md (design-docs line already updated for the reorder; add the distributed/driver invocation to Commands once stable); move this doc to `done/` when the user confirms results.

---

## 10. Open Questions — ALL RESOLVED (2026-08-13)

| Question | Decision |
|---|---|
| Worker topology for the comparison | **Lean**: 16c/16G, `DASK_WORKER_EXCLUSIVE=0` → 16 workers pack onto ~2 physical nodes. `--exclusive` 16-node topology kept only as a paper-framing variation (`v5_future_ideas.md`). |
| Seed policy | **Matched seeds** — identical seed in both runs; the architecture is the only variable. |
| Pipelining (overlap exec/update) | **Out of v1** — moved to `v5_future_ideas.md`; added later once the baseline comparison is locked. |
