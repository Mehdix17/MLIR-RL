# Distributed PPO Training for MLIR Loop-Nest Scheduling (V5.1)

Reference document for the second MLIR-RL paper. Describes the distributed
reinforcement-learning training system introduced in V5.1 of the MLIR-RL
framework: one shared policy brain trained by PPO across a fleet of Slurm/Dask
workers, with a shared cross-worker execution cache.

---

## 1. Motivation

MLIR-RL trains a PPO agent that schedules (tiles, vectors, interchanges) MLIR
loop nests. Each training iteration:

1. samples 64 benchmarks from the dataset,
2. rolls out the agent's policy on each (action sequence),
3. executes the transformed MLIR modules to measure speedup over the baseline,
4. performs one PPO update on the collected trajectory.

In the single-node system (V5), all four steps run in one process on one node:
the 64 MLIR executions are the bottleneck (MLIR compilation + `mlir-cpu-runner`
is single-threaded per benchmark), and the driver also pays the policy forward
passes and the PPO fit. Measured iteration cost: ≈ 30 s, dominated by the
64-benchmark execution wave (~15 s) and sampling (~18 s).

The scaling problem: MLIR execution is embarrassingly parallel across
benchmarks, but the single-node design serializes it through one process. More
CPUs on the node do not help (each execution is single-threaded). The fix is to
spread the execution wave across machines.

## 2. Architecture

```
                    ┌──────────────────────────────┐
                    │  Driver (Slurm job, 8c/8G)   │
                    │  - policy model (shared)     │
                    │  - PPO update (per iter)     │
                    │  - exec-cache single writer  │
                    │  - checkpoints               │
                    └──────┬───────────▲───────────┘
                     weights│           │ trajectory
                    broadcast│           │ segments + new cache entries
                    ┌───────▼───────────┴───────────┐
                    │       Dask scheduler          │
                    └───────┬───────────┬───────────┘
         ┌──────────────────┘           └──────────────────┐
   ┌─────▼──────┐  16 × dask workers (8c/3G each)   ┌──────▼─────┐
   │  worker 1  │  each: policy replica + 4 envs    │  worker 16 │
   │ (4 benches │  executed 4-parallel in threads   │  (4 benches│
   │  /iter)    │                                    │  /iter)    │
   └────────────┘                                    └────────────┘
```

**One shared brain.** The policy weights live on the driver. Each iteration:

1. The driver dispatches 64 benchmarks as **16 group tasks** (4 benchmarks per
   worker, one group task per worker).
2. Each worker loads the current weights into a local policy replica, rolls out
   its 4 benchmarks (4 environments run **in parallel within the worker**),
   and executes the terminal MLIR modules.
3. Workers return **trajectory segments** (observations/actions/rewards),
   per-benchmark speedups, and any **new execution-cache entries**.
4. The driver concatenates the 64-benchmark trajectory, runs **one PPO update**,
   persists checkpoints, and the loop repeats.

Key invariant: **workers never touch gradients or the optimizer** — they are
stateless roll-out engines. The driver is the only place weights change, which
keeps the distributed system bit-compatible with the single-node PPO semantics.

## 3. Transport: Dask over Slurm (not torch.distributed)

Chosen: **Dask** (`dask-jobqueue` SLURMCluster) — 16 worker *jobs* on the
`compute` partition, one dask worker process per job, `processes=1`.

Why Dask over torch.distributed:

| Concern | Dask | torch.distributed |
|---|---|---|
| Worker crash recovery | scheduler re-queues; failed groups surface per-iteration | all-reduce hangs need external watchdog |
| MLIR C++ state safety | worker processes are `spawn`-based, no fork | NCCL/gloo collectives assume process symmetry |
| Weight broadcast | Dask futures move the ~21 MB `state_dict` over local TCP | needs rank-0 broadcast logic |
| Scheduler/debugging | `distributed` dashboard + worker logs per job | opaque |

**Process model.** All worker processes use the `spawn` context
(`BindingsProcess.ENABLED` stays `False`; `fork` corrupts MLIR C++ state).
Executions are additionally isolated in per-benchmark child processes
(`mlir-cpu-runner` with a code-cache key).

## 4. From the legacy Dask infrastructure to distributed PPO

MLIR-RL already shipped a Dask integration before V5.1 — but it was a
different beast, and it was never even enabled. This section documents exactly
what the old code did and what V5.1 changed to make distributed training work.

### 4.1 The legacy system (V0–V5): distributed *execution*, disabled

The original `utils/dask_manager.py` (V0 era) defined a `DaskManager` with:

- `ENABLED = False` **hardcoded** — the cluster was never started; the
  `ThreadPoolExecutor` fallback (`SLURM_CPUS_PER_TASK` workers on the driver
  node) was the de-facto execution path for the whole V0–V5 line;
- hardcoded worker resources: `cores=28`, `memory='100GB'`,
  `--exclusive`, 7-day wall;
- a single use-site in `ppo.py`:
  `dm.map_states(__execute_states, states, data, ...)`.

The legacy design's division of labour: **the driver ran the entire PPO loop
in-process** — sampling, environment stepping, policy forward passes — and
only the final benchmark *execution* (`__execute_states`: apply transforms →
`mlir-cpu-runner`) was mapped onto dask workers. If it had been enabled, it
would have been *distributed benchmark execution*, not distributed training:
the trajectory, the policy, and the PPO update all stayed on the driver, and
the 64-execution wave was the only parallelized stage.

### 4.2 What V5.1 changed

**1. Enablement + ownership.** The package copy
(`rl_autoschedular_v5/utils/dask_manager.py`) became the live one:
`ENABLED` turned env-driven (`DASK_NODES`), later config-driven
(`dask_node_count` in the experiment JSON, env override for smoke tests). The
root copy stays hardcoded-off for backward compatibility.

**2. Semantics: from execution-only to full rollout.** New
`distributed.py` replaces the `map_states` pattern with `rollout_group` +
`collect_distributed_trajectory`. The parallel unit is no longer "execute one
terminal state" but **"roll out 4 benchmarks end to end"**: the worker loads
the current policy weights (broadcast by the driver), steps the environments,
executes the transformed modules, and returns the complete trajectory segment
+ speedups + new cache entries. The driver's remaining job is concatenation,
one PPO update, checkpointing, and the cache merge. This is the
single-node-vs-distributed PPO boundary that the paper compares.

**3. Cluster correctness fixes.** The old code had never run, so making the
fleet actually come up took nine smoke runs; every root cause is baked into
the final code (details in §11):

- Slurm binaries resolved absolutely (`/opt/slurm/default/bin`) — worker
  environments don't inherit the submitter's PATH;
- worker job scripts get `shebang='#!/bin/bash'` and a **deterministic**
  prologue exporting PATH/PYTHONPATH/MLIR env — Slurm jobs don't inherit
  driver env, and `.env` self-references (`$PYTHONPATH`) expand to empty in a
  bare environment;
- `Client(cluster)` races dask-jobqueue's state correction → connect by
  scheduler address after awaiting the cluster;
- worker acceptance polls `client.scheduler_info()` (workers actually
  connected), not `squeue` — a stale squeue view caused the acceptance loop to
  exit early and scale-down just-registered workers;
- no `signal.signal()` at module level (worker imports run in a Dask thread;
  the nanny restart-looped on the ValueError);
- spawn-isolated children must be module-level picklable functions (no
  closures, no name mangling).

**4. Shared execution cache (new channel).** The legacy cache was per-process
file reads with no cross-worker mechanism. V5.1 adds the
`new_cache_data` return channel (workers send their new
(benchmark, cache-key, exec-time) entries with each trajectory segment) and
the driver-side single-writer merge (§5). This is what turns the cache into a
shared, deduplicated resource.

**5. Resource engineering.** The legacy hardcodes (28c/100GB/`--exclusive`)
and the V5 driver (64c/16G) were replaced by measurement-driven, config-owned
allocations: 8c/3G workers, 8c/8G drivers (§6).

**6. Operational plumbing.** `--resume` across restarts (weights + optimizer
+ the shared cache), the experiment registry, per-iteration benchmark-failure
counters, and the anomaly watchdog — none of which existed in the legacy Dask
path.

### 4.3 Before / after

| Aspect | Legacy (V0–V5, disabled) | V5.1 (live) |
|---|---|---|
| Enablement | `ENABLED = False` hardcoded | config `dask_node_count` (env override) |
| Parallel unit | benchmark execution only (`map_states`) | full 4-bench rollout (`rollout_group`) |
| Policy weights | driver only | broadcast to workers per iteration |
| Trajectory | collected on driver | segments returned, concatenated on driver |
| PPO update | driver (unchanged) | driver, one update per iteration (unchanged semantics) |
| Exec cache | per-process file reads | shared single-writer merge + in-memory snapshot |
| Worker size | 28c / 100GB / exclusive | 8c / 3G / shared (config-owned) |
| Driver size | 64c / 16G | 8c / 8G |
| Ops | — | resume, registry, failure counters, watchdog |

## 5. Shared execution cache (key contribution)

MLIR execution dominates the iteration cost, and the same (benchmark, action
sequence) pair recurs as the policy converges. The system therefore maintains
a **shared, deduplicated execution cache**:

- **Format**: `exec_data.json`, `{bench_name: {code_cache_key: exec_time_ns}}`.
  The key encodes the full transformation history (tile sizes, loop order,
  vector flags) of the produced module.
- **Single writer**: workers return their new entries with each trajectory
  segment; the driver merges them and atomically rewrites the file
  (write-tmp + `os.replace`) once per iteration. There is no multi-writer
  corruption, and the file contains only unique (benchmark, key) pairs.
- **Live reads**: cache lookups read the file on shared Lustre, so every
  worker sees the driver's merges from the previous iteration. Measured hit
  rate ≈ **80%** (6,421 unique misses over 489 iterations ≈ 13 misses/iter
  out of 64 dispatched).
- **In-memory snapshot + periodic refresh**: per-lookup file parsing (≈ 33 ms
  per 5.3 MB load) was eliminated by snapshotting the file into worker memory
  at group-task start, refreshed every **5 iterations**
  (`DASK_CACHE_REFRESH_INTERVAL`); workers merge their *own* new execs into
  the snapshot in place, so a longer refresh interval never re-executes a
  worker's own work — it only delays other workers' entries by ≤ 5 iterations.
- **Duplicate-miss instrumentation**: the driver counts misses that already
  exist in the merged cache (cross-worker duplicates). Measured **0–2 per
  iteration (≤ 3 %)** — the single-writer design already prevents duplicate
  *work* across iterations; the residual window is the same benchmark sampled
  twice within one iteration before the driver's write lands.

Cost accounting (per iteration): cache handling adds ≈ 0.1 s (refresh) and
removes ≈ 2 s of per-lookup parsing that both the single-node path and the
initial distributed version paid — i.e. the shared cache is strictly cheaper
than the single-node lookup pattern.

## 6. Resource engineering (lean, measurement-driven)

Worker and driver sizes were **measured, not estimated** (`ps` RSS on the
compute nodes), then set with ≈ 1.5–3× headroom:

| Resource | Allocated (initial) | Measured | Allocated (final) |
|---|---|---|---|
| Worker RAM | 16 G | ~0.7 GB RSS | **3 G** (≈ 2 G effective after dask-jobqueue trim) |
| Worker CPU | 16 c | ~4 cores during exec bursts | **8 c** |
| Driver RAM | 16 G | ~3.3 GB RSS | **8 G** |
| Driver CPU | 64 c | ~2 cores | **8 c** |

The fleet's total allocation dropped from **388 CPUs / 144 GB to 276 CPUs /
112 GB** for two runs (driver fix alone frees 112 CPUs). All knobs live in the
experiment's JSON config (`dask_node_count`, `dask_worker_cores`,
`dask_worker_mem`, `dask_worker_exclusive`) with env-var overrides for
launch-time smoke tests; the launch command is plain
`sbatch scripts/train/train.sh <config>`.

Per-user HPC caps: QOS `small` allows 2,048 CPUs / 7.5 TB per user; the four
running experiments use ~544 CPUs (26%). The GPU partition's QOS `c2` caps at
384 CPUs — a fleet of > 16 workers would exceed it there, so GPU-node runs must
size down (relevant only if training moves to the `nvidia` partition).

## 7. Reproducibility and operations

- **Matched seeds**: both the transformer (V5) and LSTM ablation
  (`v5_no_transformer`) runs share `seed=42`; worker RNG streams differ by
  design, both runs reproducible.
- **Paper-parity configs**: all V5.1 configs are verified against
  `paper_original.json` over 31 shared hyperparameter keys (unittest gate).
- **Resume**: the driver reloads the latest `model_<n>.pt` + optimizer state
  and continues from `step+1`; the shared exec cache persists across restarts,
  so resumed runs keep their hit rate (verified: 50 → 55 smoke, then 2400 →
  2750 on the real run).
- **Registry + reporting**: every experiment is registered in
  `experiments.json`; `fast_report.py` auto-derives each experiment's state
  and surfaces the dask fleet, iteration count, and bench-failure counters.
- **Watchdog**: a silent cron watcher alerts only on anomalies (driver count
  < 2, workers < 16, > 5 bench failures/iter in the latest log, > 10
  cross-worker duplicate execs/iter).
- **Benchmark failure visibility**: each iteration logs the failure count;
  > 50% failures raises a red alert (guards against silent infrastructure
  degradation — the bug class that cost 9 invalid runs during development).

## 8. Results (measured so far)

All runs: 64 benches/iteration, 20,000-iteration budget, seed 42,
paper-parity hyperparameters.

| Metric | Single-node (V5) | Distributed (V5.1) |
|---|---|---|
| Iteration cost | ≈ 30 s | ≈ 24 s (16 workers) |
| Collection wave | 64 in-process | ~6 s across 16 workers |
| Iterations in equal wall time | 1,423 (cancelled at 7.1%) | 2,357 (11.8%) — **~1.7× faster** |
| Exec cache hit rate | n/a (per-process file reads) | ~80 % |
| Bench failures (ops_and_blocks) | 2–6/iter | 0–2/iter |

The speed advantage is a combination of parallel execution waves and the
shared cache; single-node was cancelled in favour of the distributed system
after the wall-time comparison. The LSTM ablation on the legacy paper dataset
shows higher failure rates (~11/64 ≈ 17 % per iteration) — likely fragile
benchmarks in the older dataset, under investigation.

## 9. Experiments matrix (second paper)

| Experiment | Dataset | Encoder | Status |
|---|---|---|---|
| v5_distributed | ops_and_blocks (12 K) | Transformer | training |
| v5_no_transformer | ops_and_blocks (12 K) | LSTM (paper) | training |
| v5_legacy_paper | legacy_paper (1,202) | Transformer | training |
| v5_no_transformer_legacy_paper | legacy_paper (1,202) | LSTM (paper) | training |

The 2 × 2 design isolates the encoder contribution (Transformer vs LSTM) and
the dataset contribution (modern 12 K-benchmark set vs the original paper set).

## 10. Limitations and future work

- **No pipelining**: collection and PPO update are strictly sequential per
  iteration; overlapping them (IMPALA-style async PPO) is the main latency
  lever, deliberately deferred to keep V5.1's comparison clean.
- **Scaling path**: at 64 benches/iteration, 16 workers form a single wave;
  more workers idle. The lever is `bench_count` (64 → 256+), not the fleet
  size.
- **Live cache broadcast**: measured cross-worker duplicates ≈ 0, so
  broadcasting the merged cache per iteration is not needed; the file-based
  read path already provides live sharing.
- **Parallel evaluation** (T7): the same Dask mechanism extends to the eval
  pipeline (checkpoint evaluation across workers).

## 11. Engineering notes (pitfalls paid for)

The distributed system went through 9 failed smoke runs; the fixes, all
baked into `dask_manager.py` / `distributed.py`, are:

1. Slurm binaries must be resolved absolutely (`/opt/slurm/default/bin`) —
   worker environments do not inherit the submitter's PATH.
2. Worker scripts need `#!/bin/bash` shebang and a **deterministic** env
   prologue (Slurm jobs do not inherit driver env; `.env` self-references like
   `$PYTHONPATH` expand to empty).
3. `Client(cluster)` races dask-jobqueue's state correction → connect by
   scheduler address after awaiting the cluster.
4. Worker acceptance must poll `client.scheduler_info()` (ground truth), not
   `squeue` (which can lag and kill just-registered workers via scale-down).
5. No `signal.signal` at module level in worker code (runs in a Dask thread;
   nanny restarts crash-loop).
6. Spawned children must be module-level picklable functions (no closures,
   no name mangling).
7. Benchmark failures must be counted and surfaced per iteration — the silent
   failure mode (all-64 fail → trained on penalty rewards) was the costliest
   bug class.

---

*Companion implementation doc: `docs/design/todo/v5_1_parallel_training.md`
(design contract + task log).*
