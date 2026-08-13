# V5 Future Ideas & Variations — Living List

**Status**: Active — a living list of experiments, variations, and follow-ups
for the V5 generation (V5 → V5.1 → V5.2 → V5.3). Ideas here are **not
versioned features**; they are things we can test and explore when a version is
locked or the comparison baseline is in. Versioned features live in their own
design docs (`v5_1_parallel_training.md`, `v5_2_full_model_eval.md`,
`v5_3_expanded_action_space.md`, `HPO_PLAN.md`).

Rules: one idea per entry, with a one-line rationale and the trigger (when to
try it). Add freely; move an idea out into its own design doc when it earns a
version slot.

---

## Distributed PPO follow-ups (post V5.1 comparison)

### 1. Pipelining — overlap exec with update
Hide the driver's ~13s `ppo_update` under the workers' execution: workers start
iteration N+1 while the driver updates on iteration N's trajectory. Iteration
wall drops from ~(exec + update) to ~max(exec, update). Cost: the update lags
one iteration (slightly off-policy), so results differ subtly from the
synchronous baseline. **Trigger**: after the V5.1 single-vs-distributed
comparison is locked and the baseline numbers are recorded.

### 2. bench_count scaling (64 → 128/256+)
With more workers (32/64 nodes), scale `bench_count` proportionally — the
paper-relevant claim for distributed training. Each node keeps 4 benchmarks.
**Trigger**: V5.1 comparison shows distributed ≈ single-node at 64 (expected);
this measures the scaling curve beyond one node's capacity.

### 3. Async distributed PPO (IMPALA-style)
Drop the synchronous barrier: workers always execute with the latest available
weights instead of waiting for the driver's update. Higher throughput, more
off-policy data. **Trigger**: if the sync barrier shows up as the bottleneck in
the comparison.

### 4. torch.distributed (GLOO) transport
Symmetric ranks, all-gather trajectory, identical `ppo_update` on every rank,
rank-0 checkpoints. Textbook alternative to Dask; rejected for v1 (all-new
machinery, one dead rank kills the job). **Trigger**: if Dask's per-iteration
serialization or worker-job overhead ever becomes measurable.

### 5. `--exclusive` 16-node topology (paper framing)
Literal one worker per physical node (`DASK_WORKER_EXCLUSIVE=1`, big cores) to
match a "16 nodes" description in the paper. ~95% idle; identical dynamics.
**Trigger**: only if the paper needs the physical-topology framing.

### 6. DASK_GROUP_TIMEOUT / straggler tuning
Per-group timeout knob for the distributed path (default 600s in v1). Tune
against observed worst-case groups. **Trigger**: if group timeouts false-trigger
or stragglers dominate the iteration.

### 7. Distributed eval
Fan the eval set (~2,363 benches) across workers instead of the single-node
eval job. **Trigger**: if eval wall-clock becomes a paper constraint (V5 doc
§3.7 originally reserved Dask for exactly this).

---

## Single-node levers (from the V5 done-doc, deferred)

### 8. Intra-node sampling/execution pipelining
Overlap the env-loop/collection phase with execution inside the single-node
loop (V5 measured ~3% of iteration). **Trigger**: if the V5.1 comparison shows
collection still dominates.

### 9. Early stopping on plateau
Stop training when reward/speedup plateaus; saves wall-clock. **Trigger**:
when training budget becomes the constraint (e.g. HPO sweeps).

### 10. Benchmark feature cache
Persist extracted features to disk; saves ~2-3 min per startup — and 16× that
for the distributed system (every worker re-extracts the full split at
startup). **Trigger**: before any run where startup time matters (all
distributed runs).

### 11. Multi-seed array runs
Same config, several seeds → variance bars for the paper. Needs the seed
policy machinery from V5.1. **Trigger**: when the paper requires error bars.

### 12. `reuse_experience` / `replay_count` experiments
Config-only knobs (already implemented) for experience replay — e.g. use them
in the distributed path to enlarge the effective batch. **Trigger**: if PPO
fit variance or sample efficiency becomes a question.

### 13. Dask for eval sweeps
Multi-thousand-bench eval across Dask nodes (the V5 §3.7 "when it would
matter" case — the infrastructure now exists via V5.1). **Trigger**: any eval
sweep too large for one node.

---

## Cross-version

### 14. Full-model eval + distributed synergy
V5.2's full-model eval worker pool (§3.6 of `v5_2_full_model_eval.md`) could
reuse V5.1's Dask worker machinery instead of its own spawn-based pool.
**Trigger**: during V5.2 implementation, if the two pools overlap.
