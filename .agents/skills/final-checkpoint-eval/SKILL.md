---
name: final-checkpoint-eval
description: Evaluate the surviving (top-9, post-cleanup) checkpoints of an experiment with N repeat runs each, aggregating per-benchmark speedups by median. Runs AFTER cleanup-checkpoints. Computes median speedup per benchmark across N eval runs and writes one json per checkpoint to eval_final/. Use when the user wants final/robust checkpoint numbers after pruning, multi-run median eval, or reduced-variance benchmark results.
---

# Final Checkpoint Evaluation — N-run Median

Runs **after** `cleanup-checkpoints`: the input is whatever `model_*.pt` files remain in
`<results_dir>/models/` (the top-9 KEEP set). Every surviving checkpoint is evaluated
`num_runs` times; per benchmark we record the **median** of the per-run speedups
(`baseline_time / exec_time`). Results land in a dedicated `<results_dir>/eval_final/`
directory — one `<checkpoint_<n>.json` per checkpoint — so the single-run `eval/`
artifacts are left untouched.

After all checkpoints are evaluated, the orchestrator **picks the best checkpoint** (max
geometric-mean speedup across the median jsons) and writes the best-checkpoint CSVs the
plotting skill consumes — these now reflect the **median** results, not single-run:
  `csvs/best_checkpoint_speedups.csv` (single row: best checkpoint + its geo-mean)
  `csvs/best_checkpoint_benchmark_family_results.csv`
  `csvs/best_checkpoint_operation_type_results.csv`

Deletion-free operation: this skill only creates files under `eval_final/` and overwrites
the best-checkpoint CSVs under `csvs/`. It does not remove or overwrite `eval/`, the
per-checkpoint ranking (`checkpoint_speedups.csv`), or `models/`.

## What drives it
- A **dedicated final-eval config** JSON that carries the two tunables up front:
  - `num_runs` (e.g. `5`)
  - `aggregator` (`"median"`)
  plus `eval_config` (the agent's base config `eval.py` needs), `results_dir`, `dataset`,
  and Slurm sizing (`cpus_per_task`, `mem`, `time`).
- Config files live in `config/<dataset>/eval/<agent>_final_eval.json`.
- The orchestrator `scripts/eval/final_eval.py` reuses `utils/csvs.load_baseline` for the
  per-dataset baseline (same one that ranks checkpoints), so numbers are comparable to the
  `checkpoint_speedups.csv` ranking.
- The batch wrapper `scripts/eval/final_eval.sh` sets up the Slurm env (conda, LLVM
  PYTHONPATH, .env) then runs the orchestrator **sequentially in one job**.

## Flow (mandatory)

### Step 1 — Run the self-test (first use / after edits)
```bash
python scripts/eval/final_eval.py <config> --self-test
```
Verifies the median aggregation (`{bench: median_speedup}`, NaN-free) on synthetic data.

### Step 2 — Create/check the config
`config/<dataset>/eval/<agent>_final_eval.json`. Confirm `num_runs` and `aggregator`
match the request (e.g. 5 / median), and `results_dir`/`eval_config` point at the
experiment.

### Step 3 — Dry run (no evaluation)
```bash
python scripts/eval/final_eval.py config/<dataset>/eval/<agent>_final_eval.json --dry-run
```
Prints: surviving checkpoints on disk (must match the cleanup KEEP set, e.g. 9), `num_runs`,
`aggregator`, and the total eval-invocation count (`len(ckpts) * num_runs`). Confirm the
checkpoint list is the intended one before submitting.

### Step 4 — Submit the batch job
```bash
sbatch scripts/eval/final_eval.sh config/<dataset>/eval/<agent>_final_eval.json
```
Sequential single job; logs to `logs/final_eval_%j.out`.

### Step 5 — Verify
- Job finished without `eval.py failed` in `logs/final_eval_*.out`.
- `ls <results_dir>/eval_final/` → one `checkpoint_<n>.json` per surviving checkpoint.
- Log line `BEST CHECKPOINT (median): <n>` printed.
- `csvs/best_checkpoint_speedups.csv` has one row = that best checkpoint; the family/op
  best-checkpoint CSVs are regenerated from the **median** results.
- Each eval_final json is `{bench_name: median_speedup}`.

## Hard rules
- **Never** run before `cleanup-checkpoints` on the experiment — this skill evaluates only
  the surviving checkpoints in `models/` so it can produce wrong coverage if `models/` is
  still the full set.
- **Never** change `num_runs`/`aggregator` inline on the command line — they live in the
  config file only, so the run is reproducible from the config.
- **Never** delete or overwrite `eval/`, `csvs/`, or `models/`. Output is `eval_final/` only.
- If `models/` is empty or has far more than the expected KEEP count (e.g. 400 unpruned),
  STOP — cleanup hasn't run; tell the user.
- `aggregator` must be `"median"` (only that mode is implemented).

## Output shape
```
<results_dir>/eval_final/checkpoint_<n>.json   →  { "bench_name": <median_speedup> | null, ... }
<results_dir>/eval_final/                      (no pngs/, no subfolders beyond the jsons)
```