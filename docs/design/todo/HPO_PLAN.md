# Hyperparameter Tuning Plan: paper_transformer

**Status**: Active — parallel track (NOT a V5 version)
**Role in V5 generation**: HPO runs **in parallel** with V5 (`v5_training_acceleration.md`) and V5.1 (`v5_1_full_model_eval.md`), finding the best Transformer hyperparameters and feeding them into training **before** V5.2 (`v5_2_expanded_action_space.md`) expands the action space. Per user decision 2026-08-06, HPO is experimentation, not a versioned feature.
**Target package**: `rl_autoschedular_paper_transformer` (results transfer to `rl_autoschedular_v5` which shares the same Transformer encoder)

## Overview

Tune the **Transformer encoder architecture** only (d_model, nhead, num_layers, ffn_dim, dropout, pooling) using **Optuna** with TPE Bayesian optimization on the **ops_and_blocks** dataset. PPO/training hyperparameters remain fixed.

## Search Space

| Parameter | Type | Range | Constraint |
|-----------|------|-------|------------|
| `transformer_d_model` | int | {32, 48, 64, 96, 128, 192, 256} | — |
| `transformer_nhead` | int | {2, 4, 8} | `d_model % nhead == 0` |
| `transformer_num_layers` | int | {1, 2, 3, 4} | — |
| `transformer_ffn_dim` | int | {64, 128, 256, 512, 1024} | — |
| `transformer_dropout` | float | [0.0, 0.3] | — |
| `transformer_pooling` | categorical | {"cls", "mean"} | — |

## Fixed Parameters

All PPO params, lr=0.001, reward_shaping=false, gae_lambda=0.95, entropy_coef=0.01, etc. remain unchanged from `paper_transformer_large.json`.

## Objective Metric

**Mean speedup across all eval benchmarks** (`eval/average_speedup`) after full 20K iteration training.

## File Structure

```
scripts/hpo/
├── base_config.json          # Template config (copy of paper_transformer_large.json)
├── train_trial.sh            # Slurm script: single trial training (with --resume)
├── eval_trial.sh             # Slurm script: single trial evaluation
├── run_hpo.py                # Main orchestrator (Optuna study + batch Slurm management)
└── analyze.py                # Post-hoc analysis + best config extraction
```

## Execution Flow

```
┌─────────────────────────────────────────────────────┐
│  run_hpo.py (Python, runs interactively on login)   │
│                                                     │
│  1. Create Optuna study (SQLite: scripts/hpo/       │
│     study.db)                                       │
│  2. Pre-generate BATCH_SIZE trial configs           │
│  3. Submit all training jobs via sbatch             │
│  4. Poll until all training jobs complete           │
│     └─ If TIMEOUT: resubmit with --resume           │
│  5. Submit all eval jobs via sbatch                 │
│  6. Poll until all eval jobs complete               │
│  7. Read results, tell Optuna                       │
│  8. Repeat from step 2                              │
└─────────────────────────────────────────────────────┘
```

## File Details

### 1. `scripts/hpo/base_config.json`

Copy of `config/ops_and_blocks/train/paper_transformer_large.json` with:
- `results_dir` changed to template: `"results/hpo/trial_{trial_id}"` (replaced at runtime)
- `tags` updated to include `"hpo"` tag

### 2. `scripts/hpo/train_trial.sh`

Slurm script modeled after `scripts/train/train.sh`. Key differences:
- Accepts `TRIAL_ID` as environment variable
- Reads config from `scripts/hpo/trials/trial_{TRIAL_ID}/config.json`
- Sets `CONFIG_FILE_PATH` to that config
- Sets `RESUME_FROM=results/hpo/trial_{TRIAL_ID}` if the directory already has model checkpoints (auto-resume on timeout)
- Same SBATCH params: `--partition=compute --mem=32G --cpus-per-task=12 --constraint=bergamo --time=7-00:00:00`
- Sources `.env`, activates conda, sets `PYTHONPATH` (same as `train.sh`)
- Calls `python scripts/train/train.py`

The `--resume` logic:
```bash
TRIAL_DIR="results/hpo/trial_${TRIAL_ID}"
if [[ -d "$TRIAL_DIR/models" ]] && ls "$TRIAL_DIR"/models/model_*.pt 1>/dev/null 2>&1; then
    export RESUME_FROM="$TRIAL_DIR"
fi
```

If the job times out and is resubmitted, it automatically picks up from the latest checkpoint.

### 3. `scripts/hpo/eval_trial.sh`

Slurm script modeled after `scripts/eval/eval.sh`. Key differences:
- Accepts `TRIAL_ID` as environment variable
- Reads config from `scripts/hpo/trials/trial_{TRIAL_ID}/config.json`
- Sets `EVAL_DIR=results/hpo/trial_{TRIAL_ID}/models`
- Evaluates only the **last checkpoint** (`EVAL_LAST_ONLY=1`)
- Sets `FORCE_RUN_ID=trial_{TRIAL_ID}` to isolate the eval run
- Same SBATCH params as training

### 4. `scripts/hpo/run_hpo.py` (Main Orchestrator)

Python script that runs on the login node. Uses Optuna for hyperparameter optimization and manages Slurm job submissions.

**Key components:**

```python
import optuna
import json
import subprocess
import time
from pathlib import Path

BATCH_SIZE = 5  # Number of parallel trials per batch
TOTAL_TRIALS = 50  # Total number of Optuna trials
POLL_INTERVAL = 60  # Seconds between Slurm status checks

def objective(trial):
    """Optuna objective: generate config, submit train+eval, return speedup."""
    # 1. Sample hyperparameters
    d_model = trial.suggest_categorical("transformer_d_model", [32, 48, 64, 96, 128, 192, 256])
    nhead = trial.suggest_categorical("transformer_nhead", [2, 4, 8])
    if d_model % nhead != 0:
        while d_model % nhead != 0:
            nhead = trial.suggest_categorical("transformer_nhead", [2, 4, 8])
    num_layers = trial.suggest_int("transformer_num_layers", 1, 4)
    ffn_dim = trial.suggest_categorical("transformer_ffn_dim", [64, 128, 256, 512, 1024])
    dropout = trial.suggest_float("transformer_dropout", 0.0, 0.3)
    pooling = trial.suggest_categorical("transformer_pooling", ["cls", "mean"])

    # 2. Generate trial config
    trial_id = trial.number
    generate_trial_config(trial_id, {
        "transformer_d_model": d_model,
        "transformer_nhead": nhead,
        "transformer_num_layers": num_layers,
        "transformer_ffn_dim": ffn_dim,
        "transformer_dropout": dropout,
        "transformer_pooling": pooling,
    })

    # 3. Submit training job
    train_job_id = submit_slurm("scripts/hpo/train_trial.sh", trial_id)

    # 4. Wait for training to complete (poll + auto-resume on timeout)
    wait_for_job(train_job_id, trial_id, job_type="train")

    # 5. Submit evaluation job
    eval_job_id = submit_slurm("scripts/hpo/eval_trial.sh", trial_id)

    # 6. Wait for evaluation to complete
    wait_for_job(eval_job_id, trial_id, job_type="eval")

    # 7. Read result
    speedup = read_trial_result(trial_id)
    return speedup
```

**Batch mode:**

```python
def run_batch_mode():
    """Submit BATCH_SIZE trials, wait for all, collect results, repeat."""
    study = optuna.create_study(
        study_name="paper_transformer_hpo",
        storage="sqlite:///scripts/hpo/study.db",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )

    completed = len(study.trials)
    while completed < TOTAL_TRIALS:
        # Determine how many trials to submit in this batch
        batch_end = min(completed + BATCH_SIZE, TOTAL_TRIALS)
        batch_trial_ids = []

        # Generate configs and submit training jobs
        for i in range(completed, batch_end):
            trial = study.ask()  # Create a new trial
            d_model = trial.suggest_categorical(...)
            # ... sample params, generate config ...
            train_job_id = submit_slurm("scripts/hpo/train_trial.sh", trial_id)
            batch_trial_ids.append((trial, train_job_id))

        # Wait for all training jobs in this batch
        for trial, train_job_id in batch_trial_ids:
            wait_for_job(train_job_id, trial.number, job_type="train")

        # Submit all eval jobs
        eval_job_ids = []
        for trial, _ in batch_trial_ids:
            eval_job_id = submit_slurm("scripts/hpo/eval_trial.sh", trial.number)
            eval_job_ids.append((trial, eval_job_id))

        # Wait for all eval jobs
        for trial, eval_job_id in eval_job_ids:
            wait_for_job(eval_job_id, trial.number, job_type="eval")

        # Collect results and tell Optuna
        for trial, _ in batch_trial_ids:
            speedup = read_trial_result(trial.number)
            study.tell(trial, speedup)

        completed = batch_end
```

**Slurm job monitoring with auto-resume:**

```python
def wait_for_job(job_id, trial_id, job_type="train"):
    """Poll squeue until job finishes. Auto-resume on timeout."""
    max_resubmits = 10
    resubmit_count = 0

    while True:
        # Check if job is still running
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True
        )
        state = result.stdout.strip()

        if state == "":  # Job finished (not in queue)
            exit_code = get_job_exit_code(job_id)
            if exit_code == 0:
                return  # Success

            # Check if we need to resume (timeout or failure)
            if resubmit_count < max_resubmits:
                if job_type == "train" and not is_training_complete(trial_id):
                    resubmit_count += 1
                    new_job_id = submit_slurm("scripts/hpo/train_trial.sh", trial_id)
                    return wait_for_job(new_job_id, trial_id, job_type)
            return  # Give up after max resubmits

        time.sleep(POLL_INTERVAL)

def is_training_complete(trial_id):
    """Check if training completed all iterations by looking at saved models."""
    models_dir = Path(f"results/hpo/trial_{trial_id}/models")
    if not models_dir.exists():
        return False
    model_files = list(models_dir.glob("model_*.pt"))
    if not model_files:
        return False
    max_step = max(int(f.stem.split("_")[1]) for f in model_files)
    return max_step >= 19999  # 20K iterations (0-indexed)
```

**Config generation:**

```python
def generate_trial_config(trial_id, params):
    """Generate a trial-specific config JSON."""
    base = json.loads(Path("scripts/hpo/base_config.json").read_text())
    base.update(params)
    base["results_dir"] = f"results/hpo/trial_{trial_id}"
    base["tags"] = ["ops_and_blocks", "paper", "transformer", "hpo", f"trial_{trial_id}"]

    trial_dir = Path(f"scripts/hpo/trials/trial_{trial_id}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps(base, indent=2))
```

**Result reading:**

```python
def read_trial_result(trial_id):
    """Read the eval speedup from the trial's eval logs."""
    eval_log = Path(f"results/hpo/trial_{trial_id}/logs/eval/average_speedup")
    if eval_log.exists():
        values = [float(line.strip()) for line in eval_log.read_text().splitlines() if line.strip()]
        if values:
            return values[-1]  # Last recorded average speedup

    # Fallback: check eval exec times JSON
    eval_json = Path(f"results/hpo/trial_{trial_id}/eval/checkpoint_final.json")
    if eval_json.exists():
        exec_times = json.loads(eval_json.read_text())
        # Compute speedup from exec times vs baselines
    return 0.0  # Failed trial
```

### 5. `scripts/hpo/analyze.py`

Post-hoc analysis script that:
- Loads the Optuna study from `scripts/hpo/study.db`
- Prints best trial and its hyperparameters
- Generates parameter importance plot
- Generates parallel coordinate plot
- Generates slice plot (each param vs speedup)
- Saves best config as `scripts/hpo/best_config.json`
- Prints top-10 trials sorted by speedup

## Key Design Decisions

1. **Each trial has its own `results_dir`**: `results/hpo/trial_{N}/` — completely isolated, no cross-contamination.

2. **Auto-resume on timeout**: `train_trial.sh` checks if `results/hpo/trial_{N}/models/` has checkpoints and sets `RESUME_FROM` accordingly. If the Slurm job is killed (timeout), the wrapper detects non-zero exit, checks if training is complete, and resubmits with resume.

3. **Eval after training**: Only the final checkpoint is evaluated (`EVAL_LAST_ONLY=1`). This gives the best speedup metric for Optuna.

4. **Batch parallelism**: 5-10 training jobs run simultaneously. All must finish before eval starts. This balances Slurm resource usage with Optuna's need for results.

5. **Optuna TPE sampler**: Bayesian optimization learns from prior trials to suggest better hyperparameters. SQLite storage persists across sessions.

6. **7-day time limit**: Per jubail cluster max. With resume, even slow trials will complete eventually.

## Resource Usage

- **Per trial**: 12 CPUs, 32G RAM, up to 7 days
- **Batch of 5**: 60 CPUs, 160G RAM simultaneously
- **Total trials**: 50 (configurable)
- **Total compute**: ~50 node-days (parallelized across batches)

## Wrapper Script Summary

1. **Generate**: Create trial configs with Optuna-sampled hyperparameters
2. **Submit training**: `sbatch scripts/hpo/train_trial.sh` for each trial in batch
3. **Monitor training**: Poll `squeue` every 60s. If job dies:
   - Check if training completed (final model exists) → move to eval
   - If not completed → resubmit with `--resume results/hpo/trial_{N}`
4. **Submit eval**: `sbatch scripts/ho/eval_trial.sh` for each completed trial
5. **Monitor eval**: Wait for all eval jobs to finish
6. **Collect**: Read `average_speedup` from eval logs, tell Optuna
7. **Repeat**: Generate next batch of trials, loop until `TOTAL_TRIALS` reached

## Verification Steps

1. **Dry run**: Generate 1 trial config, verify it loads correctly with `python -m py_compile`
2. **Single trial**: Submit 1 training job (20K iter), verify it completes and saves models
3. **Resume test**: Kill a training job mid-way, resubmit with `--resume`, verify it continues from checkpoint
4. **Eval test**: Run eval on the trained model, verify `average_speedup` is written to logs
5. **Full batch**: Submit batch of 5, verify all complete and Optuna receives results
6. **Full study**: Run 50 trials, verify study.db has all results, generate analysis
