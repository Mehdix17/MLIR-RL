#!/usr/bin/env python3
"""HPO orchestrator for paper_transformer hyperparameter tuning.

Runs on login node. Uses Optuna TPE to sample transformer architecture
hyperparameters, submits Slurm training/eval jobs in batches, collects
results, and repeats.

Usage:
    python scripts/hpo/run_hpo.py --n-trials 50 --batch-size 5
    python scripts/hpo/run_hpo.py --n-trials 10 --batch-size 3 --resume-study
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_DIR = PROJECT_ROOT / "scripts" / "hpo"
TRIALS_DIR = HPO_DIR / "trials"
RESULTS_BASE = PROJECT_ROOT / "results" / "hpo"
BASE_CONFIG = HPO_DIR / "base_config.json"
STUDY_DB = HPO_DIR / "study.db"
POLL_INTERVAL = 60  # seconds between Slurm status checks

# Precomputed valid nhead values per d_model (d_model % nhead == 0)
VALID_NHEAD = {
    32: [2, 8],
    48: [2, 4, 8],
    64: [2, 4, 8],
    96: [2, 4, 8],
    128: [2, 4, 8],
    192: [2, 4, 8],
    256: [2, 4, 8],
}


def generate_trial_config(trial_id: int, params: dict) -> Path:
    """Generate a trial-specific config JSON from the base template."""
    base = json.loads(BASE_CONFIG.read_text())
    base.update(params)
    base["results_dir"] = f"results/hpo/trial_{trial_id}"
    base["tags"] = ["ops_and_blocks", "paper", "transformer", "hpo", f"trial_{trial_id}"]

    trial_dir = TRIALS_DIR / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(base, indent=2))
    return config_path


def submit_slurm(script: str, trial_id: int, job_name: str) -> str:
    """Submit a Slurm job and return the job ID."""
    result = subprocess.run(
        [
            "sbatch",
            f"--job-name={job_name}",
            f"--output=logs/hpo/{job_name}_%j.out",
            f"--error=logs/hpo/{job_name}_%j.err",
            str(PROJECT_ROOT / script),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "TRIAL_ID": str(trial_id)},
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")
    # Parse "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def get_job_state(job_id: str) -> str:
    """Return Slurm job state letter, or empty string if not found."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_job_exit_code(job_id: str) -> int:
    """Return the exit code of a completed Slurm job via sacct."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=ExitCode", "-n", "--parsable2"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().splitlines():
        code = line.strip().split("|")[0].replace(":", "")
        if code.isdigit():
            return int(code)
    return -1


def is_training_complete(trial_id: int) -> bool:
    """Check if training completed all 20K iterations."""
    # Checkpoints are saved to nested path by base FileLogger:
    #   <results_dir>/rl_autoschedular_paper_transformer_agent/run_0/models/
    models_dir = RESULTS_BASE / f"trial_{trial_id}" / "rl_autoschedular_paper_transformer_agent" / "run_0" / "models"
    if not models_dir.exists():
        return False
    model_files = list(models_dir.glob("model_*.pt"))
    if not model_files:
        return False
    max_step = max(int(f.stem.split("_")[1]) for f in model_files)
    # Models saved every 50 steps (0, 50, ..., 19950) + final at 20000
    return max_step >= 19950


def wait_for_job(job_id: str, trial_id: int, job_type: str = "train", max_resubmits: int = 10) -> bool:
    """Poll until job finishes. Auto-resume training on timeout. Returns True on success."""
    for resubmit in range(max_resubmits + 1):
        # Wait for job to leave the queue
        while True:
            state = get_job_state(job_id)
            if state == "":
                break
            print(f"  [{job_type}] trial={trial_id} job={job_id} state={state}", flush=True)
            time.sleep(POLL_INTERVAL)

        exit_code = get_job_exit_code(job_id)
        if exit_code == 0:
            return True

        print(f"  [{job_type}] trial={trial_id} job={job_id} failed (exit={exit_code})", flush=True)

        # For training: resubmit with --resume if not complete
        if job_type == "train" and not is_training_complete(trial_id):
            print(f"  [train] trial={trial_id} resubmitting (attempt {resubmit + 1}/{max_resubmits})", flush=True)
            job_id = submit_slurm("scripts/hpo/train_trial.sh", trial_id, f"hpo-train-t{trial_id}")
        else:
            return False

    return False


def read_trial_result(trial_id: int) -> float:
    """Read the eval average_speedup from the trial's logs."""
    # Method 1: Check eval log file (written by evaluate_benchmarks in ppo.py)
    avg_speedup_file = RESULTS_BASE / f"trial_{trial_id}" / "logs" / "eval" / "average_speedup"
    if avg_speedup_file.exists():
        values = []
        for line in avg_speedup_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
        if values:
            return values[-1]

    # Method 2: Check eval exec times JSON
    eval_json = RESULTS_BASE / f"trial_{trial_id}" / "eval" / "checkpoint_final.json"
    if eval_json.exists():
        try:
            exec_times = json.loads(eval_json.read_text())
            if exec_times:
                # Read baseline times
                baseline_path = PROJECT_ROOT / "results" / "ops_and_blocks_results" / "baselines" / "mlir" / "base_eval.json"
                if baseline_path.exists():
                    baselines = json.loads(baseline_path.read_text())
                    speedups = []
                    for bench, opt_time in exec_times.items():
                        if bench in baselines and opt_time and baselines[bench]:
                            speedups.append(baselines[bench] / opt_time)
                    if speedups:
                        return sum(speedups) / len(speedups)
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"  [warn] trial={trial_id} could not read result, returning 0.0", flush=True)
    return 0.0


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample transformer architecture hyperparameters with constraint handling."""
    d_model = trial.suggest_categorical("transformer_d_model", list(VALID_NHEAD.keys()))
    nhead = trial.suggest_categorical("transformer_nhead", VALID_NHEAD[d_model])
    num_layers = trial.suggest_int("transformer_num_layers", 1, 4)
    ffn_dim = trial.suggest_categorical("transformer_ffn_dim", [64, 128, 256, 512, 1024])
    dropout = trial.suggest_float("transformer_dropout", 0.0, 0.3)
    pooling = trial.suggest_categorical("transformer_pooling", ["cls", "mean"])

    return {
        "transformer_d_model": d_model,
        "transformer_nhead": nhead,
        "transformer_num_layers": num_layers,
        "transformer_ffn_dim": ffn_dim,
        "transformer_dropout": round(dropout, 3),
        "transformer_pooling": pooling,
    }


def run_batch(study: optuna.Study, trial_ids: list[int]) -> None:
    """Run a batch of trials: submit train, wait, submit eval, wait, collect."""
    print(f"\n{'='*60}", flush=True)
    print(f"BATCH: trials {trial_ids[0]}-{trial_ids[-1]}", flush=True)
    print(f"{'='*60}", flush=True)

    # Create trials and generate configs
    pending = []
    for tid in trial_ids:
        trial = study.ask()
        params = sample_hyperparameters(trial)
        generate_trial_config(tid, params)
        print(f"  trial {tid}: {params}", flush=True)
        pending.append((trial, tid))

    # Submit all training jobs
    print(f"\n--- Submitting {len(pending)} training jobs ---", flush=True)
    train_jobs = []
    for trial, tid in pending:
        job_id = submit_slurm("scripts/hpo/train_trial.sh", tid, f"hpo-train-t{tid}")
        print(f"  trial {tid}: train job {job_id}", flush=True)
        train_jobs.append((trial, tid, job_id))

    # Wait for all training jobs
    print(f"\n--- Waiting for training jobs ---", flush=True)
    for trial, tid, job_id in train_jobs:
        success = wait_for_job(job_id, tid, job_type="train")
        status = "DONE" if success else "FAILED"
        print(f"  trial {tid}: training {status}", flush=True)

    # Submit all eval jobs
    print(f"\n--- Submitting {len(pending)} eval jobs ---", flush=True)
    eval_jobs = []
    for trial, tid, _ in train_jobs:
        job_id = submit_slurm("scripts/hpo/eval_trial.sh", tid, f"hpo-eval-t{tid}")
        print(f"  trial {tid}: eval job {job_id}", flush=True)
        eval_jobs.append((trial, tid, job_id))

    # Wait for all eval jobs
    print(f"\n--- Waiting for eval jobs ---", flush=True)
    for trial, tid, job_id in eval_jobs:
        success = wait_for_job(job_id, tid, job_type="eval")
        status = "DONE" if success else "FAILED"
        print(f"  trial {tid}: eval {status}", flush=True)

    # Collect results and tell Optuna
    print(f"\n--- Collecting results ---", flush=True)
    for trial, tid, _ in eval_jobs:
        speedup = read_trial_result(tid)
        study.tell(trial, speedup)
        print(f"  trial {tid}: speedup = {speedup:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="HPO orchestrator for paper_transformer")
    parser.add_argument("--n-trials", type=int, default=50, help="Total number of Optuna trials")
    parser.add_argument("--batch-size", type=int, default=5, help="Parallel trials per batch")
    parser.add_argument("--resume-study", action="store_true", help="Resume existing Optuna study")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only, no Slurm submission")
    args = parser.parse_args()

    # Create logs directory
    (PROJECT_ROOT / "logs" / "hpo").mkdir(parents=True, exist_ok=True)

    # Create or load Optuna study
    study = optuna.create_study(
        study_name="paper_transformer_hpo",
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    if args.resume_study:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Resuming study: {completed} trials already completed")
    else:
        completed = 0

    if args.dry_run:
        print("DRY RUN: generating configs without Slurm submission")
        for i in range(min(args.batch_size, args.n_trials)):
            trial = study.ask()
            params = sample_hyperparameters(trial)
            config_path = generate_trial_config(completed + i, params)
            print(f"  trial {completed + i}: {params}")
            print(f"    config: {config_path}")
        return

    # Main loop
    while completed < args.n_trials:
        batch_end = min(completed + args.batch_size, args.n_trials)
        trial_ids = list(range(completed, batch_end))

        run_batch(study, trial_ids)

        completed = batch_end
        print(f"\nProgress: {completed}/{args.n_trials} trials completed", flush=True)

        # Print current best
        if study.trials:
            best = study.best_trial
            print(f"Best so far: trial {best.number} speedup={best.value:.4f}", flush=True)
            print(f"  params: {best.params}", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"STUDY COMPLETE: {args.n_trials} trials", flush=True)
    print(f"{'='*60}", flush=True)
    best = study.best_trial
    print(f"Best trial: {best.number}", flush=True)
    print(f"Best speedup: {best.value:.4f}", flush=True)
    print(f"Best params: {json.dumps(best.params, indent=2)}", flush=True)

    # Save best config
    best_config = generate_trial_config(-1, best.params)
    print(f"Best config saved to: {best_config}", flush=True)


if __name__ == "__main__":
    main()
