#!/usr/bin/env python3
import os
import re
import sys
import glob
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS_FILE = os.path.join(PROJECT_ROOT, "experiments.json")


def load_experiments(dataset: str | None = None) -> list[dict]:
    """Load active experiments from the repo-root experiments.json registry.

    The report-progress skill reads this file to know which runs to report.
    """
    try:
        with open(EXPERIMENTS_FILE) as f:
            data = json.load(f)
        return [e for e in data.get("experiments", [])
                if not e.get("archived") and (dataset in (None, "all") or e.get("dataset") == dataset)]
    except Exception:
        return []


def update_experiment_states(agent_reports):
    """Write back derived states (running/done/failed/stopped/pending) to experiments.json."""
    try:
        with open(EXPERIMENTS_FILE) as f:
            data = json.load(f)
    except Exception:
        return
    states = {r["results_dir"]: r["state"] for r in agent_reports}
    updated = False
    for e in data.get("experiments", []):
        state = states.get(e["results_dir"])
        if state and state != e.get("state"):
            e["state"] = state
            updated = True
    if updated:
        tmp = EXPERIMENTS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, EXPERIMENTS_FILE)


NEW_DATASET_EXPERIMENTS = [
    {"name": name, "results_dir": results_dir, "dataset": "new", "config": name}
    for name, results_dir in {
        "v0": "results/new_dataset_results/v0_agent",
        "v4_6": "results/new_dataset_results/v4_6_agent",
        "v4_7": "results/new_dataset_results/v4_7_agent",
        "v4_8": "results/new_dataset_results/v4_8_agent",
        "v4_9_small": "results/new_dataset_results/v4_9_small_agent",
        "v4_9_large": "results/new_dataset_results/v4_9_large_agent",
    }.items()
]

def get_slurm_jobs():
    """Query squeue in the background."""
    try:
        out = subprocess.check_output(
            ["squeue", "-a", "--noheader", "--format=%i|%j|%T|%u|%R"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        jobs = {}
        user = os.environ.get("USER", "mb10856")
        for line in out.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 4 and parts[3] == user:
                jobs[parts[0]] = {
                    "name": parts[1],
                    "state": parts[2],
                    "node": parts[4] if len(parts) > 4 else "?"
                }
        return jobs
    except Exception:
        return {}

def get_lfs_quota():
    """Query lfs quota in the background."""
    try:
        user = os.environ.get("USER", "mb10856")
        out = subprocess.check_output(
            ["lfs", "quota", "-u", user, "/scratch"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        for line in out.split("\n"):
            parts = line.split()
            if parts and parts[0] == "/scratch":
                return {
                    "space_used": int(parts[1]),
                    "space_quota": int(parts[2]),
                    "space_limit": int(parts[3]),
                    "files_used": int(parts[5]),
                    "files_quota": int(parts[6]),
                    "files_limit": int(parts[7])
                }
    except Exception:
        pass
    return None

def scan_train_logs():
    """Scan training logs to identify active training job IDs and progress."""
    version_jobs = {}
    pattern = os.path.join(PROJECT_ROOT, "logs", "train_*.out")
    for fpath in glob.glob(pattern):
        try:
            # Parse config/version from head
            with open(fpath, errors="ignore") as f:
                head = f.read(512)
            m = re.search(r"Config:\s*.*/([\w_]+)\.json", head)
            if not m:
                m = re.search(r"Config:\s*.*/([\w_]+)", head)
            if not m:
                continue
            version = m.group(1)
            
            job_id_match = re.search(r"train_(\d+)\.out", fpath)
            if not job_id_match:
                continue
            job_id = job_id_match.group(1)

            # Get tail info
            file_size = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                f.seek(max(0, file_size - 4096))
                tail = f.read().decode("utf-8", errors="ignore")
                
            iters = re.findall(r"Main Loop (\d+)/(\d+)", tail)
            status = "running"
            if "TRAINING FAILED" in tail:
                status = "FAILED"
            elif "TRAINING FINISHED" in tail:
                status = "FINISHED"

            if iters:
                last_iter, total = int(iters[-1][0]), int(iters[-1][1])
            else:
                last_iter, total = 0, 20000

            distributed = "Distributed collection" in tail

            fail_m = re.findall(r"\((\d+) bench failures\)", tail)
            bench_failures = int(fail_m[-1]) if fail_m else None

            version_jobs.setdefault(version, {})
            prev = version_jobs[version].get(job_id)
            if prev is None or last_iter > prev["iteration"]:
                version_jobs[version][job_id] = {
                    "job_id": job_id,
                    "iteration": last_iter,
                    "total": total,
                    "status": status,
                    "distributed": distributed,
                    "bench_failures": bench_failures,
                    "mtime": os.path.getmtime(fpath),
                }
        except Exception:
            continue
    return version_jobs

def get_agent_stats(experiment, active_jobs, train_log_data):
    """Aggregate stats for a specific agent version."""
    version = experiment["name"]
    reg_dir = experiment["results_dir"]
    dataset = experiment["dataset"]
    config_key = os.path.splitext(os.path.basename(experiment.get("config", version)))[0]
    agent_dir = os.path.join(PROJECT_ROOT, reg_dir)
    models_dir = os.path.join(agent_dir, "models")
    eval_dir = os.path.join(agent_dir, "eval")
    
    # 1. Models Max Checkpoint (flat layout, with legacy run_N fallback)
    max_trained = 0
    models_dirs = [models_dir] + sorted(
        glob.glob(os.path.join(agent_dir, "*", "run_*", "models"))
        + glob.glob(os.path.join(agent_dir, "*", "models"))
    )
    for md in models_dirs:
        if not os.path.isdir(md):
            continue
        for f in os.listdir(md):
            m = re.match(r"model_(\d+)\.pt", f)
            if m:
                val = int(m.group(1))
                if val > max_trained:
                    max_trained = val

    # 2. Evaluation Completed Checkpoints
    evaluated_ckpts = []
    if os.path.isdir(eval_dir):
        for f in os.listdir(eval_dir):
            m = re.match(r"checkpoint_(\d+)\.json", f)
            if m:
                fpath = os.path.join(eval_dir, f)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 2:
                    evaluated_ckpts.append(int(m.group(1)))
    evaluated_ckpts.sort()
    evaluated_count = len(evaluated_ckpts)
    last_evaluated = evaluated_ckpts[-1] if evaluated_ckpts else None

    # 3. Slurm Jobs correlating to this agent
    slurm_job_id = None
    slurm_state = "N/A"
    slurm_node = "N/A"
    job_type = "N/A"
    
    # Legacy aliases for renamed experiments, newest first:
    # mlir_rl_v1_paper -> paper_original -> legacy_paper (wrong_split name never ran training)
    LEGACY_ALIASES = {
        "v5_mlir_rl_v1_paper": ["v5_paper_original", "v5_legacy_paper"],
        "v5_no_transformer_mlir_rl_v1_paper": ["v5_no_transformer_paper_original", "v5_no_transformer_legacy_paper"],
        "v5_paper_wrong_split": ["v5_legacy_paper"],
        "v5_no_transformer_paper_wrong_split": ["v5_no_transformer_legacy_paper"],
    }

    # Check if this agent is currently running training or evaluating
    log_info = None
    version_logs = dict(train_log_data.get(config_key, {}))
    for alias_key in LEGACY_ALIASES.get(config_key, []):
        for jid, v in train_log_data.get(alias_key, {}).items():
            version_logs.setdefault(jid, v)
    if version_logs:
        # Prefer the log whose job is still active; else the most recently written one.
        # (max iteration is wrong: stale logs from crashed runs can show a higher
        # loop counter than the current run, e.g. 20000 looped with 0 successful iters.)
        log_info = next((v for jid, v in version_logs.items() if jid in active_jobs), None)
        if log_info is None:
            log_info = max(version_logs.values(), key=lambda v: v.get("mtime", 0))
    if log_info:
        jid = log_info["job_id"]
        if jid in active_jobs:
            slurm_job_id = jid
            slurm_state = active_jobs[jid]["state"]
            slurm_node = active_jobs[jid]["node"]
            job_type = "Train"
            
    # Check active evaluation jobs
    eval_ckpts_evaluating = 0
    active_eval_jobs_json = os.path.join(PROJECT_ROOT, "scripts/eval/active_eval_jobs.json")
    if os.path.exists(active_eval_jobs_json):
        try:
            with open(active_eval_jobs_json) as f:
                active_evals = json.load(f)
            for jid, info in active_evals.items():
                if jid not in active_jobs or info.get("agent") not in {version, config_key}:
                    continue
                start, end, step = info["start"], info["end"], info["step"]
                eval_ckpts_evaluating += len(range(start, end + 1, step))
                if job_type != "Train":
                    slurm_job_id = jid
                    slurm_state = active_jobs[jid]["state"]
                    slurm_node = active_jobs[jid]["node"]
                    job_type = "Eval"
        except Exception:
            pass

    # 4. Pending evaluations (model-driven, grid-agnostic)
    # Pending = checkpoints that have a model file but no eval json.
    # This is the truth on disk, not a synthetic 100-grid expectation.
    # User standard is 100-grid but historic evals used 50-offset; we count any eval as covering its model.
    model_steps = []
    for md in models_dirs:
        if not os.path.isdir(md):
            continue
        for f in os.listdir(md):
            m = re.match(r"model_(\d+)\.pt", f)
            if m:
                model_steps.append(int(m.group(1)))
    model_steps = sorted(set(model_steps))
    pending_count = 0
    if model_steps:
        evaluated_set = set(evaluated_ckpts)
        pending_count = sum(1 for m in model_steps if m not in evaluated_set)
    elif log_info and log_info.get("total"):
        # Fallback when no model files yet (e.g. just started): use log total
        total = int(log_info["total"])
        expected = total // 100
        evaluated_within = sum(1 for c in evaluated_ckpts if c <= total)
        pending_count = max(0, expected - evaluated_within)
    elif max_trained > 0:
        expected = max_trained // 100
        evaluated_within = sum(1 for c in evaluated_ckpts if c <= max_trained)
        pending_count = max(0, expected - evaluated_within)
                
    # 5. Distributed dask workers (only for distributed runs whose driver is active)
    worker_count = 0
    worker_nodes: list[str] = []
    if job_type == "Train" and slurm_job_id and log_info and log_info.get("distributed"):
        workers = [i for i, j in active_jobs.items() if j["name"] == "dask"]
        worker_count = len(workers)
        worker_nodes = sorted({active_jobs[i]["node"] for i in workers})

    # 6. Derived state for the experiments registry
    if job_type == "Train" and slurm_job_id:
        state = "running"
    elif log_info and log_info["status"] == "FAILED":
        state = "failed"
    elif log_info and log_info["status"] == "FINISHED":
        state = "done"
    elif log_info:
        state = "stopped"
    else:
        state = "pending"

    return {
        "version": version,
        "dataset": dataset,
        "results_dir": reg_dir,
        "max_trained": max_trained,
        "evaluated_count": evaluated_count,
        "last_evaluated": last_evaluated,
        "evaluating_count": eval_ckpts_evaluating,
        "pending_count": pending_count,
        "slurm_job_id": slurm_job_id,
        "slurm_state": slurm_state,
        "slurm_node": slurm_node,
        "job_type": job_type,
        "worker_count": worker_count,
        "worker_nodes": worker_nodes,
        "state": state,
        "train_log": log_info
    }

def main():
    parser = argparse.ArgumentParser(description="Fast progress reporting tool")
    parser.add_argument("-d", "--dataset", default="all",
                        choices=["all", "ops_and_blocks", "mlir_rl_v1_paper", "new"], help="Dataset to report on")
    args = parser.parse_args()
    
    t0 = time.time()
    
    # Run API-heavy / IO-heavy Slurm and quota queries in parallel threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_jobs = executor.submit(get_slurm_jobs)
        f_quota = executor.submit(get_lfs_quota)
        f_logs = executor.submit(scan_train_logs)
        
        active_jobs = f_jobs.result()
        quota = f_quota.result()
        train_log_data = f_logs.result()

    agents = NEW_DATASET_EXPERIMENTS if args.dataset == "new" else load_experiments(args.dataset)
    
    # Process agent metrics in parallel
    agent_reports = []
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(get_agent_stats, experiment, active_jobs, train_log_data): experiment
            for experiment in agents
        }
        for fut in futures:
            agent_reports.append(fut.result())
            
    agent_reports.sort(key=lambda x: x["version"])

    # Persist derived states back into the experiments registry
    update_experiment_states(agent_reports)
    
    # ------------------ PRESENT REPORT ------------------
    print(f"\n## MLIR-RL Progress & Resource Report")
    print(f"**Generated in:** {time.time() - t0:.2f}s | **Dataset:** {args.dataset}")
    print()
    
    # Table 1: Active Slurm Jobs (interactive sessions hidden)
    print("### 1. Active Slurm Jobs")
    slurm_rows = []
    for rep in agent_reports:
        if rep["slurm_job_id"]:
            jid = rep["slurm_job_id"]
            state = rep["slurm_state"]
            node = rep["slurm_node"]
            jtype = rep["job_type"]
            slurm_rows.append(f"| `{jid}` | `{rep['dataset']}` | `{rep['version']}` | {jtype} | **{state}** | `{node}` |")
            if rep.get("worker_count"):
                slurm_rows.append(
                    f"| `{rep['worker_count']}× dask` | `{rep['dataset']}` | `{rep['version']}` | Workers | **RUNNING** | "
                    f"`{', '.join(rep['worker_nodes'])}` |"
                )
            
    if slurm_rows:
        print("| Job ID | Dataset | Agent Version | Job Type | State | Compute Node |")
        print("|---|---|---|---|---|---|")
        print("\n".join(slurm_rows))
    else:
        print("*No active Slurm jobs found for the current user.*")
    print()

    # Table 2: Training Progress (precise status; no Workers/Latest Ckpt — those live in Table 1 / Evaluation)
    print("### 2. Training Progress")
    print("| Dataset | Version | Iteration | Progress % | Status | Failures |")
    print("|---|---|---|---|---|---|")
    # Helper to resolve sacct state for non-active jobs (Timeout vs Cancelled vs Failed)
    def _sacct_status(jid: str) -> str:
        try:
            out = subprocess.check_output(
                ["sacct", "-j", str(jid), "--format=State", "--noheader", "--parsable2"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode().strip().splitlines()
            # sacct prints e.g. "COMPLETED" or "TIMEOUT" or "CANCELLED by ..." per line
            for line in out:
                s = line.strip().split()[0].upper()
                if "TIMEOUT" in s:
                    return "Timeout"
                if "CANCELLED" in s:
                    return "Cancelled"
                if "FAILED" in s:
                    return "Failed"
                if "COMPLETED" in s:
                    return "Finished"
            return ""
        except Exception:
            return ""

    for rep in agent_reports:
        log = rep["train_log"]
        if log:
            prog = (log['iteration']/log['total'])*100
            status = log['status']
            if rep["slurm_job_id"] and rep["job_type"] == "Train":
                status = f"Running ({rep['slurm_state']})"
            elif status == "FINISHED":
                status = "Finished"
            elif status == "FAILED":
                status = "Failed"
            elif status == "running":
                # No active job and not finished/failed → resolve via sacct
                sacct = _sacct_status(log["job_id"]) if log.get("job_id") else ""
                status = sacct if sacct in ("Timeout", "Cancelled", "Failed", "Finished") else "Timeout"
            failures = log.get("bench_failures")
            failures_s = f"**{failures}**" if failures and failures > 0 else (str(failures) if failures is not None else "-")
            print(f"| `{rep['dataset']}` | `{rep['version']}` | `{log['iteration']}` | `{prog:.1f}%` | {status} | {failures_s} |")
        else:
            print(f"| `{rep['dataset']}` | `{rep['version']}` | `0` | `0.0%` | Not Started | - |")
    print()

    # Table 3: Evaluation Progress (Max Trained removed; Last Evaluated is last column)
    print("### 3. Evaluation Progress")
    print("| Dataset | Agent Version | Evaluated | Evaluating | Pending | Last Evaluated |")
    print("|---|---|---|---|---|---|")
    for rep in agent_reports:
        le = f"`{rep['last_evaluated']}`" if rep['last_evaluated'] is not None else "`-`"
        print(f"| `{rep['dataset']}` | `{rep['version']}` | {rep['evaluated_count']} checkpoints | {rep['evaluating_count']} | {rep['pending_count']} checkpoints | {le} |")
    print()

    # Table 4: Lustre Quota
    print("### 4. Lustre Storage Quota (/scratch)")
    if quota:
        space_used_gb = quota["space_used"] / (1024 * 1024)
        space_quota_tb = quota["space_quota"] / (1024 * 1024 * 1024)
        space_limit_tb = quota["space_limit"] / (1024 * 1024 * 1024)
        space_pct = (quota["space_used"] / quota["space_quota"]) * 100
        
        files_pct_soft = (quota["files_used"] / quota["files_quota"]) * 100
        files_pct_hard = (quota["files_used"] / quota["files_limit"]) * 100
        
        print("| Metric | Used | Soft Limit (Quota) | Hard Limit | Utilized % |")
        print("|---|---|---|---|---|")
        print(f"| **Storage Space** | {space_used_gb:.1f} GB | {space_quota_tb:.2f} TB | {space_limit_tb:.2f} TB | {space_pct:.2f}% |")
        print(f"| **File Count (Inodes)** | {quota['files_used']:,} | {quota['files_quota']:,} | {quota['files_limit']:,} | **{files_pct_soft:.2f}%** of Soft / **{files_pct_hard:.2f}%** of Hard |")
        
        if quota["files_used"] > 425000:
            print("\n> [!WARNING]\n> File count exceeds **85%** of the soft limit quota. Cleanup suggested to prevent potential write failures.")
    else:
        print("*Lustre storage quota information currently unavailable.*")
    print()

if __name__ == "__main__":
    main()
