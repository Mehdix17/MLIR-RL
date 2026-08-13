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


def load_experiments() -> dict:
    """Load active experiments from the repo-root experiments.json registry.

    The report-progress skill reads this file to know which runs to report.
    """
    try:
        with open(EXPERIMENTS_FILE) as f:
            data = json.load(f)
        return {e["name"]: e["results_dir"] for e in data.get("experiments", [])}
    except Exception:
        return {}


def update_experiment_states(agent_reports):
    """Write back derived states (running/done/failed/stopped/pending) to experiments.json."""
    try:
        with open(EXPERIMENTS_FILE) as f:
            data = json.load(f)
    except Exception:
        return
    states = {r["version"]: r["state"] for r in agent_reports}
    updated = False
    for e in data.get("experiments", []):
        state = states.get(e["name"])
        if state and state != e.get("state"):
            e["state"] = state
            updated = True
    if updated:
        tmp = EXPERIMENTS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, EXPERIMENTS_FILE)


DATASET_MAPPINGS = {
    "ops_and_blocks": load_experiments(),
    "new": {
        "v0": "results/new_dataset_results/v0_agent",
        "v4_6": "results/new_dataset_results/v4_6_agent",
        "v4_7": "results/new_dataset_results/v4_7_agent",
        "v4_8": "results/new_dataset_results/v4_8_agent",
        "v4_9_small": "results/new_dataset_results/v4_9_small_agent",
        "v4_9_large": "results/new_dataset_results/v4_9_large_agent",
    }
}

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
                }
        except Exception:
            continue
    return version_jobs

def get_agent_stats(version, reg_dir, active_jobs, train_log_data):
    """Aggregate stats for a specific agent version."""
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

    # 3. Slurm Jobs correlating to this agent
    slurm_job_id = None
    slurm_state = "N/A"
    slurm_node = "N/A"
    job_type = "N/A"
    
    # Check if this agent is currently running training or evaluating
    log_info = None
    version_logs = train_log_data.get(version, {})
    if version_logs:
        # Prefer the log whose job is still active; else the most advanced one
        log_info = next((v for jid, v in version_logs.items() if jid in active_jobs), None)
        if log_info is None:
            log_info = max(version_logs.values(), key=lambda v: v["iteration"])
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
                if jid in active_jobs and info.get("agent") == version:
                    slurm_job_id = jid
                    slurm_state = active_jobs[jid]["state"]
                    slurm_node = active_jobs[jid]["node"]
                    job_type = "Eval"
                    start, end, step = info["start"], info["end"], info["step"]
                    eval_ckpts_evaluating += len(range(start, end + 1, step))
        except Exception:
            pass

    # 4. Pending evaluations
    pending_count = 0
    if max_trained > 0:
        for ckpt in range(100, max_trained + 1, 100):
            if ckpt not in evaluated_ckpts:
                pending_count += 1
                
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
        "max_trained": max_trained,
        "evaluated_count": evaluated_count,
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
    parser.add_argument("-d", "--dataset", default="ops_and_blocks", 
                        choices=["ops_and_blocks", "new"], help="Dataset to report on")
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

    agents = DATASET_MAPPINGS.get(args.dataset, DATASET_MAPPINGS["ops_and_blocks"])
    
    # Process agent metrics in parallel
    agent_reports = []
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(get_agent_stats, version, reg_dir, active_jobs, train_log_data): version
            for version, reg_dir in agents.items()
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
    
    # Table 1: Active Slurm Jobs
    print("### 1. Active Slurm Jobs")
    slurm_rows = []
    # Add our current interactive job
    for jid, info in active_jobs.items():
        if "interact" in info["name"] or "salloc" in info["name"] or "srun" in info["name"]:
            slurm_rows.append(f"| `{jid}` | interactive | Interactive Shell | **{info['state']}** | `{info['node']}` |")
            
    for rep in agent_reports:
        if rep["slurm_job_id"]:
            jid = rep["slurm_job_id"]
            state = rep["slurm_state"]
            node = rep["slurm_node"]
            jtype = rep["job_type"]
            slurm_rows.append(f"| `{jid}` | `{rep['version']}` | {jtype} | **{state}** | `{node}` |")
            if rep.get("worker_count"):
                slurm_rows.append(
                    f"| `{rep['worker_count']}× dask` | `{rep['version']}` | Workers | **RUNNING** | "
                    f"`{', '.join(rep['worker_nodes'])}` |"
                )
            
    if slurm_rows:
        print("| Job ID | Agent Version | Job Type | State | Compute Node |")
        print("|---|---|---|---|---|")
        print("\n".join(slurm_rows))
    else:
        print("*No active Slurm jobs found for the current user.*")
    print()

    # Table 2: Training Progress
    print("### 2. Training Progress")
    print("| Version | Job ID | Iteration | Progress % | Status | Workers | Bench Failures | Latest Checkpoint |")
    print("|---|---|---|---|---|---|---|---|")
    for rep in agent_reports:
        log = rep["train_log"]
        if log:
            prog = (log['iteration']/log['total'])*100
            # Correlate status with Slurm state
            status = log['status']
            if rep["slurm_job_id"] and rep["job_type"] == "Train":
                status = f"Running ({rep['slurm_state']})"
            elif status == "running":
                status = "Stopped (Timeout/Cancelled)"
            
            ckpt = rep["max_trained"] if rep["max_trained"] > 0 else "N/A"
            workers = rep.get("worker_count") or "-"
            failures = log.get("bench_failures")
            failures_s = f"**{failures}**" if failures and failures > 0 else (str(failures) if failures is not None else "-")
            print(f"| `{rep['version']}` | `{log['job_id']}` | `{log['iteration']} / {log['total']}` | `{prog:.1f}%` | {status} | `{workers}` | {failures_s} | `{ckpt}` |")
        else:
            print(f"| `{rep['version']}` | `N/A` | `0 / 20000` | `0.0%` | Not Started | `-` | - | `N/A` |")
    print()

    # Table 3: Evaluation Progress
    print("### 3. Evaluation Progress")
    print("| Agent Version | Evaluated | Evaluating | Pending | Max Trained Checkpoint |")
    print("|---|---|---|---|---|")
    for rep in agent_reports:
        print(f"| `{rep['version']}` | {rep['evaluated_count']} checkpoints | {rep['evaluating_count']} | {rep['pending_count']} checkpoints | `{rep['max_trained']}` |")
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
