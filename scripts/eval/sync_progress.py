#!/usr/bin/env python3
import os
import re
import json
import glob
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKDOWN_PATH = os.path.join(PROJECT_ROOT, "docs/results/eval_progress.md")
ACTIVE_JOBS_JSON = os.path.join(PROJECT_ROOT, "scripts/eval/active_eval_jobs.json")

AGENT_MAPPING = {
    "paper_original": {
        "results_dir": "results/ops_and_blocks_results/paper_original_agent",
        "display": "paper_original",
        "config": "config/ops_and_blocks/eval/paper_original_eval.json"
    },
    "paper_transformer_small": {
        "results_dir": "results/ops_and_blocks_results/paper_transformer_small_agent",
        "display": "paper_transformer_small",
        "config": "config/ops_and_blocks/eval/paper_transformer_small_eval.json"
    },
    "paper_transformer_large": {
        "results_dir": "results/ops_and_blocks_results/paper_transformer_large_agent",
        "display": "paper_transformer_large",
        "config": "config/ops_and_blocks/eval/paper_transformer_large_eval.json"
    }
}

def get_slurm_jobs():
    try:
        out = subprocess.check_output(
            ["squeue", "--noheader", "--format=%i|%j|%T|%u|%R"],
            timeout=10, stderr=subprocess.DEVNULL
        ).decode().strip()
        jobs = {}
        for line in out.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 4 and parts[3] == "mb10856":
                jobs[parts[0]] = {
                    "name": parts[1],
                    "state": parts[2],
                    "node": parts[4] if len(parts) > 4 else "?"
                }
        return jobs
    except Exception:
        return {}

def load_active_jobs():
    if os.path.exists(ACTIVE_JOBS_JSON):
        try:
            with open(ACTIVE_JOBS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_active_jobs(jobs):
    os.makedirs(os.path.dirname(ACTIVE_JOBS_JSON), exist_ok=True)
    with open(ACTIVE_JOBS_JSON, "w") as f:
        json.dump(jobs, f, indent=2)

def main():
    active_jobs = load_active_jobs()
    slurm_jobs = get_slurm_jobs()
    
    # 1. Update active jobs in JSON cache
    still_active = {}
    for job_id, info in active_jobs.items():
        if job_id in slurm_jobs:
            # Job is still active
            info["state"] = slurm_jobs[job_id]["state"]
            info["node"] = slurm_jobs[job_id]["node"]
            still_active[job_id] = info
        else:
            # Job has finished/been cancelled.
            # We don't keep it in active_jobs anymore.
            pass
    save_active_jobs(still_active)
    
    # 2. Gather progress stats for each agent
    agent_stats = {}
    for agent_name, paths in AGENT_MAPPING.items():
        eval_dir = os.path.join(PROJECT_ROOT, paths["results_dir"], "eval")
        models_dir = os.path.join(PROJECT_ROOT, paths["results_dir"], "models")
        
        # Scanned evaluated checkpoints
        evaluated = []
        if os.path.exists(eval_dir):
            for f in os.listdir(eval_dir):
                m = re.match(r"checkpoint_(\d+)\.json", f)
                if m:
                    # check if the file is valid/non-empty
                    fpath = os.path.join(eval_dir, f)
                    if os.path.exists(fpath) and os.path.getsize(fpath) > 2:
                        evaluated.append(int(m.group(1)))
        evaluated.sort()
        
        # Scanned trained checkpoints
        trained = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                m = re.match(r"model_(\d+)\.pt", f)
                if m:
                    trained.append(int(m.group(1)))
        trained.sort()
        
        # Filter trained checkpoints to only include step of 50/100 (multiple of 50)
        trained = [ckpt for ckpt in trained if ckpt % 50 == 0]
        
        # Calculate what checkpoints are currently being evaluated
        evaluating = []
        evaluating_jobs = []
        for job_id, info in still_active.items():
            if info["agent"] == agent_name:
                start, end, step = info["start"], info["end"], info["step"]
                job_ckpts = list(range(start, end + 1, step))
                evaluating.extend(job_ckpts)
                evaluating_jobs.append((job_id, f"{start}-{end} (step {step})", info["state"]))
        
        # To evaluate (pending): trained but neither evaluated nor currently evaluating
        pending = []
        max_tr_val = max(trained) if trained else 100
        for ckpt in range(100, max_tr_val + 1, 100):
            if ckpt not in evaluated and ckpt not in evaluating:
                pending.append(ckpt)
                
        agent_stats[agent_name] = {
            "evaluated": evaluated,
            "evaluating": sorted(list(set(evaluating))),
            "evaluating_jobs": evaluating_jobs,
            "pending": pending,
            "max_trained": max(trained) if trained else None
        }

    # 3. Generate Markdown Content
    os.makedirs(os.path.dirname(MARKDOWN_PATH), exist_ok=True)
    with open(MARKDOWN_PATH, "w") as f:
        f.write("# Evaluation Progress Tracker\n\n")
        f.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Section: Active Jobs
        f.write("## Active Slurm Jobs\n")
        if still_active:
            f.write("| Job ID | Agent Version | Checkpoint Range | State | Node |\n")
            f.write("|---|---|---|---|---|\n")
            for job_id, info in sorted(still_active.items()):
                rng = f"{info['start']}–{info['end']} (step {info['step']})"
                f.write(f"| {job_id} | `{info['agent']}` | {rng} | **{info['state']}** | `{info['node']}` |\n")
        else:
            f.write("*No active evaluation jobs in Slurm.*\n")
        f.write("\n")
        
        # Section: Summary Table
        f.write("## Agent Progress Summary\n")
        f.write("| Agent Version | Evaluated | Evaluating | Pending | Max Trained |\n")
        f.write("|---|---|---|---|---|\n")
        for agent_name in sorted(AGENT_MAPPING.keys()):
            stats = agent_stats[agent_name]
            num_eval = len(stats["evaluated"])
            num_evaling = len(stats["evaluating"])
            num_pending = len(stats["pending"])
            max_tr = stats["max_trained"] or "N/A"
            f.write(f"| `{agent_name}` | {num_eval} | {num_evaling} | {num_pending} | {max_tr} |\n")
        f.write("\n")
        
        # Section: Detailed Status
        f.write("## Detailed Status per Agent\n\n")
        for agent_name in sorted(AGENT_MAPPING.keys()):
            stats = agent_stats[agent_name]
            f.write(f"### {agent_name}\n")
            f.write(f"- **Max trained checkpoint:** {stats['max_trained'] or 'N/A'}\n")
            
            # Helper function to format list of integers to compact ranges
            def format_ranges(lst):
                if not lst:
                    return "none"
                lst = sorted(lst)
                ranges = []
                start = lst[0]
                prev = lst[0]
                for val in lst[1:]:
                    if val == prev + 100:  # step of 100
                        prev = val
                    else:
                        if start == prev:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}–{prev}")
                        start = val
                        prev = val
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}–{prev}")
                return ", ".join(ranges)

            f.write(f"- **Evaluated ({len(stats['evaluated'])}):** {format_ranges(stats['evaluated'])}\n")
            
            evaling_str = "none"
            if stats["evaluating_jobs"]:
                evaling_str = ", ".join(f"Job {jid} [{rng}]: {state}" for jid, rng, state in stats["evaluating_jobs"])
            f.write(f"- **Evaluating:** {evaling_str}\n")
            f.write(f"- **Pending ({len(stats['pending'])}):** {format_ranges(stats['pending'])}\n\n")

    print(f"Successfully synced progress to docs/results/eval_progress.md")

if __name__ == "__main__":
    main()
