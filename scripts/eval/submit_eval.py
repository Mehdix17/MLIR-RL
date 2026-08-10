#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_JOBS_JSON = os.path.join(PROJECT_ROOT, "scripts/eval/active_eval_jobs.json")
EVAL_BATCH_SH = os.path.join(PROJECT_ROOT, "scripts/eval/eval_batch.sh")

AGENT_CONFIGS = {
    "paper_original": "config/ops_and_blocks/eval/paper_original_eval.json",
    "paper_transformer_small": "config/ops_and_blocks/eval/paper_transformer_small_eval.json",
    "paper_transformer_large": "config/ops_and_blocks/eval/paper_transformer_large_eval.json"
}

def resolve_agent_config(input_arg):
    # Check if config exists directly
    if os.path.exists(input_arg):
        return input_arg
        
    # Check if config exists inside config dir
    full_config_path = os.path.join(PROJECT_ROOT, input_arg)
    if os.path.exists(full_config_path):
        return full_config_path
        
    # Check if matches agent name or input folder
    for agent, cfg in AGENT_CONFIGS.items():
        if agent in input_arg:
            return os.path.join(PROJECT_ROOT, cfg)
            
    # Try directory matching
    # e.g. results/ops_and_blocks_results/paper_transformer_small_agent/models
    # base folder name contains agent name
    for agent, cfg in AGENT_CONFIGS.items():
        # Match parts like "paper_transformer_small" or "paper-transformer-small"
        pattern = agent.replace("_", ".*")
        if re.search(pattern, input_arg):
            return os.path.join(PROJECT_ROOT, cfg)
            
    raise ValueError(f"Could not resolve agent config from argument: {input_arg}")

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
    parser = argparse.ArgumentParser(description="Submit evaluation batch for a single agent version")
    parser.add_argument("agent", help="Agent config path, name, or input folder (e.g. config/...json, paper_original, or models directory)")
    parser.add_argument("start", type=int, help="Start checkpoint number")
    parser.add_argument("end", type=int, help="End checkpoint number")
    parser.add_argument("step", type=int, nargs="?", default=100, help="Step between checkpoints (default: 100)")
    parser.add_argument("--cpus", type=int, default=None, help="Number of CPUs to allocate (overrides script default)")
    parser.add_argument("--mem", default=None, help="Memory limit to allocate (e.g. 16G, overrides script default)")
    parser.add_argument("--time", default=None, help="Slurm time limit for the job (e.g. 08:00:00, overrides script default)")
    args = parser.parse_args()

    # 1. Resolve Config Path and Agent Name
    try:
        config_path = resolve_agent_config(args.agent)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    config_name = os.path.basename(config_path)
    # Find matching agent key
    agent_name = None
    for agent, cfg in AGENT_CONFIGS.items():
        if os.path.basename(cfg) == config_name:
            agent_name = agent
            break
    if not agent_name:
        agent_name = config_name.replace("_eval.json", "")

    print(f"Resolved Config Path: {config_path}")
    print(f"Agent Version: {agent_name}")
    print(f"Range: {args.start} to {args.end} with step {args.step}")

    # 2. Submit via sbatch
    cmd = ["sbatch"]
    if args.cpus:
        cmd.append(f"--cpus-per-task={args.cpus}")
    if args.mem:
        cmd.append(f"--mem={args.mem}")
    if args.time:
        cmd.append(f"--time={args.time}")
    cmd.extend([
        EVAL_BATCH_SH,
        config_path,
        str(args.start),
        str(args.end),
        str(args.step)
    ])
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
        output = res.stdout.strip()
        print(output)
        if "Submitted batch job" not in output:
            print("Error: unexpected output from sbatch", file=sys.stderr)
            sys.exit(1)
        job_id = output.split()[-1]
    except Exception as e:
        print(f"Failed to submit sbatch job: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Update active jobs cache
    active_jobs = load_active_jobs()
    active_jobs[job_id] = {
        "agent": agent_name,
        "start": args.start,
        "end": args.end,
        "step": args.step,
        "state": "PENDING",
        "node": "N/A"
    }
    save_active_jobs(active_jobs)
    print(f"Registered Job {job_id} in active jobs cache.")

    # 4. Trigger Sync script to regenerate markdown progress tracker
    sync_script = os.path.join(PROJECT_ROOT, "scripts/eval/sync_progress.py")
    subprocess.run([sys.executable, sync_script])

if __name__ == "__main__":
    main()
