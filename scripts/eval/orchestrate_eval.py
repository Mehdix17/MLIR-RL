import json
import subprocess
import time
import os
import sys

QUEUE_FILE = "scripts/eval/eval_queue.json"
PROJECT_ROOT = "/scratch/mb10856/MLIR-RL"

def load_queue():
    with open(QUEUE_FILE, "r") as f:
        return json.load(f)

def save_queue(queue):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

def get_job_status(job_id):
    try:
        # Query sacct for the state of the job
        res = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = [line.strip() for line in res.stdout.split("\n") if line.strip()]
        if not lines:
            return "PENDING"
        # The first line is typically the main job state (e.g. RUNNING, COMPLETED, FAILED, TIMEOUT)
        state = lines[0]
        # Clean up state string if it has sub-components like RUNNING+
        if "+" in state:
            state = state.split("+")[0]
        return state
    except Exception as e:
        print(f"Error querying job {job_id}: {e}", file=sys.stderr)
        return "UNKNOWN"

def run_orchestrator():
    print("MLIR-RL Evaluation Orchestrator Started.")
    while True:
        try:
            queue = load_queue()
        except Exception as e:
            print(f"Error loading queue: {e}", file=sys.stderr)
            time.sleep(30)
            continue

        active_task = None
        active_index = -1
        
        for idx, task in enumerate(queue):
            if task["status"] in ("running", "pending"):
                active_task = task
                active_index = idx
                break

        if not active_task:
            print("All evaluation tasks completed successfully!")
            break

        if active_task["status"] == "running":
            job_id = active_task["job_id"]
            status = get_job_status(job_id)
            print(f"Task {active_index + 1}/{len(queue)} ({active_task['config']}, {active_task['start']}-{active_task['end']}) is RUNNING under Job ID {job_id}. Slurm State: {status}")
            
            if status == "COMPLETED":
                print(f"Job {job_id} completed successfully!")
                active_task["status"] = "completed"
                save_queue(queue)
            elif status in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
                print(f"Job {job_id} terminated with state: {status}. Stopping orchestrator.", file=sys.stderr)
                active_task["status"] = f"failed ({status})"
                save_queue(queue)
                sys.exit(1)
            else:
                # Still running or pending
                pass
                
        elif active_task["status"] == "pending":
            print(f"Submitting task {active_index + 1}/{len(queue)}: {active_task['config']} ({active_task['start']}-{active_task['end']})...")
            cmd = [
                "sbatch",
                "scripts/eval/eval_batch.sh",
                active_task["config"],
                str(active_task["start"]),
                str(active_task["end"]),
                str(active_task["step"])
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
                # Expecting output like: "Submitted batch job 16546546"
                output = res.stdout.strip()
                if "Submitted batch job" in output:
                    job_id = output.split()[-1]
                    active_task["status"] = "running"
                    active_task["job_id"] = job_id
                    print(f"Successfully submitted Job ID: {job_id}")
                    save_queue(queue)
                else:
                    raise RuntimeError(f"Unexpected sbatch output: {output}")
            except Exception as e:
                print(f"Failed to submit task {active_index + 1}: {e}", file=sys.stderr)
                time.sleep(60)
                continue

        # Wait 2 minutes before the next check
        time.sleep(120)

if __name__ == "__main__":
    run_orchestrator()
