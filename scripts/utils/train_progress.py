#!/usr/bin/env python3
"""Report training progression across versions.

Usage:
  python scripts/utils/train_progress.py                    # auto-detect all
  python scripts/utils/train_progress.py -v v4_6 v4_7       # specific versions
  python scripts/utils/train_progress.py -w 300             # watch mode, refresh every 300s
"""

import argparse
import json
import os
import re
import glob
import sys
import time
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERSION_REGISTRY = {
    "v4_6": {
        "results_dir": "results/new_dataset_results/v4_6_agent",
    },
    "v4_7": {
        "results_dir": "results/new_dataset_results/v4_7_agent",
    },
    "v4_8": {
        "results_dir": "results/new_dataset_results/v4_8_agent",
    },
    "v0_v2": {
        "results_dir": "results/new_dataset_results/v0_agent_v2",
    },
}


def _parse_config_version(config_path: str) -> str:
    name = os.path.basename(config_path).replace(".json", "")
    return name.replace("_", "_") if "_" in name else name


def auto_detect_versions():
    """Find versions by scanning train log files."""
    version_jobs = {}
    pattern = os.path.join(PROJECT_ROOT, "logs", "train_*.out")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath) as f:
                first_lines = "".join(f.readline() for _ in range(3))
            m = re.search(r"Config:\s*(.+)", first_lines)
            if not m:
                continue
            version = _parse_config_version(m.group(1))
            job_id = re.search(r"train_(\d+)\.out", fpath).group(1)
            if version not in version_jobs or int(job_id) > int(version_jobs[version]):
                version_jobs[version] = job_id
        except Exception:
            continue
    return version_jobs


def get_log_data(version: str, job_id: str):
    """Parse training log for a given job."""
    log_path = os.path.join(PROJECT_ROOT, "logs", f"train_{job_id}.out")
    if not os.path.exists(log_path):
        return None

    with open(log_path, errors="ignore") as f:
        text = f.read()

    iters = re.findall(r"Main Loop (\d+)/(\d+)", text)
    if not iters:
        return {"version": version, "job_id": job_id, "iteration": "?", "total": "?", "status": "loading"}

    last_iter, total = iters[-1]

    execs = re.findall(r"exec: (\d+):(\d+):(\d+)\.(\d+)", text)
    last_execs = []
    for h, m, s, ms in execs[-5:]:
        secs = int(h) * 3600 + int(m) * 60 + int(s)
        last_execs.append(secs)

    status_lines = [l for l in text.split("\n") if "TRAINING FAILED" in l or "TRAINING FINISHED" in l]
    status = "running"
    if "TRAINING FAILED" in "".join(status_lines):
        status = "FAILED"
    elif "TRAINING FINISHED" in "".join(status_lines):
        status = "FINISHED"

    return {
        "version": version,
        "job_id": job_id,
        "iteration": int(last_iter),
        "total": int(total),
        "exec_secs": last_execs,
        "status": status,
    }


def find_latest_checkpoint_file(version: str):
    """Find the latest training checkpoint JSON file."""
    reg = VERSION_REGISTRY.get(version)
    if not reg:
        return None

    results = os.path.join(PROJECT_ROOT, reg["results_dir"])
    train_dir = os.path.join(results, "run_0", "train")
    if not os.path.isdir(train_dir):
        return None

    ckpts = sorted(
        [f for f in os.listdir(train_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    return os.path.join(train_dir, ckpts[-1]) if ckpts else None


def get_checkpoint_speedups(ckpt_file: str):
    """Extract speedup data from a checkpoint JSON file."""
    if not ckpt_file or not os.path.isfile(ckpt_file):
        return []

    speedups = []
    try:
        with open(ckpt_file) as f:
            data = json.load(f)
        for bench_name, bench_data in data.items():
            if isinstance(bench_data, dict) and "speedup" in bench_data:
                spd = bench_data["speedup"]
                if spd is not None and spd > 0:
                    speedups.append(spd)
    except Exception:
        pass

    return speedups


def get_checkpoint_history(version: str, n=5):
    """Get speedup trend from the last N training checkpoints."""
    reg = VERSION_REGISTRY.get(version)
    if not reg:
        return []

    results = os.path.join(PROJECT_ROOT, reg["results_dir"])
    train_dir = os.path.join(results, "run_0", "train")
    if not os.path.isdir(train_dir):
        return []

    ckpts = sorted(
        [f for f in os.listdir(train_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    history = []
    for ckpt in ckpts[-n:]:
        iter_num = ckpt.split("_")[1].split(".")[0]
        spds = get_checkpoint_speedups(os.path.join(train_dir, ckpt))
        history.append((iter_num, spds))
    return history


def format_table(data: list[dict]):
    """Print a formatted comparison table."""
    header = ["Version", "Job", "Iter", "Status", "Exec(avg)", "MarkSpd", "Best", "Worst", "Spd Trend"]
    rows = [header]

    for d in data:
        if d.get("status") == "loading":
            spds = d.get("marker_speedups", [])
            if spds:
                avg = sum(spds) / len(spds)
                spd_str = f"{avg:.2f}x"
                best_str = f"{max(spds):.2f}x"
                worst_str = f"{min(spds):.4f}x"
            else:
                spd_str = best_str = worst_str = "-"
            rows.append([d["version"], d["job_id"], "-", "loading", "-", spd_str, best_str, worst_str, "-"])
            continue

        exec_str = f"{sum(d['exec_secs'])//len(d['exec_secs'])}s" if d.get("exec_secs") else "-"

        spds = d.get("marker_speedups", [])
        if spds:
            avg = sum(spds) / len(spds)
            spd_str = f"{avg:.2f}x"
            best_str = f"{max(spds):.2f}x"
            worst_str = f"{min(spds):.4f}x"
        else:
            spd_str = best_str = worst_str = "-"

        # Speed trend from history
        history = d.get("marker_history", [])
        trend_str = ""
        if history:
            for it, sps in history:
                if sps:
                    avg_sp = sum(sps) / len(sps)
                    trend_str += f"@{it}:{avg_sp:.2f} "
            trend_str = trend_str.strip()

        rows.append([
            d["version"], d["job_id"], str(d["iteration"]),
            d.get("status", "?"), exec_str, spd_str, best_str, worst_str, trend_str or "-",
        ])

    # Compute column widths
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*header))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

    for row in rows[1:]:
        print(fmt.format(*[str(c) for c in row]))

    # Legend
    print("\nExec(avg) = last 5 collection exec times average  |  MarkSpd = latest marker avg speedup")
    print("Spd Trend = avg speedup at last N marker checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Report training progression")
    parser.add_argument("-v", "--versions", nargs="*", help="Version names (e.g. v4_6 v4_7)")
    parser.add_argument("-w", "--watch", type=int, nargs="?", const=300, metavar="SECS",
                        help="Auto-refresh every SECS seconds (default: 300)")
    parser.add_argument("-j", "--jobs", nargs="*", help="Explicit job_id:version pairs (e.g. 16157399:v4_6)")
    args = parser.parse_args()

    if args.jobs:
        version_jobs = {}
        for pair in args.jobs:
            job_id, version = pair.split(":")
            version_jobs[version] = job_id
    else:
        version_jobs = auto_detect_versions()

    parser.add_argument("-a", "--all", action="store_true",
                        help="Show all detected versions (not just registry)")

    args = parser.parse_args()

    if args.jobs:
        version_jobs = {}
        for pair in args.jobs:
            job_id, version = pair.split(":")
            version_jobs[version] = job_id
    else:
        version_jobs = auto_detect_versions()

    if args.versions:
        versions = args.versions
    elif args.all:
        versions = sorted(version_jobs.keys())
    else:
        versions = sorted(VERSION_REGISTRY.keys())

    versions = [v for v in versions if v in version_jobs]

    def print_report(heading=True):
        if heading:
            print(f"\n=== Training Progress @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        data = []
        for version in versions:
            job_id = version_jobs.get(version)
            if not job_id:
                print(f"  {version}: no job found")
                continue

            log_data = get_log_data(version, job_id)
            if not log_data:
                print(f"  {version}: log not found")
                continue

            ckpt_file = find_latest_checkpoint_file(version)
            speedups = get_checkpoint_speedups(ckpt_file)
            history = get_checkpoint_history(version, n=5)
            log_data["marker_speedups"] = speedups
            log_data["marker_history"] = history
            log_data["marker_file"] = ckpt_file
            data.append(log_data)

        if not data:
            print("No training data found.")
            return

        format_table(data)

    import sys as _sys
    if args.watch:
        try:
            while True:
                version_jobs = auto_detect_versions()
                print(f"\n=== Training Progress @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                _sys.stdout.flush()
                print_report(heading=False)
                _sys.stdout.flush()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_report()


if __name__ == "__main__":
    main()
