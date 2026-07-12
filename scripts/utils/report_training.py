#!/usr/bin/env python3
"""Report training progression across versions.

Usage:
  python scripts/utils/report_training.py                    # auto-detect all
  python scripts/utils/report_training.py -v v4_6 v4_7       # specific versions
  python scripts/utils/report_training.py -w 300             # watch mode, refresh every 300s
  python scripts/utils/report_training.py --json out.json    # export results as JSON
"""

import argparse
import json
import os
import re
import glob
import math
import sys
import time
import subprocess
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIRS = {
    "new": "results/new_dataset_results",
    "single_ops": "results/single_ops_dataset_results",
    "ops_and_blocks": "results/ops_and_blocks_results",
}

# How many bytes to read from log file tail/head (avoids full 7-8 MB Lustre reads)
_TAIL_BYTES = 65536  # 64 KB — enough for many recent iteration lines
_HEAD_BYTES = 2048   # 2 KB — enough for the resume/config header

VERSION_REGISTRY = {
    "v0": {"results_dir": "results/new_dataset_results/v0_agent"},
    "v4_5": {"results_dir": "results/new_dataset_results/v4_5_agent"},
    "v4_6": {"results_dir": "results/new_dataset_results/v4_6_agent"},
    "v4_7": {"results_dir": "results/new_dataset_results/v4_7_agent"},
    "v4_8": {"results_dir": "results/new_dataset_results/v4_8_agent"},
    "v4_9_small": {"results_dir": "results/single_ops_dataset_results/v4_9_small_agent"},
    "v4_9_large": {"results_dir": "results/single_ops_dataset_results/v4_9_large_agent"},
    "no_hw": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_hw_agent"},
    "no_shaped_reward": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent"},
    "no_transformer": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_transformer_agent"},
    "paper_original": {"results_dir": "results/ops_and_blocks_results/paper_original_agent"},
    "paper_transformer_small": {"results_dir": "results/ops_and_blocks_results/paper_transformer_small_agent"},
    "paper_transformer_large": {"results_dir": "results/ops_and_blocks_results/paper_transformer_large_agent"},
}


def build_registry(dataset: str):
    """Build version registry for a specific dataset."""
    base = DATASET_DIRS.get(dataset, DATASET_DIRS["new"])
    registry = {
        "v0": {"results_dir": f"{base}/v0_agent"},
        "v4_6": {"results_dir": f"{base}/v4_6_agent"},
        "v4_7": {"results_dir": f"{base}/v4_7_agent"},
        "v4_8": {"results_dir": f"{base}/v4_8_agent"},
        "v4_9_small": {"results_dir": f"{base}/v4_9_small_agent"},
        "v4_9_large": {"results_dir": f"{base}/v4_9_large_agent"},
    }
    if dataset == "new":
        registry.update({
            "no_hw": {"results_dir": f"{base}/ablation_study/v45_no_hw_agent"},
            "no_shaped_reward": {"results_dir": f"{base}/ablation_study/v45_no_shaped_reward_agent"},
            "no_transformer": {"results_dir": f"{base}/ablation_study/v45_no_transformer_agent"},
        })
    elif dataset == "ops_and_blocks":
        registry.update({
            "paper_original": {"results_dir": f"{base}/paper_original_agent"},
            "paper_transformer_small": {"results_dir": f"{base}/paper_transformer_small_agent"},
            "paper_transformer_large": {"results_dir": f"{base}/paper_transformer_large_agent"},
        })
    return registry


def _parse_config_version(config_path: str) -> str:
    name = os.path.basename(config_path).replace(".json", "")
    return name


def auto_detect_versions():
    """Find versions by scanning train log files.
    Among jobs with actual iteration data, prefers the one with the highest
    last-seen iteration number. This ensures a cancelled/short job with a
    higher Slurm ID doesn't shadow a long-running job with more real progress.
    """
    # version -> (job_id, last_iter) for jobs that have iteration data
    version_jobs_data: dict[str, tuple[str, int]] = {}
    # version -> job_id for jobs with no iteration data (fallback)
    version_jobs_nodata: dict[str, str] = {}
    pattern = os.path.join(PROJECT_ROOT, "logs", "train_*.out")
    for fpath in glob.glob(pattern):
        try:
            with open(fpath, errors="ignore") as f:
                first_lines = f.read(512)
            m = re.search(r"Config:\s*.*/([\w_]+)\.json", first_lines)
            if not m:
                m = re.search(r"Config:\s*.*/([\w_]+)",
                              first_lines.replace("\\", "/"))
            if not m:
                continue
            version = _parse_config_version(m.group(1))
            job_id_match = re.search(r"train_(\d+)\.out", fpath)
            if not job_id_match:
                continue
            job_id = job_id_match.group(1)

            # Quick tail scan for iteration data
            file_size = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                f.seek(max(0, file_size - _TAIL_BYTES))
                tail = f.read().decode("utf-8", errors="ignore")
            iters = re.findall(r"Main Loop (\d+)/\d+", tail)

            if iters:
                last_iter = int(iters[-1])
                prev = version_jobs_data.get(version)
                if prev is None or last_iter > prev[1]:
                    version_jobs_data[version] = (job_id, last_iter)
            else:
                prev_id = version_jobs_nodata.get(version)
                if prev_id is None or int(job_id) > int(prev_id):
                    version_jobs_nodata[version] = job_id
        except Exception:
            continue

    # Build final map: prefer real-data jobs; fall back to no-data for new versions
    version_jobs = {v: info[0] for v, info in version_jobs_data.items()}
    for v, jid in version_jobs_nodata.items():
        if v not in version_jobs:
            version_jobs[v] = jid
    return version_jobs



def get_log_data(version: str, job_id: str):
    """Parse training log for a given job.

    Reads only the tail (_TAIL_BYTES) for iteration progress/status, and the
    head (_HEAD_BYTES) for the resume-from line — avoids loading the full 7-8 MB
    Lustre file into memory on every report invocation.
    """
    log_path = os.path.join(PROJECT_ROOT, "logs", f"train_{job_id}.out")
    if not os.path.exists(log_path):
        return None

    file_size = os.path.getsize(log_path)
    with open(log_path, "rb") as f:
        # Head: config / resume line (written once near startup)
        head = f.read(_HEAD_BYTES).decode("utf-8", errors="ignore")
        # Tail: most-recent iteration lines and final status
        f.seek(max(0, file_size - _TAIL_BYTES))
        tail = f.read().decode("utf-8", errors="ignore")

    iters = re.findall(r"Main Loop (\d+)/(\d+)", tail)
    if not iters:
        return {"version": version, "job_id": job_id, "iteration": "?", "total": "?", "status": "loading"}

    last_iter, total = iters[-1]

    execs = re.findall(r"exec: (\d+):(\d+):(\d+)\.(\d+)", tail)
    last_execs = []
    for h, m, s, ms in execs[-5:]:
        secs = int(h) * 3600 + int(m) * 60 + int(s)
        last_execs.append(secs)

    status = "running"
    if "TRAINING FAILED" in tail:
        status = "FAILED"
    elif "TRAINING FINISHED" in tail:
        status = "FINISHED"

    resume_from = None
    m = re.search(r"Resumed model.*from.*run_(\d+)/models/model_(\d+)\.pt", head)
    if m:
        resume_from = f"run_{m.group(1)}/model_{m.group(2)}"

    return {
        "version": version,
        "job_id": job_id,
        "iteration": int(last_iter),
        "total": int(total),
        "exec_secs": last_execs,
        "status": status,
        "resume_from": resume_from,
    }


def get_train_stats(train_dir: str):
    """Extract speedup data and failure count from train/checkpoint_N.json files.

    Returns dict with: speedups (list), failed (int), total (int)
    """
    empty = {"speedups": [], "failed": 0, "total": 0}
    if not train_dir or not os.path.isdir(train_dir):
        return empty

    results_file = os.path.join(train_dir, "results.json")
    ckpt_file = results_file if os.path.isfile(results_file) else None

    if not ckpt_file:
        try:
            ckpts = sorted(
                [f for f in os.listdir(train_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
                key=lambda x: int(x.split("_")[1].split(".")[0])
            )
        except (OSError, FileNotFoundError):
            return empty
        if not ckpts:
            return empty
        ckpt_file = os.path.join(train_dir, ckpts[-1])

    speedups = []
    failed = 0
    total = 0
    try:
        with open(ckpt_file) as f:
            data = json.load(f)
        for bench_data in data.values():
            if isinstance(bench_data, dict) and "speedup" in bench_data:
                total += 1
                spd = bench_data["speedup"]
                if spd is not None and spd > 0:
                    speedups.append(spd)
                elif spd is not None and spd <= 0:
                    failed += 1
    except Exception:
        pass

    return {"speedups": speedups, "failed": failed, "total": total}


def get_train_checkpoint_file(train_dir: str):
    """Get the path+name of the latest checkpoint."""
    if not train_dir or not os.path.isdir(train_dir):
        return None, None
    results_file = os.path.join(train_dir, "results.json")
    if os.path.isfile(results_file):
        return results_file, "results.json"

    try:
        ckpts = sorted(
            [f for f in os.listdir(train_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
    except (OSError, FileNotFoundError):
        return None, None
    if ckpts:
        return os.path.join(train_dir, ckpts[-1]), ckpts[-1]
    return None, None


def get_train_trend(train_dir: str, n=5):
    """Get speedup trend from the last N training checkpoints."""
    if not train_dir or not os.path.isdir(train_dir):
        return []

    try:
        ckpts = sorted(
            [f for f in os.listdir(train_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
    except (OSError, FileNotFoundError):
        return []
    trend = []
    for ckpt in ckpts[-n:]:
        iter_num = ckpt.split("_")[1].split(".")[0]
        try:
            with open(os.path.join(train_dir, ckpt)) as f:
                data = json.load(f)
            spds = [v["speedup"] for v in data.values()
                    if isinstance(v, dict) and v.get("speedup", 0) and v["speedup"] > 0]
            trend.append((iter_num, spds))
        except Exception:
            pass
    return trend


def get_latest_model_index(models_dir: str) -> int | None:
    """Get the highest model index from run_N/models/."""
    if not models_dir or not os.path.isdir(models_dir):
        return None
    indices = []
    for f in os.listdir(models_dir):
        m = re.match(r"model_(\d+)\.pt", f)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) if indices else None


def get_running_jobs():
    """Get running Slurm jobs info."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "--format=%i|%j|%T|%M|%l"],
            timeout=10, stderr=subprocess.DEVNULL
        ).decode().strip()
        jobs = {}
        for line in out.split("\n"):
            parts = line.split("|")
            if len(parts) >= 3:
                jobs[parts[0]] = {"name": parts[1], "state": parts[2],
                                  "time": parts[3] if len(parts) > 3 else "?",
                                  "limit": parts[4] if len(parts) > 4 else "?"}
        return jobs
    except subprocess.TimeoutExpired:
        print("Warning: squeue timed out — Slurm status unavailable", file=sys.stderr)
        return {}
    except Exception:
        return {}


def format_table(data: list[dict]):
    """Print a formatted comparison table."""
    header = ["Version", "Job", "Iter (%Prog)", "Status", "Model#", "AvgSpd(geo)", "Failed", "Best/Worst", "Trend"]
    rows = [header]

    for d in data:
        if d.get("status") == "loading":
            rows.append([d["version"], d["job_id"], "-", "loading", "-", "-", "-", "-", "-"])
            continue

        total = d.get("total", 0) or 1
        pct = d["iteration"] / total * 100 if isinstance(d.get("iteration"), (int, float)) else 0
        iter_str = f"{d['iteration']}/{total} ({pct:.1f}%)"
        status = d.get("status", "?")

        model_str = str(d.get("model_index", "?")) if d.get("model_index") is not None else "-"

        spds = d.get("train_speedups", [])
        failed = d.get("train_failed", 0)
        total_benchs = d.get("train_total", 0)
        if spds:
            try:
                geo = math.exp(sum(math.log(max(s, 1e-12)) for s in spds) / len(spds))
            except (ValueError, OverflowError):
                geo = 0.0
            spd_str = f"{geo:.2f}x (n={len(spds)})"
            failed_str = f"{failed}" + (f" ({failed/total_benchs*100:.1f}%)" if total_benchs else "")
            best_str = f"{max(spds):.2f}x/{min(spds):.4f}x"
        else:
            spd_str = "-"
            failed_str = f"{failed}" if failed else "-"
            best_str = "-"

        trend = d.get("train_trend", [])
        trend_str = ""
        for it, sps in trend:
            if sps:
                try:
                    geo_sp = math.exp(sum(math.log(max(s, 1e-12)) for s in sps) / len(sps))
                except (ValueError, OverflowError):
                    geo_sp = 0.0
                trend_str += f"@{it}:{geo_sp:.2f} "
        trend_str = trend_str.strip() or "-"

        if d.get("ckpt_file") == "results.json":
            trend_str += " [live]"

        rows.append([d["version"], d["job_id"], iter_str, status, model_str,
                     spd_str, failed_str, best_str, trend_str])

    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*header))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows[1:]:
        print(fmt.format(*[str(c) for c in row]))

    print("\nAvgSpd = geometric mean speedup from train/results.json")
    print("Failed = benchmark count where execution failed (speedup=0.0)")
    print("Best/Worst = best and worst individual benchmark speedups")
    print("Trend = avg speedup at last 5 snapshot checkpoints")
    print("Model# = latest model checkpoint index in run_N/models/")


def main():
    parser = argparse.ArgumentParser(description="Report training progression")
    parser.add_argument("-v", "--versions", nargs="*", help="Version names (e.g. v4_6 v4_7)")
    parser.add_argument("-d", "--dataset", choices=["new", "single_ops", "ops_and_blocks"], default="new",
                        help="Dataset to report (default: new)")
    parser.add_argument("-w", "--watch", type=int, nargs="?", const=300, metavar="SECS",
                        help="Auto-refresh every SECS seconds (default: 300)")
    parser.add_argument("-j", "--jobs", nargs="*",
                        help="Explicit job_id:version pairs (e.g. 16157399:v4_6)")
    parser.add_argument("-n", "--trend-n", type=int, default=5,
                        help="Number of checkpoints for trend (default: 5)")
    parser.add_argument("--json", metavar="FILE",
                        help="Export results as JSON to file")
    args = parser.parse_args()

    version_registry = build_registry(args.dataset)

    if args.jobs:
        version_jobs = dict(pair.split(":") for pair in args.jobs)
    else:
        version_jobs = auto_detect_versions()

    if args.versions:
        versions = args.versions
    else:
        versions = sorted(set(list(version_registry.keys()) + list(version_jobs.keys())))

    versions = [v for v in versions if v in version_jobs]
    if not versions:
        print("No training jobs found.")
        return

    running_jobs = get_running_jobs()

    def print_report():
        _t0 = time.time()
        print(f"\n=== Training Progress @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"Dataset: {args.dataset}\n")
        data = []
        for version in versions:
            job_id = version_jobs.get(version)
            if not job_id:
                continue

            log_data = get_log_data(version, job_id)
            if not log_data:
                continue

            reg = version_registry.get(version, {})
            results_dir = reg.get("results_dir", "")
            agent_dir = os.path.join(PROJECT_ROOT, results_dir)
            train_dir = os.path.join(agent_dir, "train")
            models_dir = os.path.join(agent_dir, "models")

            stats = get_train_stats(train_dir)
            trend = get_train_trend(train_dir, n=args.trend_n)
            model_idx = get_latest_model_index(models_dir)
            ckpt_file, ckpt_name = get_train_checkpoint_file(train_dir)

            log_data["train_speedups"] = stats["speedups"]
            log_data["train_failed"] = stats["failed"]
            log_data["train_total"] = stats["total"]
            log_data["train_trend"] = trend
            log_data["model_index"] = model_idx
            log_data["ckpt_file"] = ckpt_file

            # Update status from squeue
            if job_id in running_jobs and log_data.get("status") == "running":
                j = running_jobs[job_id]
                log_data["slurm_time"] = j["time"]
                log_data["slurm_state"] = j["state"]

            data.append(log_data)

        if not data:
            print("No training data found.")
            return
        format_table(data)

        print(f"\nReport generated in {time.time() - _t0:.1f}s")

        if getattr(args, 'json', None):
            export = []
            for d in data:
                row = {
                    "version": d.get("version"),
                    "job_id": d.get("job_id"),
                    "iteration": d.get("iteration"),
                    "total": d.get("total"),
                    "status": d.get("status"),
                    "model_index": d.get("model_index"),
                }
                spds = d.get("train_speedups", [])
                if spds:
                    try:
                        row["geo_mean"] = math.exp(sum(math.log(max(s, 1e-12)) for s in spds) / len(spds))
                    except (ValueError, OverflowError):
                        row["geo_mean"] = 0.0
                    row["best"] = max(spds)
                    row["worst"] = min(spds)
                    row["n_benchmarks"] = len(spds)
                row["failed"] = d.get("train_failed", 0)
                export.append(row)
            with open(args.json, "w") as f:
                json.dump(export, f, indent=2)
            print(f"\nExported to {args.json}")

    if args.watch:
        try:
            while True:
                version_jobs.update(auto_detect_versions())
                print("\033[2J\033[H")  # clear screen
                print_report()
                sys.stdout.flush()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_report()


if __name__ == "__main__":
    main()
