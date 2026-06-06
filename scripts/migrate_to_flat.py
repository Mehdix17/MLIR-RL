#!/usr/bin/env python3
"""Migrate results from run_i/ to flat structure.

For each agent dir:
  - Merge run_N/models/ -> models/
  - Merge run_N/train/ -> train/
  - Merge run_N/eval/ -> eval/
  - Merge run_N/logs/ -> logs/
  - Remove run_N/, global_markers/, old impl subdirs

Usage:
  python scripts/migrate_to_flat.py --execute   # perform migration
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results", "new_dataset_results")

IMPL_SUBDIR_NAMES = {
    "v4_5_agent", "v4_5_small", "old_agent",
    "rl_autoschedular_v45_no_hw_agent",
    "rl_autoschedular_v45_no_shaped_reward_agent",
    "rl_autoschedular_v45_no_transformer_agent",
}


def find_agent_dirs():
    """Find all experiment agent directories."""
    agents = []
    for entry in sorted(os.listdir(RESULTS)):
        full = os.path.join(RESULTS, entry)
        if os.path.isdir(full) and "_agent" in entry:
            agents.append(full)
    ablation = os.path.join(RESULTS, "ablation_study")
    if os.path.isdir(ablation):
        for entry in sorted(os.listdir(ablation)):
            full = os.path.join(ablation, entry)
            if os.path.isdir(full) and "_agent" in entry:
                agents.append(full)
    return agents


def count_files(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for _ in Path(path).rglob("*") if _.is_file())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def migrate_models(agent_dir, dry_run):
    """Collect all model_*.pt from run_N/models/, keep newest per index."""
    src_models = {}
    for entry in os.listdir(agent_dir):
        if not entry.startswith("run_") or not entry.split("_")[-1].isdigit():
            continue
        mdir = os.path.join(agent_dir, entry, "models")
        if not os.path.isdir(mdir):
            continue
        for f in os.listdir(mdir):
            m = re.match(r"model_(\d+)\.pt", f)
            if not m:
                continue
            idx = int(m.group(1))
            fpath = os.path.join(mdir, f)
            mtime = os.path.getmtime(fpath)
            if idx not in src_models or mtime > src_models[idx][1]:
                src_models[idx] = (fpath, mtime)

    if not src_models:
        return

    dst_dir = os.path.join(agent_dir, "models")
    ensure_dir(dst_dir)
    copied = 0
    for idx, (src, _) in src_models.items():
        dst = os.path.join(dst_dir, f"model_{idx}.pt")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue  # destination is newer or same
        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1
    print(f"  models/: {copied} checkpoints merged from runs")


def merge_json_files(src_dirs, out_file, dry_run):
    """Merge JSON dicts from multiple directories, later overwrites earlier."""
    merged = {}
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            continue
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith(".json"):
                continue
            fpath = os.path.join(src_dir, f)
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                if f == "results.json" or f.startswith("checkpoint_"):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            merged[k] = v  # later overwrites
                elif f.startswith("checkpoint_"):
                    pass  # already handled above
            except Exception:
                pass

    if merged:
        if not dry_run:
            with open(out_file, "w") as f:
                json.dump(merged, f, indent=2)
        return len(merged)
    return 0


def migrate_train(agent_dir, dry_run):
    """Merge train/results.json and checkpoint_*.json from runs."""
    results_src = []
    ckpt_src = {}  # ckpt_num -> (path, mtime)

    for entry in os.listdir(agent_dir):
        if not entry.startswith("run_") or not entry.split("_")[-1].isdigit():
            continue
        tdir = os.path.join(agent_dir, entry, "train")
        if not os.path.isdir(tdir):
            continue

        # results.json
        rf = os.path.join(tdir, "results.json")
        if os.path.isfile(rf):
            results_src.append(tdir)

        # checkpoint_*.json
        for f in os.listdir(tdir):
            m = re.match(r"checkpoint_(\d+)\.json", f)
            if m:
                ckpt = int(m.group(1))
                fpath = os.path.join(tdir, f)
                mtime = os.path.getmtime(fpath)
                if ckpt not in ckpt_src or mtime > ckpt_src[ckpt][1]:
                    ckpt_src[ckpt] = (fpath, mtime)

    dst_dir = os.path.join(agent_dir, "train")
    ensure_dir(dst_dir)

    # Merge results.json
    if results_src:
        merged = {}
        for tdir in sorted(results_src):
            rf = os.path.join(tdir, "results.json")
            try:
                with open(rf) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        merged[k] = v
            except Exception:
                pass
        if merged:
            dst = os.path.join(dst_dir, "results.json")
            if not dry_run:
                with open(dst, "w") as f:
                    json.dump(merged, f, indent=2)
            print(f"  train/results.json: {len(merged)} benchmarks merged")

    # Copy checkpoint files
    ckpt_count = 0
    for ckpt, (src, _) in ckpt_src.items():
        dst = os.path.join(dst_dir, f"checkpoint_{ckpt}.json")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        ckpt_count += 1
    if ckpt_count:
        print(f"  train/: {ckpt_count} checkpoints merged")


def migrate_eval(agent_dir, dry_run):
    """Merge eval/checkpoint_*.json and eval/markers/ from runs."""
    ckpt_src = {}  # ckpt -> (path, mtime)
    marker_src = {}  # ckpt -> [(path, mtime)]

    for entry in os.listdir(agent_dir):
        if not entry.startswith("run_") or not entry.split("_")[-1].isdigit():
            continue
        edir = os.path.join(agent_dir, entry, "eval")
        if not os.path.isdir(edir):
            continue

        for f in os.listdir(edir):
            m = re.match(r"checkpoint_(\d+)\.json", f)
            if m:
                ckpt = int(m.group(1))
                fpath = os.path.join(edir, f)
                mtime = os.path.getmtime(fpath)
                if ckpt not in ckpt_src or mtime > ckpt_src[ckpt][1]:
                    ckpt_src[ckpt] = (fpath, mtime)

        markers_dir = os.path.join(edir, "markers")
        if not os.path.isdir(markers_dir):
            continue
        for ckpt_dir in os.listdir(markers_dir):
            ckpt_path = os.path.join(markers_dir, ckpt_dir)
            if not os.path.isdir(ckpt_path):
                continue
            if ckpt_dir not in marker_src:
                marker_src[ckpt_dir] = []
            for mf in os.listdir(ckpt_path):
                mp = os.path.join(ckpt_path, mf)
                if os.path.isfile(mp):
                    marker_src[ckpt_dir].append((mp, os.path.getmtime(mp)))

    dst_dir = os.path.join(agent_dir, "eval")
    ensure_dir(dst_dir)

    ckpt_count = 0
    for ckpt, (src, _) in ckpt_src.items():
        dst = os.path.join(dst_dir, f"checkpoint_{ckpt}.json")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        ckpt_count += 1

    marker_count = 0
    if marker_src:
        dst_markers = os.path.join(dst_dir, "markers")
        ensure_dir(dst_markers)
        for ckpt_dir, files in marker_src.items():
            ckpt_dst = os.path.join(dst_markers, ckpt_dir)
            ensure_dir(ckpt_dst)
            for src, _ in files:
                fname = os.path.basename(src)
                dst_f = os.path.join(ckpt_dst, fname)
                if os.path.exists(dst_f) and os.path.getmtime(dst_f) >= os.path.getmtime(src):
                    continue
                if not dry_run:
                    shutil.copy2(src, dst_f)
                marker_count += 1

    if ckpt_count or marker_count:
        print(f"  eval/: {ckpt_count} checkpoints, {marker_count} markers merged")


def migrate_logs(agent_dir, dry_run):
    """Merge logs/ from runs: exec_data.json (newest), tags, concat per-iter files."""
    dst_dir = os.path.join(agent_dir, "logs")
    ensure_dir(dst_dir)

    # Collect run dirs in order
    runs = sorted(
        [d for d in os.listdir(agent_dir) if d.startswith("run_") and d.split("_")[-1].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )

    # exec_data.json — keep one with most entries (largest file)
    best_exec = None
    best_size = 0
    for run_name in runs:
        exec_f = os.path.join(agent_dir, run_name, "logs", "exec_data.json")
        if os.path.isfile(exec_f):
            sz = os.path.getsize(exec_f)
            if sz > best_size:
                best_size = sz
                best_exec = exec_f
    if best_exec:
        dst = os.path.join(dst_dir, "exec_data.json")
        if not dry_run and (not os.path.exists(dst) or os.path.getsize(dst) < best_size):
            shutil.copy2(best_exec, dst)
        print(f"  logs/exec_data.json: merged")

    # tags — keep from any run (they're identical)
    for run_name in runs:
        tags_f = os.path.join(agent_dir, run_name, "tags")
        if os.path.isfile(tags_f) and not os.path.exists(os.path.join(dst_dir, "tags")):
            if not dry_run:
                shutil.copy2(tags_f, os.path.join(dst_dir, "tags"))
            break

    # Per-iter metric files: concatenate in order
    metric_dirs = ["train", "train_ppo", "train_value"]
    for mdir_name in metric_dirs:
        out_file = os.path.join(dst_dir, mdir_name)
        ensure_dir(out_file) if not dry_run else None
        # Gather all metrics from all runs
        metrics = {}
        for run_name in runs:
            src_mdir = os.path.join(agent_dir, run_name, "logs", mdir_name)
            if not os.path.isdir(src_mdir):
                continue
            for f in os.listdir(src_mdir):
                fpath = os.path.join(src_mdir, f)
                if not os.path.isfile(fpath):
                    continue
                if f not in metrics:
                    metrics[f] = []
                with open(fpath) as fh:
                    content = fh.read()
                metrics[f].append(content)

        for metric_name, chunks in metrics.items():
            dst = os.path.join(out_file, metric_name)
            if not dry_run:
                with open(dst, "w") as fh:
                    fh.write("".join(chunks))

    # eval/ — eval_exec_times.json from latest run
    for run_name in reversed(runs):
        eval_f = os.path.join(agent_dir, run_name, "logs", "eval", "eval_exec_times.json")
        if os.path.isfile(eval_f):
            dst = os.path.join(dst_dir, "eval")
            ensure_dir(dst)
            if not dry_run:
                shutil.copy2(eval_f, os.path.join(dst, "eval_exec_times.json"))
            break

    # eval per-benchmark files (speedup, exec_time) — merge from latest run
    for run_name in reversed(runs):
        src_eval = os.path.join(agent_dir, run_name, "logs", "eval")
        if not os.path.isdir(src_eval):
            continue
        for sub in os.listdir(src_eval):
            src_sub = os.path.join(src_eval, sub)
            if not os.path.isdir(src_sub):
                continue
            dst_sub = os.path.join(dst_dir, "eval", sub)
            ensure_dir(dst_sub) if not dry_run else None
            for f in os.listdir(src_sub):
                src_f = os.path.join(src_sub, f)
                dst_f = os.path.join(dst_sub, f)
                if os.path.isfile(src_f) and not os.path.exists(dst_f):
                    if not dry_run:
                        shutil.copy2(src_f, dst_f)

    print(f"  logs/: merged")


def cleanup(agent_dir, dry_run):
    """Remove run_N/, global_markers/, old impl subdirs, duplicate models/."""
    removed = 0
    for entry in list(os.listdir(agent_dir)):
        full = os.path.join(agent_dir, entry)
        if not os.path.isdir(full):
            continue
        # Remove run_N dirs
        if entry.startswith("run_") and entry.split("_")[-1].isdigit():
            n = count_files(full)
            if not dry_run:
                shutil.rmtree(full)
            removed += n
            print(f"  Removed {entry}/ ({n} files)")
        # Remove global_markers
        elif entry == "global_markers":
            n = count_files(full)
            if not dry_run:
                shutil.rmtree(full)
            removed += n
            print(f"  Removed global_markers/ ({n} files)")
        # Remove old impl subdirs
        elif entry in IMPL_SUBDIR_NAMES:
            n = count_files(full)
            if not dry_run:
                shutil.rmtree(full)
            removed += n
            print(f"  Removed {entry}/ ({n} files)")
    return removed


def migrate_agent(agent_dir, dry_run):
    agent_name = os.path.relpath(agent_dir, PROJECT_ROOT)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}{agent_name}:")

    migrate_models(agent_dir, dry_run)
    migrate_train(agent_dir, dry_run)
    migrate_eval(agent_dir, dry_run)
    migrate_logs(agent_dir, dry_run)
    cleanup(agent_dir, dry_run)

    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("=== DRY RUN ===\n")

    agents = find_agent_dirs()
    print(f"Found {len(agents)} agent directories\n")

    for agent_dir in agents:
        migrate_agent(agent_dir, dry_run)

    if dry_run:
        print("\n=== DRY RUN COMPLETE ===")
    else:
        print("\n=== MIGRATION COMPLETE ===")


if __name__ == "__main__":
    main()
