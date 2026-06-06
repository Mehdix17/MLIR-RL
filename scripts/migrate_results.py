#!/usr/bin/env python3
"""Migrate existing results to new run_i/ structure.

Restructures agent directories from the old format:
  agent_dir/
    impl_subdir/run_N/   →  agent_dir/run_N/
    global_markers/      →  run_N/train/checkpoint_N.json
    models/              →  run_N/models/
    eval/                →  run_N/eval/

Usage:
  python scripts/migrate_results.py                    # dry-run (preview only)
  python scripts/migrate_results.py --execute           # perform migration
  python scripts/migrate_results.py -d results/new_dataset_results/v4_7_agent  # one agent
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMPL_SUBDIR_CANDIDATES = [
    "v4_5_agent", "v4_5_small", "old_agent",
    "rl_autoschedular_v45_no_hw_agent",
    "rl_autoschedular_v45_no_shaped_reward_agent",
    "rl_autoschedular_v45_no_transformer_agent",
    "rl_autoschedular_v0_agent",
    "rl_autoschedular_v0_agent_v2",
]


def find_agent_dirs(base_dir: str) -> list[str]:
    """Find all agent directories under base_dir."""
    agent_dirs = []
    base = os.path.join(PROJECT_ROOT, base_dir)
    if not os.path.isdir(base):
        return []

    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full) and "_agent" in entry:
            agent_dirs.append(full)

    # Also check ablation_study/ subdirs
    ablation = os.path.join(base, "ablation_study")
    if os.path.isdir(ablation):
        for entry in os.listdir(ablation):
            full = os.path.join(ablation, entry)
            if os.path.isdir(full) and "_agent" in entry:
                agent_dirs.append(full)

    return sorted(agent_dirs)


def find_impl_subdir(agent_dir: str) -> str | None:
    """Find the implementation subdirectory (contains run_N/ dirs)."""
    # Check known names first
    for candidate in IMPL_SUBDIR_CANDIDATES:
        cand_path = os.path.join(agent_dir, candidate)
        if os.path.isdir(cand_path):
            return cand_path

    # Fallback: find any subdir containing run_*/ dirs
    for entry in os.listdir(agent_dir):
        full = os.path.join(agent_dir, entry)
        if not os.path.isdir(full):
            continue
        for sub in os.listdir(full):
            if sub.startswith("run_") and os.path.isdir(os.path.join(full, sub)):
                return full

    return None


def find_run_dirs(agent_dir: str, impl_subdir: str | None) -> list[str]:
    """Find all run_N/ directories. Checks both impl_subdir and agent root."""
    run_dirs = {}

    # Check impl subdir
    if impl_subdir and os.path.isdir(impl_subdir):
        for entry in os.listdir(impl_subdir):
            full = os.path.join(impl_subdir, entry)
            if entry.startswith("run_") and os.path.isdir(full):
                run_dirs[entry] = full

    # Check agent root for run dirs already at top level
    for entry in os.listdir(agent_dir):
        full = os.path.join(agent_dir, entry)
        if entry.startswith("run_") and os.path.isdir(full) and entry not in run_dirs:
            run_dirs[entry] = full

    return run_dirs


def convert_global_markers_to_train(agent_dir: str, run_dir: str, dry_run: bool):
    """Convert global_markers/training/iter_N/ → train/checkpoint_N.json.

    Each iter_N/ contains one batch worth of markers (~64 benchmarks).
    We accumulate across all iterations and snapshot every 100 iters + last.
    The latest accumulated result is saved as results.json.
    Also merges any markers in global_markers/default/ (mid-crash state).
    """
    gm_path = os.path.join(agent_dir, "global_markers")
    training_path = os.path.join(gm_path, "training")
    if not os.path.isdir(training_path):
        return

    train_dir = os.path.join(run_dir, "train")
    if not dry_run:
        os.makedirs(train_dir, exist_ok=True)

    iter_dirs = sorted(
        [d for d in os.listdir(training_path)
         if d.startswith("iter_") and os.path.isdir(os.path.join(training_path, d))],
        key=lambda x: int(x.split("_")[1])
    )

    if not iter_dirs:
        return

    accumulated: dict = {}
    num_iters = len(iter_dirs)

    for iter_dir in iter_dirs:
        iter_num = int(iter_dir.split("_")[1])
        iter_path = os.path.join(training_path, iter_dir)
        for bench_file in os.listdir(iter_path):
            fpath = os.path.join(iter_path, bench_file)
            if bench_file.startswith("_") or not os.path.isfile(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                accumulated[bench_file] = data
            except Exception:
                pass

        is_last = (iter_dirs.index(iter_dir) == num_iters - 1)
        if iter_num % 100 == 0 or is_last:
            ckpt_file = os.path.join(train_dir, f"checkpoint_{iter_num}.json")
            print(f"  -> {ckpt_file} ({len(accumulated)} benchmarks cumulative)")
            if not dry_run:
                with open(ckpt_file, "w") as f:
                    json.dump(accumulated, f, indent=2)

    # Merge default/ markers (mid-crash state from last incomplete iteration)
    default_path = os.path.join(gm_path, "default")
    if os.path.isdir(default_path):
        for bench_file in os.listdir(default_path):
            fpath = os.path.join(default_path, bench_file)
            if bench_file.startswith("_") or not os.path.isfile(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                accumulated[bench_file] = data
            except Exception:
                pass
        if accumulated:
            results_file = os.path.join(train_dir, "results.json")
            print(f"  -> {results_file} (merged default/ markers, {len(accumulated)} benchmarks)")
            if not dry_run:
                with open(results_file, "w") as f:
                    json.dump(accumulated, f, indent=2)


def convert_global_markers_to_eval(agent_dir: str, run_dir: str, dry_run: bool):
    """Convert global_markers/ckpt_N/ → eval/markers/ checkpoint."""
    gm_path = os.path.join(agent_dir, "global_markers")
    if not os.path.isdir(gm_path):
        return

    eval_markers_dir = os.path.join(run_dir, "eval", "markers")
    for entry in os.listdir(gm_path):
        if not entry.startswith("ckpt_"):
            continue
        ckpt_path = os.path.join(gm_path, entry)
        if not os.path.isdir(ckpt_path):
            continue

        ckpt_num = entry.replace("ckpt_", "")
        dst = os.path.join(eval_markers_dir, ckpt_num)
        bench_count = len([f for f in os.listdir(ckpt_path)
                          if not f.startswith("_") and os.path.isfile(os.path.join(ckpt_path, f))])
        print(f"  global_markers/{entry}/ -> eval/markers/{ckpt_num}/ ({bench_count} markers)")
        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(ckpt_path, dst)


def move_agent_models(agent_dir: str, run_dir: str, dry_run: bool):
    """Move agent_dir/models/ → run_dir/models/."""
    src_models = os.path.join(agent_dir, "models")
    if not os.path.isdir(src_models):
        return

    dst_models = os.path.join(run_dir, "models")
    if not dry_run:
        os.makedirs(dst_models, exist_ok=True)

    pt_files = [f for f in os.listdir(src_models) if f.endswith(".pt")]
    if not pt_files:
        return

    print(f"  models/ -> run_0/models/ ({len(pt_files)} checkpoints)")
    for f in pt_files:
        src = os.path.join(src_models, f)
        dst = os.path.join(dst_models, f)
        try:
            if os.path.exists(dst):
                if os.path.getmtime(src) > os.path.getmtime(dst):
                    if not dry_run:
                        shutil.copy2(src, dst)
            else:
                if not dry_run:
                    shutil.copy2(src, dst)
        except (FileNotFoundError, OSError):
            pass  # Broken symlink or missing file — skip


def move_agent_eval(agent_dir: str, run_dir: str, dry_run: bool):
    """Move agent_dir/eval/checkpoint_*.json → run_dir/eval/checkpoint_*.json."""
    src_eval = os.path.join(agent_dir, "eval")
    if not os.path.isdir(src_eval):
        return

    dst_eval = os.path.join(run_dir, "eval")
    if not dry_run:
        os.makedirs(dst_eval, exist_ok=True)

    ckpt_files = [f for f in os.listdir(src_eval)
                  if f.startswith("checkpoint_") and f.endswith(".json")]
    if ckpt_files:
        print(f"  eval/checkpoint_*.json -> run_0/eval/ ({len(ckpt_files)} checkpoints)")
        for f in ckpt_files:
            src = os.path.join(src_eval, f)
            dst = os.path.join(dst_eval, f)
            if not dry_run and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Move eval logs
    src_logs = os.path.join(src_eval, "logs")
    if os.path.isdir(src_logs):
        dst_logs = os.path.join(run_dir, "logs", "eval")
        print(f"  eval/logs/ -> run_0/logs/eval/")
        if not dry_run:
            if os.path.exists(dst_logs):
                # Merge: copy files that don't exist in dst
                for root, dirs, files in os.walk(src_logs):
                    rel = os.path.relpath(root, src_logs)
                    target = os.path.join(dst_logs, rel) if rel != "." else dst_logs
                    os.makedirs(target, exist_ok=True)
                    for fname in files:
                        sf = os.path.join(root, fname)
                        df = os.path.join(target, fname)
                        if not os.path.exists(df):
                            shutil.copy2(sf, df)
            else:
                shutil.copytree(src_logs, dst_logs, dirs_exist_ok=True)


def move_exec_data(agent_dir: str, run_dir: str, dry_run: bool):
    """Move run_N/exec_data.json → run_N/logs/exec_data.json."""
    src = os.path.join(run_dir, "exec_data.json")
    if not os.path.isfile(src):
        return

    dst = os.path.join(run_dir, "logs", "exec_data.json")
    if os.path.exists(dst):
        return  # Already moved or exists in logs/

    print(f"  exec_data.json -> logs/exec_data.json")
    if not dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def merge_impl_run_dirs(agent_dir: str, impl_subdir: str, run_dirs: dict, dry_run: bool):
    """Move run_N/ dirs from impl_subdir to agent_dir root."""
    if not impl_subdir or not os.path.isdir(impl_subdir):
        return

    for entry in os.listdir(impl_subdir):
        if not entry.startswith("run_"):
            continue
        src = os.path.join(impl_subdir, entry)
        dst = os.path.join(agent_dir, entry)

        if os.path.exists(dst):
            # Merge: move files from src to dst
            print(f"  Merging {impl_subdir}/{entry}/ -> {entry}/")
            if not dry_run:
                for root, dirs, files in os.walk(src):
                    rel = os.path.relpath(root, src)
                    target = os.path.join(dst, rel) if rel != "." else dst
                    os.makedirs(target, exist_ok=True)
                    for fname in files:
                        sf = os.path.join(root, fname)
                        df = os.path.join(target, fname)
                        if not os.path.exists(df):
                            shutil.move(sf, df)
                # Cleanup empty dirs in src
                for root, dirs, files in os.walk(src, topdown=False):
                    if root != src and not os.listdir(root):
                        os.rmdir(root)
                if not os.listdir(src):
                    os.rmdir(src)
        else:
            print(f"  Moving {impl_subdir}/{entry}/ -> {entry}/")
            if not dry_run:
                shutil.move(src, dst)


def cleanup(agent_dir: str, dry_run: bool):
    """Remove empty directories."""
    for subdir in ["global_markers"]:
        path = os.path.join(agent_dir, subdir)
        if os.path.isdir(path) and not os.listdir(path):
            print(f"  Removing empty {subdir}/")
            if not dry_run:
                os.rmdir(path)

    # Remove impl subdir if empty
    for entry in os.listdir(agent_dir):
        full = os.path.join(agent_dir, entry)
        if os.path.isdir(full) and entry in IMPL_SUBDIR_CANDIDATES:
            if not os.listdir(full):
                print(f"  Removing empty impl subdir {entry}/")
                if not dry_run:
                    os.rmdir(full)


def migrate_agent(agent_dir: str, dry_run: bool):
    """Migrate one agent directory to new structure."""
    agent_name = os.path.basename(agent_dir)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}{agent_name}:")

    impl_subdir = find_impl_subdir(agent_dir)
    run_dirs = find_run_dirs(agent_dir, impl_subdir)

    if impl_subdir:
        print(f"  impl subdir: {os.path.basename(impl_subdir)}")

    # Determine primary run dir (run_0)
    primary_run = run_dirs.get("run_0")
    if not primary_run:
        primary_run = os.path.join(agent_dir, "run_0")
        if not os.path.isdir(primary_run):
            if not dry_run:
                os.makedirs(primary_run, exist_ok=True)
            print(f"  Creating run_0/")

    # 1. Merge run dirs from impl subdir to agent root
    merge_impl_run_dirs(agent_dir, impl_subdir, run_dirs, dry_run)
    # Reload run dirs after merge
    run_dirs = find_run_dirs(agent_dir, None)
    primary_run = run_dirs.get("run_0", primary_run)

    # 2. Convert global_markers/training/ -> train/
    convert_global_markers_to_train(agent_dir, primary_run, dry_run)

    # 3. Convert global_markers/ckpt_*/ -> eval/markers/
    convert_global_markers_to_eval(agent_dir, primary_run, dry_run)

    # 4. Move agent-level models/ -> run_0/models/
    move_agent_models(agent_dir, primary_run, dry_run)

    # 5. Move agent-level eval/ -> run_0/eval/
    move_agent_eval(agent_dir, primary_run, dry_run)

    # 6. Move exec_data.json from run root to logs/
    for run_name, run_dir in run_dirs.items():
        move_exec_data(agent_dir, run_dir, dry_run)

    # 7. Cleanup
    cleanup(agent_dir, dry_run)

    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser(description="Migrate results to new run_i/ structure")
    parser.add_argument("--execute", action="store_true", help="Execute migration (default: dry-run)")
    parser.add_argument("-d", "--dir", help="Single agent directory to migrate")
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("=== DRY RUN (no changes will be made) ===\n"
              "Use --execute to perform the migration.\n")

    if args.dir:
        agent_dir = args.dir
        if not os.path.isabs(agent_dir):
            agent_dir = os.path.join(PROJECT_ROOT, agent_dir)
        if not os.path.isdir(agent_dir):
            print(f"Error: Directory not found: {agent_dir}")
            sys.exit(1)
        migrate_agent(agent_dir, dry_run)
    else:
        agent_dirs = find_agent_dirs("results/new_dataset_results")
        if not agent_dirs:
            print("No agent directories found under results/new_dataset_results/")
            sys.exit(1)

        print(f"Found {len(agent_dirs)} agent directories:\n")
        for d in agent_dirs:
            print(f"  {os.path.relpath(d, PROJECT_ROOT)}")
        print()

        for agent_dir in agent_dirs:
            migrate_agent(agent_dir, dry_run)

    if dry_run:
        print("\n=== DRY RUN COMPLETE ===\n"
              "Run with --execute to apply the migration.")


if __name__ == "__main__":
    main()
