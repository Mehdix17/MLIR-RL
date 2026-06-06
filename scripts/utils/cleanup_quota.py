#!/usr/bin/env python3
"""Cleanup script: delete duplicates, obsolete markers, and thin model checkpoints.

Steps:
  1. Delete agent-root models/ dirs (duplicates of run_0/models/)
  2. Delete ALL global_markers/ dirs (migrated to train/)
  3. Delete V0 run_1/eval/markers*/ (old per-iter markers)
  4. Delete empty eval-only run dirs (3-file residuals)
  5. Thin completed run models: keep first, last, and every 50th
  6. Thin ablation run_0 models: same logic

CRITICAL: Never touches currently running training runs.
"""

import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = "/scratch/mb10856/MLIR-RL"
RESULTS = os.path.join(PROJECT_ROOT, "results", "new_dataset_results")

# These runs are CURRENTLY TRAINING — DO NOT TOUCH
RUNNING_RUNS = {
    "v4_6_agent": {"run_2"},
    "v4_7_agent": {"run_7"},
    "v4_8_agent": {"run_5"},
    "v0_agent_v2": {"run_2"},
}

# Runs that were created for eval only (no models, just exec_data/logs/tags)
# These can be deleted entirely
EVAL_ONLY_RUNS = [
    "v4_5_agent/run_5", "v4_5_agent/run_6", "v4_5_agent/run_7",
    "v4_5_agent/run_8", "v4_5_agent/run_9", "v4_5_agent/run_10",
    "v4_5_agent/run_11", "v4_5_agent/run_12", "v4_5_agent/run_13",
    "v4_5_agent/run_14", "v4_5_agent/run_15", "v4_5_agent/run_16",
    "v4_5_agent/run_17", "v4_5_agent/run_18",
    "v4_5_agent/run_70", "v4_5_agent/run_71", "v4_5_agent/run_72",
    "v4_5_agent/run_73", "v4_5_agent/run_74", "v4_5_agent/run_75",
    # Ablation eval-only runs
    "ablation_study/v45_no_hw_agent/run_80", "ablation_study/v45_no_hw_agent/run_81",
    "ablation_study/v45_no_hw_agent/run_82", "ablation_study/v45_no_hw_agent/run_83",
    "ablation_study/v45_no_hw_agent/run_84", "ablation_study/v45_no_hw_agent/run_85",
    "ablation_study/v45_no_hw_agent/run_86", "ablation_study/v45_no_hw_agent/run_87",
    "ablation_study/v45_no_hw_agent/run_88", "ablation_study/v45_no_hw_agent/run_89",
    "ablation_study/v45_no_hw_agent/run_90", "ablation_study/v45_no_hw_agent/run_91",
    "ablation_study/v45_no_hw_agent/run_92", "ablation_study/v45_no_hw_agent/run_93",
    "ablation_study/v45_no_hw_agent/run_94", "ablation_study/v45_no_hw_agent/run_95",
    "ablation_study/v45_no_hw_agent/run_96", "ablation_study/v45_no_hw_agent/run_97",
    "ablation_study/v45_no_hw_agent/run_98",
    "ablation_study/v45_no_shaped_reward_agent/run_100", "ablation_study/v45_no_shaped_reward_agent/run_101",
    "ablation_study/v45_no_shaped_reward_agent/run_102", "ablation_study/v45_no_shaped_reward_agent/run_103",
    "ablation_study/v45_no_shaped_reward_agent/run_104", "ablation_study/v45_no_shaped_reward_agent/run_105",
    "ablation_study/v45_no_shaped_reward_agent/run_106", "ablation_study/v45_no_shaped_reward_agent/run_107",
    "ablation_study/v45_no_shaped_reward_agent/run_108", "ablation_study/v45_no_shaped_reward_agent/run_109",
    "ablation_study/v45_no_shaped_reward_agent/run_110", "ablation_study/v45_no_shaped_reward_agent/run_111",
    "ablation_study/v45_no_shaped_reward_agent/run_112", "ablation_study/v45_no_shaped_reward_agent/run_113",
    "ablation_study/v45_no_shaped_reward_agent/run_114", "ablation_study/v45_no_shaped_reward_agent/run_115",
    "ablation_study/v45_no_shaped_reward_agent/run_116", "ablation_study/v45_no_shaped_reward_agent/run_117",
    "ablation_study/v45_no_shaped_reward_agent/run_118", "ablation_study/v45_no_shaped_reward_agent/run_119",
    "ablation_study/v45_no_shaped_reward_agent/run_120", "ablation_study/v45_no_shaped_reward_agent/run_122",
    "ablation_study/v45_no_shaped_reward_agent/run_123", "ablation_study/v45_no_shaped_reward_agent/run_125",
    "ablation_study/v45_no_shaped_reward_agent/run_126",
    "ablation_study/v45_no_transformer_agent/run_50", "ablation_study/v45_no_transformer_agent/run_51",
    "ablation_study/v45_no_transformer_agent/run_52", "ablation_study/v45_no_transformer_agent/run_53",
    "ablation_study/v45_no_transformer_agent/run_54", "ablation_study/v45_no_transformer_agent/run_55",
    "ablation_study/v45_no_transformer_agent/run_56", "ablation_study/v45_no_transformer_agent/run_57",
    "ablation_study/v45_no_transformer_agent/run_58", "ablation_study/v45_no_transformer_agent/run_59",
    "ablation_study/v45_no_transformer_agent/run_60", "ablation_study/v45_no_transformer_agent/run_61",
    "ablation_study/v45_no_transformer_agent/run_62", "ablation_study/v45_no_transformer_agent/run_63",
    "ablation_study/v45_no_transformer_agent/run_64", "ablation_study/v45_no_transformer_agent/run_65",
    "ablation_study/v45_no_transformer_agent/run_66", "ablation_study/v45_no_transformer_agent/run_67",
    "ablation_study/v45_no_transformer_agent/run_68", "ablation_study/v45_no_transformer_agent/run_69",
]

# Ablation runs that are NOT just eval-only — have real models, keep but thin
ABLATION_MODEL_RUNS = [
    "ablation_study/v45_no_hw_agent/run_0",
    "ablation_study/v45_no_shaped_reward_agent/run_0",
    "ablation_study/v45_no_shaped_reward_agent/run_121",
    "ablation_study/v45_no_shaped_reward_agent/run_124",
    "ablation_study/v45_no_shaped_reward_agent/run_127",
    "ablation_study/v45_no_shaped_reward_agent/run_128",
    "ablation_study/v45_no_shaped_reward_agent/run_129",
    "ablation_study/v45_no_shaped_reward_agent/run_130",
    "ablation_study/v45_no_transformer_agent/run_0",
]


def is_running(agent, run_name):
    """Check if this run is currently being trained."""
    return run_name in RUNNING_RUNS.get(agent, set())


def count_files(path):
    """Count files in a directory recursively."""
    if not os.path.isdir(path):
        return 0
    return sum(1 for _ in Path(path).rglob("*") if _.is_file())


def dir_size(path):
    """Get directory size in bytes."""
    if not os.path.isdir(path):
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def fmt_size(size):
    for unit in ["B", "K", "M", "G", "T"]:
        if abs(size) < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}P"


def step1_delete_agent_models():
    """Delete agent-root models/ dirs — duplicates of run_0/models/."""
    print("\n=== Step 1: Delete agent-root models/ duplicates ===")
    for agent in ["v4_6_agent", "v4_7_agent", "v4_8_agent"]:
        models_dir = os.path.join(RESULTS, agent, "models")
        if os.path.isdir(models_dir):
            n = count_files(models_dir)
            s = dir_size(models_dir)
            print(f"  Deleting {agent}/models/ ({n} files, {fmt_size(s)})")
            shutil.rmtree(models_dir)
            print(f"    Done — freed {n} files")


def step2_delete_global_markers():
    """Delete ALL global_markers/ dirs — migrated to train/."""
    print("\n=== Step 2: Delete all global_markers/ ===")
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(RESULTS):
        if os.path.basename(root) == "global_markers":
            # Only delete direct global_markers dirs (depth ~2-3)
            depth = root.replace(RESULTS, "").count(os.sep)
            if 2 <= depth <= 4:
                n = count_files(root)
                s = dir_size(root)
                total_files += n
                total_size += s
                print(f"  Deleting {os.path.relpath(root, RESULTS)} ({n} files, {fmt_size(s)})")
                shutil.rmtree(root)
    print(f"    Total: {total_files} files, {fmt_size(total_size)}")


def step3_delete_v0_markers():
    """Delete V0 run_1/eval/markers*/ — old per-iter markers."""
    print("\n=== Step 3: Delete V0 per-iter eval markers ===")
    v0_eval = os.path.join(RESULTS, "v0_agent_v2", "run_1", "eval")
    if not os.path.isdir(v0_eval):
        return
    total = 0
    for entry in os.listdir(v0_eval):
        if entry.startswith("markers"):
            path = os.path.join(v0_eval, entry)
            if os.path.isdir(path):
                n = count_files(path)
                s = dir_size(path)
                total += n
                print(f"  Deleting {entry} ({n} files, {fmt_size(s)})")
                shutil.rmtree(path)
    print(f"    Total: {total} files")


def step4_delete_eval_only_runs():
    """Delete empty/eval-only run dirs."""
    print("\n=== Step 4: Delete eval-only run dirs ===")
    total = 0
    for run_path in EVAL_ONLY_RUNS:
        full = os.path.join(RESULTS, run_path)
        if os.path.isdir(full):
            n = count_files(full)
            s = dir_size(full)
            total += n
            print(f"  Deleting {run_path} ({n} files, {fmt_size(s)})")
            shutil.rmtree(full)
    print(f"    Total: {total} files")


def thin_models(models_dir, keep_every=50):
    """Keep first, last, and every N-th model checkpoint. Delete the rest."""
    if not os.path.isdir(models_dir):
        return 0, 0

    pt_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pt")]
    if len(pt_files) <= keep_every:
        return 0, 0

    indices = []
    for f in pt_files:
        m = re.match(r"model_(\d+)\.pt", f)
        if m:
            indices.append((int(m.group(1)), f))

    if not indices:
        return 0, 0

    indices.sort(key=lambda x: x[0])
    first_idx = indices[0][0]
    last_idx = indices[-1][0]

    keep = set()
    for idx, fname in indices:
        if idx == first_idx or idx == last_idx or idx % keep_every == 0:
            keep.add(fname)

    deleted = 0
    freed = 0
    for idx, fname in indices:
        if fname not in keep:
            fpath = os.path.join(models_dir, fname)
            try:
                sz = os.path.getsize(fpath)
                os.remove(fpath)
                deleted += 1
                freed += sz
            except OSError:
                pass

    return deleted, freed


def step56_thin_all_models():
    """Thin model checkpoints in all completed runs."""
    print("\n=== Steps 5-6: Thin model checkpoints (keep every 50th) ===")
    total_deleted = 0
    total_freed = 0

    # Standard agents: all run dirs except running ones
    agents = ["v4_6_agent", "v4_7_agent", "v4_8_agent", "v0_agent_v2", "v4_5_agent"]
    for agent in agents:
        agent_dir = os.path.join(RESULTS, agent)
        if not os.path.isdir(agent_dir):
            continue
        for entry in sorted(os.listdir(agent_dir)):
            if not entry.startswith("run_") or not entry.split("_")[-1].isdigit():
                continue
            run_name = entry
            if is_running(agent, run_name):
                print(f"  SKIP {agent}/{run_name} (running)")
                continue
            models_dir = os.path.join(agent_dir, run_name, "models")
            if not os.path.isdir(models_dir):
                continue
            n_pt = len([f for f in os.listdir(models_dir) if f.endswith(".pt")])
            if n_pt == 0:
                continue
            deleted, freed = thin_models(models_dir, keep_every=50)
            total_deleted += deleted
            total_freed += freed
            if deleted > 0:
                kept = n_pt - deleted
                print(f"  {agent}/{run_name}: deleted {deleted} models (kept {kept}, freed {fmt_size(freed)})")

    # Ablation study agents
    print()
    for run_path in ABLATION_MODEL_RUNS:
        full = os.path.join(RESULTS, run_path)
        models_dir = os.path.join(full, "models")
        if not os.path.isdir(models_dir):
            continue
        n_pt = len([f for f in os.listdir(models_dir) if f.endswith(".pt")])
        if n_pt == 0:
            continue
        deleted, freed = thin_models(models_dir, keep_every=50)
        total_deleted += deleted
        total_freed += freed
        if deleted > 0:
            kept = n_pt - deleted
            print(f"  {run_path}: deleted {deleted} models (kept {kept}, freed {fmt_size(freed)})")

    print(f"\n    Total models deleted: {total_deleted}, freed: {fmt_size(total_freed)}")
    return total_deleted, total_freed


def check_quota():
    """Check current Lustre quota."""
    import subprocess
    try:
        out = subprocess.check_output(["lfs", "quota", "-u", os.environ["USER"], "/scratch"],
                                      stderr=subprocess.DEVNULL, timeout=10).decode()
        for line in out.split("\n"):
            if "files" in line and "quota" in line:
                return line.strip()
    except Exception:
        pass
    return "unavailable"


def main():
    print("=" * 60)
    print("MLIR-RL Cleanup Script")
    print("=" * 60)

    print(f"\nBefore: {check_quota()}")

    step1_delete_agent_models()
    step2_delete_global_markers()
    step3_delete_v0_markers()
    step4_delete_eval_only_runs()
    total_model_files, total_model_size = step56_thin_all_models()

    print(f"\n{'=' * 60}")
    print(f"After: {check_quota()}")

    # Also count remaining top-level model dirs that shouldn't exist
    for agent in ["v4_6_agent", "v4_7_agent", "v4_8_agent"]:
        mdir = os.path.join(RESULTS, agent, "models")
        if os.path.isdir(mdir):
            print(f"WARNING: {agent}/models/ still exists (should be deleted)")

    print("\n=== CLEANUP COMPLETE ===")


if __name__ == "__main__":
    main()
