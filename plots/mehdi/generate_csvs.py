#!/usr/bin/env python3
"""Generate CSV files with geometric mean speedup evolution for V4.6, V4.7, V4.8.

Reads eval checkpoint files and computes geo mean speedup vs MLIR baseline.
Outputs: v4_6/v4_6.csv, v4_7/v4_7.csv, v4_8/v4_8.csv
"""

import csv
import json
import math
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERSIONS = {
    "v0": "results/new_dataset_results/v0_agent/eval",
    "v4_6": "results/new_dataset_results/v4_6_agent/eval",
    "v4_7": "results/new_dataset_results/v4_7_agent/eval",
    "v4_8": "results/new_dataset_results/v4_8_agent/eval",
}

BASELINE_PATH = "results/new_dataset_results/baselines/mlir/eval_base.json"


def load_baseline():
    path = os.path.join(PROJECT_ROOT, BASELINE_PATH)
    with open(path) as f:
        return json.load(f)


def geo_mean_speedup(eval_data: dict, baseline: dict) -> float | None:
    """Compute geometric mean speedup across all benchmarks."""
    speedups = []
    for bench_name, opt_ns in eval_data.items():
        base_ns = baseline.get(bench_name)
        if base_ns is None or base_ns <= 0:
            continue
        if opt_ns is None or opt_ns <= 0:
            continue
        speedups.append(base_ns / opt_ns)
    if not speedups:
        return None
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def get_checkpoint_number(filename: str) -> int:
    m = re.search(r"checkpoint_(\d+)\.json", filename)
    return int(m.group(1)) if m else -1


def main():
    baseline = load_baseline()
    print(f"Baseline: {len(baseline)} benchmarks")

    for version, eval_rel in VERSIONS.items():
        eval_dir = os.path.join(PROJECT_ROOT, eval_rel)
        if not os.path.isdir(eval_dir):
            print(f"Skipping {version}: {eval_dir} not found")
            continue

        checkpoint_files = sorted(
            [f for f in os.listdir(eval_dir) if f.startswith("checkpoint_") and f.endswith(".json")],
            key=get_checkpoint_number,
        )

        rows = []
        for cf in checkpoint_files:
            ckpt_num = get_checkpoint_number(cf)
            if ckpt_num < 0:
                continue
            with open(os.path.join(eval_dir, cf)) as f:
                eval_data = json.load(f)
            gm = geo_mean_speedup(eval_data, baseline)
            if gm is not None:
                rows.append((ckpt_num, gm))

        out_dir = os.path.join(PROJECT_ROOT, "plots", "mehdi", version)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{version}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "average_speedup"])
            for ckpt, speedup in rows:
                writer.writerow([ckpt, f"{speedup:.4f}"])

        print(f"{version}: {len(rows)} checkpoints -> {csv_path}")


if __name__ == "__main__":
    main()
