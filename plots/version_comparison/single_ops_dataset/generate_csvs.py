#!/usr/bin/env python3
"""Generate CSV files for single_ops_dataset eval results.

Produces:
  1-3: Per-version eval evolution CSVs (checkpoint, average_speedup)
  4:   Per-operation type CSV (agent, group, geo_mean, count)
  5:   Per-model type CSV (agent, group, geo_mean, count)
"""

import csv
import json
import math
import os
import re
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

EVAL_DIRS = {
    "V0": ("results/single_ops_dataset_results/v0_agent/eval", "v0"),
    "V4.9-S": ("results/single_ops_dataset_results/v4_9_small_agent/eval", "v4_9_small"),
    "V4.9-L": ("results/single_ops_dataset_results/v4_9_large_agent/eval", "v4_9_large"),
}

BEST_CHECKPOINTS = {
    "V0": 4400,
    "V4.9-S": 3800,
    "V4.9-L": 3200,
}

BASELINE_PATH = "results/single_ops_dataset_results/baselines/mlir/base_eval.json"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_MODELS = [
    "albert", "bart", "bert", "convnext_tiny", "distilbert", "efficientnet_b0",
    "gat", "gin", "gpt2", "llama3_2_1b", "mobilenet_v3_small", "resnet50",
    "resnext50", "t5", "vgg16", "vit_b_16", "whisper_base", "yolov8m",
]

OP_NORMALIZE = {
    "conv": "conv2d",
    "conv_2d_nchw_fchw": "conv2d",
    "pooling": "pooling",
    "pooling_nchw_max": "pooling",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def geo_mean(speedups):
    if not speedups:
        return None
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def get_checkpoint_number(filename):
    m = re.search(r"checkpoint_(\d+)\.json", filename)
    return int(m.group(1)) if m else -1


def classify_op(bench_name):
    """Classify operation type from benchmark name."""
    if "_batch_matmul" in bench_name:
        return "batch_matmul"
    if "_conv_2d" in bench_name or bench_name.startswith("conv_"):
        return "conv2d"
    if "_matmul" in bench_name or bench_name.startswith("matmul"):
        return "matmul"
    if "_pooling" in bench_name or bench_name.startswith("pooling"):
        return "pooling"
    if "_relu" in bench_name or bench_name.startswith("relu"):
        return "relu"
    if "_generic" in bench_name or bench_name.startswith("generic"):
        return "generic"
    if "_add" in bench_name or bench_name.startswith("add"):
        return "add"
    if "_mul" in bench_name or bench_name.startswith("mul"):
        return "mul"
    if "_reduce_sum" in bench_name or bench_name.startswith("reduce_sum"):
        return "reduce_sum"
    if "_sub" in bench_name or bench_name.startswith("sub"):
        return "sub"
    return "unknown"


def extract_model(bench_name):
    """Extract model name if benchmark is model-prefixed."""
    for m in sorted(KNOWN_MODELS, key=len, reverse=True):
        if bench_name.startswith(m + "_"):
            return m
    return None


def generate_evolution_csvs(baseline):
    """Generate per-version eval evolution CSVs."""
    for version, (eval_rel, dir_name) in EVAL_DIRS.items():
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
            eval_data = load_json(os.path.join(eval_dir, cf))
            speedups = []
            for bench, opt_ns in eval_data.items():
                base_ns = baseline.get(bench)
                if base_ns is None or base_ns <= 0 or opt_ns is None or opt_ns <= 0:
                    continue
                speedups.append(base_ns / opt_ns)
            gm = geo_mean(speedups)
            if gm is not None:
                rows.append((ckpt_num, gm))

        out_dir = os.path.join(OUT_DIR, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{dir_name}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "average_speedup"])
            for ckpt, speedup in rows:
                writer.writerow([ckpt, f"{speedup:.4f}"])

        print(f"{version}: {len(rows)} checkpoints -> {csv_path}")


def generate_per_op_csv(baseline):
    """Generate per-operation type CSV using best checkpoints."""
    # Load eval data for best checkpoint per agent
    agent_bench_speedup = {}  # agent -> {bench -> speedup}
    for version, (eval_rel, _) in EVAL_DIRS.items():
        best_ckpt = BEST_CHECKPOINTS[version]
        eval_file = os.path.join(PROJECT_ROOT, eval_rel, f"checkpoint_{best_ckpt}.json")
        if not os.path.isfile(eval_file):
            print(f"Skipping {version}: {eval_file} not found")
            continue
        eval_data = load_json(eval_file)
        speedups = {}
        for bench, opt_ns in eval_data.items():
            base_ns = baseline.get(bench)
            if base_ns is None or base_ns <= 0 or opt_ns is None or opt_ns <= 0:
                continue
            speedups[bench] = base_ns / opt_ns
        agent_bench_speedup[version] = speedups

    # Group by operation type
    op_groups = defaultdict(lambda: defaultdict(list))  # op -> agent -> [speedups]
    for bench in baseline:
        op = classify_op(bench)
        for agent, bench_speedups in agent_bench_speedup.items():
            if bench in bench_speedups:
                op_groups[op][agent].append(bench_speedups[bench])

    # Write CSV
    csv_path = os.path.join(OUT_DIR, "per_operation.csv")
    agents = sorted(agent_bench_speedup.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "group", "geo_mean", "count"])
        for op in sorted(op_groups.keys()):
            for agent in agents:
                speedups = op_groups[op][agent]
                if len(speedups) >= 10:
                    gm = geo_mean(speedups)
                    writer.writerow([agent, op, f"{gm:.4f}", len(speedups)])

    print(f"Per-operation CSV: {csv_path}")


def generate_per_model_csv(baseline):
    """Generate per-model type CSV using best checkpoints."""
    agent_bench_speedup = {}
    for version, (eval_rel, _) in EVAL_DIRS.items():
        best_ckpt = BEST_CHECKPOINTS[version]
        eval_file = os.path.join(PROJECT_ROOT, eval_rel, f"checkpoint_{best_ckpt}.json")
        if not os.path.isfile(eval_file):
            continue
        eval_data = load_json(eval_file)
        speedups = {}
        for bench, opt_ns in eval_data.items():
            base_ns = baseline.get(bench)
            if base_ns is None or base_ns <= 0 or opt_ns is None or opt_ns <= 0:
                continue
            speedups[bench] = base_ns / opt_ns
        agent_bench_speedup[version] = speedups

    # Group by model (only model-prefixed benchmarks)
    model_groups = defaultdict(lambda: defaultdict(list))
    for bench in baseline:
        model = extract_model(bench)
        if model is None:
            continue
        for agent, bench_speedups in agent_bench_speedup.items():
            if bench in bench_speedups:
                model_groups[model][agent].append(bench_speedups[bench])

    csv_path = os.path.join(OUT_DIR, "per_model.csv")
    agents = sorted(agent_bench_speedup.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "group", "geo_mean", "count"])
        for model in sorted(model_groups.keys()):
            for agent in agents:
                speedups = model_groups[model][agent]
                if len(speedups) >= 1:
                    gm = geo_mean(speedups)
                    writer.writerow([agent, model, f"{gm:.4f}", len(speedups)])

    print(f"Per-model CSV: {csv_path}")


def main():
    baseline = load_json(os.path.join(PROJECT_ROOT, BASELINE_PATH))
    print(f"Baseline: {len(baseline)} benchmarks")

    generate_evolution_csvs(baseline)
    generate_per_op_csv(baseline)
    generate_per_model_csv(baseline)


if __name__ == "__main__":
    main()
