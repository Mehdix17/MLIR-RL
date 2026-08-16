#!/usr/bin/env python3
"""CSV generation for MLIR-RL experiment results — separate from plots logic.

Produces per-agent CSVs inside `<results_dir>/csvs/`:
  - checkpoint_speedups.csv                 (checkpoint, speedup) — ranking per checkpoint
  - best_checkpoint_speedups.csv            (checkpoint, speedup) — champion checkpoint, 1 row
  - best_checkpoint_benchmark_family_results.csv (benchmark_family, agent_version, speedup)
  - best_checkpoint_operation_type_results.csv   (benchmark_family, agent_version, speedup)

The eval script updates checkpoint_speedups.csv automatically after each eval
(update_checkpoint_speedup); this module's CLI rebuilds all CSVs from eval jsons
(backfill / one-shot generation).

Usage:
  python scripts/utils/csvs.py --results-dir <dir> --agent <name> --baseline <json> [--all]
"""
import argparse
import csv
import json
import math
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_BASELINES = {
    "new": "results/new_dataset_results/baselines/mlir/eval_base.json",
    "single_ops": "results/single_ops_dataset_results/baselines/mlir/base_eval.json",
    "ops_and_blocks": "results/ops_and_blocks_results/baselines/mlir/base_eval.json",
    "legacy_paper": "results/legacy_paper_results/baselines/mlir/base_eval.json",
}


def load_baseline(dataset: str) -> dict:
    path = os.path.join(PROJECT_ROOT, DATASET_BASELINES[dataset])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline not found: {path}")
    with open(path) as f:
        return json.load(f)


def compute_geo_mean_speedup(eval_data: dict, baseline: dict) -> float | None:
    speedups = [
        baseline[b] / eval_data[b]
        for b in eval_data
        if baseline.get(b, 0) > 0 and eval_data[b] is not None and eval_data[b] > 0
    ]
    if not speedups:
        return None
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def csv_path(results_dir: str, stem: str) -> str:
    return os.path.join(results_dir, "csvs", f"{stem}.csv")


def get_checkpoint_files(eval_dir: str) -> list[str]:
    if not os.path.isdir(eval_dir):
        return []
    return sorted(
        (f for f in os.listdir(eval_dir) if re.match(r"checkpoint_\d+\.json$", f)),
        key=lambda f: int(re.search(r"checkpoint_(\d+)\.json", f).group(1)),
    )


def update_checkpoint_speedup(results_dir: str, checkpoint: int, speedup: float) -> str:
    """Upsert one row into <results_dir>/csvs/checkpoint_speedups.csv (columns: checkpoint,speedup)."""
    path = csv_path(results_dir, "checkpoint_speedups")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2 and parts[0] != str(checkpoint):
                    rows.append(line.rstrip("\n"))
    else:
        header = "checkpoint,speedup\n"
    rows.append(f"{checkpoint},{speedup:.4f}")
    with open(path, "w", newline="") as f:
        f.write(header)
        f.write("\n".join(rows) + "\n")
    return path


def rebuild_checkpoint_speedups(results_dir: str, baseline: dict) -> int:
    """Regenerate checkpoint_speedups.csv from the eval jsons (idempotent). Returns row count."""
    eval_dir = os.path.join(results_dir, "eval")
    rows = []
    for cf in get_checkpoint_files(eval_dir):
        ckpt = int(re.search(r"checkpoint_(\d+)\.json", cf).group(1))
        try:
            with open(os.path.join(eval_dir, cf)) as f:
                eval_data = json.load(f)
        except Exception:
            continue
        gm = compute_geo_mean_speedup(eval_data, baseline)
        if gm is not None:
            rows.append([ckpt, f"{gm:.4f}"])
    path = csv_path(results_dir, "checkpoint_speedups")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "speedup"])
        writer.writerows(rows)
    return len(rows)


MODEL_PREFIXES = {"albert", "bart", "bert", "convnext_tiny", "distilbert", "efficientnet_b0",
                  "gat", "gin", "gpt2", "llama3_2_1b", "mobilenet_v3_small", "resnet50",
                  "resnext50", "t5", "vgg16", "vit_b_16", "whisper_base", "yolov8m"}
OP_TYPES = {"add", "conv_2d", "matmul", "pooling", "relu"}


def benchmark_group(bench_name: str) -> tuple[str, str]:
    """Group a benchmark into ('model', <model>) | ('op', <op_type>) | ('unknown', name).

    Model benchmarks start with the model prefix (albert_*, llama3_2_1b_*, ...); synthetic
    op benchmarks start with the op name (add_*, conv_2d_*, matmul_*, ...). Longest prefix
    first — model names contain underscores (llama3_2_1b, mobilenet_v3_small, ...).
    """
    for p in sorted(MODEL_PREFIXES, key=len, reverse=True):
        if bench_name.startswith(p + "_"):
            return ("model", p)
    for p in OP_TYPES:
        if bench_name.startswith(p + "_"):
            return ("op", p)
    return ("unknown", bench_name)


def generate_comparison_csv(
    results_dir: str,
    agent: str,
    baseline: dict,
    filter_type: str,   # "models_only" | "ops_only"
    exclude: list,
) -> str:
    """Per-agent best-checkpoint comparison CSV (best_checkpoint_benchmark_family_results.csv /
    best_checkpoint_operation_type_results.csv), grouped by model family / op type."""
    eval_dir = os.path.join(results_dir, "eval")
    ckpt_files = get_checkpoint_files(eval_dir)
    if not ckpt_files:
        return ""
    # best checkpoint = max geo-mean speedup across evaluated checkpoints
    best_ckpt, best_gm = None, -1.0
    for cf in ckpt_files:
        ckpt = int(re.search(r"checkpoint_(\d+)\.json", cf).group(1))
        try:
            with open(os.path.join(eval_dir, cf)) as f:
                eval_data = json.load(f)
        except Exception:
            continue
        gm = compute_geo_mean_speedup(eval_data, baseline)
        if gm is not None and gm > best_gm:
            best_ckpt, best_gm = ckpt, gm
    if best_ckpt is None:
        return ""
    with open(os.path.join(eval_dir, f"checkpoint_{best_ckpt}.json")) as f:
        eval_data = json.load(f)

    exclude_lower = {e.lower() for e in exclude}
    group_speedups: dict[str, list] = {}
    for bench_name, opt_ns in eval_data.items():
        kind, group = benchmark_group(bench_name)
        if kind == "unknown":
            continue
        if filter_type == "models_only" and kind != "model":
            continue
        if filter_type == "ops_only" and kind != "op":
            continue
        if any(x in group for x in exclude_lower):
            continue
        root = baseline.get(bench_name, 0)
        if root <= 0 or not opt_ns or opt_ns <= 0:
            continue
        group_speedups.setdefault(group, []).append(root / opt_ns)

    if filter_type == "models_only":
        stem = "best_checkpoint_benchmark_family_results"
    else:
        stem = "best_checkpoint_operation_type_results"
    rows = [[g, agent, f"{math.exp(sum(math.log(s) for s in sps) / len(sps)):.4f}"]
            for g, sps in sorted(
                group_speedups.items(),
                key=lambda kv: math.exp(sum(math.log(s) for s in kv[1]) / len(kv[1])),
                reverse=True)]
    path = csv_path(results_dir, stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark_family", "agent_version", "speedup"])
        writer.writerows(rows)
    return path


def rebuild_best_checkpoint(results_dir: str) -> int:
    """Write best_checkpoint_speedups.csv (single row: champion checkpoint + speedup)."""
    src = os.path.join(results_dir, "csvs", "checkpoint_speedups.csv")
    path = csv_path(results_dir, "best_checkpoint_speedups")
    best = None
    if os.path.isfile(src):
        with open(src) as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        ckpt, sp = int(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    if best is None or sp > best[1]:
                        best = (ckpt, sp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("checkpoint,speedup\n")
        if best:
            f.write(f"{best[0]},{best[1]:.4f}\n")
    return 1 if best else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild per-agent result CSVs from eval jsons")
    ap.add_argument("--results-dir", required=True, help="experiment results dir (e.g. results/.../v5_single_node_agent)")
    ap.add_argument("--agent", required=True, help="agent_version value written into the CSVs")
    ap.add_argument("--dataset", default="ops_and_blocks", choices=list(DATASET_BASELINES))
    ap.add_argument("--baseline", default=None, help="explicit baseline json (default: per-dataset)")
    ap.add_argument("--all", action="store_true", help="also rebuild comparison CSVs (needs benchmark-family map)")
    args = ap.parse_args()

    baseline = json.load(open(args.baseline)) if args.baseline else load_baseline(args.dataset)
    n = rebuild_checkpoint_speedups(args.results_dir, baseline)
    print(f"checkpoint_speedups.csv: {n} rows -> {csv_path(args.results_dir, 'checkpoint_speedups')}")
    m = rebuild_best_checkpoint(args.results_dir)
    print(f"best_checkpoint_speedups.csv: {'1 row' if m else 'no data'} -> {csv_path(args.results_dir, 'best_checkpoint_speedups')}")
    if args.all:
        p1 = generate_comparison_csv(args.results_dir, args.agent, baseline, "models_only", [])
        p2 = generate_comparison_csv(args.results_dir, args.agent, baseline, "ops_only", [])
        print(f"best_checkpoint_benchmark_family_results.csv: {p1}")
        print(f"best_checkpoint_operation_type_results.csv:  {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
