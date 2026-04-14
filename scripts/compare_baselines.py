#!/usr/bin/env python3
"""
Compare RL agent performance against MLIR baseline and PyTorch baselines.

Reads:
    --mlir-base       results/all/exec_times/old_base_eval.json (or new_base_eval.json)
                    {"bench_name": <ns>, ...}  (MLIR no-schedule baseline)

  --pytorch-times   results/all/exec_times/pytorch.json
                    {"bench_name": {"eager": <ns>, "compile": <ns>, "jit": <ns>}, ...}

    --rl-eval-dir     results/mehdi/old_agent/run_1/logs/eval/
                    Contains speedup/<bench_name> files (one value per checkpoint).
                    The LAST value in each file is used (from the most recent checkpoint).

Writes a CSV with one row per benchmark present in ALL sources:
  bench_name, mlir_base_ns, rl_ns,
  speedup_rl_vs_mlir,
  eager_ns, compile_ns, jit_ns,
  speedup_rl_vs_eager, speedup_rl_vs_compile, speedup_rl_vs_jit

A summary section at the bottom reports mean speedups per operation type
and overall.

Usage:
  python scripts/compare_baselines.py \\
    --mlir-base  results/all/exec_times/old_base_eval.json \\
      --pytorch-times results/all/exec_times/pytorch.json \\
    --rl-eval-dir results/mehdi/old_agent/run_1/logs/eval/ \\
      --output results/comparison.csv
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from utils.implementation import get_agent_runs_root, get_autoschedular_impl, get_base_prefix


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_mlir_base(path: str) -> dict[str, int]:
    with open(path) as f:
        data = json.load(f)
    return {k: int(v) for k, v in data.items()}


def load_pytorch_times(path: str) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)


def load_rl_speedups(eval_dir: str) -> dict[str, float]:
    """
    Read logs/eval/speedup/<bench_name> files.
    Each file has one float per line (one per evaluated checkpoint).
    Returns the LAST value (most recent checkpoint) for each benchmark.
    """
    speedup_dir = Path(eval_dir) / "speedup"
    if not speedup_dir.exists():
        raise SystemExit(f"Speedup directory not found: {speedup_dir}")

    results = {}
    for fp in speedup_dir.iterdir():
        if fp.is_file():
            lines = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
            if lines:
                results[fp.name] = float(lines[-1])
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def infer_op_type(bench_name: str) -> str:
    for op in ("conv_2d_nchw_fchw", "batch_matmul", "matmul",
               "pooling_nchw_max", "pooling_nchw_sum", "generic"):
        if op in bench_name:
            return op
    return "unknown"


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0.0:
        return None
    return round(a / b, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "bench_name", "op_type",
    "mlir_base_ns", "rl_ns", "speedup_rl_vs_mlir",
    "eager_ns", "compile_ns", "jit_ns",
    "speedup_rl_vs_eager", "speedup_rl_vs_compile", "speedup_rl_vs_jit",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL agent vs MLIR and PyTorch baselines"
    )
    parser.add_argument("config_path",   nargs="?", default=None,
                        help="Path to config JSON (positional alternative to --config)")
    parser.add_argument("--config",        default=None,
                        help="Path to config JSON (derives all paths)")
    parser.add_argument("--implementation", default=None,
                        help="Autoscheduler implementation package (default: AUTOSCHEDULER_IMPL or rl_autoschedular)")
    parser.add_argument("--mlir-base",     default=None,
                        help="Override: path to MLIR base exec times JSON")
    parser.add_argument("--pytorch-times", default=None,
                        help="Override: path to PyTorch times JSON")
    parser.add_argument("--rl-eval-dir",   default=None,
                        help="Override: path to logs/eval/ directory from eval.py run")
    parser.add_argument("--output",        default=None,
                        help="Override: output CSV path")
    args = parser.parse_args()

    config_path = args.config or args.config_path
    implementation = args.implementation or get_autoschedular_impl()

    if config_path:
        with open(config_path) as _f:
            _cfg = json.load(_f)
        exec_dir    = Path(_cfg["results_dir"]) / "exec_times"
        base_prefix = get_base_prefix(implementation)
        mlir_base_path    = args.mlir_base     or str(exec_dir / f"{base_prefix}_base_eval.json")
        pytorch_times_path= args.pytorch_times or str(exec_dir / "pytorch.json")
        output_csv        = args.output        or str(Path(_cfg["results_dir"]) / "comparison.csv")
        if args.rl_eval_dir:
            rl_eval_dir = args.rl_eval_dir
        else:
            # auto-detect latest run_N directory for selected implementation
            agent_root = get_agent_runs_root(_cfg["results_dir"], implementation)
            run_dirs = sorted(
                agent_root.glob("run_*"),
                key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0
            )
            if not run_dirs:
                raise SystemExit(f"No run_N directories found in {agent_root}")
            rl_eval_dir = str(run_dirs[-1] / "logs" / "eval")
            print(f"Auto-detected RL eval dir: {rl_eval_dir}")
    else:
        if not (args.mlir_base and args.pytorch_times and args.rl_eval_dir):
            parser.error("Provide --config, or all of --mlir-base, --pytorch-times, --rl-eval-dir")
        mlir_base_path     = args.mlir_base
        pytorch_times_path = args.pytorch_times
        rl_eval_dir        = args.rl_eval_dir
        output_csv         = args.output or "results/comparison.csv"

    # Load all three sources
    mlir_base     = load_mlir_base(mlir_base_path)
    pytorch_times = load_pytorch_times(pytorch_times_path)
    rl_speedups   = load_rl_speedups(rl_eval_dir)

    # Intersection of benchmarks across all sources
    common = set(mlir_base) & set(pytorch_times) & set(rl_speedups)
    if not common:
        raise SystemExit(
            "No benchmarks found in all three sources. "
            "Check that paths are correct and that eval.py has been run."
        )
    print(f"Benchmarks in all sources: {len(common)}")

    rows = []
    for name in sorted(common):
        base_ns  = mlir_base[name]
        speedup  = rl_speedups[name]
        rl_ns    = int(base_ns / speedup) if speedup != 0 else None

        pt       = pytorch_times[name]
        eager_ns   = pt.get("eager")
        compile_ns = pt.get("compile")
        jit_ns     = pt.get("jit")

        rows.append({
            "bench_name":            name,
            "op_type":               infer_op_type(name),
            "mlir_base_ns":          base_ns,
            "rl_ns":                 rl_ns,
            "speedup_rl_vs_mlir":    round(speedup, 4),
            "eager_ns":              eager_ns,
            "compile_ns":            compile_ns,
            "jit_ns":                jit_ns,
            "speedup_rl_vs_eager":   safe_div(eager_ns,   rl_ns),
            "speedup_rl_vs_compile": safe_div(compile_ns, rl_ns),
            "speedup_rl_vs_jit":     safe_div(jit_ns,     rl_ns),
        })

    # Write CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_path}")

    # -----------------------------------------------------------------------
    # Summary statistics per op type
    # -----------------------------------------------------------------------
    def _mean(vals):
        valid = [v for v in vals if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else None

    by_op: dict[str, list] = defaultdict(list)
    for r in rows:
        by_op[r["op_type"]].append(r)

    print("\n--- Mean speedups (RL / baseline) per operation type ---")
    speedup_cols = [
        ("vs_mlir",   "speedup_rl_vs_mlir"),
        ("vs_eager",  "speedup_rl_vs_eager"),
        ("vs_compile","speedup_rl_vs_compile"),
        ("vs_jit",    "speedup_rl_vs_jit"),
    ]
    header = f"{'op_type':<28} {'count':>5}  " + \
             "  ".join(f"{lbl:>12}" for lbl, _ in speedup_cols)
    print(header)
    print("-" * len(header))

    global_rows = []
    for op_type in sorted(by_op):
        op_rows = by_op[op_type]
        global_rows.extend(op_rows)
        counts = len(op_rows)
        means  = [_mean([r[col] for r in op_rows]) for _, col in speedup_cols]
        parts  = "  ".join(f"{(str(m) if m else 'N/A'):>12}" for m in means)
        print(f"{op_type:<28} {counts:>5}  {parts}")

    # Overall
    overall_means = [_mean([r[col] for r in global_rows]) for _, col in speedup_cols]
    parts = "  ".join(f"{(str(m) if m else 'N/A'):>12}" for m in overall_means)
    print("-" * len(header))
    print(f"{'OVERALL':<28} {len(global_rows):>5}  {parts}")


if __name__ == "__main__":
    main()
