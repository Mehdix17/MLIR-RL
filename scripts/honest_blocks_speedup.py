"""
honest_blocks_speedup.py
------------------------
Computes honest model-level speedup for bufferization-crash models.
Measures baseline for ALL blocks (heavy + generic), reports:
  - heavy-only speedup (existing)
  - honest speedup  = (heavy_baseline + generic_baseline) / (heavy_optimized + generic_baseline)

Usage:
  source ~/envs/mlir/bin/activate && set -a && source .env && set +a
  export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5 CONFIG_FILE_PATH=config/full_model_optim.json
  python scripts/honest_blocks_speedup.py --models albert bert --output results/full_model/honest_blocks.json
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.debug", override=False)

if not os.getenv("CONFIG_FILE_PATH"):
    print("Error: CONFIG_FILE_PATH not set.")
    sys.exit(1)

from utils.implementation import get_autoschedular_impl, import_autoschedular_module

IMPL = get_autoschedular_impl()
Execution = import_autoschedular_module("execution", IMPL).Execution

COMPUTE_HEAVY_OPS = {"matmul", "batch_matmul", "conv_2d", "conv_1d", "conv_3d",
                     "batch_matmul_transpose_a", "batch_matmul_transpose_b",
                     "matmul_transpose_a", "matmul_transpose_b",
                     "batch_reduce_matmul", "depthwise_conv_2d"}

def is_compute_heavy(code: str) -> bool:
    for op in COMPUTE_HEAVY_OPS:
        if f"linalg.{op}" in code:
            return True
    return False

def measure_baseline(code: str, name: str) -> int:
    try:
        exec_engine = Execution("")
        r = exec_engine.execute_code(code, name, [])
        if r[1] and r[0] > 0:
            return r[0]
    except Exception:
        pass
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--blocks-dir", default="data/nn/blocks_v10")
    parser.add_argument("--v10-results-dir", default="results/full_model")
    parser.add_argument("--output", default="results/full_model/honest_blocks.json")
    args = parser.parse_args()

    blocks_dir = Path(args.blocks_dir)
    v10_dir = Path(args.v10_results_dir)

    results = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # Load v10 results for heavy-optimized times
        v10_path = v10_dir / f"blocks_v10_{model_name}.json"
        if not v10_path.exists():
            print(f"  No v10 results for {model_name}, skipping.")
            continue
        with open(v10_path) as f:
            v10 = json.load(f)

        if "error" in v10:
            print(f"  v10 error: {v10['error']}")
            continue

        cps = v10.get("checkpoints", {})
        if not cps:
            print(f"  No checkpoints in v10 results.")
            continue

        heavy_baseline = next(iter(cps.values()))["baseline_ns"]
        heavy_optimized = next(iter(cps.values()))["optimized_ns"]
        checkpoint_name = next(iter(cps.keys()))

        # Read all block files
        model_blocks_dir = blocks_dir / model_name
        block_files = sorted(model_blocks_dir.glob("*.mlir"))
        print(f"  Total blocks: {len(block_files)}")

        # Separate heavy vs generic
        heavy_files = []
        generic_files = []
        for bf in block_files:
            with open(bf) as f:
                code = f.read()
            if is_compute_heavy(code):
                heavy_files.append(bf)
            else:
                generic_files.append(bf)

        print(f"  Heavy: {len(heavy_files)}, Generic: {len(generic_files)}")

        # Measure generic block baselines
        print(f"  Measuring {len(generic_files)} generic baselines ...")
        generic_baseline_total = 0
        generic_errors = 0
        generic_results = {}

        for i, bf in enumerate(generic_files):
            with open(bf) as f:
                code = f.read()
            bn = bf.stem
            bl = measure_baseline(code, f"{model_name}_{bn}_generic_base")
            if bl > 0:
                generic_baseline_total += bl
                generic_results[bn] = bl
            else:
                generic_errors += 1
            if (i + 1) % 200 == 0:
                print(f"    [{i+1}/{len(generic_files)}] done")

        print(f"  Generic baseline total: {generic_baseline_total:,} ns ({generic_baseline_total/1e6:.2f} ms), {generic_errors} errors")

        # Compute honest speedup
        total_baseline = heavy_baseline + generic_baseline_total
        total_optimized = heavy_optimized + generic_baseline_total
        honest_speedup = total_baseline / total_optimized if total_optimized > 0 else 1.0
        heavy_speedup = heavy_baseline / heavy_optimized if heavy_optimized > 0 else 1.0

        print(f"\n  Heavy speedup: {heavy_speedup:.4f}x")
        print(f"  Honest speedup (with generics): {honest_speedup:.4f}x")
        print(f"  Heavy pct of total baseline: {heavy_baseline/total_baseline*100:.1f}%")

        results[model_name] = {
            "model": model_name,
            "checkpoint": checkpoint_name,
            "total_blocks": len(block_files),
            "heavy_blocks": len(heavy_files),
            "generic_blocks": len(generic_files),
            "generic_errors": generic_errors,
            "heavy_baseline_ns": heavy_baseline,
            "heavy_optimized_ns": heavy_optimized,
            "generic_baseline_ns": generic_baseline_total,
            "total_baseline_ns": total_baseline,
            "total_optimized_ns": total_optimized,
            "heavy_speedup": heavy_speedup,
            "honest_speedup": honest_speedup,
            "heavy_pct_of_baseline": heavy_baseline / total_baseline * 100 if total_baseline > 0 else 0,
        }

        # Save incremental
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {args.output}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Heavy%':>8} {'HeavySp':>10} {'HonestSp':>10} {'Diff':>8}")
    print("-" * 56)
    for m in args.models:
        if m in results:
            r = results[m]
            diff = (r["heavy_speedup"] - r["honest_speedup"]) / r["honest_speedup"] * 100
            print(f"{m:<20} {r['heavy_pct_of_baseline']:>7.1f}% {r['heavy_speedup']:>10.4f}x {r['honest_speedup']:>10.4f}x {diff:>+7.1f}%")

if __name__ == "__main__":
    main()
