"""
optimize_model_via_blocks.py
----------------------------
For models that can't be bufferized end-to-end: extracts operation blocks
from the full model, filters to compute-heavy blocks only (matmul/conv),
optimizes each independently with the RL agent, and computes weighted speedup.

Supports comparing multiple checkpoints to track training progress.

Usage:
  source ~/envs/mlir/bin/activate && set -a && source .env && set +a
  export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5 CONFIG_FILE_PATH=config/full_model_optim.json
  python scripts/optimize_model_via_blocks.py \
      --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
      --model distilbert
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.debug", override=False)

if not os.getenv("CONFIG_FILE_PATH"):
    print("Error: CONFIG_FILE_PATH not set.")
    sys.exit(1)

from utils.config import Config
from utils.implementation import get_autoschedular_impl, import_autoschedular_module

IMPL = get_autoschedular_impl()
HiearchyModel = import_autoschedular_module("model", IMPL).HiearchyModel
Execution = import_autoschedular_module("execution", IMPL).Execution
ActionSpace = import_autoschedular_module("actions", IMPL).ActionSpace
Observation = import_autoschedular_module("observation", IMPL).Observation
extract_bench_features_from_code = import_autoschedular_module("state", IMPL).extract_bench_features_from_code
OperationState = import_autoschedular_module("state", IMPL).OperationState
BenchmarkFeatures = import_autoschedular_module("state", IMPL).BenchmarkFeatures
TiledFusion = import_autoschedular_module("actions.tiled_fusion", IMPL).TiledFusion

from data_utils.extract_blocks import extract_blocks_from_file

_config = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ops that benefit from RL tiling/vectorization
COMPUTE_HEAVY_OPS = {"matmul", "batch_matmul", "conv_2d", "conv_1d", "conv_3d",
                     "batch_matmul_transpose_a", "batch_matmul_transpose_b",
                     "matmul_transpose_a", "matmul_transpose_b",
                     "batch_reduce_matmul", "depthwise_conv_2d"}


def _has_compute_heavy(code: str) -> bool:
    """Check if a block contains compute-heavy linalg ops."""
    for op in COMPUTE_HEAVY_OPS:
        if f"linalg.{op}" in code:
            return True
    return False


def infer_greedy_schedule(model, bench_features, operation_tag):
    """Run greedy RL inference for one operation."""
    operation_features = bench_features.operations[operation_tag].copy()
    for action in operation_features.pre_actions:
        operation_features = action.update_features(operation_features)

    producer_tag = None
    producer_operand_idx = None
    producer_features = None
    if operation_features.producers:
        producer_tag = operation_features.producers[-1][0]
        producer_operand_idx = min(idx for t, idx in operation_features.producers if t == producer_tag)
        producer_features = bench_features.operations[producer_tag].copy()

    try:
        bench_idx = list(bench_features.operations.keys()).index(operation_tag)
    except ValueError:
        bench_idx = 0

    state = OperationState(
        bench_idx=bench_idx, bench_name=bench_features.bench_name,
        operation_tag=operation_tag,
        original_operation_features=bench_features.operations[operation_tag].copy(),
        operation_features=operation_features,
        producer_tag=producer_tag, producer_operand_idx=producer_operand_idx,
        producer_features=producer_features,
        transformation_history=[[]], terminal=False,
    )

    while not state.terminal:
        obs = Observation.from_state(state).to(DEVICE)
        with torch.no_grad():
            actions_index, _, _ = model.sample(obs, greedy=True)
        try:
            action = ActionSpace.action_by_index(actions_index[0], state)
        except Exception:
            break
        try:
            state.record_action(action)
        except Exception:
            pass
        if isinstance(action, TiledFusion):
            try:
                action.update_producer_features(state, bench_features)
            except Exception:
                pass
        state.operation_features = action.update_features(state.operation_features)
        state.terminal = action.terminal or state.step_count >= _config.truncate

    return state.transformation_history[0]


def apply_actions_to_code(code, actions, op_tag):
    """Apply a list of actions to code."""
    result = code
    for action in actions:
        action.operation_tag = op_tag
        try:
            result = action.apply(result)
        except Exception:
            break
    return result


def measure_exec(exec_engine, code, bench_name):
    """Measure execution time in ns. Returns 0 on failure."""
    try:
        r = exec_engine.execute_code(code, bench_name, [])
        if r[1] and r[0] > 0:
            return r[0]
    except Exception:
        pass
    return 0


def process_model(model_name: str, checkpoint_paths: List[str], blocks_dir_base: str,
                  models_dir: str, window_size: int = 10, stride: int = 5,
                  skip_extraction: bool = False) -> dict:
    """Process one model across multiple checkpoints. Returns results dict."""
    model_input = Path(models_dir) / f"{model_name}_linalg.mlir"
    blocks_dir = Path(blocks_dir_base) / model_name

    # --- Step 1: Extract blocks ---
    block_files = sorted(blocks_dir.glob("*.mlir"))
    if skip_extraction and block_files:
        print(f"  Skipping extraction: {len(block_files)} blocks already exist in {blocks_dir}")
        written, skipped = len(block_files), 0
    else:
        print(f"  Extracting blocks (window={window_size}, stride={stride}) ...")
        written, skipped = extract_blocks_from_file(
            input_path=str(model_input), output_dir=str(blocks_dir),
            model_name=model_name, window_size=window_size, stride=stride,
            max_depth=10, max_paths=50,
            batch_candidates=[1, 2, 4, 8, 16, 32, 64],
            batch_fallback=None, manifest_path=None,
        )
        print(f"  Extracted {written} blocks ({skipped} skipped).")

    block_files = sorted(blocks_dir.glob("*.mlir"))
    if not block_files:
        return {"model": model_name, "error": "no blocks extracted"}

    # --- Step 2: Filter to compute-heavy blocks ---
    heavy_files = []
    skipped_generic = 0
    for bf in block_files:
        with open(bf) as f:
            code = f.read()
        if _has_compute_heavy(code):
            heavy_files.append(bf)
        else:
            skipped_generic += 1
    print(f"  Filtered: {len(heavy_files)} compute-heavy / {len(block_files)} total ({skipped_generic} generic skipped)")

    if not heavy_files:
        return {"model": model_name, "error": "no compute-heavy blocks after filter"}

    # --- Step 3: Measure baselines once (same for all checkpoints) ---
    print(f"  Measuring baseline for {len(heavy_files)} blocks ...")
    exec_engine = Execution("")
    block_baselines = {}
    for i, bf in enumerate(heavy_files):
        with open(bf) as f:
            code = f.read()
        bn = bf.stem
        baseline_ns = measure_exec(exec_engine, code, f"{model_name}_{bn}_base")
        if baseline_ns > 0:
            block_baselines[bn] = {"code": code, "baseline_ns": baseline_ns}
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(heavy_files)}] baselines done")

    valid_blocks = list(block_baselines.keys())
    total_baseline = sum(block_baselines[b]["baseline_ns"] for b in valid_blocks)
    print(f"  Baselines: {len(valid_blocks)} blocks, total={total_baseline:,} ns ({total_baseline/1e6:.2f} ms)")

    # --- Step 4: Optimize with each checkpoint ---
    checkpoint_results = {}

    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem  # e.g. model_791
        print(f"\n  --- Checkpoint: {ckpt_name} ---")

        model = HiearchyModel().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()

        total_optimized = 0
        errors = 0

        for i, block_name in enumerate(valid_blocks):
            bdata = block_baselines[block_name]
            code = bdata["code"]
            baseline = bdata["baseline_ns"]

            # Extract features
            try:
                bench_features = extract_bench_features_from_code(
                    bench_name=f"{model_name}_{block_name}", code=code,
                    root_execution_time=baseline,
                )
            except Exception:
                errors += 1
                total_optimized += baseline
                continue

            if not bench_features.operation_tags:
                total_optimized += baseline
                continue

            # Get schedules
            schedules = {}
            for tag in bench_features.operation_tags:
                try:
                    actions = infer_greedy_schedule(model, bench_features, tag)
                    schedules[tag] = actions
                except Exception:
                    schedules[tag] = []

            # Apply schedules
            opt_code = code
            for tag, actions in schedules.items():
                opt_code = apply_actions_to_code(opt_code, actions, tag)

            # Measure optimized
            opt_ns = measure_exec(exec_engine, opt_code, f"{model_name}_{block_name}_{ckpt_name}_opt")
            if opt_ns <= 0:
                opt_ns = baseline
            total_optimized += opt_ns

            if (i + 1) % 100 == 0:
                current = total_baseline / (total_optimized if total_optimized > 0 else 1)
                print(f"    [{i+1}/{len(valid_blocks)}] partial speedup={current:.4f}x ({errors} err)")

        speedup = total_baseline / total_optimized if total_optimized > 0 else 1.0
        checkpoint_results[ckpt_name] = {
            "baseline_ns": total_baseline,
            "optimized_ns": total_optimized,
            "speedup": speedup,
            "blocks_used": len(valid_blocks),
            "errors": errors,
        }
        print(f"    FINAL: {total_baseline:,} -> {total_optimized:,} = {speedup:.4f}x")

    return {
        "model": model_name,
        "method": "block_based_compute_heavy",
        "window_size": window_size,
        "stride": stride,
        "total_blocks_extracted": len(block_files),
        "compute_heavy_blocks": len(valid_blocks),
        "generic_skipped": skipped_generic,
        "total_baseline_ns": total_baseline,
        "checkpoints": checkpoint_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Block-based model optimization — multi-checkpoint compare.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint .pt files to compare")
    parser.add_argument("--model", required=True, help="Model name (stem without _linalg)")
    parser.add_argument("--models-dir", default="data/nn/raw_bench")
    parser.add_argument("--blocks-dir", default="data/nn/blocks_v10")
    parser.add_argument("--output", required=True, help="Results JSON path")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--skip-extraction", action="store_true", help="Skip block extraction if blocks already exist")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Window={args.window_size}, stride={args.stride}")
    if args.skip_extraction:
        print("Skip extraction: enabled")
    print()

    result = process_model(
        model_name=args.model,
        checkpoint_paths=args.checkpoints,
        blocks_dir_base=args.blocks_dir,
        models_dir=args.models_dir,
        window_size=args.window_size,
        stride=args.stride,
        skip_extraction=args.skip_extraction,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")

    # Summary
    if "checkpoints" in result:
        print(f"\n{'Checkpoint':<20} {'Baseline':>16} {'Optimized':>16} {'Speedup':>10} {'Blocks':>8} {'Errors':>8}")
        print("-" * 80)
        bl = result.get("total_baseline_ns", 0)
        for ckpt_name, cr in result["checkpoints"].items():
            print(f"{ckpt_name:<20} {cr['baseline_ns']:>16,} {cr['optimized_ns']:>16,} "
                  f"{cr['speedup']:>10.4f}x {cr['blocks_used']:>8} {cr['errors']:>8}")


if __name__ == "__main__":
    main()
