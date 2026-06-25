#!/usr/bin/env python3
"""
V4.9 evaluation demo — runs eval output step-by-step with delays.

Uses REAL benchmark execution times from the original V4.9-Large eval (checkpoint 3200).
Output format matches eval.py exactly.

Usage:
    python3 demo/eval_demo.py --config demo/eval_v49.json --checkpoint 3200
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"

# ── Real data from V4.9-Large checkpoint 3200 on Dalma (Intel Xeon) ──────────
BENCHMARKS = {
    "matmul_256_768_128":                         {"base_ns": 61_500_000,  "opt_ns":  7_052_967},
    "conv_2d_nchw_fchw_128_240_7_7_256_1_1_4_4":  {"base_ns": 153_000_000, "opt_ns": 21_163_283},
    "pooling_nchw_max_256_512_120_120_1_60_60":    {"base_ns": 841_700_000, "opt_ns": 286_147_122},
}

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def _blue(msg):   return f"{BLUE}{msg}{RESET}"
def _green(msg):  return f"{GREEN}{msg}{RESET}"
def _yellow(msg): return f"{YELLOW}{msg}{RESET}"

def print_info(msg):
    ts = datetime.now().strftime("%m-%d %H:%M")
    print(f"{_blue(f'{ts} - [INFO]    {msg}')}", flush=True)

def print_success(msg):
    ts = datetime.now().strftime("%m-%d %H:%M")
    print(f"{_green(f'{ts} - [SUCCESS]    {msg}')}", flush=True)


def print_header(config, checkpoint):
    now = datetime.now().strftime("%a %b %d %H:%M:%S +04 %Y")
    print("=" * 60, flush=True)
    print(f"Evaluation started at {now}", flush=True)
    print(f"Job ID:   16335713", flush=True)
    print("Version:  v4_9 (rl_autoschedular_v4_9)", flush=True)
    print(f"Config:   {config}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Node:     bn004", flush=True)
    print("=" * 60, flush=True)
    print(f"EVAL_DIR: /scratch/mb10856/MLIR-RL/demo/models", flush=True)


def print_config():
    cfg = {
        "max_num_stores_loads": 7, "max_num_loops": 7, "max_num_load_store_dim": 7,
        "num_tile_sizes": 7, "num_pad_multiples": 3, "num_unroll_factors": 3,
        "vect_size_limit": 512, "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
        "interchange_mode": "enumerate", "exploration": ["entropy"],
        "init_epsilon": 0.5, "normalize_bounds": "max", "normalize_adv": "standard",
        "reuse_experience": "none", "benchmarks_folder_path": "demo/benchmarks",
        "bench_count": 3, "replay_count": 10, "nb_iterations": 2000,
        "ppo_epochs": 4, "ppo_batch_size": 32, "value_epochs": 0,
        "value_batch_size": 32, "value_coef": 0.5, "value_clip": False,
        "entropy_coef": 0.01, "lr": 0.001, "truncate": 5, "json_file": "",
        "eval_json_file": "demo/bench_eval_base.json",
        "tags": ["single_ops", "v4_9_large", "demo"], "debug": False,
        "main_exec_data_file": "", "results_dir": "demo",
        "implementation": "rl_autoschedular_v4_9", "hardware_auto_detect": True,
        "hardware_l1_kb": 0, "hardware_l2_kb": 0, "hardware_l3_kb": 0,
        "hardware_physical_cores": 0, "hardware_logical_cores": 0,
        "hardware_simd_width": 0, "hardware_clock_mhz": 0,
        "reward_shaping_enabled": False, "reward_shaping_scale": 0.05,
        "reward_shaping_clip": 0.1, "reward_shaping_weight_ai": 1.0,
        "reward_shaping_weight_vectorizable": 0.1, "reward_shaping_weight_parallel": 0.1,
        "reward_shaping_vectorization_bonus": 0.0,
        "transformer_d_model": 256, "transformer_nhead": 8,
        "transformer_num_layers": 3, "transformer_ffn_dim": 1024,
        "transformer_dropout": 0.1, "transformer_activation": "gelu",
        "transformer_pooling": "cls", "transformer_use_action_history_token": False,
        "eval_runs": 1, "eval_aggregation": "min",
        "ppo_clip_range": 0.2, "gae_lambda": 0.95, "max_grad_norm": 0.5,
    }
    print_info(f"Using device: cpu")
    time.sleep(15)
    print_info(f"Loaded {len(BENCHMARKS)} benchmark(s): {', '.join(BENCHMARKS.keys())}")
    print_info(f"Config: {cfg}")
    print_info("Autoscheduler implementation: rl_autoschedular_v4_9")
    print_success("Logging to: demo")
    time.sleep(3)
    print_success("Model initialized")


def print_bench_table():
    time.sleep(2)
    print_info("Evaluation started...")
    bench_names = list(BENCHMARKS.keys())
    print_info(f"Benchmarks to evaluate ({len(bench_names)}): {', '.join(bench_names)}")
    print()

    header = f"{'Benchmark':<52} {'Baseline(ms)':<14} {'Optimized(ms)':<14} {'Speedup':<10} {'Outcome':<10}"
    print_info(header)

    speedups = []
    for name, d in BENCHMARKS.items():
        time.sleep(1)
        base_ms = d["base_ns"] / 1e6
        opt_ms = d["opt_ns"] / 1e6
        speedup = d["base_ns"] / d["opt_ns"]
        speedups.append(speedup)
        outcome = "OK" if speedup >= 0.8 else "SLOW"
        print(f"{_blue(f'  {name:<50} {base_ms:<14.2f} {opt_ms:<14.2f} {speedup:<10.2f}x {outcome:<10}')}")

    valid = [s for s in speedups if s > 0]
    avg = sum(valid) / len(valid) if valid else 0
    geo_mean = math.exp(sum(math.log(max(s, 1e-12)) for s in speedups) / len(speedups))
    arith_mean = sum(speedups) / len(speedups)
    num_failed = sum(1 for s in speedups if s <= 0)
    print_info(f"  ---  Average speedup (valid): {avg:.2f}x  |  Failed: {num_failed}/{len(speedups)}")

    eval_time = timedelta(seconds=601)
    print_info(f"Evaluation time: {eval_time}")

    return speedups, avg, geo_mean, arith_mean


def write_results(checkpoint, speedups, avg, geo_mean, arith_mean):
    time.sleep(1)
    ckpt_dir = DEMO_DIR / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"checkpoint_{checkpoint}_v49.json"
    ckpt_data = {name: d["opt_ns"] for name, d in BENCHMARKS.items()}
    with open(ckpt_file, "w") as f:
        json.dump(ckpt_data, f, indent=2)
    print_success(f"Saved eval results to eval/checkpoint_{checkpoint}_v49.json")

    logs_dir = ckpt_dir / "logs" / f"{checkpoint}_v49"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "final_speedup", "w") as f:
        for s in speedups:
            f.write(f"{s}\n")
    with open(logs_dir / "average_speedup", "w") as f:
        f.write(f"{geo_mean}\n")
    with open(logs_dir / "arithmetic_mean_speedup", "w") as f:
        f.write(f"{arith_mean}\n")
    print_info(f"Copied logs to eval/logs/{checkpoint}_v49/")

    eval_exec_dir = DEMO_DIR / "logs" / "eval"
    eval_exec_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_exec_dir / f"eval_exec_times_{checkpoint}.json", "w") as f:
        json.dump({name: d["opt_ns"] for name, d in BENCHMARKS.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="V4.9 evaluation demo")
    parser.add_argument("--config", default="demo/eval_v49.json", help="Path to eval config")
    parser.add_argument("--checkpoint", default="3200", help="Checkpoint number")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    checkpoint = args.checkpoint

    print(f"Submitting job: scripts/eval/eval.sh", flush=True)
    print(f"With arguments: {args.config} --checkpoint {checkpoint}", flush=True)
    print("-" * 40, flush=True)
    print(f"  Mail notifications \u2192 mehdi.benchikh1@gmail.com", flush=True)
    print(f"\u2713 Job submitted: 16335713", flush=True)
    print(f"  Job name: mlir-eval", flush=True)
    print(f"  Log file: /scratch/mb10856/MLIR-RL/logs/eval_16335713.out", flush=True)
    print(flush=True)
    print("Waiting for log file...", flush=True)
    print(f"  Waiting... (Job state: PENDING)", flush=True)
    time.sleep(2)
    print(f"  Waiting... (Job state: RUNNING)", flush=True)
    time.sleep(2)
    print(flush=True)
    print(f"\u2713 Log file created, monitoring output...", flush=True)
    print("=" * 60, flush=True)
    print("(Press Ctrl+C to stop monitoring)", flush=True)
    print(flush=True)

    print_header(config_path, checkpoint)
    print()

    print_config()
    print()

    speedups, avg, geo_mean, arith_mean = print_bench_table()
    print()

    write_results(checkpoint, speedups, avg, geo_mean, arith_mean)

    print_info("Evaluation done.")
    now = datetime.now().strftime("%a %b %d %H:%M:%S +04 %Y")
    print(f"Evaluation completed at {now}")
    print()


if __name__ == "__main__":
    main()
