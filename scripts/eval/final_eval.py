#!/usr/bin/env python3
"""Final-checkpoint evaluation: N runs per checkpoint, median speedup aggregation.

Runs AFTER cleanup-checkpoints. Evaluates the surviving model_*.pt checkpoints in
<results_dir>/models/ (the top-9 KEEP set). For each checkpoint it runs eval.py
`num_runs` times and, per benchmark, records the median of the per-run speedups
(baseline_time / exec_time). Writes one json per checkpoint into
<results_dir>/eval_final/checkpoint_<n>.json = {bench_name: median_speedup}.

Reads a dedicated final-eval config (num_runs + aggregator live there explicitly).

Usage:
  export CONFIG_FILE_PATH=<path to the agent's base eval config>   # eval.py needs it
  python scripts/eval/final_eval.py <final_eval_config.json> [--dry-run]
  python scripts/eval/final_eval.py <config> --self-test
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Reuse the dataset baseline loader already used to build ranking CSVs.
from utils.csvs import load_baseline, benchmark_group


def geo_mean(speedups) -> float | None:
    vals = [float(s) for s in speedups if s and s > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def write_best_checkpoint_csvs(results_dir: str) -> int | None:
    """Pick the best checkpoint from eval_final/ (median results) and write the
    best-checkpoint CSVs the plotting skill consumes:
      csvs/best_checkpoint_speedups.csv
      csvs/best_checkpoint_benchmark_family_results.csv
      csvs/best_checkpoint_operation_type_results.csv
    Returns the best checkpoint number, or None if eval_final/ has no usable jsons.
    """
    eval_final = os.path.join(results_dir, "eval_final")
    if not os.path.isdir(eval_final):
        return None
    best_ckpt, best_gm = None, -1.0
    best_data = None
    for f in os.listdir(eval_final):
        m = re.match(r"checkpoint_(\d+)\.json$", f)
        if not m:
            continue
        with open(os.path.join(eval_final, f)) as fh:
            data = json.load(fh)
        gm = geo_mean(data.values())
        if gm is not None and gm > best_gm:
            best_ckpt, best_gm, best_data = int(m.group(1)), gm, data
    if best_ckpt is None:
        return None

    agent = os.path.basename(os.path.normpath(results_dir))
    if agent.endswith("_agent"):
        agent = agent[:-6]

    csvs = os.path.join(results_dir, "csvs")
    os.makedirs(csvs, exist_ok=True)

    # best_checkpoint_speedups.csv
    with open(os.path.join(csvs, "best_checkpoint_speedups.csv"), "w") as f:
        f.write("checkpoint,speedup\n")
        f.write(f"{best_ckpt},{best_gm:.4f}\n")

    # group the best checkpoint's median speedups by model family / op type
    fam_sp, op_sp = {}, {}
    for bench, sp in best_data.items():
        if not sp or sp <= 0:
            continue
        kind, group = benchmark_group(bench)
        if kind == "model":
            fam_sp.setdefault(group, []).append(sp)
        elif kind == "op":
            op_sp.setdefault(group, []).append(sp)

    def write_grouped(groups, spec):
        order = sorted(groups, key=lambda g: geo_mean(groups[g]) or 0, reverse=True)
        with open(os.path.join(csvs, spec), "w") as f:
            f.write("benchmark_family,agent_version,speedup\n")
            for g in order:
                f.write(f"{g},{agent},{geo_mean(groups[g]):.4f}\n")

    write_grouped(fam_sp, "best_checkpoint_benchmark_family_results.csv")
    write_grouped(op_sp, "best_checkpoint_operation_type_results.csv")
    return best_ckpt


def median_speedup_per_benchmark(baseline: dict, run_times: list[dict]) -> dict:
    """Aggregate N per-run exec-time dicts into {bench: median_speedup}.

    For each benchmark, compute baseline/exec_time per run (skipping None/0 runs)
    and take the median. If no valid run, the benchmark maps to None.
    """
    benches = set().union(*(r.keys() for r in run_times)) if run_times else set()
    out: dict = {}
    for b in benches:
        sps = [
            baseline[b] / r[b]
            for r in run_times
            if baseline.get(b, 0) > 0 and r.get(b) and r[b] > 0
        ]
        out[b] = statistics.median(sps) if sps else None
    return out


def _run_eval_env(config_path: str, results_dir: str, ckpt: int) -> dict:
    """Env for one eval.py invocation targeting exactly one checkpoint."""
    eval_dir = os.path.join(results_dir, "models")
    env = dict(os.environ)
    env["CONFIG_FILE_PATH"] = config_path
    env["EVAL_DIR"] = eval_dir
    env["EVAL_START"] = str(ckpt)
    env["EVAL_END"] = str(ckpt)
    env["EVAL_STRIDE"] = "1"
    env["EVAL_FORCE"] = "1"  # reset per-run completion file so each run re-evaluates
    return env


def get_surviving_checkpoints(results_dir: str) -> list[int]:
    models_dir = os.path.join(results_dir, "models")
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        int(m.group(1)) for f in os.listdir(models_dir)
        if (m := re.match(r"model_(\d+)\.pt$", f))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Final N-run median checkpoint evaluation")
    ap.add_argument("config", help="Path to the final-eval config json")
    ap.add_argument("--dry-run", action="store_true", help="List checkpoints/runs, don't evaluate")
    ap.add_argument("--self-test", action="store_true", help="Verify aggregator on synthetic data")
    args = ap.parse_args()

    if args.self_test:
        baseline = {"a": 100.0, "b": 200.0, "c": 0.0}
        # bench 'a': per-run [2,2,3,3,100] -> median 3 ; bench 'b': [1 x5] -> 1 ; 'c' baseline 0 -> None
        t3 = 100.0 / 3.0
        runs = [
            {"a": 50.0, "b": 200.0, "c": 10.0},
            {"a": 50.0, "b": 200.0, "c": 10.0},
            {"a": t3, "b": 200.0, "c": 10.0},
            {"a": t3, "b": 200.0, "c": 10.0},
            {"a": 1.0, "b": 200.0, "c": 10.0},
        ]
        med = median_speedup_per_benchmark(baseline, runs)
        assert abs(med["a"] - 3.0) < 1e-6, med
        assert abs(med["b"] - 1.0) < 1e-6, med
        assert med["c"] is None, med
        print(f"self-test PASS: {med}")
        return 0

    cfg = json.load(open(args.config))
    num_runs = int(cfg["num_runs"])
    assert cfg["aggregator"] == "median", "only 'median' aggregator supported"
    config_path = os.path.join(PROJECT_ROOT, cfg["eval_config"])
    results_dir = os.path.join(PROJECT_ROOT, cfg["results_dir"])
    baseline = load_baseline(cfg["dataset"])

    ckpts = get_surviving_checkpoints(results_dir)
    print(f"checkpoints on disk: {len(ckpts)} -> {ckpts}")
    print(f"num_runs: {num_runs} | aggregator: {cfg['aggregator']} | total eval invocations: {len(ckpts) * num_runs}")

    if args.dry_run:
        print("DRY RUN — nothing evaluated.")
        return 0

    eval_final = os.path.join(results_dir, "eval_final")
    os.makedirs(eval_final, exist_ok=True)

    for ckpt in ckpts:
        run_times = []
        for run in range(1, num_runs + 1):
            print(f"  ckpt {ckpt} run {run}/{num_runs} ...")
            r = subprocess.run(
                [sys.executable, "scripts/eval/eval.py"],
                cwd=PROJECT_ROOT, env=_run_eval_env(config_path, results_dir, ckpt),
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"    eval.py failed (exit {r.returncode}) — aborting ckpt {ckpt}")
                print(r.stderr[-2000:])
                return 1
            # eval.py's per-run exec times land at <logs>/eval/exec_times (single-run shape).
            src = os.path.join(results_dir, "logs", "eval", "eval_exec_times.json")
            with open(src) as f:
                run_times.append(json.load(f))
        out_file = os.path.join(eval_final, f"checkpoint_{ckpt}.json")
        with open(out_file, "w") as f:
            json.dump(median_speedup_per_benchmark(baseline, run_times), f, indent=2)
        print(f"  wrote {out_file}")

    best = write_best_checkpoint_csvs(results_dir)
    if best is not None:
        print(f"BEST CHECKPOINT (median): {best} -> csvs/best_checkpoint_speedups.csv")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())