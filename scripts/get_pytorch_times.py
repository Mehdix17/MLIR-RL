#!/usr/bin/env python3
"""
Measure PyTorch execution times (eager, torch.compile, torch.jit) for each
benchmark in a directory of .mlir files.

When called with --config, only the eval split (eval_json_file) is measured —
pytorch times are only used for comparison against RL eval results, so measuring
train benchmarks would waste time.

For each .mlir file the script:
  1. Parses the MLIR to extract operation type and tensor shapes.
  2. Builds the equivalent PyTorch operation.
  3. Measures wall-clock time (nanoseconds) with warm-up, using:
       - eager  : plain PyTorch
       - compile: torch.compile  (requires torch >= 2.0, skipped otherwise)
       - jit    : torch.jit.trace

Output JSON format (matches get_base.py flat format per mode):
  {
    "bench_name": {"eager": <ns>, "compile": <ns>, "jit": <ns>},
    ...
  }

Usage:
  python scripts/get_pytorch_times.py \\
      --benchmarks-dir data/all \\
      --output results/all/exec_times/pytorch.json   # default if omitted
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

NUM_WARMUP = 10
NUM_TRIALS = 20
HAS_COMPILE = hasattr(torch, "compile")


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def measure_ns(fn: Callable, warmup: int = NUM_WARMUP, trials: int = NUM_TRIALS) -> int:
    for _ in range(warmup):
        fn()
    start = time.perf_counter_ns()
    for _ in range(trials):
        fn()
    end = time.perf_counter_ns()
    return (end - start) // trials


# ---------------------------------------------------------------------------
# MLIR parsing helpers
# ---------------------------------------------------------------------------

def _normalize(code: str) -> str:
    return re.sub(r'\s+', ' ', code)


def parse_tensor_shape(type_str: str) -> tuple[int, ...]:
    """'tensor<1x64x64x64xf32>' -> (1, 64, 64, 64)"""
    m = re.search(r'tensor<([\dx]+)x\w+>', type_str)
    if not m:
        return ()
    return tuple(int(d) for d in m.group(1).split('x'))


def parse_main_args(code: str) -> list[tuple[int, ...]]:
    """Return shapes of every %argN in the @main function signature."""
    norm = _normalize(code)
    m = re.search(r'func\.func @main\((.+?)\)\s*->', norm)
    if not m:
        return []
    sig = m.group(1)
    shapes = []
    for part in re.split(r'%arg\d+\s*:', sig):
        s = parse_tensor_shape(part)
        if s:
            shapes.append(s)
    return shapes


def get_linalg_op(code: str) -> Optional[str]:
    m = re.search(r'linalg\.(\w+)', code)
    return m.group(1) if m else None


def _parse_dense(code: str, attr: str) -> tuple[int, int]:
    """Parse strides= or dilations= from MLIR attributes."""
    # dense<N> -> (N, N)
    m = re.search(rf'{attr}\s*=\s*dense<(\d+)>', code)
    if m:
        v = int(m.group(1))
        return (v, v)
    # dense<[M, N]> -> (M, N)
    m = re.search(rf'{attr}\s*=\s*dense<\[(\d+),\s*(\d+)\]>', code)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (1, 1)


# ---------------------------------------------------------------------------
# PyTorch function builder
# ---------------------------------------------------------------------------

def build_pytorch_fns(
    mlir_code: str,
) -> tuple[Optional[dict[str, Callable]], Optional[str]]:
    """
    Parse an MLIR benchmark and return a dict of callables:
        {"eager": fn, "compile": fn, "jit": fn}
    Returns (None, error_message) on failure.
    """
    op = get_linalg_op(mlir_code)
    args = parse_main_args(mlir_code)

    if not op:
        return None, "linalg op not found"
    if not args:
        return None, "cannot parse @main args"

    # -----------------------------------------------------------------------
    # conv_2d_nchw_fchw
    # -----------------------------------------------------------------------
    if op == 'conv_2d_nchw_fchw':
        if len(args) < 2:
            return None, "not enough args for conv2d"
        inp    = torch.randn(*args[0])
        weight = torch.randn(*args[1])
        strides   = _parse_dense(mlir_code, 'strides')
        dilations = _parse_dense(mlir_code, 'dilations')

        def eager_fn():
            return F.conv2d(inp, weight, stride=strides, dilation=dilations)

        def _conv_for_trace(x):
            return F.conv2d(x, weight, stride=strides, dilation=dilations)

        jit_fn  = torch.jit.trace(_conv_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # matmul  (2-D)
    # -----------------------------------------------------------------------
    elif op == 'matmul':
        if len(args) < 2:
            return None, "not enough args for matmul"
        a = torch.randn(*args[0])
        b = torch.randn(*args[1])

        def eager_fn():
            return torch.mm(a, b)

        def _mm_for_trace(x, y):
            return torch.mm(x, y)

        jit_fn  = torch.jit.trace(_mm_for_trace, (a, b))
        jit_run = lambda: jit_fn(a, b)

    # -----------------------------------------------------------------------
    # batch_matmul  (3-D)
    # -----------------------------------------------------------------------
    elif op == 'batch_matmul':
        if len(args) < 2:
            return None, "not enough args for batch_matmul"
        a = torch.randn(*args[0])
        b = torch.randn(*args[1])

        def eager_fn():
            return torch.bmm(a, b)

        def _bmm_for_trace(x, y):
            return torch.bmm(x, y)

        jit_fn  = torch.jit.trace(_bmm_for_trace, (a, b))
        jit_run = lambda: jit_fn(a, b)

    # -----------------------------------------------------------------------
    # pooling_nchw_max / pooling_nchw_sum
    # -----------------------------------------------------------------------
    elif op in ('pooling_nchw_max', 'pooling_nchw_sum'):
        if len(args) < 2:
            return None, "not enough args for pooling"
        inp    = torch.randn(*args[0])
        kernel = args[1]          # shape (kH, kW) from the kernel tensor arg
        strides = _parse_dense(mlir_code, 'strides')

        if op == 'pooling_nchw_max':
            def eager_fn():
                return F.max_pool2d(inp, kernel_size=kernel, stride=strides)

            def _pool_for_trace(x):
                return F.max_pool2d(x, kernel_size=kernel, stride=strides)
        else:
            def eager_fn():
                return F.avg_pool2d(inp, kernel_size=kernel, stride=strides)

            def _pool_for_trace(x):
                return F.avg_pool2d(x, kernel_size=kernel, stride=strides)

        jit_fn  = torch.jit.trace(_pool_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # generic  (reduction — treat as torch.sum on reduction dims)
    # -----------------------------------------------------------------------
    elif op == 'generic':
        if len(args) < 2:
            return None, "not enough args for generic"
        in_shape  = args[0]
        out_shape = args[-1]
        inp = torch.randn(*in_shape)
        reduce_dims = [i for i, (si, so) in enumerate(zip(in_shape, out_shape)) if so == 1]

        if reduce_dims:
            def eager_fn():
                return inp.sum(dim=reduce_dims, keepdim=True)

            def _sum_for_trace(x):
                return x.sum(dim=reduce_dims, keepdim=True)
        else:
            def eager_fn():
                return inp.sum()

            def _sum_for_trace(x):
                return x.sum()

        jit_fn  = torch.jit.trace(_sum_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    else:
        return None, f"unsupported linalg op: {op}"

    # -----------------------------------------------------------------------
    # torch.compile wrapper  (optional)
    # -----------------------------------------------------------------------
    if HAS_COMPILE:
        compiled = torch.compile(eager_fn)
        compile_run = lambda: compiled()
    else:
        compile_run = None

    return {
        "eager":   eager_fn,
        "compile": compile_run,
        "jit":     jit_run,
    }, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure PyTorch eager/compile/jit times for MLIR benchmarks"
    )
    parser.add_argument("--require-base-success", action="store_true", help="Only run benchmarks that have > 0 time in base.json")
    parser.add_argument("--config", default=None,
                        help="Path to config JSON (derives benchmarks-dir and output)")
    parser.add_argument("--benchmarks-dir", default=None,
                        help="Override: directory containing .mlir benchmark files")
    parser.add_argument("--output", default=None,
                        help="Override: output JSON path")
    parser.add_argument("--warmup",  type=int, default=NUM_WARMUP)
    parser.add_argument("--trials",  type=int, default=NUM_TRIALS)
    args = parser.parse_args()

    eval_names: set[str] | None = None  # if set, only measure these benchmarks

    if args.config:
        with open(args.config) as _f:
            _cfg = json.load(_f)
        bench_dir   = Path(args.benchmarks_dir or _cfg["benchmarks_folder_path"])
        output_path = Path(args.output) if args.output else \
                      Path(_cfg["results_dir"]) / "exec_times" / "pytorch.json"
        # Only measure eval benchmarks when eval_json_file is provided.
        # Otherwise measure all benchmarks.
        eval_json_str = _cfg.get("eval_json_file", "").strip()
        if eval_json_str:
            eval_json = Path(eval_json_str)
            if eval_json.exists():
                with open(eval_json) as _f:
                    eval_names = set(json.load(_f).keys())
            else:
                eval_names = None
        else:
            eval_names = None
    else:
        if not args.benchmarks_dir:
            parser.error("Provide --config or --benchmarks-dir")
        bench_dir = Path(args.benchmarks_dir)
        output_path = Path(args.output) if args.output else \
                      Path("results") / bench_dir.name / "exec_times" / "pytorch.json"

    if not bench_dir.is_dir():
        raise SystemExit(f"Not a directory: {bench_dir}")

    all_mlir_files = sorted(bench_dir.glob("*.mlir"))
    if eval_names is not None:
        mlir_files = [f for f in all_mlir_files if f.stem in eval_names]
        print(f"Found {len(all_mlir_files)} .mlir files in {bench_dir}; "
              f"measuring {len(mlir_files)} eval benchmarks only")
    else:
        mlir_files = all_mlir_files
        print(f"Found {len(mlir_files)} .mlir files in {bench_dir}")
    if HAS_COMPILE:
        # Compute nodes don't have g++ in PATH; use clang++ from the LLVM build instead.
        try:
            import torch._inductor.config as _inductor_cfg
            _inductor_cfg.cpp.cxx = "clang++"
        except Exception:
            pass
    else:
        print("torch.compile not available (requires torch >= 2.0) — compile times will be null")

    results: dict[str, dict] = {}
    skipped = 0

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            print(f"Resuming from existing output file. Found {len(results)} completed benchmarks.")
        except Exception as e:
            print(f"Failed to load existing {output_path}: {e}")

    # Load base.json to filter out failures if requested
    base_successes = set()
    if args.require_base_success:
        base_path = Path("results") / bench_dir.name / "exec_times" / "base.json"
        if args.config:
            with open(args.config) as cfgf:
                cfg = json.load(cfgf)
            base_path = Path(cfg["results_dir"]) / "exec_times" / "base.json"
        
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_data = json.load(f)
            base_successes = {k for k, v in base_data.items() if isinstance(v, (int, float)) and v > 0}
            print(f"Loaded {len(base_successes)} successful benchmarks from {base_path}")
        else:
            print(f"Warning: --require-base-success passed but {base_path} not found.")

    remaining_files = []
    for f in mlir_files:
        already_done = f.stem in results and results[f.stem].get("eager") not in (None, "N/A") and not str(results[f.stem].get("eager")).startswith("FAILED")
        
        if not already_done:
            if args.require_base_success and f.stem not in base_successes:
                continue # Skip files that failed in MLIR base.json
            remaining_files.append(f)

    print(f"Remaining benchmarks to process: {len(remaining_files)} / {len(mlir_files)}")

    import concurrent.futures

    def _try_measure(fn, label):
        if fn is None:
            return None
        try:
            return measure_ns(fn, warmup=args.warmup, trials=args.trials)
        except Exception as exc:
            return f"FAILED: {exc}"

    def process_file(args_tuple):
        i, mlir_file = args_tuple
        bench_name = mlir_file.stem
        code = mlir_file.read_text()

        fns, err = build_pytorch_fns(code)
        if fns is None:
            return i, bench_name, None, err

        entry: dict[str, Optional[int]] = {}
        entry["eager"]   = _try_measure(fns["eager"],   "eager")
        entry["compile"] = _try_measure(fns["compile"], "compile")
        entry["jit"]     = _try_measure(fns["jit"],     "jit")

        if (entry["eager"] is None or str(entry["eager"]).startswith("FAILED")) and (entry["jit"] is None or str(entry["jit"]).startswith("FAILED")):
            return i, bench_name, None, "all modes failed"

        return i, bench_name, entry, None

    process_args = [(i, f) for i, f in enumerate(remaining_files)]
    
    import os
    import multiprocessing
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    mp_context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context) as executor:
        futures = {executor.submit(process_file, arg): arg for arg in process_args}
        for fut in concurrent.futures.as_completed(futures):
            i, f = futures[fut]
            bench_name = f.stem
            try:
                _, _, entry, err = fut.result()
            except Exception as e:
                err = f"Uncaught Error: {e}"
                entry = None

            if err:
                print(f"[{i+1}/{len(remaining_files)}] SKIP {bench_name}: {err}")
                skipped += 1
            else:
                formatted_entry = {}
                for k, v in entry.items():
                    if isinstance(v, int):
                        formatted_entry[k] = v // 1000
                    else:
                        formatted_entry[k] = "N/A"
                results[bench_name] = formatted_entry
                print(f"[{i+1}/{len(remaining_files)}] {bench_name}: "
                      f"eager={formatted_entry.get('eager')}µs  "
                      f"compile={formatted_entry.get('compile')}µs  "
                      f"jit={formatted_entry.get('jit')}µs")
                
                # Write intermediary
                if (i+1) % 50 == 0:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f_out:
                        json.dump(results, f_out, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} measured, {skipped} skipped.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
