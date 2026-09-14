#!/usr/bin/env python3
"""
Build a JSON file mapping generated MLIR basenames to measured execution times.

Usage:
  python scripts/build_generated_exec_json.py --out data/generated/execution_times_train.json --limit 2
"""
import argparse
import glob
import json
import os
import statistics
from typing import List

import shutil
import subprocess


def _find_tool(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    llvm = os.getenv('LLVM_BUILD_PATH')
    if llvm:
        candidate = os.path.join(llvm, 'bin', name)
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate_code_cmd(code: str, tmp_file_path: str, timeout: float | None = None) -> tuple[int, bool]:
    """Lower and run MLIR using mlir-opt and mlir-cpu-runner. Returns (exec_time_ns, True)"""
    mlir_opt = _find_tool('mlir-opt')
    mlir_cpu_runner = _find_tool('mlir-cpu-runner')
    if not mlir_opt or not mlir_cpu_runner:
        raise RuntimeError('mlir-opt or mlir-cpu-runner not found in PATH or LLVM_BUILD_PATH')

    opt_flags = (
        "-loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor "
        "-one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' "
        "-convert-vector-to-scf -convert-linalg-to-loops -buffer-deallocation-pipeline -scf-forall-to-parallel "
        "-convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine "
        "-convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm "
        "-convert-math-to-llvm -convert-math-to-libm -reconcile-unrealized-casts -canonicalize -cse"
    )

    shared_libs = ''
    llvm = os.getenv('LLVM_BUILD_PATH')
    if llvm:
        libs = [
            os.path.join(llvm, 'lib', 'libmlir_runner_utils.so'),
            os.path.join(llvm, 'lib', 'libmlir_c_runner_utils.so'),
            os.path.join(llvm, 'lib', 'libomp.so'),
        ]
        shared_libs = ','.join([p for p in libs if os.path.exists(p)])

    cmd = f"{mlir_opt} {opt_flags} {tmp_file_path} | {mlir_cpu_runner} /dev/stdin"
    if shared_libs:
        cmd = f"{mlir_opt} {opt_flags} {tmp_file_path} | {mlir_cpu_runner} /dev/stdin -shared-libs={shared_libs}"

    # Ensure OMP threads environment
    os.environ['OMP_NUM_THREADS'] = os.getenv('OMP_NUM_THREADS', '1')

    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Execution failed: {e.output}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Execution timed out")

    # The mlir-cpu-runner prints the time (ns) as the last line
    lines = out.strip().splitlines()
    if not lines:
        raise RuntimeError('No output from runner')
    last = lines[-1].strip()
    try:
        val = int(last)
    except ValueError:
        raise RuntimeError(f"Unexpected runner output: {last}")
    return val, True
import tempfile
import pathlib


def measure_code(code: str, tries: int = 3) -> float:
    execs = []
    # write to a temporary file for mlir-cpu-runner path-based evaluator
    with tempfile.NamedTemporaryFile(suffix='.mlir', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(code.encode())

    try:
        for _ in range(tries):
            t, passed = evaluate_code_cmd(code, tmp_path, timeout=None)
            if not passed or t is None:
                raise RuntimeError("Evaluation failed for code")
            execs.append(t)
    finally:
        try:
            pathlib.Path(tmp_path).unlink()
        except Exception:
            pass

    return float(statistics.median(execs))


def build(files: List[str], out: str, tries: int = 3):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = {}
    for f in files:
        name = os.path.basename(f).rsplit('.', 1)[0]
        print(f"Measuring {name} from {f}")
        with open(f, 'r') as fh:
            code = fh.read()
        t = measure_code(code, tries=tries)
        data[name] = t
        print(f"  {name}: {t}s")

    with open(out, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote {out} with {len(data)} entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join("data", "generated", "code files"))
    parser.add_argument("--out", default="data/generated/execution_times_train.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N files (0=all)")
    parser.add_argument("--tries", type=int, default=3)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_linalg.mlir")))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise SystemExit("No MLIR files found in dir")

    build(files, args.out, tries=args.tries)


if __name__ == "__main__":
    main()
