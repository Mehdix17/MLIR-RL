from tqdm import tqdm
import argparse
import traceback
import json
import os
import signal
from pathlib import Path

try:
    from mlir.ir import Context, Module, MemRefType
    from mlir.dialects.func import FuncOp
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

from utils.implementation import (
    get_autoschedular_impl,
    get_base_file_path,
    import_autoschedular_module,
)

# Per-file timeout in seconds. Files that take longer are killed (et=-1).
FILE_TIMEOUT = 600


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("MLIR execution timed out")


parser = argparse.ArgumentParser(description="Get base execution times for MLIR benchmarks.")
parser.add_argument("--config", default=None, help="Path to config JSON (derives benchmarks-dir and output)")
parser.add_argument("--benchmarks-dir", default=None, help="Override: path to folder containing .mlir files")
parser.add_argument("--output", default=None, help="Override: path to output JSON file")
parser.add_argument("--implementation", default=None, help="Autoscheduler implementation package (default: AUTOSCHEDULER_IMPL or rl_autoschedular)")
parser.add_argument("--timeout", type=int, default=FILE_TIMEOUT, help=f"Per-file timeout in seconds (default: {FILE_TIMEOUT})")
args = parser.parse_args()

implementation = args.implementation or get_autoschedular_impl(config_path=args.config)
Execution = import_autoschedular_module("execution", implementation).Execution

if args.config:
    with open(args.config) as _f:
        _cfg = json.load(_f)
    path_to_folder = args.benchmarks_dir or _cfg["benchmarks_folder_path"]
    # If --output is explicitly given, use it; otherwise write to the
    # dataset-level generic base.json (shared across implementations).
    if args.output:
        output_path = args.output
    else:
        output_path = str(get_base_file_path(_cfg["results_dir"], implementation=None))
else:
    if not args.benchmarks_dir or not args.output:
        parser.error("Provide --config, or both --benchmarks-dir and --output")
    path_to_folder = args.benchmarks_dir
    output_path    = args.output

if not os.path.isdir(path_to_folder):
    print(f"Error: {path_to_folder} is not a valid directory.")
    raise SystemExit(1)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)

output_data = {}
exec = Execution("")

print(f"Using autoscheduler implementation: {implementation}")

code_files = [f for f in os.listdir(path_to_folder) if f.endswith('.mlir')]
files_tqdm = tqdm(code_files, unit='file')
for code_file in files_tqdm:
    bench_name = code_file.replace('.mlir', '')
    files_tqdm.set_postfix_str(bench_name)
    full_path = os.path.join(path_to_folder, code_file)
    with open(full_path, 'r') as f:
        code = f.read()

    # Quick pre-check: parse the MLIR and verify it has a main function.
    # This catches malformed files instantly (<0.1s) instead of waiting
    # for the subprocess + full pass pipeline to fail.
    # NOTE: We DON'T check input types here — the bufferization pass
    # converts tensors→memrefs before __create_params runs.
    try:
        with Context():
            pre_module = Module.parse(code)
            pre_funcs = [op for op in pre_module.body.operations if isinstance(op, FuncOp)]
            pre_main = next((op for op in pre_funcs if op.name.value == 'main'), None)
            if pre_main is None:
                print(f"\nFailed to execute {bench_name}: No main function found", flush=True)
                output_data[bench_name] = -1
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                continue
    except Exception as e:
        print(f"\nFailed to execute {bench_name}: Pre-check failed — {e}", flush=True)
        output_data[bench_name] = -1
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        continue

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)
        et, _, _ = exec.execute_code(code, bench_name, [])
        signal.alarm(0)
    except TimeoutError:
        print(f"\nFailed to execute {bench_name}: timed out after {args.timeout}s", flush=True)
        et = -1
    except Exception as e:
        print(f"\nFailed to execute {bench_name}: {e}", flush=True)
        traceback.print_exc()
        et = -1
    finally:
        signal.alarm(0)
    output_data[bench_name] = et

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
