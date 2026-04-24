from tqdm import tqdm
import argparse
import traceback
import json
import os
from pathlib import Path

from utils.implementation import (
    get_autoschedular_impl,
    get_base_file_path,
    import_autoschedular_module,
)

parser = argparse.ArgumentParser(description="Get base execution times for MLIR benchmarks.")
parser.add_argument("--config", default=None, help="Path to config JSON (derives benchmarks-dir and output)")
parser.add_argument("--benchmarks-dir", default=None, help="Override: path to folder containing .mlir files")
parser.add_argument("--output", default=None, help="Override: path to output JSON file")
parser.add_argument("--implementation", default=None, help="Autoscheduler implementation package (default: AUTOSCHEDULER_IMPL or rl_autoschedular)")
args = parser.parse_args()

implementation = args.implementation or get_autoschedular_impl(config_path=args.config)
Execution = import_autoschedular_module("execution", implementation).Execution

if args.config:
    with open(args.config) as _f:
        _cfg = json.load(_f)
    path_to_folder = args.benchmarks_dir or _cfg["benchmarks_folder_path"]
    output_path    = args.output or str(get_base_file_path(_cfg["results_dir"], implementation))
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
    try:
        et, _, _ = exec.execute_code(code, bench_name, [])
    except Exception as e:
        print(f"\nFailed to execute {bench_name}: {e}", flush=True)
        traceback.print_exc()
        et = -1
    output_data[bench_name] = et

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
