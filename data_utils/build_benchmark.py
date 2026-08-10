"""
build_benchmark.py
------------------
Dataset builder: generates random MLIR operations/graphs, benchmarks their execution
time, and writes results to a JSON file.

Evaluation backend is chosen automatically at startup:
  1. Bindings (default, recommended) — in-process via MLIR Python bindings.
  2. CMD (fallback) — subprocess via mlir-opt | mlir-cpu-runner.

The fallback is activated when the MLIR Python bindings are unavailable or when
--backend cmd is passed explicitly.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import traceback
import argparse
import numpy as np
import yaml
from dataclasses import asdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Backend detection — bindings primary, cmd fallback
# ---------------------------------------------------------------------------

_USE_BINDINGS = False

try:
    from rl_autoschedular.evaluation import evaluate_code_with_bindings_and_timeout
    from rl_autoschedular.observation import main_wrapper as _wrap_bindings
    _USE_BINDINGS = True
except Exception:
    pass  # bindings not available — will use cmd

from rl_autoschedular.evaluation import evaluate_code_with_cmd_and_timeout
from rl_autoschedular.observation import (
    transform_wrapper as _wrap_cmd,
    __inline,
    extract_op_features_from_affine_code,
)

# ---------------------------------------------------------------------------
# CMD-only helpers (fix() is only needed in the cmd path because transform_wrapper
# produces a @main that calls @matmul and needs the return type patched in)
# ---------------------------------------------------------------------------

import re

_TMP_FILE_BINDINGS = 'tmp/tmp-2.mlir'
_TMP_FILE_CMD      = 'tmp/test_fill.mlir'


def _fix_cmd_code(code: str, tmp_file: str) -> str:
    """Patches the @main wrapper produced by transform_wrapper so that the
    scf.for return type matches the matmul output shape."""
    _, (start, _) = _extract_function_safe(code, "main")
    if start is None:
        return code

    shape = None
    for line in code.splitlines():
        if 'func.func @matmul' in line:
            find = re.search(r"(tensor<\d+x[\d+x]*f32>)", line)
            if find:
                shape = find.group(0)
            break

    if shape is None:
        return code

    patched = '\n'.join(
        code.splitlines()[:start] +
        f"""func.func @main(){{
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {{
    %outputmain = func.call @matmul() : () -> {shape}
    }}
    return
}}
}}""".splitlines()
    )
    return patched


def _extract_function_safe(code: str, name: str):
    """Safe wrapper around extract_function that returns (fn, (start, None)) or
    (None, (None, None)) when the function is not found."""
    try:
        from rl_autoschedular.observation import extract_function
        return extract_function(code, name)
    except Exception:
        return None, (None, None)


# ---------------------------------------------------------------------------
# Unified evaluation — dispatches to bindings or cmd
# ---------------------------------------------------------------------------

def _evaluate(code: str, tmp_file: str, timeout: float, use_bindings: bool):
    """Run code and return (exec_time, passed)."""
    if use_bindings:
        return evaluate_code_with_bindings_and_timeout(code, timeout)
    else:
        return evaluate_code_with_cmd_and_timeout(code, tmp_file, timeout)


def _wrap(raw_operation: str, tmp_file: str, maps=None,
          additional_function=None, use_bindings: bool = True):
    """Wrap a raw MLIR operation into a benchmarkable @main function."""
    if use_bindings:
        return _wrap_bindings(raw_operation, tmp_file, maps=maps,
                              additional_function=additional_function)
    else:
        wrapped = _wrap_cmd(raw_operation, maps=maps,
                            additional_function=additional_function)
        wrapped = __inline(wrapped, tmp_file)
        return _fix_cmd_code(wrapped, tmp_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build a benchmark dataset of MLIR operations with execution times.'
    )
    parser.add_argument('--input_file',  required=True,
                        help='YAML config file (shapes, operations, amounts).')
    parser.add_argument('--output_file', required=True,
                        help='Output JSON file for the dataset.')
    parser.add_argument('--backend', choices=['bindings', 'cmd'], default=None,
                        help='Evaluation backend. Default: bindings if available, else cmd.')
    parser.add_argument('--timeout', type=float, default=300,
                        help='Timeout per evaluation in seconds (default: 300).')
    args = parser.parse_args()

    # Resolve backend
    if args.backend == 'cmd':
        use_bindings = False
    elif args.backend == 'bindings':
        if not _USE_BINDINGS:
            raise RuntimeError(
                "Bindings backend requested but MLIR Python bindings are not available."
            )
        use_bindings = True
    else:
        use_bindings = _USE_BINDINGS  # auto

    tmp_file = _TMP_FILE_BINDINGS if use_bindings else _TMP_FILE_CMD
    backend_name = 'bindings' if use_bindings else 'cmd'
    print(f"[build_benchmark] Using backend: {backend_name}")

    # Load config
    with open(args.input_file, 'r') as f:
        config = yaml.safe_load(f)

    # Import shape globals and generators
    from data_utils.mlir_generators import (
        LINALG_OPERATION_GENERATORS,
        BATCH_SIZES, HEIGHTS, CHANNELS, KERNELS, DILATIONS, STRIDES, SIZES,
        randomSubGraph, randomblocks,
        generate_resnet_block, generate_residual_block_mlir,
        vgg, bert, convnext,
    )

    shapes = config.get('SHAPES', {})
    BATCH_SIZES.extend(shapes.get('BATCH_SIZES', []))
    HEIGHTS.extend(shapes.get('HEIGHTS', []))
    CHANNELS.extend(shapes.get('CHANNELS', []))
    KERNELS.extend(shapes.get('KERNELS', []))
    DILATIONS.extend(shapes.get('DILATIONS', []))
    STRIDES.extend(shapes.get('STRIDES', []))
    SIZES.extend(shapes.get('SIZES', []))

    # Build operations config from YAML, or use default (randomSubGraph)
    raw_ops = config.get('OPERATIONS', {})
    if raw_ops:
        operations_config = {
            name: (LINALG_OPERATION_GENERATORS[name], amount)
            for name, amount in raw_ops.items()
            if amount > 0 and name in LINALG_OPERATION_GENERATORS
        }
    else:
        operations_config = {
            "randomSubGraph": (randomSubGraph, 200),
        }

    print(f"[build_benchmark] Operations: {list(operations_config.keys())}")
    print(f"[build_benchmark] Total: {sum(a for _, (_, a) in operations_config.items())}")

    all_operations = {}

    for operation_name, (generator, amount) in tqdm(operations_config.items(),
                                                     desc="linalg operations"):
        for i in tqdm(range(amount), desc=operation_name):
            exec_time = None
            maps = None
            additional_function = None

            try:
                res = generator()

                if isinstance(res, tuple):
                    raw_operation, additional_tuple = res
                    if isinstance(additional_tuple, tuple):
                        maps, additional_function = additional_tuple
                    else:
                        maps = additional_tuple
                else:
                    raw_operation = res

                # Build wrapped code
                wrapped = _wrap(raw_operation, tmp_file, maps=maps,
                                additional_function=additional_function,
                                use_bindings=use_bindings)

                # Evaluate
                exec_time, assertion = _evaluate(wrapped, tmp_file,
                                                 args.timeout, use_bindings)

                if assertion and exec_time is not None:
                    exec_time = np.median(
                        [exec_time] +
                        [_evaluate(wrapped, tmp_file, args.timeout, use_bindings)[0]
                         for _ in range(2)]
                    )

            except Exception as e:
                print(f"\033[91;1mError generating '{operation_name}' #{i}: {e}\033[0m")
                traceback.print_exc()
                continue

            if exec_time:
                all_operations[f"{operation_name}_{i}"] = {
                    "operation": raw_operation,
                    "transform_wrapped_operation": wrapped,
                    "execution_time": exec_time,
                }

    with open(args.output_file, 'w') as f:
        json.dump(all_operations, f)

    print(f"\n[build_benchmark] Done. {len(all_operations)} entries written to {args.output_file}")


if __name__ == '__main__':
    main()
