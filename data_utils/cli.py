#!/usr/bin/env python3
"""
orchestrate.py
--------------
Single CLI entry point for the MLIR-RL data pipeline.

Subcommands:
  build-benchmark     — Generate a synthetic benchmark dataset (JSON).
  vision              — Convert a torchvision model to linalg MLIR.
  transformer         — Convert a HuggingFace transformer model to linalg MLIR.
  wrap                — Wrap an existing .mlir file with a timed @main.
  strip               — Strip large weight constants from a .mlir file.

Environment checks:
  LLVM_BUILD_PATH     — Required for the CMD evaluation backend.
  MLIR_SHARED_LIBS    — Required for the Python bindings backend.
  LLVM_BUILD_PATH/bin/mlir-opt, mlir-cpu-runner — Required for CMD backend.

Examples:
  python -m data_utils.orchestrate build-benchmark \\
      --input_file config/example.json --output_file data/dataset.json

  python -m data_utils.orchestrate vision \\
      --model resnet50 --output-dir data/generated/code_files

  python -m data_utils.orchestrate transformer \\
      --model bert --backend onnx

  python -m data_utils.orchestrate wrap \\
      --input model.mlir --model-name forward --output model_wrapped.mlir

  python -m data_utils.orchestrate strip \\
      --input huge_model_linalg.mlir --replace
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _check_env(require_cmd: bool = False, require_bindings: bool = False):
    warnings = []

    if require_cmd:
        llvm = os.getenv("LLVM_BUILD_PATH")
        if not llvm:
            warnings.append("LLVM_BUILD_PATH is not set (needed for CMD backend).")
        else:
            for binary in ["mlir-opt", "mlir-cpu-runner"]:
                path = os.path.join(llvm, "bin", binary)
                if not os.path.isfile(path):
                    warnings.append(f"  {path} not found.")

    if require_bindings:
        libs = os.getenv("MLIR_SHARED_LIBS", "")
        if not libs:
            warnings.append(
                "MLIR_SHARED_LIBS is not set (needed for bindings backend)."
            )
        try:
            from mlir.ir import Context  # noqa: F401
        except ImportError:
            warnings.append(
                "MLIR Python bindings not importable "
                "(pip install mlir or build from source)."
            )

    if warnings:
        print("[orchestrate] Environment warnings:")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()


# ---------------------------------------------------------------------------
# Subcommand: build-benchmark
# ---------------------------------------------------------------------------

def _cmd_build_benchmark(args):
    _check_env(
        require_cmd=args.backend == "cmd",
        require_bindings=(args.backend in ("bindings", None)),
    )
    from data_utils.build_benchmark import main as _main
    # Reconstruct argv for build_benchmark's argparse
    extra = [
        "--input_file",  args.input_file,
        "--output_file", args.output_file,
        "--timeout",     str(args.timeout),
    ]
    if args.backend:
        extra += ["--backend", args.backend]
    sys.argv = ["build_benchmark"] + extra
    _main()


# ---------------------------------------------------------------------------
# Subcommand: vision
# ---------------------------------------------------------------------------

def _cmd_vision(args):
    _check_env(require_bindings=(args.backend == "direct"))
    from data_utils.vision2mlir import main as _main
    extra = ["--model", args.model, "--backend", args.backend]
    if args.output_dir:
        extra += ["--output-dir", args.output_dir]
    if args.strip_weights:
        extra.append("--strip-weights")
    if args.verbose:
        extra.append("--verbose")
    sys.argv = ["vision2mlir"] + extra
    _main()


# ---------------------------------------------------------------------------
# Subcommand: transformer
# ---------------------------------------------------------------------------

def _cmd_transformer(args):
    _check_env(require_bindings=(args.backend == "direct"))
    from data_utils.transformers2mlir import main as _main
    extra = ["--model", args.model, "--backend", args.backend]
    if args.output_dir:
        extra += ["--output-dir", args.output_dir]
    if args.strip_weights:
        extra.append("--strip-weights")
    if args.verbose:
        extra.append("--verbose")
    sys.argv = ["transformers2mlir"] + extra
    _main()


# ---------------------------------------------------------------------------
# Subcommand: wrap
# ---------------------------------------------------------------------------

def _cmd_wrap(args):
    from data_utils.wrap_mlir import main_wrapper
    main_wrapper(args.input, args.model_name, args.output)
    print(f"Wrapped '{args.model_name}': {args.input} → {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: strip
# ---------------------------------------------------------------------------

def _cmd_strip(args):
    from data_utils.strip_mlir import strip_weights
    output = args.output
    if args.replace:
        output = args.input + ".tmp"
    reduction = strip_weights(args.input, output, verbose=args.verbose)
    if args.replace:
        os.replace(output, args.input)
        print(f"Replaced {args.input} in-place ({reduction:.1f}% reduction).")
    else:
        print(f"Stripped to {output} ({reduction:.1f}% reduction).")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrate",
        description="MLIR-RL data pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-benchmark
    p_bench = sub.add_parser(
        "build-benchmark",
        help="Generate a synthetic MLIR benchmark dataset (JSON).",
    )
    p_bench.add_argument("--input_file",  required=True,
                         help="YAML config (shapes, operations, amounts).")
    p_bench.add_argument("--output_file", required=True,
                         help="Output JSON dataset file.")
    p_bench.add_argument("--backend", choices=["bindings", "cmd"], default=None,
                         help="Evaluation backend (default: auto-detect).")
    p_bench.add_argument("--timeout", type=float, default=300)

    # vision
    p_vis = sub.add_parser(
        "vision",
        help="Convert a torchvision model to linalg MLIR.",
    )
    VISION_MODELS = [
        "resnet18", "resnet50", "efficientnet_b0",
        "mobilenet_v2", "mobilenet_v3_small", "densenet121", "vit_b_16",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "vgg11",
    ]
    p_vis.add_argument("--model", default="resnet18", choices=VISION_MODELS)
    p_vis.add_argument("--output-dir", default=None,
                       help="Output directory (default: data/generated/code_files).")
    p_vis.add_argument("--backend", choices=["onnx", "direct"], default="onnx")
    p_vis.add_argument("--strip-weights", action="store_true", default=True)
    p_vis.add_argument("--verbose", action="store_true")

    # transformer
    p_tr = sub.add_parser(
        "transformer",
        help="Convert a HuggingFace transformer model to linalg MLIR.",
    )
    TRANSFORMER_MODELS = [
        "bert", "distilbert", "roberta", "albert",
        "gpt2", "t5", "bart", "lstm",
    ]
    p_tr.add_argument("--model", default="distilbert", choices=TRANSFORMER_MODELS)
    p_tr.add_argument("--output-dir", default=None)
    p_tr.add_argument("--backend", choices=["onnx", "direct"], default="onnx")
    p_tr.add_argument("--strip-weights", action="store_true", default=True)
    p_tr.add_argument("--verbose", action="store_true")

    # wrap
    p_wrap = sub.add_parser(
        "wrap",
        help="Wrap an existing .mlir model file with a timed @main.",
    )
    p_wrap.add_argument("--input",      required=True, help="Input .mlir file.")
    p_wrap.add_argument("--model-name", required=True,
                        help="Name of the forward function (e.g. 'forward', 'main_graph').")
    p_wrap.add_argument("--output",     required=True, help="Output .mlir file.")

    # strip
    p_strip = sub.add_parser(
        "strip",
        help="Strip large weight constants from a .mlir file.",
    )
    p_strip.add_argument("input", help="Input .mlir file.")
    p_strip.add_argument("-o", "--output",  default=None,
                         help="Output file (default: <input>_stripped.mlir).")
    p_strip.add_argument("-r", "--replace", action="store_true",
                         help="Overwrite input file.")
    p_strip.add_argument("-v", "--verbose", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "build-benchmark": _cmd_build_benchmark,
        "vision":          _cmd_vision,
        "transformer":     _cmd_transformer,
        "wrap":            _cmd_wrap,
        "strip":           _cmd_strip,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
