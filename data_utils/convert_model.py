#!/usr/bin/env python3
"""
convert_model.py
----------------
High-level script to convert any supported neural network model to linalg MLIR.

Auto-detects model type (vision/transformer/gnn) from model_catalog.py,
then delegates to the appropriate converter (vision2mlir/transformers2mlir/gnn2mlir).

Usage:
    python -m data_utils.convert_model --model resnet18 --output-dir data/raw_bench/
    python -m data_utils.convert_model --model bert --output-dir data/raw_bench/
    python -m data_utils.convert_model --model gcn --type gnn --output-dir data/raw_bench/
    python -m data_utils.convert_model --model resnet50 --batch-size 1 --backend direct
"""

from __future__ import annotations

import argparse
import sys
import os

from data_utils.model_catalog import VISION_MODELS, TRANSFORMER_MODELS, GNN_MODELS


def detect_model_type(model: str) -> str:
    """Auto-detect model type from catalog."""
    if model in VISION_MODELS:
        return "vision"
    elif model in TRANSFORMER_MODELS:
        return "transformer"
    elif model in GNN_MODELS:
        return "gnn"
    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Available models:\n"
            f"  Vision: {', '.join(VISION_MODELS)}\n"
            f"  Transformer: {', '.join(TRANSFORMER_MODELS)}\n"
            f"  GNN: {', '.join(GNN_MODELS)}"
        )


def convert_vision(model: str, output_dir: str, backend: str, batch_size: int,
                   strip_weights: bool, verbose: bool):
    """Convert vision model to MLIR."""
    from data_utils.convert.vision2mlir import main as vision_main
    
    args = ["vision2mlir", "--model", model, "--output-dir", output_dir, "--backend", backend]
    if batch_size > 0:
        args += ["--batch-size", str(batch_size)]
    if strip_weights:
        args.append("--strip-weights")
    if verbose:
        args.append("--verbose")
    
    sys.argv = args
    vision_main()


def convert_transformer(model: str, output_dir: str, backend: str, batch_size: int,
                        strip_weights: bool, verbose: bool):
    """Convert transformer model to MLIR."""
    from data_utils.convert.transformers2mlir import main as transformer_main
    
    args = ["transformers2mlir", "--model", model, "--output-dir", output_dir, "--backend", backend]
    if batch_size > 0:
        args += ["--batch-size", str(batch_size)]
    if strip_weights:
        args.append("--strip-weights")
    if verbose:
        args.append("--verbose")
    
    sys.argv = args
    transformer_main()


def convert_gnn(model: str, output_dir: str, backend: str, batch_size: int,
                strip_weights: bool, verbose: bool, keep_onnx: bool):
    """Convert GNN model to MLIR."""
    from data_utils.convert.gnn2mlir import main as gnn_main
    
    args = ["gnn2mlir", "--model", model, "--output-dir", output_dir, "--backend", backend]
    if batch_size > 0:
        args += ["--batch-size", str(batch_size)]
    if strip_weights:
        args.append("--strip-weights")
    if keep_onnx:
        args.append("--keep-onnx")
    if verbose:
        args.append("--verbose")
    
    sys.argv = args
    gnn_main()


def main():
    parser = argparse.ArgumentParser(
        prog="convert_model",
        description="Convert any supported neural network model to linalg MLIR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m data_utils.convert_model --model resnet18 --output-dir data/raw_bench/
    python -m data_utils.convert_model --model bert --output-dir data/raw_bench/
    python -m data_utils.convert_model --model gcn --type gnn --output-dir data/raw_bench/
    python -m data_utils.convert_model --model resnet50 --batch-size 1 --backend direct

Available models:
    Vision:      {vision}
    Transformer: {transformer}
    GNN:         {gnn}
        """.format(
            vision=", ".join(VISION_MODELS),
            transformer=", ".join(TRANSFORMER_MODELS),
            gnn=", ".join(GNN_MODELS),
        )
    )
    
    parser.add_argument("--model", required=True,
                        help="Model name (auto-detected from catalog)")
    parser.add_argument("--type", choices=["vision", "transformer", "gnn"],
                        default=None,
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for MLIR file")
    parser.add_argument("--backend", choices=["onnx", "direct"], default="onnx",
                        help="Conversion backend (default: onnx)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Static batch size (0 = keep dynamic)")
    parser.add_argument("--strip-weights", action="store_true", default=True,
                        help="Strip large weight constants (default: True)")
    parser.add_argument("--no-strip-weights", action="store_false", dest="strip_weights",
                        help="Keep weight constants")
    parser.add_argument("--keep-onnx", action="store_true", default=False,
                        help="Keep intermediate ONNX files (GNN only)")
    parser.add_argument("--verbose", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Auto-detect or validate model type
    model_type = args.type
    if model_type is None:
        try:
            model_type = detect_model_type(args.model)
            print(f"[convert_model] Auto-detected type: {model_type}")
        except ValueError as e:
            print(f"[convert_model] Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[convert_model] Converting {args.model} ({model_type}) → {args.output_dir}/")
    
    # Dispatch to appropriate converter
    try:
        if model_type == "vision":
            convert_vision(args.model, args.output_dir, args.backend, args.batch_size,
                          args.strip_weights, args.verbose)
        elif model_type == "transformer":
            convert_transformer(args.model, args.output_dir, args.backend, args.batch_size,
                               args.strip_weights, args.verbose)
        elif model_type == "gnn":
            convert_gnn(args.model, args.output_dir, args.backend, args.batch_size,
                       args.strip_weights, args.verbose, args.keep_onnx)
        
        print(f"[convert_model] ✓ Conversion complete")
        print(f"[convert_model] Output: {args.output_dir}/{args.model}_linalg.mlir")
        
    except Exception as e:
        print(f"[convert_model] ✗ Conversion failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
