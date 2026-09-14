#!/usr/bin/env python3
"""
extract_from_model.py
---------------------
High-level script to extract benchmarks (ops and/or blocks) from a full-model MLIR file.

Supports:
  - Extract only single operations (--ops-only)
  - Extract only multi-op blocks (--blocks-only)
  - Extract both (default)

Usage:
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --ops-only
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --blocks-only
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --batch-size 1
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess


def extract_ops(input_file: str, output_dir: str, batch_size: int, model_name: str,
                manifest_dir: str | None, verbose: bool) -> int:
    """Extract single operations from MLIR file. Returns number of ops extracted."""
    from data_utils.extract.extract_ops import extract_from_file
    
    ops_dir = os.path.join(output_dir, "ops")
    os.makedirs(ops_dir, exist_ok=True)
    
    print(f"[extract_from_model] Extracting single ops → {ops_dir}/")
    
    try:
        count = extract_from_file(
            input_path=input_file,
            output_dir=ops_dir,
            batch_size=batch_size,
            model_name=model_name,
            min_parallel_loops=2,
            require_reduction=True,
            generic_ratio=None,
        )
        print(f"[extract_from_model] ✓ Extracted {count} single ops")
        return count
    except Exception as e:
        print(f"[extract_from_model] ✗ Ops extraction failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 0


def extract_blocks(input_file: str, output_dir: str, batch_size: int, model_name: str,
                   window_size: int, stride: int, manifest_dir: str | None,
                   verbose: bool) -> int:
    """Extract multi-op blocks from MLIR file. Returns number of blocks extracted."""
    from data_utils.extract.extract_blocks import extract_blocks_from_file
    
    blocks_dir = os.path.join(output_dir, "blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    
    print(f"[extract_from_model] Extracting multi-op blocks → {blocks_dir}/")
    
    manifest_path = None
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"{model_name}_blocks_manifest.json")
    
    try:
        written, skipped = extract_blocks_from_file(
            input_path=input_file,
            output_dir=blocks_dir,
            model_name=model_name,
            window_size=window_size,
            stride=stride,
            max_depth=64,
            max_paths=5000,
            batch_candidates=[batch_size],
            batch_fallback=None,
            manifest_path=manifest_path,
            skip_pure_elementwise=False,
        )
        print(f"[extract_from_model] ✓ Extracted {written} blocks ({skipped} skipped)")
        return written
    except Exception as e:
        print(f"[extract_from_model] ✗ Blocks extraction failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        prog="extract_from_model",
        description="Extract benchmarks (ops and/or blocks) from a full-model MLIR file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract both ops and blocks (default)
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/
    
    # Extract only single operations
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --ops-only
    
    # Extract only multi-op blocks
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --blocks-only
    
    # With custom batch size and block parameters
    python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ \\
        --batch-size 4 --window-size 7 --stride 4
        """
    )
    
    parser.add_argument("--input", required=True,
                        help="Input *_linalg.mlir file")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for extracted benchmarks")
    
    # Extraction mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--ops-only", action="store_true",
                            help="Extract only single operations (skip blocks)")
    mode_group.add_argument("--blocks-only", action="store_true",
                            help="Extract only multi-op blocks (skip ops)")
    
    # Common options
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for dynamic dimensions (default: 1)")
    parser.add_argument("--model-name", default=None,
                        help="Model name prefix (default: derived from input filename)")
    
    # Block-specific options
    parser.add_argument("--window-size", type=int, default=5,
                        help="Block extraction window size (default: 5)")
    parser.add_argument("--stride", type=int, default=3,
                        help="Block extraction stride (default: 3)")
    
    # Manifest options
    parser.add_argument("--manifest-dir", default=None,
                        help="Directory for extraction manifest JSON files")
    
    parser.add_argument("--verbose", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input):
        print(f"[extract_from_model] Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Derive model name if not provided
    model_name = args.model_name
    if model_name is None:
        model_name = os.path.basename(args.input).replace("_linalg.mlir", "").replace(".mlir", "")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[extract_from_model] Input: {args.input}")
    print(f"[extract_from_model] Output: {args.output_dir}/")
    print(f"[extract_from_model] Model: {model_name}")
    print(f"[extract_from_model] Batch size: {args.batch_size}")
    
    # Determine extraction mode
    extract_ops_flag = not args.blocks_only
    extract_blocks_flag = not args.ops_only
    
    if args.ops_only:
        print(f"[extract_from_model] Mode: ops only")
    elif args.blocks_only:
        print(f"[extract_from_model] Mode: blocks only")
    else:
        print(f"[extract_from_model] Mode: ops + blocks")
    
    print()
    
    # Run extractions
    ops_count = 0
    blocks_count = 0
    
    if extract_ops_flag:
        ops_count = extract_ops(
            input_file=args.input,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            model_name=model_name,
            manifest_dir=args.manifest_dir,
            verbose=args.verbose,
        )
        print()
    
    if extract_blocks_flag:
        blocks_count = extract_blocks(
            input_file=args.input,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            model_name=model_name,
            window_size=args.window_size,
            stride=args.stride,
            manifest_dir=args.manifest_dir,
            verbose=args.verbose,
        )
    
    # Summary
    print()
    print("=" * 60)
    print(f"[extract_from_model] Extraction complete")
    print(f"  Single ops:  {ops_count}")
    print(f"  Multi blocks: {blocks_count}")
    print(f"  Output dir:  {args.output_dir}/")
    if extract_ops_flag:
        print(f"  Ops dir:     {args.output_dir}/ops/")
    if extract_blocks_flag:
        print(f"  Blocks dir:  {args.output_dir}/blocks/")
    print("=" * 60)


if __name__ == "__main__":
    main()
