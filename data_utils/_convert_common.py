"""
_convert_common.py
------------------
Shared conversion utilities for vision2mlir, transformers2mlir, and gnn2mlir.

Contains the ONNX export → shape-inference → torch-mlir import → linalg lowering
pipeline, the direct torch_mlir route, file post-processing, and weight stripping.

Each ``*2mlir.py`` file keeps only its model-specific loading logic and CLI
entry point; all boilerplate conversion machinery lives here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Sequence

import torch

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NON_STRIPPED_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "non_stripped_models")


# ── Post-processing helpers ─────────────────────────────────────────────────

def specialize_tensor_types(content: str, static_batch_size: int) -> tuple[str, int]:
    """Replace ``?`` dims in tensor types with *static_batch_size*.

    Returns (modified_content, number_of_replacements).
    """
    replaced = 0

    def _repl(m: re.Match) -> str:
        nonlocal replaced
        token = m.group(0)
        count = token.count("?")
        if count:
            replaced += count
            token = token.replace("?", str(static_batch_size))
        return token

    content = re.sub(r"tensor<[^>]+>", _repl, content)
    content = re.sub(r"!torch\.vtensor<\[[^\]]+\],[^>]+>", _repl, content)
    return content, replaced


def postprocess_linalg_file(linalg_path: str, static_batch_size: int = 0) -> None:
    """Normalize entry name (``@main_graph`` → ``@main``) and optionally
    specialize dynamic ``?`` dimensions with a static batch size."""
    with open(linalg_path, "r") as fh:
        content = fh.read()

    changed = False
    if "@main_graph" in content:
        content = content.replace("@main_graph", "@main")
        changed = True
        print("  Renamed @main_graph -> @main")

    if static_batch_size > 0 and "?" in content:
        content, replaced = specialize_tensor_types(content, static_batch_size)
        if replaced:
            changed = True
            print(f"  Specialized dynamic dims with static batch="
                  f"{static_batch_size} ({replaced} replacement(s))")

    if changed:
        with open(linalg_path, "w") as fh:
            fh.write(content)


# ── Weight stripping / backup ────────────────────────────────────────────────

def backup_and_strip(output_path: str, strip_weights: bool,
                     verbose: bool = False) -> None:
    """Optionally back up the un-stripped file, then strip large weights."""
    import shutil
    os.makedirs(NON_STRIPPED_DIR, exist_ok=True)
    backup_path = os.path.join(NON_STRIPPED_DIR, os.path.basename(output_path))
    shutil.copy2(output_path, backup_path)
    print(f"  Non-stripped backup: {backup_path}")

    if not strip_weights:
        return
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > 1:
        print(f"  File is {size_mb:.1f} MB — stripping weights...")
        from data_utils.strip_mlir import strip_weights as _strip
        reduction = _strip(output_path, output_path, verbose=verbose)
        print(f"  Stripped: {reduction:.1f}% reduction.")


def strip_only(output_path: str, strip_weights: bool,
               verbose: bool = False) -> None:
    """Strip large weights without creating a backup (for GNN pipeline)."""
    if not strip_weights:
        return
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > 1:
        print(f"  File is {size_mb:.1f} MB — stripping weights...")
        from data_utils.strip_mlir import strip_weights as _strip
        reduction = _strip(output_path, output_path, verbose=verbose)
        print(f"  Stripped: {reduction:.1f}% reduction.")


# ── ONNX → linalg pipeline ─────────────────────────────────────────────────

def _cleanup_onnx_intermediates(base: str, keep_onnx: bool) -> None:
    """Remove intermediate ONNX files unless *keep_onnx* is set."""
    if keep_onnx:
        return
    for ext in (".onnx", ".onnx.data", "_inferred.onnx"):
        p = base + ext
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed intermediate: {os.path.basename(p)}")


def _shape_infer_subprocess(base: str) -> str:
    """Run symbolic shape inference as a subprocess (transformers2mlir style).

    Returns the path to the inferred ONNX file on success, or the original
    path if inference failed.
    """
    onnx_input = f"{base}.onnx"
    result = subprocess.run(
        [
            sys.executable, "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", f"{base}.onnx",
            "--output", f"{base}_inferred.onnx",
            "--auto_merge",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"{base}_inferred.onnx"
    print(f"  Shape inference warning: {result.stderr}")
    return onnx_input


def _shape_infer_inprocess(base: str) -> str:
    """Run symbolic shape inference in-process (vision2mlir style).

    Returns the path to the inferred ONNX file.
    """
    import onnx
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        model_proto = onnx.load(f"{base}.onnx")
        inferred = SymbolicShapeInference.infer_shapes(
            model_proto, auto_merge=True, guess_output_rank=True,
        )
        out = f"{base}_inferred.onnx"
        onnx.save(inferred, out)
        return out
    except ImportError:
        pass
    # Fallback to onnx.shape_inference
    m = onnx.load(f"{base}.onnx")
    m2 = onnx.shape_inference.infer_shapes(m)
    out = f"{base}_inferred.onnx"
    onnx.save(m2, out)
    return out


def _shape_infer_fallback_copy(base: str) -> str:
    """Copy .onnx as-is when no shape inference is available (gnn2mlir style)."""
    import shutil
    shutil.copy(f"{base}.onnx", f"{base}_inferred.onnx")
    return f"{base}_inferred.onnx"


def _import_onnx(base: str, onnx_input: str, opset: int) -> None:
    """Import ONNX file to torch MLIR with CLI fallback."""
    torch_mlir_path = f"{base}_torch.mlir"

    # Try torch-mlir-import-onnx first
    result = subprocess.run(
        ["torch-mlir-import-onnx", onnx_input, "-o", torch_mlir_path,
         "--opset-version", str(opset)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    # Fallback: python -m torch_mlir.tools.import_onnx
    result2 = subprocess.run(
        [sys.executable, "-m", "torch_mlir.tools.import_onnx",
         onnx_input, "-o", torch_mlir_path,
         "--opset-version", str(opset)],
        capture_output=True, text=True,
    )
    if result2.returncode != 0:
        raise RuntimeError(
            f"ONNX import failed:\n{result.stderr}\n{result2.stderr}"
        )


def _lower_to_linalg(base: str, method: str = "flags") -> None:
    """Lower torch MLIR to linalg-on-tensors.

    *method* controls the lowering syntax:
      - ``"flags"``: four separate ``torch-mlir-opt`` flags (vision/transformers)
      - ``"pipeline"``: ``--pass-pipeline=builtin.module(...)`` (gnn)
    """
    torch_mlir_path = f"{base}_torch.mlir"
    linalg_path = f"{base}_linalg.mlir"

    if method == "pipeline":
        torch_mlir_opt = os.path.join(os.path.dirname(sys.executable), "torch-mlir-opt")
        passes = (
            "torch-lower-to-backend-contract,"
            "torch-backend-to-linalg-on-tensors-backend-pipeline"
        )
        subprocess.run(
            [torch_mlir_opt, torch_mlir_path,
             f"--pass-pipeline=builtin.module({passes})",
             "-o", linalg_path],
            check=True,
        )
    else:
        opt_flags = [
            "--convert-torch-onnx-to-torch",
            "--torch-decompose-complex-ops",
            "--convert-torch-to-linalg",
            "--torch-backend-to-linalg-on-tensors-backend-pipeline",
        ]
        result = subprocess.run(
            ["torch-mlir-opt", torch_mlir_path] + opt_flags + ["-o", linalg_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"torch-mlir-opt failed:\n{result.stderr}")


def onnx_route(
    model: torch.nn.Module,
    model_name: str,
    example_inputs,
    output_dir: str,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset: int = 18,
    keep_onnx: bool = False,
    static_batch_size: int = 0,
    shape_infer: str = "inprocess",
    lowering_method: str = "flags",
    do_postprocess: bool = True,
) -> str:
    """Full ONNX → linalg conversion pipeline.

    Parameters
    ----------
    model : torch.nn.Module
    model_name : str
    example_inputs : tensor or tuple of tensors
    output_dir : str
    input_names : list[str], optional
        ONNX input names.  Auto-generated as ``["input_0", ...]`` if *None*.
    output_names : list[str], optional
        ONNX output names.  Defaults to ``["output"]``.
    opset : int
    keep_onnx : bool
    static_batch_size : int
        If > 0, replace ``?`` dims in the output.  0 preserves dynamic shapes.
    shape_infer : str
        ``"inprocess"`` | ``"subprocess"`` | ``"fallback_copy"``
    lowering_method : str
        ``"flags"`` | ``"pipeline"``
    do_postprocess : bool
        Whether to call :func:`postprocess_linalg_file` after lowering.

    Returns
    -------
    str
        Path to the produced ``*_linalg.mlir`` file.
    """
    base = os.path.join(output_dir, model_name)
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(example_inputs)
                      if isinstance(example_inputs, tuple) else 1)]
    if output_names is None:
        output_names = ["output"]

    # Step 1 — export to ONNX
    print(f"  Exporting {model_name} to ONNX (opset {opset})...")
    torch.onnx.export(
        model, example_inputs, f"{base}.onnx",
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

    # Step 2 — shape inference
    print("  Running shape inference...")
    if shape_infer == "subprocess":
        onnx_input = _shape_infer_subprocess(base)
    elif shape_infer == "fallback_copy":
        onnx_input = _shape_infer_fallback_copy(base)
    else:
        onnx_input = _shape_infer_inprocess(base)

    # Step 3 — import to torch MLIR
    print("  Importing ONNX to torch-mlir...")
    _import_onnx(base, onnx_input, opset)

    # Step 4 — lower to linalg
    print("  Lowering to linalg MLIR...")
    _lower_to_linalg(base, method=lowering_method)

    linalg_path = f"{base}_linalg.mlir"

    # Step 5 — post-process
    if do_postprocess:
        postprocess_linalg_file(linalg_path, static_batch_size)

    # Step 6 — cleanup
    _cleanup_onnx_intermediates(base, keep_onnx)

    return linalg_path


# ── Direct torch_mlir route ─────────────────────────────────────────────────

def direct_route(
    model: torch.nn.Module,
    model_name: str,
    example_inputs,
    output_dir: str,
    *,
    func_name: str = "main",
    try_compile: bool = True,
    static_batch_size: int = 0,
    do_postprocess: bool = True,
) -> str:
    """Convert via ``torch_mlir.compile`` (preferred) or ``export_and_import``.

    Parameters
    ----------
    model : torch.nn.Module
    model_name : str
    example_inputs : tensor or tuple of tensors
    output_dir : str
    func_name : str
        Function name for the MLIR module.
    try_compile : bool
        Whether to attempt ``torch_mlir.compile`` before falling back to
        ``torch_mlir.fx.export_and_import``.
    static_batch_size : int
        If > 0, replace ``?`` dims in the output.
    do_postprocess : bool
        Whether to call :func:`postprocess_linalg_file`.

    Returns
    -------
    str
        Path to the produced ``*_linalg.mlir`` file.
    """
    from torch_mlir.compiler_utils import OutputType

    print(f"  Using direct torch_mlir route for {model_name}...")
    mlir_module = None

    if try_compile:
        try:
            import torch_mlir
            if hasattr(torch_mlir, "compile") and callable(torch_mlir.compile):
                try:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                        use_tracing=True, func_name=func_name,
                    )
                except TypeError:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                        func_name=func_name,
                    )
        except Exception as e:
            print(f"  torch_mlir.compile failed: {e}")

    if mlir_module is None:
        from torch_mlir.fx import export_and_import
        kwargs = dict(
            output_type=OutputType.LINALG_ON_TENSORS,
            func_name=func_name,
        )
        if isinstance(example_inputs, tuple):
            mlir_module = export_and_import(model, *example_inputs, **kwargs)
        else:
            mlir_module = export_and_import(model, example_inputs, **kwargs)

    linalg_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
    with open(linalg_path, "w") as f:
        f.write(str(mlir_module))

    if do_postprocess:
        postprocess_linalg_file(linalg_path, static_batch_size)

    return linalg_path
