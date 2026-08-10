#!/usr/bin/env python3
"""
vision2mlir.py
--------------
Convert torchvision models to linalg-on-tensors MLIR.

Conversion pipeline (primary — ONNX route):
  PyTorch model → ONNX (torch.onnx.export)
                → shape inference (onnxruntime symbolic_shape_infer)
                → torch MLIR (torch-mlir-import-onnx)
                → linalg MLIR (torch-mlir-opt lowering passes)

Fallback (when torch-mlir CLI tools are unavailable):
  PyTorch model → linalg MLIR via torch_mlir.fx.export_and_import (in-process)

Supported models: resnet18, resnet50, efficientnet_b0, mobilenet_v3_small,
                  densenet121, vit_b_16, convnext_tiny, convnext_small,
                  convnext_base, convnext_large, mobilenet_v2, vgg11.
"""

import torch
import torchvision.models as models
import argparse
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "code_files")
NON_STRIPPED_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "non_stripped_models")

SUPPORTED_MODELS = [
    "resnet18", "resnet50",
    "efficientnet_b0",
    "mobilenet_v2", "mobilenet_v3_small",
    "densenet121",
    "vit_b_16",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "vgg11",
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _load_model(model_name: str) -> torch.nn.Module:
    loaders = {
        "resnet18":          lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet50":          lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "efficientnet_b0":   lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "mobilenet_v2":      lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        "mobilenet_v3_small":lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "densenet121":       lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "vit_b_16":          lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
        "convnext_tiny":     lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small":    lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
        "convnext_base":     lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT),
        "convnext_large":    lambda: models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT),
        "vgg11":             lambda: models.vgg11(weights=models.VGG11_Weights.DEFAULT),
    }
    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {SUPPORTED_MODELS}")
    return loaders[model_name]().eval()


# ---------------------------------------------------------------------------
# Primary: ONNX route
# ---------------------------------------------------------------------------

def convert_onnx_route(model, model_name: str, output_dir: str,
                        strip_weights: bool, opset: int = 18,
                        keep_onnx: bool = False) -> str:
    """Export model via ONNX → shape inference → torch-mlir → linalg MLIR.

    On success, intermediate ONNX files (.onnx, .onnx.data, _inferred.onnx)
    are deleted unless keep_onnx=True. On failure they are always kept so the
    pipeline can be resumed from step 3/4 without re-exporting from PyTorch.

    Returns:
        Path to the produced *_linalg.mlir file.
    """
    import onnx
    try:
        import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
        _has_shape_infer = True
    except ImportError:
        _has_shape_infer = False

    base = os.path.join(output_dir, model_name)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Step 1 — export to ONNX
    print(f"  Exporting {model_name} to ONNX (opset {opset})...")
    torch.onnx.export(
        model, dummy_input, f"{base}.onnx",
        opset_version=opset,
        input_names=["input"], output_names=["output"],
    )

    # Step 2 — shape inference
    onnx_input = f"{base}.onnx"
    if _has_shape_infer:
        print("  Running shape inference...")
        model_proto = onnx.load(f"{base}.onnx")
        inferred = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(
            model_proto, auto_merge=True, guess_output_rank=True
        )
        onnx_input = f"{base}_inferred.onnx"
        onnx.save(inferred, onnx_input)

    # Step 3 — import to torch MLIR
    print("  Importing ONNX to torch-mlir...")
    result = subprocess.run(
        ["torch-mlir-import-onnx", onnx_input, "-o", f"{base}_torch.mlir"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"torch-mlir-import-onnx failed:\n{result.stderr}")

    # Step 4 — lower to linalg
    print("  Lowering to linalg MLIR...")
    opt_cmd = [
        "torch-mlir-opt",
        f"{base}_torch.mlir",
        "--convert-torch-onnx-to-torch",
        "--torch-decompose-complex-ops",
        "--convert-torch-to-linalg",
        "--torch-backend-to-linalg-on-tensors-backend-pipeline",
        "-o", f"{base}_linalg.mlir",
    ]

    result = subprocess.run(opt_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"torch-mlir-opt failed:\n{result.stderr}")

    # Clean up intermediate ONNX files on success
    if not keep_onnx:
        for candidate in [f"{base}.onnx", f"{base}.onnx.data", f"{base}_inferred.onnx"]:
            if os.path.exists(candidate):
                os.remove(candidate)
                print(f"  Removed intermediate: {os.path.basename(candidate)}")

    return f"{base}_linalg.mlir"


# ---------------------------------------------------------------------------
# Fallback: direct torch_mlir API
# ---------------------------------------------------------------------------

def convert_direct_route(model, model_name: str, output_dir: str,
                          strip_weights: bool) -> str:
    """Export model directly using torch_mlir.fx.export_and_import.

    Returns:
        Path to the produced .mlir file.
    """
    from torch_mlir.fx import export_and_import
    from torch_mlir.compiler_utils import OutputType

    print(f"  Using direct torch_mlir route for {model_name}...")
    dummy_input = torch.randn(1, 3, 224, 224)
    mlir_module = export_and_import(
        model, dummy_input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model_name,
    )
    output_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
    mlir_str = str(mlir_module)
    with open(output_path, 'w') as f:
        f.write(mlir_str)
    return output_path


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _backup_and_maybe_strip(output_path: str, strip_weights_flag: bool, verbose: bool):
    """Save a non-stripped backup, then strip the output file if enabled."""
    import shutil
    os.makedirs(NON_STRIPPED_DIR, exist_ok=True)
    backup_path = os.path.join(NON_STRIPPED_DIR, os.path.basename(output_path))
    shutil.copy2(output_path, backup_path)
    print(f"  Non-stripped backup: {backup_path}")

    if not strip_weights_flag:
        return
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > 1:  # strip anything non-trivially large
        print(f"  File is {size_mb:.1f} MB — stripping weights...")
        from data_utils.strip_mlir import strip_weights
        reduction = strip_weights(output_path, output_path, verbose=verbose)
        print(f"  Stripped: {reduction:.1f}% reduction.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert torchvision models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=SUPPORTED_MODELS,
        help="Vision model to convert.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend. 'onnx' (default) uses the ONNX route; "
             "'direct' uses the torch_mlir Python API directly.",
    )
    parser.add_argument(
        "--strip-weights", action="store_true", default=True,
        help="Strip large weight constants (default: enabled).",
    )
    parser.add_argument(
        "--keep-onnx", action="store_true", default=False,
        help="Keep intermediate .onnx/.onnx.data/_inferred.onnx files after a "
             "successful pipeline run (default: delete on success).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.model}...")
    model = _load_model(args.model)

    output_path = None
    if args.backend == "onnx":
        try:
            output_path = convert_onnx_route(
                model, args.model, args.output_dir, args.strip_weights,
                keep_onnx=args.keep_onnx,
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct torch_mlir route...")
            output_path = convert_direct_route(
                model, args.model, args.output_dir, args.strip_weights
            )
    else:
        output_path = convert_direct_route(
            model, args.model, args.output_dir, args.strip_weights
        )

    _backup_and_maybe_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
