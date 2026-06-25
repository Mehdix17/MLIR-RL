#!/usr/bin/env python3
"""
vision2mlir.py
--------------
Convert vision models to linalg-on-tensors MLIR.

Conversion pipeline (primary — ONNX route):
  PyTorch model → ONNX (torch.onnx.export)
                → shape inference (onnxruntime symbolic_shape_infer)
                → torch MLIR (torch-mlir-import-onnx)
                → linalg MLIR (torch-mlir-opt lowering passes)

Fallback (when torch-mlir CLI tools are unavailable):
  PyTorch model → linalg MLIR via torch_mlir.fx.export_and_import (in-process)

Supported models: see data_utils.model_catalog.VISION_MODELS.
"""

import torch
import torchvision.models as tv_models
import argparse
import os

from data_utils.model_catalog import VISION_MODELS, MODEL_REGISTRY
from data_utils._convert_common import (
    PROJECT_ROOT,
    onnx_route,
    direct_route,
    backup_and_strip,
)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "code_files")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_TV_WEIGHTS = {
    "resnet18":           tv_models.ResNet18_Weights,
    "resnet50":           tv_models.ResNet50_Weights,
    "resnext50_32x4d":    tv_models.ResNeXt50_32X4D_Weights,
    "efficientnet_b0":    tv_models.EfficientNet_B0_Weights,
    "mobilenet_v2":       tv_models.MobileNet_V2_Weights,
    "mobilenet_v3_small": tv_models.MobileNet_V3_Small_Weights,
    "densenet121":        tv_models.DenseNet121_Weights,
    "vit_b_16":           tv_models.ViT_B_16_Weights,
    "convnext_tiny":      tv_models.ConvNeXt_Tiny_Weights,
    "convnext_small":     tv_models.ConvNeXt_Small_Weights,
    "convnext_base":      tv_models.ConvNeXt_Base_Weights,
    "convnext_large":     tv_models.ConvNeXt_Large_Weights,
    "vgg11":              tv_models.VGG11_Weights,
    "vgg16":              tv_models.VGG16_Weights,
}


def _load_model(model_name: str) -> torch.nn.Module:
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None or entry["category"] != "vision":
        raise ValueError(f"Unknown vision model: {model_name}. Choose from: {VISION_MODELS}")

    framework = entry["framework"]

    if framework == "torchvision":
        tv_name = entry["torchvision_name"]
        weights_cls = _TV_WEIGHTS.get(tv_name)
        loader_fn = getattr(tv_models, tv_name)
        return loader_fn(weights=weights_cls.DEFAULT).eval()

    if framework == "ultralytics":
        from ultralytics import YOLO
        yolo = YOLO(f"{model_name}.pt")
        return yolo.model.eval()

    raise ValueError(f"Unsupported framework '{framework}' for {model_name}")


def _get_dummy_input_size(model_name: str) -> int:
    entry = MODEL_REGISTRY.get(model_name, {})
    return entry.get("input_shape", (224, 224))[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert vision models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=VISION_MODELS,
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
    dummy_input_size = _get_dummy_input_size(args.model)
    dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)

    output_path = None
    if args.backend == "onnx":
        try:
            output_path = onnx_route(
                model, args.model, dummy_input, args.output_dir,
                input_names=["input"], output_names=["output"],
                opset=18, keep_onnx=args.keep_onnx,
                shape_infer="inprocess",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct torch_mlir route...")
            output_path = direct_route(
                model, args.model, dummy_input, args.output_dir,
                func_name=args.model, try_compile=False,
            )
    else:
        output_path = direct_route(
            model, args.model, dummy_input, args.output_dir,
            func_name=args.model, try_compile=False,
        )

    backup_and_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
