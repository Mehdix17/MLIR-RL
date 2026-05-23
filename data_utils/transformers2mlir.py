#!/usr/bin/env python3
"""
transformers2mlir.py
--------------------
Convert HuggingFace / sequence transformer models to linalg-on-tensors MLIR.

Conversion pipeline (primary — ONNX route):
  PyTorch model → ONNX (torch.onnx.export)
                → shape inference (onnxruntime symbolic_shape_infer)
                → torch MLIR (torch-mlir-import-onnx / torch_mlir.tools.import_onnx)
                → linalg MLIR (torch-mlir-opt lowering passes)

Fallback (when CLI tools are unavailable):
  PyTorch model → linalg MLIR via torch_mlir.compile or torch_mlir.fx.export_and_import

Supported models: bert, distilbert, roberta, albert, gpt2, t5, bart, lstm.
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import subprocess
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "code_files")
NON_STRIPPED_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "non_stripped_models")

SUPPORTED_MODELS = [
    "bert", "distilbert", "roberta", "albert",
    "gpt2", "t5", "bart", "lstm",
]


# ---------------------------------------------------------------------------
# Model + input factory
# ---------------------------------------------------------------------------

def _load_model_and_inputs(model_name: str):
    """Return (model, example_inputs, input_names, output_names)."""
    name = model_name.lower()

    if name == "bert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").eval()
        inputs = tokenizer(
            "Hello from MLIR", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "distilbert":
        from transformers import DistilBertTokenizer, DistilBertModel
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "roberta":
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "albert":
        from transformers import AlbertTokenizer, AlbertModel
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        model = AlbertModel.from_pretrained("albert-base-v2").eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "gpt2":
        from transformers import GPT2Tokenizer, GPT2Model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2Model.from_pretrained("gpt2-medium").eval()
        base_model.config.use_cache = False

        class _GPT2ExportWrapper(nn.Module):
            """Wrap GPT2Model to avoid past_key_values tracing issues."""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                cache_position = torch.arange(
                    0, input_ids.shape[1], device=input_ids.device
                )
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                ).last_hidden_state

        model = _GPT2ExportWrapper(base_model).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "t5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5EncoderModel.from_pretrained("t5-small").eval()
        input_ids = tokenizer(
            "Studies show that owning a dog is good for you",
            padding="max_length", max_length=16, truncation=True,
            return_tensors="pt",
        ).input_ids
        return model, (input_ids,), ["input_ids"], ["last_hidden_state"]

    if name == "bart":
        from transformers import BartTokenizer, BartForConditionalGeneration
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").eval()
        model.config.use_cache = False
        enc = tokenizer(
            ["The cat sat on the mat."],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=16,
        )
        example_inputs = (enc["input_ids"], enc["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "lstm":
        class LSTMSeq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(512, 768, 2, batch_first=True)
                self.decoder = nn.LSTM(768, 768, 2, batch_first=True)
                self.proj    = nn.Linear(768, 512)
            def forward(self, x):
                out, (h, c) = self.encoder(x)
                dec, _      = self.decoder(out, (h, c))
                return self.proj(dec)
        model = LSTMSeq2Seq().eval()
        example_inputs = torch.randn(1, 100, 512)
        return model, example_inputs, ["input"], ["output"]

    raise ValueError(f"Unknown model: {model_name}. Choose from: {SUPPORTED_MODELS}")


# ---------------------------------------------------------------------------
# Primary: ONNX route
# ---------------------------------------------------------------------------

def convert_onnx_route(model, model_name: str, example_inputs,
                        input_names, output_names,
                        output_dir: str, strip_weights: bool,
                        opset: int = 17, keep_onnx: bool = False) -> str:
    """Export via ONNX → shape inference → torch-mlir → linalg MLIR.

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

    # Step 1 — ONNX export (JIT trace for large-model protobuf stability)
    print(f"  Exporting {model_name} to ONNX (opset {opset})...")
    torch.onnx.export(
        model, example_inputs, f"{base}.onnx",
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=False,
    )
    try:
        onnx.checker.check_model(f"{base}.onnx")
    except Exception as e:
        print(f"  ONNX checker warning (non-fatal): {e}")

    # Step 2 — shape inference
    onnx_input = f"{base}.onnx"
    if _has_shape_infer:
        print("  Running shape inference...")
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
            onnx_input = f"{base}_inferred.onnx"
        else:
            print(f"  Shape inference warning: {result.stderr}")

    # Step 3 — import to torch MLIR (prefer CLI, fall back to Python module)
    print("  Importing ONNX to torch-mlir...")
    import_cmd = [
        "torch-mlir-import-onnx", onnx_input,
        "-o", f"{base}_torch.mlir",
        "--opset-version", str(opset),
    ]
    result = subprocess.run(import_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try Python module fallback
        result2 = subprocess.run(
            [sys.executable, "-m", "torch_mlir.tools.import_onnx",
             onnx_input, "-o", f"{base}_torch.mlir",
             "--opset-version", str(opset)],
            capture_output=True, text=True,
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"ONNX import failed:\n{result.stderr}\n{result2.stderr}"
            )

    # Step 4 — lower to linalg
    print("  Lowering to linalg MLIR...")
    opt_flags = [
        "--convert-torch-onnx-to-torch",
        "--torch-decompose-complex-ops",
        "--convert-torch-to-linalg",
        "--torch-backend-to-linalg-on-tensors-backend-pipeline",
    ]

    result = subprocess.run(
        ["torch-mlir-opt", f"{base}_torch.mlir"] + opt_flags + ["-o", f"{base}_linalg.mlir"],
        capture_output=True, text=True,
    )
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
# Fallback: direct torch_mlir Python API
# ---------------------------------------------------------------------------

def convert_direct_route(model, model_name: str, example_inputs,
                          output_dir: str, strip_weights: bool) -> str:
    """Convert using torch_mlir.compile (or fx.export_and_import as sub-fallback).

    Returns:
        Path to the produced .mlir file.
    """
    from torch_mlir.compiler_utils import OutputType

    print(f"  Using direct torch_mlir route for {model_name}...")
    mlir_module = None

    # Try torch_mlir.compile first
    if hasattr(torch, '__version__'):
        try:
            import torch_mlir
            if hasattr(torch_mlir, 'compile') and callable(torch_mlir.compile):
                try:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                        use_tracing=True,
                    )
                except TypeError:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                    )
        except Exception as e:
            print(f"  torch_mlir.compile failed: {e}")

    # Sub-fallback: fx.export_and_import
    if mlir_module is None:
        from torch_mlir.fx import export_and_import
        if isinstance(example_inputs, tuple):
            mlir_module = export_and_import(
                model, *example_inputs,
                output_type=OutputType.LINALG_ON_TENSORS,
            )
        else:
            mlir_module = export_and_import(
                model, example_inputs,
                output_type=OutputType.LINALG_ON_TENSORS,
            )

    mlir_str = str(mlir_module)
    output_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
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
        description="Convert HuggingFace / transformer models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="distilbert", choices=SUPPORTED_MODELS,
        help="Transformer model to convert.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend.",
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
    model, example_inputs, input_names, output_names = _load_model_and_inputs(args.model)

    output_path = None
    if args.backend == "onnx":
        try:
            output_path = convert_onnx_route(
                model, args.model, example_inputs, input_names, output_names,
                args.output_dir, args.strip_weights,
                keep_onnx=args.keep_onnx,
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            traceback.print_exc()
            print("  Falling back to direct torch_mlir route...")
            output_path = convert_direct_route(
                model, args.model, example_inputs, args.output_dir, args.strip_weights
            )
    else:
        output_path = convert_direct_route(
            model, args.model, example_inputs, args.output_dir, args.strip_weights
        )

    _backup_and_maybe_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
