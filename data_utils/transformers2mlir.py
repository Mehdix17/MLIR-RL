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

Supported models: see data_utils.model_catalog.TRANSFORMER_MODELS.
"""

import torch
import torch.nn as nn
import argparse
import os
import traceback

from data_utils.model_catalog import TRANSFORMER_MODELS, MODEL_REGISTRY, TRANSFORMER_INPUT_SPECS
from data_utils._convert_common import (
    PROJECT_ROOT,
    onnx_route,
    direct_route,
    backup_and_strip,
)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "code_files")


# ---------------------------------------------------------------------------
# Model + input factory
# ---------------------------------------------------------------------------

def _hf_repo(model_name: str) -> str:
    """Get HuggingFace repo ID from the catalog."""
    entry = MODEL_REGISTRY.get(model_name, {})
    return entry.get("hf_repo", model_name)


def _load_model_and_inputs(model_name: str):
    """Return (model, example_inputs, input_names, output_names)."""
    name = model_name.lower()
    repo = _hf_repo(name)

    if name == "bert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        inputs = tokenizer(
            "Hello from MLIR", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "distilbert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "roberta":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "albert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "gpt2":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModel.from_pretrained(repo).eval()
        base_model.config.use_cache = False

        class _GPT2ExportWrapper(nn.Module):
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
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        input_ids = tokenizer(
            "Studies show that owning a dog is good for you",
            padding="max_length", max_length=16, truncation=True,
            return_tensors="pt",
        ).input_ids
        return model, (input_ids,), ["input_ids"], ["last_hidden_state"]

    if name == "bart":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        model.config.use_cache = False
        enc = tokenizer(
            ["The cat sat on the mat."],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=16,
        )
        example_inputs = (enc["input_ids"], enc["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "llama3_2_1b":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.float32
        ).eval()
        model.config.use_cache = False
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        return model, (encoded["input_ids"],), ["input_ids"], ["logits"]

    if name == "whisper_base":
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained(repo).eval()
        encoder = model.encoder
        dummy_input = torch.randn(1, 80, 3000)
        return encoder, (dummy_input,), ["input_features"], ["last_hidden_state"]

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

    raise ValueError(f"Unknown model: {model_name}. Choose from: {TRANSFORMER_MODELS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace / transformer models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="distilbert", choices=TRANSFORMER_MODELS,
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
            output_path = onnx_route(
                model, args.model, example_inputs, args.output_dir,
                input_names=input_names, output_names=output_names,
                opset=17, keep_onnx=args.keep_onnx,
                shape_infer="subprocess",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            traceback.print_exc()
            print("  Falling back to direct torch_mlir route...")
            output_path = direct_route(
                model, args.model, example_inputs, args.output_dir,
            )
    else:
        output_path = direct_route(
            model, args.model, example_inputs, args.output_dir,
        )

    backup_and_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
