#!/usr/bin/env python3
"""
measure_full_model_baselines.py
--------------------------------
Measure PyTorch (eager, JIT) execution times for all models in
data/nn/raw_bench/*_linalg.mlir.  Reads cached MLIR baselines from
results/full_model/baselines/full_model.json.

Output: CSV with columns:
  model, mlir_baseline_ns, pytorch_eager_ns, pytorch_jit_ns, mlir_rl_opt_ns

Timing: 10 warmup + 20 measure iterations → median.
"""

import os, sys, json, time, gc, csv
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cpu")
WARMUP = 10
MEASURE = 20

# ---------------------------------------------------------------------------
# Model + input factory
# ---------------------------------------------------------------------------

def _vision_model(name):
    import torchvision.models as tv_models
    loaders = {
        "vgg11":              lambda: tv_models.vgg11(weights=tv_models.VGG11_Weights.IMAGENET1K_V1).eval(),
        "resnet18":           lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1).eval(),
        "resnet50":           lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1).eval(),
        "efficientnet_b0":    lambda: tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval(),
        "mobilenet_v3_small": lambda: tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval(),
        "resnext50":          lambda: tv_models.resnext50_32x4d(weights=tv_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).eval(),
        "convnext_tiny":      lambda: tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval(),
        "densenet121":        lambda: tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1).eval(),
    }
    if name in loaders:
        return loaders[name]()
    if name == "vit_b_16":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1).eval()
        class ViTWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                # Return plain tensor for JIT compatibility
                if hasattr(out, 'logits'):
                    return out.logits
                return out
        return ViTWrapper(model)
    raise ValueError(f"Unknown vision model: {name}")

def _vision_input():
    return (torch.randn(1, 3, 224, 224),)

def _load_t5():
    from transformers import T5Tokenizer, T5EncoderModel
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5EncoderModel.from_pretrained("t5-small").eval()
    ids = tokenizer("Studies show that owning a dog is good for you",
                     padding="max_length", max_length=16, truncation=True,
                     return_tensors="pt").input_ids
    # Wrap to return tensor only (not dict) for JIT
    class T5Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).last_hidden_state
    return T5Wrapper(model), (ids,)

def _load_gpt2():
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2").eval()
    model.config.use_cache = False
    enc = tokenizer("Hello from MLIR-RL!", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class GPT2Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            out = self.m(input_ids=ids, attention_mask=mask)
            return out.last_hidden_state
    return GPT2Wrapper(model), (input_ids, attention_mask)

def _load_bert():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()
    enc = tokenizer("Hello from MLIR", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class BertWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    return BertWrapper(model), (input_ids, attention_mask)

def _load_distilbert():
    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
    enc = tokenizer("Hello from MLIR-RL!", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class DistilBertWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    return DistilBertWrapper(model), (input_ids, attention_mask)

def _load_roberta():
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").eval()
    enc = tokenizer("Hello from MLIR-RL!", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class RobertaWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    return RobertaWrapper(model), (input_ids, attention_mask)

def _load_albert():
    from transformers import AlbertTokenizer, AlbertModel
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertModel.from_pretrained("albert-base-v2").eval()
    enc = tokenizer("Hello from MLIR-RL!", return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class AlbertWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    return AlbertWrapper(model), (input_ids, attention_mask)

def _load_bart():
    from transformers import BartTokenizer, BartModel
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartModel.from_pretrained("facebook/bart-base").eval()
    enc = tokenizer(["The cat sat on the mat."], return_tensors="pt",
                     padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    class BartWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state
    return BartWrapper(model), (input_ids, attention_mask)

def _load_lstm():
    class LSTMSeq2Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.LSTM(512, 768, 2, batch_first=True)
            self.decoder = torch.nn.LSTM(768, 768, 2, batch_first=True)
            self.proj = torch.nn.Linear(768, 512)
        def forward(self, x):
            out, (h, c) = self.encoder(x)
            dec, _ = self.decoder(out, (h, c))
            return self.proj(dec)
    model = LSTMSeq2Seq().eval()
    return model, (torch.randn(1, 100, 512),)

def _load_gcn():
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_utils.gnn2mlir import GCN
    model = GCN().eval()
    return model, (torch.randn(128, 64), torch.randn(128, 128))

def _load_gat():
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_utils.gnn2mlir import GAT
    model = GAT().eval()
    return model, (torch.randn(128, 64), torch.randn(128, 128))

MODEL_LOADERS = {
    # Vision (9)
    "vgg11":              lambda: (_vision_model("vgg11"),              _vision_input()),
    "resnet18":           lambda: (_vision_model("resnet18"),           _vision_input()),
    "resnet50":           lambda: (_vision_model("resnet50"),           _vision_input()),
    "efficientnet_b0":    lambda: (_vision_model("efficientnet_b0"),    _vision_input()),
    "mobilenet_v3_small": lambda: (_vision_model("mobilenet_v3_small"), _vision_input()),
    "resnext50":          lambda: (_vision_model("resnext50"),          _vision_input()),
    "convnext_tiny":      lambda: (_vision_model("convnext_tiny"),      _vision_input()),
    "densenet121":        lambda: (_vision_model("densenet121"),        _vision_input()),
    "vit_b_16":           lambda: (_vision_model("vit_b_16"),           _vision_input()),
    # Transformers (6)
    "t5":                 _load_t5,
    "gpt2":               _load_gpt2,
    "bert":               _load_bert,
    "distilbert":         _load_distilbert,
    "roberta":            _load_roberta,
    "albert":             _load_albert,
    "bart":               _load_bart,
    # LSTM (1)
    "lstm":               _load_lstm,
    # GNN (2)
    "gcn":                _load_gcn,
    "gat":                _load_gat,
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _run_model(model, inputs):
    """Run model with inputs (tuple or single tensor)."""
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)

def _time_model(model, inputs, n_warmup=WARMUP, n_measure=MEASURE):
    """Return median execution time in ns."""
    # Warmup
    for _ in range(n_warmup):
        _run_model(model, inputs)

    # Measure
    times = []
    for _ in range(n_measure):
        start = time.perf_counter_ns()
        _run_model(model, inputs)
        end = time.perf_counter_ns()
        times.append(end - start)

    return int(np.median(times))

def _time_jit(model, inputs, n_warmup=WARMUP, n_measure=MEASURE):
    """Trace + time. Falls back to script if trace fails. Returns median ns or None."""
    try:
        traced = torch.jit.trace(model, inputs)
        traced = traced.eval()
        for _ in range(n_warmup + 10):
            _run_model(traced, inputs)
        return _time_model(traced, inputs, n_warmup, n_measure)
    except Exception:
        pass
    # Fallback: script
    try:
        scripted = torch.jit.script(model)
        scripted = scripted.eval()
        for _ in range(n_warmup + 10):
            _run_model(scripted, inputs)
        return _time_model(scripted, inputs, n_warmup, n_measure)
    except Exception as e:
        print(f"    JIT script also failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read cached MLIR baselines
    baseline_json = PROJECT_ROOT / "results" / "full_model" / "baselines" / "full_model.json"
    mlir_baselines = {}
    if baseline_json.exists():
        raw = json.loads(baseline_json.read_text())
        for k, v in raw.items():
            if isinstance(v, dict) and v.get("baseline_ns"):
                mlir_baselines[k] = v["baseline_ns"]

    output_path = PROJECT_ROOT / "results" / "full_model" / "baselines" / "full_baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for name in sorted(MODEL_LOADERS.keys()):
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        # MLIR baseline (cached only)
        mlir_ns = mlir_baselines.get(name, None)
        if mlir_ns:
            print(f"  MLIR baseline (cached): {mlir_ns:,} ns")
        else:
            print(f"  MLIR baseline: not available")

        # PyTorch
        try:
            print(f"  Loading PyTorch model ...")
            model, inputs = MODEL_LOADERS[name]()

            # Move to device
            model = model.to(DEVICE)
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(DEVICE) for x in inputs)
            else:
                inputs = inputs.to(DEVICE)

            print(f"  Timing eager ({WARMUP} warmup + {MEASURE} measure) ...")
            eager_ns = _time_model(model, inputs)
            print(f"  PyTorch eager: {eager_ns:,} ns")

            print(f"  Timing JIT ({WARMUP} warmup + {MEASURE} measure) ...")
            jit_ns = _time_jit(model, inputs)
            if jit_ns:
                print(f"  PyTorch JIT:   {jit_ns:,} ns")
            else:
                print(f"  PyTorch JIT:   FAILED")

            del model
            gc.collect()

            results.append({
                "model": name,
                "mlir_baseline_ns": mlir_ns if mlir_ns else "",
                "pytorch_eager_ns": eager_ns,
                "pytorch_jit_ns": jit_ns if jit_ns else "",
                "mlir_rl_opt_ns": "",
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model": name,
                "mlir_baseline_ns": mlir_ns if mlir_ns else "",
                "pytorch_eager_ns": "",
                "pytorch_jit_ns": "",
                "mlir_rl_opt_ns": "",
            })

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "mlir_baseline_ns", "pytorch_eager_ns",
            "pytorch_jit_ns", "mlir_rl_opt_ns"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved → {output_path}")
    print(f"\n{'Model':<20s} {'MLIR (ns)':>15s} {'Eager (ns)':>15s} {'JIT (ns)':>15s}")
    print("-" * 67)
    for r in results:
        mlir = f"{int(r['mlir_baseline_ns']):>15,}" if r['mlir_baseline_ns'] else "           N/A"
        eag  = f"{int(r['pytorch_eager_ns']):>15,}" if r['pytorch_eager_ns'] else "           N/A"
        jit  = f"{int(r['pytorch_jit_ns']):>15,}" if r['pytorch_jit_ns'] else "           N/A"
        print(f"{r['model']:<20s} {mlir} {eag} {jit}")


if __name__ == "__main__":
    main()
