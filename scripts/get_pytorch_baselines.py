#!/usr/bin/env python3
"""
get_pytorch_baselines.py
-------------------------
Measures PyTorch eager + JIT execution times for models listed in
results/full_model/pytorch_models.json.

Output: results/full_model/pytorch_times.json
  {"albert": {"eager_ns": ..., "jit_ns": ...}, ...}

Usage:
  python scripts/get_pytorch_baselines.py
  python scripts/get_pytorch_baselines.py --models gpt2-large gpt2-medium
  python scripts/get_pytorch_baselines.py --models vit_b_16 --output /tmp/test.json
"""

import os, sys, json, time, gc, argparse
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cpu")
WARMUP = 10
MEASURE = 20

# ---------------------------------------------------------------------------
# Generic vision input (3x224x224 random)
# ---------------------------------------------------------------------------
def _vision_input():
    return (torch.randn(1, 3, 224, 224),)


# ---------------------------------------------------------------------------
# Vision models
# ---------------------------------------------------------------------------
def _load_vision(model_name, cfg):
    import torchvision.models as tv_models
    fn = getattr(tv_models, cfg["fn"])
    # Handle dot-notation in weights (e.g. "ResNet18_Weights.IMAGENET1K_V1")
    weights_parts = cfg["weights"].split(".")
    weights_obj = tv_models
    for part in weights_parts:
        weights_obj = getattr(weights_obj, part)
    model = fn(weights=weights_obj).eval()
    if model_name == "vit_b_16":
        model = _VitWrapper(model)
        return model, _vision_input()
    return model, _vision_input()


class _VitWrapper(torch.nn.Module):
    """Only for eager mode — JIT fails on ViT. trace: graph diff, script: internal ops."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out.logits


# ---------------------------------------------------------------------------
# Transformer models
# ---------------------------------------------------------------------------
def _load_transformer(model_name, cfg):
    transformers = __import__("transformers")
    tokenizer_cls = getattr(transformers, cfg["tokenizer"])
    model_cls = getattr(transformers, cfg["cls"])

    tokenizer = tokenizer_cls.from_pretrained(cfg["hf"])
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(cfg["hf"]).eval()

    if "gpt2" in model_name and "use_cache" in dir(model.config):
        model.config.use_cache = False

    text = "Studies show that owning a dog is good for you"
    enc = tokenizer(text, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    mask = enc.get("attention_mask", None)

    if mask is not None:
        model = _HFWrapper(model, has_mask=True)
        inputs = (input_ids, mask)
    else:
        model = _HFWrapper(model, has_mask=False)
        inputs = (input_ids,)

    return model, inputs


class _HFWrapper(torch.nn.Module):
    def __init__(self, m, has_mask=True):
        super().__init__()
        self.m = m
        self.has_mask = has_mask

    def forward(self, ids, mask=None):
        if self.has_mask:
            out = self.m(input_ids=ids, attention_mask=mask)
        else:
            out = self.m(ids)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


# ---------------------------------------------------------------------------
# LSTM (custom)
# ---------------------------------------------------------------------------
def _load_lstm(model_name, cfg):
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

    return LSTMSeq2Seq().eval(), (torch.randn(1, 100, 512),)


# ---------------------------------------------------------------------------
# GNN (local)
# ---------------------------------------------------------------------------
def _load_gnn(model_name, cfg):
    sys.path.insert(0, str(PROJECT_ROOT))
    mod = __import__(cfg["src"], fromlist=[cfg["cls"]])
    cls = getattr(mod, cfg["cls"])
    model = cls().eval()
    return model, (torch.randn(128, 64), torch.randn(128, 128))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _run_model(model, inputs):
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _time_model(model, inputs):
    for _ in range(WARMUP):
        _run_model(model, inputs)
    times = []
    for _ in range(MEASURE):
        start = time.perf_counter_ns()
        _run_model(model, inputs)
        times.append(time.perf_counter_ns() - start)
    return int(np.median(times))


def _time_jit(model, inputs):
    for method in ["trace", "script"]:
        try:
            if method == "trace":
                jitted = torch.jit.trace(model, inputs)
            else:
                jitted = torch.jit.script(model)
            jitted = jitted.eval()
            for _ in range(WARMUP + 10):
                _run_model(jitted, inputs)
            return _time_model(jitted, inputs)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
LOADERS = {
    "vision":      _load_vision,
    "transformer": _load_transformer,
    "lstm":        _load_lstm,
    "gnn":         _load_gnn,
}


def process_model(model_name, cfg):
    """Load + time a single model. Returns (eager_ns, jit_ns | None)."""
    # Find the model in the config
    model_type = None
    model_cfg = None
    for mt, models in cfg.items():
        if isinstance(models, dict) and model_name in models:
            model_type = mt
            model_cfg = models[model_name]
            break
    if model_type is None or model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in config")

    loader = LOADERS[model_type]
    model, inputs = loader(model_name, model_cfg)

    model = model.to(DEVICE)
    if isinstance(inputs, tuple):
        inputs = tuple(x.to(DEVICE) for x in inputs)
    else:
        inputs = inputs.to(DEVICE)

    print(f"    Eager  ({WARMUP} warmup + {MEASURE} measure) ...", end=" ", flush=True)
    eager_ns = _time_model(model, inputs)
    print(f"{eager_ns:,} ns")

    print(f"    JIT    ({WARMUP} warmup + {MEASURE} measure) ...", end=" ", flush=True)
    jit_ns = _time_jit(model, inputs)
    if jit_ns:
        print(f"{jit_ns:,} ns")
    else:
        print("FAILED")

    del model
    gc.collect()
    return eager_ns, jit_ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch eager + JIT baseline measurement")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to measure (default: all in config)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "results/full_model/pytorch_models.json"),
                        help="Config JSON mapping models to loaders")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results/full_model_1/pytorch_times.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Collect all model names from config
    all_models = set()
    for model_type, models in cfg.items():
        if isinstance(models, dict):
            all_models.update(models.keys())

    target_models = args.models if args.models else sorted(all_models)
    # Validate
    unknown = set(target_models) - all_models
    if unknown:
        print(f"Unknown models (not in config): {unknown}")
        sys.exit(1)

    # Load existing results if any
    output_path = Path(args.output)
    results = {}
    if output_path.exists():
        results = json.loads(output_path.read_text())
        already = set(results.keys()) & set(target_models)
        if already:
            print(f"Skipping {len(already)} models already measured: {sorted(already)}")

    pending = sorted(set(target_models) - set(results.keys()))
    if not pending:
        print("All models already measured.")
        return

    print(f"Models to measure: {len(pending)}")
    print(f"Device: {DEVICE}")

    for model_name in pending:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            eager_ns, jit_ns = process_model(model_name, cfg)
            results[model_name] = {
                "eager_ns": eager_ns,
                "jit_ns": jit_ns if jit_ns else None,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"eager_ns": None, "jit_ns": None, "error": str(e)}

        # Save incrementally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Model':<22s} {'Eager (ns)':>15s} {'JIT (ns)':>15s}")
    print("-" * 55)
    for model_name in sorted(results):
        r = results[model_name]
        e = f"{r['eager_ns']:>15,}" if r.get("eager_ns") else "           FAIL"
        j = f"{r['jit_ns']:>15,}" if r.get("jit_ns") else "           FAIL"
        print(f"{model_name:<22s} {e} {j}")

    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
