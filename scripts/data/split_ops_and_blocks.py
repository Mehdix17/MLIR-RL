"""Stratified split for ops_and_blocks dataset.

Multi-level stratification:
  - Category: single_op vs block
  - Within single_op: model + operation type
  - Within block: model

Usage:
    python scripts/data/split_ops_and_blocks.py \\
        --input results/ops_and_blocks_results/baselines/mlir/base.json \\
        --output-dir results/ops_and_blocks_results/baselines/mlir/ \\
        --eval-ratio 0.2 --seed 42
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

KNOWN_MODELS = [
    "albert", "bart", "bert", "convnext_tiny", "densen121", "deberta", "distilbert",
    "efficientnet_b0", "gat", "gcn", "gin", "gpt2", "gpt2-large", "gpt2-medium",
    "llama3_2_1b", "lstm", "mobilenet_v3_small", "resnet18", "resnet50", "resnext50",
    "roberta", "t5", "vgg11", "vgg16", "vit_b_16", "whisper_base", "yolov8m",
]

parser = argparse.ArgumentParser(description="Split ops_and_blocks dataset with multi-level stratification.")
parser.add_argument("--input", required=True, help="Path to base exec times JSON")
parser.add_argument("--output-dir", required=True, help="Directory to write train/eval JSONs")
parser.add_argument("--eval-ratio", type=float, default=0.2, help="Fraction for eval (default: 0.2)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

with open(args.input) as f:
    data = json.load(f)

valid = {k: v for k, v in data.items() if v > 0}
dropped = len(data) - len(valid)
if dropped:
    print(f"Dropped {dropped} failed entries (exec_time <= 0)")


def classify(bench_name):
    """Return (category, model, op_type) for stratification."""
    # Check if block
    is_block = "_block_" in bench_name
    # Extract model prefix
    model = None
    for m in sorted(KNOWN_MODELS, key=len, reverse=True):
        if bench_name.startswith(m + "_"):
            model = m
            break
    # Extract operation type
    if model:
        rest = bench_name[len(model) + 1:]
    else:
        rest = bench_name
    if is_block:
        return ("block", model or "unknown", "block")
    # Single op
    if model:
        # Model-prefixed: model_op_index → op is everything before last _<digits>
        rest_no_idx = re.sub(r'_\d+$', '', rest)
        op = rest_no_idx
    else:
        # Synthetic: op_dims → op is first token
        op = rest.split("_")[0]
    return ("single_op", model or "synthetic", op)


# Group by stratification key
groups = defaultdict(list)
for k in valid:
    groups[classify(k)].append(k)

random.seed(args.seed)
train_keys = []
eval_keys = []

for strat_key, keys in sorted(groups.items()):
    cat, model, op = strat_key
    random.shuffle(keys)
    n_eval = max(1, round(len(keys) * args.eval_ratio))
    if n_eval >= len(keys):
        n_eval = max(1, len(keys) // 2)
    eval_keys.extend(keys[:n_eval])
    train_keys.extend(keys[n_eval:])
    print(f"  {cat:<12s} {model:<20s} {op:<15s}  total={len(keys):4d}  train={len(keys)-n_eval:4d}  eval={n_eval:4d}")

train_data = {k: valid[k] for k in train_keys}
eval_data = {k: valid[k] for k in eval_keys}

os.makedirs(args.output_dir, exist_ok=True)
train_path = os.path.join(args.output_dir, "base_train.json")
eval_path = os.path.join(args.output_dir, "base_eval.json")

with open(train_path, "w") as f:
    json.dump(train_data, f, indent=2)
with open(eval_path, "w") as f:
    json.dump(eval_data, f, indent=2)

print(f"\nTotal valid: {len(valid)}")
print(f"Train: {len(train_data)} → {train_path}")
print(f"Eval:  {len(eval_data)} → {eval_path}")
