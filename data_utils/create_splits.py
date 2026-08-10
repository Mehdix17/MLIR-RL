#!/usr/bin/env python3
"""
create_splits.py
----------------
Create train/eval splits for extracted MLIR benchmarks.

Supported modes:
  - seen-batch:   train and eval both contain the same batch domains.
  - unseen-batch: eval holds out one or more batch sizes.
  - cross-model:  eval holds out one or more models.

Outputs JSON with explicit file lists and split metadata.
"""

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict


def _collect_files(roots: list[str]) -> list[dict]:
    records = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".mlir"):
                    continue
                path = os.path.join(dirpath, fname)
                model = os.path.basename(os.path.dirname(path))
                batch_match = re.search(r"_bs(\d+)_", fname)
                batch = int(batch_match.group(1)) if batch_match else None

                op_type = "unknown"
                prefix = f"{model}_"
                if fname.startswith(prefix):
                    tail = fname[len(prefix):]
                    m = re.match(r"(.+)_\d+\.mlir$", tail)
                    if m:
                        op_tag = m.group(1)
                        op_tag = re.sub(r"_bs\d+$", "", op_tag)
                        op_type = op_tag

                origin = "dynamic" if "/dynamic/" in path.replace("\\", "/") else "real"
                records.append({
                    "path": os.path.abspath(path),
                    "model": model,
                    "batch": batch,
                    "batch_label": str(batch) if batch is not None else "static",
                    "op_type": op_type,
                    "origin": origin,
                })
    return sorted(records, key=lambda x: x["path"])


def _split_group(items: list[dict], train_ratio: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    if not items:
        return [], []
    shuffled = list(items)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    n_train = int(round(len(shuffled) * train_ratio))
    n_train = max(1, min(len(shuffled) - 1, n_train))
    return shuffled[:n_train], shuffled[n_train:]


def _build_seen_batch_split(files: list[dict], train_ratio: float, rng: random.Random) -> tuple[list[dict], list[dict], dict]:
    groups = defaultdict(list)
    for rec in files:
        groups[(rec["model"], rec["batch_label"])].append(rec)

    train = []
    eval_set = []
    for _, recs in groups.items():
        tr, ev = _split_group(recs, train_ratio, rng)
        train.extend(tr)
        eval_set.extend(ev)

    meta = {
        "grouping": "(model, batch_label)",
        "num_groups": len(groups),
    }
    return train, eval_set, meta


def _build_unseen_batch_split(files: list[dict], holdout_batches: list[int] | None) -> tuple[list[dict], list[dict], dict]:
    dynamic_batches = sorted({rec["batch"] for rec in files if rec["batch"] is not None})
    if holdout_batches:
        holdout = sorted({int(b) for b in holdout_batches if int(b) > 0})
    elif dynamic_batches:
        holdout = [dynamic_batches[-1]]
    else:
        holdout = []

    train = []
    eval_set = []
    for rec in files:
        if rec["batch"] in holdout:
            eval_set.append(rec)
        else:
            train.append(rec)

    meta = {
        "dynamic_batches_available": dynamic_batches,
        "holdout_batches": holdout,
    }
    return train, eval_set, meta


def _build_cross_model_split(files: list[dict], holdout_models: list[str] | None,
                             train_ratio: float) -> tuple[list[dict], list[dict], dict]:
    models = sorted({rec["model"] for rec in files})
    if holdout_models:
        holdout = sorted({m for m in holdout_models if m in models})
    else:
        k = max(1, int(math.ceil(len(models) * (1.0 - train_ratio)))) if models else 0
        holdout = models[-k:] if k else []

    train = [rec for rec in files if rec["model"] not in holdout]
    eval_set = [rec for rec in files if rec["model"] in holdout]

    meta = {
        "models": models,
        "holdout_models": holdout,
    }
    return train, eval_set, meta


def _serialize_split(mode: str, train: list[dict], eval_set: list[dict],
                     metadata: dict, output_path: str):
    payload = {
        "mode": mode,
        "counts": {
            "train": len(train),
            "eval": len(eval_set),
            "total": len(train) + len(eval_set),
        },
        "metadata": metadata,
        "train": [rec["path"] for rec in sorted(train, key=lambda x: x["path"])],
        "eval": [rec["path"] for rec in sorted(eval_set, key=lambda x: x["path"])],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote split: {output_path} (train={len(train)}, eval={len(eval_set)})")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation split files for extracted MLIR benchmarks.")
    parser.add_argument(
        "--mode",
        choices=["seen-batch", "unseen-batch", "cross-model", "all"],
        default="all",
        help="Split protocol to generate.",
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        default=[],
        help="One or more benchmark roots (e.g. data/nn/real data/nn/synthetic/dynamic).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where split JSON files are written.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train ratio used for seen-batch and auto cross-model modes.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for seen-batch mode.")
    parser.add_argument(
        "--holdout-batches",
        type=int,
        nargs="+",
        default=None,
        help="Explicit holdout batches for unseen-batch mode (default: largest seen dynamic batch).",
    )
    parser.add_argument(
        "--holdout-models",
        nargs="+",
        default=None,
        help="Explicit holdout models for cross-model mode.",
    )
    args = parser.parse_args()

    if not args.input_roots:
        raise ValueError("--input-roots is required (provide one or more benchmark roots).")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1).")

    files = _collect_files(args.input_roots)
    if not files:
        raise ValueError("No .mlir benchmark files found under --input-roots.")

    rng = random.Random(args.seed)
    requested_modes = ["seen-batch", "unseen-batch", "cross-model"] if args.mode == "all" else [args.mode]

    for mode in requested_modes:
        if mode == "seen-batch":
            train, eval_set, meta = _build_seen_batch_split(files, args.train_ratio, rng)
        elif mode == "unseen-batch":
            train, eval_set, meta = _build_unseen_batch_split(files, args.holdout_batches)
        else:
            train, eval_set, meta = _build_cross_model_split(files, args.holdout_models, args.train_ratio)

        split_path = os.path.join(args.output_dir, f"{mode}.json")
        _serialize_split(mode, train, eval_set, meta, split_path)


if __name__ == "__main__":
    main()