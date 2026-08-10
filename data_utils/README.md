# data_utils — Reference

## Quick Start

For reproducibility, two high-level scripts automate the full pipeline:

```bash
# 1. Convert any model to MLIR (auto-detects type)
python -m data_utils.convert_model --model resnet18 --output-dir data/raw_bench/

# 2. Extract benchmarks (ops and/or blocks)
python -m data_utils.extract_from_model --input data/raw_bench/resnet18_linalg.mlir --output-dir data/benchmarks/

# 3. (Optional) Low-level CLI for individual operations
python -m data_utils.cli vision --model resnet18
python -m data_utils.cli strip model.mlir --replace
```

---

## High-Level Scripts

### `convert_model.py` — NN model → linalg MLIR

Auto-detects model type (vision/transformer/gnn) from `model_catalog.py`, then delegates to the appropriate converter.

```bash
# Auto-detect model type
python -m data_utils.convert_model --model resnet18 --output-dir data/raw_bench/
python -m data_utils.convert_model --model bert --output-dir data/raw_bench/
python -m data_utils.convert_model --model gcn --output-dir data/raw_bench/

# Explicit type (if needed)
python -m data_utils.convert_model --model resnet50 --type vision --output-dir data/raw_bench/

# With options
python -m data_utils.convert_model --model resnet50 --output-dir data/raw_bench/ \
    --batch-size 1 --backend direct --verbose
```

**Available models:**
- **Vision** (14): resnet18, resnet50, resnext50, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, densenet121, vit_b_16, convnext_tiny/small/base/large, vgg11
- **Transformer** (14): bert, distilbert, roberta, albert, deberta, electra, gpt2, t5, bart, switch_base_8_moe, lstm, lstm_seq2seq, gru, bilstm
- **GNN** (4): gcn, graphsage, gat, gin

**Output:** `{output_dir}/{model}_linalg.mlir`

---

### `extract_from_model.py` — linalg MLIR → benchmarks

Extracts single operations and/or multi-op blocks from a full-model MLIR file.

```bash
# Extract both ops and blocks (default)
python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/

# Extract only single operations
python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --ops-only

# Extract only multi-op blocks
python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ --blocks-only

# With custom parameters
python -m data_utils.extract_from_model --input model_linalg.mlir --output-dir data/benchmarks/ \
    --batch-size 4 --window-size 7 --stride 4 --manifest-dir data/manifests/
```

**Output structure:**
```
{output_dir}/
├── ops/       # Single operations (if not --blocks-only)
└── blocks/    # Multi-op blocks (if not --ops-only)
```

---

## Low-Level CLI (`cli.py`)

For individual operations, use the low-level CLI (formerly `orchestrate.py`):

```bash
# Convert models
python -m data_utils.cli vision       --model resnet18
python -m data_utils.cli transformer  --model bert
python -m data_utils.cli gnn          --model gcn

# Post-processing
python -m data_utils.cli wrap         --input model.mlir --model-name foo --output wrapped.mlir
python -m data_utils.cli strip        model.mlir --replace
```

---

## Directory Structure

```
data_utils/
├── convert_model.py          # High-level: model → MLIR
├── extract_from_model.py     # High-level: MLIR → benchmarks
├── cli.py                    # Low-level CLI (all operations)
├── model_catalog.py          # Central model registry
│
├── convert/                  # Model converters
│   ├── vision2mlir.py        #   torchvision/ultralytics
│   ├── transformers2mlir.py  #   HuggingFace transformers
│   └── gnn2mlir.py           #   Graph Neural Networks
│
├── extract/                  # Benchmark extraction
│   ├── extract_ops.py        #   Single op extraction
│   ├── extract_blocks.py     #   Multi-op block extraction
│   └── batch_policy.py       #   Batch size selection
│
├── generate/                 # Synthetic benchmark generation
│   ├── mlir_generators.py    #   Op generator library
│   ├── generate_synthetic.py #   Generate synthetic .mlir files
│   └── id_allocator.py       #   Sequential ID allocation
│
└── postprocess/              # MLIR file cleanup
    ├── wrap_mlir.py          #   Add timed @main wrapper
    └── strip_mlir.py         #   Remove weight constants
```

---

## Environment Setup

```bash
# Activate conda environment
source ~/envs/mlir/bin/activate
set -a && source .env && set +a

# Required environment variables (from .env):
# - LLVM_BUILD_PATH: LLVM/MLIR build directory
# - MLIR_SHARED_LIBS: MLIR shared libraries path
# - AST_DUMPER_BIN_PATH: C++ AST dumper binary (for block extraction)
```

---

## Full Reproducibility Pipeline

```bash
# Step 1: Convert model to MLIR
python -m data_utils.convert_model --model resnet18 --output-dir data/raw_bench/

# Step 2: Extract benchmarks
python -m data_utils.extract_from_model \
    --input data/raw_bench/resnet18_linalg.mlir \
    --output-dir data/benchmarks/resnet18/ \
    --batch-size 1

# Step 3: (Optional) Generate baseline timings
python scripts/baseline/get_base.py --config config/ops_and_blocks/train/v4_9_small.json

# Step 4: Train RL agent
sbatch scripts/train/train.sh config/ops_and_blocks/train/v4_9_small.json
```

---

## Dataset Download

For immediate reproducibility without running the full pipeline, the pre-generated dataset is available:
https://drive.google.com/file/d/1y-mTblP_-uQv2wJmY1awIrUgbvE4dP_P/view?usp=sharing
