# config/ — MLIR-RL Configuration

Active datasets: `ops_and_blocks`, `legacy_paper`.

## Directory Layout

```
config/
├── ops_and_blocks/
│   ├── train/                (6)  Training configs
│   └── eval/                 (6)  Eval configs
├── v5/                      (7)  V5 unified configs (train + eval in one file)
└── README.md
```

## `ops_and_blocks/` — Ops + Blocks Dataset

~8,093 benchmarks (single-op + multi-op blocks). Data: `data/ops_and_blocks/` (flat; splits via JSON configs).

### `train/` (6)

| Config | Impl | Notes |
|--------|------|-------|
| `v0.json` | `rl_autoschedular_v0` | Baseline LSTM |
| `v4_9_small.json` | `rl_autoschedular_v4_9` | Small transformer |
| `v4_9_large.json` | `rl_autoschedular_v4_9` | Large transformer |
| `paper_original.json` | `rl_autoschedular_paper` | Paper LSTM |
| `paper_transformer_small.json` | `rl_autoschedular_paper_transformer` | Paper transformer (small) |
| `paper_transformer_large.json` | `rl_autoschedular_paper_transformer` | Paper transformer (large) |

### `eval/` (6)

Matching eval configs for each training variant above.

## `v5/` — V5 Unified Configs

V5 configs combine train + eval in a single JSON file. Includes `json_file` and `eval_json_file` fields pointing to baseline splits in `results/`.

| Config | Dataset | Notes |
|--------|---------|-------|
| `v5_single_node.json` | ops_and_blocks | CPU-only single-node training |
| `v5_distributed.json` | ops_and_blocks | Distributed PPO (V5.1) |
| `v5_no_transformer.json` | ops_and_blocks | Ablation: no transformer encoder |
| `v5_legacy_paper.json` | legacy_paper | Paper reproduction with V5 |
| `v5_no_transformer_legacy_paper.json` | legacy_paper | No-transformer ablation on legacy_paper |

## Usage

```bash
# V4-style (separate train/eval configs)
sbatch scripts/train/train.sh config/ops_and_blocks/train/v4_9_small.json
sbatch scripts/eval/eval.sh config/ops_and_blocks/eval/v4_9_small_eval.json

# V5-style (unified config)
sbatch scripts/train/train.sh config/v5/v5_single_node.json
```
