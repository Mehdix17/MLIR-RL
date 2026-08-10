# config/ — MLIR-RL Configuration

Single active dataset: `ops_and_blocks`.

## Directory Layout

```
config/
├── ops_and_blocks/
│   ├── train/                (6)  Training configs
│   └── eval/                 (6)  Eval configs
└── README.md
```

## `ops_and_blocks/` — Ops + Blocks Dataset

~8,962 benchmarks (single-op + multi-op blocks). Data: `data/ops_and_blocks/{train,eval}/`.

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

## Usage

```bash
sbatch scripts/train/train.sh config/ops_and_blocks/train/v4_9_small.json
sbatch scripts/eval/eval.sh config/ops_and_blocks/eval/v4_9_small_eval.json
```
