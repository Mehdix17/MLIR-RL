# config/ — MLIR-RL Configuration

No files at root. Organized by purpose into subfolders.

## Directory Layout

```
config/
├── train/                  Training/version configs (auto-selected by Slurm scripts)
├── ablation/               Ablation study configs (V4.5 minus one feature)
├── ablation_full_model/    Ablation full-model eval variants
├── eval/                   Per-checkpoint eval configs (Dalma cluster)
├── eval_bergamo/           Per-checkpoint eval configs (Bergamo cluster)
├── full_model/             Full-model end-to-end optimization
├── test/                   Sanity/minimal test configs
└── misc/                   One-off / special-purpose configs
```

## Subfolder Details

### `train/` — Version Configs (8 files)
Used for training and evaluation. Auto-selected by `train.sh` and `eval.sh` via array job mode: `config/train/${VERSION}.json`.

| File | Implementation | Key Features |
|------|---------------|--------------|
| `baseline.json` | `rl_autoschedular` | Baseline LSTM policy |
| `v1.json` | `rl_autoschedular_v1` | Hardware-aware observation |
| `v2.json` | `rl_autoschedular_v2` | Shaped reward |
| `v2_5.json` | `rl_autoschedular_v2_5` | Hardened shaped reward |
| `v3.json` | `rl_autoschedular_v3` | Transformer encoder |
| `v4.json` | `rl_autoschedular_v4` | Integrated V1+V2+V3 |
| `v4_5.json` | `rl_autoschedular_v4_5` | Robust Integrated (trained) |
| `train1.json` | `rl_autoschedular` | Shorter baseline training run |

### `ablation/` — Ablation Study (6 files)
Each disables exactly one novelty from V4.5. All use `results/ablation_no_*` dirs.

| File | Disabled Feature |
|------|-----------------|
| `v45_no_hw.json` / `v45_no_hw_blocks.json` | Hardware-awareness |
| `v45_no_shaped_reward.json` / `v45_no_reward_blocks.json` | Shaped reward |
| `v45_no_transformer.json` / `v45_no_trans_blocks.json` | Transformer (uses LSTM) |

The `_blocks` variants are identical to base but with different tags. Used for block-based eval (`data/all`).

### `ablation_full_model/` — Ablation Full-Model (3 files)
Same ablations but pointing at `data/full_model` for full-model evaluation.

| File | Ablation |
|------|---------|
| `v45_full_model.json` | None (full V4.5 for full-model) |
| `v45_no_hw_full_model.json` | No hardware |
| `v45_no_trans_full_model.json` | No transformer |

### `eval/` — Per-Checkpoint Dalma Eval (15 files)
V4.5 evaluation on Dalma cluster, one config per checkpoint (700–1999).

Naming: `v4_5_eval_<ckpt>.json` → writes to `results/V4_5_agent_ckpt<ckpt>/`.
`v4_5_eval.json` is the scan-mode variant (all checkpoints).

### `eval_bergamo/` — Per-Checkpoint Bergamo Eval (14 files)
Same structure but for Bergamo cluster (AMD EPYC 9754). Checkpoints 800–1999 only.

Naming: `v4_5_bergamo_<ckpt>.json` → writes to `results/V4_5_agent_bergamo_ckpt<ckpt>/`.
`v4_5_eval_bergamo.json` is the scan-mode variant.

### `full_model/` — Full-Model Optimization (1 file)
| File | Purpose |
|------|---------|
| `full_model_optim.json` | V4.5 full-model end-to-end optimization |

### `test/` — Sanity / Test (3 files)
Reduced-scale configs for quick validation.

| File | Scale |
|------|-------|
| `test.json` | Minimal: 5 benches, 5 replays, 5 iters |
| `v3_sanity.json` | V3 small transformer: d_model=64, 15 benches |
| `v4_sanity.json` | V4 medium transformer: d_model=128, 15 benches |

### `misc/` — Special Purpose (3 files)
| File | Purpose |
|------|---------|
| `albert.json` | ALBERT-specific baseline |
| `example.json` | Template with all features enabled (reference) |
| `hardware_eval.json` | Cross-hardware evaluation (V4.5) |

## Usage

```bash
# Training
sbatch scripts/train/train.sh config/train/v4_5.json

# Evaluation
sbatch scripts/eval/eval.sh config/train/v4_5.json

# Array job (auto-selects from config/train/)
sbatch --array=0-6 scripts/train/train.sh

# Ablation eval
sbatch scripts/eval/eval.sh config/ablation/v45_no_hw.json

# Per-checkpoint eval
sbatch scripts/eval/eval.sh config/eval/v4_5_eval_800.json
```
