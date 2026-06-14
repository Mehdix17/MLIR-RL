# Quick Test Folder

Fast testing of MLIR-RL training and evaluation.

## Structure

```
quick_test/
├── input/              # 8 sample MLIR files (add operations)
├── output/
│   └── baselines.json  # Quick baselines for training
├── config_train.json   # Quick training config (100 iterations)
├── config_eval.json    # Eval config (uses V4.9 small checkpoints)
└── README.md           # This file
```

## Key Difference

- **Training**: Uses `quick_test/input/` (8 benchmarks) for fast iteration
- **Evaluation**: Uses `data/single_ops_dataset/all/` (full dataset) with V4.9 small model checkpoints from `results/single_ops_dataset_results/v4_9_small_agent/models/`

## Usage

### 0. Set Environment (Interactive Only)

```bash
source ~/envs/mlir/bin/activate
export PYTHONPATH=/scratch/mb10856/MLIR-RL:/scratch/mb10856/MLIR-RL/rl_autoschedular:$PYTHONPATH
set -a && source .env && set +a
```

### 1. Generate Baselines

```bash
python scripts/baseline/get_base.py \
  --benchmarks-dir quick_test/input \
  --output quick_test/output/baselines.json \
  --implementation rl_autoschedular_v4_9
```

### 2. Train (Quick)

```bash
sbatch scripts/train/train.sh quick_test/config_train.json
```

**Training time:** ~5-10 minutes (100 iterations, 8 benchmarks)

### 3. Evaluate (using V4.9 small checkpoints)

```bash
sbatch scripts/eval/eval.sh quick_test/config_eval.json --checkpoint 100
```

**Note:** Evaluation uses model checkpoints from `results/single_ops_dataset_results/v4_9_small_agent/models/`

**Evaluation time:** ~2-5 minutes

### 4. Check Results

```bash
python scripts/utils/report_training.py -d single_ops -v quick_test
python scripts/utils/report_eval.py -d single_ops -v quick_test
```

## Config Differences from Full Training

| Parameter | Quick Train | Full Train | Quick Eval | Full Eval |
|-----------|-------------|------------|------------|-----------|
| `nb_iterations` | 100 | 10,000 | 2,000 | 2,000 |
| `bench_count` | 8 | 64 | 32 | 32 |
| `ppo_batch_size` | 16 | 32 | 64 | 64 |
| `replay_count` | 2 | 10 | 10 | 10 |
| `truncate` | **5** | 5 | **5** | 5 |

**Note:** `truncate` must match between train/eval for model compatibility.

## Output Files

- `output/train/results.json` — Cumulative training results
- `output/train/models/model_100.pt` — Saved model checkpoint
- `output/eval/checkpoint_100.json` — Evaluation results

## Troubleshooting

### ModuleNotFoundError: No module named 'rl_autoschedular_v4_9'

Add `rl_autoschedular/` to PYTHONPATH:
```bash
export PYTHONPATH=/scratch/mb10856/MLIR-RL:/scratch/mb10856/MLIR-RL/rl_autoschedular:$PYTHONPATH
```

### Baselines show -1 (timeout)

Increase timeout or use smaller MLIR files:
```bash
python scripts/baseline/get_base.py --benchmarks-dir quick_test/input \
  --output quick_test/output/baselines.json --timeout 30
```

### Training not finding baselines

Ensure `json_file` in config points to generated baselines:
```json
"json_file": "quick_test/output/baselines.json"
```
