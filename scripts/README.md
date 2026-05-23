# scripts/ — MLIR-RL Orchestration

Subfolders for shell scripts, root for Python entry-points.

| Subfolder   | Purpose |
|-------------|---------|
| `train/`    | Slurm training scripts (train.sh, train_condo.sh) |
| `eval/`     | Slurm evaluation scripts (eval.sh, cluster variants) |
| `baseline/` | Baseline timing scripts (get_base.sh, get_pytorch_times.sh) |
| `full_model/` | Full-model end-to-end optimization scripts |
| `checkpoint/` | Checkpoint scan and analysis scripts |
| `pipeline/` | Full pipeline orchestration (pipeline.sh) |
| `data/`     | Dataset processing scripts |
| `utils/`    | Utility scripts (submit_and_monitor.sh, test_torch_mlir.sh) |

## Root Python entry points

All `.py` files are at `scripts/` root, called by shell scripts in subfolders:

- `train.py` — PPO training loop (called by `train/train.sh`)
- `eval.py` — Evaluation loop (called by `eval/eval.sh`)
- `get_base.py`, `get_pytorch_times.py` — Baseline timing
- `split_json.py` — Train/eval split
- `optimize_full_model.py`, `optimize_model_via_blocks.py` — Full-model optimization
- `preprocess_model.py`, `add_timing_wrapper.py` — Full-model preprocessing
- `measure_full_model_baselines.py` — PyTorch baseline measurement
- `merge_ckpt_scan.py` — Checkpoint scan results merger
- `ablation_eval.py` — Ablation study evaluation
- `build_checkpoint_comparison.py`, `honest_blocks_speedup.py` — Analysis
- `test_torch_mlir_compile.py` — MLIR binding sanity check
