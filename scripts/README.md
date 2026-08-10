# scripts/ — MLIR-RL Orchestration

No files at root. Everything organized into subfolders by purpose.

## Directory Layout

```
scripts/
├── train/           Training entry points + Slurm wrappers
├── eval/            Evaluation + ablation eval entry points + Slurm wrappers
├── baseline/        MLIR & PyTorch baseline timing scripts
├── data/            Dataset generation, processing, splitting
├── full_model/      Full-model end-to-end optimization
├── checkpoint/      Checkpoint scan + results merger
├── utils/           Reporting, sanity checks, workflow helpers
├── setup/           Conda env export/import, MLIR install tests
├── hpo/             Hyperparameter optimization (Optuna)
└── plots/           Plot generation + reporting
```

## Subfolder Details

### `train/`
| File | Purpose |
|------|---------|
| `train.py` | PPO training loop (core entry point) |
| `train.sh` | Slurm training wrapper |
| `train_generated.sh` | Train on generated benchmarks |
| `train_generated_fullscale.sh` | Full-scale generated training |
| `train_v0_full.sh` | V0 full dataset training |
| `train_v4_full.sh` | V4 full dataset training |

### `eval/`
| File | Purpose |
|------|---------|
| `eval.py` | Evaluation loop (core entry point) |
| `eval.sh` | Generic Slurm eval (resolves impl from config) |
| `ablation_eval.py` | Ablation study evaluation |
| `eval_batch.sh` | Batch eval runner |
| `eval_generated.sh` | Eval on generated benchmarks |
| `orchestrate_eval.py` | Multi-job eval orchestrator |
| `run_rl_eval.sh` | RL eval via `optimize_model_via_blocks.py` |
| `submit_checkpoint_evals.sh` | Batch submit checkpoint evaluations |
| `submit_eval.py` | Programmatic eval submission |
| `sync_progress.py` | Sync eval progress tracking |

### `baseline/`
| File | Purpose |
|------|---------|
| `get_base.py` | MLIR baseline execution timing |
| `get_base.sh` | Slurm wrapper for `get_base.py` |
| `get_base_raw.sh` | Baseline on raw (unextracted) models |
| `get_pytorch_baselines.py` | Canonical PyTorch eager + JIT timing for 22 models |
| `get_pytorch_times.py` | Block-level PyTorch timing |
| `get_pytorch_times.sh` | Slurm wrapper for `get_pytorch_times.py` |
| `get_pytorch_raw.sh` | PyTorch timing on raw models |

### `data/`
| File | Purpose |
|------|---------|
| `split_json.py` | Stratified train/eval split of benchmark JSON |
| `build_checkpoint_comparison.py` | Merges full-model + block results into comparison CSV/JSON |
| `build_generated_exec_json.py` | Build exec timing JSON for generated benchmarks |
| `generate_all_datasets.sh` | Generate all benchmark datasets |
| `generate_all_models.sh` | Generate all model MLIR files |
| `gen_onnx.sh` | ONNX export helper |
| `regen_all.sh` | Regenerate all benchmarks |

### `utils/`
| File | Purpose |
|------|---------|
| `fast_report.py` | Unified fast reporting (training + eval + quota) |
| `report_training.py` | Training progress report |
| `report_eval.py` | Eval progress report |
| `classify_benchmarks.py` | Classify benchmarks by type |
| `summarize_results.py` | Summarize experiment results |
| `submit_and_monitor.sh` | Submit Slurm job + auto-tail output |
| `monitor_torch_mlir.sh` | Monitor torch-mlir compilation jobs |
| `pipeline.sh` | End-to-end pipeline helper |
| `test_torch_mlir_compile.py` | MLIR binding sanity check |
| `test_torch_mlir.sh` | Shell wrapper for the above |

### `setup/`
| File | Purpose |
|------|---------|
| `create_conda_env_from_export.sh` | Create conda env from export file |
| `export_conda_env.sh` | Export current conda env |
| `install_and_test_torch_mlir.sh` | Install and test torch-mlir |

### `full_model/`
| File | Purpose |
|------|---------|
| `optimize_full_model.py` | Main orchestrator for full-model RL optimization |
| `optimize_model_via_blocks.py` | Block-based fallback optimization |
| `preprocess_model.py` | Runs C++ AST dumper to tag linalg ops |
| `add_timing_wrapper.py` | Wraps @main with @nanoTime() |
| `optimize_full_model.sh` | Slurm array job wrapper |
| `merge_full_model_results.sh` | Merges chunk files into unified JSON + CSV |
| `get_pytorch_full_times.sh` | Full PyTorch baseline timing |

### `checkpoint/`
| File | Purpose |
|------|---------|
| `merge_ckpt_scan.py` | Merges per-model checkpoint scan results |
| `ckpt_scan_all.sh` | Full checkpoint scan (all models, all ckpts + merge) |
| `submit_ckpt_scan.sh` | Submit checkpoint scan as Slurm array job |

### `hpo/`
| File | Purpose |
|------|---------|
| `run_hpo.py` | Optuna HPO runner |
| `analyze.py` | Analyze HPO study results |
| `train_trial.sh` | Train a single HPO trial |
| `eval_trial.sh` | Eval a single HPO trial |
| `get_baselines.sh` | Get baselines for HPO |
| `base_config.json` | Base HPO config template |
| `trials/` | Per-trial configs and results |

### `plots/`
| File | Purpose |
|------|---------|
| `generate_plots.py` | Generate evolution + comparison plots |
| `generate_report.py` | Generate plot report |
| `benchmark_families.json` | Benchmark family definitions |

## Standard Workflow

```bash
# 1. MLIR baseline
sbatch scripts/baseline/get_base.sh config/ops_and_blocks/train/v4_9_small.json

# 2. Train
sbatch scripts/train/train.sh config/ops_and_blocks/train/v4_9_small.json

# 3. Evaluate
sbatch scripts/eval/eval.sh config/ops_and_blocks/eval/v4_9_small_eval.json
```
