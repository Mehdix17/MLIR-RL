# scripts/ — MLIR-RL Orchestration

No files at root. Everything organized into subfolders by purpose.

## Directory Layout

```
scripts/
├── train/           Training entry points + Slurm wrappers
├── eval/            Evaluation + ablation eval entry points + Slurm wrappers
├── baseline/        MLIR & PyTorch baseline timing scripts
├── full_model/      Full-model end-to-end optimization
├── checkpoint/      Checkpoint scan + results merger
├── data/            Dataset processing (split, analysis)
├── pipeline/        Full pipeline orchestration (one-shot)
└── utils/           Sanity checks + workflow helpers
```

## Subfolder Details

### `train/`
| File | Purpose |
|------|---------|
| `train.py` | PPO training loop (core entry point) |
| `train.sh` | Slurm training wrapper (also handles array jobs) |
| `train_condo.sh` | Condo-cluster training variant |

### `eval/`
| File | Purpose |
|------|---------|
| `eval.py` | Evaluation loop (core entry point) |
| `ablation_eval.py` | Ablation study evaluation (requires `EVAL_DIR`) |
| `eval.sh` | Generic Slurm eval (resolves impl from config) |
| `eval_condo.sh` | Condo-cluster eval variant |
| `eval_v4_5.sh` | V4.5 eval with specific settings |
| `eval_v4_5_bergamo.sh` | V4.5 eval on Bergamo cluster |
| `bergamo_eval.sh` | V4.5 eval on Bergamo (AMD EPYC 9754, 256 cores) |
| `bigmem_bergamo_eval.sh` | Big-memory Bergamo variant |
| `dalma_eval.sh` | V4.5 eval on Dalma (Intel Xeon E5-2680 v4) |
| `jubail_eval.sh` | V4.5 eval on Jubail (AMD EPYC 7742, 128 cores) |
| `ablation_bergamo_eval.sh` | Ablation eval on Bergamo |
| `run_rl_eval.sh` | RL eval via `optimize_model_via_blocks.py` |

### `baseline/`
| File | Purpose |
|------|---------|
| `get_base.py` | MLIR baseline execution timing |
| `get_pytorch_times.py` | Block-level PyTorch timing (eager + JIT per op) |
| `get_pytorch_baselines.py` | **Canonical** PyTorch eager + JIT timing for 22 models |
| `get_base.sh` | Slurm wrapper for `get_base.py` (chunked) |
| `get_pytorch_times.sh` | Slurm wrapper for `get_pytorch_times.py` (chunked) |
| `get_blocks_baseline.sh` | Block-level baselines for all 20 models |
| `get_blocks_baseline_fresh.sh` | Fresh block baselines (overwrite mode) |
| `submit_blocks_baselines.sh` | Submit all 20 models' block baselines |
| `submit_old_models_baselines.sh` | Fresh baselines for 10 older models |

### `full_model/`
| File | Purpose |
|------|---------|
| `optimize_full_model.py` | Main orchestrator for full-model RL optimization |
| `optimize_model_via_blocks.py` | Block-based fallback optimization |
| `preprocess_model.py` | Runs C++ AST dumper to tag linalg ops |
| `add_timing_wrapper.py` | Wraps @main with @nanoTime() |
| `optimize_full_model.sh` | Slurm array job wrapper (tasks 0–19) |
| `merge_full_model_results.sh` | Merges chunk files into unified JSON + CSV |
| `fix_jit_times.sh` | Fix JIT times for gpt2 + vit models |
| `fix_jit_times2.sh` | Fix JIT times v2 (same purpose) |
| `get_pytorch_full_times.sh` | Full PyTorch baseline timing |
| `run_blocks_v10.sh` | Block-based optimization v10 config |
| `run_honest_blocks.sh` | Honest block speedup computation |

### `checkpoint/`
| File | Purpose |
|------|---------|
| `merge_ckpt_scan.py` | Merges per-model checkpoint scan results |
| `ckpt_scan_all.sh` | Full checkpoint scan (all models, all ckpts + merge) |
| `run_checkpoint_scan.sh` | Single checkpoint scan run |
| `run_ckpt_scan.sh` | Same as above (alternate) |
| `submit_ckpt_scan.sh` | Submit checkpoint scan as Slurm array job |

### `data/`
| File | Purpose |
|------|---------|
| `split_json.py` | Stratified train/eval split of benchmark JSON |
| `build_checkpoint_comparison.py` | Merges full-model + block results into comparison CSV/JSON |
| `honest_blocks_speedup.py` | Computes honest model-level speedup for bufferization-crash models |

### `pipeline/`
| File | Purpose |
|------|---------|
| `pipeline.sh` | Full one-shot pipeline: get_base → get_pytorch_times → split_json → train |

### `utils/`
| File | Purpose |
|------|---------|
| `test_torch_mlir_compile.py` | MLIR binding sanity check |
| `test_torch_mlir.sh` | Shell wrapper for the above |
| `submit_and_monitor.sh` | Submit Slurm job + auto-tail output |

## Standard Workflow

```bash
# 1. MLIR baseline
sbatch scripts/baseline/get_base.sh config/train/v4_5.json

# 2. PyTorch baseline
sbatch scripts/baseline/get_pytorch_times.sh config/train/v4_5.json

# 3. Split train/eval
python scripts/data/split_json.py config/train/v4_5.json

# 4. Train
sbatch scripts/train/train.sh config/train/v4_5.json

# 5. Evaluate
sbatch scripts/eval/eval.sh config/train/v4_5.json

# Or one-shot:
bash scripts/pipeline/pipeline.sh config/train/baseline.json
```
