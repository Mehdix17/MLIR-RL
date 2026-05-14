# GEMINI.md - MLIR-RL Project Context

## Project Overview
MLIR-RL is a reinforcement learning auto-scheduler for MLIR loop nests. It uses PPO to learn optimal loop transformations (tiling, vectorization, etc.) to minimize execution time.

## Critical Environment Setup (Mandatory)
Before running any Python scripts or tools, you MUST:
1. **Activate Conda Env:** `source ~/envs/mlir/bin/activate`
2. **Load Environment Variables:** `set -a && source .env && set +a`
3. **Set Config Path:** `export CONFIG_FILE_PATH=config/<your_config>.json`

## Singletons & Initialization
- `utils.config.Config` is a singleton that reads `CONFIG_FILE_PATH` at **first import**.
- `utils.dask_manager.DaskManager` is a singleton (disabled by default).
- **Rule:** Never import `Config` or modules that use it before setting `CONFIG_FILE_PATH`.

## Implementation Versioning
The project uses a versioned package structure. Each `vN` is a standalone copy:
- `rl_autoschedular`: Baseline (Do NOT touch).
- `rl_autoschedular_v1`: Hardware-aware observation.
- `rl_autoschedular_v2`: Shaped reward.
- `rl_autoschedular_v3`: Transformer loop-nest encoder.
- `rl_autoschedular_v4+`: Future novelties.

**Import Rule:** Do NOT mix imports between packages (e.g., `v3` must only import from `v3`).

## Eval & Crash Resilience
- MLIR crashes (`SIGABRT`) are caught by a signal handler in `scripts/eval.py`.
- Eval results are saved incrementally in `markers/` files.
- To resume a crashed eval, just restart it; it will skip already evaluated benchmarks.

## LLVM/MLIR Bindings
- Python bindings are in `llvm-project/build/tools/mlir/python_packages/mlir_core`.
- If symlinks are broken (from another user's build), use the fix script in `AGENTS.md`.

## Key Commands
- **Train:** `sbatch scripts/train.sh config/train1.json`
- **Eval:** `sbatch scripts/eval.sh config/v3.json`
- **Dashboard:** `cd dashboard && streamlit run dashboard.py`
- **Verification:** `python -c "from mlir.ir import Context; print('OK')"`
