# utils/ — Shared Utilities

Common utilities used across all RL agent packages and scripts.

## Modules

### config.py
**Config singleton** — reads `CONFIG_FILE_PATH` environment variable on first import.

```python
from utils.config import Config
config = Config()  # Singleton, reads JSON config
```

All agent packages and scripts use this to access hyperparameters.

### implementation.py
**Implementation loader** — dynamically imports the correct RL agent package based on config.

```python
from utils.implementation import load_implementation
agent_cls, env_cls = load_implementation(config.implementation)
```

### dask_manager.py
**Dask distributed manager** — manages worker pools for parallel benchmark execution.

**Note**: Currently disabled. Uses `ThreadPoolExecutor` fallback with `SLURM_CPUS_PER_TASK` workers.

### bindings_process.py
**MLIR bindings process manager** — handles MLIR C++ state in separate process.

**Important**: `BindingsProcess.ENABLED` must stay `False` — fork corrupts MLIR C++ state.

### file_logger.py
**JSON file logger** — writes structured logs for training/eval metrics.

### log.py
**Logging setup** — configures Python logging with consistent formatting.

### singleton.py
**Singleton pattern** — base class for singleton implementations.

### gpt2_jit_compat.py
**GPT2 JIT compatibility** — handles PyTorch JIT compilation issues for GPT2 benchmarks.

## Key Gotchas

1. **Config is a singleton** — first import reads `CONFIG_FILE_PATH`, subsequent imports reuse
2. **No cross-package imports** — each `rl_autoschedular_vX` is standalone
3. **BindingsProcess.ENABLED = False** — never enable, causes MLIR state corruption
4. **DaskManager disabled** — ThreadPoolExecutor is the active parallel backend

## Usage

```bash
# Set config before any imports
export CONFIG_FILE_PATH=config/ops_and_blocks/train/v4_9_small.json

# In Python
from utils.config import Config
config = Config()
print(config.implementation)  # "rl_autoschedular_v4_9"
```
