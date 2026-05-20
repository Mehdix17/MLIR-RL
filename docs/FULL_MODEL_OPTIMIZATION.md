# Full-Model End-to-End RL Optimization

Optimizes complete neural network `.mlir` model files with the trained RL auto-scheduler agent
and benchmarks end-to-end execution time, comparable to MLIR baseline, PyTorch, and PyTorch JIT.

## Overview

```
Full model .mlir  ──→  [1] AST dumper  ──→  tagged code + op features
                             │
                       [2] RL agent greedy inference per operation ──→ action schedules
                             │
                       [3] Apply all schedules in-place (transform dialect)
                             │
                       [4] Add @nanoTime() timing wrapper
                             │
                       [5] Execution.execute_code() ──→ optimized exec time
```

The RL agent (PPO, v4.5) was trained on real operation blocks extracted from these same models,
so it generalizes to the full-model setting.

## Results

| Model | Baseline (ns) | Optimized (ns) | Speedup | Status |
|-------|---------------|----------------|---------|--------|
| gcn | 9,467,782 | 7,497,137 | 1.26x | OK |
| lstm | 210,823,182 | 48,445,669 | 4.35x | OK |
| efficientnet_b0 | 472,554,392 | 403,942,517 | 1.17x | OK |
| mobilenet_v3_small | 63,002,714 | 61,561,367 | 1.02x | OK |
| resnet18 | 2,304,724,956 | 1,268,257,584 | 1.82x | OK |
| albert | — | — | — | FIXED (job 15650956) |
| bert | — | — | — | FIXED (job 15650956) |
| roberta | — | — | — | AST dumper OOM |
| bart, convnext_tiny, deberta, densenet121, distilbert, gat, gpt2, vit_b_16 | — | — | — | exec timeout |
| resnext50 | — | — | — | feature extraction error |

## Architecture

### Scripts

| File | Purpose |
|------|---------|
| `scripts/optimize_full_model.py` | Main orchestrator — tags, times baseline, runs RL agent, applies schedules, measures optimized |
| `scripts/optimize_full_model.sh` | Slurm array job wrapper (tasks 0–19, one per model) |
| `scripts/preprocess_model.py` | Runs C++ AST dumper to inject `{tag = "operation_NNN"}` on all linalg ops |
| `scripts/add_timing_wrapper.py` | Wraps `@main` with `@nanoTime()` calls, returns `(tensor, i64)` |
| `scripts/merge_full_model_results.sh` | Post-hoc merger: chunk files → merged JSON + CSV |

### Config

`config/full_model_optim.json` — uses `rl_autoschedular_v4_5` implementation.

### How It Works

**Step 1 — Tagging** (`preprocess_model.py`):
Runs the C++ AST dumper (`tools/ast_dumper/build/bin/AstDumper`) on the raw model file.
The dumper outputs:
```
<operation graph and features>
########################################
<tagged full code with {tag = "operation_NNN"} on every linalg op>
```
We extract the tagged code portion. The `dense_resource` weight constants stay as-is (they compile fine).

**Step 2 + 3 — RL Agent Inference** (`optimize_full_model.py`):
- For each tagged operation in the full model, builds an `OperationState` and greedily samples
  the optimal action sequence using the trained PPO model.
- Agent uses the v4_5 checkpoint at `results/experiment3/v4_5_agent/run_0/models/model_715.pt`.
- Actions: Interchange, Tiling, TiledParallelization, TiledFusion, Vectorization, NoTransformation.
- Inference is pure state-space — no code executed during this phase.

**Step 4 — Apply Schedules In-Place**:
- All per-operation action sequences are applied to the full model code using
  `transforms.py` functions, which target specific tags via `transform.structured.match`.
- The model file stays whole — no extract/reassemble needed.
- Transform dialect code is appended to the model and run through `mlir-opt -transform-interpreter`.

**Step 5 — Timing**:
- `add_timing_wrapper.py` injects `%t0 = @nanoTime()` at `@main` entry and `%t1` before return.
- `@main` return is changed to `(original_tensor, i64)` with `llvm.emit_c_interface`.
- `Execution.execute_code()` compiles via MLIR bindings (full pass pipeline: bufferization → lowering → LLVM),
  JIT-executes, and returns the delta in nanoseconds.

## Commands

### Environment Setup

```bash
cd /scratch/mb10856/MLIR-RL
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/full_model_optim.json
export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
```

### Quick Test (Single Model, Interactive)

```bash
python scripts/optimize_full_model.py \
    --checkpoint results/experiment3/v4_5_agent/run_0/models/model_715.pt \
    --models gcn \
    --output /tmpdata/opencode/gcn_test.json
```

### Full Run (All 20 Models, Slurm Array)

```bash
# Submit array job
sbatch --array=0-19 scripts/optimize_full_model.sh

# Or with custom config/checkpoint
sbatch --array=0-19 scripts/optimize_full_model.sh config/full_model_optim.json /path/to/model.pt

# Monitor
squeue -u $USER

# After all jobs complete, merge chunks
bash scripts/merge_full_model_results.sh
# → results/full_model/merged.json
# → results/full_model/merged.csv
```

### Run Specific Models (Slurm)

```bash
sbatch scripts/optimize_full_model.sh config/full_model_optim.json /path/to/model.pt gcn distilbert albert
```

## Model List (Array Index)

| Index | Model |
|-------|-------|
| 0 | albert |
| 1 | bart |
| 2 | bert |
| 3 | convnext_tiny |
| 4 | deberta |
| 5 | densenet121 |
| 6 | distilbert |
| 7 | efficientnet_b0 |
| 8 | gat |
| 9 | gcn |
| 10 | gpt2 |
| 11 | lstm |
| 12 | mobilenet_v3_small |
| 13 | resnet18 |
| 14 | resnet50 |
| 15 | resnext50 |
| 16 | roberta |
| 17 | t5 |
| 18 | vgg11 |
| 19 | vit_b_16 |

## Output Format

Each chunk file (`results/full_model/chunk_N.json`) contains:

```json
{
  "model_name": {
    "model": "model_name",
    "baseline_ns": 10343173,
    "optimized_ns": 7526907,
    "speedup": 1.374,
    "num_ops": 13,
    "schedules": {
      "operation_0": ["I(1,0)", "T(64,16)", "T(2,4)", "V()"],
      ...
    }
  }
}
```

Merged output adds CSV with columns: `Model,Baseline(ns),Optimized(ns),Speedup`.

## Edge Cases Handled

- **`dense_resource` constants**: Left as-is; MLIR pass pipeline and JIT handle them correctly.
- **`{-# dialect_resources #-}` blocks**: Preserved outside the module at file top-level.
- **Timing wrapper** places `%t0` at `@main` entry and `%t1` before return — measures full forward pass.
- **Execution engine** (`v4_5`) returns 4-tuple `(time, ok, cache_miss, error_msg)` — orchestrator handles both 3-tuple (baseline) and 4-tuple formats.
- **RL agent inference** is pure greedy sampling — no exploration, no code execution during inference.
- **Multi-value returns**: `@main -> (type1, type2)` now correctly parsed for models like albert and bert.

## Known Issues / Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Could not find func.func @main` | Multi-value return `-> (type1, type2)` not matched by regex | Fixed: regex now handles `(type1, ...)` |
| `AST dumper failed` | Model too large (e.g. roberta, vgg11) — memory exhaustion | Increase Slurm `--mem` or use smaller models |
| `baseline_exec_failed` with no error | JIT compilation timeout on large models with dense_resource | Increase `EXEC_TIMEOUT` env var and Slurm `--time` |
| `feature extraction error: 'operation_NNN'` | AST dumper output missing operation tag in graph | Model has unsupported op types or AST dumper bug |

## Full Command Reference

### Interactive (Single Model)

```bash
cd /scratch/mb10856/MLIR-RL
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
export CONFIG_FILE_PATH=config/full_model_optim.json

python scripts/optimize_full_model.py \
    --checkpoint results/experiment3/v4_5_agent/run_0/models/model_715.pt \
    --models gcn \
    --output results/full_model/test.json
```

### Slurm Array (All Models)

```bash
cd /scratch/mb10856/MLIR-RL
sbatch --array=0-19 scripts/optimize_full_model.sh
```

### Slurm (Specific Models)

```bash
sbatch scripts/optimize_full_model.sh config/full_model_optim.json \
    /scratch/mb10856/MLIR-RL/results/experiment3/v4_5_agent/run_0/models/model_715.pt \
    gcn distilbert albert
```

### Merge Results

```bash
bash scripts/merge_full_model_results.sh
# → results/full_model/merged.json + merged.csv
```

### Preprocess + Wrap Only (Skip RL)

```bash
# Tag model
python scripts/preprocess_model.py \
    --input data/nn/raw_bench/gcn_linalg.mlir \
    --output data/nn/tagged/gcn_tagged.mlir

# Add timing wrapper
python scripts/add_timing_wrapper.py \
    --input data/nn/tagged/gcn_tagged.mlir \
    --output data/nn/wrapped/gcn_wrapped.mlir
```

### Testing the Wrapper Independently

```bash
python3 -c "
from mlir.ir import Context, Module
with open('data/nn/tagged/gcn_tagged.mlir') as f: code = f.read()
from scripts.optimize_full_model import add_timing_wrapper
wrapped = add_timing_wrapper(code)
with Context():
    m = Module.parse(wrapped)
    print('MLIR parse OK')
"
```

## Implementation Notes

- No existing files were modified. All new code is in `scripts/` and `config/`.
- Uses the same `Execution` engine that powers training/eval — zero-initialized inputs,
  full MLIR → LLVM lowering, JIT execution.
- The RL agent was trained on extracted blocks from these same models (v4.5, `data/all/`).
  Greedy inference at eval time avoids reward hacking.
