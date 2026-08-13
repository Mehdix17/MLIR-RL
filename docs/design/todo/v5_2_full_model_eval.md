# V5.2: Full-Model Evaluation Support — Design

**Status**: Design Phase
**Version**: **V5.2** of the new MLIR-RL generation (V5 → V5.1 → V5.2 → V5.3)
**Target**: `rl_autoschedular_v5` package (single package, backward compatible with V4.9 checkpoints)
**Base**: `rl_autoschedular_paper_transformer` structure (Transformer encoder only — no HW features, no shaped reward; both abandoned in V5) — V5.2 inherits V5's package + V5's accelerated pipeline
**Depends on**: V5 (`v5_training_acceleration.md`) — full-model eval needs the accelerated pipeline
**Precedes**: V5.3 (expanded action space, `v5_3_expanded_action_space.md`)
**Scope**: Block/Op training (unchanged) → Full-Model Evaluation (new)

> **Checkpoint compatibility note**: §8.4 says "V5 loads V4.9 checkpoints
> directly — same model architecture (TransformerEncoder + policy/value heads)."
> This holds only if the observation space matches. Since V5 drops HW features
> from the observation (paper_transformer structure), the observation size
> differs from v4_9 — **checkpoint compat is with `paper_transformer`
> checkpoints, not v4_9**. The architect must confirm this against
> `observation.py` before locking the design.

> **Already-implemented overlap**: `scripts/checkpoint/ckpt_scan_all.sh` +
> `submit_ckpt_scan.sh` already do a partial full-model eval of trained checkpoints
> on full `.mlir` files (block-based, `rl_autoschedular_v4_5`, results in
> `results/full_model/scan/`). This plan must **reconcile with, not duplicate,
> those scripts** — reuse their evaluation plumbing where possible and extend it.
> See §5.4 below.

---

## 1. Problem Statement

Current V4.9 and paper packages train and evaluate on **extracted operation blocks** (~12K files in `new_dataset/all/`). Each block contains a single operation with its immediate producer.

**Goal**: Evaluate the block-trained policy on **full `.mlir` model files** (ResNet18, T5, GPT-2, etc.) without retraining.

**Key Insight**: The per-op features extracted from full models are identical to block features. The policy can schedule each op in topological order on the full model, applying transforms to the complete Module.

---

## 2. Architecture Overview

### 2.1 Package Structure: `rl_autoschedular_v5/`

```
rl_autoschedular_v5/
├── config.yaml                    # + full_model_* fields
├── state.py                       # + FullModelFeatures, extract_full_model_features()
├── benchmarks.py                  # Benchmarks (block) + FullModelBenchmarks
├── actions/                       # UNCHANGED (6 per-op actions)
├── observation.py                 # UNCHANGED (Transformer encoder)
├── model.py                       # UNCHANGED (HiearchyModel)
├── env.py                         # UNCHANGED (block Env)
├── execution.py                   # + execute_model(), Module caching
├── train.py                       # UNCHANGED (block PPO)
├── eval.py                        # UNCHANGED (block eval)
├── eval_fullmodel.py              # NEW: full-model evaluation entry point
├── ppo.py                         # UNCHANGED
└── utils/
    ├── config.py                  # + full_model_* fields
    └── implementation.py          # + BASELINE_PREFIX_FULL
```

### 2.2 Training vs Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (unchanged)                      │
├─────────────────────────────────────────────────────────────────┤
│  new_dataset/all/train/*.mlir  →  Benchmarks  →  PPO on blocks  │
│  ~12K single-op blocks                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Checkpoint: model_XXX.pt
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FULL-MODEL EVAL (new)                        │
├─────────────────────────────────────────────────────────────────┤
│  data/full_models/*.mlir  →  FullModelBenchmarks                │
│       ↓                                                           │
│  For each full model:                                             │
│    1. Parse full MLIR → FullModelFeatures (graph + all ops)     │
│    2. Topological order through ALL ops                          │
│    3. For each op:                                               │
│         a. Extract per-op features (same as training)            │
│         b. Policy (greedy) → action sequence                     │
│         c. Apply transform to FULL Module                        │
│    4. Execute full transformed model                             │
│    5. Speedup = baseline_time / optimized_time                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Full-Model Infrastructure (Weeks 1-2)

### 3.1 AST Dumper Enhancement (`utils/ast_dumper/`)

**Current**: Parses `.mlir` → per-op features + `#BEGIN_GRAPH` (producer→consumer edges)

**New**: Add `--full-model` flag that additionally outputs:
- Tensor shapes and element types for each graph edge
- Loop-carried dependency information
- Per-op loop nest depth and total loop count
- Global stats: num_ops, num_loops_total, max_depth, graph_diameter, connected_components, memory_footprint

**Output Format** (appended to existing):
```
########################################
#BEGIN_FULL_MODEL_STATS
num_operations: 765
num_loops_total: 2340
max_depth: 7
total_arithmetic_ops: 15000000
graph_diameter: 12
num_connected_components: 1
has_cyclic_dependencies: false
memory_footprint_bytes: 2147483648
#END_FULL_MODEL_STATS

#BEGIN_EDGE_ATTRS
op_1 --> op_2 : shape=[1,32,32,3], dtype=f32
op_2 --> op_3 : shape=[1,32,32,64], dtype=f32
...
#END_EDGE_ATTRS

#BEGIN_LOOP_CARRIED_DEPS
op_5: loop_2 (reduction)
...
#END_LOOP_CARRIED_DEPS
```

### 3.2 State Representation (`rl_autoschedular_v5/state.py`)

```python
from dataclasses import dataclass
import networkx as nx

@dataclass
class FullModelStats:
    num_operations: int
    num_loops_total: int
    max_depth: int
    total_arithmetic_ops: int
    graph_diameter: int
    num_connected_components: int
    has_cyclic_dependencies: bool
    memory_footprint_bytes: int

@dataclass
class FullModelFeatures:
    graph: nx.DiGraph                    # nodes=op_tags, edges=producer→consumer
    op_features: dict[str, OperationFeatures]  # same OperationFeatures as block
    global_stats: FullModelStats
    edge_attrs: dict[tuple[str, str], dict]    # (producer, consumer) → {shape, dtype}
    loop_carried_deps: dict[str, list[str]]    # op_tag → list of loop args with carried deps

def extract_full_model_features(code: str) -> FullModelFeatures:
    """Parse AST dumper output with --full-model flag."""
    # 1. Run ast_dumper with --full-model
    # 2. Parse existing sections (ops, graph)
    # 3. Parse new sections (stats, edge_attrs, loop_carried_deps)
    # 4. Build nx.DiGraph
    # 5. Return FullModelFeatures
```

### 3.3 Benchmarks Loader (`rl_autoschedular_v5/benchmarks.py`)

```python
class FullModelBenchmarks:
    """Load full .mlir model files for evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models_dir = Path(config.full_model_benchmarks_path)
        self.baseline_times = self._load_baseline_times(config.full_model_json_file)
        self.train_models = self._load_split(config.full_model_train_split)
        self.eval_models = self._load_split(config.full_model_eval_split)
        
        # Cache: model_name → FullModelFeatures
        self._features_cache: dict[str, FullModelFeatures] = {}
    
    def get_model_features(self, model_name: str) -> FullModelFeatures:
        if model_name not in self._features_cache:
            mlir_file = self.models_dir / f"{model_name}.mlir"
            with open(mlir_file) as f:
                code = f.read()
            self._features_cache[model_name] = extract_full_model_features(code)
        return self._features_cache[model_name]
    
    def __len__(self) -> int:
        return len(self.eval_models)  # or train_models
    
    def __getitem__(self, idx: int) -> FullModelFeatures:
        model_name = self.eval_models[idx]
        return self.get_model_features(model_name)
```

### 3.4 Config Fields (`utils/config.py`)

```python
# Add to Config class:
full_model_benchmarks_path: str = ""           # data/full_models/
full_model_json_file: str = ""                 # data/full_models_base_times.json
full_model_train_split: str = ""               # data/full_models_train.json
full_model_eval_split: str = ""                # data/full_models_eval.json
full_model_exec_timeout: int = 14400           # 4 hours (seconds)
full_model_op_order: str = "topological"       # "topological" | "reverse_topological"
full_model_memory_limit_gb: int = 120          # kill if RSS exceeds
```

### 3.5 Execution Engine (`rl_autoschedular_v5/execution.py`)

```python
def execute_model(
    self, 
    code: str, 
    model_name: str, 
    transform_history: list[list[Action]], 
    root_exec_time: int,
    timeout_s: int = 14400
) -> tuple[int, bool, bool, Optional[str]]:
    """
    Execute full model with profiling-based timeout and memory monitoring.
    
    Args:
        code: Full MLIR module code
        model_name: Benchmark name
        transform_history: All transforms applied (for cache key)
        root_exec_time: Baseline time (ns) for dynamic timeout
        timeout_s: Hard timeout ceiling
    
    Returns:
        (exec_time_ns, success, cache_miss, error_msg)
    """
    # Dynamic timeout: min(4h, max(10min, root_exec_time * 5))
    # Memory monitor: spawn watchdog thread, kill if RSS > 120GB
    # Module caching: keep Module.parse() result alive during episode
    # Use __execute_bufferized_code as primary (not fallback)
```

**Module Caching Pattern**:
```python
class FullModelExecutor:
    def __init__(self, code: str):
        with Context():
            self.module = Module.parse(code)
            self.pm = PassManager.parse(pass_pipeline)
            self.inputs, self.outputs_struct = self._create_params()
    
    def apply_and_execute(self, transform_sequence: list[Action]) -> int:
        # Apply transforms incrementally to self.module
        for action in transform_sequence:
            action.apply(self.module)
        
        # Run passes
        self.pm.run(self.module.operation)
        
        # Execute
        return self._invoke()
```

### 3.6 Persistent MLIR Worker Pool (moved from V5.0)

**Why it lives here, not in V5.0** — in block training a worker pool saves
only ~1-3% of wall-clock: every execution is a different code string, exact
repeats hit the execution-time cache before any process work, and a freshly
forked child is nearly free (it inherits the already-imported MLIR bindings).
In full-model eval the economics flip: each model is parsed once, transforms
are applied incrementally to the same module, and several models can be
evaluated in parallel on one node. Keeping one executor process per model
avoids repeated process startup and lets the parsed module live for the whole
rollout.

**What it is**: N long-lived worker processes (N ≤ `--cpus-per-task`), each
owning a `FullModelExecutor` for one model. The parent dispatches model
rollouts to free workers and collects per-model exec times.

**Design requirements** (carried over from the V5.0 analysis):
- **Spawn-based only** — `multiprocessing.get_context("spawn")`, never fork
  (fork corrupts MLIR C++ state; `BindingsProcess.ENABLED` must stay `False`).
  Spawn cost is paid once per worker at eval start, then amortized over the
  whole rollout.
- **Worker lifecycle**: workers start at eval start, import MLIR once, receive
  `(model_name, code)`, parse → loop over transform sequences → apply →
  execute → return `(exec_time_ns, success, error)`.
- **Crash recovery**: a worker SIGABRT only kills its own model rollout — the
  parent detects death, respawns the worker, marks that model failed, and the
  existing `mlir-opt | mlir-cpu-runner` fallback still applies.
- **Timeout**: per-model dynamic timeout (`root_exec_time * 5`, clamped) — the
  same rule as block execution.
- **Memory**: ~0.7-1G per worker (measured, see
  `v5_training_acceleration.md` §Memory calibration) — 64 workers fit the eval
  budget.

**Deferred from V5.0 by decision** (`v5_training_acceleration.md` §Scope
decisions). V5.0 ships without it; V5.2 implements it only if parallel
full-model eval needs it — a sequential eval over ~19 models is fine without.

### 3.7 DaskManager — evaluated, not worth enabling

**Current state**: `DaskManager` (`utils/dask_manager.py`) is disabled by
default — `ENABLED = int(os.getenv('DASK_NODES', '0')) > 0`, and `DASK_NODES`
is not set in `.env`. The fallback is a `ThreadPoolExecutor(max_workers =
SLURM_CPUS_PER_TASK)` (`dask_manager.py:122-130`), which is what all current
and planned runs use. `dask-jobqueue` is installed (requirements.txt), so
enabling it is a pure env-var flip — the question is whether it's worth it.

**What it would do**: spin up extra whole Slurm nodes (`--nodes=1 --exclusive`,
28 cores / 100GB per worker job, walltime 7-00) and dispatch tasks to them,
giving parallelism beyond one node's 128 cores.

**Why it does not help**:
1. **It does not touch the expensive part.** The dominant cost (parse,
   bufferize, lower, JIT, run — ~95% of iteration) happens inside a fresh
   child process per execution (`execution.py` `__execute_bufferized_code_isolated`,
   called from every task). Dask workers do not remove that child spawn — each
   task still pays it. Dask's long-lived workers only persist *data* (the
   `Benchmarks` object, `main_exec_data`, imported modules) — and that data is
   already loaded once in the parent and shared via singletons.
2. **Training caps at `bench_count=64` anyway.** One wave per iteration; one
   node's 64 threads already cover it. Extra Dask nodes would sit idle between
   waves.
3. **Eval is not the bottleneck.** Eval (2,363 benches) is the only workload
   that could use >128 workers, but it is a separate Slurm job, already ~5x
   faster at 64 workers, and eval wall-clock is not the paper's constraint.

**Costs**:
- Extra `--exclusive` node requests → queue latency on a busy cluster and
  wasted resources (a whole node per Dask worker job).
- ~378 lines of machinery (cluster management, persistent futures, worker
  restart on timeout) with new failure modes.
- Contradicts the V5 scope decision: "no code changes to `dask_manager.py`"
  (`v5_training_acceleration.md` §Approach) — enabling Dask changes the
  execution architecture under the platform run.

**Relation to §3.6**: Dask's persistence is data-only — it does not provide
the parse-once-per-model reuse that makes a worker pool worthwhile in
full-model eval. If parallel full-model eval is ever needed, the spawn-based
pool in §3.6 is the simpler, cheaper alternative.

**When it WOULD matter**: cross-node parallelism for a workload bigger than
one node — e.g. full-model eval over many large models at once, or a
multi-thousand-bench eval sweep. Then it is the right tool (already
installed); until then, keep `DASK_NODES` unset.

---

## 4. Phase 2: Full-Model Evaluation Pipeline (Weeks 2-3)

### 4.1 Evaluation Entry Point (`rl_autoschedular_v5/eval_fullmodel.py`)

```python
def evaluate_full_models(
    model: HiearchyModel,
    full_benchmarks: FullModelBenchmarks,
    block_benchmarks: Benchmarks,  # for feature extraction compatibility
    checkpoint_step: int
) -> dict[str, int]:
    """
    Greedy rollout on full models using block-trained policy.
    
    Args:
        model: Trained policy (block mode)
        full_benchmarks: Full model dataset
        block_benchmarks: Block dataset (for feature extraction parity)
        checkpoint_step: For logging
    
    Returns:
        {model_name: exec_time_ns}
    """
    model.eval()
    results = {}
    
    for model_name in full_benchmarks.eval_models:
        print(f"Evaluating full model: {model_name}")
        
        # 1. Get full model features (graph + all ops)
        full_features = full_benchmarks.get_model_features(model_name)
        
        # 2. Determine op order
        op_order = get_topological_order(full_features.graph)
        if Config().full_model_op_order == "reverse_topological":
            op_order = list(reversed(op_order))
        
        # 3. Parse full model once, keep Module alive
        executor = FullModelExecutor(full_features.code)
        
        # 4. Schedule each op in order
        all_transforms = []
        for op_tag in op_order:
            # Extract per-op features (same as block training)
            op_features = full_features.op_features[op_tag]
            
            # Build OperationState compatible with block policy
            state = build_op_state_from_full_model(
                op_tag=op_tag,
                op_features=op_features,
                full_features=full_features,
                block_benchmarks=block_benchmarks  # for normalization stats
            )
            
            # Greedy policy rollout
            obs = Observation.from_state(state)
            with torch.no_grad():
                actions_index, _, _ = model.sample(obs.to(device), greedy=True)
            
            # Convert to actions and apply to FULL module
            op_transforms = actions_index_to_transforms(actions_index, state)
            for action in op_transforms:
                action.apply(executor.module)
            
            all_transforms.append(op_transforms)
        
        # 5. Execute full transformed model
        exec_time, success, cache_miss, error = executor.execute_all(transform_history=all_transforms)
        
        if success:
            baseline = full_benchmarks.baseline_times[model_name]
            speedup = baseline / exec_time
            results[model_name] = exec_time
            print(f"  {model_name}: {exec_time/1e6:.1f}ms (speedup: {speedup:.2f}x)")
        else:
            results[model_name] = None
            print(f"  {model_name}: FAILED - {error}")
    
    return results
```

### 4.2 Per-Op State Building (Bridge Block → Full Model)

```python
def build_op_state_from_full_model(
    op_tag: str,
    op_features: OperationFeatures,
    full_features: FullModelFeatures,
    block_benchmarks: Benchmarks
) -> OperationState:
    """
    Build OperationState identical to what block Env.reset() produces.
    Uses same feature normalization, producer selection, etc.
    """
    # Producer selection: prefer last producer (same as Env.__init_op_state)
    producer_tag = None
    producer_operand_idx = None
    producer_features = None
    
    if op_features.producers:
        producer_tag = op_features.producers[-1][0]
        producer_operand_idx = min(idx for t, idx in op_features.producers if t == producer_tag)
        producer_features = full_features.op_features[producer_tag].copy()
    
    return OperationState(
        bench_idx=0,  # not used in eval
        bench_name=full_features.bench_name,
        operation_tag=op_tag,
        original_operation_features=op_features.copy(),
        operation_features=op_features.copy(),
        producer_tag=producer_tag,
        producer_operand_idx=producer_operand_idx,
        producer_features=producer_features,
        transformation_history=[[]],
        terminal=False,
    )
```

### 4.3 Results Logging

```python
# Save to: results/<exp>/run_N/eval_fullmodel/checkpoint_<step>.json
{
    "checkpoint": 500,
    "timestamp": "2026-07-22T10:30:00Z",
    "results": {
        "resnet18_linalg": 123456789,
        "vgg11_linalg": 234567890,
        "t5_linalg": null,  # failed
        "gpt2_linalg": 9876543210
    },
    "speedups": {
        "resnet18_linalg": 1.83,
        "vgg11_linalg": 2.15,
        "gpt2_linalg": 1.42
    },
    "failed": ["t5_linalg"]
}
```

---

## 5. Phase 3: Integration & Testing (Weeks 3-4)

### 5.1 Slurm Script: `scripts/eval/eval_fullmodel.sh`

```bash
#!/bin/bash
#SBATCH --job-name=eval_fullmodel
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=compute
# For very large models (GPT-2+), raise:
# #SBATCH --mem=256G
# #SBATCH --time=8:00:00
# Note: full-model eval is CPU-bound MLIR execution (no model training),
# so it stays on the Jubail `compute` partition — do NOT burn C2 GPU quota
# (team cap ~5 GPUs) on it. GPU is not needed here.

source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/new_dataset/train/v5_fullmodel_eval.json

# Args: checkpoint_step
CHECKPOINT=${1:-500}

python -m rl_autoschedular_v5.eval_fullmodel \
    --config $CONFIG_FILE_PATH \
    --checkpoint $CHECKPOINT \
    --output results/fullmodel_eval_${CHECKPOINT}.json
```

### 5.2 Test Matrix

| Stage | Models | Expected Time | Success Criteria |
|-------|--------|---------------|------------------|
| Parse test | 5 small (<50MB) | <5 min | All ops extracted, graph correct |
| Transform apply | ResNet18, VGG11 | <10 min | All transforms apply, no MLIR crash |
| Execution | ResNet18, VGG11 | <30 min | Speedup measured, matches block baseline |
| Medium models | T5, BERT-base (~200MB) | <1 hr | Completes on Jubail compute (480GB) |
| Large models | GPT-2 (950MB) | <2 hr | Completes on compute partition (128-core node) |
| XL models | 2.5GB model (~2000 ops) | <1 hr | Completes on compute with high `--mem` |

### 5.3 Comparison Baseline

Compare full-model eval speedups against:
1. **Block baseline**: Same checkpoint evaluated on block eval set
2. **Bag-of-schedules**: Sum of per-op speedups (current V4.9 paper method)
3. **Full-model greedy**: This V5 method

| Model | Block Eval | Bag-of-Schedules | Full-Model Greedy (V5) |
|-------|------------|------------------|------------------------|
| ResNet18 | 1.83x | 1.83x | Target: ≥1.83x |
| T5 | N/A | 10.07x | Target: ≥10.07x |
| GPT-2 | N/A | N/A | Functional |

### 5.4 Reconcile with Existing `ckpt_scan` Scripts (already-implemented overlap)

`scripts/checkpoint/ckpt_scan_all.sh` + `scripts/checkpoint/submit_ckpt_scan.sh`
already evaluate trained checkpoints against full `.mlir` files (v4_5 policy,
`results/full_model/scan/<model>_scan.json`, 19 models × 14 checkpoints). They
use a block-based approach with parallel block evaluation.

**Design requirement**: V5.2 must **reuse and extend** these scripts, not
re-implement them:

- Reuse the checkpoints→models iteration and scan-result layout
  (`results/full_model/scan/`, `merge_ckpt_scan.py`).
- Replace the v4_5 policy import with the `rl_autoschedular_v5` policy; keep
  checkpoint compatibility (V5 loads V4.9 checkpoints — same architecture).
- The new `eval_fullmodel.py` provides the per-model greedy rollout; the scan
  scripts become the batch driver over checkpoints × models.
- Keep `ckpt_scan_all.sh` on the `compute` partition (CPU-bound; already
  `--cpus-per-task=128 --mem=300G` after the bergamo-constraint removal).

| Script | Status | V5.2 action |
|--------|--------|-------------|
| `ckpt_scan_all.sh` | ✅ works (v4_5) | Point at `rl_autoschedular_v5`; keep on `compute` |
| `submit_ckpt_scan.sh` | ✅ works | Same change; keep resource params |
| `merge_ckpt_scan.py` | ✅ works | Extend for `eval_fullmodel` output shape |
| `eval_fullmodel.py` | ❌ new | The greedy full-model rollout (this plan §4.1) |

---

## 6. Data Layout

```
data/
├── data/
│   ├── full_models/
│   │   ├── resnet18_linalg.mlir
│   │   ├── vgg11_linalg.mlir
│   │   ├── t5_linalg.mlir
│   │   ├── gpt2_linalg.mlir
│   │   └── ... (20+ models)
│   ├── full_models_base_times.json      # {"resnet18_linalg": 123456789, ...}
│   ├── full_models_train.json           # ["resnet18_linalg", "vgg11_linalg", ...]
│   └── full_models_eval.json            # ["t5_linalg", "gpt2_linalg", ...]
│
├── config/
│   └── new_dataset/
│       └── train/
│           └── v5_fullmodel_eval.json   # Training config + full_model_* fields
│
└── results/
    └── <experiment>/
        └── run_N/
            └── eval_fullmodel/
                ├── checkpoint_100.json
                ├── checkpoint_200.json
                └── ...
```

---

## 7. Resource Estimates (Jubail compute partition)

| Model | Parse | Transform (all ops) | Execute | Peak Mem | Total |
|-------|-------|---------------------|---------|----------|-------|
| ResNet18 (50MB, ~20 ops) | 2s | 10s | 30s | 2 GB | ~1 min |
| VGG11 (80MB, ~30 ops) | 3s | 15s | 45s | 3 GB | ~1.5 min |
| T5 (200MB, ~100 ops) | 8s | 60s | 2 min | 8 GB | ~4 min |
| BERT-base (300MB, ~150 ops) | 12s | 90s | 3 min | 12 GB | ~6 min |
| GPT-2 (950MB, ~765 ops) | 45s | 8 min | 10 min | 40 GB | ~20 min |
| 2.5GB model (~2000 ops) | 2 min | 20 min | 30 min | 100 GB | ~55 min |

**Hardware Requirements**:
- Jubail compute partition standard nodes (128 cores, 480 GB): all models up to and including GPT-2
- High-memory cases (>100 GB peak): request higher `--mem` on the same compute partition (or the 1TB node subset if still needed)
- `--cpus-per-task=64` for parallel MLIR pass execution
- **No GPU needed** — full-model eval is CPU-bound MLIR execution; keep it off the C2 `nvidia` partition (team GPU cap ~5).

---

## 8. Key Technical Decisions (Requires Confirmation)

### 8.1 Transform Application Target
**Question**: During full-model eval, when scheduling `op_i`, apply transform to:
- **Option A**: Full Module (affects all ops, matches training Env behavior)
- **Option B**: Cloned submodule of just `op_i` (isolated, but loses cross-op effects)

**Recommendation**: Option A (full Module). Matches `Env.step()` semantics where transforms are applied to the benchmark's full code.

### 8.2 Feature Freshness
**Question**: After applying transform for `op_i`, how to get features for `op_{i+1}`?
- **Option A**: Re-run AST dumper on modified full model (slow, ~2-45s per op)
- **Option B**: Update features analytically (fast, matches block `Env.__update_state_infos`)

**Recommendation**: Option B. Use same `action.update_features()` logic as block training. Only re-parse if analytical update fails.

### 8.3 Op Order
**Question**: Process ops in topological order (producers→consumers) or reverse?
- **Topological**: Producers scheduled first → fusion opportunities visible to consumers
- **Reverse**: Consumers scheduled first → can pull producers into fusion

**Recommendation**: Configurable via `full_model_op_order`. Default: `topological`.

### 8.4 Checkpoint Compatibility
**Question**: Will V5 load V4.9 checkpoints directly?
- **Yes**: Same model architecture (TransformerEncoder + policy/value heads)
- **Config**: V5 config must match V4.9 config for observation/action spaces

---

## 9. Implementation Checklist

### Phase 1: Infrastructure
- [ ] AST dumper `--full-model` flag
- [ ] `FullModelStats`, `FullModelFeatures` dataclasses
- [ ] `extract_full_model_features(code: str)` 
- [ ] `FullModelBenchmarks` class
- [ ] Config fields: `full_model_*`
- [ ] `execute_model()` with dynamic timeout + memory monitor
- [ ] Module caching in `FullModelExecutor`
- [ ] Persistent worker pool (spawn-based, §3.6) — only if parallel full-model eval is needed

### Phase 2: Evaluation Pipeline
- [ ] `eval_fullmodel.py` entry point
- [ ] `build_op_state_from_full_model()` bridge function
- [ ] Topological sort utilities
- [ ] Results logging to `eval_fullmodel/checkpoint_XXX.json`

### Phase 3: Integration
- [ ] `eval_fullmodel.sh` Slurm script
- [ ] Config: `v5_fullmodel_eval.json`
- [ ] Test on 5 small models
- [ ] Test on ResNet18, VGG11
- [ ] Test on T5, BERT-base (Jubail)
- [ ] Test on GPT-2 (Bergamo)
- [ ] Compare speedups vs block baseline

---

## 10. Future Extensions (Post-V5)

| Extension | Description | Effort |
|-----------|-------------|--------|
| **GNN Encoder** | Add GraphSAGE/GAT on full model graph for global context | Medium |
| **Hierarchical Actions** | Global (vectorize all) + Group (fuse chain) + Local (per-op) | Large |
| **Full-Model PPO** | Train end-to-end on full models (curriculum: small→large) | Very Large |
| **Fusion Heuristics** | Learned fusion policies (FuseChain, TileFuseWindow) | Medium |
| **Cost Model** | Replace execution with learned cost model for faster eval | Large |

---

## 11. References

- [FUTURE_WORKS_RL_FULL_MODEL_SUPPORT.md](FUTURE_WORKS_RL_FULL_MODEL_SUPPORT.md) - Original full-model training design
- [TRAINING_AND_EVALUATION.md](pipeline/TRAINING_AND_EVALUATION.md) - Current training/eval workflow
- [VERSIONS.md](design/VERSIONS.md) - Package version history
- [CONFIG.md](design/CONFIG.md) - Config schema reference