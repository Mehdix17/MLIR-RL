# MLIR-RL — Agent Quick-Start

Reinforcement-learning auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC cluster, Conda env at `~/envs/mlir`.

## Must-Do Environment Setup

**Interactive / local use** needs these three steps **in this order**:

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

**Slurm scripts** (`train.sh`, `eval.sh`, `get_base.sh`, `get_pytorch_times.sh`) already handle this internally — they source `.env` and activate the conda env. You only need the steps above for running Python scripts directly or using `pipeline.sh`.

The `.env` file sets:

- `PYTHONPATH` to the in-tree LLVM build’s MLIR Python bindings
- `LD_LIBRARY_PATH` to the Conda env **and** a GCC-14 `libstdc++` (required on this cluster)
- `LLVM_BUILD_PATH`, `MLIR_SHARED_LIBS`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`
- Neptune credentials (`NEPTUNE_PROJECT`, `NEPTUNE_API_TOKEN`)

**Critical:** `LD_LIBRARY_PATH` must include both `~/envs/mlir/lib` and the GCC-14 path. Missing either causes `ImportError` or `GLIBCXX_3.4.29` errors.

## CONFIG_FILE_PATH — Set Before Any Python

`utils.config.Config` is a **singleton** that reads `CONFIG_FILE_PATH` **at first import**. Any Python script that touches `Config` (directly or transitively) will fail if `CONFIG_FILE_PATH` is not set before the import.

```bash
export CONFIG_FILE_PATH=config/train1.json
```

The Slurm scripts set this automatically. For custom scripts, always import `dotenv` and load `.env` first, then set `CONFIG_FILE_PATH`.

## LLVM Build Gotchas

`llvm-project/` is a full LLVM/MLIR build (release/19.x). If it was compiled by another user, the MLIR Python bindings are broken symlinks. Fix:

```bash
cd llvm-project
git checkout HEAD -- $(git ls-files mlir/python/mlir/ | tr '\n' ' ')
find build/tools/mlir/python_packages/mlir_core -type l | while read link; do
    target=$(readlink "$link")
    if echo "$target" | grep -q "OTHER_USER"; then
        new_target=$(echo "$target" | sed 's|/scratch/OTHER_USER/|/scratch/YOUR_USER/|g')
        rm "$link" && ln -s "$new_target" "$link"
    fi
done
```

Use `rm + ln -s` (not `ln -sf`) — `-f` can fail with "Permission denied" on broken symlinks to inaccessible paths.

## Implementation Packages (Versioned Agents)

| Package                | Purpose                              |
| ---------------------- | ------------------------------------ |
| `rl_autoschedular`     | Baseline — **must remain untouched** |
| `rl_autoschedular_v1`  | **Hardware-aware observation** (our contribution) |
| `rl_autoschedular_v2`  | Shaped reward                        |
| `rl_autoschedular_v2_5`| Hardened shaped reward (fixes V2)    |
| `rl_autoschedular_v3`  | Transformer loop-nest encoder        |
| `rl_autoschedular_v4`  | Integrated V1 + V2 + V3              |
| `rl_autoschedular_v4_5`| **Robust Integrated** (Integration + Hardened Reliability, trained) |
| `rl_autoschedular_v45_no_hw`     | Ablation: hardware-awareness disabled |
| `rl_autoschedular_v45_no_shaped_reward` | Ablation: no reward shaping |
| `rl_autoschedular_v45_no_transformer` | Ablation: baseline policy (no transformer) |
| `rl_autoschedular_v5`+ | Future novelties (one per version)   |

Each `vN` is a **full standalone copy** of the baseline with internal imports redirected to itself. Do **not** mix imports between packages.

### Ablation Study Structure

Three ablation implementations remove exactly one novelty from V4.5:

| Ablation | Disabled Feature | Impl Package | Results Dir |
|----------|-----------------|--------------|-------------|
| No HW | `hardware_auto_detect: false`, all HW overrides = 0 | `rl_autoschedular_v45_no_hw` | `results/ablation_no_hw/` |
| No Shaped Reward | `reward_shaping_enabled: false` | `rl_autoschedular_v45_no_shaped_reward` | `results/ablation_no_shaped_reward/` |
| No Transformer | Uses LSTM instead of Transformer | `rl_autoschedular_v45_no_transformer` | `results/ablation_no_transformer/` |

Each uses `scripts/ablation_eval.py` (nearly identical to `eval.py` but requires `EVAL_DIR`). Configs: use `v45_no_*_blocks.json` for block-based eval, `v45_no_*_full_model.json` for full-model eval. Current status: no_hw and no_transformer already evaluated; no_shaped_reward currently evaluating in 2 jobs (`rew-eval-low`, `rew-eval-high`).

## Hardware-Aware Observation (V1 — Our Contribution)

V1 adds a 7-element hardware feature vector to every observation, auto-detected at module import from the running machine:

- **L1 cache** (KB, normalized /256), **L2 cache** (/4096), **L3 cache** (/65536)
- **Physical cores** (/256), **Logical cores** (/512)
- **SIMD width** (/1024), **Clock MHz** (/6000)

Detection reads `/sys/devices/system/cpu/cpu0/cache/` and `/proc/cpuinfo`. Controlled by config fields:
- `hardware_auto_detect: true` → auto-detect (recommended for single-machine runs)
- `hardware_l1_kb`, `hardware_l2_kb`, ... → manual overrides for cross-hardware experiments

The hardware vector is concatenated to the LSTM output in `model.py` forward pass. Policy and value networks now consume `[op_lstm, producer_lstm, hardware_features]`, allowing them to condition decisions on hardware properties.

**Key files:** `rl_autoschedular_v1/observation.py` (HardwareFeatures class), `rl_autoschedular_v1/model.py` (LSTMEmbedding integration).

**Verification:**
```bash
python -c "from rl_autoschedular_v1.observation import HARDWARE_VECTOR; print(HARDWARE_VECTOR)"
```

## Multi-Cluster Evaluation Mission

**Goal:** Evaluate the already-trained V4.5 agent on 3 clusters with different CPUs to validate that hardware-aware observation enables cross-hardware generalization.

**Training hardware:** V4.5 was trained on **Dalma** (Intel Xeon E5-2680 v4, 28 cores, 112 GB RAM).

**Eval set:** `results/experiment3/exec_times/base_eval.json` (3015 benchmarks from the flat `data/all/` directory). Multi-hardware scripts hardcode `EVAL_DIR=results/experiment3/v4_5_agent/run_0/models` and use `EVAL_LAST_ONLY=1` (evaluates only `model_1999.pt`).

### Cluster Specs

| Cluster | Slurm constraint | CPUs/node | CPU model | RAM/node |
|---------|-----------------|-----------|-----------|----------|
| Dalma (trained on) | `--constraint=dalma` | 28 | Intel Xeon E5-2680 v4 (Broadwell) | 112 GB |
| Jubail | `-C jubail` | 128 | AMD EPYC 7742 (Zen 2) | 480 GB |
| Bergamo | `-C bergamo` | 256 | AMD EPYC 9754 (Zen 4) | 1 TB |

### Evaluation Commands

```bash
sbatch scripts/dalma_eval.sh config/v4_5.json     # training hardware (baseline)
sbatch scripts/jubail_eval.sh config/v4_5.json    # different AMD Zen 2
sbatch scripts/bergamo_eval.sh config/v4_5.json   # different AMD Zen 4, 256 cores
```

All scripts set `EVAL_LAST_ONLY=1` internally (evaluates only the last checkpoint, `model_1999.pt`).

**Hypothesis:** The hardware-aware agent should adapt scheduling decisions (tile sizes, vectorization, parallelism) to each cluster's CPU characteristics, producing competitive speedups across all three architectures despite training on only one.

## Architecture: AST & Feature Pipeline

The RL agent does **not** parse MLIR directly. A **C++ AST dumper** (`tools/ast_dumper/`, env var `AST_DUMPER_BIN_PATH`) serves as the bridge between raw MLIR and the Python observation tensor:

```
.mlir file  →  C++ AST dumper  →  structured text (operations, loops, graph)  →  state.py  →  BenchmarkFeatures  →  observation.py  →  torch.Tensor
```

**What the AST dumper outputs** (parsed by `state.py::__extract_bench_features_from_ast_result()`):
- `#START_OPERATION` blocks — each linalg op with its tag, name, type
- `#START_NESTED_LOOPS` — loop bounds, steps, iterator types (parallel/reduction)
- `#START_LOAD_DATA` / `#START_STORE_DATA` — memory access patterns per loop dimension
- `#START_OP_COUNT` — arithmetic op counts (mul, add, etc.)
- `#BEGIN_GRAPH` — producer→consumer edges between operations

**What the observation tensor contains** (built by `observation.py::Observation.from_state()`):
1. **OpFeatures** — operation type one-hot, loop upper bounds, iterator types, load/store access matrices, arithmetic counts
2. **ProducerOpFeatures** — same features for the producing operation (or zeros)
3. **ActionHistory** — multi-hot encoding of previously applied transformations
4. **NumLoops** — current loop nest depth
5. **ActionMask** — bitmask of currently legal actions

**Key files:** `rl_autoschedular/state.py` (feature extraction + `BenchmarkFeatures` dataclass), `rl_autoschedular/observation.py` (tensor construction), `rl_autoschedular/benchmarks.py` (bulk loading).

## Safety & Hardened Reliability (V4.5+)

Starting with V4.5, the system includes proactive safeguards against MLIR instability:

1. **Process Isolation:** All JIT transformations and execution profiling run in independent subprocesses. If a transformation sequence causes a native crash (e.g., LLVM assertion failure), the worker process dies, but the main RL training/evaluation loop remains unaffected.
2. **Success-Contingent Rewards:** To prevent the agent from learning "risky" behavior (e.g., aggressive tiling that causes timeouts), shaped rewards are only granted if the final code executes successfully. If it fails, all intermediate rewards are zeroed, and a total failure penalty is applied.
3. **Stability Rails:** The action space now masks out transformations that are statistically correlated with instability, such as vectorizing deeply nested loops (>6) or exceeding a transformation complexity threshold (>4 steps).
4. **Resilient Markers:** Progress is saved incrementally in `results/experiment3/global_markers/`. If a job is interrupted, it will automatically resume from the last completed benchmark.

## Model Benchmarking & Results Organization

The system evaluates learned policies on collections of MLIR operations across three domains:

### **Benchmark Datasets**

- **Vision Operations** (`data/nn/vision_operations.json`, `conv_2d_vision_operations.json`) — Conv2D, pooling, ReLU; commonly used to debug scheduling
- **Full Neural Net** (`data/nn/train_operations.json`) — MatMul, Conv2D, Add, pooling mixed workloads
- **Synthetic / Domain-Specific** (`data/linalg/`, `data/gnn/`) — Pure linalg, graph neural net ops

Each benchmark JSON contains:
```json
{
  "operation_name_1": 5.2,     // baseline execution time (ms), unoptimized
  "operation_name_2": 12.1,
  ...
}
```

### **Results Directory Structure**

```
results/
  ├─ experiment3/              # Primary active experiment
  │  ├─ exec_times/            # Cached baseline timings (don't edit)
  │  │  ├─ v4_5_base.json      # Unoptimized times for v4_5 agent
  │  │  ├─ v4_base.json
  │  │  ├─ ...
  │  │  └─ pytorch_times.json  # PyTorch reference times
  │  │
  │  ├─ v4_5_agent/             # Canonical agent dir (from implementation.py)
  │  │  ├─ run_0/
  │  │  │  ├─ models/           # model_0.pt .. model_1999.pt
  │  │  │  ├─ logs/train/       # reward, entropy, final_speedup (per epoch)
  │  │  │  ├─ logs/train_ppo/   # PPO loss metrics
  │  │  │  ├─ logs/eval/        # eval_exec_times.json, speedup files
  │  │  │  ├─ exec_data.json    # schedule→execution-time mapping
  │  │  │  └─ tags              # metadata tags
  │  │  └─ run_N/               # repeated runs (same structure)
  │  │
  │  ├─ v4_agent/, v2_5_agent/, ...  # Same structure, older versions
  │  └─ global_markers/         # V4.5+ resilience: iteration-level checkpoints
  │
  ├─ ablation_no_hw/            # Ablation study: hardware-awareness removed
  │  ├─ exec_times/              # base.json, base_eval.json, pytorch.json
  │  └─ rl_autoschedular_v45_no_hw_agent/
  ├─ ablation_no_shaped_reward/  # Ablation: no reward shaping (currently running)
  │  ├─ exec_times/
  │  └─ rl_autoschedular_v45_no_shaped_reward_agent/
  ├─ ablation_no_transformer/    # Ablation: baseline policy
  │  ├─ exec_times/
  │  └─ rl_autoschedular_v45_no_transformer_agent/
  └─ full_model/                # Full benchmark runs
     ├─ baselines/              # full_baselines.csv, blocks_baseline.json
     ├─ blocks/                 # per-block detail for block-based models
     ├─ eval/                   # honest_blocks.json
     ├─ scan/                   # checkpoint scan results
     └─ summary/                # merged results (final_merged.csv)
```

### **Evaluation Results**

Eval saves results in `<agent>/run_0/logs/eval/eval_exec_times.json` as `{bench_name: optimized_time_ns}` (same format as `exec_times/base_eval.json`). Speedup per checkpoint in `logs/eval/final_speedup`, average in `logs/eval/average_speedup`.

### **Workflow: From Data to Benchmarking**

1. **Collect Benchmarks** — Generate MLIR code + baseline times (vision2mlir, tf2mlir, gnn2mlir in `data_utils/`)
2. **Split for Training** — `split_json.py` divides into train/eval (e.g., 70%/30%)
3. **Train Agent** — `train.sh` runs PPO on train split, outputs checkpoints + loss curves to Neptune
4. **Evaluate** — `eval.sh` loads best checkpoint, optimizes eval split, logs results to `logs/eval/eval_exec_times.json`
5. **Compare** — Dashboard aggregates speedups across all versions and ablations for comparative analysis

### **Benchmark Analysis Quick Start**

**View averaged speedups** (across all test benchmarks):
```bash
python -c "
import json
with open('results/experiment3/v4_5_agent/run_0/logs/eval/eval_exec_times.json') as f:
    d = json.load(f)
    print(f\"Optimized {len(d)} benches\")
"
```

**Compare two agents**:
```bash
python -c "
import json
with open('results/experiment3/v4_5_agent/run_0/logs/eval/eval_exec_times.json') as f:
    v45 = json.load(f)
with open('results/experiment3/v4_agent/run_0/logs/eval/eval_exec_times.json') as f:
    v4 = json.load(f)
print(f\"V4.5 optimized {len(v45)} benches, V4 optimized {len(v4)} benches\")
"
```

**Monitor training via Neptune** — Link set in `.env` (`NEPTUNE_PROJECT`) — view loss, reward, PPO metrics live during training runs.

## Full-Model End-to-End Optimization: Architecture & Challenges

### Overview

Full-model optimization applies the trained RL agent (v4.5, checkpoint 715) to complete `.mlir` model files, producing end-to-end execution time improvements. Success rate: **9/19 models** with average speedup **2.67x** (t5: 10.07x, lstm: 4.53x). Remaining 10 models use block-based fallback (average 1.046x due to missing context).

**Key challenges**:
1. **MLIR Bufferization** — Transform-dialect pass crashes on certain linalg patterns (10 models affected)
2. **PyTorch JIT** — 4 models (gpt2/vit_b_16) fail JIT tracing due to HuggingFace library design (detailed in [PYTORCH_JIT_DEBUGGING.md](docs/PYTORCH_JIT_DEBUGGING.md))
3. **AST Dumper Robustness** — Timeout handling and worker crash recovery required for large models (fixed in v4.5+)
4. **Schedule Application Bottleneck** — Old per-action subprocess approach took ~25h for gpt2 (950MB file); now batched in-process transforms: ~2–3 min

### PyTorch JIT Failures: Root Cause & Workarounds

| Model | Issue | Root Cause | Current Workaround |
|-------|-------|-----------|------------------|
| gpt2, gpt2-large, gpt2-medium | Trace fails | `'Tensor' has no attribute 'get_seq_length'` | Use eager times only |
| gpt2 (all variants) | Script fails | `GPT2Config.__init__` uses `**kwargs` — incompatible with `torch.jit.script()` | Same |
| vit_b_16 | Trace fails | Dynamic attention paths produce different graphs per invocation | Use eager times only |
| vit_b_16 | Script fails | `'Tensor' has no attribute 'logits'` — internal ops not in TorchScript scope | Same |

**Impact**: 4/22 models (18%) have empty PyTorch JIT columns in baselines CSV. All 4 still execute in eager mode (valid benchmark) and participate in full-model RL optimization.

**Future Fixes** (in priority order):
1. **Model Surgery** (2–4h) — Extract & freeze model components, trace only forward path
2. **Custom TorchScript Ops** (8–16h) — Implement custom CUDA kernels for attention/ops
3. **ONNX Intermediate** (1–2h) — Trace via ONNX Runtime (not applicable for CPU-only benchmarking)

See [PYTORCH_JIT_DEBUGGING.md](docs/PYTORCH_JIT_DEBUGGING.md) for detailed analysis, reproduction steps, and implementation roadmap.

### MLIR Bufferization Crashes: Workarounds

**Problem**: 10 models fail MLIR `one-shot-bufferize` transform-dialect pass with `op was not bufferized`. These are MLIR compiler bugs, not pipeline bugs. Affected: albert, bert, distilbert, bart, deberta, convnext_tiny, densenet121, gat, gpt2, vit_b_16.

**Current Approach**:
1. **Full-model** (checkpoint 715): Succeeds for 9 models (t5, lstm, vgg11, resnet18/50, resnext50, gcn, efficientnet_b0, mobilenet_v3_small)
2. **Block-based fallback** (checkpoint 1999): Decompose into extracted operation blocks (window=5, stride=3), optimize each block independently
3. **Three-level execution pipeline**:
   - v4_5 bindings execution (300s cap)
   - Multiprocess bindings (7200s timeout)
   - `mlir-opt -one-shot-bufferize` subprocess + `mlir-cpu-runner` JIT (7200s timeout for large models)

See [FULL_MODEL.md § 4](docs/FULL_MODEL.md#4-root-cause-analysis-mlir-baseline-failures) for technical details.

## Documentation References

For more detailed guides and architectural decisions, refer to the following documents:

- [Main README config properties](README.md) — Exhaustive list of `CONFIG_FILE_PATH` fields.
- [HPC Setup Guide](docs/HPC%20Setup.md) — Slurm cluster specific instructions.
- [Training Guide](docs/TRAINING_GUIDE.md) — Comprehensive guide on training the RL agent.
- [RL Agent Tutorial](docs/RL_AGENT_TUTORIAL.md) — Walkthrough of the RL framework and logic.
- [Pipeline Orchestration](docs/PIPELINE.md) — Full lifecycle of MLIR baseline up to evaluation.
- [Full-Model End-to-End Optimization](docs/FULL_MODEL.md) — Complete architecture, bug fixes, results, and known limitations for full-model RL optimization.
- [PyTorch JIT Debugging](docs/PYTORCH_JIT_DEBUGGING.md) — Root cause analysis of gpt2/vit_b_16 JIT failures and proposed workarounds (model surgery, custom ops, ONNX).
- [Dashboard Guide](docs/dashboard.md) — Streamlit evaluation instructions.
- [Data Utils](data_utils/README.md) — Tools for generating synthetic MLIR datasets and extraction operations.
- [Novelties](docs/NOVELTIES.md) and [Versions](docs/VERSIONS.md) — Changelog and upcoming version plans.

## Manuscript & Paper Citations

**CRITICAL REMINDER:** When writing or editing our current manuscript, **always cite the published or officially archived papers**. Do not cite the raw master's thesis manuscripts directly in the text.

- **Baseline Environment (2024 Focus)**
  - **Manuscript Path:** `manuscript/references/Bendib 2024.md`
  - **Always Cite the Paper:** Use the citation `bendib2024reinforcement` (Master's thesis).
- **Extended System / Our Contributions (2025 Focus)**
  - **Manuscripts Paths:** `manuscript/references/Rafik & Djad 2025.md` and `manuscript/references/Nassim & Mohamed 2025.md`
  - **Always Cite the Paper:** These converged into the arXiv preprint. Use the citation `tirichine2025reinforcement`.

## Config-Driven Workflow

All scripts read `CONFIG_FILE_PATH` (env var) or accept a config path as `$1`. The config JSON selects the implementation:

```json
{
  "implementation": "rl_autoschedular_v3",
  "results_dir": "results/my_experiment",
  ...
}
```

**Implementation resolution order** (from `utils/implementation.py`):

1. `AUTOSCHEDULER_IMPL` env var
2. `--implementation` flag (where supported)
3. Config file `"implementation"` field
4. Default: `rl_autoschedular`

Train/eval Slurm scripts derive the implementation from the config. Override with a second positional arg:

```bash
sbatch scripts/train.sh config/my_config.json rl_autoschedular_v2
```

### Slurm Array Job Mode

`train.sh` and `eval.sh` support Slurm array jobs to run multiple versions:

```bash
sbatch --array=0-3 scripts/train.sh    # task 0→baseline, 1→v1, 2→v2, 3→v3
sbatch --array=0-2 scripts/eval.sh     # same mapping
```

When `SLURM_ARRAY_TASK_ID` is set and no config is provided, the script auto-selects `config/<version>.json`.

### Standard Pipeline Order

```bash
# 1. MLIR baseline (Slurm)
sbatch scripts/get_base.sh config/my_config.json

# 2. PyTorch baseline (Slurm or local)
sbatch scripts/get_pytorch_times.sh config/my_config.json

# 3. Split into train / eval
python scripts/split_json.py config/my_config.json

# 4. Train (Slurm)
sbatch scripts/train.sh config/my_config.json

# 5. Evaluate checkpoints (Slurm)
sbatch scripts/eval.sh config/my_config.json

# 6. Compare in dashboard
cd dashboard && streamlit run dashboard.py --server.fileWatcherType none
```

Or use the one-shot pipeline:
```bash
bash scripts/pipeline.sh config/baseline.json
bash scripts/pipeline.sh config/baseline.json "rl_autoschedular_v1,rl_autoschedular_v3"
```

### Benchmark Format Requirements

The RL agent works on **extracted operation blocks**, NOT full model `.mlir` files. Every `.mlir` file in `benchmarks_folder_path` must have:

- **`{tag = "operation_NNN"}`** annotations on each linalg op — required by the AST dumper for feature extraction
- **`@nanoTime()` timing calls** — measures block execution time
- **Weights passed as function arguments** (not inlined constants) — keeps files small and self-contained
- **A `func.func @main`** entry point returning `(output_tensor, i64)` (tensor + execution delta)

Example block-ready format (`data/all/albert_block_301.mlir`, 29 lines):
```
func.func @main(%arg_0_in_0: tensor<...>, ..., %arg_0_out_0: tensor<...>, ...) -> (tensor<...>, i64) {
  %t0 = call @nanoTime() : () -> i64
  %v0 = linalg.batch_matmul {tag = "operation_211"} ins(...) outs(...) -> ...
  %v1 = linalg.generic {tag = "operation_212"} ... -> ...
  %t1 = call @nanoTime() : () -> i64
  %delta = arith.subi %t1, %t0 : i64
  return %v4, %delta : tensor<...>, i64
}
```

**Full model `.mlir` files are NOT directly usable** — including those in `data/nn/raw_bench/` (e.g. `albert_linalg.mlir`, `resnet18_torch.mlir`). These lack operation tags, timing wrappers, and contain thousands of lines with inlined weight constants. They must be processed into blocks first (see below).

The `json_file` and `eval_json_file` config fields map benchmark **names** to baseline execution times (ns). Each name must match a `.mlir` file in `benchmarks_folder_path` (e.g. `"albert_block_0": 12345` → loads `benchmarks_folder_path/albert_block_0.mlir`).

### Extract Blocks from Raw Model

```bash
python data_utils/extract_blocks.py --input data/nn/raw_bench/albert_linalg.mlir --output-dir data/nn/extracted_blocks/
```

`extract_blocks.py` windows consumer→producer paths into fixed-size blocks (default: 5 ops, stride 3). Alternative: `extract_ops.py` for single-operation files. Then run `scripts/get_base.py` for baseline times and `scripts/split_json.py` for train/eval split.

### Blocks (`data/nn/blocks/`)

Per-model block directories (20 models), extracted from full model `.mlir` files via `extract_blocks.py` (window=5 stride=3). Failed blocks moved to `data/nn/failed_blocks/`. Models: `albert/`, `bart/`, `bert/`, `convnext_tiny/`, `densenet121/`, `deberta/`, `distilbert/`, `efficientnet_b0/`, `gat/`, `gcn/`, `gpt2/`, `gpt2-large/`, `gpt2-medium/`, `lstm/`, `mobilenet_v3_small/`, `resnet18/`, `resnet50/`, `resnext50/`, `t5/`, `vgg11/`, `vit_b_16/`.

**Important:** `benchmarks.py` does `os.path.join(benchmarks_folder_path, bench_name + ".mlir")` — no subdirectory recursion. To evaluate a model, point `benchmarks_folder_path` directly at the model subdir (e.g., `data/nn/blocks/albert`). The flat `data/all/` (~17K files) is what V4.5 was trained on.

### Full-Model End-to-End Evaluation

For models that bufferize end-to-end, the pipeline does not decompose into blocks. 9/19 models succeed; the rest fail MLIR's `one-shot-bufferize` pass and use block-based estimation instead.

**Pipeline:**
```
Full .mlir → AST dumper (preprocess_model.py) → tagged code with {tag = "operation_NNN"}
  → RL agent greedy inference per op (optimize_full_model.py) → apply schedules in-place
  → add @nanoTime() wrapper (add_timing_wrapper.py) → execute → optimized time
```

**Key scripts:**
- `scripts/optimize_full_model.py` — Main orchestrator: tagging, baseline, RL inference, schedule application (batched in-process transforms — parse once, not per-action), execution
- `scripts/optimize_full_model.sh` — Slurm array job wrapper (tasks 0–19, one per model)
- `scripts/preprocess_model.py` — Runs C++ AST dumper to tag linalg ops
- `scripts/add_timing_wrapper.py` — Wraps `@main` with `@nanoTime()`, returns `(tensor, i64)`
- `scripts/measure_full_model_baselines.py` — PyTorch eager + JIT timing for all models
- `scripts/merge_full_model_results.sh` — Merges chunk files into unified JSON + CSV

**Checkpoints:** Full-model uses `model_715.pt` (iteration 715, less aggressive). Block-based eval uses `model_1999.pt` (final checkpoint).

**Three-level execution pipeline** (increasing robustness):
1. V4.5 bindings execution (300s cap)
2. Multiprocess bindings with 7200s timeout
3. `mlir-opt -one-shot-bufferize` subprocess + `mlir-cpu-runner` JIT with 7200s timeout

**Key known issues:**
- 10 models fail bufferization (albert, bart, bert, convnext_tiny, deberta, densenet121, distilbert, gat, gpt2, vit_b_16) → use block-based estimation
- roberta: AST dumper fails (`tm_tensor` dialect not found)
- See `docs/FULL_MODEL.md` for full details (395-line guide with bug fixes, edge cases, results)

### Canonical Results Table

`results/full_model/baselines/full_baselines.csv` is the master comparison table:
```
model,mlir_baseline_ns,pytorch_eager_ns,pytorch_jit_ns,mlir_rl_opt_ns
```

Visit `results/full_model/summary/final_merged.csv` for the unified per-model summary (full-model for 9 models, block-based for 10).

## Results Directory Layout

```
results/<experiment>/
  exec_times/
    <prefix>_base.json          # unoptimized MLIR baseline
    <prefix>_base_train.json    # train split
    <prefix>_base_eval.json     # eval split
    pytorch.json                # PyTorch baselines
  <impl_agent>/                 # e.g. old_agent, v1_agent, v2_agent
    run_0/
      models/                   # model_4.pt, model_9.pt, ...
      logs/
        train/
        eval/
      exec_data.json
      eval/
        markers/                # crash-resilience markers
        eval_exec_times.json    # {bench_name: optimized_time_ns}
```

`<prefix>` is resolved by `utils/implementation.py`:
- `rl_autoschedular` → `old` (agent dir: `old_agent`)
- `new_rl_autoschedular` → `new` (agent dir: `new_agent`)
- `rl_autoschedular_vN` → `vN` (agent dir: `vN_agent`)

The config fields `json_file` and `eval_json_file` default to `""` — when empty, paths are auto-derived from `results_dir` and the current implementation via `get_split_file_path()`.

## Singletons & Import Order

`utils.config.Config` and `utils.dask_manager.DaskManager` are **singletons**. `Config` reads `CONFIG_FILE_PATH` at first import. Importing `utils.config` before setting `CONFIG_FILE_PATH` will fail.

`scripts/train.py` and `scripts/eval.py` load `.env` and `.env.debug` via `python-dotenv` at the very top. `.env.debug` is optional (typically absent). Custom scripts should follow the same pattern.

## Operational Gotchas

- **DaskManager is disabled by default** (`ENABLED = False` in `utils/dask_manager.py`). Distributed execution across Slurm workers only activates if you set `ENABLED = True` and `DASK_NODES` env var. All execution runs single-process on the Slurm node otherwise.
- **SIGABRT handler** — `train.py` and `eval.py` install a signal handler that catches native MLIR crashes (`SIGABRT`) and converts them to Python exceptions so training can continue past a bad schedule.
- **`submit_and_monitor.sh`** — Submits a Slurm script, waits for the job to start, then tails its output (`scripts/submit_and_monitor.sh scripts/train.sh config/train1.json`).
- **Check running jobs before submitting** — `squeue -u mb10856`. Two reward-eval jobs currently use 2 nodes (`rew-eval-low`, `rew-eval-high`). Submitting eval with `--cpus-per-task=4` avoids `AssocGrpCpuLimit`.

### Critical: BindingsProcess.ENABLED Must Stay False

`utils/bindings_process.py` has `ENABLED = False`. **Do NOT set it to True.** Reasons:

1. **Fork corrupts MLIR C++ state** — Linux `fork()` duplicates MLIR C++ globals inconsistently. Transform subprocesses fail with `exit code: 1` (12M+ errors per eval run), producing speedup=1.0 for all benchmarks.
2. **Dill-based spawn is too slow** — A dill/subprocess replacement was attempted but each subprocess launch takes 1-2s. With ~24K operations per eval, this is impractical.
3. **In-process (False) works** — Transforms + execution run correctly in-process. The SIGABRT handler catches most MLIR crashes. Some `convnext_tiny_block_*` benchmarks crash with `CollectDiagnosticsToStringScope` assertion — a reproducible MLIR binding bug in certain operation patterns.

If a benchmark crashes with SIGABRT during eval, the process dies and any in-progress eval results are lost. See **Eval Crash Resilience** below for the mitigation.

### Eval Crash Resilience

The eval saves per-benchmark results incrementally via **marker files** at `<agent>/run_0/eval/markers/<bench_name>`. Each marker is a JSON containing `rewards`, `speedup`, `exec_time`, `cache_miss`.

**How it works:**
1. Before evaluating a benchmark, `__execute_states` checks if a marker file exists → skips with "Skipped (cached)" if found
2. After successful evaluation, writes marker atomically via temp file + rename
3. If SIGABRT kills the process mid-eval, only the current benchmark is lost

**To resume a crashed eval:**
```bash
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
export AUTOSCHEDULER_IMPL=rl_autoschedular_v1 CONFIG_FILE_PATH=config/v1.json
export EVAL_LAST_ONLY=1 EVAL_DIR=results/new_experiment/v1_agent/run_0/models
# Markers survive crashes — restart auto-skips completed benchmarks
python scripts/eval.py
```

**To launch eval safely** (survives terminal closure):
```python
import subprocess, os
p = subprocess.Popen(['python', 'scripts/eval.py'],
    stdout=open('logs/eval_myrun.out','w'), stderr=open('logs/eval_myrun.err','w'),
    env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    preexec_fn=os.setpgrp)
```

**To clear markers and start fresh:** delete `<agent>/run_0/eval/markers/` directory.

### Evaluation Results

Eval saves results in multiple formats:
- `eval_exec_times.json` — `{bench_name: optimized_time_ns}` (same format as `base.json`). Saved incrementally after each benchmark.
- `eval/final_speedup` — one speedup per benchmark per checkpoint
- `eval/exec_time/<bench_name>` — per-benchmark optimized times
- `eval/markers/<bench_name>` — crash-resilience marker files

### EVAL_LAST_ONLY — Fast Single-Checkpoint Eval

Env var `EVAL_LAST_ONLY=1` evaluates only the last model checkpoint (e.g., `model_1999.pt`). For 41-checkpoint eval, omit this var.

### Slurm CPU Limits

When submitting eval jobs, use low CPU count to avoid `AssocGrpCpuLimit`:
```bash
sbatch --cpus-per-task=4 --mem=16G --time=02:00:00 ...
```

The eval script auto-discovers `EVAL_DIR` from `results_dir` + implementation. When running outside Slurm, set `EVAL_DIR` explicitly or let the script auto-discover.

### Interactive Session — NEVER Cancel

The user's Slurm interactive session (`squeue` shows as "interactive") is the session running OpenCode. **Never cancel it** — it will kill this agent's connection.

## C++ Tools

- `tools/ast_dumper/` — **Critical bridge**: parses `.mlir` files into structured text consumed by `state.py`. Without it, the RL agent cannot extract features from MLIR. Build artifacts in `tools/ast_dumper/build/`. Env var: `AST_DUMPER_BIN_PATH`.
- `tools/vectorizer/` — Auto-vectorization analysis tool. Env var: `VECTORIZER_BIN_PATH`.
- `tools/pre_vec/` — Pre-vectorizer analysis (no build dir checked in).

Referenced by env vars in `.env`. These C++ tools are called as subprocesses from Python (not linked as libraries).

## Testing & Validation

There is **no pytest suite**. Validation is done via:
- Python compile checks (`python -m py_compile <file>`)
- Import smoke tests (e.g. `python -c "import rl_autoschedular_v3.model"`)
- End-to-end sanity runs on a single benchmark
- Slurm job logs in `logs/`

The only test scripts are `scripts/test_torch_mlir.sh` and `scripts/test_torch_mlir_compile.py` (torch-mlir sanity checks).

## Code Style

- `.flake8` ignores `E501` (line length) and `E402` (module-level import not at top).
- No auto-formatter or type-checker configured.
- Use `from utils.log import print_info, print_success` for consistent colored CLI output.

## Quick Commands

See `quick_commands.txt` for copy-paste workflow commands. Key verification:

```bash
# Verify MLIR Python bindings work
python -c "from mlir.ir import Context; print('OK')"
```
