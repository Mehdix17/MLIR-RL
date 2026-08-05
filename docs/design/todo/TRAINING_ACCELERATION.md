# Training Pipeline, Limitations, and Acceleration Opportunities

## How Training Is Done

The training pipeline is invoked via `scripts/train/train.sh`, which calls the unified entry point `scripts/train/train.py`. This script dynamically imports the implementation package (e.g. `rl_autoschedular_paper_transformer`) based on the config's `implementation` field or the `AUTOSCHEDULER_IMPL` env var.

### Entry Point (`scripts/train/train.py`)

1. Installs a SIGABRT handler (MLIR native crashes → Python exception, not hard kill).
2. Loads `Config`, `Benchmarks`, `Execution` singleton, builds `HiearchyModel`, moves it to `device`.
3. Checks for `--resume`: if a run directory already has model checkpoints, loads the latest and resumes from `start_step`.
4. Runs the PPO training loop (see below).

**Important:** The unified `scripts/train/train.py` does **not** call `evaluate_benchmarks`. Evaluation is a separate process.

### PPO Loop (`scripts/train/train.py:159`)

```
for step in range(start_step, cfg.nb_iterations):
    trajectory = collect_trajectory(train_data, model, step)
    if cfg.value_epochs > 0:
        value_update(trajectory, model, optimizer)
    ppo_update(trajectory, model, optimizer)
    if step % 50 == 0:
        save checkpoint
```

No evaluation happens inside this loop. The loop only does: collect → update → checkpoint.

### Trajectory Collection (`ppo.py:collect_trajectory`)

1. Samples `bench_count` random benchmarks from the training set.
2. For each benchmark, creates an `Env` and steps through it until all operations are terminal. At each step, the model samples an action (on `device` — CPU on Bergamo, GPU on C2/nvidia nodes).
3. After all benchmarks reach terminal state, `DaskManager.map_objs(__execute_states, ...)` executes them in parallel.
4. Execution results (rewards, speedups, exec times) are collected and the trajectory is returned.

### PPO Update (`ppo.py:ppo_update`)

- `cfg.ppo_epochs` (default 4) epochs over the trajectory.
- Batch size: `cfg.ppo_batch_size` (default 64 — standard for academic PPO, should not be changed).
- Adam optimizer, gradient clipping at 0.5, entropy bonus.
- The model forward pass runs on `device`.

### Execution (`execution.py`)

- `Execution.execute_code()` checks a JSON cache first.
- On cache miss: bufferizes and lowers the MLIR module through a ~20-pass pipeline, then runs it via `ExecutionEngine` in an isolated child process (`multiprocessing.Process`).
- If the child crashes (SIGABRT) or times out, falls back to `mlir-opt | mlir-cpu-runner` subprocess.
- Dynamic timeout: `root_exec_time * 5`, capped at 300s.
- All compilation and execution is **CPU-only**. MLIR's `ExecutionEngine` runs LLVM-compiled code on the CPU. This is true regardless of which node type the job runs on — MLIR compilation never uses GPU.

### Parallelism (`dask_manager.py`)

- `DaskManager` is **disabled** by default (`ENABLED = DASK_NODES > 0`, and `.env` does not set `DASK_NODES`).
- When disabled, falls back to a `ThreadPoolExecutor` with `max_workers = SLURM_CPUS_PER_TASK` (12 threads from `train.sh`).
- Each thread calls `__execute_states`, which creates a fresh `Execution` singleton and `Env` per benchmark.

### Evaluation (separate from training)

- Evaluation is submitted as a separate Slurm job: `sbatch scripts/eval/eval.sh config/.../eval.json --checkpoint N`.
- `eval.sh` runs `scripts/eval/eval.py`, which loads a checkpoint and calls `evaluate_benchmarks()` on the full eval set.
- The per-package `train.py` files (e.g. `rl_autoschedular_paper_transformer/train.py`) do call `evaluate_benchmarks` every 100 steps, but these are not used by the Slurm pipeline — `scripts/train/train.py` is the actual entry point.

---

## Cluster Hardware

### Available Machines and Partitions

| Machine / Partition | Node Type | Hardware | Access | Current `train.sh` Uses |
|---------------------|-----------|----------|--------|------------------------|
| **Jubail HPC** — `compute` partition | Bergamo nodes (`bn001-058`) | 256 CPU cores, ~1TB RAM, **no GPU** (`Gres=null`) | Default | ✅ Yes (`--constraint=bergamo`) |
| **C2 QOS** — `nvidia` partition | GPU nodes (e.g. `cn009`) | 128 CPU cores + **3× A100 (80GB)** each | Via `--qos=c2 -p nvidia --gres=gpu:a100:1` | ❌ No |
| **Dalma** partition | V100 nodes (`dn003-005,007-008,011-014`) | CPU + **2× V100** (16-32GB each) | Via `-p dalma --gres=gpu:v100:1` | ❌ No |
| **Kindi** (standalone) | Not on Slurm | AMD EPYC 7742 64-core + **8× A100 (80GB)** | SSH only (`kindi.abudhabi.nyu.edu:4410`) | ❌ No |
| **Green HPC** (NYC) | Various | Shared, often busy | Via Slurm at Greene | ❌ No |

### Key Facts

- **Training currently runs on CPU-only Bergamo nodes** (`compute` partition, `--constraint=bergamo`). `Gres=(null)` means no GPU. `torch.cuda.is_available()` returns `False`, `device = torch.device("cpu")`.
- **You (mb10856) have C2 QOS access** — confirmed via `sacctmgr`. You can submit GPU jobs.
- **C2 QOS has a limit of ~5 concurrent GPUs** across the team (GrpTRES includes `gres+` cap).
- **GPU+CPU on the same node IS possible**: C2 GPU nodes like cn009 have 128 CPU cores + 3 A100 GPUs on the same machine. Dalma V100 nodes also have both.
- **MLIR compilation is always CPU** regardless of node type. The `ExecutionEngine` JIT-compiles to CPU machine code. GPU would only accelerate the PyTorch model (forward pass, PPO gradients).

### How to Request a GPU Node

```bash
# Interactive access to a C2 GPU node
srun --pty -n1 -q c2 -p nvidia --gres=gpu:a100:1 bash

# For a training job on GPU
sbatch --qos=c2 --partition=nvidia --gres=gpu:a100:1 --cpus-per-task=64 --mem=128G \
  scripts/train/train.sh config/ops_and_blocks/train/paper_transformer_small.json
```

---

## Limitations

| Limitation | Where | Impact |
|------------|-------|--------|
| **MLIR compilation/execution dominates** | `ppo.py` timings (`exec` >> `sampling`) | Each iteration spends most of its time compiling and running MLIR code, not doing RL. This is CPU work regardless of node type. |
| **Training runs on CPU-only Bergamo nodes** | `train.sh: --constraint=bergamo` | The entire pipeline — model inference, PPO gradients, MLIR compilation, code execution — runs on CPU. No GPU acceleration for the model. |
| **Only 12 CPU threads used** | `train.sh: --cpus-per-task=12` | Bergamo nodes have 256 cores. Only 12 are requested, limiting parallel benchmark execution. |
| **Process-isolated execution overhead** | `execution.py:263-277` | Each benchmark spawns a `multiprocessing.Process`, serializes the MLIR module to a string, forks, re-parses, re-creates MLIR context, compiles, and runs. Per-benchmark overhead is significant. |
| **ThreadPoolExecutor, not true distributed** | `dask_manager.py:122-130` | 12 threads on one node. MLIR C++ bindings and the GIL limit real parallelism. |
| **Small model on CPU** | `config`: `d_model=64`, `nhead=2`, `num_layers=2` | The Transformer is tiny, but on CPU the forward/backward for 64 benchmarks × multiple steps per benchmark still adds up. |
| **Cache miss cost** | `Execution.execute_code` | First-time transformations pay full compile+run cost. Only repeated transformation sequences hit the JSON cache. |
| **GPU unused even when available** | Code paths | The pipeline *can* run on GPU (code checks `cuda.is_available()`), but `train.sh` hardcodes bergamo constraint. Switching to C2/nvidia nodes would enable GPU for the model with zero code changes. |

---

## How to Accelerate Training

### 1. Increase CPU Parallelism (quickest win)

- `train.sh` currently requests `--cpus-per-task=12`. Bergamo nodes have 256 cores. Increase to `--cpus-per-task=64` or `128`.
- The `ThreadPoolExecutor` in `dask_manager.py` automatically uses `SLURM_CPUS_PER_TASK` as `max_workers`.
- More parallel benchmarks = proportionally faster execution phase.
- **Caveat**: memory. Each worker loads MLIR bindings and compiles independently. Increase `--mem` to `64G` or `128G`.

### 2. Run on GPU Nodes (C2 QOS / nvidia partition)

Bergamo is CPU-only, but you have access to C2 QOS GPU nodes. Two strategies:

**Strategy A — GPU for the model only (zero code changes):**

Change `train.sh` to submit to nvidia partition instead of bergamo:
```bash
#SBATCH --partition=nvidia
#SBATCH --qos=c2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Remove: #SBATCH --constraint=bergamo
```

The existing code already handles this: `device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")`. The model moves to GPU, PPO updates run on GPU, MLIR compilation uses the 64 CPU cores on the same node.

**Benefit**: The model forward pass and PPO gradient computation move to GPU. Even though the model is small (`d_model=64`), this frees CPU cores for MLIR compilation.

**Limitation**: The model is tiny. GPU inference for 64 observations is barely faster than CPU. The real bottleneck (MLIR compilation) is still CPU. Net speedup: modest, maybe 1.1-1.3x.

**Strategy B — GPU + CPU overlap (requires code changes):**

On a GPU node, model inference runs on GPU while MLIR compilation runs on CPU cores. They don't compete. You could pipeline:
- While CPU pool executes benchmark batch N, GPU samples actions for batch N+1.

This requires refactoring `collect_trajectory` into a producer-consumer pattern (see below). The benefit depends on the ratio of sampling time to execution time — since execution dominates, the speedup is limited to the sampling fraction.

**Strategy C — Kindi machine (8× A100, no Slurm):**

Kindi has 8 A100s and 128 CPU cores. It's a standalone machine (SSH access, no Slurm). You could run training directly:
```bash
ssh -p 4410 <netid>@kindi.abudhabi.nyu.edu
# Set up conda env, run training directly
python scripts/train/train.py  # with CUDA visible
```

**Limitation**: No Slurm means no job scheduling, no checkpoint/restart, no mail notifications. And the 8 GPUs are shared with other team members.

### 3. Persistent MLIR Worker Pool (highest impact for CPU execution)

**Current**: `__execute_states` spawns a new `multiprocessing.Process` per benchmark. Each process must:
- Create a new MLIR `Context`
- Parse the module from string
- Run `transform_bufferize_and_lower_v`
- Create a `PassManager`, parse the ~20-pass pipeline, run it
- Create an `ExecutionEngine`
- Invoke `main` twice
- Die

**Proposal**: Keep a pool of N long-lived worker processes (one per CPU thread). Each worker holds a persistent MLIR `Context` and receives (code_str, bench_name, seq, root_exec_time) via a queue.

**What can be reused across benchmarks**:
- The MLIR `Context` object (creation is expensive)
- The `PassManager` pipeline string (identical for every benchmark — only the module differs)

**What cannot be reused**:
- The `Module` itself (each benchmark has different code)
- The `ExecutionEngine` instance (tied to a specific module)

**Effect on learning**: None. This is purely an execution-side optimization. The agent sees the same (state, action, reward) tuples — we're just computing the rewards faster. The RL algorithm, reward function, and action space are untouched.

**Effort**: Medium. Requires rewriting `__execute_bufferized_code_isolated` to use a persistent pool. Workers must handle SIGABRT gracefully (run each benchmark in a sub-process or catch the signal internally).

### 4. Overlap Model Inference with Execution (GPU nodes only)

**Current**: `collect_trajectory` is sequential: sample all actions → then execute all benchmarks → then PPO update.

**Proposal**: Split `bench_count` into sub-batches. While the CPU pool executes batch N, the model samples actions for batch N+1.

**Prerequisite**: This only helps if the model inference runs on a different accelerator than MLIR compilation:
- On **Bergamo (CPU-only)**: model inference and MLIR compilation compete for the same CPU cores → pipelining gives little benefit.
- On **C2/nvidia (GPU+CPU)**: model inference on GPU, MLIR on CPU → they don't compete → pipelining keeps both busy.

**Expected benefit**: Limited. The model is tiny, so sampling is fast (seconds) while execution dominates (minutes). Pipelining saves at most the sampling fraction of each iteration.

### 5. Increase `bench_count`

- Currently 64. Increasing to 128 or 256 means more diverse gradient signal per iteration.
- But also longer execution phase per iteration (more MLIR compilations).
- This is an **algorithmic** change — it changes the RL training dynamics. Test carefully.

### 6. Multiple Seed Training in Parallel

- On Bergamo (256 cores), run multiple training instances with different seeds:
  - `sbatch --array=0-3 --cpus-per-task=32 --mem=64G` → 4 trainers × 32 cores = 128 cores.
  - Doesn't speed up a single run, but gives 4× data points for variance analysis.

---

## What NOT to Change (Academic Constraints)

| Item | Reason |
|------|--------|
| `ppo_batch_size=64` | Standard PPO batch size for academic reproducibility. |
| `opt_level=3` | Changes the reward signal (execution times). Must stay consistent. |
| `evaluate_benchmarks` during training | Already not called by `scripts/train/train.py`. Eval is separate. |
| Reward function (`-20.0` penalty, speedup ratio) | Core to the RL formulation. |
| Action space (tiling, interchange, vectorization, etc.) | Core to the problem formulation. |

---

## Recommended Quick Wins

| Priority | Change | Effort | Expected Speedup |
|----------|--------|--------|------------------|
| 1 | Increase `--cpus-per-task` from 12 to 64-128 in `train.sh` (+ `--mem`) | 2 lines | 3-5x on execution phase |
| 2 | Persistent MLIR worker pool | Medium (`execution.py`) | 2-3x on execution phase (amortizes context creation) |
| 3 | Run on C2 GPU node: `--partition=nvidia --qos=c2 --gres=gpu:a100:1` (remove bergamo constraint) | 4 lines in `train.sh` | 1.1-1.3x (model on GPU, frees CPU for compilation) |
| 4 | Pipeline sampling + execution (requires GPU node) | Medium-High (`ppo.py`) | Marginal (sampling is small fraction of iteration time) |

**Bottom line**: The dominant cost is MLIR compilation/execution, which is always CPU. The biggest win is **#1** (more CPU threads — Bergamo has 256 cores, you're using 12). Adding **#2** (persistent workers) compounds this by eliminating per-benchmark process spawn overhead. Running on GPU nodes (#3) helps the model but not the bottleneck.