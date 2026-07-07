# MLIR-RL Training and Evaluation Pipeline Guide

This document explains how the Reinforcement Learning (RL) training and evaluation pipelines work in MLIR-RL, including execution flows, resource requirements, caching mechanisms, parallel fallbacks, and architecture constraints.

---

## 1. The RL Training Process

The training process uses a Reinforcement Learning agent (using Proximal Policy Optimization - PPO) to search for optimal compilation schedules (such as loop tiling, loop interchange, and vectorization) for MLIR operations. Training alternates between two sequential phases:

### Phase A: Simulation & Trajectory Collection (Rollout)
1. **Suggesting Schedules**: The agent’s policy network receives structural features of an MLIR loop nest and suggests scheduling actions.
2. **Timing Execution**: To calculate a speedup reward, the schedule must be compiled and timed:
   * **Cache Lookup**: The execution history is stored in `exec_data.json`. If a schedule was previously run, its execution time is looked up instantly (**0ms**).
   * **JIT Execution**: On a cache miss, the MLIR code is JIT-compiled into LLVM IR and executed on the CPU to measure runtime in nanoseconds (**100–300ms**).
3. **Parallel Compilations**: When resolving multiple cache misses in a batch, executions are run in parallel using a local `ThreadPoolExecutor` fallback scaled dynamically by Slurm's `SLURM_CPUS_PER_TASK`.

### Phase B: Neural Network Weight Updates
1. **PPO Backpropagation**: Observations, actions, and rewards collected during Phase A are fed into the policy and value networks to perform PyTorch backpropagation.
2. **Thread Capping**: PyTorch is explicitly restricted to **4 threads** (`torch.set_num_threads(4)`). Since the policy network is small, capping the threads prevents multi-threading context-switching overhead and maximizes CPU efficiency.

---

## 2. The RL Evaluation Process

Evaluation measures the agent's generalization performance on a separate validation set of MLIR benchmarks without modifying neural network weights.

### Architecture Consistency (Genoa vs. Bergamo)
* **Bergamo Pinned**: To ensure timing measurements are consistent, both training and evaluation jobs must run on the exact same CPU architecture. All evaluation scripts are configured to use:
  ```bash
  #SBATCH --constraint=bergamo
  ```
  Evaluating on different nodes (like Genoa or Milan) invalidates comparisons due to CPU clock and cache differences.

### Sequential Batch Checkpoint Scanning
Rather than launching dozens of concurrent Slurm tasks which spam the scheduler queue and trigger memory bottlenecks, evaluations are run in batches (e.g., checkpoints `100` to `1000` with step `100`).
* **Sequential Loop**: A single batch job executes the evaluation of checkpoints sequentially, but uses the parallel thread pool (`12 CPUs`) to evaluate individual benchmarks within each checkpoint rapidly.
* **Saving Results**: The evaluation times for each checkpoint are saved as a standalone JSON file under the run's evaluation folder:
  ```
  results/<experiment>/<agent_dir>/run_N/eval/checkpoint_<N>.json
  ```
  This is identical to the V4.9 evaluation format and contains execution statistics (bench_name $\rightarrow$ optimized_time_ns).

---

## 3. Slurm Resource Configurations

Training and evaluation co-exist stably under the following Slurm resource guidelines:

| Job Type | Recommended Resources | Constraint | Notes |
| :--- | :--- | :--- | :--- |
| **Training** | `--cpus-per-task=12 --mem=32G` | `bergamo` | 12 CPUs allow high concurrency during MLIR parallel compile rollouts; 32GB RAM prevents cgroup OOM failures. |
| **Evaluation** | `--cpus-per-task=12 --mem=32G` | `bergamo` | Evaluates individual checkpoints sequentially; uses parallel execution threads for fast benchmark compilation. |

---

## 4. Operational Commands

### Launching Training (from scratch or resuming)
```bash
# Start new training
sbatch scripts/train/train.sh config/ops_and_blocks/train/paper_original.json

# Resume training from checkpoint
sbatch scripts/train/train.sh config/ops_and_blocks/train/paper_original.json --resume results/ops_and_blocks_results/paper_original_agent/run_0
```

### Launching Batch Evaluation
```bash
# Evaluate checkpoints 100 to 1000 with step 100
sbatch scripts/eval/eval_batch.sh config/ops_and_blocks/eval/paper_original_eval.json 100 1000 100
```
