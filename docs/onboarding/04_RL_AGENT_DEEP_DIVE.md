# Onboarding: How the RL Agent Works

> **Module 4**: A deep-dive exploration of the Reinforcement Learning agent — detailing its state representation, action space, reward functions, compilation safety rails, and the PPO training loop.

---

## 1. The MDP (Markov Decision Process) Formulation

Our autoscheduler optimizes loop nest schedules by framing compiling as a **Markov Decision Process (MDP)**:

- **State ($S$)**: The structural representation of the current MLIR program (loop bounds, tensor shapes, memory maps, hardware specifications, and action history).
- **Action ($A$)**: A compilation decision applied to a specific structured operation (e.g., tile size choice, fusion decision, vectorization flag).
- **Transition ($P(s' | s, a)$)**: Applying the chosen transform to the MLIR module and updating the program representation state.
- **Reward ($R$)**: Terminal speedup relative to the unoptimized baseline program. Intermediate shaped rewards can guide search trajectories (though disabled in V4.9).
- **Episode**: A single benchmark compilation session. The agent starts with the raw MLIR file, sequentially schedules all eligible structured operations, runs the compiled code, gets the execution time reward, and ends the episode.

---

## 2. State Representation & Feature Extraction

The agent observes the MLIR program through a concatenated feature vector, defined in `rl_autoschedular_v4_9/observation.py`.

### A. Core Operation Features
For each structured operation (such as `linalg.generic` or `linalg.matmul`), we extract:
1. **Operation Type (One-Hot)**: Matmul, batch matmul, convolution (nhwc/nchw), pooling, generic, etc.
2. **Loop Structure**: Parallel vs reduction dimensions, nesting depth, and loop iteration bounds.
3. **Memory Access Patterns**: Input and output tensor shapes (log-normalized), ranks, and indexing maps (affine access maps translated into loop-dimension-to-tensor-dimension matrices).
4. **Operation Complexity**: Counts of arithmetic operations (adds, multiplies, divisions, etc.) within the loop body.

### B. Hardware-Aware Features (V4.5+)
When hardware-aware observation is enabled, the agent detects the properties of the execution host at runtime (reading `/sys/devices/system/cpu/` and `/proc/cpuinfo`):
- **Caches**: L1, L2, L3 cache sizes (in KB, normalized).
- **Cores**: Count of physical and logical CPU cores (using `SLURM_CPUS_PER_TASK` on clusters).
- **SIMD**: SIMD register width in bits (e.g., 512 for AVX-512, 256 for AVX2, 128 for SSE/Neon).
- **Clock**: Processor base clock frequency.

These features are concatenated into the final state representation, forcing the policy to learn **architecture-specific scheduling rules** (e.g., choose smaller tile sizes on CPUs with smaller L2 caches).

### C. Representation Encoders: LSTM vs. Transformer

```
LSTM Encoder:
Concatenated (producer, consumer) features ──► Bidirectional LSTM ──► Fixed-size Embedding

Transformer Encoder (V4.5+):
[CLS] Token ──┐
Summary Token ┼──► Input Projections ──► Self-Attention ──► LayerNorm ──► CLS Pooling
Loop Tokens   ──┘
```

- **LSTM Embedding (`rl_autoschedular_v0`)**: Processes loop structures sequentially. It struggles to capture relationships in deep, nested loops or complex multi-operation blocks.
- **Transformer Embedding (`rl_autoschedular_v4_9`)**: Treats each loop dimension as a token in a sequence. A prepended `[CLS]` token acts as the global state summarizer. Using multi-head self-attention, the model can relate outermost loops (tiling) with innermost loops (vectorization) directly. Role, token-type, and depth embeddings maintain the nesting hierarchy.

---

## 3. Action Space & Action Masking

The agent steps through each operation in dependency order and selects a sequence of transformations.

### A. Supported Actions
1. **Interchange ($I$)**: Permutes loop order to align memory accesses with cache lines.
2. **Tiling ($T$)**: Selects block partition sizes for each loop dimension from a discrete set of powers of two (e.g., 8, 16, 32, 64, 128, 256).
3. **Tiled Parallelization ($TP$)**: Annotates outer tiled dimensions to execute concurrently using multi-threading.
4. **Tiled Fusion ($TPF$)**: Merges a producer loop nest into a consumer loop nest to keep data warm in caches.
5. **Vectorization ($V$)**: Directs the compiler to lower the innermost loop into SIMD vector instructions.
6. **No Transformation ($NT$)**: Terminal action indicating that the schedule for the current operation is finished.

### B. Action Masking & Stability Rails
Compiler transformations are prone to crashes if illegal parameters are chosen (e.g., swapping a reduction dimension with a parallel dimension in an unsafe way, or tiling to sizes larger than the loop bounds).

To prevent this, `ActionSpace.action_mask` evaluates the current `OperationState` and filters out illegal choices:
- Swapping non-permutative dimensions is masked.
- Tile sizes exceeding loop bounds are masked.
- **Stability Rail (depth limit)**: If loop nesting is too deep ($>6$), vectorization is explicitly masked to bypass known LLVM diagnostic stack leaks that cause native segment faults.

---

## 4. Environment Transitions & Trajectory Walkthrough

To clarify how the environment moves from state to state during an episode, let us look at a concrete **2-operation block example**:
- **Operation 0 (Producer)**: Matrix Multiplication (`linalg.matmul`)
- **Operation 1 (Consumer)**: Element-wise Bias Add (`linalg.generic`)

The environment traverses operations in **reverse topological order** (Consumer $\rightarrow$ Producer) to ensure that downstream consumers are scheduled before we decide how to fuse upstream producers.

```
                  [Episode Commences: Load Benchmark]
                                   │
                                   ▼
                   [State S_0: Observe Operation 1]
                        (Element-wise Bias Add)
                                   │
                     Agent chooses Action Sequence:
                     - Tiling (T) to factor 32
                     - Vectorization (V)
                     - Terminal action (NT)
                                   │
                                   ▼
             [Transition step() & get_next_op_state()]
           Updates features & sets focus to Operation 0
                                   │
                                   ▼
                   [State S_1: Observe Operation 0]
                        (Matrix Multiplication)
                                   │
                     Agent chooses Action Sequence:
                     - Loop Interchange (I)
                     - Tiling (T) to factor 16
                     - Tiled Fusion (TPF) with Op 1
                     - Terminal action (NT)
                                   │
                                   ▼
             [Transition step() & get_next_op_state()]
           No more ops remain -> Set Terminal = True
                                   │
                                   ▼
                    [Episode Ends: JIT Execution]
           Transforms MLIR ──► Compiles ──► Profiles Time
                                   │
                                   ▼
                      [Terminal Reward Returned]
```

At the end of the episode, the accumulated schedule of compilation transforms is built programmatically and sent to the JIT engine.

---

## 5. Execution Safety & Process Isolation

Optimizing compiler schedules at runtime introduces stability challenges. Invalid configurations can crash the compilation worker with a `SIGABRT` or hang the compiler in an infinite loop.

To prevent compiler crashes from killing the RL training process, our state-of-the-art versions (V4.5+) use a **Process-Isolated Execution Engine** (`rl_autoschedular_v4_9/execution.py`):

```
                       [Parent Python Process]
                                  │
                       Spawns isolated worker via
                        multiprocessing.Process
                                  │
                                  ▼
                      [Child Worker Subprocess]
                     Parses MLIR ──► Compiles ──► JIT Runs
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼ (crashes / aborts)                              ▼ (successful JIT run)
    Child Aborted                                     Returns Wall-Clock Time
         │                                                 │
         ▼                                                 ▼
Parent catches abort status                        Parent retrieves timing
Falls back to:                                             │
`mlir-cpu-runner` CLI tool                                 │
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                       Resumes Training Loop
```

- **SIGABRT Handler**: Re-registered immediately after MLIR context initialization in `train.py` to intercept native aborts and bubble them up as Python `RuntimeError` exceptions.
- **Dynamic Timeout**: Calculated per benchmark as `unoptimized_baseline_time * 5` (with a floor of 2s). Prevents pathological, extremely slow tile selections from hanging training runs.
- **CLI Fallback**: If the JIT runtime python bindings fail, the runner invokes `mlir-cpu-runner` via a subprocess command line before logging the benchmark execution as failed.

---

## 6. Reward Design & Mathematical Formulation

### A. Terminal Reward
If compilation and execution are successful, the terminal reward measures optimized speedup:
$$R_{\text{terminal}} = \log_{10}\left(\frac{\text{unoptimized\_baseline\_time}}{\text{optimized\_execution\_time}}\right)$$
*Example*: If a benchmark runs in $200\mu s$ unoptimized, and the agent schedules it to run in $50\mu s$ ($4\times$ speedup):
$$R_{\text{terminal}} = \log_{10}(4.0) \approx +0.602$$
If compilation fails, or execution crashes/times out:
$$R_{\text{terminal}} = -20.0 \text{ (flat penalty)}$$

### B. Shaped Rewards (Ablation Settings)
In older versions (V2 – V4.5), intermediate shaped rewards were computed at each step from static loop properties:
- **Static Efficiency Score**:
  $$S_{\text{efficiency}} = w_{\text{ai}} \cdot \log_{10}(1.0 + AI) + w_{\text{vec}} \cdot V_{\text{score}} + w_{\text{par}} \cdot P_{\text{ratio}}$$
  Where:
  * **Arithmetic Intensity ($AI$)**: Estimated FLOPS divided by estimated bytes moved:
    $$AI = \frac{\text{Loops\_Extent} \times \text{Op\_Count}}{\sum_{\text{accesses}} (\text{Access\_Dimensions} \times 4 \text{ bytes})}$$
  * **Parallel Loop Ratio ($P_{\text{ratio}}$)**: Parallel loops divided by loop nest depth.
  * **Vectorizable Score ($V_{\text{score}}$)**: Binary indicator ($1.0$ if vectorizable).
- **The Lesson of Shaped Rewards (V4.9)**: Dense shaped rewards caused **Entropy Collapse**. The agent learned to exploit the static estimators (e.g., getting vectorization bonuses) even if the final compiled code ran slower or crashed. 

> [!IMPORTANT]
> In `rl_autoschedular_v4_9` and the final Paper Packages, the shaped reward is hardcoded to return `0.0`. The policy relies strictly on terminal wall-clock execution speedup rewards.

---

## 7. Actor-Critic PPO Training Loop

We train the policy using **Proximal Policy Optimization (PPO)**. Policy and Value functions share the token embedding backbone:

```
                  ┌──► Policy Head (MLP) ──► Action Probabilities (Categorical)
State (Tokens) ──► Embedding Backbone 
                  └──► Value Head (MLP) ──► Estimated State Value (Scalar)
```

### The Training Step
1. **Sample Batch**: Pick a batch of 8 to 16 benchmarks using the complexity-based batch policy.
2. **Collect Trajectories**: For each benchmark, run the agent. Save actions, log probabilities, and state values.
3. **Run Kernels**: Execute the scheduled MLIR codes in isolated child processes and retrieve execution times.
4. **Generalized Advantage Estimation (GAE)**: Compute advantage targets ($A_t$) using $\gamma = 0.99$ and $\lambda = 0.95$.
5. **Optimize (PPO Update)**: Run 4 optimization epochs on the batch:
   - Compute PPO clipped ratio loss:
     $$L_{\text{policy}}(\theta) = -\mathbb{E} \left[ \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) \right]$$
   - Add Value MSE loss and Entropy bonus ($H$):
     $$L_{\text{total}} = L_{\text{policy}} + c_1 L_{\text{value}} - c_2 H$$
   - Update model weights using Adam (learning rate $= 3\times 10^{-4}$).

In the next module, we will detail the evolution of different agent versions in the project.
