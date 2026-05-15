# Thesis Proposal: Advanced RL-based Automatic Code Optimization in MLIR

## 1. Baseline Architecture Overview

The current system utilizes a structured approach to optimize `Linalg` and `Affine` operations in MLIR:

- **State Representation:** Flat vectors representing loop bounds and access matrices, processed via an LSTM-based encoder.
- **Agent:** PPO (Proximal Policy Optimization) making discrete decisions on a per-operation basis.
- **Action Space:** Tiling, Parallelization, Fusion, Interchange, and Vectorization.
- **Traversal:** Reverse traversal (Consumer $\rightarrow$ Producer) of the computation graph.

---

## 2. Proposed Novelties

### Novelty 1: Hardware-Aware Observation Space

**The Problem:** The current agent is "blind" to the hardware it is optimizing for. A policy trained on a 4-core laptop might be suboptimal for a 64-core server.
**The Solution:** Inject hardware specifications directly into the observation vector.

- **Features to add:** L1/L2/L3 cache sizes (in KB), number of physical vs. logical cores, SIMD vector width (e.g., 256 for AVX2, 512 for AVX-512), and clock speed.
- **Impact:** This enables **Cross-Hardware Portability**. The agent learns to adjust tile sizes and parallelization strategies based on available cache and cores.

### Novelty 2: Deep Loop Nest Parsing (Transformer-based Encoder)

**The Problem:** LSTMs are sequential and struggle with long-range dependencies in complex nested loops.
**The Solution:** Replace the recursive LSTM with a **Transformer Encoder**.

- **Mechanism:** Treat each loop level in a nest as a "token." Use Self-Attention to allow the model to weigh the importance of the outermost loop (for tiling) against the innermost loop (for vectorization) simultaneously.
- **Positional Encoding:** Use structural encoding to maintain the hierarchy of the loop nest (which loop is inside which).

### Novelty 3: Expanded Transformation Action Space (Future Work - V5)

**The Problem:** The current set of actions is limited to high-level loop transformations (Tiling, Parallelization, Fusion, Interchange, Vectorization). The agent cannot express schedules that combine high-level tiling with low-level memory layout changes or instruction-level parallelism optimizations.
**The Solution:** Implement finer-grained transformations available in the MLIR Transform Dialect. This is planned for future work (e.g., **`rl_autoschedular_v5`**):

- **Pad (`P`):** Pads operation dimensions to multiples of powers of 2, ensuring aligned memory accesses and efficient vectorization.
- **Pack (`PK`):** Reorganizes data into blocked/tiled layouts using `transform.structured.pack`, improving cache locality for tiled access patterns.
- **Unroll (`U`):** Tiles loops and then unrolls them with `transform.loop.unroll`, exposing instruction-level parallelism and reducing loop overhead.

See [`docs/Novelties/v5_action_space_expansion.md`](Novelties/v5_action_space_expansion.md) for full planned implementation details.

### Novelty 4: Guided Search Strategy (Beam Search / MCTS) (Future Work - V6)

**The Problem:** The RL agent currently suffers from "Greedy Brittleness" — it makes a single "greedy" choice per step. If one choice is wrong, the entire schedule fails because it cannot backtrack.
**The Solution:** Implement a search layer during the inference/evaluation phase. _Note: This is currently not yet implemented and is planned for future work (e.g., `rl_autoschedular_v6`)._

- **Beam Search:** Instead of picking the top action, keep the $K$ most likely transformation sequences and evaluate them all.
- **Monte Carlo Tree Search (MCTS):** Use the RL policy as a "prior" to guide an MCTS exploration. This helps the agent find deep optimization sequences that a standard greedy policy would miss.

### Novelty 5: Multi-Objective and Shaped Rewards (Future Work)

**The Problem:** Speedup is a "sparse" reward (only known at the end of execution) and doesn't account for other factors like energy or memory.
**The Solution:** _ **Reward Shaping:** Add intermediate rewards based on static analysis:
_ _Arithmetic Intensity:_ Ops/Byte ratio. \* _Vectorization Ratio:_ Percentage of loops successfully vectorized.

- **Multi-Objective:** Create a weighted reward: $R = w_1(Speedup) + w_2(PeakMemoryUsage)$. This is critical for edge computing where memory is constrained.

### Novelty 6: Meta-Learning for Fast Adaptation (MAML) (Future Work)

**The Problem:** Training an RL agent from scratch for every new dialect or hardware is expensive.
**The Solution:** Use **Model-Agnostic Meta-Learning (MAML)**.

- **Goal:** Train the agent on a wide variety of "tasks" (different kernels like MatMul, Convolutions, Softmax).
- **Outcome:** The agent develops a "meta-policy" that can adapt to a completely new, unseen kernel with only 5–10 optimization steps.

---

## 3. Implementation Roadmap

This roadmap is divided into four phases, moving from environment enhancements to architectural shifts.

### Phase 1: Environment & Observation (Weeks 1-4)

- **Task 1.1:** Update `observation.py` to include the Hardware Feature vector (Cores, Cache, SIMD).
- **Task 1.2:** Modify the reward function in `env.py` to include "Shaped Rewards" (Arithmetic intensity).
- **Task 1.3:** Conduct a baseline test: See if the agent learns to pick different tile sizes for two different (simulated) cache sizes.

### Phase 2: Action Space Expansion (Planned for V5)

- **Task 2.1:** Integrate **Pad**, **Pack**, and **Unroll** into new actions definitions and transforms.
- **Task 2.2:** Update `action_mask` logic for all three new actions (Pad/Pack per-dimension validity, Unroll divisibility checks and terminal restrictions).
- **Task 2.3:** Benchmark performance on the existing dataset to measure speedup from expanded schedule space.

### Phase 3: Architectural Upgrade (Weeks 9-14)

- **Task 3.1:** Replace the LSTM in `model.py` with a **Transformer Block**.
- **Task 3.2:** Implement the tokenization logic for loop nests (converting loop features into sequence tokens).
- **Task 3.3:** Retrain the model on the full dataset. This will be the most compute-intensive phase.

### Phase 4: Advanced Search & Meta-Learning (Future Work - V6)

- **Task 4.1:** Implement **Beam Search** in the `evaluation.py` script. Compare "Greedy RL" vs "Beam Search RL." _(Planned for V6)_
- **Task 4.2:** (Optional/Advanced) Implement the MAML outer loop in `ppo.py` to enable fast adaptation to new kernels.
- **Task 4.3:** Final evaluation: Compare your final system against the previous student's baseline and standard MLIR `O3` optimizations.

---

## 4. Summary of Expected Contributions

1.  **A more robust encoder:** Moving from LSTMs to Transformers for code structural analysis.
2.  **Hardware-aware policies:** A single model capable of optimizing for different CPUs.
3.  **Efficiency gains:** Through future Pad, Pack, and Unroll actions (V5) not present in the baseline, enabling finer-grained control over memory layout and instruction-level parallelism.
4.  **Better Search (Planned for V6):** Using Beam Search to find higher-performing schedules than greedy inference (addressing greedy brittleness).
