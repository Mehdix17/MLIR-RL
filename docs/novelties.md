# RL Agent Architecture — Proposed Novelties

**Date:** 2026-03-06  
**Context:** Post-option-2 roadmap. Assumes the 4 benchmark generation blockers are fixed
(static shapes, `@main` entry point, no elided weights, single-op granularity) and the agent
is being trained on a rich single-operation dataset (`data/nn/other/` or equivalent).

---

## Current Architecture Reference

Before describing what to add, here is what already exists:

| Component | Current implementation |
|---|---|
| **Observation** | Concatenated flat vector: `[op_type \| loop_bounds \| iterator_types \| load_access_matrices \| store_access_matrices \| arith_op_counts \| producer_features \| action_history \| action_mask]` |
| **Encoder** | `OpFeatures` MLP (Linear→ELU×2) → LSTM over `(consumer, producer)` pair → flat embedding (411-dim) + action history |
| **Policy backbone** | 3× Linear-ReLU shared trunk → separate head per action type |
| **Value network** | Same LSTM → 3× Linear-ReLU → scalar |
| **RL algorithm** | PPO with off-policy correction, ε-greedy + entropy exploration, experience replay |
| **Action space** | Discrete: `{Tiling, TiledParallelization, TiledFusion, Interchange, Vectorization, NoTransform}` |
| **Reward** | Empirical speedup: $r = t_{\text{original}} / t_{\text{transformed}}$, measured at episode end only |

**Key files:** `rl_autoschedular/model.py`, `rl_autoschedular/observation.py`,
`rl_autoschedular/actions/`, `rl_autoschedular/env.py`, `rl_autoschedular/ppo.py`

---

## Novelty 1 — Hardware-Aware Observation

### Motivation

The current observation has **no information about the target machine**. The agent learns one
fixed policy regardless of whether it targets a 16-core server with 32 MB L3 cache or a 4-core
laptop with 6 MB. A tile size of 256 may be optimal on one machine and cache-thrashing on
another. This means:
- The trained policy is **not portable** — it must be retrained for each new target machine.
- The agent cannot learn generalizable rules like *"tile to fit in L2 cache"* because the
  cache size is not in the observation.

### What to Add

Append a `HardwareFeatures` observation part to the existing `Observation.parts` list in
`observation.py`. It is a fixed vector collected once at startup via Linux system interfaces:

```python
class HardwareFeatures(ObservationPart):
    _cached: Optional[torch.Tensor] = None

    FEATURES = [
        'l1d_cache_kb',   # L1 data cache in KB
        'l2_cache_kb',    # L2 cache in KB
        'l3_cache_kb',    # L3 cache in KB
        'num_cores',      # Physical cores
        'simd_width',     # AVX2 = 256, AVX-512 = 512, SSE = 128
        'cache_line_b',   # Cache line size in bytes (usually 64)
    ]

    @classmethod
    def size(cls) -> int:
        return len(cls.FEATURES)

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        if cls._cached is None:
            cls._cached = cls._collect()
        return cls._cached

    @classmethod
    def _collect(cls) -> torch.Tensor:
        import subprocess, re
        lscpu = subprocess.check_output('lscpu', text=True)
        # parse and normalize values ...
```

**Data sources:**
- `/sys/devices/system/cpu/cpu0/cache/index*/size` — cache sizes per level
- `lscpu | grep "CPU(s)"` — core count
- `lscpu | grep Flags` — detect `avx512f`, `avx2`, `sse4` for SIMD width
- Normalize by dividing by known maximum to keep values in `[0, 1]`

All values are **log-normalized** (same as `normalize_bounds = 'log'` in the config) since
cache sizes span orders of magnitude (32 KB L1 vs 32 MB L3).

### Why It's Novel Here

Most existing learned schedulers (TVM AutoTVM, Ansor, Halide auto-scheduler) either learn
separate models per hardware target or require explicit hardware templates. Adding hardware
features to the observation lets **one model generalize across targets**, with reuse of the
learned policy as a strong prior even on unseen hardware configurations.

### Implementation Effort

**Low.** This is purely additive:
1. Add `HardwareFeatures` class to `observation.py` (~30 lines)
2. Append to `Observation.parts`
3. The `LSTMEmbedding.embedding_size` in `model.py` adjusts automatically based on
   `OpFeatures.size()` — but note `embedding_size = 411` is hardcoded. It needs to be set
   to `OpFeatures.size() + ProducerOpFeatures.size() + HardwareFeatures.size()` dynamically.

---

## Novelty 2 — Graph Neural Network Over the Computation DAG

### Motivation

The LSTM currently processes exactly **two nodes**: `(consumer, producer)`. This ignores:
- Operations with **multiple producers** (both inputs of a matmul can be outputs of convs)
- **Downstream consumers** — whether a conv feeds a pooling or a relu changes the
  cache-reuse pattern and optimal tile strategy
- The `producers: list[tuple[str, int]]` and `consumers: list[tuple[str, int]]` fields are
  already stored in `OperationFeatures` but only the first selected producer is used

### What to Add

Replace `LSTMEmbedding` with a **message-passing GNN** that operates on the local
computation subgraph centered on the current operation:

```
                      ┌─────────┐
                conv  │  node_0 │ ← encoded with OpFeatures vector
                      └────┬────┘
                           │ edge: data dependency
                      ┌────▼────┐
              current │  node_1 │ ← target node (the op being scheduled)
                      └────┬────┘
                           │
                      ┌────▼────┐
                relu  │  node_2 │
                      └─────────┘
```

**Architecture:**
1. Each node is initialized with its `OpFeatures` vector
2. K rounds of message-passing aggregate neighbor embeddings:
   - `h_v^{k+1} = MLP(h_v^k, mean(h_u^k for u in neighbors(v)))`
3. The target node's final embedding replaces the LSTM output
4. `ActionHistory` is appended as before

**Graph construction:** `BenchmarkFeatures.operations` already stores the full
producer/consumer graph. Building the PyTorch Geometric (PyG) `Data` object is straightforward
since all the edges are explicit.

### Why It's Novel Here

Most existing approaches to loop-nest scheduling (Tiramisu, Tensor Comprehensions, MLIR
Sandbox experiments) treat each operation as independent. Using the data-flow context means
the agent can learn **cross-op patterns** like:
- "When my output feeds a reduction, tile the reduction dimension small"
- "When my two inputs are both large convs, their tile choices interact via L2 cache sharing"

This is the primary structural novelty over the current LSTM pair approach.

### Implementation Effort

**High.** Requires:
1. Adding `torch_geometric` as a dependency
2. Rewriting `LSTMEmbedding` in `model.py` as a GNN encoder
3. Modifying `Observation.from_state()` to produce a graph instead of a flat tensor — or
   keeping the flat tensor for compatibility and building the graph separately
4. PPO batching changes: GNN models require graph-batched data (`Batch.from_data_list`)
   which is different from the current simple `torch.cat` over observations

---

## Novelty 3 — Tree Search at Inference (Beam Search / MCTS)

### Motivation

The trained policy is currently deployed **greedily** (`greedy=True` in `model.sample()`).
A single forward pass picks the action at each step. This is cheap but sub-optimal: the agent
has no way to backtrack from a bad early tiling choice before actually executing the code.

At **evaluation time**, spending extra compute to search over the action space is acceptable
(a few seconds of search is fine for an offline scheduler). The trained policy is a strong
prior that makes this search efficient.

### What to Add

**Beam Search:** Keep the B most probable action sequences at each step, expand each one,
prune back to B:

```python
def beam_search(env, model, initial_state, beam_width=5, depth=Config().truncate):
    beams = [(initial_state, [], 0.0)]  # (state, action_seq, cumulative_log_p)
    for _ in range(depth):
        candidates = []
        for state, seq, score in beams:
            if state.terminal:
                candidates.append((state, seq, score))
                continue
            distributions = model.policy_model(Observation.from_state(state))
            # Take top-beam_width actions by probability
            for action_index in top_k_actions(distributions, beam_width):
                action = ActionSpace.action_by_index(action_index, state)
                new_state = env.step(state, action)
                new_score = score + action_log_prob(distributions, action_index)
                candidates.append((new_state, seq + [action], new_score))
        beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

    # Execute all terminal beams, return the best speedup
    results = [env.apply_and_run_sequence(seq) for _, seq, _ in beams]
    return max(results, key=lambda r: r[1])  # best speedup
```

`Env.apply_and_run_sequence()` already exists and supports running a full action sequence —
it just needs to be called once per beam rather than once per episode.

**MCTS variant:** Use the value network as the rollout estimator to avoid executing all
leaves, falling back to execution only for the top-1 final sequence.

### Why It's Novel Here

The current pipeline has no search at inference at all. Even beam width=2 can improve
results significantly when early tiling decisions are ambiguous. The value network already
provides a state quality estimate — using it to prune the beam is a natural extension.

### Implementation Effort

**Medium.** Add a `search.py` module or extend `evaluation/evaluate.py`. No changes to the
model architecture. The main complexity is managing the state-copy cost (each beam maintains
its own `OperationState`).

---

## Novelty 4 — Meta-Learning for Fast Adaptation (MAML-style)

### Motivation

The current PPO training produces one fixed policy. When a new class of operations appears
(e.g., a new model family with unique conv strides or non-standard tensor layouts), the agent
must be fully retrained or will generalize poorly.

**Meta-learning** trains the *initialization* of the model such that a small number of
gradient steps on new data produces a well-adapted policy — without full retraining.

### What to Add

Implement **MAML (Model-Agnostic Meta-Learning)** around the existing PPO loop:

```
Meta-training outer loop:
  For each iteration:
    Sample K operation families as "tasks" (e.g., conv_2d, matmul, pooling)
    For each task:
      Clone the current model θ
      Run N inner PPO steps on this task's benchmarks → θ'_task
      Compute loss on a held-out set of the same task using θ'_task
    Outer update: optimize θ to minimize average loss across all tasks
```

**Partitioning strategy:** Use operation type (`OperationType.Conv`, `.Matmul`, etc.) and
shape classes (small/medium/large by loop bounds) as task definitions. This is a natural
partition given the existing `OperationFeatures.operation_type` enum.

At deployment on a new target:
1. Load the meta-trained checkpoint
2. Run 10–20 rollouts on the new benchmark type
3. Take K gradient steps (inner loop only)
4. Deploy the adapted policy

### Why It's Novel Here

The dominant approaches (TVM Ansor, MLIR Sandbox) either transfer-learn by fine-tuning a
large pretrained model or run cost-model-guided random search from scratch on new hardware.
MAML-based meta-RL would allow **principled fast adaptation** from a small number of actual
hardware measurements — important when execution time is expensive (e.g., on specialized
accelerators).

### Implementation Effort

**High.** Requires:
1. A second-order gradient computation through the PPO inner loop
2. Task partitioning logic for the benchmark dataset
3. A modified training loop in `ppo.py` with inner/outer update separation
4. Higher GPU memory due to unrolled computation graphs

Consider using `higher` (PyTorch meta-learning library) for the inner loop differentiation.

---

## Novelty 5 — Analytical Reward Shaping via Roofline Model

### Motivation

The current reward is **sparse**: the agent only receives a signal at the **end of the episode**
when the fully-transformed code is executed. This makes credit assignment hard — if the agent
applies 3 tiling actions and then 1 vectorization, it does not know which action drove the speedup.

The observation already captures everything needed to compute a **dense intermediate reward**
from a static cost model at each action step.

### What to Add

After each tiling action, compute the **arithmetic intensity** and **cache efficiency** of
the resulting loop nest analytically using the information already in `OperationFeatures`:

$$\text{arithmetic intensity} = \frac{\text{FLOPs per iteration}}{\text{bytes accessed per iteration}}$$

$$\text{FLOPs} = \prod_{i \in \text{parallel loops}} UB_i \cdot \sum_{\text{arith ops}} \text{count}$$

$$\text{bytes} = \sum_{\text{loads}} \prod_{d \in \text{accessed dims}} UB_d \cdot \text{element\_size}$$

A tile is **cache-efficient** when its working set fits in L2 cache. Define the shaped reward:

$$r_t^{\text{shaped}} = \alpha \cdot \Delta(\text{roofline efficiency}) + (1 - \alpha) \cdot r_T^{\text{empirical}}$$

where $r_T^{\text{empirical}}$ is the true speedup at episode end and $\alpha$ decays over
training (start with $\alpha=0.7$, anneal to $0.1$).

**All inputs are available at zero cost:**
- `nested_loops[i].upper_bound` — loop bounds
- `op_count` — arithmetic op counts (`+`, `-`, `*`, `/`, `exp`)
- `load_data`, `store_data` — access patterns for bytes-moved computation

The Roofline model maximum performance is:

$$P_{\max} = \min\!\left(\text{peak FLOP/s},\; \text{AI} \times \text{peak bandwidth}\right)$$

If tiling brings AI closer to the hardware ridge point, the action receives a positive shaped reward.

### Implementation Effort

**Medium.** The static analysis is pure arithmetic on existing fields:
1. Add `compute_roofline_efficiency(op_features, tile_sizes, hw_features)` utility function
2. Modify `Env.step()` to call it and blend the reward
3. Add `hw_features` parameter (links to Novelty 1)

This novelty **synergizes with Novelty 1** (hardware-aware observation): the hardware
features needed for the Roofline bound (peak bandwidth, SIMD throughput) are the same ones
added to the observation.

---

## Recommended Implementation Order

Given the trade-off between research impact and implementation cost:

```
Phase 1 (additive, no architecture change):
  ✦ Novelty 1 — Hardware-aware observation       (1–2 days)
  ✦ Novelty 5 — Analytical reward shaping        (3–4 days)
  → These two compound: reward shaping uses hardware info from novelty 1

Phase 2 (inference-time only, no training change):
  ✦ Novelty 3 — Beam search at inference         (2–3 days)
  → Immediately improves evaluation results on existing checkpoints

Phase 3 (architectural, requires retraining):
  ✦ Novelty 2 — GNN over computation graph       (1–2 weeks)
  → Most structurally novel, biggest potential gain

Phase 4 (training algorithm):
  ✦ Novelty 4 — MAML meta-learning               (2–3 weeks)
  → High-risk, high-reward; requires careful task partitioning
```

---

## Integration Map

```
observation.py          → add HardwareFeatures (Novelty 1)
                           modify Observation.parts to include it

model.py                → replace LSTMEmbedding with GNN (Novelty 2)

env.py                  → blend shaped reward in Env.step() (Novelty 5)

evaluation/evaluate.py  → wrap with beam search (Novelty 3)

ppo.py                  → add outer MAML loop (Novelty 4)
```

---

## Related Work

| Approach | Venue | Relation |
|---|---|---|
| Halide auto-scheduler | PLDI 2019 | Learned cost model, no hardware features in obs |
| TVM AutoTVM | OSDI 2018 | Black-box tuning, no RL policy |
| TVM Ansor | OSDI 2020 | Random program generation + cost model, no policy gradient |
| MLIR Sandbox | CGO 2022 | Manual transform schedule, no learning |
| Tiramisu RL | ArXiv 2021 | RL for polyhedral scheduling, single-machine assumption |
| MetaSchedule (TVM) | ArXiv 2022 | Closest prior work; uses cost model but no meta-learning |

The combination of **hardware-aware observation + analytical reward shaping + GNN encoder**
would be novel relative to all of the above.
