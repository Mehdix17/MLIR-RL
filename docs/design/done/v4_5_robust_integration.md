# V4.5: Robust Integration (Hardened Reliability & Safety) — Design

**Status**: complete (historical — the reliability engineering survives in V5's execution.py)
**Date**: 2026-05-17
**Novelty scope**: Stability, Reliability, and Safety (Success-Contingent RL)
**Package**: `rl_autoschedular_v4_5`
**Config selector**: `"implementation": "rl_autoschedular_v4_5"`
**VERSIONS.md**: [V4.5 entry](../VERSIONS.md)
**Survives in V5**: ✅ yes — process isolation, dynamic timeouts, success-contingent reward negation, and the multi-engine fallback carry into V5's `execution.py` / `env.py` via the paper_transformer base. The shaped-reward aspect is abandoned; the reliability pillars are not.

> **Note**: this file also contains the former `v4_5_reliability_logic.md` deep-dive (merged 2026-08-06) — see §11.

## 1. Overview

Version 4.5 represents the "Hardened" evolution of the integrated V4 agent. While V4 successfully combined Hardware-Awareness, Shaped Rewards, and Transformer Encoding, it suffered from a high failure rate (~50%) due to aggressive incentives and native MLIR instability. 

V4.5 introduces **Defensive Reinforcement Learning**—a suite of architectural and algorithmic safeguards designed to achieve a near-zero failure rate without sacrificing the performance gains of the integrated model.

## 2. Problem Statement: The V4 "Reliability Gap"

In large-scale evaluations of V4, several critical issues were identified:
1.  **Native Crashes:** Certain transformation sequences (especially in ResNet/ViT models) triggered SIGABRTs in the LLVM/MLIR C++ bindings, killing the entire training/evaluation process.
2.  **Risk-Incentive Misalignment:** The **Shaped Rewards** (V2) provided "partial credit" for transforms that improved arithmetic intensity even if the final binary crashed. This encouraged the agent to gamble on unstable schedules.
3.  **Strict Timeouts:** Static 30s timeouts penalized large kernels that required more time for JIT compilation and profiling, marking valid optimizations as failures.
4.  **Greedy Complexity:** The agent often attempted 5+ nested transformations, leading to code so complex it triggered internal compiler assertions.

## 3. Solution: Hardened Robust Integration

V4.5 fixes these issues through four primary pillars of reliability:

### 1. Process Isolation (The "Sandbox")
All high-risk operations—Transform Dialect interpretation, JIT compilation, and binary execution—are wrapped in isolated `multiprocessing.Process` workers.

**Improvement:** If the MLIR bindings crash with a SIGABRT or Segfault, only the "Sandbox" process dies. The main RL loop catches the exit code and continues training, treating the crash as a standard execution failure (-20.0 penalty).

```python
# rl_autoschedular_v4_5/execution.py
def __run_isolated_exec(func, *args):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=lambda: queue.put(func(*args)))
    process.start()
    process.join(timeout=dynamic_timeout)
    if process.is_alive():
        process.terminate()
        return None, "Timeout"
    return queue.get() # Result from sandbox
```

### 2. Success-Contingent Reward Negation
To eliminate the "Gambler's Incentive," V4.5 treats all intermediate rewards as **provisional**.

**Improvement:** If the final execution fails (Timeouts, Crashes, or OOM), the environment **zeroes out all rewards** earned during that episode. The agent receives only the final failure penalty. This forces the policy to prioritize "runnability" as a hard constraint.

```python
# rl_autoschedular_v4_5/env.py
final_reward = self.__action_reward(True, exec_succeeded, ...)

if not exec_succeeded:
    # Negate ALL intermediate shaped rewards
    rewards = [0.0] * len(rewards)
    
# Add the final penalty to the end
rewards.append(final_reward)
```

### 3. Stability Rails (Action Masking)
V4.5 proactively prunes the action space to avoid sequences that are statistically correlated with compiler instability.

**Key Constraints:**
-   **Sequence Boundary:** The agent is restricted to terminal actions (`NT` or `V`) as it reaches the end of the predefined `order` sequence or approaches the environment's `truncate` limit. This prevents runaway transformation complexity while respecting the intended scheduling strategy.
-   **Depth Limit:** `Vectorization` is masked out if the loop-nest depth is > 6, avoiding a known diagnostic buffer assertion in the bindings.

```python
# rl_autoschedular_v4_5/actions/__init__.py
if not state.terminal:
    # Force termination based on config boundaries
    if state.step_count >= len(cfg.order) or state.step_count >= cfg.truncate - 1:
        allow_action(NoTransformation)
        
    # Forbid vectorization in deep nests
    if len(state.operation_features.nested_loops) > 6:
        forbid_action(Vectorization)
```

### 4. Multi-Engine Fallback
If the high-performance Python bindings fail to execute the code, the engine automatically falls back to the stable `mlir-cpu-runner` command-line utility.

**Improvement:** This distinguishes between "Binding Bugs" and "Optimization Bugs," ensuring the agent is only penalized if the MLIR code is truly unrunnable.

## 4. Comparison: V4 vs. V4.5

| Feature | V4 (Integrated) | V4.5 (Robust) |
| :--- | :--- | :--- |
| **Execution** | In-process (Risky) | Isolated Subprocess (Safe) |
| **Failures** | 50.1% | **Targeting < 1%** |
| **Incentive** | Partial credit for failures | All-or-nothing (Success-contingent) |
| **Timeout** | Static 30s | **Dynamic (10x Baseline)** |
| **Recovery** | Signal Handler (Partial) | Process Re-spawn (Full) |
| **Complexity** | Unlimited | Respects Config Boundaries |

## 5. Configuration

V4.5 is the default implementation for the `experiment3` series. It uses the `rl_autoschedular_v4_5` package.

```json
{
  "implementation": "rl_autoschedular_v4_5",
  "results_dir": "results/experiment3",
  "reward_shaping_enabled": true,
  "transformer_num_layers": 3
}
```

## 6. Expected Impact

1.  **Zero-Crash Training:** Training jobs should never FAILED on Slurm due to MLIR assertions.
2.  **Stable Evaluation:** The evaluation CDF (Cumulative Distribution Function) will no longer show a "cliff" at 1.0x speedup caused by silent failures.
3.  **Higher Peak Performance:** By allowing up to 300s for profiling, the agent can discover highly complex, multi-tiled optimizations that were previously killed by short timeouts.

## 7. References

-   `docs/V4_FAILURE_INVESTIGATION.md`: The root-cause analysis that led to V4.5.
-   `rl_autoschedular_v4_5/execution.py`: Implementation of isolated profiling.
-   `rl_autoschedular_v4_5/env.py`: Implementation of reward negation logic.

---

## 8. Deep-Dive: Reliability Engineering Insights (merged from `v4_5_reliability_logic.md`)

This section summarizes the technical rationale and architectural insights behind the "Hardening" features in V4.5, specifically addressing the trade-off between pure RL exploration and compiler stability.

### 11.1 Dynamic Timeouts (The "10x Tailored Deadline")

**Insight:** Static timeouts (e.g., 30s) are inherently unfair to large, complex benchmarks.

*   **Source:** The baseline execution time (`root_exec_time`) is taken from the **V0 (Unoptimized)** results.
*   **The 10x Margin:** We allow the optimized code to run up to 10x slower than the unoptimized baseline (capped at 300s).
*   **Why allow "Slow" code?**
    1.  **Stronger Gradients:** In RL, knowing that an action made the code "3x slower" provides a much richer mathematical signal to PPO than simply "Timed Out." It turns a binary failure into a measurable penalty.
    2.  **JIT Overhead:** Optimizing code (tiling, fusion) increases the complexity of the MLIR source. This requires the LLVM backend to work harder during Just-In-Time (JIT) compilation. A static timeout often kills a valid optimization simply because the compiler was taking its time.

### 11.2 Dynamic Sequence Boundaries (Respecting the "Order")

**Insight:** Runaway complexity is a primary trigger for native compiler crashes. Instead of hardcoded Python limits, V4.5 treats the **Config as the Source of Truth**.

*   **Rationale:** The original `ActionSpace` already enforces the length of the `order` list. If an agent reaches the end of the predefined strategy, the system throws an exception unless it stops.
*   **Safety Buffer:** The environment's `truncate` setting acts as an absolute hard wall. If the agent somehow exceeds the `order` length, `truncate` forcefully ends the episode.
*   **Implementation:** V4.5 keeps the `ActionSpace` logic clean and baseline-compatible, relying on these existing configuration-driven boundaries rather than adding redundant hardcoded masking for sequence length.

### 11.3 Physical Boundary Safeguards (Depth Limit)

**Insight:** You cannot "train" an agent to ignore a bug in the underlying C++ compiler if that bug causes a catastrophic crash.

*   **Environmental Context:** This bug is a **native C++ limitation** in the MLIR Python bindings (diagnostic buffer overflow). It exists in the shared environment and therefore **affects all versions (V0 - V4.5).**
*   **Why V2 and V4 are most affected:** The introduction of **Shaped Rewards** encouraged "Greedy Exploration." The agent discovered that creating extremely deep loop nests maximized intermediate rewards, inadvertently driving itself into the "Crash Zone" (depth > 6).
*   **Safe Exploration:** In V4.5 and V2.5, we explicitly mask `Vectorization` if the loop nest is too deep. We aren't "cheating" for the agent; we are defining the **Physical Boundaries** of the environment. This keeps the agent focused on finding the best *runnable* schedule within the limits of the current compiler technology.

### 11.4 Summary of Safeguards

| Feature | Logic | Goal |
| :--- | :--- | :--- |
| **Dynamic Timeout** | 10x `root_exec_time` | Fairer feedback & JIT tolerance |
| **Sequence Rail** | Respects `order` & `truncate` | Stay within user-defined strategies |
| **Depth Limit** | Max 6 loops for Vectorization | Avoid native MLIR C++ assertions |
| **Negation** | Zero rewards on failure | Prioritize runnability over partial gains |
