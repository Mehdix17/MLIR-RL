# V4: Combined Enhancements (V1 + V2 + V3) — Design

**Status**: complete (historical)
**Date**: 2026-05-10
**Novelty scope**: Integrated model combining Hardware-Aware Observation (V1) + Shaped Reward (V2) + Transformer Loop-Nest Encoder (V3)
**Package**: `rl_autoschedular_v4`
**Config selector**: `"implementation": "rl_autoschedular_v4"`
**VERSIONS.md**: [V4 entry](../VERSIONS.md)
**Survives in V5**: ⚠️ partially — the Transformer encoder (V3) survives; hardware features (V1) and shaped reward (V2) are abandoned. V4's ~50% failure rate led to V4.5's reliability engineering, which V5 inherits.

## 1. Overview

Version 4 represents the integration of all early-stage enhancements into a single, comprehensive RL agent. It combines the **Hardware-Aware Observation** (V1), **Shaped Reward** (V2), and the **Transformer Loop-Nest Encoder** (V3) to maximize scheduling performance, cross-hardware generalization, and training stability.

## 2. Problem Statement

V1, V2, and V3 each targeted a distinct orthogonal component of the RL pipeline — state observation, reward signal, and neural architecture — in isolation. V4's goal was to verify they compose into a single stronger agent.

## 3. Solution: Integrated Components

V4 brings together the following novelties:

1. **Hardware-Aware Observation (from V1)**: The agent receives explicit features about the target hardware (L1/L2/L3 cache sizes, physical/logical core counts, SIMD width, clock speed). This allows the schedule to adapt its tiling and parallelization optimally to different microarchitectures.
2. **Shaped Reward (from V2)**: Instead of relying solely on sparse, delayed execution time improvements, V4 uses intermediate reward shaping (based on heuristics like arithmetic intensity and vectorizability). This guides the agent during early training steps and accelerates convergence.
3. **Transformer Loop-Nest Encoder (from V3)**: The underlying MLIR loop structures are processed using an attention-based sequence encoder. This enables the agent to better capture nested dependencies and complex data-flow patterns compared to simple flattened MLP layers.

## 4. Implementation

- `rl_autoschedular_v4/*`: full standalone copy combining `rl_autoschedular_v1` (explicit hardware features), `rl_autoschedular_v2` (intermediate, dense shaped rewards driven by arithmetic intensity/vectorizability), and `rl_autoschedular_v3` (Transformer loop-nest architecture). Internal imports redirected to `rl_autoschedular_v4`.

## 5. Configuration

To use V4, ensure your JSON config contains:

```json
{
  "implementation": "rl_autoschedular_v4",
  "hardware_auto_detect": true,
  "reward_shaping_enabled": true,
  "reward_shaping_scale": 0.5
}
```

## 6. Results / Validation

- Proven to synergize hardware constraints with representation learning (VERSIONS.md).

## 7. Comparison vs Previous

| Feature | V3 (Transformer) | V4 (Combined) |
| :--- | :--- | :--- |
| **Encoder** | Transformer | Transformer |
| **Hardware-Aware** | No | Yes (from V1) |
| **Shaped Reward** | No | Yes (from V2) |

## 8. How to Use

```bash
sbatch scripts/train.sh config/v4.json
sbatch scripts/eval.sh config/v4.json
```

## 9. What is Unchanged

- PPO training algorithm, action space, environment dynamics, MLIR execution engine.

## 10. Limitations & Lessons Learned

- **~50% failure rate** due to aggressive incentives from shaped rewards: the agent learned to push the MLIR compiler into failing states (MLIR bindings crashing). This is the "V4 reliability gap" that motivated V4.5's hardening (see `v4_5_robust_integration.md`).
- Shaped reward + Transformer also seed the entropy-collapse problem later diagnosed in V4.6/4.7/4.8 and fixed in V4.9 by removing shaped reward entirely.
- **Historical verdict (2026-08-06)**: the V1 (hardware) and V2 (shaped reward) components were ultimately found unhelpful and are **abandoned in V5**; only the V3 Transformer contribution survives.
