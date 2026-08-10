# Design Docs — Completed Versions (V0 → V4.5)

This folder documents the **completed** MLIR-RL versions, one file per version,
in a unified structure so the evolution can be read linearly. Every file follows
the same template:

```
Metadata block (Status / Date / Novelty scope / Package / Config selector / VERSIONS.md link / Survives in V5?)
## 1. Overview
## 2. Problem Statement
## 3. Solution
## 4. Implementation
## 5. Configuration
## 6. Results / Validation
## 7. Comparison vs Previous
## 8. How to Use
## 9. What is Unchanged
## 10. Limitations & Lessons Learned
```

## Reading order

Read top to bottom — each version builds on the previous:

| # | Doc | Version | What it adds | Survives in V5? |
|---|-----|---------|--------------|-----------------|
| 1 | `v0_original_baseline.md` | V0 (baseline) | The original `rl_autoschedular` package — state, obs, actions, model, reward, env, PPO | Foundation (all versions fork from it) |
| 2 | `v0_model_detailed.md` | V0 appendix | Deep dive into `HiearchyModel` (LSTM embedding, policy/value heads, hierarchical action index) | Reference only (V5 uses Transformer) |
| 3 | `v1_hardware_aware_observation.md` | V1 | Hardware features in the observation | ❌ **Abandoned** (explored in v4_9, unhelpful) |
| 4 | `v2_shaped_reward.md` | V2 | Dense shaped reward | ❌ **Abandoned** (entropy collapse in v4.9) |
| 5 | `v2_5_hardened_shaped_reward.md` | V2.5 | V4.5 hardening ported back to V2 (fair baseline) | ❌ Historical baseline artifact |
| 6 | `v3_transformer_loop_nest_encoder.md` | V3 | Transformer encoder replacing LSTM | ✅ **Core contribution — carried into V5** |
| 7 | `v4_combined_model.md` | V4 | V1 + V2 + V3 integrated | ⚠️ Parts (V3 only; V1+V2 abandoned) |
| 8 | `v4_5_robust_integration.md` | V4.5 | Reliability engineering (process isolation, reward negation, stability rails, fallback) | ✅ Reliability features survive in V5's `execution.py` |
| 9 | `v4_9_no_shaped_reward.md` | V4.9 | Entropy-collapse fix — shaped reward disabled; last version with HW features | ⚠️ Lessons (no shaped reward, HW abandoned) carry into V5 |

## Version map

- V4.6 / V4.7 / V4.8 are **config variants** of `rl_autoschedular_v4_5` (reward-fixed runs) — documented in `../VERSIONS.md`, no separate design doc.
- V4.9 (no shaped reward / entropy fix) is documented here; it is historically pivotal because it validated both V5 abandonment decisions (shaped reward, HW features).
- Paper, Paper Transformer are documented in `../VERSIONS.md`.
- The **V5 generation** (V5 platform, V5.1 full-model eval, V5.2 action space) lives in `../todo/`.
- Full history with validation details: [`../VERSIONS.md`](../VERSIONS.md).

## Key conventions

- **Package isolation**: each `rl_autoschedular_vN` is a fully standalone package (no cross-package imports). V5 follows the `paper_transformer` structure (own `utils/`).
- **Config singleton**: `utils/config.Config` reads `CONFIG_FILE_PATH` at first import.
- **One novelty per version** (V0–V4.5 rule); V4.6-4.8 are config variants, V4.9 is the entropy fix, and V5 is the platform version that V5.1/V5.2 extend.
