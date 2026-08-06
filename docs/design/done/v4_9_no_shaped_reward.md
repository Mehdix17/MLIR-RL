# V4.9: No Shaped Reward (Entropy Collapse Fix) — Design

**Status**: complete — **historically pivotal**: validates both V5 abandonment decisions (shaped reward, and later HW features)
**Date**: 2026-06-10
**Novelty scope**: Entropy-collapse fix by removing shaped reward
**Package**: `rl_autoschedular_v4_9`
**Config selector**: `"implementation": "rl_autoschedular_v4_9"`
**VERSIONS.md**: [V4.9 entry](../VERSIONS.md)
**Survives in V5**: ⚠️ indirectly — V5 inherits the *lesson* (no shaped reward; HW features explored here and abandoned), the Transformer encoder, and all V4.5 reliability pillars. The package itself is not the V5 base (`paper_transformer` is).

## 1. Overview

V4.9 is the **entropy-collapse fix**: it disables the shaped reward entirely by
hardcoding `__shaped_reward()` to return `0.0`, while keeping the V4.5
reliability engineering intact. It is a full standalone copy of
`rl_autoschedular_v4_5` (Transformer + HW features + V4.5 hardening) with only
the reward shaping neutralized. This is the last version where hardware-aware
observation is active — the user explored it here and found it unhelpful,
leading to its abandonment in V5.

## 2. Problem Statement: Entropy Collapse

During `single_ops_dataset` experiments, **V4.6 / V4.7 / V4.8 all suffered entropy
collapse mid-training**:

- Entropy dropped to **zero** (from a healthy 1.0-3.0), freezing the policy into
  producing identical actions forever.
- **Root cause**: shaped reward + Transformer encoder → the policy converges to a
  deterministic local optimum that maximizes the static shaped-reward heuristics
  (arithmetic intensity, vectorizability) instead of actual execution speedup.
- V0 (no shaped reward, LSTM) survived with healthy entropy through 14K+
  iterations — the contrast implicated the shaped reward, not the Transformer.

## 3. Solution: Remove the Shaped Reward

- `rl_autoschedular_v4_9/env.py`: `__shaped_reward()` **hardcoded to return
  0.0** (shaped reward disabled).
- All helper methods (`__static_efficiency_score`,
  `__estimate_arithmetic_intensity`, etc.) **kept as dead code** (not deleted)
  for reference.
- `reward_shaping_enabled: false` in both train and eval configs.
- Everything else — Transformer encoder, HW features, process isolation,
  dynamic timeouts, stability rails, multi-engine fallback — unchanged from
  V4.5.

## 4. Implementation

- `rl_autoschedular_v4_9/*`: full standalone copy of `rl_autoschedular_v4_5`
  with internal imports redirected to `rl_autoschedular_v4_9`.
- **Config variants** (config-driven, one package):

| Variant | Transformer | Config (train) | Config (eval) |
|---------|-------------|----------------|---------------|
| **V4.9 small** | d=64, 2 heads, 2 layers, ffn=128 (V4.7 arch) | `config/single_ops_dataset/train/v4_9_small.json` | `config/single_ops_dataset/eval/v4_9_small_eval.json` |
| **V4.9 large** | d=256, 8 heads, 3 layers, ffn=1024 (V4.8 arch) | `config/single_ops_dataset/train/v4_9_large.json` | `config/single_ops_dataset/eval/v4_9_large_eval.json` |

## 5. Configuration

```json
{
  "implementation": "rl_autoschedular_v4_9",
  "reward_shaping_enabled": false,
  "entropy_coef": 0.01,
  "bench_count": 64,
  "ppo_batch_size": 32,
  "nb_iterations": 10000
}
```

## 6. Results / Validation (measured 2026-08-06 from `results/`)

| Dataset | Variant | Benches | Median | p90 | Max |
|---------|---------|---------|--------|-----|-----|
| single_ops | small | 1,224 | 1.16x | 4.72x | 45.5x |
| single_ops | large | 1,224 | 1.68x | 7.10x | 47.8x |
| ops_and_blocks | small | 6,553 | 1.21x | 9.88x | 124,209x* |
| ops_and_blocks | large | 6,553 | 1.48x | 10.96x | 105,107x* |

\* The ops_and_blocks **means (~87x) and maxes are pathological outliers** —
tiny/degenerate benchmarks whose baseline is near-zero make speedup explode. The
**median is the honest statistic** here: 1.16-1.68x.

Validation performed (2026-06-10):
- All 15 Python files compile (`python -m py_compile`).
- All 4 config files are valid JSON.
- No `v4_5` references remain in the V4.9 package.
- `reward_shaping_enabled: false` in both train and eval configs.

## 7. Comparison vs Previous

| Feature | V4.6/4.7/4.8 | V4.9 |
| :--- | :--- | :--- |
| **Shaped reward** | Yes (fixed scale 0.05) | **None** (hardcoded 0.0) |
| **Entropy** | Collapses to 0 mid-training | Stays healthy (> 0.1 expected) |
| **HW features** | Yes | Yes (last version to have them) |
| **Transformer** | Yes | Yes (unchanged) |
| **V4.5 reliability** | Yes | Yes (unchanged) |

## 8. How to Use

```bash
sbatch scripts/train/train.sh config/single_ops_dataset/train/v4_9_small.json
sbatch scripts/eval/eval.sh config/single_ops_dataset/eval/v4_9_small_eval.json --checkpoint N
```

## 9. What is Unchanged

- Transformer encoder, HW features, action space, PPO algorithm, reward
  function for the *terminal* reward (`-20.0` penalty, `log10` speedup),
  `opt_level=3`.
- All V4.5 reliability features (process isolation, dynamic timeouts, stability
  rails, mlir-cpu-runner fallback).

## 10. Limitations & Lessons Learned

- Shaped-reward code kept as dead code (not deleted) for reference — a
  deliberate archaeology choice.
- **Lesson 1 (shaped reward)**: shaped reward + Transformer → entropy collapse.
  Removing it restores healthy exploration. **V5 drops shaped reward entirely**
  (the `reward_shaping_*` config fields are gone).
- **Lesson 2 (HW features)**: explored in this version and found **unhelpful** —
  the user observed no benefit from hardware-aware observation in V4.9. **V5
  drops HW features** (the `hardware_*` config fields are gone).
- **Lesson 3 (metrics)**: always report **median/p90**, not mean — the
  ops_and_blocks mean (~87x) is meaningless next to the median (~1.2-1.5x)
  because degenerate benchmarks inflate the tail.
- **V5 base note**: V5 is based on `paper_transformer` (which itself = V4.9
  minus HW features minus shaped reward), not on V4.9 directly — V4.9's
  checkpoint layout and observation size differ (HW features present).
