# rl_autoschedular/ — RL Agent Implementations

This directory contains all RL agent implementations. Each subdirectory is a **fully standalone package** with no cross-imports.

## Package Overview

| Package | Encoder | HW Features | Shaped Reward | Status |
|---------|---------|-------------|---------------|--------|
| `rl_autoschedular_v0` | LSTM | ❌ | ❌ | Baseline |
| `rl_autoschedular_v1` | LSTM | ✅ | ❌ | HW-aware observations |
| `rl_autoschedular_v2` | LSTM | ✅ | ✅ | Shaped reward |
| `rl_autoschedular_v2_5` | LSTM | ✅ | ✅ | Hardened shaped reward |
| `rl_autoschedular_v3` | Transformer | ✅ | ✅ | Transformer encoder |
| `rl_autoschedular_v4` | Transformer | ✅ | ✅ | Combined enhancements |
| `rl_autoschedular_v4_5` | Transformer | ✅ | ✅ | Robust integration |
| `rl_autoschedular_v4_9` | Transformer | ✅ | ❌ | Entropy collapse fix |
| `rl_autoschedular_v5` | Transformer | ❌ | ❌ | V5 platform (CPU-only) |
| `rl_autoschedular_paper` | LSTM | ❌ | ❌ | Paper reproduction |
| `rl_autoschedular_paper_transformer` | Transformer | ❌ | ❌ | Paper transformer |

### Ablation Variants

| Package | Missing Feature |
|---------|-----------------|
| `rl_autoschedular_v45_no_hw` | No hardware features |
| `rl_autoschedular_v45_no_shaped_reward` | No shaped reward |
| `rl_autoschedular_v45_no_transformer` | LSTM instead of transformer |

## Package Structure

Each package contains:
```
rl_autoschedular_vX/
├── __init__.py
├── agent.py          # Main RL agent (actor-critic)
├── environment.py    # Gym-like environment
├── model.py          # Neural network architecture
├── config.py         # Package-specific config
└── ...               # Additional modules
```

## Key Design Decisions

### Package Isolation
Each version is **fully standalone** — no imports between packages. This ensures:
- Reproducibility of historical experiments
- No accidental behavior changes from shared code
- Easy comparison between versions

### Config Singleton
All packages use `utils.config.Config` singleton that reads `CONFIG_FILE_PATH` env var.

### Hardware Features
- V1+: Hardware-aware observations (CPU cache sizes, vector width)
- V4.5+: Full hardware embedding with GPU support (theoretical)
- V5: CPU-only (no GPUOccupier)

### Entropy Collapse
**Known issue**: Shaped reward + Transformer → policy collapses to zero entropy.
**Fix**: Disable shaped reward (V4.9) or use `entropy_coef ≥ 0.05`.

## Usage

Select implementation via config:
```json
{
  "implementation": "rl_autoschedular_v4_9"
}
```

Then run:
```bash
sbatch scripts/train/train.sh config/ops_and_blocks/train/v4_9_small.json
```

## Version History

- **v0**: Original LSTM baseline
- **v1**: Added hardware-aware observations
- **v2**: Added shaped reward (intermediate step rewards)
- **v2.5**: Hardened shaped reward (better scaling)
- **v3**: Transformer encoder with self-attention
- **v4**: Combined v1+v2+v3 enhancements
- **v4.5**: Robust integration, isolation fixes
- **v4.9**: Fixed entropy collapse by removing shaped reward
- **v5**: New platform for V5.1/V5.2 development (CPU-only, no eval-in-training)

## Paper Reproduction

The `paper` and `paper_transformer` packages reproduce the original paper results:
- Use `interchange_mode="pointers"`
- No hardware features
- No shaped reward
- Process-isolated execution

See `docs/design/VERSIONS.md` for detailed design notes.
