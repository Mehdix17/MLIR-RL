# Parameter Counts: V5 Transformer vs LSTM Variants

Measured 2026-08-17 by instantiating each model stack with its real config and
summing `p.numel()` (verified, not estimated).

## Summary

| Variant | Embedding | Embedding params | Full model (Value+Policy) |
|---|---|---|---|
| `v5` (transformer) | TransformerEmbedding (d_model=64, nhead=2, ffn=128, 2 layers, cls pool) | 129,088 | 3,893,047 |
| `v5_no_transformer` (LSTM) | LSTMEmbedding (hidden = OpFeatures.size() = 711) | 4,111,100 | 12,519,599 |
| `paper_original` (LSTM) | LSTMEmbedding (hidden = 411, hardcoded) | 2,147,900 | 8,285,999 |

- Transformer is ~32× lighter than the v5 LSTM embedding (129K vs 4.1M) and
  ~3.2× lighter full-model (3.9M vs 12.5M).
- LSTM cost scales quadratically in hidden size: `nn.LSTM(512, h)` has
  ~4·(512·h + h²) weights.

## ⚠️ LSTM hidden-size discrepancy (v5_no_transformer ≠ paper_original)

| Variant | LSTM hidden size | Source |
|---|---|---|
| `paper_original` | 411 (hardcoded) | `rl_autoschedular_paper/model.py:261` |
| `v45_no_transformer` | 512 (`self.hidden_size`, + HardwareFeatures in output) | `rl_autoschedular_v45_no_transformer/model.py` |
| `v5_no_transformer` | `OpFeatures.size()` = 711 | `rl_autoschedular_v5_no_transformer/model.py:249` |

`v5_no_transformer` was expected to reuse the paper's LSTM (hidden=411) but
derives it from `OpFeatures.size()` (711). Both packages have identical
`OpFeatures.size() == 711`, so the only difference is the hardcoded 411 vs the
dynamic lookup. Fixed 2026-08-17 by hardcoding 411 to match `paper_original`.

Note: the trained checkpoints of `v5_no_transformer` currently in flight were
produced with hidden=711 — the fix applies to future launches/resumes from
scratch, NOT to existing checkpoints (state dicts are shape-incompatible, and
the running job keeps its already-loaded architecture).