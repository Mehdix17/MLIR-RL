# MLIR-RL V5.1 Experiment Status

Generated: 2026-08-24 · Dataset: merged `ops_and_blocks` + `legacy_paper`

## 1. At-a-glance

| Experiment | Dataset | Train | Per-ckpt Eval | Final (median) Eval | Best Ckpt (speedup) | Plots | Cleanup |
|---|---|---|---|---|---|---|---|
| `v5` | ops_and_blocks | ✅ done (20000) | ✅ 200/200 | ✅ 9 ckpts | 10950 (4.42) | ✅ all 4 | ✅ 9 kept |
| `v5_no_transformer` | ops_and_blocks | ✅ done (20000) | 🔄 186/200 | ⏳ queued | 350 (4.13) | ⏳ wait final | ✅ 10 kept |
| `v5` | legacy_paper | ✅ done (20000) | 🔄 199/200 | ✅ 9 ckpts | 5350 (1.55) | ✅ 2 (op-only) | ✅ 9 kept |
| `v5_no_transformer` | legacy_paper | 🏃 70%(~14100) | 🔄 113 **running** | ❌ n/a | 4150 (1.68) | ⏳ wait | ⏳ wait finish |

## 2. Training Progress

| Version | Iteration | Progress | Status | Failures |
|---|---|---|---|---|
| `v5` | 20000 | 100% | stopped | 1 |
| `v5_no_transformer` | 20000 | 100% | stopped | 1 |
| `v5` | 20000 | 100% | stopped | 6 |
| `v5_no_transformer` | ~14100 | ~70% | **running** (train 17270594 + 13 dask workers bn002/bn003) | 6 |

## 3. Eval Progress

Grid = checkpoints 50..19950 step 100 (200 total). Backlogs filling now.

| Agent | Evaluated | Missing | Active Job |
|---|---|---|---|
| `v5` | 200/200 | none | — |
| `v5_no_transformer` | 186/200 | 14 (18850..19950) | 17381174 RUNNING (18550-19950) |
| `v5` | 199/200 | 1 (19450) | 17381175 RUNNING (19450) |
| `v5_no_transformer` | 113 | grows w/ training | 17381176 PENDING (11550-14050) |

## 4. Final Eval (5-run median, `eval_final/`)

| Agent | ckpts | Best (median) | Status |
|---|---|---|---|
| `v5` | 9 | 10950 (4.42) | ✅ done |
| `v5` | 9 | 5350 (1.55) | ✅ done |
| `v5_no_transformer` | — | 350 (4.13, single-run) | ⏳ job 17381237 PENDING (needs eval backlog 1st) |
| `v5_no_transformer` | — | 4150 (1.68, single-run) | ❌ can't final until training done |

## 5. Best Checkpoints (final / CSV)

| Agent | Best ckpt | Median speedup | Note |
|---|---|---|---|
| `v5` | 10950 | **4.42** | top of all 4 |
| `v5_no_transformer` | 350 | 4.13 (pre-final) | final pending |
| `v5` | 5350 | 1.55 | op-only dataset lower ceiling |
| `v5_no_transformer` | 4150 | 1.68 (pre-final) | final pending |

## 6. Plot Generation (`plots/experimentation_plots/`)

| Agent | Evolution | Family | Family (no-LLaMA) | Op-type |
|---|---|---|---|---|
| `v5` | ✅ | ✅ | ✅ | ✅ |
| `v5` | ✅ | — (op-only) | — | ✅ |
| `v5_no_transformer` | ⏳ after final | ⏳ | ⏳ | ⏳ |
| `v5_no_transformer` | ⏳ after training | — | — | ⏳ |

## 7. Checkpoint Cleanup (models/)

| Agent | Before | Kept | Anchor | Note |
|---|---|---|---|---|
| `v5` | — | 9 | 20000 | done earlier |
| `v5_no_transformer` | 399 | 9 (+20000) | kept | ✅ applied this session (consented) |
| `v5` | — | 9 | 20000 | done earlier |
| `v5_no_transformer` | 276 | — | — | untouched — still training |

## 8. Outstanding / Remaining

1. **v5_no_transformer final eval** — job 17381237 queued behind eval backlog; run final → regenerate best-ckpt CSVs → build its 3 plots.
2. **v5_no_transformer_legacy_paper** — finish training (~14100/20000) → re-submit per-ckpt evals for new ckpts → cleanup-checkpoints → final eval → op-only plots.
3. **Re-verify eval backlogs** after jobs drain (squeue) — confirm 200/200, 200/200 clean.
4. **Lustre** — inodes 54% soft / 27% hard, fine (space 3%).
5. Optionally: remove stale entries in `scripts/eval/active_eval_jobs.json` if reports show phantom PENDING.