# MLIR-RL Version Evolution Log

This document tracks the evolution of versioned RL agents.

Rules used in this repository:
- `rl_autoschedular_v0` is the baseline (renamed from `rl_autoschedular`).
- Each version folder (`rl_autoschedular_v1`, `rl_autoschedular_v2`, ...) implements exactly one novelty.
- Every completed version must add one section here with implementation details and validation notes.
- **V5 generation convention (2026-08-06)**: V5 is the *platform* version (accelerated pipeline + the `rl_autoschedular_v5` package); V5.1 and V5.2 are feature increments on the same package. The base is the `paper_transformer` structure — Transformer encoder only (the core contribution). **Hardware-aware observation and shaped reward are abandoned** (explored through V1/V2/V4.5/V4.9, found unhelpful). Design docs live in `docs/design/todo/`: `v5_training_acceleration.md` (V5), `v5_1_full_model_eval.md` (V5.1), `v5_2_expanded_action_space.md` (V5.2). HPO (`HPO_PLAN.md`) is a parallel track, not a version.

## Entry Template
Use this template for each completed version:

### Vx - <Novelty Name>
- Status: <complete|in progress|planned>
- Date completed: <YYYY-MM-DD>
- Novelty scope: <single novelty only>
- Package: `rl_autoschedular_vx`
- Config selector: `"implementation": "rl_autoschedular_vx"`
- Key code changes:
	- <file>: <what changed>
- How to run:
	- <commands>
- Validation:
	- <checks performed>
- Notes/limitations:
	- <important caveats>

---

## Version Entries

### V1 - Hardware-Aware Observation
- Status: complete
- Date completed: 2026-04-22
- Novelty scope: Hardware-aware observation only
- Package: `rl_autoschedular_v1`
- Config selector: `"implementation": "rl_autoschedular_v1"`

Key code changes:
- `rl_autoschedular_v1/*`: full standalone copy of baseline package with internal imports redirected to `rl_autoschedular_v1`.
- `rl_autoschedular_v1/observation.py`:
	- Added hardware feature extraction and normalization.
	- Added `HardwareFeatures` observation part with fields:
		- L1 cache KB
		- L2 cache KB
		- L3 cache KB
		- physical cores
		- logical cores
		- SIMD width bits
		- CPU clock MHz
	- Hardware values are sourced from config overrides, or auto-detected when enabled.
- `rl_autoschedular_v1/model.py`:
	- Updated embedding output shape to include `HardwareFeatures.size()`.
	- Concatenated hardware features into the final embedding used by policy/value networks.

Cross-version routing and config support (required to run by config only):
- `utils/implementation.py`:
	- Generalized implementation mapping to support baseline, legacy `new`, and `vN` packages.
	- Added config-aware implementation resolution.
- `utils/config.py`:
	- Added optional config fields with defaults:
		- `implementation`
		- `hardware_auto_detect`
		- `hardware_l1_kb`, `hardware_l2_kb`, `hardware_l3_kb`
		- `hardware_physical_cores`, `hardware_logical_cores`
		- `hardware_simd_width`, `hardware_clock_mhz`
- `scripts/train.sh`, `scripts/eval.sh`, `scripts/get_base.sh`, `scripts/get_pytorch_times.sh`:
	- Resolve implementation from config by default.
	- Preserve explicit CLI override behavior.
- `scripts/get_base.py`, `scripts/split_json.py`:
	- Resolve implementation from config when provided.
	- Write to shared `exec_times/base.json` by default (no per-implementation prefix).

Dashboard and comparison support:
- `dashboard/dashboard.py`:
	- Refactored from fixed old/new implementations to dynamic implementation token discovery.
	- Added implementation multi-select and generalized per-implementation metrics/charts/tables.
	- Supports comparison across baseline and any `vN` implementation in one run view.

Config/doc updates:
- `config/example.json`, `config/train1.json`, `config/albert.json`: added `implementation` (and hardware fields in example).
- `quick_commands.txt`: updated for config-driven implementation selection.
- `README.md`: documented new config fields.

How to run (example):
1. Set in config:
	 - `"implementation": "rl_autoschedular_v1"`
2. Run pipeline:
	 - `sbatch scripts/get_base.sh config/train1.json`
	 - `python scripts/split_json.py config/train1.json`
	 - `sbatch scripts/train.sh config/train1.json`
	 - `sbatch scripts/eval.sh config/train1.json`
3. Launch dashboard:
	 - `streamlit run dashboard/dashboard.py --server.fileWatcherType none`

Validation performed:
- Python compile checks passed for updated files.
- Import smoke test passed for `rl_autoschedular_v1.model`.
- Implementation mapping verified:
	- `rl_autoschedular -> old_agent / old`
	- `rl_autoschedular_v1 -> v1_agent / v1`
	- `new_rl_autoschedular -> new_agent / new`
- No remaining baseline package import references inside `rl_autoschedular_v1` code.

Notes/limitations:
- V1 does not change reward shaping, model architecture type, or action space.
- V1 novelty is strictly hardware-aware observation.

### V2 - Shaped Reward
- Status: complete
- Date completed: 2026-04-22
- Novelty scope: Reward shaping only
- Package: `rl_autoschedular_v2`
- Config selector: `"implementation": "rl_autoschedular_v2"`

Key code changes:
- `rl_autoschedular_v2/*`: full standalone copy of baseline package with internal imports redirected to `rl_autoschedular_v2`.
- `rl_autoschedular_v2/env.py`:
	- Added dense, intermediate shaped reward from static operation features.
	- Added static efficiency score combining:
		- arithmetic intensity proxy
		- vectorizability indicator
		- parallel-loop ratio
	- Added bounded per-step shaping term and optional explicit vectorization bonus.
	- Preserved terminal execution reward and changed final aggregation to:
		- `final_step_reward = shaped_reward + terminal_speedup_reward`
	- Kept failure penalties unchanged.

Config support added:
- `utils/config.py` default fields:
	- `reward_shaping_enabled`
	- `reward_shaping_scale`
	- `reward_shaping_clip`
	- `reward_shaping_weight_ai`
	- `reward_shaping_weight_vectorizable`
	- `reward_shaping_weight_parallel`
	- `reward_shaping_vectorization_bonus`
- `config/example.json`:
	- Added all reward shaping fields with example values.

How to run (example):
1. Set in config:
	 - `"implementation": "rl_autoschedular_v2"`
2. Optional tuning in config:
	 - `"reward_shaping_enabled": true`
	 - `"reward_shaping_scale": 1.0`
	 - `"reward_shaping_clip": 2.0`
3. Run pipeline:
	 - `sbatch scripts/get_base.sh <config>`
	 - `python scripts/split_json.py <config>`
	 - `sbatch scripts/train.sh <config>`
	 - `sbatch scripts/eval.sh <config>`

Validation performed:
- Python compile checks passed for V2 and routing files.
- Implementation mapping verified:
	- `rl_autoschedular_v2 -> v2_agent / v2`
- Import smoke test passed for `rl_autoschedular_v2.env`.
- Config defaults for shaped reward fields resolved correctly.

Notes/limitations:
- V2 does not modify observation architecture or action space.
- Shaped reward uses static proxies and should be tuned per benchmark family if needed.

### V2.5 - Hardened Shaped Reward (Fair Baseline)
- Status: complete
- Date completed: 2026-05-12
- Novelty scope: Stability/Reliability engineering ported back to V2 to serve as a fair baseline.
- Package: `rl_autoschedular_v2_5`
- Config selector: `"implementation": "rl_autoschedular_v2_5"`

Key code changes:
- Ported the "4 Pillars of Hardening" from V4.5 into the V2 architecture.
- Added Process Isolation for executing JIT/MLIR bindings.
- Added Success-Contingent Reward Negation (zeros out all rewards if final execution fails).
- Added Dynamic Timeouts based on profiling-based margin (10x baseline, max 300s).
- Added Stability Rails (action masking for deep nests and config execution bounds).

How to run:
1. Set in config:
         - `"implementation": "rl_autoschedular_v2_5"`
         - `"results_dir": "results/experiment3"`
         - `"reward_shaping_enabled": true`
2. Run pipeline:
         - `sbatch scripts/train.sh <config>`

Validation performed:
- Eliminates gambler's incentive and handles runtime execution failures safely.

Notes/limitations:
- Used strictly as a performance baseline against V4.5 in `experiment3` to measure the isolated value of Hardware-Awareness and Transformers.

### V3 - Transformer Loop-Nest Encoder
- Status: complete
- Date completed: 2026-04-22
- Novelty scope: Transformer-based encoder only (replacing LSTM embedding)
- Package: `rl_autoschedular_v3`
- Config selector: `"implementation": "rl_autoschedular_v3"`

Key code changes:
- `rl_autoschedular_v3/*`: full standalone copy of baseline package with internal imports redirected to `rl_autoschedular_v3`.
- `rl_autoschedular_v3/model.py`:
	- Replaced `LSTMEmbedding` with `TransformerEmbedding`.
	- Added structured tokenization for consumer/producer operation features:
		- CLS token
		- consumer summary token
		- producer summary token
		- per-loop tokens for consumer loop nest
		- per-loop tokens for producer loop nest
		- optional action-history token
	- Added structural embeddings:
		- role embeddings (global/consumer/producer)
		- token-type embeddings (cls/summary/loop/action)
		- depth embeddings (loop level)
	- Added configurable pooling (`cls` or masked `mean`) and padding-mask support.
	- Kept policy/value heads and PPO interfaces unchanged by preserving embedding output contract.

Config support added:
- `utils/config.py` default fields:
	- `transformer_d_model`
	- `transformer_nhead`
	- `transformer_num_layers`
	- `transformer_ffn_dim`
	- `transformer_dropout`
	- `transformer_activation`
	- `transformer_pooling`
	- `transformer_use_action_history_token`
- `config/example.json`:
	- Added all transformer fields with default values.
- `README.md`:
	- Documented transformer config fields under versioned-agent settings.

How to run (example):
1. Set in config:
	 - `"implementation": "rl_autoschedular_v3"`
2. Optional architecture tuning:
	 - `"transformer_d_model": 256`
	 - `"transformer_nhead": 8`
	 - `"transformer_num_layers": 3`
	 - `"transformer_pooling": "cls"`
3. Run pipeline:
	 - `sbatch scripts/get_base.sh <config>`
	 - `python scripts/split_json.py <config>`
	 - `sbatch scripts/train.sh <config>`
	 - `sbatch scripts/eval.sh <config>`

Validation performed:
- Python compile checks passed for V3 model and config/routing files.
- Import smoke test passed for `rl_autoschedular_v3.model`.
- Implementation mapping verified:
	- `rl_autoschedular_v3 -> v3_agent / v3`
- No remaining baseline package import references inside `rl_autoschedular_v3` Python files.
- End-to-end sanity run (single benchmark, single existing env `~/envs/mlir`) completed:
	- Train run: `results/test_v3_sanity/v3_agent/run_0`
	- Eval run: `results/test_v3_sanity/v3_agent/run_1`
	- Benchmark: `albert_sl128_bs1_generic_1`
	- Eval final speedup: `2.1240696519147306`
	- Eval cumulative reward: `0.3271687539095493`
	- Eval execution time: `57371`
- Architecture ablation smoke results (CPU embedding forward, synthetic observation):
	- A (`pooling=cls`, `layers=2`, `history_token=false`): `output_size=690`, `params=7022271`, `avg_ms=3.5852`
	- B (`pooling=cls`, `layers=3`, `history_token=false`): `output_size=690`, `params=8601791`, `avg_ms=4.8569`
	- C (`pooling=mean`, `layers=3`, `history_token=false`): `output_size=690`, `params=8601791`, `avg_ms=6.9716`
	- D (`pooling=cls`, `layers=3`, `history_token=true`): `output_size=256`, `params=8511679`, `avg_ms=4.8396`

Notes/limitations:
- V3 does not modify reward shaping, environment dynamics, or action space.
- Per-loop token validity for producer operations is inferred from non-zero static feature signals.
- Controlled cross-version comparison was intentionally skipped per user decision.
- Merge-readiness constraints kept in V3 implementation:
	- no script forks (selection remains config-driven via `implementation`)
	- novelty-specific config namespace only (`transformer_*`)
	- policy/value external interfaces preserved for future ultimate-version assembly

### V4 - Combined Enhancements (V1 + V2 + V3)
- Status: complete
- Date completed: 2026-05-10
- Novelty scope: Integrated model combining Hardware-Aware Observation, Shaped Reward, and Transformer Loop-Nest Encoder.
- Package: `rl_autoschedular_v4`
- Config selector: `"implementation": "rl_autoschedular_v4"`

Key code changes:
- `rl_autoschedular_v4/*`: Combines `rl_autoschedular_v1` (explicit hardware features), `rl_autoschedular_v2` (intermediate, dense shaped rewards driven by arithmetic intensity/vectorizability), and `rl_autoschedular_v3` (Transformer loop-nest architecture).

How to run (example):
1. Set in config:
         - `"implementation": "rl_autoschedular_v4"`
         - `"hardware_auto_detect": true`
         - `"reward_shaping_enabled": true`
         - `"reward_shaping_scale": 0.5`
2. Run pipeline:
         - `sbatch scripts/train.sh <config>`
         - `sbatch scripts/eval.sh <config>`

Validation performed:
- Proven to synergize hardware constraints with representation learning.

Notes/limitations:
- Due to aggressive incentives from shaped rewards, the agent learned to push the MLIR compiler into failing states. It experienced ~50% failure rate due to MLIR bindings crashing. Handled in V4.5.

### V4.5 - Robust Integration (Hardened Reliability & Safety)
- Status: in progress
- Date completed: 2026-05-17
- Novelty scope: Stability, Reliability, and Safety (Success-Contingent RL)
- Package: `rl_autoschedular_v4_5`
- Config selector: `"implementation": "rl_autoschedular_v4_5"`

Key code changes:
- `rl_autoschedular_v4_5/execution.py`:
	- Implemented **Process Isolation** for JIT compilation and profiling using `multiprocessing.Process`.
	- Implemented **Dynamic Timeouts** based on unoptimized execution time (10x baseline, max 300s).
	- Implemented **mlir-cpu-runner Fallback** for failed Python binding executions.
- `rl_autoschedular_v4_5/env.py`:
	- Implemented **Success-Contingent Reward Negation**: zeroes out ALL intermediate shaped rewards if final code fails to run.
- `rl_autoschedular_v4_5/actions/__init__.py`:
	- Implemented **Stability Rails**: Masks vectorization for depth > 6 and enforces terminal actions based on `order` and `truncate` limits.
- `rl_autoschedular_v4_5/ppo.py`:
	- Implemented **Resilient Markers**: Persistent global marker directory to ensure eval/train resumption after SIGABRT.

How to run:
1. Set in config:
	 - `"implementation": "rl_autoschedular_v4_5"`
	 - `"results_dir": "results/experiment3"`
2. Run pipeline:
	 - `sbatch scripts/train_condo.sh config/v4_5.json`

Validation:
- [X] Verified 4-value return from isolated `execute_code`.
- [X] Verified reward negation logic with synthetic failures.
- [X] Verified stability rails mask correct actions in deep nests.

Notes/limitations:
- Focus is strictly on eliminating the 50% failure rate observed in V4.
- Success-Contingent rewards may initially slow down learning but ensure the final policy is 100% runnable.


### V4.6 — Reward-Fixed Classic Transformer
- Status: complete
- Date completed: 2026-05-31
- Novelty scope: Corrected reward shaping + optimizer checkpointing (same Classic Transformer as V4.5)
- Package: `rl_autoschedular_v4_5` (shared with V4.5)
- Config selector: `"implementation": "rl_autoschedular_v4_5"`
- Config file: `config/new_dataset/train/v4_6.json`

Key fixes from V4.5:
- **Reward shaping scale**: 1.0 → **0.05** (shaped reward = 10% of terminal, not 20×)
- **Reward shaping clip**: 2.0 → **0.1**
- **Vectorization bonus**: 0.2 → **0.0** (scaled with `reward_shaping_scale`)
- **Slowdown penalty**: Zero intermediate rewards when speedup < 1.0
- **Hardware cores**: Prefers `SLURM_CPUS_PER_TASK` over `os.cpu_count()`
- **Optimizer checkpointing**: Adam state saved with model (dict format: `{'model': ..., 'optimizer': ..., 'step': ...}`)
- **Execution timeout**: Min reduced 30s → 2s, multiplier 10x → 5x (`MIN_EXEC_TIMEOUT` env var, default 2)

Architecture:
- Transformer: d_model=256, nhead=8, num_layers=3, ffn_dim=1024
- New dataset: 18-model block benchmarks, 8822 train / 2163 eval
- Training: nb_iterations=10000, bench_count=64, batch_size=32

Results (at iter 153, 5h timeout): avg speedup **1.82x**, best **16.69x**

Config mismatch discovered (2026-06-10):
- **Eval config** (`v4_6_eval.json`) used old reward values (scale=1.0, clip=2.0, vec_bonus=0.2)
- **Train config** (`v4_6.json`) used corrected values (scale=0.05, clip=0.1, vec_bonus=0.0)
- **Impact: None** — reward shaping only affects logged metrics during eval, not action selection or exec times

Notes:
- Training resumed after timeout, 72h time limit applied

### V4.7 — Reward-Fixed Small Transformer
- Status: complete
- Date completed: 2026-05-31
- Novelty scope: Same reward fixes as V4.6 but with reduced transformer for faster training
- Package: `rl_autoschedular_v4_5` (shared with V4.5)
- Config selector: `"implementation": "rl_autoschedular_v4_5"`
- Config file: `config/new_dataset/train/v4_7.json`

Key differences from V4.6:
- **Transformer**: d_model=**64**, nhead=**2**, num_layers=**2**, ffn_dim=**128** (3.6× fewer params)
- **Training speed**: ~62s/iter vs V4.6's 97s/iter (36% faster per iteration)
- All reward fixes and timeout improvements identical to V4.6

Results (at iter 238, 5h timeout):
- Avg speedup: **1.42x** (regressed from earlier peak of **2.59x** at iter 102)
- Best: **7.66x**
- Hypothesis: Smaller model converges faster initially but lacks capacity to sustain improvement — gradient noise or shaped reward exploitation causes regression

Config mismatch discovered (2026-06-10):
- **Eval config** (`v4_7_eval.json`) used old reward values (scale=1.0, clip=2.0, vec_bonus=0.2)
- **Train config** (`v4_7.json`) used corrected values (scale=0.05, clip=0.1, vec_bonus=0.0)
- **Impact: None** — same reason as V4.6

Notes:
- Peaked early (2.59x at iter 102) then degraded — classic overfitting to bad schedules
- Valuable as an ablation: confirms d_model≥256 needed for stable convergence

### V4.8 — Reward-Fixed Classic Transformer (Eval-Corrected)
- Status: complete
- Date completed: 2026-06-01
- Novelty scope: Identical architecture to V4.6 with corrected eval config (fixes V4.6 eval config mismatch)
- Package: `rl_autoschedular_v4_5` (shared with V4.5)
- Config selector: `"implementation": "rl_autoschedular_v4_5"`
- Config file: `config/new_dataset/train/v4_8.json`

Key differences from V4.6:
- **Eval config**: Uses corrected reward values (scale=0.05, clip=0.1, vec_bonus=0.0) matching training config — fixes V4.6's eval config discrepancy
- Training config: Identical to V4.6 (Classic Transformer, same reward fixes)

Results (at iter 155, 5h timeout): avg speedup **1.98x**, best **14.59x**

Notes:
- Slightly better than V4.6 (1.98x vs 1.82x at same iteration) — likely noise, not systematic
- Second run confirms Classic Transformer with reward fixes is viable
- Eval config correction ensures fair comparison with train-era rewards

### V4.9 — No Shaped Reward (Entropy Collapse Fix)
- Status: complete
- Date completed: 2026-06-10
- Novelty scope: Entropy collapse fix by removing shaped reward
- Package: `rl_autoschedular_v4_9`
- Config selector: `"implementation": "rl_autoschedular_v4_9"`

Discovery (2026-06-10):
- During single_ops_dataset experimentations, V4.6/V4.7/V4.8 all suffered **entropy collapse** mid-training
- Entropy dropped to zero (from healthy 1.0-3.0), causing policy to freeze and produce identical actions
- Root cause: shaped reward + transformer encoder → policy converges to deterministic local optimum
- V0 (no shaped reward, LSTM) survived — entropy healthy throughout 14K+ iterations

Key code changes:
- `rl_autoschedular_v4_9/*`: full standalone copy of `rl_autoschedular_v4_5` with internal imports redirected to `rl_autoschedular_v4_9`
- `rl_autoschedular_v4_9/env.py`:
  - `__shaped_reward()` hardcoded to return 0.0 (shaped reward disabled)
  - All helper methods (`__static_efficiency_score`, `__estimate_arithmetic_intensity`, etc.) kept as dead code

Variants (config-driven):
- **V4.9 small** (`config/single_ops_dataset/train/v4_9_small.json`): V4.7 transformer (d=64, 2 heads, 2 layers, ffn=128)
- **V4.9 large** (`config/single_ops_dataset/train/v4_9_large.json`): V4.8 transformer (d=256, 8 heads, 3 layers, ffn=1024)

Config files:
- `config/single_ops_dataset/train/v4_9_small.json` — training config for small variant
- `config/single_ops_dataset/train/v4_9_large.json` — training config for large variant
- `config/single_ops_dataset/eval/v4_9_small_eval.json` — eval config for small variant
- `config/single_ops_dataset/eval/v4_9_large_eval.json` — eval config for large variant

Validation:
- All 15 Python files compile successfully (`python -m py_compile`)
- All 4 config files are valid JSON
- No v4_5 references remain in V4.9 package
- `reward_shaping_enabled: false` in both train and eval configs

Notes/limitations:
- V4.9 keeps all V4.5 reliability features (process isolation, dynamic timeouts, stability rails)
- Shaped reward code kept as dead code (not deleted) for reference
- Expected: entropy stays > 0.1 throughout training (like V0)
- Expected: eval speedup improves over training (not flat like V4.6/V4.7/V4.8)
- **V4.9's HW features and shaped reward are the last use of both** — V5 abandons them (see V5 entry). Detailed design + measured results: [`docs/design/done/v4_9_no_shaped_reward.md`](done/v4_9_no_shaped_reward.md)

### Paper - LSTM Baseline (Paper Artifact)
- Status: complete
- Novelty scope: Paper artifact — LSTM encoder, no HW features, no shaped reward, `interchange_mode="pointers"`, process-isolated
- Package: `rl_autoschedular_paper`
- Config selector: `"implementation": "rl_autoschedular_paper"`
- Notes: Used for paper single-op benchmarks (18 models). Clean-room baseline for the paper's comparisons.

### Paper Transformer - Transformer Ablation (Paper Artifact)
- Status: complete
- Novelty scope: Paper ablation — Transformer encoder (self-attention, CLS pooling), no HW features, no shaped reward, `interchange_mode="pointers"`, process-isolated
- Package: `rl_autoschedular_paper_transformer`
- Config selector: `"implementation": "rl_autoschedular_paper_transformer"`
- Notes: **This package is the structural base for V5** — verified 2026-08-06: paper_transformer = v4_9 minus HW observation minus shaped reward, and is fully standalone (own `utils/config.py`), unlike v4_9 which imports the shared root `utils.config`. The Transformer encoder (the core contribution) is identical in both.

---

## V5 Generation (current research line, 2026-08-06)

### V5 - Accelerated Training Platform (in progress)
- Status: **in progress (design phase)** — see `docs/design/todo/v5_training_acceleration.md`
- Novelty scope: Platform version — no new RL novelty; infrastructure that V5.1/V5.2 depend on
- Package: `rl_autoschedular_v5` (new standalone package, `paper_transformer` structure)
- Config selector: `"implementation": "rl_autoschedular_v5"`
- Base: `rl_autoschedular_paper_transformer` structure — Transformer encoder only
- Abandoned features (explored in v4_9, found unhelpful): hardware-aware observation (`hardware_*` config fields), reward shaping (`reward_shaping_*` config fields) — dead fields dropped from V5 config
- Key changes (planned):
  - `rl_autoschedular_v5/*`: standalone copy of paper_transformer structure, internal imports redirected to `rl_autoschedular_v5`
  - `utils/config.py`: drop `hardware_*` and `reward_shaping_*` fields
  - Slurm resources: `--cpus-per-task=64` (train, matches `bench_count=64`) / 128 (eval), `--mem=128G`/`256G`, `--time=7-00:00:00` (compute partition max), no GPU (see doc)
  - **No GPU decision**: GPU accelerates only the ~3% sampling/PPO fraction (~1.05-1.1x); CPU parallelism 12→64/128 accelerates the ~95% MLIR-exec fraction (5-10x). V5 runs on the Jubail `compute` partition only.
  - GPUOccupier: rejected for V5 (CPU-only pipeline; `start()` requires CUDA)
  - Optional (architect's call): persistent MLIR worker pool (spawn-based, never fork)
- How to run: `sbatch scripts/train/train.sh config/.../v5.json`
- Validation: `python -m py_compile` all package files; `bash -n` scripts; smoke run via sbatch; iteration wall-clock 50s → 8-12s

### V5.1 - Full-Model Evaluation Support (planned)
- Status: **planned (design phase)** — see `docs/design/todo/v5_1_full_model_eval.md`
- Novelty scope: Evaluate the block-trained policy on full `.mlir` model files (ResNet18, T5, GPT-2) without retraining
- Package: `rl_autoschedular_v5` (extends V5 in place)
- Key insight: per-op features from full models are identical to block features; schedule each op in topological order on the full Module
- Reuses existing `scripts/checkpoint/ckpt_scan_all.sh` + `submit_ckpt_scan.sh` (already do partial full-model eval, v4_5) — reconcile, don't duplicate
- Checkpoint compat: with **paper_transformer** checkpoints, NOT v4_9 (observation size differs — HW features dropped)
- Note: runs on `compute` partition (CPU-bound MLIR execution; no GPU)

### V5.2 - Expanded Transformation Action Space (planned)
- Status: **planned (design phase)** — see `docs/design/todo/v5_2_expanded_action_space.md`
- Novelty scope: Expanded transformation action space only
- Package: `rl_autoschedular_v5` (extends V5 in place)
- Config selector: `"implementation": "rl_autoschedular_v5"`
- Base: `rl_autoschedular_paper_transformer` action space (6 actions) — transforms referenced live in `rl_autoschedular_v4_9/transforms.py` (padding, unrolling, packing already implemented there) and are ported into V5
- Sequencing note: user confirmed V5 → V5.1 → V5.2 order (no swap). Consequence accepted: V5.1's full-model numbers describe the 6-action agent; re-run after V5.2 if the paper needs them for the expanded agent.

> **HPO (parallel track, not a version)**: `docs/design/todo/HPO_PLAN.md` — Optuna TPE tuning of Transformer architecture (d_model, nhead, num_layers, ffn_dim, dropout, pooling) on ops_and_blocks, running in parallel with V5/V5.1, feeding hyperparameters before V5.2.

