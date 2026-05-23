# MLIR-RL Version Evolution Log

This document tracks the evolution of versioned RL agents.

Rules used in this repository:
- `rl_autoschedular_v0` is the baseline (renamed from `rl_autoschedular`).
- Each version folder (`rl_autoschedular_v1`, `rl_autoschedular_v2`, ...) implements exactly one novelty.
- Every completed version must add one section here with implementation details and validation notes.

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

### V4 - Action Space Expansion (Pad + Pack + Unroll)
- Status: complete
- Date completed: 2026-04-24
- Novelty scope: Expanded transformation action space only
- Package: `rl_autoschedular_v4`
- Config selector: `"implementation": "rl_autoschedular_v4"`

Key code changes:
- `rl_autoschedular_v4/*`: full standalone copy of baseline package with internal imports redirected to `rl_autoschedular_v4`.
- `rl_autoschedular_v4/transforms.py`:
	- Added `transform_pad()`: `structured.match` → `structured.pad` → re-annotate tag.
	- Added `transform_pack()`: `structured.match` → `structured.pack` → re-annotate tag.
	- Added `transform_unroll()`: `structured.match` → `structured.convert_to_loops` → `loop.unroll`.
- `rl_autoschedular_v4/actions/pad.py`:
	- New `Pad` action (`symbol = 'P'`).
	- Per-dimension categorical parameter: 0 = no pad, 1 = pad to multiple 2, 2 = multiple 4, etc.
	- Updates loop upper bounds in `update_features` by rounding up to the chosen multiple.
- `rl_autoschedular_v4/actions/pack.py`:
	- New `Pack` action (`symbol = 'PK'`).
	- Per-dimension categorical parameter reusing tile-size encoding (powers of 2).
	- Updates loop upper bounds in `update_features` to `ceil(orig / pack_size)`.
- `rl_autoschedular_v4/actions/unroll.py`:
	- New `Unroll` action (`symbol = 'U'`, `terminal = True`).
	- Single categorical parameter mapping to unroll factor (2, 4, 8, ...).
	- Only allowed when `producer_tag is None` (safe to lose tag after `convert_to_loops`).
- `rl_autoschedular_v4/actions/__init__.py`:
	- Registered `Pad`, `Pack`, `Unroll` in `ActionSpace.supported_actions`.
	- Action space expands from 6 → 9 actions; model auto-adapts via dynamic `ActionSpace` queries.
- `utils/config.py`:
	- Added defaults: `num_pad_multiples = 3`, `num_unroll_factors = 3`.

How to run (example):
1. Set in config:
	 - `"implementation": "rl_autoschedular_v4"`
2. Run pipeline:
	 - `sbatch scripts/get_base.sh config/train1.json`
	 - `python scripts/split_json.py config/train1.json`
	 - `sbatch scripts/train.sh config/train1.json`
	 - `sbatch scripts/eval.sh config/train1.json`
3. Launch dashboard:
	 - `streamlit run dashboard/dashboard.py --server.fileWatcherType none`

Validation performed:
- Python compile checks passed for all new/modified files.
- Import smoke test passed for `rl_autoschedular_v4.model`.
- `ActionSpace.size()` verified as 9 with symbols `['NT', 'T', 'TP', 'TPF', 'I', 'V', 'P', 'PK', 'U']`.
- Dummy-state mask tests passed for Pad, Pack, and Unroll.
- No remaining baseline package import references inside `rl_autoschedular_v4` Python files.

Notes/limitations:
- V4 does not modify observation architecture, reward shaping, or model type.
- `Unroll` is terminal because `convert_to_loops` destroys the structured-op tag.
- Pack access-pattern tracking is approximate in `update_features` (loop bounds only).
- New config fields have backward-compatible defaults; existing configs work without modification.

### V4.5 - Hardened Robust Integration
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
