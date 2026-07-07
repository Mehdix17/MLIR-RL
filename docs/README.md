# MLIR-RL Documentation Sitemap

Welcome to the MLIR-RL documentation repository. This directory is organized into thematic subfolders to make finding active operational and design specifications easy.

---

## 📂 Directory Map

### 1. ⚙️ [Pipeline & Operations](pipeline/)
Guides on setting up, training, and evaluating RL agents on MLIR loop nests:
* [TRAINING_AND_EVALUATION.md](pipeline/TRAINING_AND_EVALUATION.md) — Quick commands cheat-sheet for Slurm and local operations.
* [TRAINING_MANUAL.md](pipeline/TRAINING_MANUAL.md) — Comprehensive guide on RL agent architecture, rollout simulation, PyTorch thread capping, and hyperparameters.
* [PIPELINE.md](pipeline/PIPELINE.md) — Lifecycle of dataset processing, baseline timings, and train/eval splits.
* [COMMANDS.md](pipeline/COMMANDS.md) — Exhaustive index of CLI arguments and scripts.

### 2. 📊 [Results & Logging](results/)
Details on training outputs, cache files, and performance analytics:
* [RESULTS.md](results/RESULTS.md) — Experimental results tables across datasets.
* [RESULTS_ARCHITECTURE.md](results/RESULTS_ARCHITECTURE.md) — Layout of `run_N` logs, models, and cumulative training progress JSON outputs.
* [DASHBOARD.md](results/DASHBOARD.md) — Instructions for launching and utilizing the Streamlit comparison dashboard.

### 3. 🧠 [Design & Architecture](design/)
Theoretical specifications, observation features, action spaces, and config files:
* [CONFIG.md](design/CONFIG.md) — Explanation of hyperparameter configuration schema.
* [VERSIONS.md](design/VERSIONS.md) — Design logs of all package versions (`v0` to `v4_9`) and paper baselines.
* [METHODOLOGY_ALIGNMENT.md](design/METHODOLOGY_ALIGNMENT.md) — Methodological notes aligning the code with scientific paper implementations.
* **Architecture Novelties**: Detailed specs on [v0 Baseline](design/v0_model_detailed.md), [v1 Hardware Observations](design/v1_hardware_aware_observation.md), [v2 Shaped Rewards](design/v2_shaped_reward.md), [v3 Transformer Encoder](design/v3_transformer_loop_nest_encoder.md), [v4 Combined Model](design/v4_combined_model.md), and [v4.5 Reliability Systems](design/v4_5_reliability_logic.md).

### 4. 🔬 [Failure Investigations](investigations/)
Deep-dives into historical design flaws, compiler errors, and mitigations:
* [ENTROPY_COLLAPSE_INVESTIGATION.md](investigations/ENTROPY_COLLAPSE_INVESTIGATION.md) — Root cause analysis and fixes for policy collapses under shaped rewards.
* [V4_FAILURE_INVESTIGATION.md](investigations/V4_FAILURE_INVESTIGATION.md) — Investigation into why legacy V4 runs suffered high crash rates.
* [LLVM_DEFENSE.md](investigations/LLVM_DEFENSE.md) — Guide to fixing broken symbol links inside standard compiler builds.
* [FUTURE_WORKS_RL_FULL_MODEL_SUPPORT.md](investigations/FUTURE_WORKS_RL_FULL_MODEL_SUPPORT.md) — Roadmaps for full-model compilation instead of block-based operations.

### 5. 🗄️ [Historical Archive & Post-Mortems](archive/)
Quarantined logs of resolved bug fixes and legacy issues:
* [GPT2_JIT_FIX.md](archive/GPT2_JIT_FIX.md) — Hotfix for GPT2 benchmark compile crashes.
* [LLAMA_BF16_FIX.md](archive/LLAMA_BF16_FIX.md) — Hotfix for LLaMA 16-bit float tensor conversions.
* [PYTORCH_JIT_DEBUGGING.md](archive/PYTORCH_JIT_DEBUGGING.md) — Debug logs of legacy memory errors.
* [TRAIN_FAILURES_2026_06_24.md](archive/TRAIN_FAILURES_2026_06_24.md) — Fixes for TiledFusion boundary checks.
* [FIX1_MODEL_SURGERY.md](archive/FIX1_MODEL_SURGERY.md) — Post-mortem notes on model weight surgeries.

### 6. 🚀 [Onboarding Guides](onboarding/)
Step-by-step onboarding modules for new developers:
* [01_WHAT_IS_MLIR.md](onboarding/01_WHAT_IS_MLIR.md) — Foundational MLIR concepts.
* [02_PROJECT_SETUP.md](onboarding/02_PROJECT_SETUP.md) — Detailed environment setup and LLVM troubleshooting.
* [03_ARCHITECTURE_OVERVIEW.md](onboarding/03_ARCHITECTURE_OVERVIEW.md) — Codebase layout and logic flow.
* [04_RL_AGENT_DEEP_DIVE.md](onboarding/04_RL_AGENT_DEEP_DIVE.md) — RL agents, actors, critics, and feature observations.
* [05_AGENT_VERSIONS.md](onboarding/05_AGENT_VERSIONS.md) — Implementation differences (v0 to v4.9).
* [06_ACTIONS_AND_OPTIMIZATIONS.md](onboarding/06_ACTIONS_AND_OPTIMIZATIONS.md) — Available transformations and hierarchy.
* [07_CONTRIBUTIONS.md](onboarding/07_CONTRIBUTIONS.md) — Git flows and development checks.
* [08_DATA_PIPELINE.md](onboarding/08_DATA_PIPELINE.md) — Generating splits and baseline runtimes.
* [09_MONITORING_AND_EVALUATION.md](onboarding/09_MONITORING_AND_EVALUATION.md) — Neptune tracking and evaluation commands.

### 7. 💻 [HPC & System Setup](hpc/)
Setup instructions for Slurm cluster runs:
* [HPC_SETUP.md](hpc/HPC_SETUP.md) — Cluster modules, GCC environments, and Conda paths.
* [HPC_HARDWARE.md](hpc/HPC_HARDWARE.md) — Details on CPU node partitions (Bergamos vs. Genoas).

### 8. 📁 [Data & Extraction](data/)
* [NEW_DATASET.md](data/NEW_DATASET.md) — Instructions for splitting data and converting networks to MLIR.
* [EXTRACT_OPS_README.md](data/EXTRACT_OPS_README.md) — CLI flags for operation extraction scripts.
* [FULL_MODEL.md](data/FULL_MODEL.md) — Details on complete MLIR model structures.
