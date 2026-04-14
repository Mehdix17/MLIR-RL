# MLIR-RL Pipeline Overview
> Generated: 2026-02-21

---

## The three pipelines

Implementation can be selected with `AUTOSCHEDULER_IMPL` (default: `rl_autoschedular`).
Use `AUTOSCHEDULER_IMPL=new_rl_autoschedular` once the enhanced package is added.

```
Pipeline 1 — Benchmark generation
  (a) synthetic:  mlir_generators.py → build_benchmark.py → JSON dataset
  (b) from model: vision2mlir.py / transformers2mlir.py → _linalg.mlir
                  → wrap_mlir.py → code_files/*.mlir
                  → scripts/get_base.py  → base_exec_times.json   ← MISSING STEP

Pipeline 2 — RL training
  config JSON + json_file (exec times) + benchmarks_folder_path (*.mlir)
  → scripts/train.py → results/<experiment>/<old_agent|new_agent>/run_N/{models/, logs/, exec_data.json}

Pipeline 3 — Evaluation
  config JSON + trained checkpoint (.pt)
  → scripts/eval.py → results/<experiment>/<old_agent|new_agent>/run_N/exec_data.json
```

---

## What each folder/file means

**`config/*.json`** — hyperparameters for the RL agent. The two critical fields for connecting the pipelines are:
- `"json_file"` — path to a `{"bench_name": execution_time_ns}` map; this is the **baseline** execution time the RL agent tries to beat
- `"benchmarks_folder_path"` — folder containing the `.mlir` files, one file per key in `json_file`
- Everything else is PPO tuning (lr, batch_size, epochs, etc.)

**`data/nn/`** layout:
```
code_files/          ← final wrapped .mlir files (the ones scripts/train.py reads)
generated/
  code_files/        ← _torch.mlir + _linalg.mlir (pipeline output)
non_stripped_models/ ← weight-containing backups
gen/                 ← old data_generation_random.py (legacy)
```
There are currently **no JSON files** in `data/nn/` — that JSON (mapping bench names → baseline times) must be created by running `scripts/get_base.py` or `scripts/build_generated_exec_json.py`.

**`scripts/`**:

| Script | Purpose |
|---|---|
| `generate_all_models.sh` | Slurm array job — one task per model, runs the MLIR export pipeline |
| `train_generated.sh` | Slurm job — runs `scripts/train.py` with `config/generated-benchmarks.json` |
| `eval_generated.sh` | Slurm job — runs evaluation against a saved checkpoint |
| `build_generated_exec_json.py` | Measures baseline execution time for each `.mlir` in a folder → writes the JSON needed by `scripts/train.py` |
| `train_example.sh` | Example train job with custom config |

**`results/<experiment>/<old_agent|new_agent>/run_N/`** — one folder per training/eval run:
- `models/` — checkpoint `.pt` files saved periodically
- `exec_data.json` — execution times logged *during* training for each benchmark
- `logs/` — stdout/stderr logs

**`results/<experiment>/exec_times/`** — shared baseline data for that experiment:
- `{old|new}_base.json`
- `{old|new}_base_train.json`
- `{old|new}_base_eval.json`
- `pytorch.json`

---

## The missing link: how `_linalg.mlir` → training data

This is the gap between pipeline 1b and pipeline 2. The current `generated/code_files/` has
the MLIR, but `scripts/train.py` can't use it yet. The full chain is:

```
Step 1 ✅  vision2mlir.py / transformers2mlir.py
           → generated/code_files/{model}_linalg.mlir

Step 2 ❌  wrap_mlir.py  (adds timed @main entry point)
           → data/nn/code_files/{model}.mlir

Step 3 ❌  scripts/get_base.py  (runs each .mlir, measures baseline ns)
           → data/nn/base_exec_times.json
              {"model_name": 823519234, ...}

Step 4 ❌  split into train/eval JSON files
           → data/nn/train_exec_times.json
           → data/nn/eval_exec_times.json

Step 5 ❌  create/update config JSON pointing to those files + code_files/
```

---

## What's broken in `generate_all_models.sh`

The script is outdated — it still uses the old file names and is missing the GCC14 fix:

1. References `models-to-onnx.py` and `vision-to-mlir.py` (deleted, now `transformers2mlir.py` / `vision2mlir.py`)
2. Missing `LD_LIBRARY_PATH` fix for GCC14 `libstdc++`
3. Missing `--backend direct` for convnext_tiny and resnet50
4. GPT2/bart/t5 are excluded with a comment — the direct route might work for GPT2

---

## Adding GPT2

GPT2 is already declared in `transformers2mlir.py`'s `SUPPORTED_MODELS`. The ONNX route is
known broken (dynamic control flow), but the `--backend direct` route may work. The steps would be:

1. Test locally first:
   ```bash
   LD_LIBRARY_PATH="$GCC14_LIB:$LD_LIBRARY_PATH" \
   PATH="/home/tb3654/.conda/envs/mlir/bin:$PATH" \
   python data_utils/transformers2mlir.py --model gpt2 --backend direct \
     --output-dir data/nn/generated/code_files
   ```
2. If that works, fix `generate_all_models.sh` to use the new scripts and include GPT2
3. Then run the missing steps 2–5 above to get it into training

---

## Next steps (suggested order)

1. Fix `generate_all_models.sh` (update script names, add GCC14 fix, add `--backend direct` cases)
2. Write a `wrap_all.sh` or extend the orchestrator to batch-wrap `generated/code_files/*_linalg.mlir` → `data/nn/code_files/`
3. Run `scripts/get_base.py` on `data/nn/code_files/` to produce `base_exec_times.json`
4. Split into train/eval JSON and create a `config/nn-models.json`
5. Test GPT2 locally with `--backend direct`, then add to Slurm array
