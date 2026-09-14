---
name: plot-experimentation-results
description: Unified runbook for generating line evolution charts and benchmark family bar comparison plots for MLIR-RL experimentations. Reads per-experiment CSVs (results/<exp>_agent/csvs/) and writes only images into plots/. Does NOT generate or update CSVs — that is utils/csvs.py's job.
---

# MLIR-RL Experimentation Plotting Skill

Use this skill to generate:
- **Line evolution charts**: checkpoint iteration vs. geometric mean speedup
- **Model family bar charts**: speedup per model (bert, gpt2, resnet50, ...) — excludes synthetic op benchmarks
- **Operation type bar charts**: speedup per op type (add, conv_2d, matmul, pooling, relu) — only synthetic op benchmarks

CSV data is separate from plots: each experiment owns its CSVs in
`results/<exp>_agent/csvs/` (`checkpoint_speedups.csv`,
`best_checkpoint_speedups.csv`,
`best_checkpoint_benchmark_family_results.csv`,
`best_checkpoint_operation_type_results.csv`). They are updated automatically
after each eval (eval.py hook) and regenerable at any time with `utils/csvs.py`.
This skill only consumes them and writes PNGs.

**Best-checkpoint plots are final numbers — generate them AFTER `final-checkpoint-eval`.**
The `best_checkpoint_*results*.csv` files reflect the **median** of 5 eval runs (written by
`final_eval.py`, which picks the best checkpoint and saves it to `csvs/`). Running the
comparison plots before that yields stale single-run bests. The `checkpoint_evolution`
line chart can be made at any time — it reads the per-checkpoint ranking CSV.

---

## 🎯 Step 1. Ask the User (Interactive Menu)

Use the `ask_question` tool with `is_multi_select: false` for each item:

1. **Dataset**: suggest `ops_and_blocks` | `new` | `single_ops`
2. **Agent versions**: suggest the set of paper or ablation agents based on the dataset
3. **Output folder**: defaults to `plots/experimentation_plots/<dataset>/<experimentation>/`,
   where `<experimentation>` is the underscore-joined agent list (e.g.
   `plots/experimentation_plots/ops_and_blocks/v5_distributed/`). Images are written
   directly into that dir (no `pngs/` subfolder, no numbered `exp<N>` dirs). Pass a
   custom path via `--out-dir` if it doesn't suit.
4. **Refresh CSVs first?** If new eval results exist since the last plot, regenerate the
   experiments' CSVs before plotting:
   ```bash
   python utils/csvs.py --results-dir results/<dataset>_results/<exp>_agent --agent <exp> [--all]
   ```
   (eval.py already auto-updates `checkpoint_speedups.csv` after each eval — this is only
   needed when the CSVs are missing or the comparison CSVs are stale.)

---

## 🛠️ Step 2. Generate the Plots

Run the plots per experimentation session. PNGs land directly in `<out-dir>/`; no CSVs are written by this step. When the dataset matches the run, the default `<out-dir>` is `plots/experimentation_plots/<dataset>/<agents>/`:

### 1. Checkpoint Evolution Line Chart
```bash
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
python scripts/plots/generate_plots.py \
  -d <dataset> -m evolution --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/checkpoint_evolution.png` (reads `results/<exp>_agent/csvs/checkpoint_speedups.csv`)
*Only multiples-of-100 checkpoints are included for a smooth curve.*

### 2. Model Family Comparison (all model families)
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type models_only --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/best_checkpoint_benchmark_family_results.png`
*Uses best checkpoint per agent (highest overall geo-mean). Excludes all op-type benchmarks.*

### 3. Model Family Comparison (without LLaMA)
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type models_only --exclude llama3_2_1b --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/best_checkpoint_benchmark_family_results_no_llama3.png`

### 4. Operation Type Comparison
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type ops_only --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/best_checkpoint_operation_type_results.png`
*Only shows bars for the 5 synthetic op families (see Benchmark Classification below).*

---

## 🗂️ Benchmark Classification

Classification is driven by [scripts/plots/benchmark_families.json](file:///scratch/mb10856/MLIR-RL/scripts/plots/benchmark_families.json).
Each benchmark name maps to exactly one family — there is no overlap between model families and op types.

**Model families** (`--filter-type models_only`):
`albert`, `bart`, `bert`, `convnext_tiny`, `distilbert`, `efficientnet_b0`, `gat`, `gin`, `gpt2`, `llama3_2_1b`, `mobilenet_v3_small`, `resnet50`, `resnext50`, `t5`, `vgg16`, `vit_b_16`, `whisper_base`, `yolov8m`

**Operation types** (`--filter-type ops_only`):
`add`, `conv_2d`, `matmul`, `pooling`, `relu`

**Unclassified** (`unknown`): `bench_N` style benchmarks from the `new_dataset` baseline — not used in ops_and_blocks plots.

> If a new benchmark family is added to a dataset, update `scripts/plots/benchmark_families.json` by re-running the generation script.

---

## 🎨 Step 3. Customize Aesthetics

Edit the `USER-CUSTOMIZABLE PLOTTING PARAMETERS` block at the top of [scripts/plots/generate_plots.py](file:///scratch/mb10856/MLIR-RL/scripts/plots/generate_plots.py):
- `AGENT_COLORS`: color per agent display name
- `FONT_SETTINGS`: title, label, tick, legend font sizes
- `LINE_STYLE`: line width, marker, markersize, grid alpha

### V5 paper palette (required)

Use the Okabe–Ito colorblind-safe palette consistently in V5 figures:

| Experiment | Color | Hex |
|---|---|---|
| `v5` (`v5_distributed` or `v5_legacy_paper` on disk) | deep academic blue | `#0072B2` |
| `v5_no_transformer` | vermilion | `#D55E00` |

These are configured in `AGENT_COLORS`; do not substitute the Matplotlib default blue.

`legacy_paper` contains operation benchmarks only. Generate its evolution and
operation-type charts; do not generate model-family or no-LLaMA charts when
`best_checkpoint_benchmark_family_results.csv` has no data rows.

Override titles and paths at runtime:
- `--title "My Title"` — custom plot title
- `--csv path/to/file.csv` — direct CSV path override (legacy aggregate CSVs)
- `--png path/to/file.png` — direct PNG path override

---

## 📊 Step 4. Present Results

After running, provide the user with:
1. Clickable links to the generated PNGs in `<out-dir>/` (CSVs live in each experiment's `results/<exp>_agent/csvs/`, not in the plot output)
2. A brief summary table: agent | best checkpoint | geo-mean speedup

