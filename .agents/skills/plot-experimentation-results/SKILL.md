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

---

## 🎯 Step 1. Ask the User (Interactive Menu)

Use the `ask_question` tool with `is_multi_select: false` for each item:

1. **Dataset**: suggest `ops_and_blocks` | `new` | `single_ops`
2. **Agent versions**: suggest the set of paper or ablation agents based on the dataset
3. **Output folder**: suggest numbered paths based on the base directory `plots/experimentation_plots/`.
   - Auto-detect the next unused `exp<N>` directory by checking which ones already exist.
   - Suggestions should be absolute-style paths like:
     - `plots/experimentation_plots/exp1` (next available)
     - `plots/experimentation_plots/exp2`
   - Let the user write a custom path if none of the suggestions suit them.
4. **Refresh CSVs first?** If new eval results exist since the last plot, regenerate the
   experiments' CSVs before plotting:
   ```bash
   python utils/csvs.py --results-dir results/<dataset>_results/<exp>_agent --agent <exp> [--all]
   ```
   (eval.py already auto-updates `checkpoint_speedups.csv` after each eval — this is only
   needed when the CSVs are missing or the comparison CSVs are stale.)

---

## 🛠️ Step 2. Generate the Plots

Run four plots per experimentation session into the chosen `<out-dir>` using `--out-dir=<path>` (use `=` to avoid argparse ambiguity). PNGs land in `<out-dir>/pngs/`; no CSVs are written by this step:

### 1. Checkpoint Evolution Line Chart
```bash
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
python scripts/plots/generate_plots.py \
  -d <dataset> -m evolution --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/pngs/checkpoint_evolution.png` (reads `results/<exp>_agent/csvs/checkpoint_speedups.csv`)
*Only multiples-of-100 checkpoints are included for a smooth curve.*

### 2. Model Family Comparison (all model families)
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type models_only --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/pngs/best_checkpoint_benchmark_family_results.png`
*Uses best checkpoint per agent (highest overall geo-mean). Excludes all op-type benchmarks.*

### 3. Model Family Comparison (without LLaMA)
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type models_only --exclude llama3_2_1b --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/pngs/best_checkpoint_benchmark_family_results_no_llama3.png`

### 4. Operation Type Comparison
```bash
python scripts/plots/generate_plots.py \
  -d <dataset> -m comparison --filter-type ops_only --out-dir=<out-dir> \
  -a <agent1> <agent2> ...
```
→ Saves: `<out-dir>/pngs/best_checkpoint_operation_type_results.png`
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

Override titles and paths at runtime:
- `--title "My Title"` — custom plot title
- `--csv path/to/file.csv` — direct CSV path override (legacy aggregate CSVs)
- `--png path/to/file.png` — direct PNG path override

---

## 📋 Step 4. Generate the Experimentation Report

After all plots are generated, run the report script to produce `experimentation_report.md` in the experiment directory (at the same level as `csvs/` and `pngs/`):

```bash
source ~/envs/mlir/bin/activate && set -a && source .env && set +a
python scripts/plots/generate_report.py \
  --exp-dir=<out-dir> \
  -d <dataset> \
  -a <agent1> <agent2> ...
```

The report includes:
- **Best checkpoint summary**: ranked table with peak geo-mean speedup per agent
- **Per-agent detailed stats**: valid benchmarks, failed count, geo-mean, arith-mean, best/worst speedup
- **Model family performance table**: all families × all agents
- **Model family table excluding LLaMA** (if the no-LLaMA CSV exists)
- **Operation type performance table**: add, conv_2d, matmul, pooling, relu
- **Top-5 individual benchmarks**: model benchmarks and op-type benchmarks per agent
- **Plot references**: list of all generated PNGs

---

## 📊 Step 5. Present Results

After running, provide the user with:
1. Clickable links to each generated PNG in `<out-dir>/pngs/` (CSVs live in each experiment's `results/<exp>_agent/csvs/`, not in the plot output)
2. Clickable link to `<out-dir>/experimentation_report.md`
3. A brief summary table: agent | best checkpoint | geo-mean speedup

