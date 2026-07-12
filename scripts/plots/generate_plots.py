#!/usr/bin/env python3
"""MLIR-RL Experimentation Plot Generator.

Generates:
  1. Line charts showing evolution of agent speedups across checkpoints (mode: evolution).
  2. Grouped bar charts: model families only, model families without LLaMA,
     or operation types only (mode: comparison).

Automatically creates output CSV files if they do not exist.

Usage examples:
  # Checkpoint evolution line chart
  python scripts/plots/generate_plots.py -d ops_and_blocks \
      -a paper_original paper_transformer_small paper_transformer_large \
      -m evolution --out-dir plots/experimentation_plots/exp1

  # Model family bar chart (no op benchmarks)
  python scripts/plots/generate_plots.py -d ops_and_blocks \
      -a paper_original paper_transformer_small paper_transformer_large \
      -m comparison --filter-type models_only \
      --out-dir plots/experimentation_plots/exp1

  # Same but without LLaMA
  python scripts/plots/generate_plots.py -d ops_and_blocks \
      -a paper_original paper_transformer_small paper_transformer_large \
      -m comparison --filter-type models_only --exclude llama3_2_1b \
      --out-dir plots/experimentation_plots/exp1

  # Operation type bar chart (only add/conv_2d/matmul/pooling/relu benchmarks)
  python scripts/plots/generate_plots.py -d ops_and_blocks \
      -a paper_original paper_transformer_small paper_transformer_large \
      -m comparison --filter-type ops_only \
      --out-dir plots/experimentation_plots/exp1
"""

import os
import sys
import json
import math
import csv
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# USER-CUSTOMIZABLE PLOTTING PARAMETERS  (Edit these to style your plots!)
# ==============================================================================
FONT_SETTINGS = {
    "title_size": 16,
    "label_size": 13,
    "tick_size": 11,
    "legend_size": 11,
    "title_weight": "bold",
    "label_weight": "bold",
}

# Color palette mapping for agents (used consistently in line and bar charts)
AGENT_COLORS = {
    "V0":                 "#2563eb",   # Royal Blue
    "V4.6":               "#16a34a",   # Forest Green
    "V4.7":               "#db2777",   # Hot Pink
    "V4.8":               "#ea580c",   # Deep Orange
    "V4.9-S":             "#0d9488",   # Teal
    "V4.9-L":             "#dc2626",   # Red
    "paper_original":     "#e11d48",   # Crimson Red
    "paper_tf_small":     "#2ea42e",   # Balanced Green
    "paper_tf_large":     "#4f46e5",   # Indigo Blue
}

# Fallback colors for agents not listed above
FALLBACK_COLORS = [
    "#2563eb", "#16a34a", "#db2777", "#ea580c",
    "#0d9488", "#dc2626", "#4f46e5", "#f59e0b", "#7c3aed",
]

LINE_STYLE = {
    "linewidth": 1.75,
    "marker": "o",
    "markersize": 4,
    "grid_alpha": 0.3,
}
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to the ground-truth benchmark → family mapping JSON
BENCHMARK_FAMILIES_JSON = os.path.join(PROJECT_ROOT, "scripts", "plots", "benchmark_families.json")

# Operation-type family names (as stored in benchmark_families.json)
OP_FAMILIES = {"add", "conv_2d", "matmul", "pooling", "relu"}

DATASET_DIRS = {
    "new":          "results/new_dataset_results",
    "single_ops":   "results/single_ops_dataset_results",
    "ops_and_blocks": "results/ops_and_blocks_results",
}

DATASET_BASELINES = {
    "new":          "results/new_dataset_results/baselines/mlir/eval_base.json",
    "single_ops":   "results/single_ops_dataset_results/baselines/mlir/base_eval.json",
    "ops_and_blocks": "results/ops_and_blocks_results/baselines/mlir/base_eval.json",
}

AGENT_DISPLAY_NAMES = {
    "v0":                      "V0",
    "v4_5":                    "V4.5",
    "v4_6":                    "V4.6",
    "v4_7":                    "V4.7",
    "v4_8":                    "V4.8",
    "v4_9_small":              "V4.9-S",
    "v4_9_large":              "V4.9-L",
    "paper_original":          "paper_original",
    "paper_transformer_small": "paper_tf_small",
    "paper_transformer_large": "paper_tf_large",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_baseline(dataset: str) -> dict:
    path = os.path.join(PROJECT_ROOT, DATASET_BASELINES[dataset])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_benchmark_families() -> dict:
    """Load the ground-truth benchmark → family mapping."""
    if not os.path.isfile(BENCHMARK_FAMILIES_JSON):
        raise FileNotFoundError(
            f"Benchmark family mapping not found: {BENCHMARK_FAMILIES_JSON}\n"
            "Run: python scripts/plots/build_benchmark_families.py"
        )
    with open(BENCHMARK_FAMILIES_JSON) as f:
        return json.load(f)


def get_benchmark_family(bench_name: str, families: dict) -> str:
    """Return the benchmark family from the mapping; fall back to 'unknown'."""
    return families.get(bench_name, "unknown")


def get_checkpoint_files(eval_dir: str) -> list[str]:
    """Return checkpoint_*.json files at multiples of 100, sorted ascending."""
    if not os.path.isdir(eval_dir):
        return []
    files = []
    for f in os.listdir(eval_dir):
        m = re.search(r"checkpoint_(\d+)\.json", f)
        if m and int(m.group(1)) % 100 == 0:
            files.append(f)
    files.sort(key=lambda x: int(re.search(r"checkpoint_(\d+)\.json", x).group(1)))
    return files


def compute_geo_mean_speedup(eval_data: dict, baseline: dict) -> float | None:
    speedups = [
        baseline[b] / eval_data[b]
        for b in eval_data
        if baseline.get(b, 0) > 0 and eval_data[b] is not None and eval_data[b] > 0
    ]
    if not speedups:
        return None
    return math.exp(sum(math.log(s) for s in speedups) / len(speedups))


def find_best_checkpoint(agent_dir: str, baseline: dict) -> int | None:
    eval_dir = os.path.join(agent_dir, "eval")
    best_ckpt, best_gm = None, -1.0
    for cf in get_checkpoint_files(eval_dir):
        ckpt_num = int(re.search(r"checkpoint_(\d+)\.json", cf).group(1))
        try:
            with open(os.path.join(eval_dir, cf)) as f:
                eval_data = json.load(f)
        except Exception:
            continue
        gm = compute_geo_mean_speedup(eval_data, baseline)
        if gm is not None and gm > best_gm:
            best_gm, best_ckpt = gm, ckpt_num
    return best_ckpt


def next_exp_dir() -> str:
    """Auto-detect the next available exp<N> directory name."""
    base = os.path.join(PROJECT_ROOT, "plots", "experimentation_plots")
    n = 1
    while os.path.exists(os.path.join(base, f"exp{n}")):
        n += 1
    return f"plots/experimentation_plots/exp{n}"


# ── CSV generators ────────────────────────────────────────────────────────────

def generate_evolution_csv(dataset: str, agents: list, baseline: dict, csv_path: str):
    base_dir = os.path.join(PROJECT_ROOT, DATASET_DIRS[dataset])
    print(f"Generating evolution CSV -> {csv_path}")
    rows = []
    for agent in agents:
        eval_dir = os.path.join(base_dir, f"{agent}_agent", "eval")
        ckpt_files = get_checkpoint_files(eval_dir)
        display = AGENT_DISPLAY_NAMES.get(agent, agent)
        print(f"  {agent}: {len(ckpt_files)} checkpoints...")
        for cf in ckpt_files:
            ckpt_num = int(re.search(r"checkpoint_(\d+)\.json", cf).group(1))
            try:
                with open(os.path.join(eval_dir, cf)) as f:
                    eval_data = json.load(f)
            except Exception:
                continue
            gm = compute_geo_mean_speedup(eval_data, baseline)
            if gm is not None:
                rows.append([display, ckpt_num, f"{gm:.4f}"])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_version", "checkpoint", "speedup"])
        writer.writerows(rows)
    print(f"  CSV written: {csv_path}")


def generate_comparison_csv(
    dataset: str,
    agents: list,
    baseline: dict,
    csv_path: str,
    filter_type: str,   # "models_only" | "ops_only"
    exclude: list,      # list of family names to exclude
):
    """Build a CSV of geo-mean speedup grouped by benchmark family.

    filter_type:
      - "models_only": only model-family benchmarks (no op-type benchmarks)
      - "ops_only":    only the 5 op-type families (add/conv_2d/matmul/pooling/relu)
    """
    base_dir = os.path.join(PROJECT_ROOT, DATASET_DIRS[dataset])
    families_map = load_benchmark_families()
    exclude_lower = {e.lower() for e in exclude}
    label = "ops" if filter_type == "ops_only" else "models"
    print(f"Generating comparison CSV -> {csv_path}  [filter={filter_type}]")

    rows = []
    for agent in agents:
        agent_dir = os.path.join(base_dir, f"{agent}_agent")
        best_ckpt = find_best_checkpoint(agent_dir, baseline)
        if best_ckpt is None:
            print(f"  Warning: no checkpoints for {agent}. Skipping.")
            continue
        print(f"  {agent}: best checkpoint = {best_ckpt}")
        with open(os.path.join(agent_dir, "eval", f"checkpoint_{best_ckpt}.json")) as f:
            eval_data = json.load(f)

        family_speedups: dict[str, list] = {}
        for bench_name, opt_ns in eval_data.items():
            fam = get_benchmark_family(bench_name, families_map)
            if fam == "unknown":
                continue

            # Apply filter
            is_op = fam in OP_FAMILIES
            if filter_type == "models_only" and is_op:
                continue
            if filter_type == "ops_only" and not is_op:
                continue

            # Apply exclude list (partial match)
            if any(ex in fam.lower() for ex in exclude_lower):
                continue

            base_ns = baseline.get(bench_name, 0)
            if base_ns > 0 and opt_ns is not None and opt_ns > 0:
                family_speedups.setdefault(fam, []).append(base_ns / opt_ns)

        display = AGENT_DISPLAY_NAMES.get(agent, agent)
        for fam, speedups in family_speedups.items():
            if len(speedups) >= 1:
                gm = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
                rows.append([fam, display, f"{gm:.4f}"])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark_family", "agent_version", "speedup"])
        writer.writerows(rows)
    print(f"  CSV written: {csv_path}")


# ── Plot renderers ────────────────────────────────────────────────────────────

def plot_evolution(csv_path: str, png_path: str, custom_title: str = None):
    df = pd.read_csv(csv_path)
    agents = df["agent_version"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, agent in enumerate(agents):
        adf = df[df["agent_version"] == agent].sort_values("checkpoint")
        color = AGENT_COLORS.get(agent, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        ax.plot(
            adf["checkpoint"], adf["speedup"],
            label=agent, color=color,
            linewidth=LINE_STYLE["linewidth"],
            marker=LINE_STYLE["marker"],
            markersize=LINE_STYLE["markersize"],
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="MLIR Baseline (1.0×)")
    title = custom_title or "Agent Performance Evolution Across Checkpoints"
    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.set_xlabel("Training Iteration", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Geometric Mean Speedup (×)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], frameon=True)
    ax.grid(alpha=LINE_STYLE["grid_alpha"], linestyle="--")
    ax.tick_params(axis="both", labelsize=FONT_SETTINGS["tick_size"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")


def plot_comparison(
    csv_path: str,
    png_path: str,
    custom_title: str = None,
    exclude: list = None,
):
    df = pd.read_csv(csv_path)
    if exclude:
        ex_lower = [e.lower() for e in exclude]
        df = df[~df["benchmark_family"].str.lower().apply(
            lambda x: any(e in x for e in ex_lower)
        )]

    families = sorted(df["benchmark_family"].unique())
    agents   = sorted(df["agent_version"].unique())
    n_agents = len(agents)

    fig, ax = plt.subplots(figsize=(max(12, len(families) * 1.4), 7))
    width = 0.8 / n_agents
    x = np.arange(len(families))

    for i, agent in enumerate(agents):
        adf = df[df["agent_version"] == agent].set_index("benchmark_family")
        vals = [adf.loc[fam, "speedup"] if fam in adf.index else float("nan") for fam in families]
        color = AGENT_COLORS.get(agent, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        offset = (i - (n_agents - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=color, label=agent, zorder=3)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2, label="Baseline 1.0×")
    ax.set_xticks(x)
    display_names = [fam.replace("_", " ").title() for fam in families]
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=FONT_SETTINGS["tick_size"])

    title = custom_title or "Benchmark Family Speedup Comparison (Best Checkpoint)"
    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.set_xlabel("Benchmark Family", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Geometric Mean Speedup (×)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper right", frameon=True)
    ax.tick_params(axis="y", labelsize=FONT_SETTINGS["tick_size"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLIR-RL Plot Generator")
    parser.add_argument("-d", "--dataset",
                        choices=["new", "single_ops", "ops_and_blocks"], required=True)
    parser.add_argument("-a", "--agents", nargs="+", required=True,
                        help="Agent directory prefixes (e.g. paper_original paper_transformer_small)")
    parser.add_argument("-m", "--mode",
                        choices=["evolution", "comparison"], required=True,
                        help="'evolution' = line chart  |  'comparison' = grouped bar chart")
    parser.add_argument("--filter-type",
                        choices=["models_only", "ops_only"], default="models_only",
                        help="[comparison] 'models_only' = only model-family benchmarks  |  "
                             "'ops_only' = only op-type benchmarks (add/conv_2d/matmul/pooling/relu)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="[comparison] Benchmark families to exclude (e.g. llama3_2_1b)")
    parser.add_argument("--out-dir",
                        help="Output directory for CSVs and PNGs "
                             f"(default: next auto-incremented exp<N> under plots/experimentation_plots/)")
    parser.add_argument("--csv",  help="Override CSV path directly")
    parser.add_argument("--png",  help="Override PNG path directly")
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--force-csv", action="store_true",
                        help="Rebuild CSV even if it already exists")
    args = parser.parse_args()

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.out_dir:
        out_dir = (args.out_dir if os.path.isabs(args.out_dir)
                   else os.path.join(PROJECT_ROOT, args.out_dir))
    else:
        out_dir = os.path.join(PROJECT_ROOT, next_exp_dir())

    # ── Resolve file names based on mode / filter ─────────────────────────────
    if args.mode == "evolution":
        default_stem = "checkpoint_evolution"
    elif args.filter_type == "ops_only":
        default_stem = "operation_type_results"
    elif args.exclude:
        # "no_<family>" suffix for common exclusions
        excl_tag = "_no_" + "_no_".join(e.split("_")[0] for e in args.exclude)
        default_stem = f"best_checkpoint_results{excl_tag}"
    else:
        default_stem = "best_checkpoint_results"

    csv_path = (args.csv if args.csv else os.path.join(out_dir, "csvs", f"{default_stem}.csv"))
    png_path = (args.png if args.png else os.path.join(out_dir, "pngs", f"{default_stem}.png"))

    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)
    if not os.path.isabs(png_path):
        png_path = os.path.join(PROJECT_ROOT, png_path)

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline = load_baseline(args.dataset)

    # ── Generate CSV if missing (or forced) ───────────────────────────────────
    if not os.path.isfile(csv_path) or args.force_csv:
        if args.mode == "evolution":
            generate_evolution_csv(args.dataset, args.agents, baseline, csv_path)
        else:
            generate_comparison_csv(
                args.dataset, args.agents, baseline, csv_path,
                args.filter_type, args.exclude,
            )
    else:
        print(f"Using existing CSV: {csv_path}")

    # ── Render plot ───────────────────────────────────────────────────────────
    if args.mode == "evolution":
        plot_evolution(csv_path, png_path, args.title)
    else:
        # exclude is already applied at CSV generation; pass [] here
        plot_comparison(csv_path, png_path, args.title, [])


if __name__ == "__main__":
    main()
