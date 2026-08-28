#!/usr/bin/env python3
"""MLIR-RL plot generator — READS result CSVs, writes only images (plots/).

CSV generation lives in scripts/utils/csvs.py (per-agent, inside
results/<agent>_agent/eval/). This script consumes those CSVs and renders PNGs.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIRS = {
    "new": "results/new_dataset_results",
    "single_ops": "results/single_ops_dataset_results",
    "ops_and_blocks": "results/ops_and_blocks_results",
    "legacy_paper": "results/legacy_paper_results",
}

FONT_SETTINGS = {
    "title_size": 16,
    "label_size": 13,
    "tick_size": 11,
    "legend_size": 11,
    "title_weight": "bold",
    "label_weight": "bold",
}

AGENT_COLORS = {
    "paper_original": "#4C72B0",
    "paper_transformer_large": "#DD8452",
    "paper_transformer_small": "#55A868",
    "v5_single_node": "#C44E52",
    "v5_distributed": "#0072B2",
    "v5_legacy_paper": "#0072B2",
    "v5_no_transformer": "#D55E00",
}

FALLBACK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

LINE_STYLE = {
    "linewidth": 2.2,
    "marker": "o",
    "markersize": 7,
}

AGENT_DISPLAY_NAMES = {
    "paper_original": "paper_original",
    "paper_transformer_small": "paper_tf_small",
    "paper_transformer_large": "paper_tf_large",
    "v5_single_node": "v5_single_node",
    "v5_distributed": "v5",
    "v5_legacy_paper": "v5",
}

EVOLUTION_CSV = "checkpoint_speedups.csv"
COMPARISON_CSV = "best_checkpoint_benchmark_family_results.csv"
OPS_CSV = "best_checkpoint_operation_type_results.csv"


def get_benchmark_family(bench_name: str, families: dict) -> str:
    for fam, members in families.items():
        if bench_name in members:
            return fam
    return "unknown"


def next_exp_dir(dataset: str, agents: list) -> str:
    name = "_".join(agents)
    return os.path.join("plots", "experimentation_plots", dataset, name)


def load_agent_csv(dataset: str, agent: str, stem: str, csv_override: str | None) -> pd.DataFrame:
    """Read one agent's per-agent CSV from its results dir; --csv overrides (legacy)."""
    if csv_override:
        path = csv_override if os.path.isabs(csv_override) else os.path.join(PROJECT_ROOT, csv_override)
        if not os.path.isfile(path):
            sys.exit(f"ERROR: CSV not found: {path}")
        return pd.read_csv(path)
    path = os.path.join(PROJECT_ROOT, DATASET_DIRS[dataset], f"{agent}_agent", "csvs", stem)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: no {stem} for agent '{agent}' — generate it first:\n"
                 f"  python utils/csvs.py --results-dir results/.../{agent}_agent "
                 f"--agent {agent} [--all]\n  (expected: {path})")
    df = pd.read_csv(path)
    # Per-experiment CSVs carry no agent_version column — tag rows by source agent
    if "agent_version" not in df.columns:
        df["agent_version"] = agent
    return df


def plot_evolution(df: pd.DataFrame, png_path: str, custom_title: str = None):
    agents = df["agent_version"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, agent in enumerate(agents):
        adf = df[df["agent_version"] == agent].sort_values("checkpoint")
        color = AGENT_COLORS.get(agent, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        ax.plot(
            adf["checkpoint"], adf["speedup"],
            label=AGENT_DISPLAY_NAMES.get(agent, agent), color=color,
            linewidth=LINE_STYLE["linewidth"],
            marker=LINE_STYLE["marker"],
            markersize=LINE_STYLE["markersize"],
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="MLIR Baseline (1.0×)")
    title = custom_title or "Agent Performance Evolution Across Checkpoints"
    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.set_xlabel("Training Iteration", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Geometric Mean Speedup (×)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper left", frameon=True)
    ax.tick_params(axis="x", labelsize=FONT_SETTINGS["tick_size"])
    ax.tick_params(axis="y", labelsize=FONT_SETTINGS["tick_size"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")


def plot_comparison(
    df: pd.DataFrame,
    png_path: str,
    custom_title: str = None,
    exclude: list = None,
):
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

    for i, agent in enumerate(agents):
        adf = df[df["agent_version"] == agent].set_index("benchmark_family")
        color = AGENT_COLORS.get(agent, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        x = np.arange(len(families)) + (i - (n_agents - 1) / 2) * width
        ax.bar(
            x, [adf.loc[f, "speedup"] if f in adf.index else 0.0 for f in families],
            width=width, label=AGENT_DISPLAY_NAMES.get(agent, agent), color=color, edgecolor="white", linewidth=0.5,
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="MLIR Baseline (1.0×)")
    title = custom_title or "Best Checkpoint Speedup by Benchmark Family"
    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.set_xlabel("Benchmark Family", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Geometric Mean Speedup (×)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=FONT_SETTINGS["tick_size"])
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


def main():
    parser = argparse.ArgumentParser(description="MLIR-RL Plot Generator (reads CSVs, writes PNGs only)")
    parser.add_argument("-d", "--dataset",
                        choices=["new", "single_ops", "ops_and_blocks", "legacy_paper"], required=True)
    parser.add_argument("-a", "--agents", nargs="+", required=True,
                        help="Agent directory prefixes (e.g. paper_original paper_transformer_small)")
    parser.add_argument("-m", "--mode",
                        choices=["evolution", "comparison"], required=True,
                        help="'evolution' = line chart  |  'comparison' = grouped bar chart")
    parser.add_argument("--filter-type",
                        choices=["models_only", "ops_only"], default="models_only",
                        help="[comparison] 'models_only' = best_checkpoint_results.csv  |  "
                             "'ops_only' = operation_type_results.csv")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="[comparison] Benchmark families to exclude (e.g. llama3_2_1b)")
    parser.add_argument("--out-dir",
                        help="Output directory for PNGs "
                             f"(default: next auto-incremented exp<N> under plots/experimentation_plots/)")
    parser.add_argument("--csv", help="Read a single (legacy aggregate) CSV instead of per-agent CSVs")
    parser.add_argument("--png", help="Override PNG path directly")
    parser.add_argument("--title", help="Custom plot title")
    args = parser.parse_args()

    # ── Resolve output directory (images only) ────────────────────────────────
    if args.out_dir:
        out_dir = (args.out_dir if os.path.isabs(args.out_dir)
                   else os.path.join(PROJECT_ROOT, args.out_dir))
    else:
        out_dir = os.path.join(PROJECT_ROOT, next_exp_dir(args.dataset, args.agents))

    # ── Resolve file names based on mode / filter ─────────────────────────────
    if args.mode == "evolution":
        stem = EVOLUTION_CSV
        png_stem = "checkpoint_evolution"
    elif args.filter_type == "ops_only":
        stem = OPS_CSV
        png_stem = "best_checkpoint_operation_type_results"
    else:
        stem = COMPARISON_CSV
        excl_tag = ("_no_" + "_no_".join(e.split("_")[0] for e in args.exclude)) if args.exclude else ""
        png_stem = f"best_checkpoint_benchmark_family_results{excl_tag}"

    png_path = (args.png if args.png else os.path.join(out_dir, f"{png_stem}.png"))
    if not os.path.isabs(png_path):
        png_path = os.path.join(PROJECT_ROOT, png_path)

    # ── Load per-agent CSVs (plots only read; csvs.py generates) ─────────────
    frames = [load_agent_csv(args.dataset, agent, stem, args.csv) for agent in args.agents]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if args.mode == "evolution":
        plot_evolution(df, png_path, args.title)
    else:
        plot_comparison(df, png_path, args.title, args.exclude)


if __name__ == "__main__":
    main()
