#!/usr/bin/env python3
"""Generate a markdown experimentation report from an experiment directory.

Reads CSVs in <exp-dir>/csvs/ and the underlying eval JSON files to produce
<exp-dir>/experimentation_report.md with key statistics and tables.

Usage:
  python scripts/plots/generate_report.py --exp-dir plots/experimentation_plots/exp1 \
      -d ops_and_blocks -a paper_original paper_transformer_small paper_transformer_large
"""

import os
import sys
import json
import math
import argparse
import re
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIRS = {
    "new":            "results/new_dataset_results",
    "single_ops":     "results/single_ops_dataset_results",
    "ops_and_blocks": "results/ops_and_blocks_results",
}

DATASET_BASELINES = {
    "new":            "results/new_dataset_results/baselines/mlir/eval_base.json",
    "single_ops":     "results/single_ops_dataset_results/baselines/mlir/base_eval.json",
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

OP_FAMILIES = {"add", "conv_2d", "matmul", "pooling", "relu"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_baseline(dataset: str) -> dict:
    path = os.path.join(PROJECT_ROOT, DATASET_BASELINES[dataset])
    with open(path) as f:
        return json.load(f)


def load_benchmark_families() -> dict:
    path = os.path.join(PROJECT_ROOT, "scripts", "plots", "benchmark_families.json")
    with open(path) as f:
        return json.load(f)


def geo_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_eval(agent_dir: str, checkpoint: int) -> dict:
    path = os.path.join(agent_dir, "eval", f"checkpoint_{checkpoint}.json")
    with open(path) as f:
        return json.load(f)


def compute_speedups(eval_data: dict, baseline: dict) -> tuple[list, int, int]:
    """Returns (speedup_list, failed_count, total_count)."""
    speedups, failed, total = [], 0, 0
    for bench, opt_ns in eval_data.items():
        base_ns = baseline.get(bench, 0)
        if base_ns <= 0:
            continue
        total += 1
        if opt_ns is None or opt_ns <= 0:
            failed += 1
        else:
            speedups.append(base_ns / opt_ns)
    return speedups, failed, total


def get_top_benchmarks(eval_data: dict, baseline: dict, families: dict,
                       n: int = 5, op_type: bool = False) -> list[dict]:
    """Return top-n benchmarks by speedup (filtered by op_type flag)."""
    results = []
    for bench, opt_ns in eval_data.items():
        base_ns = baseline.get(bench, 0)
        if base_ns <= 0 or opt_ns is None or opt_ns <= 0:
            continue
        fam = families.get(bench, "unknown")
        is_op = fam in OP_FAMILIES
        if op_type != is_op:
            continue
        results.append({"benchmark": bench, "family": fam, "speedup": base_ns / opt_ns})
    results.sort(key=lambda x: x["speedup"], reverse=True)
    return results[:n]


# ── CSV loaders ───────────────────────────────────────────────────────────────

def find_csv(csvs_dir: str, stem_patterns: list[str]) -> pd.DataFrame | None:
    """Load the first CSV whose filename exactly equals stem+'.csv' or contains the pattern."""
    for stem in stem_patterns:
        # Try exact match first
        exact = os.path.join(csvs_dir, f"{stem}.csv")
        if os.path.isfile(exact):
            df = pd.read_csv(exact)
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            return df
    return None



# ── Report builder ────────────────────────────────────────────────────────────

def build_report(exp_dir: str, dataset: str, agents: list[str]) -> str:
    csvs_dir = os.path.join(exp_dir, "csvs")
    pngs_dir = os.path.join(exp_dir, "pngs")

    baseline = load_baseline(dataset)
    families_map = load_benchmark_families()
    base_results_dir = os.path.join(PROJECT_ROOT, DATASET_DIRS[dataset])

    evo_df   = find_csv(csvs_dir, ["checkpoint_evolution"])
    model_df = find_csv(csvs_dir, ["best_checkpoint_results"])
    op_df    = find_csv(csvs_dir, ["operation_type_results"])

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Experimentation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Dataset**: `{dataset}`  ",
        f"**Agents**: {', '.join(f'`{a}`' for a in agents)}  ",
        f"**Experiment directory**: `{os.path.relpath(exp_dir, PROJECT_ROOT)}`",
        "",
    ]

    # ── Section 1: Best Checkpoint Summary ───────────────────────────────────
    lines += [
        "---",
        "",
        "## 1. Best Checkpoint Summary",
        "",
    ]

    if evo_df is not None:
        evo_df["speedup"] = pd.to_numeric(evo_df["speedup"], errors="coerce")
        evo_df["checkpoint"] = pd.to_numeric(evo_df["checkpoint"], errors="coerce")
        best_rows = evo_df.loc[evo_df.groupby("agent_version")["speedup"].idxmax()]

        lines.append("| Agent | Best Checkpoint | Peak Geo-Mean Speedup |")
        lines.append("|:------|:--------------:|----------------------:|")
        for _, row in best_rows.sort_values("speedup", ascending=False).iterrows():
            lines.append(f"| `{row['agent_version']}` | {int(row['checkpoint'])} | **{row['speedup']:.4f}×** |")
        lines.append("")

    # Detailed stats from eval JSON for each agent's best checkpoint
    lines += [
        "### Per-Agent Detailed Stats (at best checkpoint)",
        "",
        "| Agent | Best CP | Valid | Failed | Geo-Mean | Arith-Mean | Best Speedup | Worst Speedup |",
        "|:------|:-------:|------:|-------:|---------:|-----------:|-------------:|---------------:|",
    ]

    agent_best: dict[str, int] = {}  # agent_key → best checkpoint number

    for agent in agents:
        agent_dir = os.path.join(base_results_dir, f"{agent}_agent")
        display = AGENT_DISPLAY_NAMES.get(agent, agent)

        # Determine best checkpoint from evolution CSV
        if evo_df is not None and display in evo_df["agent_version"].values:
            sub = evo_df[evo_df["agent_version"] == display]
            best_ckpt = int(sub.loc[sub["speedup"].idxmax(), "checkpoint"])
        else:
            # Fall back: scan eval dir
            eval_dir = os.path.join(agent_dir, "eval")
            best_ckpt, best_gm = None, -1.0
            if os.path.isdir(eval_dir):
                for fname in os.listdir(eval_dir):
                    m = re.search(r"checkpoint_(\d+)\.json", fname)
                    if m and int(m.group(1)) % 100 == 0:
                        try:
                            ed = json.load(open(os.path.join(eval_dir, fname)))
                        except Exception:
                            continue
                        sp, _, _ = compute_speedups(ed, baseline)
                        gm = geo_mean(sp) if sp else 0.0
                        if gm > best_gm:
                            best_gm, best_ckpt = gm, int(m.group(1))

        if best_ckpt is None:
            lines.append(f"| `{display}` | — | — | — | — | — | — | — |")
            continue

        agent_best[agent] = best_ckpt
        try:
            eval_data = load_eval(agent_dir, best_ckpt)
        except FileNotFoundError:
            lines.append(f"| `{display}` | {best_ckpt} | — | — | — | — | — | — |")
            continue

        speedups, failed, total = compute_speedups(eval_data, baseline)
        if not speedups:
            lines.append(f"| `{display}` | {best_ckpt} | 0 | {failed} | — | — | — | — |")
            continue

        gm     = geo_mean(speedups)
        am     = sum(speedups) / len(speedups)
        best_s = max(speedups)
        worst_s= min(speedups)
        lines.append(
            f"| `{display}` | {best_ckpt} | {len(speedups)} | {failed} "
            f"| {gm:.4f}× | {am:.2f}× | {best_s:.2f}× | {worst_s:.4f}× |"
        )

    lines.append("")

    # ── Section 2: Model Family Performance ──────────────────────────────────
    if model_df is not None:
        model_df["speedup"] = pd.to_numeric(model_df["speedup"], errors="coerce")
        agents_in_csv = sorted(model_df["agent_version"].unique())
        families_in_csv = sorted(model_df["benchmark_family"].unique())

        lines += [
            "---",
            "",
            "## 2. Model Family Performance (Best Checkpoint)",
            "",
            "Geo-mean speedup per model family across all agents.",
            "",
        ]

        header = "| Model Family | " + " | ".join(f"`{a}`" for a in agents_in_csv) + " |"
        sep    = "|:-------------|" + "|".join(":------:" for _ in agents_in_csv) + "|"
        lines += [header, sep]

        for fam in families_in_csv:
            row_parts = [f"**{fam.replace('_', ' ').title()}**"]
            for ag in agents_in_csv:
                sub = model_df[(model_df["benchmark_family"] == fam) & (model_df["agent_version"] == ag)]
                if sub.empty:
                    row_parts.append("—")
                else:
                    row_parts.append(f"{sub['speedup'].values[0]:.4f}×")
            lines.append("| " + " | ".join(row_parts) + " |")

        # Highlight best / worst family per agent
        lines += [""]
        lines.append("**Top-3 families by geo-mean speedup (averaged across agents):**")
        lines.append("")
        avg_by_fam = (model_df.groupby("benchmark_family")["speedup"]
                      .mean().sort_values(ascending=False).head(3))
        for rank, (fam, spd) in enumerate(avg_by_fam.items(), 1):
            lines.append(f"{rank}. `{fam}` — avg geo-mean **{spd:.4f}×**")
        lines.append("")


    # ── Section 3: Operation Type Performance ────────────────────────────────
    if op_df is not None:
        op_df["speedup"] = pd.to_numeric(op_df["speedup"], errors="coerce")
        agents_in_csv = sorted(op_df["agent_version"].unique())
        op_families = sorted(op_df["benchmark_family"].unique())

        lines += [
            "---",
            "",
            "## 3. Operation Type Performance (Best Checkpoint)",
            "",
            "Only synthetic operation-type benchmarks "
            "(`add`, `conv_2d`, `matmul`, `pooling`, `relu`).",
            "",
        ]

        header = "| Operation | " + " | ".join(f"`{a}`" for a in agents_in_csv) + " |"
        sep    = "|:----------|" + "|".join(":------:" for _ in agents_in_csv) + "|"
        lines += [header, sep]

        for fam in op_families:
            row_parts = [f"**{fam.replace('_', ' ').title()}**"]
            for ag in agents_in_csv:
                sub = op_df[(op_df["benchmark_family"] == fam) & (op_df["agent_version"] == ag)]
                row_parts.append(f"{sub['speedup'].values[0]:.4f}×" if not sub.empty else "—")
            lines.append("| " + " | ".join(row_parts) + " |")
        lines.append("")

    # ── Section 4: Top Individual Benchmarks ─────────────────────────────────
    lines += [
        "---",
        "",
        "## 4. Top Individual Benchmark Performances",
        "",
        "Best individual benchmark speedups from each agent's best checkpoint.",
        "",
    ]

    for agent in agents:
        display = AGENT_DISPLAY_NAMES.get(agent, agent)
        best_ckpt = agent_best.get(agent)
        if best_ckpt is None:
            continue

        agent_dir = os.path.join(base_results_dir, f"{agent}_agent")
        try:
            eval_data = load_eval(agent_dir, best_ckpt)
        except FileNotFoundError:
            continue

        top5 = get_top_benchmarks(eval_data, baseline, families_map, n=5, op_type=False)
        top5_ops = get_top_benchmarks(eval_data, baseline, families_map, n=5, op_type=True)

        lines.append(f"### `{display}` (checkpoint {best_ckpt})")
        lines.append("")
        lines.append("**Top-5 model benchmarks:**")
        lines.append("")
        lines.append("| Rank | Benchmark | Family | Speedup |")
        lines.append("|:----:|:----------|:-------|--------:|")
        for rank, row in enumerate(top5, 1):
            lines.append(f"| {rank} | `{row['benchmark']}` | {row['family']} | {row['speedup']:.2f}× |")

        lines.append("")
        lines.append("**Top-5 operation-type benchmarks:**")
        lines.append("")
        lines.append("| Rank | Benchmark | Op Type | Speedup |")
        lines.append("|:----:|:----------|:--------|--------:|")
        for rank, row in enumerate(top5_ops, 1):
            lines.append(f"| {rank} | `{row['benchmark']}` | {row['family']} | {row['speedup']:.2f}× |")
        lines.append("")

    # ── Section 5: Plot References ────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Generated Plots",
        "",
    ]

    rel_exp = os.path.relpath(exp_dir, PROJECT_ROOT)
    for fname in sorted(os.listdir(pngs_dir)):
        if fname.endswith(".png"):
            stem = fname.replace(".png", "").replace("_", " ").title()
            lines.append(f"- **{stem}**: `{rel_exp}/pngs/{fname}`")
    lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate experimentation_report.md from CSVs and eval files")
    parser.add_argument("--exp-dir", required=True,
                        help="Path to experiment directory (e.g. plots/experimentation_plots/exp1)")
    parser.add_argument("-d", "--dataset",
                        choices=["new", "single_ops", "ops_and_blocks"], required=True)
    parser.add_argument("-a", "--agents", nargs="+", required=True,
                        help="Agent directory prefixes used in this experiment")
    args = parser.parse_args()

    exp_dir = (args.exp_dir if os.path.isabs(args.exp_dir)
               else os.path.join(PROJECT_ROOT, args.exp_dir))

    if not os.path.isdir(exp_dir):
        print(f"Error: experiment directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    report = build_report(exp_dir, args.dataset, args.agents)

    out_path = os.path.join(exp_dir, "experimentation_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
