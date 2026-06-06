#!/usr/bin/env python3
"""
Generate PNG bar charts from graph CSVs.
"""

import csv, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = "plots"
G2_DIR = os.path.join(OUT_DIR, "multi_hardware", "graphs")
G3_DIR = os.path.join(OUT_DIR, "full_model_comparison", "graphs_tayeb")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

COLORS = {"Prior Agent": "#d62728", "Our Agent": "#2ca02c",
          "Our Agent (with HW-F)": "#2ca02c",
          "Our Agent (without HW-F)": "#ff7f0e"}
BASELINE_COLOR = "#7f7f7f"

AGENT_RENAME = {"No-Reward (HW)": "Our Agent (with HW-F)",
                "No-Reward (No-HW)": "Our Agent (without HW-F)"}

CLUSTER_DISPLAY = {
    "bergamo": "Bergamo (AMD EPYC 9754, 256c — train cluster)",
    "dalma": "Dalma (Intel Xeon E5-2680 v4, 28c)",
    "jubail": "Jubail (AMD EPYC 7742, 128c)",
}


def make_bar(ax, labels, groups, width=0.35, baseline_label="Baseline 1.0x"):
    """Grouped bar chart. groups = {label: [values]}."""
    x = np.arange(len(labels))
    n = len(groups)
    w = width / n
    for i, (name, vals) in enumerate(groups.items()):
        offset = (i - (n - 1) / 2) * w
        color = COLORS.get(name, "#1f77b4")
        ax.bar(x + offset, vals, w, label=name, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(y=1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1, label=baseline_label)


# ============================================================
# GRAPH 1: Overall Performance
# ============================================================
def plot_graph1():
    rows = []
    G1_DIR = os.path.join(OUT_DIR, "version comparison", "v0_vs_v45_no_reward_tayeb")
    with open(os.path.join(G1_DIR, "graph1_performance.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    labels = [r["op_type"].replace("generic", "relu") for r in rows]
    v0_geo = [float(r["V0_geo_mean"]) if r["V0_geo_mean"] else 0 for r in rows]
    nw_geo = [float(r["NoReward_geo_mean"]) if r["NoReward_geo_mean"] else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    make_bar(ax, labels, {"Prior Agent": v0_geo, "Our Agent": nw_geo},
             baseline_label="MLIR Baseline (1.0x)")
    ax.set_ylabel("Geometric Mean Speedup")
    ax.set_title("Prior Agent vs Our Agent (by operation type)")
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))
    fig.tight_layout()
    fig.savefig(os.path.join(G1_DIR, "graph1_performance.png"))
    plt.close(fig)
    print("graph1_performance.png")


# ============================================================
# GRAPH 2: Multi-Hardware (combined 2-subplot figures)
# ============================================================
def _load_graph2_csv(fname):
    rows = []
    path = os.path.join(G2_DIR, fname)
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None, None, None
    agents = sorted(set(r["agent"] for r in rows))
    if len(agents) < 2:
        return None, None, None
    return rows, agents, fname


def _build_graph2_data(rows, agents, filter_out=None, rename_op=None):
    """Extract grouped bars from rows. Returns (labels, {agent: [values]})."""
    groups_ordered = []
    seen = set()
    for r in rows:
        g = r["group"]
        if filter_out and g in filter_out:
            continue
        if rename_op and g in rename_op:
            g = rename_op[g]
        if g not in seen:
            groups_ordered.append(g)
            seen.add(g)

    agent_data = {}
    for agent_raw in agents:
        agent = AGENT_RENAME.get(agent_raw, agent_raw)
        d = {}
        for r in rows:
            g = r["group"]
            if filter_out and g in filter_out:
                continue
            if rename_op and g in rename_op:
                g = rename_op[g]
            if r["agent"] == agent_raw:
                d[g] = float(r["geo_mean"]) if r["geo_mean"] else 0
        agent_data[agent] = [d.get(g, 0) for g in groups_ordered]
    return groups_ordered, agent_data


def plot_graph2():
    os.makedirs(G2_DIR, exist_ok=True)

    for cluster in ["bergamo", "dalma", "jubail"]:
        op_rows, op_agents, _ = _load_graph2_csv(f"graph2_{cluster}_optype.csv")
        model_rows, model_agents, _ = _load_graph2_csv(f"graph2_{cluster}_model.csv")

        if not op_rows or not model_rows:
            print(f"  Skip {cluster}: missing data")
            continue

        fig, (ax_op, ax_model) = plt.subplots(2, 1, figsize=(14, 8))

        # Top: by op_type (exclude block, rename generic → relu)
        op_labels, op_data = _build_graph2_data(op_rows, op_agents,
                                                 filter_out={"block"},
                                                 rename_op={"generic": "relu"})
        make_bar(ax_op, op_labels, op_data, baseline_label="MLIR Baseline (1.0x)")
        ax_op.set_ylabel("Geometric Mean Speedup")
        ax_op.set_title(f"By Operation Type")
        max_op = max(v for vals in op_data.values() for v in vals if v)
        if max_op > 10:
            ax_op.set_yscale("log")
        ax_op.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))

        # Bottom: by model (exclude synthetic)
        model_labels, model_data = _build_graph2_data(model_rows, model_agents,
                                                       filter_out={"synthetic"})
        make_bar(ax_model, model_labels, model_data, baseline_label="MLIR Baseline (1.0x)")
        ax_model.set_ylabel("Geometric Mean Speedup")
        ax_model.set_title(f"By Model")
        max_md = max(v for vals in model_data.values() for v in vals if v)
        if max_md > 10:
            ax_model.set_yscale("log")
        ax_model.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))

        # Shared figure title
        fig.suptitle(CLUSTER_DISPLAY.get(cluster, cluster), fontsize=14, fontweight="bold")

        # Single legend at figure bottom
        handles, labels = ax_op.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 0.02))

        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        out = os.path.join(G2_DIR, f"graph2_{cluster}.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  {out}")


# ============================================================
# GRAPH 3: Full Model Support
# ============================================================
def plot_graph3():
    rows = []
    path = os.path.join(G3_DIR, "graph3_fullmodel.csv")
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    labels = [r["model"] for r in rows]
    v0_spd = [float(r["V0_total_speedup"]) if r["V0_total_speedup"] else 0 for r in rows]
    nw_spd = [float(r["No-Reward_total_speedup"]) if r["No-Reward_total_speedup"] else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    make_bar(ax, labels, {"Prior Agent": v0_spd, "Our Agent": nw_spd}, width=0.4,
             baseline_label="MLIR Baseline (1.0x)")
    ax.set_ylabel("Total Model Speedup")
    ax.set_title("Full Model Speedup (Prior Agent vs Our Agent)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(G3_DIR, exist_ok=True)
    fig.savefig(os.path.join(G3_DIR, "graph3_fullmodel.png"))
    plt.close(fig)
    print("graph3_fullmodel.png")


if __name__ == "__main__":
    os.makedirs(G2_DIR, exist_ok=True)
    plot_graph1()
    plot_graph2()
    plot_graph3()
    print("\nAll plots done.")
