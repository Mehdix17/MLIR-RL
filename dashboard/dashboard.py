"""
MLIR-RL Evaluation Dashboard — Multi-page Streamlit app.

Run:
    streamlit run dashboard/dashboard.py --server.fileWatcherType none

Data is read from dashboard/data/ (populated from plots/).
"""

import sys
from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_version_comparison_grouped,
    load_ablation_real,
    load_full_model_comparison,
    load_benchmark_classification,
    AGENT_COLORS,
    MODEL_COLORS,
    DATA_DIR,
)
from utils.plotting import PLOTLY_THEME, add_baseline_hline

st.set_page_config(page_title="MLIR-RL Dashboard", page_icon="⚡", layout="wide")

st.markdown("# MLIR-RL Evaluation Dashboard")

st.divider()

# ── Load data ──────────────────────────────────────────────────────────────
df_grouped = load_version_comparison_grouped()
df_abl     = load_ablation_real()
df_bench   = load_benchmark_classification()
df_full    = load_full_model_comparison()

# ── Summary cards ──────────────────────────────────────────────────────────
import numpy as np

def calculate_geo_mean(series):
    vals = series[series > 0].dropna()
    if len(vals) == 0:
        return 0.0
    return np.exp(np.mean(np.log(vals)))

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_models = df_grouped["model"].nunique() if not df_grouped.empty else 0
    st.metric("Model Families", n_models)
with col2:
    n_bench = df_bench["benchmark"].nunique() if not df_bench.empty else 0
    st.metric("Benchmarks", f"{n_bench:,}")
with col3:
    best = df_grouped["v45_speedup"].max() if not df_grouped.empty and "v45_speedup" in df_grouped.columns else 0
    st.metric("Best Our Agent Speedup", f"{best:.2f}×")
with col4:
    geo_mean_val = calculate_geo_mean(df_grouped["v45_speedup"]) if not df_grouped.empty and "v45_speedup" in df_grouped.columns else 0
    st.metric("Geo-Mean Our Agent Speedup", f"{geo_mean_val:.2f}×")

st.divider()

# ── Previous Agent vs Our Agent overview bar chart ──────────────────────────────────────────
st.markdown("## Previous Agent vs Our Agent — Geometric Mean Speedup by Model")

if not df_grouped.empty and "v0_speedup" in df_grouped.columns and "v45_speedup" in df_grouped.columns:
    # Melt into long format for grouped bars
    melt = df_grouped[["model", "v0_speedup", "v45_speedup"]].copy()
    # Normalise model names (upper-case in some CSVs)
    melt["model"] = melt["model"].str.upper()
    melt = melt.melt(id_vars="model", var_name="agent", value_name="speedup")
    melt["agent"] = melt["agent"].map({"v0_speedup": "Previous Agent", "v45_speedup": "Our Agent"})

    fig = px.bar(
        melt, x="model", y="speedup", color="agent",
        barmode="group",
        color_discrete_map={"Previous Agent": AGENT_COLORS["Previous Agent"], "Our Agent": AGENT_COLORS["Our Agent"]},
        labels={"model": "Model", "speedup": "Geometric Mean Speedup (×)", "agent": ""},
        height=420,
    )
    add_baseline_hline(fig)
    fig.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=0, r=0, t=20, b=60))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("version_comparison/grouped.csv not found in dashboard/data/.")

st.divider()

# ── Ablation overview ──────────────────────────────────────────────────────
st.markdown("## Ablation Study — Geometric Mean Speedup by Model")

if not df_abl.empty:
    col_map = {
        "v45_speedup": "Our Agent",
        "ntr_speedup": "No-Transformer",
        "nhw_speedup": "No-HW-Features",
        "nrw_speedup": "No-reward",
    }
    abl_cols = [c for c in col_map if c in df_abl.columns]
    melt_abl = df_abl[["model"] + abl_cols].melt(id_vars="model", var_name="agent", value_name="speedup")
    melt_abl["agent"] = melt_abl["agent"].map(col_map)
    melt_abl["model"] = melt_abl["model"].str.upper()

    color_map_abl = {
        "Our Agent": "#dc2626",        # red
        "No-Transformer": "#16a34a",   # green
        "No-HW-Features": "#359CDB",   # blue
        "No-reward": "#eab308",        # yellow
    }
    fig_abl = px.bar(
        melt_abl, x="model", y="speedup", color="agent", barmode="group",
        color_discrete_map=color_map_abl,
        labels={"model": "Model", "speedup": "Geometric Mean Speedup (×)", "agent": ""},
        height=420,
    )
    add_baseline_hline(fig_abl)
    fig_abl.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(l=0, r=0, t=20, b=60))
    fig_abl.update_traces(marker_line_width=0)
    st.plotly_chart(fig_abl, use_container_width=True)
else:
    st.info("ablation/real.csv not found in dashboard/data/.")

st.divider()

# ── Benchmark classification breakdown ────────────────────────────────────
st.markdown("## Benchmark Dataset")

if not df_bench.empty:
    c1, c2 = st.columns(2)
    with c1:
        fam_counts = df_bench["model_family"].value_counts().reset_index()
        fam_counts.columns = ["model_family", "count"]
        fig_fam = px.bar(fam_counts, x="model_family", y="count",
                         labels={"model_family": "Model Family", "count": "Benchmarks"},
                         height=340, color="model_family",
                         color_discrete_map=MODEL_COLORS)
        fig_fam.update_layout(**PLOTLY_THEME, showlegend=False,
                              xaxis_tickangle=-30, margin=dict(l=0, r=0, t=20, b=60))
        fig_fam.update_traces(marker_line_width=0)
        st.markdown("### By Model Family")
        st.plotly_chart(fig_fam, use_container_width=True)

    with c2:
        op_counts = df_bench["op_type"].value_counts().reset_index()
        op_counts.columns = ["op_type", "count"]
        fig_op = px.pie(op_counts, names="op_type", values="count",
                        height=340, hole=0.4,
                        color="op_type")
        fig_op.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=20))
        st.markdown("### By Operation Type")
        st.plotly_chart(fig_op, use_container_width=True)

st.divider()
st.caption("Navigate using the sidebar pages: **Version Comparison**, **Ablation Study**, **Multi-Hardware**, **Full Model**.")
