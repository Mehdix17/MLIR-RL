'''Multi-Hardware -- No-Reward (HW) agent across Bergamo, Dalma, Jubail clusters.'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_hardware_model,
    load_hardware_optype,
)
from utils.plotting import PLOTLY_THEME, add_baseline_hline

st.set_page_config(page_title="MLIR-RL -- Multi-Hardware", page_icon="⚡", layout="wide")

st.markdown("# Multi-Hardware Evaluation")
st.caption("No-Reward (HW-aware) agent evaluated on three HPC clusters: Bergamo, Dalma, Jubail")

CLUSTERS = ["bergamo", "dalma", "jubail"]

CLUSTER_COLORS = {
    "bergamo": "#dc2626",   # red
    "dalma": "#359CDB",     # blue
    "jubail": "#16a34a",    # green
}

# Load all clusters
dfs_model  = {c: load_hardware_model(c)  for c in CLUSTERS}
dfs_optype = {c: load_hardware_optype(c) for c in CLUSTERS}

# 1. Cross-cluster comparison - by model
st.divider()
st.markdown("## Cross-Cluster Speedup by Model")

metric_col = "geo_mean"

frames_model = []
for c in CLUSTERS:
    df = dfs_model[c]
    if not df.empty and metric_col in df.columns:
        df = df[["cluster", "group", metric_col]].copy()
        df.columns = ["cluster", "model", "speedup"]
        df["cluster"] = df["cluster"].str.capitalize()
        frames_model.append(df)

if frames_model:
    df_all_model = pd.concat(frames_model, ignore_index=True)
    cluster_color_map = {c.capitalize(): v for c, v in CLUSTER_COLORS.items()}
    fig_model = px.bar(
        df_all_model, x="model", y="speedup", color="cluster", barmode="group",
        color_discrete_map=cluster_color_map,
        labels={"model": "Model", "speedup": "Geometric Mean Speedup (x)", "cluster": "Cluster"},
        height=440,
    )
    add_baseline_hline(fig_model)
    fig_model.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=0, r=0, t=20, b=60))
    fig_model.update_traces(marker_line_width=0)
    st.plotly_chart(fig_model, use_container_width=True)
else:
    st.info("Hardware model CSV files not found.")

# 2. Cross-cluster comparison - by op-type
st.divider()
st.markdown("## Cross-Cluster Speedup by Operation Type")

frames_op = []
for c in CLUSTERS:
    df = dfs_optype[c]
    if not df.empty and metric_col in df.columns:
        df = df[["cluster", "group", metric_col]].copy()
        df.columns = ["cluster", "op_type", "speedup"]
        df["cluster"] = df["cluster"].str.capitalize()
        frames_op.append(df)

if frames_op:
    df_all_op = pd.concat(frames_op, ignore_index=True)
    fig_op = px.bar(
        df_all_op, x="op_type", y="speedup", color="cluster", barmode="group",
        color_discrete_map=cluster_color_map,
        labels={"op_type": "Operation Type", "speedup": "Geometric Mean Speedup (x)", "cluster": "Cluster"},
        height=380,
    )
    add_baseline_hline(fig_op)
    fig_op.update_layout(**PLOTLY_THEME, xaxis_tickangle=0,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02),
                         margin=dict(l=0, r=0, t=20, b=40))
    fig_op.update_traces(marker_line_width=0)
    st.plotly_chart(fig_op, use_container_width=True)

# 3. Per-cluster details
st.divider()
st.markdown("## Per-Cluster Detail")

sel_cluster = st.selectbox("Select cluster", [c.capitalize() for c in CLUSTERS])
sel_cluster_key = sel_cluster.lower()

tab_model, tab_op = st.tabs(["By Model", "By Op-Type"])

with tab_model:
    df_m = dfs_model[sel_cluster_key]
    if not df_m.empty:
        fig_m = px.bar(df_m, x="group", y=metric_col,
                       labels={"group": "Model", metric_col: "Geometric Mean Speedup (x)"},
                       height=340)
        fig_m.update_traces(marker_color=CLUSTER_COLORS[sel_cluster_key], marker_line_width=0)
        add_baseline_hline(fig_m)
        fig_m.update_layout(**PLOTLY_THEME, showlegend=False,
                            xaxis_tickangle=-20, margin=dict(l=0, r=0, t=20, b=60))
        st.plotly_chart(fig_m, use_container_width=True)

with tab_op:
    df_o = dfs_optype[sel_cluster_key]
    if not df_o.empty:
        fig_o = px.bar(df_o, x="group", y=metric_col,
                       labels={"group": "Op Type", metric_col: "Geometric Mean Speedup (x)"},
                       height=300)
        fig_o.update_traces(marker_color=CLUSTER_COLORS[sel_cluster_key], marker_line_width=0)
        add_baseline_hline(fig_o)
        fig_o.update_layout(**PLOTLY_THEME, showlegend=False,
                            xaxis_tickangle=0, margin=dict(l=0, r=0, t=20, b=40))
        st.plotly_chart(fig_o, use_container_width=True)
