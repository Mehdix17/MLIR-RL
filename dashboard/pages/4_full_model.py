'''Full Model Comparison -- Previous Agent vs Our Agent across full neural network models.'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_full_model_comparison,
    AGENT_COLORS,
)
from utils.plotting import PLOTLY_THEME, add_baseline_hline

st.set_page_config(page_title="MLIR-RL -- Full Model", page_icon="⚡", layout="wide")

st.markdown("# Full Model Comparison")
st.caption("Previous Agent vs Our Agent -- end-to-end speedup on complete neural network models.")

# Load data
df = load_full_model_comparison()

if df.empty:
    st.warning("full_model/comparison.csv not found in dashboard/data/.")
    st.stop()

# Bar chart: total speedup per model
v0_col = "V0_total_speedup"
nr_col = "No-Reward_total_speedup"

st.markdown("## Total Speedup by Neural Network Model")

agent_cols = [c for c in [v0_col, nr_col] if c in df.columns]
if agent_cols:
    melt = df[["model"] + agent_cols].copy()
    melt["model"] = melt["model"].str.upper()
    melt = melt.melt(id_vars="model", var_name="agent", value_name="speedup")
    label_map = {v0_col: "Previous Agent", nr_col: "Our Agent"}
    melt["agent"] = melt["agent"].map(label_map)
    color_map = {
        "Previous Agent": AGENT_COLORS["Previous Agent"],
        "Our Agent": AGENT_COLORS["Our Agent"],
    }

    fig_sp = px.bar(
        melt, x="model", y="speedup", color="agent", barmode="group",
        color_discrete_map=color_map,
        labels={"model": "Model", "speedup": "Geometric Speedup (x)", "agent": ""},
        height=440,
    )
    add_baseline_hline(fig_sp)
    fig_sp.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02),
                         margin=dict(l=0, r=0, t=20, b=60))
    fig_sp.update_traces(marker_line_width=0)
    st.plotly_chart(fig_sp, use_container_width=True)
