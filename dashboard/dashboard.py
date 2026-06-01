"""
MLIR-RL Evaluation Dashboard — Multi-page Streamlit app.

Run:
    streamlit run dashboard/dashboard.py --server.fileWatcherType none

CSV files are read from results/new_dataset_results/dashboard/.
Generate them first with: python scripts/generate_dashboard_csvs.py
"""

import sys
from pathlib import Path

# Ensure dashboard utils is importable
_dashboard_dir = Path(__file__).resolve().parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))

from utils.data import set_csv_dir

# ── CSV path config ──
_CSV_DIR = _dashboard_dir.parent / "results" / "new_dataset_results" / "dashboard"
_csv_env = __import__("os").environ.get("DASHBOARD_CSV_DIR", "")
if _csv_env:
    _CSV_DIR = Path(_csv_env)
_CSV_DIR.mkdir(parents=True, exist_ok=True)
set_csv_dir(_CSV_DIR)

# ── Home page ──
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_agent_registry, load_benchmarks_df,
    get_family_colors,
)
from utils.plotting import PLOTLY_THEME

st.set_page_config(page_title="MLIR-RL Dashboard", page_icon="⚡", layout="wide")

st.markdown("# ⚡ MLIR-RL Evaluation Dashboard")
st.caption("Reinforcement learning auto-scheduler for MLIR loop nests")

registry = load_agent_registry()
df_all = load_benchmarks_df()

st.divider()

# ── Summary cards ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Agents", len(registry))
with col2:
    st.metric("Benchmarks", df_all["benchmark"].nunique() if "benchmark" in df_all.columns else 0)
with col3:
    st.metric("Model Families", df_all["family"].nunique() if "family" in df_all.columns else 0)
with col4:
    st.metric("Data Points", f"{len(df_all):,}")

st.divider()

# ── Agent registry table ──
st.markdown("## Agents")
st.dataframe(
    registry[["agent_key", "display_name", "description", "category", "csv_file"]],
    use_container_width=True, hide_index=True,
    column_config={
        "agent_key": st.column_config.TextColumn("Key", width="small"),
        "display_name": "Agent",
        "description": "Description",
        "category": "Category",
        "csv_file": "CSV",
    },
)

st.divider()

# ── Benchmark distribution by family ──
st.markdown("## Benchmark Distribution by Model Family")
if not df_all.empty and "family" in df_all.columns:
    family_totals = df_all["family"].value_counts().reset_index()
    family_totals.columns = ["family", "count"]
    fig = px.bar(
        family_totals, x="family", y="count", color="family",
        color_discrete_map=get_family_colors(),
        labels={"family": "Model Family", "count": "Benchmarks"},
        height=420,
    )
    fig.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                      margin=dict(l=0, r=0, t=20, b=60), showlegend=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # Family breakdown table
    st.markdown("### Per-Family Counts")
    counts = df_all.groupby(["family", "agent"])["benchmark"].count().unstack(fill_value=0)
    st.dataframe(counts.reset_index(), use_container_width=True, hide_index=True)
else:
    st.info("No benchmark data loaded. Run `scripts/generate_dashboard_csvs.py` first.")

st.divider()
st.caption(f"Data source: `{_CSV_DIR}`")
st.caption("Navigate using the sidebar pages: **Version Comparison** and **Ablation Study**.")
