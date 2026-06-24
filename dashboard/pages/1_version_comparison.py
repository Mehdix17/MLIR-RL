"""Version Comparison — V0 vs V4.5 across models and benchmarks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_version_comparison_grouped,
    load_version_comparison_per_bench,
    load_graph1_performance,
    load_benchmark_classification,
    AGENT_COLORS,
    MODEL_COLORS,
)
from utils.plotting import PLOTLY_THEME, add_baseline_hline

st.set_page_config(page_title="MLIR-RL — Version Comparison", page_icon="⚡", layout="wide")

st.markdown("# Version Comparison")
st.caption("Previous Agent vs Our Agent")

# ── Data ────────────────────────────────────────────────────────────────────
df_grouped  = load_version_comparison_grouped()
df_per_bench = load_version_comparison_per_bench()
df_graph1   = load_graph1_performance()
df_bench_class = load_benchmark_classification()

# ── 1. Per-model grouped speedup ────────────────────────────────────────────
st.divider()
st.markdown("## Geometric Mean Speedup by Model")

if not df_grouped.empty:
    melt = df_grouped[["model", "v0_speedup", "v45_speedup"]].copy()
    melt["model"] = melt["model"].str.upper()
    melt = melt.melt(id_vars="model", var_name="agent", value_name="speedup")
    melt["agent"] = melt["agent"].map({"v0_speedup": "Previous Agent", "v45_speedup": "Our Agent"})
    fig = px.bar(
        melt, x="model", y="speedup", color="agent", barmode="group",
        color_discrete_map={"Previous Agent": AGENT_COLORS["Previous Agent"], "Our Agent": AGENT_COLORS["Our Agent"]},
        labels={"model": "Model", "speedup": "Geometric Mean Speedup (×)", "agent": ""},
        height=440,
    )
    add_baseline_hline(fig)
    fig.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=0, r=0, t=20, b=60))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# ── 2. Per-operation-type speedup ────────────────────────────────────────────
st.divider()
st.markdown("## Previous Agent vs Our Agent — Speedup by Operation Type")
st.caption("Compares baseline Previous Agent against Our Agent across op-types.")

if not df_graph1.empty:
    op_col = df_graph1.columns[0]
    geo_data = []
    for col_suffix, label in [("V0_geo_mean", "Previous Agent"), ("NoReward_geo_mean", "Our Agent")]:
        if col_suffix in df_graph1.columns:
            tmp = df_graph1[[op_col, col_suffix]].copy()
            tmp.columns = ["op_type", "geo_mean"]
            tmp["agent"] = label
            geo_data.append(tmp)

    if geo_data:
        df_geo = pd.concat(geo_data, ignore_index=True)
        fig_geo = px.bar(
            df_geo, x="op_type", y="geo_mean", color="agent", barmode="group",
            color_discrete_map={"Previous Agent": AGENT_COLORS["Previous Agent"], "Our Agent": AGENT_COLORS["Our Agent"]},
            labels={"op_type": "Operation Type", "geo_mean": "Geometric Mean Speedup (×)", "agent": ""},
            height=380,
        )
        add_baseline_hline(fig_geo)
        fig_geo.update_layout(**PLOTLY_THEME, xaxis_tickangle=0,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              margin=dict(l=0, r=0, t=20, b=40))
        fig_geo.update_traces(marker_line_width=0)
        st.plotly_chart(fig_geo, use_container_width=True)

# ── 3. Benchmark Dataset ────────────────────────────────────────────────────
st.divider()
st.markdown("## Benchmark Dataset")

if not df_bench_class.empty:
    c1, c2 = st.columns(2)

    with c1:
        fam_counts = df_bench_class[~df_bench_class["model_family"].isin(["legacy"])]["model_family"].value_counts().reset_index()
        fam_counts.columns = ["model_family", "count"]
        fig_fam = px.pie(fam_counts, names="model_family", values="count",
                         height=340, hole=0.4, color="model_family")
        fig_fam.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=20))
        st.markdown("### By Model Family")
        st.plotly_chart(fig_fam, use_container_width=True)

    with c2:
        op_counts = df_bench_class[~df_bench_class["op_type"].isin(["block"])]["op_type"].value_counts().reset_index()
        op_counts.columns = ["op_type", "count"]
        fig_op = px.pie(op_counts, names="op_type", values="count",
                        height=340, hole=0.4, color="op_type")
        fig_op.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=20))
        st.markdown("### By Operation Type")
        st.plotly_chart(fig_op, use_container_width=True)

    st.markdown("### Benchmark Counts")
    col_a, col_b = st.columns(2)
    with col_a:
        fam_table = df_bench_class[~df_bench_class["model_family"].isin(["legacy"])]["model_family"].value_counts().reset_index()
        fam_table.columns = ["Model Family", "Count"]
        st.dataframe(fam_table, use_container_width=True, hide_index=True)
    with col_b:
        op_table = df_bench_class[~df_bench_class["op_type"].isin(["block"])]["op_type"].value_counts().reset_index()
        op_table.columns = ["Operation Type", "Count"]
        st.dataframe(op_table, use_container_width=True, hide_index=True)

# ── 4. Per-benchmark detail ──────────────────────────────────────────────────
st.divider()
st.markdown("## Per-Benchmark Detail (Our Agent vs Previous Agent)")
st.caption("Select a model family or operation type, then pick specific benchmarks to compare.")

if not df_per_bench.empty:
    filter_by = st.radio("Filter benchmarks by", ["Model Family", "Operation Type"], horizontal=True)

    if filter_by == "Model Family":
        families = sorted(df_per_bench["model"].unique()) if "model" in df_per_bench.columns else []
        selected_group = st.selectbox("Select Model Family", options=families)
        df_filtered = df_per_bench[df_per_bench["model"] == selected_group] if selected_group else pd.DataFrame()
    else:
        import re
        def extract_op_type(bench_name):
            parts = bench_name.split("_")
            known_ops = ["batch_matmul", "conv2d", "generic", "matmul", "pooling"]
            for op in known_ops:
                if op in parts:
                    return op
            return "other"
        df_per_bench = df_per_bench.copy()
        df_per_bench["op_type"] = df_per_bench["benchmark"].apply(extract_op_type)
        op_types = sorted(df_per_bench["op_type"].unique())
        selected_group = st.selectbox("Select Operation Type", options=op_types)
        df_filtered = df_per_bench[df_per_bench["op_type"] == selected_group] if selected_group else pd.DataFrame()

    if not df_filtered.empty:
        bench_names = sorted(df_filtered["benchmark"].unique())
        selected_benchmarks = st.multiselect(
            "Select Benchmarks",
            options=bench_names,
            default=bench_names[:min(5, len(bench_names))],
        )

        if selected_benchmarks:
            df_plot = df_filtered[df_filtered["benchmark"].isin(selected_benchmarks)]

            melt_bench = df_plot[["benchmark", "v0_speedup", "v45_speedup"]].copy()
            melt_bench = melt_bench.melt(id_vars="benchmark", var_name="agent", value_name="speedup")
            melt_bench["agent"] = melt_bench["agent"].map({"v0_speedup": "Previous Agent", "v45_speedup": "Our Agent"})

            fig_bench = px.bar(
                melt_bench, x="benchmark", y="speedup", color="agent", barmode="group",
                color_discrete_map={"Previous Agent": AGENT_COLORS["Previous Agent"], "Our Agent": AGENT_COLORS["Our Agent"]},
                labels={"benchmark": "Benchmark", "speedup": "Geometric Speedup (×)", "agent": ""},
                height=450,
            )
            add_baseline_hline(fig_bench)
            fig_bench.update_layout(**PLOTLY_THEME, xaxis_tickangle=-35,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                   margin=dict(l=0, r=0, t=20, b=100))
            fig_bench.update_traces(marker_line_width=0)
            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.info("Select at least one benchmark to display.")
    else:
        st.info("No benchmarks found for the selected filter.")
else:
    st.info("version_comparison/per_bench.csv not found in dashboard/data/.")
