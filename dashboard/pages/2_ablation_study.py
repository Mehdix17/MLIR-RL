"""Ablation Study — V4.5 vs No-HW vs No-Reward vs No-Transformer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data import (
    load_ablation_summary, load_benchmarks_df,
    get_family_colors, get_agent_colors, get_agent_display_name,
    build_benchmark_index, parse_model_name,
)
from utils.plotting import (
    grouped_bar, speedup_horizontal_line,
    build_search_options, PLOTLY_THEME,
)

st.set_page_config(page_title="MLIR-RL — Ablation Study", page_icon="⚡", layout="wide")

st.markdown("# Ablation Study")
st.caption("V4.5 (Robust) vs three ablations: No Hardware Observation, No Shaped Reward, No Transformer")

ABLATION_AGENTS = ["v4_5", "no_transformer", "no_hw", "no_reward"]

df_all = load_benchmarks_df()
if df_all.empty:
    st.warning("No data loaded. Run `scripts/generate_dashboard_csvs.py` first.")
    st.stop()

# Filter to ablation agents
df_abl = df_all[df_all["agent"].isin(ABLATION_AGENTS)]
if df_abl.empty:
    st.warning("No ablation data found.")
    st.stop()

summary = load_ablation_summary()

st.divider()

# ── Overall ──
st.markdown("## Overall Average Speedup")
overall = df_abl.groupby("agent")["speedup"].agg(["mean", "median", "count"]).reset_index()
overall["agent_display"] = overall["agent"].apply(get_agent_display_name)

col1, col2 = st.columns([2, 1])
with col1:
    fig_overall = px.bar(
        overall, x="agent_display", y="mean", color="agent",
        color_discrete_map=get_agent_colors(),
        labels={"agent_display": "", "mean": "Avg Speedup (×)"},
        height=380,
    )
    speedup_horizontal_line(fig_overall)
    fig_overall.update_layout(**PLOTLY_THEME, xaxis_tickangle=-15,
                              margin=dict(l=0, r=0, t=20, b=60),
                              showlegend=False)
    fig_overall.update_traces(marker_line_width=0)
    st.plotly_chart(fig_overall, use_container_width=True)

with col2:
    st.markdown("#### Summary")
    st.dataframe(
        overall[["agent_display", "mean", "median", "count"]]
        .rename(columns={"agent_display": "Agent", "mean": "Avg Sp", "median": "Median", "count": "N"}),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ── Per-Family ──
st.markdown("## Speedup by Model Family")
if not summary.empty:
    families = sorted(summary["family"].unique())
    speedup_cols = [c for c in summary.columns if c.endswith("_speedup")]
    agents_in_summary = [c.replace("_speedup", "") for c in speedup_cols]

    melt_rows = []
    for _, row in summary.iterrows():
        family = row["family"]
        for agent in agents_in_summary:
            col = f"{agent}_speedup"
            if col in row and pd.notna(row[col]):
                melt_rows.append({
                    "family": family,
                    "agent": get_agent_display_name(agent),
                    "speedup": row[col],
                    "agent_key": agent,
                })
    df_melt = pd.DataFrame(melt_rows)

    if not df_melt.empty:
        color_map = {get_agent_display_name(k): v for k, v in get_agent_colors().items()}
        fig_fam = grouped_bar(
            df_melt, x_col="family", y_col="speedup", color_col="agent",
            x_label="Model Family", y_label="Avg Speedup (×)",
            color_map=color_map, height=420,
        )
        speedup_horizontal_line(fig_fam)
        st.plotly_chart(fig_fam, use_container_width=True)

        # Summary table
        st.markdown("### Per-Family Summary")
        pivot = df_melt.pivot(index="family", columns="agent", values="speedup")
        st.dataframe(pivot.reset_index(), use_container_width=True, hide_index=True)
else:
    st.info("No ablation summary data. Run `scripts/generate_dashboard_csvs.py`.")

st.divider()

# ── Per-Benchmark Drill-Down ──
st.markdown("## Per-Benchmark Drill-Down")

df_benches = df_abl.head(1)
all_benches = sorted(df_abl["benchmark"].unique())
search_options = build_search_options(df_abl)

col_search, col_family = st.columns([3, 1])
with col_search:
    selected_bench = st.selectbox(
        "Search benchmark", options=all_benches,
        index=None, placeholder="Type benchmark name...",
        key="abl_bench",
        label_visibility="collapsed",
    )
with col_family:
    families_sorted = sorted(df_abl["family"].unique())
    sel_family = st.selectbox("Filter family", ["All"] + families_sorted, key="abl_fam")

if selected_bench:
    bench_rows = df_abl[df_abl["benchmark"] == selected_bench]
    baseline = bench_rows.iloc[0]["mlir_baseline"] if not bench_rows.empty else 0

    names = ["MLIR Baseline"]
    values = [1.0]
    colors = ["#94a3b8"]

    for _, row in bench_rows.iterrows():
        sp = row["speedup"]
        agent = row["agent"]
        names.append(get_agent_display_name(agent))
        values.append(sp)
        colors.append(get_agent_colors().get(agent, "#6b7280"))

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(x=names, y=values, marker_color=colors, marker_line_width=0))
    speedup_horizontal_line(fig_bench)
    fig_bench.update_layout(**PLOTLY_THEME, xaxis_tickangle=-15,
                            margin=dict(l=0, r=0, t=50, b=80),
                            yaxis_title="Speedup (×)",
                            title=f"Speedup — {selected_bench}")
    st.plotly_chart(fig_bench, use_container_width=True)

    st.markdown("#### Benchmark Details")
    display = bench_rows[["agent_display", "mlir_baseline", "mlir_rl_exec_time", "speedup"]]
    display = display.rename(columns={
        "agent_display": "Agent", "mlir_baseline": "MLIR Baseline (ns)",
        "mlir_rl_exec_time": "Optimized Time (ns)", "speedup": "Speedup",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
