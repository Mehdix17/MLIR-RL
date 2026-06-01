"""Version Comparison — compare any two agents side by side."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data import (
    load_version_comparison, load_agent_registry, load_benchmarks_df,
    get_family_colors, get_agent_colors, get_agent_display_name,
    build_benchmark_index, parse_model_name,
)
from utils.plotting import (
    grouped_bar, speedup_horizontal_line, per_benchmark_comparison,
    build_search_options, PLOTLY_THEME,
)

st.set_page_config(page_title="MLIR-RL — Version Comparison", page_icon="⚡", layout="wide")

st.markdown("# Version Comparison")
st.caption("Compare any two agents across model families and individual benchmarks.")

registry = load_agent_registry()
comp_df = load_version_comparison()
df_all = load_benchmarks_df()

if df_all.empty or registry.empty:
    st.warning("No data loaded. Run `scripts/generate_dashboard_csvs.py` first.")
    st.stop()

agent_keys = sorted(registry["agent_key"].unique())
agent_options = {k: get_agent_display_name(k) for k in agent_keys}

col_a, col_b = st.columns(2)
with col_a:
    agent_a = st.selectbox("Agent A", agent_keys, index=0,
                           format_func=lambda k: agent_options[k])
with col_b:
    default_b = agent_keys[1] if len(agent_keys) > 1 else agent_keys[0]
    agent_b = st.selectbox("Agent B", agent_keys,
                           index=min(1, len(agent_keys) - 1),
                           format_func=lambda k: agent_options[k])

if agent_a == agent_b:
    st.warning("Select two different agents to compare.")
    st.stop()

st.divider()

# ── Data preparation ──
df_a = df_all[df_all["agent"] == agent_a].copy()
df_b = df_all[df_all["agent"] == agent_b].copy()

common = set(df_a["benchmark"]) & set(df_b["benchmark"])
st.caption(f"**{len(common):,}** common benchmarks between {agent_options[agent_a]} and {agent_options[agent_b]}")

df_a = df_a[df_a["benchmark"].isin(common)]
df_b = df_b[df_b["benchmark"].isin(common)]

# Per-family averages
fam_avg_a = df_a.groupby("family")["speedup"].mean().reset_index()
fam_avg_a["agent"] = agent_options[agent_a]
fam_avg_b = df_b.groupby("family")["speedup"].mean().reset_index()
fam_avg_b["agent"] = agent_options[agent_b]
fam_avg = pd.concat([fam_avg_a, fam_avg_b], ignore_index=True)

# ── Per-Family Bar Chart ──
st.markdown("## Speedup by Model Family")
fig_fam = grouped_bar(
    fam_avg, x_col="family", y_col="speedup", color_col="agent",
    x_label="Model Family", y_label="Avg Speedup (×)",
    color_map={agent_options[agent_a]: "#2563eb",
               agent_options[agent_b]: "#16a34a"},
    height=400,
)
speedup_horizontal_line(fig_fam)
st.plotly_chart(fig_fam, use_container_width=True)

st.divider()

# ── Win/Loss Summary ──
merged = df_a.set_index("benchmark")["speedup"].rename("sp_a").reset_index()
merged_b = df_b.set_index("benchmark")["speedup"].rename("sp_b").reset_index()
merged = merged.merge(merged_b, on="benchmark")
merged["a_wins"] = merged["sp_a"] > merged["sp_b"]
wins_a = merged["a_wins"].sum()
wins_b = len(merged) - wins_a
ties = (merged["sp_a"] == merged["sp_b"]).sum()

col_w, col_l, col_t = st.columns(3)
with col_w:
    st.metric(f"{agent_options[agent_a]} Wins", wins_a)
with col_l:
    st.metric(f"{agent_options[agent_b]} Wins", wins_b)
with col_t:
    st.metric("Ties", ties)

st.divider()

# ── Per-Benchmark Drill-Down ──
st.markdown("## Per-Benchmark Drill-Down")

search_options = build_search_options(df_a)
all_benches = sorted(df_a["benchmark"].unique())

# Group options by model
model_groups = sorted(search_options.keys())
col_search, col_family = st.columns([3, 1])
with col_search:
    selected_bench = st.selectbox(
        "Search benchmark", options=all_benches,
        index=None, placeholder="Type benchmark name...",
        label_visibility="collapsed",
    )
with col_family:
    families_sorted = sorted(df_a["family"].unique())
    sel_family = st.selectbox("Filter family", ["All"] + families_sorted, key="v2_fam")

if selected_bench:
    label_map = {agent_a: agent_options[agent_a], agent_b: agent_options[agent_b]}

    # Build wide-format row for the benchmark
    row_a = df_a[df_a["benchmark"] == selected_bench].iloc[0]
    row_b = df_b[df_b["benchmark"] == selected_bench].iloc[0]
    wide_row = {
        "benchmark": selected_bench,
        "mlir_baseline": row_a["mlir_baseline"],
        "pytorch_eager": row_a["pytorch_eager"],
        "pytorch_jit": row_a["pytorch_jit"],
        f"{agent_a}_exec_time": row_a["mlir_rl_exec_time"],
        f"{agent_a}_speedup": row_a["speedup"],
        f"{agent_b}_exec_time": row_b["mlir_rl_exec_time"],
        f"{agent_b}_speedup": row_b["speedup"],
    }
    wide_df = pd.DataFrame([wide_row])

    tab1, tab2 = st.tabs(["Execution Time", "Speedup"])

    with tab1:
        fig_exec = per_benchmark_comparison(
            wide_df, selected_bench,
            [agent_a, agent_b], label_map,
        )
        st.plotly_chart(fig_exec, use_container_width=True)

    with tab2:
        fig_sp = per_benchmark_comparison(
            wide_df, selected_bench,
            [agent_a, agent_b], label_map,
        )
        fig_sp.data[0].y = None
        sp_values = []
        sp_names = []
        base = row_a["mlir_baseline"]
        if base and base > 0:
            for agent in [agent_a, agent_b]:
                col = f"{agent}_exec_time"
                et = wide_row[col]
                if et and et > 0:
                    sp_names.append(label_map.get(agent, agent))
                    sp_values.append(base / et)
        fig_sp.data = []
        fig_sp.add_trace(go.Bar(x=sp_names, y=sp_values,
                                marker_color=["#2563eb", "#16a34a"],
                                marker_line_width=0))
        speedup_horizontal_line(fig_sp)
        fig_sp.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                             margin=dict(l=0, r=0, t=50, b=80),
                             yaxis_title="Speedup (×)",
                             title=f"Speedup — {selected_bench}")
        st.plotly_chart(fig_sp, use_container_width=True)

    # Show benchmark details
    st.markdown("#### Benchmark Details")
    cols = ["benchmark", "family", "mlir_baseline",
            f"{agent_a}_exec_time", f"{agent_a}_speedup",
            f"{agent_b}_exec_time", f"{agent_b}_speedup"]
    cols_renamed = {
        "benchmark": "Benchmark", "family": "Family",
        "mlir_baseline": "MLIR Baseline (ns)",
        f"{agent_a}_exec_time": f"{agent_options[agent_a]} Time (ns)",
        f"{agent_a}_speedup": f"{agent_options[agent_a]} Speedup",
        f"{agent_b}_exec_time": f"{agent_options[agent_b]} Time (ns)",
        f"{agent_b}_speedup": f"{agent_options[agent_b]} Speedup",
    }
    show = wide_df[[c for c in cols if c in wide_df.columns]].rename(
        columns={k: v for k, v in cols_renamed.items() if k in wide_df.columns})
    st.dataframe(show, use_container_width=True, hide_index=True)

# ── Full per-family detail table ──
st.markdown("### Per-Family Detail")
show_fam = fam_avg.pivot(index="family", columns="agent", values="speedup")
show_fam.columns = [f"{c} Speedup" for c in show_fam.columns]
st.dataframe(show_fam.reset_index(), use_container_width=True, hide_index=True)
