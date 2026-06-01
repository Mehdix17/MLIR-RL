"""Shared plot utilities for the dashboard."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.data import get_family_colors, get_agent_colors, get_agent_display_name

PLOTLY_THEME = dict(
    template="plotly_white",
    font_family="sans-serif",
)


def grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str,
                title: str = "", x_label: str = "", y_label: str = "",
                color_map: dict = None, height: int = 420):
    """Grouped bar chart."""
    fig = px.bar(
        df, x=x_col, y=y_col, color=color_col,
        barmode="group", color_discrete_map=color_map,
        labels={x_col: x_label, y_col: y_label, color_col: ""},
        height=height,
        category_orders={x_col: sorted(df[x_col].unique())},
    )
    fig.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=0, r=0, t=40, b=60),
                      title=title)
    fig.update_traces(marker_line_width=0)
    return fig


def speedup_horizontal_line(fig: go.Figure):
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                  annotation_text="No improvement",
                  annotation_position="top right")


def per_benchmark_comparison(df_wide: pd.DataFrame, benchmark: str,
                             agents: list[str], label_map: dict) -> go.Figure:
    """Single benchmark bar chart: MLIR baseline + all agents + PyTorch eager/jit."""
    row = df_wide[df_wide["benchmark"] == benchmark]
    if row.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_THEME, title=f"No data for {benchmark}")
        return fig

    row = row.iloc[0]
    names = ["MLIR Baseline"]
    values = [row.get("mlir_baseline", 0) or 0]
    colors = ["#94a3b8"]

    for agent in agents:
        col = f"{agent}_exec_time"
        if col in row and row[col] and row[col] > 0:
            names.append(label_map.get(agent, agent))
            values.append(row[col])
            colors.append(get_agent_colors().get(agent, "#6b7280"))

    for mode in ["pytorch_eager", "pytorch_jit"]:
        if mode in row and row[mode] and row[mode] > 0:
            names.append({
                "pytorch_eager": "PyTorch Eager",
                "pytorch_jit": "PyTorch JIT",
            }[mode])
            values.append(row[mode] * 1000)  # convert us -> ns
            colors.append({"pytorch_eager": "#f59e0b", "pytorch_jit": "#8b5cf6"}[mode])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors, marker_line_width=0))
    fig.update_layout(
        **PLOTLY_THEME,
        title=f"Execution Time — {benchmark}",
        xaxis_tickangle=-20,
        margin=dict(l=0, r=0, t=50, b=80),
        yaxis_title="Time (ns)",
    )
    return fig


def per_benchmark_ablation(df_wide: pd.DataFrame, benchmark: str,
                           agent_keys: list[str],
                           label_map: dict) -> go.Figure:
    """Bar chart for ablation: MLIR baseline + speedup bars per agent."""
    row = df_wide[df_wide["benchmark"] == benchmark]
    if row.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_THEME, title=f"No data for {benchmark}")
        return fig

    row = row.iloc[0]
    names = ["MLIR Baseline"]
    values = [1.0]
    colors = ["#94a3b8"]

    for agent in agent_keys:
        col = f"{agent}_speedup"
        if col in row and row[col] and row[col] > 0:
            names.append(label_map.get(agent, agent))
            values.append(row[col])
            colors.append(get_agent_colors().get(agent, "#6b7280"))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors, marker_line_width=0))
    speedup_horizontal_line(fig)
    fig.update_layout(
        **PLOTLY_THEME,
        title=f"Speedup — {benchmark}",
        xaxis_tickangle=-20,
        margin=dict(l=0, r=0, t=50, b=80),
        yaxis_title="Speedup (×)",
    )
    return fig


def build_search_options(df: pd.DataFrame) -> dict:
    """Group benchmarks by model prefix for autocomplete.
    Returns {model_name: [benchmark_names]}"""
    groups = {}
    for _, row in df.iterrows():
        bench = row["benchmark"]
        model = bench.split("_")[0]
        if bench.startswith("llama3"):
            model = "llama3"
        groups.setdefault(model, []).append(bench)
    return groups
