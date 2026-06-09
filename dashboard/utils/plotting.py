"""Shared plot utilities for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PLOTLY_THEME = dict(
    template="plotly_white",
    font_family="sans-serif",
)


def add_baseline_hline(fig: go.Figure, y: float = 1.0, label: str = "Baseline (1×)"):
    fig.add_hline(
        y=y, line_dash="dash", line_color="#888",
        annotation_text=label, annotation_position="top right",
    )


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    color_map: dict = None,
    barmode: str = "group",
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    height: int = 420,
    xangle: int = -20,
) -> go.Figure:
    kwargs = dict(
        x=x_col, y=y_col,
        labels={x_col: x_label, y_col: y_label},
        height=height,
        barmode=barmode,
    )
    if color_col:
        kwargs["color"] = color_col
        kwargs["color_discrete_map"] = color_map or {}
        kwargs["labels"][color_col] = ""
    fig = px.bar(df, **kwargs)
    fig.update_layout(
        **PLOTLY_THEME,
        title=title,
        xaxis_tickangle=xangle,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    fig.update_traces(marker_line_width=0)
    return fig
