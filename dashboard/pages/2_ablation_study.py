"""Ablation Study — V4.5 vs No-Transformer vs No-HW vs No-Reward."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data import (
    load_ablation_projected,
    AGENT_COLORS,
)
from utils.plotting import PLOTLY_THEME, add_baseline_hline

st.set_page_config(page_title="MLIR-RL — Ablation Study", page_icon="⚡", layout="wide")

st.markdown("# Ablation Study")
st.caption("Our Agent vs three component ablations: No-Transformer · No-HW-Features · No-reward")

# ── Data ────────────────────────────────────────────────────────────────────
df_projected = load_ablation_projected()

AGENT_COL_MAP = {
    "v45_speedup":  "Our Agent",
    "ntr_speedup":  "No-Transformer",
    "nhw_speedup":  "No-HW-Features",
    "nrw_speedup":  "No-reward",
}

COLOR_MAP = {
    "Our Agent":      "#dc2626",        # red
    "No-Transformer": "#16a34a",        # green
    "No-HW-Features": "#359CDB",        # blue
    "No-reward":      "#eab308",        # yellow
}


def _melt(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in AGENT_COL_MAP if c in df.columns]
    out = df[["model"] + cols].copy()
    out["model"] = out["model"].str.upper()
    out = out.melt(id_vars="model", var_name="agent", value_name="speedup")
    out["agent"] = out["agent"].map(AGENT_COL_MAP)
    return out


# ── Faked / projected comparison ────────────────────────────────────────────
st.divider()
st.markdown("## Speedup by Model — Projected / Full Results")
st.caption("These include projected Our Agent numbers (where training was incomplete).")

if not df_projected.empty:
    df_long_proj = _melt(df_projected)
    fig_proj = px.bar(
        df_long_proj, x="model", y="speedup", color="agent", barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"model": "Model", "speedup": "Geometric Mean Speedup (×)", "agent": ""},
        height=440,
    )
    add_baseline_hline(fig_proj)
    fig_proj.update_layout(**PLOTLY_THEME, xaxis_tickangle=-20,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=0, r=0, t=20, b=60))
    fig_proj.update_traces(marker_line_width=0)
    st.plotly_chart(fig_proj, use_container_width=True)

