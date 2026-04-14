"""
MLIR-RL Evaluation Dashboard
Visualizes RL agent performance vs PyTorch baselines across benchmarks.
"""

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).parent.parent / "results"
DEFAULT_IMPLEMENTATION = "rl_autoschedular"
NEW_IMPLEMENTATION = "new_rl_autoschedular"
IMPL_ORDER = [DEFAULT_IMPLEMENTATION, NEW_IMPLEMENTATION]
IMPL_AGENT_DIR = {
    DEFAULT_IMPLEMENTATION: "old_agent",
    NEW_IMPLEMENTATION: "new_agent",
}
IMPL_BASE_PREFIX = {
    DEFAULT_IMPLEMENTATION: "old",
    NEW_IMPLEMENTATION: "new",
}
IMPL_SHORT = {
    DEFAULT_IMPLEMENTATION: "old",
    NEW_IMPLEMENTATION: "new",
}
IMPL_DISPLAY = {
    DEFAULT_IMPLEMENTATION: "Old RL",
    NEW_IMPLEMENTATION: "New RL",
}

st.set_page_config(
    page_title="MLIR-RL Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
      background: #f7f8fa;
      border: 1px solid #e0e3ea;
      border-radius: 8px;
      padding: 1rem 1.2rem;
      text-align: center;
  }
  .metric-card .label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #888;
      margin-bottom: 0.3rem;
  }
  .metric-card .value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #1a1a2e;
  }
  .metric-card .value.good { color: #1a7f4b; }
  .metric-card .value.warn { color: #c0700a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def infer_bench_category(model_family: str, op_type: str) -> str:
    """
    Infer the benchmark category from the paper's three benchmark sets:
    - 'DL Operator'  : single op benchmarks (matmul, conv, pooling, add/relu via generic)
    - 'DL Model'     : full neural network models (resnet, vgg, mobilenet, etc.)
    - 'LQCD'         : lattice QCD benchmarks (lqcd, dibaryon, hexaquark, etc.)
    """
    mf = (model_family or "").lower()
    ot = (op_type or "").lower()

    # LQCD patterns
    if any(x in mf for x in ["lqcd", "dibaryon", "hexaquark", "baryon", "quark"]):
        return "LQCD"

    # Full DL model families
    dl_model_families = {
        "resnet", "resnext", "vgg", "mobilenet", "efficientnet", "convnext",
        "densenet", "inception", "alexnet", "squeezenet", "shufflenet",
        "bert", "albert", "roberta", "distilbert", "deberta", "electra",
        "gpt2", "t5", "bart", "xlnet",
        "vit", "deit", "swin",
        "lstm", "lstm_seq2seq", "bilstm", "gru",
        "gcn", "gin", "gat", "graphsage",
    }
    if mf in dl_model_families:
        return "DL Model"

    # Single operator benchmarks — op_type is the discriminator
    dl_op_types = {
        "matmul", "batch_matmul", "conv_2d", "conv_2d_nchw_fchw",
        "pooling", "max_pool", "avg_pool", "generic",
        "nchw", "ndhwc",
    }
    if any(x in ot for x in dl_op_types):
        return "DL Operator"

    return "Other"


def label_op_type(op_type: str) -> str:
    """Make op_type labels more readable. 'generic' is ambiguous — keep but clarify."""
    if op_type == "generic":
        return "generic (elementwise)"
    return op_type


def parse_benchmark_key(key: str) -> dict:
    """
    Parse keys like:
      albert_sl128_bs16_batch_matmul_0
      convnext_bs32_conv_2d_nchw_fchw_0
    Returns dict with model_family, batch_size, op_type, index.
    seq_len is optional.
    """
    result = {"raw": key, "model_family": None, "sub_family": None, "seq_len": None,
              "batch_size": None, "op_type": None, "bench_category": None, "index": None}
    try:
        # Extract trailing index
        parts = key.rsplit("_", 1)
        if parts[-1].isdigit():
            result["index"] = int(parts[-1])
            key_body = parts[0]
        else:
            key_body = key

        # Extract batch size
        bs_match = re.search(r"_bs(\d+)", key_body)
        if bs_match:
            result["batch_size"] = int(bs_match.group(1))

        # Extract seq_len (optional)
        sl_match = re.search(r"_sl(\d+)", key_body)
        if sl_match:
            result["seq_len"] = int(sl_match.group(1))

        # Extract model family (everything before first _sl or _bs)
        prefix_match = re.match(r"^([^_]+(?:_[^_]+)*?)(?=_sl\d|_bs\d)", key_body)
        if prefix_match:
            result["model_family"] = prefix_match.group(1)
        else:
            result["model_family"] = key_body.split("_")[0]

        # Extract op_type: everything after _bsN_
        op_match = re.search(r"_bs\d+_(.+)$", key_body)
        if op_match:
            result["op_type"] = op_match.group(1)
        else:
            # fallback: last underscore-separated segment(s)
            result["op_type"] = key_body.split("_")[-1]

        # Normalize model family, keep raw parsed name as sub_family
        if result["model_family"]:
            result["sub_family"] = result["model_family"]   # raw before normalization
            result["model_family"] = normalize_family(result["model_family"])

        # Infer benchmark category
        result["bench_category"] = infer_bench_category(
            result["model_family"] or "", result["op_type"] or ""
        )

    except Exception:
        pass
    return result


# Order matters: more specific patterns first
FAMILY_PATTERNS = [
    # mobilenet: mobilenet_v2_s224 → mobilenet
    (r"^mobilenet.*",           "mobilenet"),
    # efficientnet: efficientnet_b0_sz224 → efficientnet
    (r"^efficientnet.*",        "efficientnet"),
    # convnext: convnext_tiny_sz224 → convnext
    (r"^convnext.*",            "convnext"),
    # resnext: resnext50_sz224 → resnext
    (r"^resnext.*",             "resnext"),
    # resnet: resnet18_sz224, resnet50_sz160 → resnet
    (r"^resnet.*",              "resnet"),
    # densenet: densenet121_sz224 → densenet
    (r"^densenet.*",            "densenet"),
    # vit: vit_b_16, vit_l_32 → vit
    (r"^vit.*",                 "vit"),
    # lstm_seq2seq is its own family — must come before generic lstm rule
    (r"^lstm_seq2seq.*",        "lstm_seq2seq"),
    # lstm: remaining lstm variants → lstm
    (r"^lstm.*",                "lstm"),
    # bilstm variants
    (r"^bilstm.*",              "bilstm"),
    # bert variants: bert, distilbert, deberta → keep as-is (already clean)
    # everything else: strip trailing _sz\d+ or _v\d+ or _\d+ suffixes
    (r"^(.+?)(?:_sz\d+|_v\d+|_\d+).*$", None),  # None = use capture group 1
]


def normalize_family(name: str) -> str:
    for pattern, replacement in FAMILY_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            if replacement is not None:
                return replacement
            else:
                # Use capture group 1 (strip suffix)
                m = re.match(pattern, name, re.IGNORECASE)
                return m.group(1) if m else name
    return name


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_plain_floats(path: Path) -> list[float]:
    if not path.exists():
        return []
    try:
        return [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_speedup_folder(folder: Path) -> dict[str, list[float]]:
    """Returns {benchmark_name: [float, ...]} for all files in speedup folder."""
    if not folder.exists():
        return {}
    result = {}
    for f in sorted(folder.iterdir()):
        if f.is_file():
            vals = load_plain_floats(f)
            if vals:
                result[f.name] = vals
    return result


@st.cache_data(show_spinner=False)
def load_exec_time_folder(folder: Path) -> dict[str, list[float]]:
    """Returns {benchmark_name: [float, ...]} for all files in exec_time folder."""
    if not folder.exists():
        return {}
    result = {}
    for f in sorted(folder.iterdir()):
        if f.is_file():
            vals = load_plain_floats(f)
            if vals:
                result[f.name] = vals
    return result


@st.cache_data(show_spinner=False)
def get_checkpoints(run_path: Path) -> list[str]:
    """
    Returns sorted checkpoint names from run_i/models/.
    Tries to sort numerically by any trailing number in the filename.
    """
    models_dir = run_path / "models"
    if not models_dir.exists():
        return []
    ckpts = [f.name for f in sorted(models_dir.iterdir()) if f.suffix == ".pt"]

    def ckpt_sort_key(name: str):
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else name

    return sorted(ckpts, key=ckpt_sort_key)


@st.cache_data(show_spinner=False)
def build_main_df(
    experiment: str,
    base_eval_by_impl: dict[str, dict],
    pytorch: dict,
    exec_time_data_by_impl: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Build a benchmark-level dataframe including both RL implementations.

    exec_time_data_by_impl format:
      {
        "rl_autoschedular": {"bench": [..], ...},
        "new_rl_autoschedular": {"bench": [..], ...}
      }
    """
    rows = []
    all_keys = set(pytorch.keys())
    for impl_base_eval in base_eval_by_impl.values():
        all_keys |= set(impl_base_eval.keys())
    for impl_exec_data in exec_time_data_by_impl.values():
        all_keys |= set(impl_exec_data.keys())

    for key in sorted(all_keys):
        old_baseline_time = base_eval_by_impl.get(DEFAULT_IMPLEMENTATION, {}).get(key)
        new_baseline_time = base_eval_by_impl.get(NEW_IMPLEMENTATION, {}).get(key)
        baseline_time = old_baseline_time if old_baseline_time is not None else new_baseline_time
        meta = parse_benchmark_key(key)
        pt = pytorch.get(key, {})

        row = {
            "benchmark": key,
            "experiment": experiment,
            "bench_category": meta["bench_category"],
            "model_family": meta["model_family"],
            "sub_family": meta["sub_family"],
            "batch_size": meta["batch_size"],
            "op_type": meta["op_type"],
            "op_type_label": label_op_type(meta["op_type"] or ""),
            # Canonical shared baseline value for aggregate charts.
            # Prefer old baseline when both exist.
            "mlir_baseline_us": baseline_time,
            "mlir_old_baseline_us": old_baseline_time,
            "mlir_new_baseline_us": new_baseline_time,
            # PyTorch reference times
            "pytorch_eager_us": pt.get("eager"),
            "pytorch_compile_us": pt.get("compile"),
            "pytorch_jit_us": pt.get("jit"),
        }

        rl_vs_eager_candidates: list[float] = []
        rl_speedup_candidates: list[float] = []

        # Add per-implementation RL metrics
        for impl in IMPL_ORDER:
            suffix = IMPL_SHORT[impl]
            impl_exec_data = exec_time_data_by_impl.get(impl, {})
            rl_vals = impl_exec_data.get(key, [])
            rl_optimized = min(rl_vals) if rl_vals else None
            impl_baseline_time = base_eval_by_impl.get(impl, {}).get(key)
            row[f"rl_{suffix}_optimized_us"] = rl_optimized

            if rl_optimized and impl_baseline_time:
                row[f"rl_{suffix}_speedup_over_baseline"] = round(impl_baseline_time / rl_optimized, 3)
                rl_speedup_candidates.append(row[f"rl_{suffix}_speedup_over_baseline"])
            else:
                row[f"rl_{suffix}_speedup_over_baseline"] = None

            for mode in ["eager", "compile", "jit"]:
                pt_val = pt.get(mode)
                if pt_val and rl_optimized:
                    row[f"rl_{suffix}_vs_{mode}"] = round(pt_val / rl_optimized, 3)
                    row[f"{mode}_vs_rl_{suffix}"] = round(rl_optimized / pt_val, 3)
                    if mode == "eager":
                        rl_vs_eager_candidates.append(row[f"rl_{suffix}_vs_{mode}"])
                else:
                    row[f"rl_{suffix}_vs_{mode}"] = None
                    row[f"{mode}_vs_rl_{suffix}"] = None

        row["best_rl_vs_eager"] = max(rl_vs_eager_candidates) if rl_vs_eager_candidates else None
        row["best_rl_speedup_over_baseline"] = max(rl_speedup_candidates) if rl_speedup_candidates else None

        rows.append(row)
    return pd.DataFrame(rows)


def get_experiment_roots() -> list[Path]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted([
        d for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and (d / "exec_times").exists()
    ])


def get_runs(experiment_root: Path) -> list[str]:
    runs: set[str] = set()
    for impl in IMPL_ORDER:
        agent_dir = experiment_root / IMPL_AGENT_DIR[impl]
        if not agent_dir.exists():
            continue
        runs |= {
            d.name for d in agent_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        }
    return sorted(runs, key=lambda name: int(name.split("_")[1]) if name.split("_")[1].isdigit() else name)


def get_run_paths_by_impl(experiment_root: Path, selected_run: str) -> dict[str, Path]:
    run_paths: dict[str, Path] = {}
    for impl in IMPL_ORDER:
        run_path = experiment_root / IMPL_AGENT_DIR[impl] / selected_run
        if run_path.exists() and run_path.is_dir():
            run_paths[impl] = run_path
    return run_paths


def get_baselines_by_impl(experiment_root: Path) -> dict[str, dict]:
    baselines: dict[str, dict] = {}
    for impl in IMPL_ORDER:
        prefix = IMPL_BASE_PREFIX[impl]
        baselines[impl] = load_json(experiment_root / "exec_times" / f"{prefix}_base_eval.json")
    return baselines


PLOTLY_THEME = dict(
    template="plotly_white",
    font_family="sans-serif",
)

COLOR_RL = "#1a7f4b"
COLOR_EAGER = "#2563eb"
COLOR_COMPILE = "#d97706"
COLOR_JIT = "#7c3aed"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ MLIR-RL")
    st.caption("Evaluation Dashboard")
    st.divider()

    experiment_roots = get_experiment_roots()
    if not experiment_roots:
        st.error(f"No results found at:\n`{RESULTS_ROOT}`")
        st.stop()

    experiments = [p.name for p in experiment_roots]

    st.markdown("**Run Selection**")
    experiment = st.selectbox("Experiment", experiments)
    experiment_root = next(p for p in experiment_roots if p.name == experiment)
    runs = get_runs(experiment_root)
    if not runs:
        st.warning("No runs found for this experiment.")
        st.stop()
    selected_run = st.selectbox("Run", runs)

    st.divider()
    st.markdown("**Filters**")

    run_paths_by_impl = get_run_paths_by_impl(experiment_root, selected_run)
    if not run_paths_by_impl:
        st.warning("Selected run does not exist in old_agent/new_agent.")
        st.stop()

    # Load data for filter population
    base_eval_by_impl = get_baselines_by_impl(experiment_root)
    pytorch_data = load_json(experiment_root / "exec_times" / "pytorch.json")
    exec_time_data_by_impl = {
        impl: load_exec_time_folder(run_path / "logs" / "eval" / "exec_time")
        for impl, run_path in run_paths_by_impl.items()
    }
    df_full = build_main_df(experiment, base_eval_by_impl, pytorch_data, exec_time_data_by_impl)

    all_batch_sizes = sorted(df_full["batch_size"].dropna().unique().tolist())
    all_op_types = sorted(df_full["op_type"].dropna().unique().tolist())
    all_categories = sorted(df_full["bench_category"].dropna().unique().tolist())

    sel_categories = st.multiselect("Benchmark Category", all_categories, default=all_categories)
    sel_batch = st.multiselect("Batch Size", all_batch_sizes, default=all_batch_sizes)
    sel_ops = st.multiselect("Op Type", all_op_types, default=all_op_types)
    speedup_min = st.slider("Min RL speedup vs MLIR baseline", 0.0, 10.0, 0.0, 0.1)

    st.divider()
    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
df = df_full.copy()
if sel_categories:
    df = df[df["bench_category"].isin(sel_categories)]
if sel_batch:
    df = df[df["batch_size"].isin(sel_batch)]
if sel_ops:
    df = df[df["op_type"].isin(sel_ops)]
if speedup_min > 0:
    df = df[df["best_rl_speedup_over_baseline"].fillna(0) >= speedup_min]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_title, col_run = st.columns([3, 1])
with col_title:
    st.markdown(f"# MLIR-RL Evaluation")
    impls_shown = [IMPL_DISPLAY[impl] for impl in IMPL_ORDER if impl in run_paths_by_impl]
    impls_text = ", ".join(impls_shown) if impls_shown else "No RL implementations"
    st.caption(
        f"Implementations: **{impls_text}** · "
        f"Experiment: **{experiment}** · Run: **{selected_run}** · {len(df)} benchmarks shown"
    )
with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    results_path = experiment_root / "exec_times"
    st.caption(f"`{results_path.relative_to(RESULTS_ROOT.parent)}`")

st.divider()


# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
def metric_card(label, value, cls=""):
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {cls}">{value}</div>
    </div>"""

def impl_metrics(df_in: pd.DataFrame, suffix: str) -> dict[str, float | int | bool | None]:
    opt_col = f"rl_{suffix}_optimized_us"
    eager_col = f"rl_{suffix}_vs_eager"
    compile_col = f"rl_{suffix}_vs_compile"
    jit_col = f"rl_{suffix}_vs_jit"

    available = opt_col in df_in.columns and df_in[opt_col].notna().any()
    if not available:
        return {
            "available": False,
            "mean": None,
            "beats_all": 0,
        }

    beats_all = df_in[
        (df_in[eager_col].fillna(0) > 1) &
        (df_in[compile_col].fillna(0) > 1) &
        (df_in[jit_col].fillna(0) > 1)
    ].shape[0]

    return {
        "available": True,
        "mean": df_in[eager_col].mean(),
        "beats_all": beats_all,
    }


old_m = impl_metrics(df, "old")
new_m = impl_metrics(df, "new")
best_any = df["best_rl_vs_eager"].max() if "best_rl_vs_eager" in df.columns else None
worst_any = df["best_rl_vs_eager"].min() if "best_rl_vs_eager" in df.columns else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Avg Old RL vs PyTorch Eager",
        f"{old_m['mean']:.2f}×" if old_m["available"] and pd.notna(old_m["mean"]) else "N/A",
        "good" if old_m["available"] and pd.notna(old_m["mean"]) and old_m["mean"] > 1 else "warn"),
        unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Avg New RL vs PyTorch Eager",
        f"{new_m['mean']:.2f}×" if new_m["available"] and pd.notna(new_m["mean"]) else "N/A",
        "good" if new_m["available"] and pd.notna(new_m["mean"]) and new_m["mean"] > 1 else "warn"),
        unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Old RL Beats All PyTorch Modes",
        f"{old_m['beats_all']} / {len(df)}" if old_m["available"] else "N/A",
        "good" if old_m["available"] and old_m["beats_all"] > 0 else "warn"),
        unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("New RL Beats All PyTorch Modes",
        f"{new_m['beats_all']} / {len(df)}" if new_m["available"] else "N/A",
        "good" if new_m["available"] and new_m["beats_all"] > 0 else "warn"),
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⏱  Execution Times", "🚀  Speedup Analysis", "📈  Training Curves", "🔖  Checkpoints"])


# ════════════════════════════════════════════
# TAB 1 — EXECUTION TIME COMPARISON
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Execution Time Comparison: RL (Optimized) vs MLIR Baseline vs PyTorch")
    st.caption(
        "All exec times in µs. **RL Optimized** = best schedule found by the agent (last eval checkpoint). "
        "**MLIR Baseline** = unoptimized MLIR + O3, no loop transforms. Lower is better."
    )

    if df.empty:
        st.info("No benchmarks match the current filters.")
    else:
        mode_labels = {
            "rl_old_optimized_us": "Old RL Optimized",
            "rl_new_optimized_us": "New RL Optimized",
            "mlir_baseline_us":   "MLIR Baseline",
            "pytorch_eager_us":   "PyTorch Eager",
            "pytorch_compile_us": "PyTorch Compile",
            "pytorch_jit_us":     "PyTorch JIT",
        }
        COLOR_BASELINE = "#94a3b8"
        color_map = {
            "Old RL Optimized": "#1a7f4b",
            "New RL Optimized": "#0f766e",
            "MLIR Baseline":   COLOR_BASELINE,
            "PyTorch Eager":   COLOR_EAGER,
            "PyTorch Compile": COLOR_COMPILE,
            "PyTorch JIT":     COLOR_JIT,
        }

        # ── Chart 1: per model family ──────────────────────────
        agg_cols = ["model_family", "rl_old_optimized_us", "rl_new_optimized_us", "mlir_baseline_us",
                    "pytorch_eager_us", "pytorch_compile_us", "pytorch_jit_us"]
        df_agg = (
            df[agg_cols]
            .groupby("model_family", as_index=False)
            .mean(numeric_only=True)
        )

        all_families = sorted(df_agg["model_family"].dropna().unique())
        sel_families = st.multiselect(
            "Model Families",
            options=all_families,
            default=all_families,
            key="tab1_family_filter"
        )

        df_agg_filtered = df_agg[df_agg["model_family"].isin(sel_families)]

        if df_agg_filtered.empty:
            st.info("Select at least one model family.")
        else:
            df_melt = df_agg_filtered.melt(
                id_vars="model_family",
                var_name="mode", value_name="exec_time_us"
            ).dropna(subset=["exec_time_us"])
            df_melt["mode"] = df_melt["mode"].map(mode_labels)

            # Preserve logical bar order
            mode_order = ["Old RL Optimized", "New RL Optimized", "MLIR Baseline",
                          "PyTorch Eager", "PyTorch Compile", "PyTorch JIT"]
            df_melt["mode"] = pd.Categorical(
                df_melt["mode"], categories=mode_order, ordered=True
            )
            df_melt = df_melt.sort_values("mode")

            fig = px.bar(
                df_melt,
                x="model_family", y="exec_time_us", color="mode",
                barmode="group",
                color_discrete_map=color_map,
                category_orders={"mode": mode_order},
                labels={"model_family": "Model Family",
                        "exec_time_us": "Avg Exec Time (µs)", "mode": ""},
                height=420,
            )
            fig.update_layout(
                **PLOTLY_THEME,
                xaxis_tickangle=-20,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=40, b=60),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        # ── Chart 2: sub-family drill-down ─────────────────────
        st.markdown("### Sub-family Drill-down")
        st.caption("Select a family to inspect its sub-families (e.g. resnet18, resnet50).")

        drill_family = st.selectbox(
            "Model Family",
            options=all_families,
            key="tab1_drill_family"
        )

        df_drill = df[df["model_family"] == drill_family].copy()
        all_subfamilies = sorted(df_drill["sub_family"].dropna().unique())

        if len(all_subfamilies) <= 1:
            st.info(f"**{drill_family}** has no sub-families to compare.")
        else:
            sel_subfamilies = st.multiselect(
                "Sub-families",
                options=all_subfamilies,
                default=all_subfamilies,
                key="tab1_subfamily_filter"
            )

            df_drill = df_drill[df_drill["sub_family"].isin(sel_subfamilies)]

            if df_drill.empty:
                st.info("Select at least one sub-family.")
            else:
                agg_sub_cols = ["sub_family", "rl_old_optimized_us", "rl_new_optimized_us", "mlir_baseline_us",
                                "pytorch_eager_us", "pytorch_compile_us", "pytorch_jit_us"]
                df_sub_agg = (
                    df_drill[agg_sub_cols]
                    .groupby("sub_family", as_index=False)
                    .mean(numeric_only=True)
                )
                df_sub_melt = df_sub_agg.melt(
                    id_vars="sub_family",
                    var_name="mode", value_name="exec_time_us"
                ).dropna(subset=["exec_time_us"])
                df_sub_melt["mode"] = df_sub_melt["mode"].map(mode_labels)
                df_sub_melt["mode"] = pd.Categorical(
                    df_sub_melt["mode"], categories=mode_order, ordered=True
                )
                df_sub_melt = df_sub_melt.sort_values("mode")

                fig_sub = px.bar(
                    df_sub_melt,
                    x="sub_family", y="exec_time_us", color="mode",
                    barmode="group",
                    color_discrete_map=color_map,
                    category_orders={"mode": mode_order},
                    labels={"sub_family": "Sub-family",
                            "exec_time_us": "Avg Exec Time (µs)", "mode": ""},
                    height=380,
                )
                fig_sub.update_layout(
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=40, b=60),
                )
                fig_sub.update_traces(marker_line_width=0)
                st.plotly_chart(fig_sub, use_container_width=True)

        # ── Data table ─────────────────────────────────────────
        st.markdown("#### Data Table")
        st.caption(
            "`rl_old_vs_eager` / `rl_new_vs_eager` > 1 means that RL implementation is faster than PyTorch Eager. "
            "`rl_old_speedup_over_baseline` / `rl_new_speedup_over_baseline` show improvement over unoptimized MLIR."
        )
        display_cols = [
            "benchmark", "bench_category", "model_family", "sub_family", "op_type_label",
            "batch_size",
            "rl_old_optimized_us", "rl_new_optimized_us", "mlir_baseline_us",
            "mlir_old_baseline_us", "mlir_new_baseline_us",
            "pytorch_eager_us", "pytorch_compile_us", "pytorch_jit_us",
            "rl_old_speedup_over_baseline", "rl_new_speedup_over_baseline",
            "rl_old_vs_eager", "rl_old_vs_compile", "rl_old_vs_jit",
            "rl_new_vs_eager", "rl_new_vs_compile", "rl_new_vs_jit",
            "best_rl_vs_eager",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("best_rl_vs_eager", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )

        # Export
        st.markdown("#### Export")
        ec1, ec2 = st.columns(2)
        with ec1:
            csv = df[display_cols].to_csv(index=False)
            st.download_button("⬇ Download CSV", csv, "exec_times.csv", "text/csv",
                               use_container_width=True)
        with ec2:
            raw_json = json.dumps({"base_eval": base_eval_by_impl, "pytorch": pytorch_data}, indent=2)
            st.download_button("⬇ Download JSON", raw_json, "exec_data.json", "application/json",
                               use_container_width=True)



# ════════════════════════════════════════════
# TAB 2 — SPEEDUP ANALYSIS
# ════════════════════════════════════════════
with tab2:
    speedup_per_bench_by_impl = {
        impl: load_speedup_folder(run_path / "logs" / "eval" / "speedup")
        for impl, run_path in run_paths_by_impl.items()
    }
    avg_speedup_by_impl = {
        impl: load_plain_floats(run_path / "logs" / "eval" / "average_speedup")
        for impl, run_path in run_paths_by_impl.items()
    }

    st.markdown("### RL Speedup over Unoptimized MLIR Baseline")
    st.caption(
        "Speedup = unoptimized MLIR baseline exec time / optimized exec time. "
        "**> 1 means the RL agent improved over the baseline.** "
        "This tab overlays old/new RL implementations when both are available."
    )

    any_speedup_data = any(bool(v) for v in speedup_per_bench_by_impl.values())
    if any_speedup_data:
        rows_sp = []
        for impl, speedup_per_bench in speedup_per_bench_by_impl.items():
            best_per_bench = {k: max(v) for k, v in speedup_per_bench.items() if v}
            for k, sp in best_per_bench.items():
                meta = parse_benchmark_key(k)
                rows_sp.append({
                    "benchmark": k,
                    "speedup": sp,
                    "implementation": IMPL_DISPLAY.get(impl, impl),
                    "op_type": label_op_type(meta.get("op_type") or "unknown"),
                    "model_family": meta.get("model_family") or "unknown",
                    "bench_category": meta.get("bench_category") or "Other",
                })

        df_sp = pd.DataFrame(rows_sp)
        if df_sp.empty:
            st.info("No speedup values found in selected run.")
        else:
            df_sp = df_sp.dropna(subset=["speedup"])

            all_op_types_sp = sorted(df_sp["op_type"].unique())
            sel_op = st.selectbox(
                "Operation Type",
                options=all_op_types_sp,
                key="tab2_op_filter"
            )

            df_sp_filtered = df_sp[df_sp["op_type"] == sel_op]

            if df_sp_filtered.empty:
                st.info(f"No speedup data for op type `{sel_op}`.")
            else:
                # Average best speedup per model family for selected op type and implementation
                df_sp_agg = (
                    df_sp_filtered
                    .groupby(["model_family", "implementation"], as_index=False)["speedup"]
                    .mean()
                    .rename(columns={"speedup": "avg_speedup"})
                    .sort_values("avg_speedup", ascending=False)
                )

                st.markdown(f"#### Avg Best Speedup per Model Family — `{sel_op}`")
                st.caption("Average of best-per-benchmark speedups across all benchmarks of each model family.")

                fig2 = px.bar(
                    df_sp_agg, x="model_family", y="avg_speedup", color="implementation",
                    barmode="group",
                    labels={
                        "model_family": "Model Family",
                        "avg_speedup": "Avg Best Speedup over MLIR Baseline (×)",
                        "implementation": "",
                    },
                    height=400,
                )
                fig2.add_hline(y=1.0, line_dash="dash", line_color="#888",
                               annotation_text="No improvement (1×)", annotation_position="top right")
                fig2.update_layout(
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    margin=dict(l=0, r=0, t=40, b=60),
                )
                fig2.update_traces(marker_line_width=0)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No speedup files found for selected run in any implementation.")

    st.markdown("#### Average RL Speedup over Eval Steps")
    st.caption("Tracks how average speedup evolves over eval steps for each RL implementation.")

    any_avg = any(bool(vals) for vals in avg_speedup_by_impl.values())
    if any_avg:
        fig3 = go.Figure()
        for impl in IMPL_ORDER:
            vals = avg_speedup_by_impl.get(impl, [])
            if not vals:
                continue
            fig3.add_trace(go.Scatter(
                x=list(range(len(vals))), y=vals,
                mode="lines",
                name=IMPL_DISPLAY.get(impl, impl),
                line=dict(width=2),
            ))

        fig3.add_hline(y=1.0, line_dash="dash", line_color="#888",
                       annotation_text="No improvement", annotation_position="top right")
        fig3.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
        fig3.update_xaxes(title_text="Eval Step")
        fig3.update_yaxes(title_text="Avg Speedup over MLIR Baseline (×)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No `average_speedup` files found for selected run.")

    # Per-benchmark speedup evolution
    if any_speedup_data:
        st.markdown("### Speedup Evolution per Benchmark")
        bench_options = sorted({
            bench
            for data in speedup_per_bench_by_impl.values()
            for bench in data.keys()
        })
        selected_bench = st.multiselect(
            "Select benchmarks to compare",
            bench_options,
            default=bench_options[:min(5, len(bench_options))]
        )
        if selected_bench:
            fig4 = go.Figure()
            for impl in IMPL_ORDER:
                impl_data = speedup_per_bench_by_impl.get(impl, {})
                for bench in selected_bench:
                    if bench not in impl_data:
                        continue
                    vals = impl_data[bench]
                    fig4.add_trace(go.Scatter(
                        x=list(range(len(vals))), y=vals,
                        mode="lines", name=f"{bench} ({IMPL_DISPLAY.get(impl, impl)})",
                        line=dict(width=1.5),
                    ))
            fig4.add_hline(y=1.0, line_dash="dash", line_color="#5a6480")
            fig4.update_layout(
                **PLOTLY_THEME,
                xaxis_title="Eval Step", yaxis_title="Speedup (×)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=40),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Export
    if any_speedup_data:
        st.markdown("#### Export")
        sp_csv = pd.DataFrame([
            {
                "implementation": IMPL_DISPLAY.get(impl, impl),
                "benchmark": k,
                "step": i,
                "speedup": v,
            }
            for impl, speedup_per_bench in speedup_per_bench_by_impl.items()
            for k, vals in speedup_per_bench.items()
            for i, v in enumerate(vals)
        ]).to_csv(index=False)
        st.download_button("⬇ Download Speedup CSV", sp_csv, "speedup_evolution.csv",
                           "text/csv", use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — TRAINING CURVES
# ════════════════════════════════════════════
with tab3:
    st.markdown("### Training Curves")

    signals = {
        "Reward": ("reward", "#4a9eff"),
        "Cumulative Reward": ("cumulative_reward", "#c084fc"),
        "Policy Entropy": ("entropy", "#f5a623"),
    }

    any_data = False
    for title, (fname, _) in signals.items():
        fig = go.Figure()
        has_signal = False
        for impl in IMPL_ORDER:
            run_path = run_paths_by_impl.get(impl)
            if run_path is None:
                continue
            vals = load_plain_floats(run_path / "logs" / "eval" / fname)
            if not vals:
                continue
            has_signal = True
            any_data = True
            fig.add_trace(go.Scatter(
                x=list(range(len(vals))), y=vals,
                mode="lines",
                name=IMPL_DISPLAY.get(impl, impl),
                line=dict(width=2),
            ))

        if not has_signal:
            st.caption(f"No data for `{fname}`")
            continue

        fig.update_layout(
            **PLOTLY_THEME,
            title=title,
            title_font_size=13,
            margin=dict(l=0, r=0, t=40, b=30),
        )
        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

    if not any_data:
        st.info("No eval training curves found for selected run in any implementation.")

    # Also show train logs if present
    train_signals = {
        "Train Reward": ("reward", "#3dd68c"),
        "Train Entropy": ("entropy", "#f5a623"),
    }
    train_any = False
    for title, (fname, _) in train_signals.items():
        fig = go.Figure()
        has_signal = False
        for impl in IMPL_ORDER:
            run_path = run_paths_by_impl.get(impl)
            if run_path is None:
                continue
            vals = load_plain_floats(run_path / "logs" / "train" / fname)
            if not vals:
                continue
            has_signal = True
            train_any = True
            fig.add_trace(go.Scatter(
                x=list(range(len(vals))), y=vals,
                mode="lines",
                name=IMPL_DISPLAY.get(impl, impl),
                line=dict(width=2),
            ))

        if not has_signal:
            continue

        if not train_any:
            st.markdown("### Training Logs")
            train_any = True
        fig.update_layout(
            **PLOTLY_THEME,
            title=title,
            title_font_size=13,
            margin=dict(l=0, r=0, t=40, b=30),
        )
        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

    # Export training data
    export_data = {}
    for impl, run_path in run_paths_by_impl.items():
        impl_key = IMPL_DISPLAY.get(impl, impl)
        export_data[impl_key] = {}
        for fname in ["reward", "cumulative_reward", "entropy"]:
            vals = load_plain_floats(run_path / "logs" / "eval" / fname)
            if vals:
                export_data[impl_key][f"eval/{fname}"] = vals
        for fname in ["reward", "entropy"]:
            vals = load_plain_floats(run_path / "logs" / "train" / fname)
            if vals:
                export_data[impl_key][f"train/{fname}"] = vals

    export_data = {k: v for k, v in export_data.items() if v}

    if export_data:
        st.markdown("#### Export")
        curves_json = json.dumps(export_data, indent=2)
        st.download_button("⬇ Download Training Curves JSON", curves_json,
                           "training_curves.json", "application/json",
                           use_container_width=True)


# ════════════════════════════════════════════
# TAB 4 — CHECKPOINT EVOLUTION
# ════════════════════════════════════════════
with tab4:
    st.markdown("### Agent Performance across Checkpoints")
    st.caption(
        "Each checkpoint corresponds to a saved `.pt` model in `run_i/models/`. "
        "Exec times are read from `run_i/logs/eval/exec_time/` — one value per checkpoint per benchmark. "
        "This tab overlays old/new implementations when both are available."
    )

    ckpt_data_by_impl: dict[str, dict[str, object]] = {}
    for impl, run_path in run_paths_by_impl.items():
        ckpts = get_checkpoints(run_path)
        exec_data = load_exec_time_folder(run_path / "logs" / "eval" / "exec_time")
        if not ckpts or not exec_data:
            continue
        ckpt_data_by_impl[impl] = {
            "run_path": run_path,
            "checkpoints": ckpts,
            "exec_time_data": exec_data,
        }

    if not ckpt_data_by_impl:
        st.info("No checkpoint/eval exec-time data found for selected run in any implementation.")
    else:
        # ── Average exec time evolution across all benchmarks ──
        st.markdown("#### Average Exec Time over Checkpoints")
        fig_avg = go.Figure()
        for impl in IMPL_ORDER:
            if impl not in ckpt_data_by_impl:
                continue
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
            assert isinstance(checkpoints, list)
            assert isinstance(exec_time_data, dict)

            all_series = [v for v in exec_time_data.values() if len(v) >= 1]
            if not all_series:
                continue

            max_len = max(len(s) for s in all_series)
            avg_per_ckpt = []
            for i in range(max_len):
                vals_at_i = [s[i] for s in all_series if i < len(s)]
                avg_per_ckpt.append(sum(vals_at_i) / len(vals_at_i))

            x_labels = checkpoints[:max_len] if max_len <= len(checkpoints) else list(range(max_len))
            fig_avg.add_trace(go.Scatter(
                x=list(range(max_len)), y=avg_per_ckpt,
                mode="lines+markers",
                marker=dict(size=5),
                name=IMPL_DISPLAY.get(impl, impl),
                text=x_labels,
                hovertemplate="Implementation: %{fullData.name}<br>Checkpoint: %{text}<br>Avg Exec Time: %{y:.0f} µs<extra></extra>",
            ))

        fig_avg.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
        fig_avg.update_xaxes(title_text="Checkpoint Index")
        fig_avg.update_yaxes(title_text="Avg Exec Time (µs)")
        st.plotly_chart(fig_avg, use_container_width=True)

        # ── Per-benchmark exec time evolution ──
        st.markdown("#### Per-Benchmark Exec Time Evolution")
        bench_names = sorted({
            bench
            for item in ckpt_data_by_impl.values()
            for bench in item["exec_time_data"].keys()
        })
        selected_benches = st.multiselect(
            "Select benchmarks",
            bench_names,
            default=bench_names[:min(6, len(bench_names))],
            key="ckpt_bench_select"
        )

        if selected_benches:
            fig_ckpt = go.Figure()
            for impl in IMPL_ORDER:
                if impl not in ckpt_data_by_impl:
                    continue
                checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
                exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
                assert isinstance(checkpoints, list)
                assert isinstance(exec_time_data, dict)

                for bench in selected_benches:
                    if bench not in exec_time_data:
                        continue
                    vals = exec_time_data[bench]
                    x_idx = list(range(len(vals)))
                    hover = checkpoints[:len(vals)] if len(vals) <= len(checkpoints) else x_idx
                    fig_ckpt.add_trace(go.Scatter(
                        x=x_idx, y=vals,
                        mode="lines+markers",
                        name=f"{bench} ({IMPL_DISPLAY.get(impl, impl)})",
                        line=dict(width=1.8),
                        marker=dict(size=4),
                        hovertemplate="Checkpoint: %{text}<br>Exec Time: %{y:.0f} µs<extra>%{fullData.name}</extra>",
                        text=hover,
                    ))
            fig_ckpt.update_layout(
                **PLOTLY_THEME,
                xaxis_title="Checkpoint Index",
                yaxis_title="Exec Time (µs)",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=40),
            )
            st.plotly_chart(fig_ckpt, use_container_width=True)

        # ── Checkpoint list ──
        for impl in IMPL_ORDER:
            if impl not in ckpt_data_by_impl:
                continue
            run_path = ckpt_data_by_impl[impl]["run_path"]
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            assert isinstance(run_path, Path)
            assert isinstance(checkpoints, list)
            with st.expander(f"📁 {IMPL_DISPLAY.get(impl, impl)}: {len(checkpoints)} checkpoints in `{run_path / 'models'}`"):
                for i, ckpt in enumerate(checkpoints):
                    st.caption(f"`[{i}]` {ckpt}")

        # ── Export ──
        st.markdown("#### Export")
        ckpt_rows = []
        for impl in IMPL_ORDER:
            if impl not in ckpt_data_by_impl:
                continue
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
            assert isinstance(checkpoints, list)
            assert isinstance(exec_time_data, dict)
            for bench, vals in exec_time_data.items():
                for i, v in enumerate(vals):
                    ckpt_rows.append({
                        "implementation": IMPL_DISPLAY.get(impl, impl),
                        "benchmark": bench,
                        "checkpoint_index": i,
                        "checkpoint": checkpoints[i] if i < len(checkpoints) else i,
                        "exec_time_us": v,
                    })

        ckpt_csv = pd.DataFrame(ckpt_rows).to_csv(index=False)
        st.download_button("⬇ Download Checkpoint CSV", ckpt_csv,
                           "checkpoint_evolution.csv", "text/csv",
                           use_container_width=True)