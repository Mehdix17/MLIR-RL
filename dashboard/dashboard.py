"""
MLIR-RL Evaluation Dashboard
Visualizes RL agent performance vs PyTorch baselines across benchmarks.
"""
import json
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.stats import wilcoxon as _wilcoxon

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).parent.parent / "results"
LEGACY_AGENT_DIR = {
    "old": "old_agent",
    "new": "new_agent",
}
LEGACY_DISPLAY = {
    "old": "Baseline RL",
    "new": "New RL",
}

st.set_page_config(
    page_title="MLIR-RL Dashboard",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def infer_bench_category(model_family: str, op_type: str) -> str:
    mf = (model_family or "").lower()
    ot = (op_type or "").lower()
    if any(x in mf for x in ["lqcd", "dibaryon", "hexaquark", "baryon", "quark"]):
        return "LQCD"
    dl_model_families = {
        "resnet",
        "resnext",
        "vgg",
        "mobilenet",
        "efficientnet",
        "convnext",
        "densenet",
        "inception",
        "alexnet",
        "squeezenet",
        "shufflenet",
        "bert",
        "albert",
        "roberta",
        "distilbert",
        "deberta",
        "electra",
        "gpt2",
        "t5",
        "bart",
        "xlnet",
        "vit",
        "deit",
        "swin",
        "lstm",
        "lstm_seq2seq",
        "bilstm",
        "gru",
        "gcn",
        "gin",
        "gat",
        "graphsage",
    }
    if mf in dl_model_families:
        return "DL Model"
    dl_op_types = {
        "matmul",
        "batch_matmul",
        "conv_2d",
        "conv_2d_nchw_fchw",
        "pooling",
        "max_pool",
        "avg_pool",
        "generic",
        "nchw",
        "ndhwc",
    }
    if any(x in ot for x in dl_op_types):
        return "DL Operator"
    return "Other"


def label_op_type(op_type: str) -> str:
    if op_type == "generic":
        return "generic (elementwise)"
    return op_type


def parse_benchmark_key(key: str) -> dict:
    result = {
        "raw": key,
        "model_family": None,
        "sub_family": None,
        "seq_len": None,
        "batch_size": None,
        "op_type": None,
        "bench_category": None,
        "index": None,
    }
    try:
        parts = key.rsplit("_", 1)
        if parts[-1].isdigit():
            result["index"] = int(parts[-1])
            key_body = parts[0]
        else:
            key_body = key
        bs_match = re.search(r"_bs(\d+)", key_body)
        if bs_match:
            result["batch_size"] = int(bs_match.group(1))
        sl_match = re.search(r"_sl(\d+)", key_body)
        if sl_match:
            result["seq_len"] = int(sl_match.group(1))
        prefix_match = re.match(r"^([^_]+(?:_[^_]+)*?)(?=_sl\d|_bs\d)", key_body)
        if prefix_match:
            result["model_family"] = prefix_match.group(1)
        else:
            result["model_family"] = key_body.split("_")[0]
        op_match = re.search(r"_bs\d+_(.+)$", key_body)
        if op_match:
            result["op_type"] = op_match.group(1)
        else:
            result["op_type"] = key_body.split("_")[-1]
        if result["model_family"]:
            result["sub_family"] = result["model_family"]
            result["model_family"] = normalize_family(result["model_family"])
        result["bench_category"] = infer_bench_category(
            result["model_family"] or "", result["op_type"] or ""
        )
    except Exception:
        pass
    return result


FAMILY_PATTERNS = [
    (r"^mobilenet.*", "mobilenet"),
    (r"^efficientnet.*", "efficientnet"),
    (r"^convnext.*", "convnext"),
    (r"^resnext.*", "resnext"),
    (r"^resnet.*", "resnet"),
    (r"^densenet.*", "densenet"),
    (r"^vit.*", "vit"),
    (r"^lstm_seq2seq.*", "lstm_seq2seq"),
    (r"^lstm.*", "lstm"),
    (r"^bilstm.*", "bilstm"),
    (r"^(.+?)(?:_sz\d+|_v\d+|_\d+).*$", None),
]


def normalize_family(name: str) -> str:
    for pattern, replacement in FAMILY_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            if replacement is not None:
                return replacement
            m = re.match(pattern, name, re.IGNORECASE)
            return m.group(1) if m else name
    return name


def sort_impl_tokens(tokens: list[str]) -> list[str]:
    def key(token: str):
        if token == "old":
            return (0, 0)
        if token == "new":
            return (1, 0)
        v_match = re.fullmatch(r"v(\d+)", token)
        if v_match:
            return (2, int(v_match.group(1)))
        return (3, token)

    return sorted(tokens, key=key)


def token_to_agent_dir(token: str) -> str:
    return LEGACY_AGENT_DIR.get(token, f"{token}_agent")


def token_display(token: str) -> str:
    if token in LEGACY_DISPLAY:
        return LEGACY_DISPLAY[token]
    if re.fullmatch(r"v\d+", token):
        return f"{token.upper()} RL"
    return token


def token_col(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", token).strip("_").lower()


def parse_schedule_string(s: str) -> dict:
    """Parse a schedule string into counts of each transformation type.
    Returns {"I": n, "T": n, "V": n, "TP": n, "TPF": n, "NT": n, "total": n}
    """
    counts = {"I": 0, "T": 0, "V": 0, "TP": 0, "TPF": 0, "NT": 0}
    pattern = r"\b(TPF|TP|I|T|V|NT)\("
    for m in re.finditer(pattern, s):
        kind = m.group(1)
        if kind in counts:
            counts[kind] += 1
    counts["total"] = sum(counts.values())
    return counts


def compute_rolling_stats(values: list[float], window: int = 10) -> tuple[list[float], list[float]]:
    """Rolling mean and std. Returns (means, stds) same length as input."""
    s = pd.Series(values)
    rolling = s.rolling(window, min_periods=1)
    return rolling.mean().tolist(), rolling.std().tolist()


# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
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
def load_eval_exec_times_json(run_path: Path) -> dict[str, list[float]]:
    """Load {prefix}_agent/run_N/logs/eval/eval_exec_times.json → {bench: [time_ns]}"""
    path = run_path / "logs" / "eval" / "eval_exec_times.json"
    if not path.exists():
        return {}
    data = load_json(path)
    if not data:
        return {}
    return {k: [v] for k, v in data.items() if v is not None and v > 0}


def load_agent_exec_times(run_path: Path) -> dict[str, list[float]]:
    """Unified exec time loader: tries eval_exec_times.json first, falls back to per-benchmark folder."""
    result = load_eval_exec_times_json(run_path)
    if result:
        return result
    return load_exec_time_folder(run_path / "logs" / "eval" / "exec_time")


@st.cache_data(show_spinner=False)
def load_exec_data_json(run_path: Path) -> dict:
    """Load {prefix}_agent/run_N/exec_data.json"""
    path = run_path / "exec_data.json"
    return load_json(path)


@st.cache_data(show_spinner=False)
def get_checkpoints(run_path: Path) -> list[str]:
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
    impl_tokens: tuple[str, ...],
    base_eval_by_impl: dict[str, dict],
    pytorch: dict,
    exec_time_data_by_impl: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    rows = []
    all_keys = set(pytorch.keys())
    for impl_base_eval in base_eval_by_impl.values():
        all_keys |= set(impl_base_eval.keys())
    for impl_exec_data in exec_time_data_by_impl.values():
        all_keys |= set(impl_exec_data.keys())

    for key in sorted(all_keys):
        baseline_time = None
        for token in impl_tokens:
            candidate = base_eval_by_impl.get(token, {}).get(key)
            if candidate is not None:
                baseline_time = candidate
                break

        meta = parse_benchmark_key(key)
        pt = pytorch.get(key, {})

        row = {
            "benchmark": key,
            "experiment": experiment,
            "bench_category": meta["bench_category"],
            "model_family": meta["model_family"],
            "sub_family": meta["sub_family"],
            "batch_size": meta["batch_size"],
            "seq_len": meta["seq_len"],
            "op_type": meta["op_type"],
            "op_type_label": label_op_type(meta["op_type"] or ""),
            "mlir_baseline_us": baseline_time,
            "pytorch_eager_us": pt.get("eager"),
            "pytorch_compile_us": pt.get("compile"),
            "pytorch_jit_us": pt.get("jit"),
        }

        rl_vs_eager_candidates: list[float] = []
        rl_speedup_candidates: list[float] = []

        for token in impl_tokens:
            suffix = token_col(token)
            impl_exec_data = exec_time_data_by_impl.get(token, {})
            rl_vals = impl_exec_data.get(key, [])
            rl_optimized = min(rl_vals) if rl_vals else None
            impl_baseline_time = base_eval_by_impl.get(token, {}).get(key)

            row[f"mlir_{suffix}_baseline_us"] = impl_baseline_time
            row[f"rl_{suffix}_optimized_us"] = rl_optimized

            if rl_optimized and impl_baseline_time:
                sp = round(impl_baseline_time / rl_optimized, 3)
                row[f"rl_{suffix}_speedup_over_baseline"] = sp
                rl_speedup_candidates.append(sp)
            else:
                row[f"rl_{suffix}_speedup_over_baseline"] = None

            for mode in ["eager", "compile", "jit"]:
                pt_val = pt.get(mode)
                if pt_val and rl_optimized:
                    # PyTorch times are in µs, RL/MLIR times are in ns — convert
                    pt_ns = pt_val * 1000
                    row[f"rl_{suffix}_vs_{mode}"] = round(pt_ns / rl_optimized, 3)
                    row[f"{mode}_vs_rl_{suffix}"] = round(rl_optimized / pt_ns, 3)
                    if mode == "eager":
                        rl_vs_eager_candidates.append(row[f"rl_{suffix}_vs_{mode}"])
                else:
                    row[f"rl_{suffix}_vs_{mode}"] = None
                    row[f"{mode}_vs_rl_{suffix}"] = None

        row["best_rl_vs_eager"] = (
            max(rl_vs_eager_candidates) if rl_vs_eager_candidates else None
        )
        row["best_rl_speedup_over_baseline"] = (
            max(rl_speedup_candidates) if rl_speedup_candidates else None
        )

        rows.append(row)
    return pd.DataFrame(rows)


def get_experiment_roots() -> list[Path]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(
        [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and (d / "exec_times").exists()]
    )


def discover_impl_tokens(experiment_root: Path) -> list[str]:
    tokens: set[str] = set()
    for d in experiment_root.iterdir():
        if d.is_dir() and d.name.endswith("_agent"):
            if d.name == "old_agent":
                tokens.add("old")
            elif d.name == "new_agent":
                tokens.add("new")
            else:
                tokens.add(d.name.removesuffix("_agent"))
    exec_times_dir = experiment_root / "exec_times"
    if exec_times_dir.exists():
        for f in exec_times_dir.glob("*_base_eval.json"):
            tokens.add(f.name.removesuffix("_base_eval.json"))
    return sort_impl_tokens(list(tokens))


def get_runs(experiment_root: Path, impl_tokens: list[str]) -> list[str]:
    runs: set[str] = set()
    for token in impl_tokens:
        agent_dir = experiment_root / token_to_agent_dir(token)
        if not agent_dir.exists():
            continue
        runs |= {
            d.name
            for d in agent_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        }
    return sorted(
        runs,
        key=lambda name: int(name.split("_")[1])
        if name.split("_")[1].isdigit()
        else name,
    )


def get_run_paths_by_impl(
    experiment_root: Path, selected_run: str, impl_tokens: list[str]
) -> dict[str, Path]:
    run_paths: dict[str, Path] = {}
    for token in impl_tokens:
        run_path = experiment_root / token_to_agent_dir(token) / selected_run
        if run_path.exists() and run_path.is_dir():
            run_paths[token] = run_path
    return run_paths


def get_baselines_by_impl(
    experiment_root: Path, impl_tokens: list[str]
) -> dict[str, dict]:
    baselines: dict[str, dict] = {}
    generic_data = load_json(experiment_root / "exec_times" / "base_eval.json")

    for token in impl_tokens:
        specific_path = experiment_root / "exec_times" / f"{token}_base_eval.json"
        if specific_path.exists():
            baselines[token] = load_json(specific_path)
        elif generic_data:
            baselines[token] = generic_data
        else:
            baselines[token] = {}
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
    st.markdown("## \u26a1 MLIR-RL")
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

    all_impl_tokens = discover_impl_tokens(experiment_root)
    if not all_impl_tokens:
        st.warning("No implementation folders or base files found for this experiment.")
        st.stop()

    selected_impl_tokens = st.multiselect(
        "Implementations",
        options=all_impl_tokens,
        default=all_impl_tokens,
        format_func=token_display,
    )
    if not selected_impl_tokens:
        st.warning("Select at least one implementation.")
        st.stop()

    runs = get_runs(experiment_root, selected_impl_tokens)
    if not runs:
        st.warning("No runs found for this experiment.")
        st.stop()
    selected_run = st.selectbox("Run", runs)

    st.divider()
    st.markdown("**Filters**")

    run_paths_by_impl = get_run_paths_by_impl(
        experiment_root, selected_run, selected_impl_tokens
    )
    if not run_paths_by_impl:
        st.warning("Selected run does not exist for the chosen implementations.")
        st.stop()

    active_impl_tokens = [t for t in selected_impl_tokens if t in run_paths_by_impl]

    # Load data for filter population
    base_eval_by_impl = get_baselines_by_impl(experiment_root, selected_impl_tokens)
    pytorch_data = load_json(experiment_root / "exec_times" / "pytorch.json")
    exec_time_data_by_impl = {
        impl: load_agent_exec_times(run_path)
        for impl, run_path in run_paths_by_impl.items()
    }
    df_full = build_main_df(
        experiment,
        tuple(selected_impl_tokens),
        base_eval_by_impl,
        pytorch_data,
        exec_time_data_by_impl,
    )

    all_batch_sizes = sorted(df_full["batch_size"].dropna().unique().tolist())
    all_op_types = sorted(df_full["op_type"].dropna().unique().tolist())
    all_categories = sorted(df_full["bench_category"].dropna().unique().tolist())

    sel_categories = st.multiselect("Benchmark Category", all_categories, default=all_categories)
    sel_batch = st.multiselect("Batch Size", all_batch_sizes, default=all_batch_sizes)
    sel_ops = st.multiselect("Op Type", all_op_types, default=all_op_types)
    speedup_min = st.slider("Min RL speedup vs MLIR baseline", 0.0, 10.0, 0.0, 0.1)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("\U0001f504 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c2:
        if st.button("\U0001f4c4 Export Report", use_container_width=True):
            st.session_state["_export_report"] = True
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
    st.markdown("# MLIR-RL Evaluation")
    impls_shown = [token_display(impl) for impl in active_impl_tokens]
    impls_text = ", ".join(impls_shown) if impls_shown else "No RL implementations"
    st.caption(
        f"Implementations: **{impls_text}** \u00b7 "
        f"Experiment: **{experiment}** \u00b7 Run: **{selected_run}** \u00b7 {len(df)} benchmarks shown"
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


def impl_metrics(df_in: pd.DataFrame, suffix: str) -> dict:
    opt_col = f"rl_{suffix}_optimized_us"
    eager_col = f"rl_{suffix}_vs_eager"
    compile_col = f"rl_{suffix}_vs_compile"
    jit_col = f"rl_{suffix}_vs_jit"

    available = opt_col in df_in.columns and df_in[opt_col].notna().any()
    if not available:
        return {"available": False, "mean": None, "beats_all": 0}

    beats_all = df_in[
        (df_in[eager_col].fillna(0) > 1)
        & (df_in[jit_col].fillna(0) > 1)
    ].shape[0]

    eager_vals = df_in[eager_col].dropna()
    mean_val = eager_vals.mean() if len(eager_vals) > 0 else None

    return {"available": True, "mean": mean_val, "beats_all": beats_all}


summary_rows = []
for token in active_impl_tokens:
    suffix = token_col(token)
    metrics = impl_metrics(df, suffix)
    summary_rows.append(
        {
            "Implementation": token_display(token),
            "Avg RL vs PyTorch Eager": (
                f"{metrics['mean']:.2f}\u00d7"
                if metrics["available"] and pd.notna(metrics["mean"])
                else "N/A"
            ),
            "Beats PyTorch (Eager+JIT)": (
                f"{metrics['beats_all']} / {len(df)}"
                if metrics["available"]
                else "N/A"
            ),
        }
    )

if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
    )
else:
    st.info("No implementation metrics available for selected run.")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRE-LOAD SHARED DATA
# ─────────────────────────────────────────────
speedup_per_bench_by_impl = {
    impl: load_speedup_folder(run_path / "logs" / "eval" / "speedup")
    for impl, run_path in run_paths_by_impl.items()
}
# For agents without speedup folder, generate from eval_exec_times.json + baseline
for impl, run_path in run_paths_by_impl.items():
    if not speedup_per_bench_by_impl.get(impl):
        exec_times = load_eval_exec_times_json(run_path)
        if exec_times:
            baseline = base_eval_by_impl.get(impl, {})
            speedup_per_bench_by_impl[impl] = {
                bench: [baseline[bench] / times[0]]
                for bench, times in exec_times.items()
                if bench in baseline and baseline[bench] and baseline[bench] > 0
            }
avg_speedup_by_impl = {
    impl: load_plain_floats(run_path / "logs" / "eval" / "average_speedup")
    for impl, run_path in run_paths_by_impl.items()
}
exec_data_json_by_impl = {
    impl: load_exec_data_json(run_path)
    for impl, run_path in run_paths_by_impl.items()
}
ckpt_data_by_impl: dict[str, dict] = {}
for impl, run_path in run_paths_by_impl.items():
    ckpts = get_checkpoints(run_path)
    exec_data_folder = load_agent_exec_times(run_path)
    if ckpts and exec_data_folder:
        ckpt_data_by_impl[impl] = {
            "run_path": run_path,
            "checkpoints": ckpts,
            "exec_time_data": exec_data_folder,
        }

# Speedup dataframe (used in multiple tabs)
rows_sp = []
for impl, speedup_per_bench in speedup_per_bench_by_impl.items():
    best_per_bench = {k: max(v) for k, v in speedup_per_bench.items() if v}
    for k, sp in best_per_bench.items():
        meta = parse_benchmark_key(k)
        rows_sp.append(
            {
                "benchmark": k,
                "speedup": sp,
                "implementation": impl,
                "impl_display": token_display(impl),
                "op_type": label_op_type(meta.get("op_type") or "unknown"),
                "model_family": meta.get("model_family") or "unknown",
                "bench_category": meta.get("bench_category") or "Other",
                "batch_size": meta.get("batch_size"),
                "seq_len": meta.get("seq_len"),
            }
        )
df_sp = pd.DataFrame(rows_sp).dropna(subset=["speedup"])


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_labels = [
    "\u23f1  Exec Times",
    "\U0001f680  Speedup",
    "\U0001f4c8  Training",
    "\U0001f516  Checkpoints",
    "\U0001f52c  Schedules",
    "\u2694\ufe0f  Head-to-Head",
    "\U0001f4ca  Correlation",
]
(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
) = st.tabs(tab_labels)


# ════════════════════════════════════════════
# TAB 1 — EXECUTION TIME COMPARISON
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Execution Time Comparison: RL (Optimized) vs MLIR Baseline vs PyTorch")
    st.caption(
        "All exec times in \u00b5s. **RL Optimized** = best schedule found. "
        "**MLIR Baseline** = unoptimized MLIR + O3. Lower is better."
    )

    if df.empty:
        st.info("No benchmarks match the current filters.")
    else:
        rl_modes = [
            (
                f"rl_{token_col(token)}_optimized_us",
                f"{token_display(token)} Optimized",
            )
            for token in active_impl_tokens
        ]
        mode_labels = {k: v for k, v in rl_modes}
        mode_labels.update(
            {
                "mlir_baseline_us": "MLIR Baseline",
                "pytorch_eager_us": "PyTorch Eager",
                "pytorch_compile_us": "PyTorch Compile",
                "pytorch_jit_us": "PyTorch JIT",
            }
        )

        COLOR_BASELINE = "#94a3b8"
        rl_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        color_map = {
            "MLIR Baseline": COLOR_BASELINE,
            "PyTorch Eager": COLOR_EAGER,
            "PyTorch Compile": COLOR_COMPILE,
            "PyTorch JIT": COLOR_JIT,
        }
        for i, (_, label) in enumerate(rl_modes):
            color_map[label] = rl_palette[i % len(rl_palette)]

        mode_order = [label for _, label in rl_modes] + [
            "MLIR Baseline",
            "PyTorch Eager",
            "PyTorch Compile",
            "PyTorch JIT",
        ]

        # ── Chart 1: per model family BAR with error bars ──
        agg_cols = (
            ["model_family"]
            + [c for c, _ in rl_modes]
            + [
                "mlir_baseline_us",
                "pytorch_eager_us",
                "pytorch_compile_us",
                "pytorch_jit_us",
            ]
        )
        agg_cols_present = [c for c in agg_cols if c in df.columns]

        all_families = sorted(df["model_family"].dropna().unique())
        sel_families = st.multiselect(
            "Model Families",
            options=all_families,
            default=all_families,
            key="tab1_family_filter",
        )

        if sel_families:
            df_family = df[df["model_family"].isin(sel_families)]

            # Grouped bar with mean
            df_agg = df_family[agg_cols_present].groupby("model_family", as_index=False).mean(
                numeric_only=True
            )

            # Error bars (std)
            df_std = (
                df_family[agg_cols_present]
                .groupby("model_family", as_index=False)
                .std(numeric_only=True)
            )

            if not df_agg.empty:
                df_melt = (
                    df_agg.melt(
                        id_vars="model_family", var_name="mode", value_name="exec_time_us"
                    )
                    .dropna(subset=["exec_time_us"])
                )
                df_melt["mode"] = df_melt["mode"].map(mode_labels)
                df_melt["mode"] = pd.Categorical(
                    df_melt["mode"], categories=mode_order, ordered=True
                )
                df_melt = df_melt.sort_values("mode")

                # Merge std
                df_std_melt = (
                    df_std.melt(
                        id_vars="model_family", var_name="mode", value_name="std"
                    )
                    .dropna(subset=["std"])
                )
                df_std_melt["mode"] = df_std_melt["mode"].map(mode_labels)

                # Build figure with error bars
                fig = go.Figure()
                for mode_name in mode_order:
                    subset = df_melt[df_melt["mode"] == mode_name]
                    std_subset = df_std_melt[df_std_melt["mode"] == mode_name]
                    if subset.empty:
                        continue
                    error_y = None
                    if not std_subset.empty:
                        merged = subset.merge(std_subset, on=["model_family", "mode"], how="left")
                        error_y = merged["std"].tolist()

                    fig.add_trace(
                        go.Bar(
                            x=subset["model_family"],
                            y=subset["exec_time_us"],
                            name=mode_name,
                            marker_color=color_map.get(mode_name),
                            error_y=dict(type="data", array=error_y, visible=True, thickness=1.5)
                            if error_y
                            else None,
                            marker_line_width=0,
                        )
                    )

                fig.update_layout(
                    barmode="group",
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    margin=dict(l=0, r=0, t=40, b=60),
                    xaxis_title="Model Family",
                    yaxis_title="Avg Exec Time (\u00b5s)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Chart 2: sub-family drill-down ──
        st.markdown("### Sub-family Drill-down")
        st.caption("Select a family to inspect its sub-families.")

        drill_family = st.selectbox(
            "Model Family", options=all_families, key="tab1_drill_family"
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
                key="tab1_subfamily_filter",
            )
            df_drill = df_drill[df_drill["sub_family"].isin(sel_subfamilies)]

            if not df_drill.empty:
                agg_sub_cols = (
                    ["sub_family"]
                    + [c for c, _ in rl_modes]
                    + [
                        "mlir_baseline_us",
                        "pytorch_eager_us",
                        "pytorch_compile_us",
                        "pytorch_jit_us",
                    ]
                )
                agg_sub_cols = [c for c in agg_sub_cols if c in df_drill.columns]
                df_sub_agg = (
                    df_drill[agg_sub_cols]
                    .groupby("sub_family", as_index=False)
                    .mean(numeric_only=True)
                )
                df_sub_melt = (
                    df_sub_agg.melt(
                        id_vars="sub_family", var_name="mode", value_name="exec_time_us"
                    )
                    .dropna(subset=["exec_time_us"])
                )
                df_sub_melt["mode"] = df_sub_melt["mode"].map(mode_labels)
                df_sub_melt["mode"] = pd.Categorical(
                    df_sub_melt["mode"], categories=mode_order, ordered=True
                )
                df_sub_melt = df_sub_melt.sort_values("mode")

                fig_sub = px.bar(
                    df_sub_melt,
                    x="sub_family",
                    y="exec_time_us",
                    color="mode",
                    barmode="group",
                    color_discrete_map=color_map,
                    category_orders={"mode": mode_order},
                    labels={
                        "sub_family": "Sub-family",
                        "exec_time_us": "Avg Exec Time (\u00b5s)",
                        "mode": "",
                    },
                    height=380,
                )
                fig_sub.update_layout(
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    margin=dict(l=0, r=0, t=40, b=60),
                )
                fig_sub.update_traces(marker_line_width=0)
                st.plotly_chart(fig_sub, use_container_width=True)

        # ── Top/Bottom N ──
        st.markdown("### Top & Bottom Benchmarks")
        col_top, col_bot = st.columns(2)
        n_show = st.number_input("Show top/bottom N", 3, 50, 10, key="top_n")

        display_cols = [
            "benchmark",
            "bench_category",
            "model_family",
            "op_type_label",
            "batch_size",
            "mlir_baseline_us",
            "pytorch_eager_us",
            "best_rl_vs_eager",
        ]
        for token in active_impl_tokens:
            suffix = token_col(token)
            display_cols.extend(
                [
                    f"rl_{suffix}_optimized_us",
                    f"rl_{suffix}_speedup_over_baseline",
                    f"rl_{suffix}_vs_eager",
                ]
            )
        display_cols = [c for c in display_cols if c in df.columns]

        df_sorted = df[display_cols].sort_values(
            "best_rl_vs_eager", ascending=False, na_position="last"
        )

        with col_top:
            st.markdown(f"**Top {n_show} by RL vs Eager**")
            st.dataframe(
                df_sorted.head(n_show),
                use_container_width=True,
                hide_index=True,
            )
        with col_bot:
            st.markdown(f"**Bottom {n_show} by RL vs Eager**")
            st.dataframe(
                df_sorted.tail(n_show).sort_values(
                    "best_rl_vs_eager", ascending=True, na_position="first"
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ── Full data table ──
        st.markdown("#### Full Data Table")
        st.caption(
            "`rl_<impl>_vs_eager > 1` means RL is faster than PyTorch Eager. "
            "Sorted by best RL vs Eager."
        )
        st.dataframe(
            df_sorted,
            use_container_width=True,
            hide_index=True,
        )

        # Export
        st.markdown("#### Export")
        ec1, ec2 = st.columns(2)
        with ec1:
            csv = df[display_cols].to_csv(index=False)
            st.download_button(
                "\u2b07 Download CSV", csv, "exec_times.csv", "text/csv", use_container_width=True
            )
        with ec2:
            raw_json = json.dumps(
                {"base_eval": base_eval_by_impl, "pytorch": pytorch_data}, indent=2
            )
            st.download_button(
                "\u2b07 Download JSON",
                raw_json,
                "exec_data.json",
                "application/json",
                use_container_width=True,
            )


# ════════════════════════════════════════════
# TAB 2 — SPEEDUP ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown("### RL Speedup over Unoptimized MLIR Baseline")
    st.caption(
        "Speedup = baseline / optimized. **> 1 means RL improved over baseline.**"
    )

    any_speedup_data = any(bool(v) for v in speedup_per_bench_by_impl.values())
    if not any_speedup_data:
        st.info("No speedup files found for selected run in any implementation.")
    elif df_sp.empty:
        st.info("No speedup values after filtering.")
    else:
        # ── Best-so-far Convergence ──
        st.markdown("#### Best-So-Far Speedup Convergence")
        st.caption(
            "Cumulative maximum of average speedup across eval steps. "
            "Shows how quickly each implementation approaches its final performance."
        )
        any_avg = any(bool(vals) for vals in avg_speedup_by_impl.values())
        if any_avg:
            fig_best_sofar = go.Figure()
            for impl in active_impl_tokens:
                vals = avg_speedup_by_impl.get(impl, [])
                if not vals:
                    continue
                best_sofar = []
                cur_max = 0
                for v in vals:
                    cur_max = max(cur_max, v)
                    best_sofar.append(cur_max)
                fig_best_sofar.add_trace(
                    go.Scatter(
                        x=list(range(len(best_sofar))),
                        y=best_sofar,
                        mode="lines+markers",
                        name=token_display(impl),
                        line=dict(width=2),
                        marker=dict(size=4),
                    )
                )
            fig_best_sofar.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#888",
                annotation_text="No improvement",
                annotation_position="top right",
            )
            fig_best_sofar.update_layout(
                **PLOTLY_THEME,
                margin=dict(l=0, r=0, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_best_sofar.update_xaxes(title_text="Eval Step")
            fig_best_sofar.update_yaxes(title_text="Best-So-Far Avg Speedup (\u00d7)")
            st.plotly_chart(fig_best_sofar, use_container_width=True)

        # ── Per-op-type speedup bar (existing) ──
        st.markdown("#### Avg Best Speedup per Model Family by Op Type")
        all_op_types_sp = sorted(df_sp["op_type"].unique())
        if all_op_types_sp:
            sel_op = st.selectbox(
                "Operation Type", options=all_op_types_sp, key="tab2_op_filter"
            )
            df_sp_filtered = df_sp[df_sp["op_type"] == sel_op]
            if not df_sp_filtered.empty:
                df_sp_agg = (
                    df_sp_filtered.groupby(
                        ["model_family", "impl_display"], as_index=False
                    )["speedup"]
                    .mean()
                    .rename(columns={"speedup": "avg_speedup"})
                    .sort_values("avg_speedup", ascending=False)
                )
                fig2 = px.bar(
                    df_sp_agg,
                    x="model_family",
                    y="avg_speedup",
                    color="impl_display",
                    barmode="group",
                    labels={
                        "model_family": "Model Family",
                        "avg_speedup": "Avg Best Speedup (\u00d7)",
                        "impl_display": "",
                    },
                    height=400,
                )
                fig2.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="#888",
                    annotation_text="No improvement",
                    annotation_position="top right",
                )
                fig2.update_layout(
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    margin=dict(l=0, r=0, t=40, b=60),
                )
                fig2.update_traces(marker_line_width=0)
                st.plotly_chart(fig2, use_container_width=True)

        # ── Average speedup evolution line chart ──
        st.markdown("#### Average RL Speedup over Eval Steps (Per-Step)")
        if any_avg:
            fig3 = go.Figure()
            for impl in active_impl_tokens:
                vals = avg_speedup_by_impl.get(impl, [])
                if not vals:
                    continue
                fig3.add_trace(
                    go.Scatter(
                        x=list(range(len(vals))),
                        y=vals,
                        mode="lines",
                        name=token_display(impl),
                        line=dict(width=2),
                    )
                )
            fig3.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#888",
                annotation_text="No improvement",
                annotation_position="top right",
            )
            fig3.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
            fig3.update_xaxes(title_text="Eval Step")
            fig3.update_yaxes(title_text="Avg Speedup (\u00d7)")
            st.plotly_chart(fig3, use_container_width=True)

        # ── Per-benchmark speedup evolution ──
        st.markdown("### Speedup Evolution per Benchmark")
        bench_options = sorted(
            {
                bench
                for data in speedup_per_bench_by_impl.values()
                for bench in data.keys()
            }
        )
        if bench_options:
            selected_bench = st.multiselect(
                "Select benchmarks",
                bench_options,
                default=bench_options[: min(5, len(bench_options))],
            )
            if selected_bench:
                fig4 = go.Figure()
                for impl in active_impl_tokens:
                    impl_data = speedup_per_bench_by_impl.get(impl, {})
                    for bench in selected_bench:
                        if bench not in impl_data:
                            continue
                        vals = impl_data[bench]
                        fig4.add_trace(
                            go.Scatter(
                                x=list(range(len(vals))),
                                y=vals,
                                mode="lines",
                                name=f"{bench} ({token_display(impl)})",
                                line=dict(width=1.5),
                            )
                        )
                fig4.add_hline(y=1.0, line_dash="dash", line_color="#5a6480")
                fig4.update_layout(
                    **PLOTLY_THEME,
                    xaxis_title="Eval Step",
                    yaxis_title="Speedup (\u00d7)",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=40, b=40),
                )
                st.plotly_chart(fig4, use_container_width=True)

        # ── RL vs PyTorch (Eager & JIT) per Model Family ──
        st.markdown("#### RL vs PyTorch (Eager & JIT) per Model Family")
        st.caption("Speedup of RL over PyTorch Eager/JIT (>1 = RL faster).")

        # Build comparison dataframe from df
        cmp_rows = []
        for token in active_impl_tokens:
            suffix = token_col(token)
            opt_col = f"rl_{suffix}_optimized_us"
            eager_col = f"rl_{suffix}_vs_eager"
            jit_col = f"rl_{suffix}_vs_jit"
            if opt_col not in df.columns:
                continue
            for _, row in df[df[opt_col].notna()].iterrows():
                family = row.get("model_family", "unknown")
                eager_val = row.get(eager_col)
                jit_val = row.get(jit_col)
                if eager_val is not None and jit_val is not None:
                    cmp_rows.append({
                        "model_family": family,
                        "implementation": token_display(token),
                        "vs_eager": eager_val,
                        "vs_jit": jit_val,
                    })

        df_cmp = pd.DataFrame(cmp_rows)
        if not df_cmp.empty:
            # Aggregate by model family
            df_cmp_agg = df_cmp.groupby(
                ["model_family", "implementation"], as_index=False
            )[["vs_eager", "vs_jit"]].mean()

            fig_cmp = go.Figure()
            families_sorted = sorted(df_cmp_agg["model_family"].unique())
            for impl_label in sorted(df_cmp_agg["implementation"].unique()):
                sub = df_cmp_agg[df_cmp_agg["implementation"] == impl_label]
                # Reindex to ensure all families present
                sub_idx = sub.set_index("model_family").reindex(families_sorted)
                fig_cmp.add_trace(
                    go.Bar(
                        x=families_sorted,
                        y=sub_idx["vs_eager"].tolist(),
                        name=f"{impl_label} vs Eager",
                        marker_line_width=0,
                    )
                )
                fig_cmp.add_trace(
                    go.Bar(
                        x=families_sorted,
                        y=sub_idx["vs_jit"].tolist(),
                        name=f"{impl_label} vs JIT",
                        marker_line_width=0,
                    )
                )
            fig_cmp.add_hline(
                y=1.0, line_dash="dash", line_color="#888",
                annotation_text="Equal to PyTorch", annotation_position="top right",
            )
            fig_cmp.update_layout(
                barmode="group",
                **PLOTLY_THEME,
                xaxis_tickangle=-20,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=60),
                xaxis_title="Model Family",
                yaxis_title="Speedup vs PyTorch (\u00d7)",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Export
        sp_csv = pd.DataFrame(
            [
                {
                    "implementation": token_display(impl),
                    "benchmark": k,
                    "step": i,
                    "speedup": v,
                }
                for impl, speedup_per_bench in speedup_per_bench_by_impl.items()
                for k, vals in speedup_per_bench.items()
                for i, v in enumerate(vals)
            ]
        ).to_csv(index=False)
        st.download_button(
            "\u2b07 Download Speedup CSV",
            sp_csv,
            "speedup_evolution.csv",
            "text/csv",
            use_container_width=True,
        )


# ════════════════════════════════════════════
# TAB 3 — TRAINING CURVES
# ════════════════════════════════════════════
with tab3:
    st.markdown("### Training Curves")
    st.caption("Eval signals with rolling mean \u00b1 std band (window=10).")

    rolling_window = st.slider("Rolling window", 1, 50, 10, key="rolling_window")

    signals = {
        "Reward": ("reward", "#4a9eff"),
        "Cumulative Reward": ("cumulative_reward", "#c084fc"),
        "Policy Entropy": ("entropy", "#f5a623"),
    }

    any_data = False
    for title, (fname, color) in signals.items():
        fig = go.Figure()
        has_signal = False
        for impl in active_impl_tokens:
            run_path = run_paths_by_impl.get(impl)
            if run_path is None:
                continue
            vals = load_plain_floats(run_path / "logs" / "eval" / fname)
            if not vals:
                continue
            has_signal = True
            any_data = True
            label = token_display(impl)
            if rolling_window > 1 and len(vals) >= rolling_window:
                rm, rs = compute_rolling_stats(vals, rolling_window)
                x = list(range(len(vals)))
                # Raw (faint)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=vals,
                        mode="lines",
                        name=f"{label} (raw)",
                        line=dict(width=0.5, color=color),
                        opacity=0.3,
                        showlegend=False,
                    )
                )
                # Rolling mean
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=rm,
                        mode="lines",
                        name=label,
                        line=dict(width=2.5, color=color),
                    )
                )
                # Fill band
                upper = [m + s if pd.notna(s) else m for m, s in zip(rm, rs)]
                lower = [m - s if pd.notna(s) else m for m, s in zip(rm, rs)]
                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                        line=dict(width=0),
                        name=f"{label} \u00b11\u03c3",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(vals))),
                        y=vals,
                        mode="lines",
                        name=label,
                        line=dict(width=2, color=color),
                    )
                )

        if not has_signal:
            st.caption(f"No data for `{fname}`")
            continue
        fig.update_layout(
            **PLOTLY_THEME,
            title=title,
            title_font_size=13,
            margin=dict(l=0, r=0, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

    if not any_data:
        st.info("No eval training curves found for selected run.")

    # Train logs
    train_signals = {
        "Train Reward": ("reward", "#3dd68c"),
        "Train Entropy": ("entropy", "#f5a623"),
    }
    train_any = False
    train_header_shown = False
    for title, (fname, color) in train_signals.items():
        fig = go.Figure()
        has_signal = False
        for impl in active_impl_tokens:
            run_path = run_paths_by_impl.get(impl)
            if run_path is None:
                continue
            vals = load_plain_floats(run_path / "logs" / "train" / fname)
            if not vals:
                continue
            has_signal = True
            if not train_header_shown:
                st.markdown("### Training Logs")
                train_header_shown = True
            train_any = True
            label = token_display(impl)
            if rolling_window > 1 and len(vals) >= rolling_window:
                rm, rs = compute_rolling_stats(vals, rolling_window)
                x = list(range(len(vals)))
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=vals,
                        mode="lines",
                        name=f"{label} (raw)",
                        line=dict(width=0.5, color=color),
                        opacity=0.3,
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=rm,
                        mode="lines",
                        name=label,
                        line=dict(width=2.5, color=color),
                    )
                )
                upper = [m + s if pd.notna(s) else m for m, s in zip(rm, rs)]
                lower = [m - s if pd.notna(s) else m for m, s in zip(rm, rs)]
                fig.add_trace(
                    go.Scatter(
                        x=x + x[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
                        line=dict(width=0),
                        name=f"{label} \u00b11\u03c3",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(vals))),
                        y=vals,
                        mode="lines",
                        name=label,
                        line=dict(width=2, color=color),
                    )
                )

        if not has_signal:
            continue
        fig.update_layout(
            **PLOTLY_THEME,
            title=title,
            title_font_size=13,
            margin=dict(l=0, r=0, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

    if not train_any:
        st.caption("No training logs found.")

    # Export
    export_data = {}
    for impl, run_path in run_paths_by_impl.items():
        impl_key = token_display(impl)
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
        st.download_button(
            "\u2b07 Download Training Curves JSON",
            curves_json,
            "training_curves.json",
            "application/json",
            use_container_width=True,
        )


# ════════════════════════════════════════════
# TAB 4 — CHECKPOINT EVOLUTION
# ════════════════════════════════════════════
with tab4:
    st.markdown("### Agent Performance across Checkpoints")
    st.caption(
        "Each checkpoint corresponds to a saved `.pt` model. "
        "Exec times from `logs/eval/exec_time/` folders."
    )

    if not ckpt_data_by_impl:
        st.info("No checkpoint/eval exec-time data found for selected run.")
    else:
        # ── Average exec time evolution ──
        st.markdown("#### Average Exec Time over Checkpoints")
        fig_avg = go.Figure()
        for impl in active_impl_tokens:
            if impl not in ckpt_data_by_impl:
                continue
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
            all_series = [v for v in exec_time_data.values() if len(v) >= 1]
            if not all_series:
                continue
            max_len = max(len(s) for s in all_series)
            avg_per_ckpt = []
            for i in range(max_len):
                vals_at_i = [s[i] for s in all_series if i < len(s)]
                avg_per_ckpt.append(sum(vals_at_i) / len(vals_at_i))
            x_labels = (
                checkpoints[:max_len]
                if max_len <= len(checkpoints)
                else list(range(max_len))
            )
            fig_avg.add_trace(
                go.Scatter(
                    x=list(range(max_len)),
                    y=avg_per_ckpt,
                    mode="lines+markers",
                    marker=dict(size=5),
                    name=token_display(impl),
                    text=x_labels,
                    hovertemplate="Implementation: %{fullData.name}<br>"
                    "Checkpoint: %{text}<br>"
                    "Avg Exec Time: %{y:.0f} \u00b5s<extra></extra>",
                )
            )
        fig_avg.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
        fig_avg.update_xaxes(title_text="Checkpoint Index")
        fig_avg.update_yaxes(title_text="Avg Exec Time (\u00b5s)")
        st.plotly_chart(fig_avg, use_container_width=True)

        # ── Delta from first checkpoint ──
        st.markdown("#### Improvement from First Checkpoint (\u0394 Exec Time)")
        st.caption(
            "Shows how much each checkpoint improves over the first one. "
            "More negative = more improvement."
        )
        fig_delta = go.Figure()
        for impl in active_impl_tokens:
            if impl not in ckpt_data_by_impl:
                continue
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
            all_series = [v for v in exec_time_data.values() if len(v) >= 1]
            if not all_series:
                continue
            max_len = max(len(s) for s in all_series)
            avg_per_ckpt = []
            for i in range(max_len):
                vals_at_i = [s[i] for s in all_series if i < len(s)]
                avg_per_ckpt.append(sum(vals_at_i) / len(vals_at_i))
            if not avg_per_ckpt:
                continue
            first_val = avg_per_ckpt[0]
            delta = [v - first_val for v in avg_per_ckpt]
            fig_delta.add_trace(
                go.Scatter(
                    x=list(range(max_len)),
                    y=delta,
                    mode="lines+markers",
                    marker=dict(size=5),
                    name=token_display(impl),
                )
            )
        fig_delta.add_hline(
            y=0, line_dash="dash", line_color="#888", annotation_text="No change"
        )
        fig_delta.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
        fig_delta.update_xaxes(title_text="Checkpoint Index")
        fig_delta.update_yaxes(title_text="\u0394 Avg Exec Time (\u00b5s)")
        st.plotly_chart(fig_delta, use_container_width=True)

        # ── Per-benchmark exec time evolution ──
        st.markdown("#### Per-Benchmark Exec Time Evolution")
        bench_names = sorted(
            {
                bench
                for item in ckpt_data_by_impl.values()
                for bench in item["exec_time_data"].keys()
            }
        )
        selected_benches = st.multiselect(
            "Select benchmarks",
            bench_names,
            default=bench_names[: min(6, len(bench_names))],
            key="ckpt_bench_select",
        )
        if selected_benches:
            fig_ckpt = go.Figure()
            for impl in active_impl_tokens:
                if impl not in ckpt_data_by_impl:
                    continue
                checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
                exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
                for bench in selected_benches:
                    if bench not in exec_time_data:
                        continue
                    vals = exec_time_data[bench]
                    x_idx = list(range(len(vals)))
                    hover = (
                        checkpoints[: len(vals)]
                        if len(vals) <= len(checkpoints)
                        else x_idx
                    )
                    fig_ckpt.add_trace(
                        go.Scatter(
                            x=x_idx,
                            y=vals,
                            mode="lines+markers",
                            name=f"{bench} ({token_display(impl)})",
                            line=dict(width=1.8),
                            marker=dict(size=4),
                            hovertemplate="Checkpoint: %{text}<br>"
                            "Exec Time: %{y:.0f} \u00b5s"
                            "<extra>%{fullData.name}</extra>",
                            text=hover,
                        )
                    )
            fig_ckpt.update_layout(
                **PLOTLY_THEME,
                xaxis_title="Checkpoint Index",
                yaxis_title="Exec Time (\u00b5s)",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=40),
            )
            st.plotly_chart(fig_ckpt, use_container_width=True)

        # ── Checkpoint list ──
        for impl in active_impl_tokens:
            if impl not in ckpt_data_by_impl:
                continue
            run_path = ckpt_data_by_impl[impl]["run_path"]
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            with st.expander(
                f"\U0001f4c1 {token_display(impl)}: {len(checkpoints)} checkpoints in `{run_path / 'models'}`"
            ):
                for i, ckpt in enumerate(checkpoints):
                    st.caption(f"`[{i}]` {ckpt}")

        # ── Export ──
        st.markdown("#### Export")
        ckpt_rows = []
        for impl in active_impl_tokens:
            if impl not in ckpt_data_by_impl:
                continue
            checkpoints = ckpt_data_by_impl[impl]["checkpoints"]
            exec_time_data = ckpt_data_by_impl[impl]["exec_time_data"]
            for bench, vals in exec_time_data.items():
                for i, v in enumerate(vals):
                    ckpt_rows.append(
                        {
                            "implementation": token_display(impl),
                            "benchmark": bench,
                            "checkpoint_index": i,
                            "checkpoint": checkpoints[i]
                            if i < len(checkpoints)
                            else i,
                            "exec_time_us": v,
                        }
                    )
        if ckpt_rows:
            ckpt_csv = pd.DataFrame(ckpt_rows).to_csv(index=False)
            st.download_button(
                "\u2b07 Download Checkpoint CSV",
                ckpt_csv,
                "checkpoint_evolution.csv",
                "text/csv",
                use_container_width=True,
            )


# ════════════════════════════════════════════
# TAB 5 — SCHEDULE EXPLORATION
# ════════════════════════════════════════════
with tab5:
    st.markdown("### Schedule Exploration Analysis")
    st.caption(
        "Analyzes **all** schedules explored by the RL agent (from `exec_data.json`), "
        "not just the best one. Reveals exploration behavior."
    )

    any_exec_data = any(bool(v) for v in exec_data_json_by_impl.values())
    if not any_exec_data:
        st.info(
            "No `exec_data.json` files found for selected run. "
            "Schedule exploration data is needed for this tab."
        )
    else:
        # Build exploration dataframe
        expl_rows = []
        for impl, exec_data in exec_data_json_by_impl.items():
            for bench, schedules in exec_data.items():
                if not schedules:
                    continue
                meta = parse_benchmark_key(bench)
                times = list(schedules.values())
                best_time = min(times)
                baseline = base_eval_by_impl.get(impl, {}).get(bench)
                best_speedup = baseline / best_time if baseline and best_time else None
                # Parse all schedule strings
                transform_counts = Counter()
                total_transforms = 0
                for s in schedules:
                    parsed = parse_schedule_string(s)
                    for k, v in parsed.items():
                        if k != "total":
                            transform_counts[k] += v
                        else:
                            total_transforms += v
                expl_rows.append(
                    {
                        "implementation": impl,
                        "impl_display": token_display(impl),
                        "benchmark": bench,
                        "model_family": meta["model_family"],
                        "op_type": label_op_type(meta["op_type"] or "unknown"),
                        "bench_category": meta.get("bench_category", "Other"),
                        "num_schedules": len(schedules),
                        "best_time_ns": best_time,
                        "mean_time_ns": sum(times) / len(times),
                        "std_time_ns": pd.Series(times).std(),
                        "best_speedup": best_speedup,
                        "baseline_ns": baseline,
                        "I_count": transform_counts.get("I", 0),
                        "T_count": transform_counts.get("T", 0),
                        "V_count": transform_counts.get("V", 0),
                        "TP_count": transform_counts.get("TP", 0),
                        "TPF_count": transform_counts.get("TPF", 0),
                        "NT_count": transform_counts.get("NT", 0),
                        "total_transforms": total_transforms,
                    }
                )
        df_expl = pd.DataFrame(expl_rows)
        if df_expl.empty:
            st.info("No schedule exploration data after parsing.")
        else:
            # ── Exploration count per benchmark ──
            st.markdown("#### Number of Unique Schedules Explored per Benchmark")
            st.caption(
                "Higher = more exploration. Shows whether the agent converges quickly or "
                "needs many attempts."
            )
            df_expl_counts = (
                df_expl.groupby("model_family")["num_schedules"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            fig_expl_count = go.Figure()
            for impl_label in sorted(df_expl["impl_display"].unique()):
                sub = (
                    df_expl[df_expl["impl_display"] == impl_label]
                    .groupby("model_family")["num_schedules"]
                    .mean()
                    .reset_index()
                )
                fig_expl_count.add_trace(
                    go.Bar(
                        x=sub["model_family"],
                        y=sub["num_schedules"],
                        name=impl_label,
                        marker_line_width=0,
                    )
                )
            fig_expl_count.update_layout(
                barmode="group",
                **PLOTLY_THEME,
                xaxis_tickangle=-20,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=60),
                xaxis_title="Model Family",
                yaxis_title="Avg Unique Schedules Explored",
            )
            st.plotly_chart(fig_expl_count, use_container_width=True)

            # ── Violin/Box: distribution of execution times ──
            st.markdown("#### Distribution of Explored Execution Times")
            st.caption(
                "Violin plot showing the spread of all explored schedule execution times "
                "per model family. Hover for benchmark names."
            )
            sel_impl_expl = st.selectbox(
                "Implementation",
                options=sorted(df_expl["impl_display"].unique()),
                key="expl_violin_impl",
            )
            df_expl_impl = df_expl[df_expl["impl_display"] == sel_impl_expl]

            # Build per-benchmark rows with individual times
            violin_rows = []
            for impl, exec_data in exec_data_json_by_impl.items():
                if token_display(impl) != sel_impl_expl:
                    continue
                for bench, schedules in exec_data.items():
                    if not schedules:
                        continue
                    meta = parse_benchmark_key(bench)
                    for s, t in schedules.items():
                        violin_rows.append(
                            {
                                "benchmark": bench,
                                "model_family": meta["model_family"],
                                "exec_time_ns": t,
                            }
                        )
            df_violin = pd.DataFrame(violin_rows)
            if not df_violin.empty:
                fig_violin = px.violin(
                    df_violin,
                    x="model_family",
                    y="exec_time_ns",
                    box=True,
                    points="outliers",
                    labels={
                        "model_family": "Model Family",
                        "exec_time_ns": "Exec Time (ns)",
                    },
                    height=450,
                )
                fig_violin.update_layout(
                    **PLOTLY_THEME,
                    xaxis_tickangle=-20,
                    margin=dict(l=0, r=0, t=20, b=60),
                )
                fig_violin.update_traces(marker=dict(size=2))
                st.plotly_chart(fig_violin, use_container_width=True)

            # ── Transformation type frequency ──
            st.markdown("#### Transformation Type Usage Frequency")
            st.caption(
                "How often each transformation type appears in explored schedules "
                "(sum over all schedules)."
            )
            trans_cols = ["I_count", "T_count", "V_count", "TP_count", "TPF_count", "NT_count"]
            trans_labels = ["Interchange", "Tiling", "Vectorize", "TileParallel", "TileFuse", "NoTransform"]

            fig_trans = go.Figure()
            for impl_label in sorted(df_expl["impl_display"].unique()):
                sub = df_expl[df_expl["impl_display"] == impl_label]
                totals = [sub[c].sum() for c in trans_cols]
                fig_trans.add_trace(
                    go.Bar(
                        x=trans_labels,
                        y=totals,
                        name=impl_label,
                        marker_line_width=0,
                    )
                )
            fig_trans.update_layout(
                barmode="group",
                **PLOTLY_THEME,
                xaxis_tickangle=-15,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=40),
                xaxis_title="Transformation Type",
                yaxis_title="Total Occurrences",
            )
            st.plotly_chart(fig_trans, use_container_width=True)

            # ── Pareto frontier: exploration breadth vs best result ──
            st.markdown("#### Pareto: Exploration Breadth vs Best Speedup")
            st.caption(
                "Each point = one benchmark. X = schedules explored, Y = best speedup. "
                "Ideally, fewer schedules with higher speedup is better."
            )
            fig_pareto = go.Figure()
            for impl_label in sorted(df_expl["impl_display"].unique()):
                sub = df_expl[df_expl["impl_display"] == impl_label].dropna(
                    subset=["best_speedup"]
                )
                if sub.empty:
                    continue
                fig_pareto.add_trace(
                    go.Scatter(
                        x=sub["num_schedules"],
                        y=sub["best_speedup"],
                        mode="markers",
                        name=impl_label,
                        marker=dict(size=7, opacity=0.7),
                        text=sub["benchmark"],
                        hovertemplate="%{text}<br>"
                        "Schedules: %{x}<br>"
                        "Best Speedup: %{y:.1f}\u00d7<extra></extra>",
                    )
                )
            fig_pareto.add_hline(y=1.0, line_dash="dash", line_color="#888")
            fig_pareto.update_layout(
                **PLOTLY_THEME,
                xaxis_title="Unique Schedules Explored",
                yaxis_title="Best Speedup (\u00d7)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=20, b=40),
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

            # ── Schedule complexity vs speedup ──
            st.markdown("#### Schedule Complexity vs Speedup")
            st.caption(
                "Does using more transformations help or hurt? "
                "X-axis = number of transformations in schedule."
            )
            complexity_rows = []
            for impl, exec_data in exec_data_json_by_impl.items():
                for bench, schedules in exec_data.items():
                    baseline = base_eval_by_impl.get(impl, {}).get(bench)
                    for s, t in schedules.items():
                        parsed = parse_schedule_string(s)
                        speedup = baseline / t if baseline and t else None
                        complexity_rows.append(
                            {
                                "implementation": token_display(impl),
                                "benchmark": bench,
                                "total_transforms": parsed["total"],
                                "exec_time_ns": t,
                                "speedup": speedup,
                            }
                        )
            df_cmplx = pd.DataFrame(complexity_rows).dropna(subset=["speedup"])
            if not df_cmplx.empty:
                fig_cmplx = go.Figure()
                for impl_label in sorted(df_cmplx["implementation"].unique()):
                    sub = df_cmplx[df_cmplx["implementation"] == impl_label]
                    fig_cmplx.add_trace(
                        go.Box(
                            x=sub["total_transforms"],
                            y=sub["speedup"],
                            name=impl_label,
                            boxpoints="outliers",
                            marker=dict(size=3),
                        )
                    )
                fig_cmplx.add_hline(y=1.0, line_dash="dash", line_color="#888")
                fig_cmplx.update_layout(
                    barmode="group",
                    **PLOTLY_THEME,
                    xaxis_title="Number of Transformations in Schedule",
                    yaxis_title="Speedup (\u00d7)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=20, b=40),
                    boxmode="group",
                )
                st.plotly_chart(fig_cmplx, use_container_width=True)

            # Export
            st.markdown("#### Export")
            expl_csv = df_expl.to_csv(index=False)
            st.download_button(
                "\u2b07 Download Exploration CSV",
                expl_csv,
                "schedule_exploration.csv",
                "text/csv",
                use_container_width=True,
            )


# ════════════════════════════════════════════
# TAB 6 — HEAD-TO-HEAD COMPARISON
# ════════════════════════════════════════════
with tab6:
    st.markdown("### Implementation Head-to-Head Comparison")
    st.caption(
        "Direct per-benchmark comparison between pairs of implementations. "
        "Shows which implementation wins on each benchmark."
    )

    if len(active_impl_tokens) < 2:
        st.info("Select at least 2 implementations to compare head-to-head.")
    else:
        # Build paired comparison data
        pair_rows = []
        for token in active_impl_tokens:
            suffix = token_col(token)
            opt_col = f"rl_{suffix}_optimized_us"
            if opt_col not in df.columns:
                continue
            # Filter benchmarks with data for this impl
            for _, row in df[df[opt_col].notna()].iterrows():
                pair_rows.append(
                    {
                        "benchmark": row["benchmark"],
                        "implementation": token,
                        "impl_display": token_display(token),
                        "optimized_us": row[opt_col],
                        "model_family": row["model_family"],
                        "op_type": row["op_type_label"],
                        "batch_size": row["batch_size"],
                    }
                )
        df_pair = pd.DataFrame(pair_rows)
        if df_pair.empty:
            st.info("Not enough data for head-to-head comparison.")
        else:
            # Pairwise comparison grid
            impls = sorted(df_pair["implementation"].unique(), key=sort_impl_tokens)
            if len(impls) >= 2:
                pairs = list(combinations(impls, 2))
                st.markdown(f"### Pairwise Comparisons ({len(pairs)} pairs)")

                for impl_a, impl_b in pairs:
                    st.markdown(f"#### {token_display(impl_a)} vs {token_display(impl_b)}")

                    # Get common benchmarks
                    df_a = df_pair[df_pair["implementation"] == impl_a][["benchmark", "optimized_us"]]
                    df_b = df_pair[df_pair["implementation"] == impl_b][["benchmark", "optimized_us"]]
                    merged = df_a.merge(df_b, on="benchmark", suffixes=("_a", "_b")).dropna()

                    if merged.empty:
                        st.caption(f"No common benchmarks with data for both implementations.")
                        continue

                    # Win/loss/tie
                    merged["a_wins"] = merged["optimized_us_a"] < merged["optimized_us_b"]
                    wins_a = merged["a_wins"].sum()
                    wins_b = len(merged) - wins_a
                    ties = (merged["optimized_us_a"] == merged["optimized_us_b"]).sum()

                    col_wl, col_stat = st.columns(2)
                    with col_wl:
                        st.markdown(
                            f"**{token_display(impl_a)} wins:** {wins_a} &nbsp;|&nbsp; "
                            f"**{token_display(impl_b)} wins:** {wins_b} &nbsp;|&nbsp; "
                            f"**Ties:** {ties} &nbsp;(out of {len(merged)} benchmarks)"
                        )

                    # Wilcoxon test
                    if HAS_SCIPY and len(merged) >= 5:
                        try:
                            stat, pval = _wilcoxon(merged["optimized_us_a"], merged["optimized_us_b"])
                            sig = "significant" if pval < 0.05 else "not significant"
                            with col_stat:
                                st.caption(
                                    f"Wilcoxon p={pval:.4f} ({sig}, n={len(merged)})"
                                )
                        except Exception:
                            pass
                    elif not HAS_SCIPY:
                        with col_stat:
                            st.caption("Install `scipy` for Wilcoxon test.")

                    # Scatter plot
                    merged["speedup_a"] = merged["optimized_us_a"] / merged["optimized_us_b"]
                    merged["speedup_b"] = merged["optimized_us_b"] / merged["optimized_us_a"]

                    # Add metadata for coloring
                    meta_map = {}
                    for _, r in df_pair.iterrows():
                        meta_map[r["benchmark"]] = r["model_family"]
                    merged["model_family"] = merged["benchmark"].map(meta_map)

                    max_val = max(
                        merged["optimized_us_a"].max(), merged["optimized_us_b"].max()
                    )

                    fig_scatter = px.scatter(
                        merged,
                        x="optimized_us_a",
                        y="optimized_us_b",
                        color="model_family",
                        hover_data=["benchmark"],
                        labels={
                            "optimized_us_a": f"{token_display(impl_a)} Exec Time (\u00b5s)",
                            "optimized_us_b": f"{token_display(impl_b)} Exec Time (\u00b5s)",
                        },
                        height=420,
                    )
                    # Diagonal line (A = B)
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode="lines",
                            name="Equal Performance",
                            line=dict(dash="dash", color="#888", width=1),
                            showlegend=True,
                        )
                    )
                    fig_scatter.update_layout(
                        **PLOTLY_THEME,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=0, r=0, t=40, b=40),
                    )
                    fig_scatter.update_traces(marker=dict(size=6, opacity=0.7))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.divider()

            # Win/Loss summary matrix
            st.markdown("### Win/Loss Summary Matrix")
            st.caption(
                "Row beats column = row implementation had lower exec time. "
                "Number shows win count over common benchmarks."
            )
            matrix_data = {}
            for impl_a in impls:
                matrix_data[token_display(impl_a)] = {}
                for impl_b in impls:
                    if impl_a == impl_b:
                        matrix_data[token_display(impl_a)][token_display(impl_b)] = "\u2014"
                    else:
                        df_a = df_pair[df_pair["implementation"] == impl_a][
                            ["benchmark", "optimized_us"]
                        ]
                        df_b = df_pair[df_pair["implementation"] == impl_b][
                            ["benchmark", "optimized_us"]
                        ]
                        merged = df_a.merge(
                            df_b, on="benchmark", suffixes=("_a", "_b")
                        ).dropna()
                        if merged.empty:
                            matrix_data[token_display(impl_a)][
                                token_display(impl_b)
                            ] = "N/A"
                        else:
                            wins = (
                                merged["optimized_us_a"] < merged["optimized_us_b"]
                            ).sum()
                            matrix_data[token_display(impl_a)][
                                token_display(impl_b)
                            ] = f"{wins}/{len(merged)}"
            df_matrix = pd.DataFrame(matrix_data)
            st.dataframe(df_matrix, use_container_width=True)

            # Export
            st.markdown("#### Export")
            h2h_csv = df_pair.to_csv(index=False)
            st.download_button(
                "\u2b07 Download Head-to-Head CSV",
                h2h_csv,
                "head_to_head.csv",
                "text/csv",
                use_container_width=True,
            )


# ════════════════════════════════════════════
# TAB 7 — CORRELATION & IMPACT
# ════════════════════════════════════════════
with tab7:
    st.markdown("### Correlation & Impact Analysis")
    st.caption(
        "Explores how benchmark properties (batch size, sequence length, model family) "
        "correlate with RL speedup."
    )

    if df_sp.empty:
        st.info("No speedup data available for correlation analysis.")
    else:
        # ── Batch Size vs Speedup ──
        st.markdown("#### Batch Size vs Speedup")
        st.caption("Scatter: each point = one benchmark. Color = implementation.")
        df_bs = df_sp.dropna(subset=["batch_size", "speedup"])
        if not df_bs.empty:
            fig_bs = px.scatter(
                df_bs,
                x="batch_size",
                y="speedup",
                color="impl_display",
                facet_col="bench_category",
                hover_data=["benchmark", "model_family"],
                labels={
                    "batch_size": "Batch Size",
                    "speedup": "Best Speedup (\u00d7)",
                    "impl_display": "",
                    "bench_category": "Category",
                },
                height=400,
                opacity=0.7,
            )
            fig_bs.add_hline(y=1.0, line_dash="dash", line_color="#888")
            fig_bs.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=40, b=40))
            fig_bs.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_bs, use_container_width=True)
        else:
            st.caption("No batch size data available.")

        # ── Sequence Length vs Speedup ──
        st.markdown("#### Sequence Length vs Speedup")
        df_sl = df_sp.dropna(subset=["seq_len", "speedup"])
        if not df_sl.empty:
            fig_sl = px.scatter(
                df_sl,
                x="seq_len",
                y="speedup",
                color="impl_display",
                hover_data=["benchmark", "model_family"],
                labels={
                    "seq_len": "Sequence Length",
                    "speedup": "Best Speedup (\u00d7)",
                    "impl_display": "",
                },
                height=380,
                opacity=0.7,
            )
            fig_sl.add_hline(y=1.0, line_dash="dash", line_color="#888")
            fig_sl.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
            fig_sl.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_sl, use_container_width=True)
        else:
            st.caption("No sequence length data available (seq_len present in NLP benchmarks only).")

        # ── Speedup by Benchmark Category ──
        st.markdown("#### Speedup Distribution by Benchmark Category")
        df_cat = df_sp.dropna(subset=["bench_category", "speedup"])
        if not df_cat.empty:
            fig_cat = px.box(
                df_cat,
                x="bench_category",
                y="speedup",
                color="impl_display",
                points="outliers",
                labels={
                    "bench_category": "Benchmark Category",
                    "speedup": "Best Speedup (\u00d7)",
                    "impl_display": "",
                },
                height=400,
            )
            fig_cat.add_hline(y=1.0, line_dash="dash", line_color="#888")
            fig_cat.update_layout(**PLOTLY_THEME, margin=dict(l=0, r=0, t=20, b=40))
            st.plotly_chart(fig_cat, use_container_width=True)

        # ── Speedup Correlation Table ──
        st.markdown("#### Per-Category Speedup Summary")
        summary_cat = (
            df_cat.groupby(["bench_category", "impl_display"])["speedup"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
        summary_cat["mean"] = summary_cat["mean"].round(2)
        summary_cat["median"] = summary_cat["median"].round(2)
        summary_cat["std"] = summary_cat["std"].round(2)
        st.dataframe(summary_cat, use_container_width=True, hide_index=True)

        # ── Op Type vs Speedup boxplot ──
        st.markdown("#### Speedup Distribution by Operation Type")
        df_op = df_sp.dropna(subset=["op_type", "speedup"])
        top_ops = df_op["op_type"].value_counts().head(15).index.tolist()
        df_op_filtered = df_op[df_op["op_type"].isin(top_ops)]

        if not df_op_filtered.empty:
            fig_op = px.box(
                df_op_filtered,
                x="op_type",
                y="speedup",
                color="impl_display",
                points="outliers",
                labels={
                    "op_type": "Operation Type",
                    "speedup": "Best Speedup (\u00d7)",
                    "impl_display": "",
                },
                height=420,
            )
            fig_op.add_hline(y=1.0, line_dash="dash", line_color="#888")
            fig_op.update_layout(
                **PLOTLY_THEME,
                xaxis_tickangle=-25,
                margin=dict(l=0, r=0, t=20, b=60),
            )
            st.plotly_chart(fig_op, use_container_width=True)

        # Export
        st.markdown("#### Export")
        corr_csv = df_sp.to_csv(index=False)
        st.download_button(
            "\u2b07 Download Correlation CSV",
            corr_csv,
            "correlation_data.csv",
            "text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# REPORT EXPORT
# ─────────────────────────────────────────────
if st.session_state.get("_export_report"):
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    report_lines = [
        "<html><head><meta charset='utf-8'>",
        "<title>MLIR-RL Report</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "body { font-family: sans-serif; margin: 2rem; }",
        "h1 { color: #1a1a2e; }",
        "h2 { color: #444; margin-top: 2rem; }",
        ".plot { margin: 1rem 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 1rem 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f7f8fa; }",
        "</style></head><body>",
        f"<h1>MLIR-RL Evaluation Report</h1>",
        f"<p>Experiment: <strong>{experiment}</strong> &mdash; Run: <strong>{selected_run}</strong></p>",
        f"<p>Implementations: {', '.join(token_display(t) for t in active_impl_tokens)}</p>",
        f"<p>Generated: {ts}</p>",
        f"<p>Benchmarks shown: {len(df)}</p>",
        "<hr>",
        "<h2>Summary</h2>",
        pd.DataFrame(summary_rows).to_html(index=False),
        "<h2>Execution Times</h2>",
        df.filter(regex="^(benchmark|bench_category|model_family|op_type|batch_size|mlir_baseline|pytorch|best_rl|rl_)")
        .head(50)
        .to_html(index=False),
        "<h2>Speedup Data</h2>",
        df_sp.groupby(["bench_category", "impl_display"])["speedup"]
        .agg(["mean", "count"])
        .reset_index()
        .to_html(index=False),
        "</body></html>",
    ]
    report_html = "\n".join(report_lines)

    st.sidebar.download_button(
        "\u2b07 Download Report (HTML)",
        report_html,
        f"mlir_rl_report_{experiment}_{selected_run}_{ts}.html",
        "text/html",
        use_container_width=True,
    )
    st.session_state["_export_report"] = False
