"""Data loading utilities for the dashboard.

All CSV/image files are read from dashboard/data/ (populated from plots/).
"""

import pandas as pd
from pathlib import Path

# Resolved once at import time — always points to dashboard/data/
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


def data_path(filename: str) -> Path:
    return DATA_DIR / filename


def load_csv(filename: str) -> pd.DataFrame:
    path = data_path(filename)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def img_path(filename: str) -> str:
    """Return absolute path string for st.image()."""
    return str(data_path(filename))


# ── Loaders per section ─────────────────────────────────────────────────────

def load_version_comparison_grouped() -> pd.DataFrame:
    """Per-model grouped speedup: V0 vs V4.5 (+ PyTorch).
    Columns: model, mlir_baseline, pytorch_eager, pytorch_jit,
             mlir_rl_v0, mlir_rl_v45, count,
             v0_speedup, v45_speedup, pytorch_eager_speedup, pytorch_jit_speedup
    """
    return load_csv("version_comparison/grouped.csv")


def load_version_comparison_per_bench() -> pd.DataFrame:
    """Per-benchmark V0 vs V4.5 (benchmarks where V4.5 outperforms V0).
    Columns: benchmark, model, mlir_baseline, v0_exec_time, v0_speedup,
             v45_exec_time, v45_speedup, v45_vs_v0_ratio, pytorch_eager, pytorch_jit
    """
    return load_csv("version_comparison/per_bench.csv")


def load_graph1_performance() -> pd.DataFrame:
    """V0 vs Our Agent speedup per op-type (graph1).
    Columns: op_type, V0_geo_mean, V0_avg, V0_count,
             NoReward_geo_mean, NoReward_avg, NoReward_count
    """
    return load_csv("version_comparison/graph1_performance.csv")


def load_ablation_real() -> pd.DataFrame:
    """Ablation: per-model speedup — real V4.5 numbers.
    Columns: model, v45_speedup, v45_n, ntr_speedup, ntr_n,
             nhw_speedup, nhw_n, nrw_speedup, nrw_n
    """
    return load_csv("ablation/real.csv")


def load_ablation_projected() -> pd.DataFrame:
    """Ablation: per-model speedup — projected V4.5 numbers."""
    return load_csv("ablation/projected.csv")


def load_full_model_comparison() -> pd.DataFrame:
    """Full model comparison: V0 vs Our Agent per model.
    Columns: model, bench_count, sum_baseline_ns,
             V0_valid, V0_total_speedup, V0_sum_opt_ns,
             No-Reward_valid, No-Reward_total_speedup, No-Reward_sum_opt_ns
    """
    return load_csv("full_model/comparison.csv")


def load_hardware_model(cluster: str) -> pd.DataFrame:
    """Per-model speedup on a given cluster.
    Columns: cluster, agent, group, group_type, geo_mean, avg, count
    """
    return load_csv(f"hardware/{cluster}_model.csv")


def load_hardware_optype(cluster: str) -> pd.DataFrame:
    """Per-op-type speedup on a given cluster."""
    return load_csv(f"hardware/{cluster}_optype.csv")


def load_benchmark_classification() -> pd.DataFrame:
    """Benchmark metadata.
    Columns: benchmark, eval_set, category, model_family, full_model, op_type
    """
    return load_csv("benchmarks/classification.csv")


# ── Color palettes ────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "Previous Agent": "#dc2626",      # red
    "Our Agent": "#16a34a",           # green
    "No-Transformer": "#16a34a",      # green
    "No-HW-Features": "#359CDB",
    "No-reward": "#eab308",
    "PyTorch Eager": "#8b5cf6",
    "PyTorch JIT": "#ec4899",
}

MODEL_COLORS = {
    "albert": "#2563eb",
    "bart": "#dc2626",
    "bert": "#16a34a",
    "convnext_tiny": "#9333ea",
    "convnext": "#9333ea",
    "densenet": "#ea580c",
    "distilbert": "#0891b2",
    "efficientnet": "#ca8a04",
    "gat": "#4f46e5",
    "gcn": "#0d9488",
    "gin": "#7c3aed",
    "gpt2": "#b45309",
    "llama3": "#be123c",
    "mobilenet": "#15803d",
    "resnet": "#1d4ed8",
    "resnext": "#0369a1",
    "t5": "#7e22ce",
    "vgg": "#b91c1c",
    "vit": "#0f766e",
    "whisper": "#92400e",
    "yolo": "#166534",
    "other": "#cbd5e1",
}

CLUSTER_COLORS = {
    "bergamo": "#dc2626",   # red
    "dalma": "#359CDB",     # blue
    "jubail": "#16a34a",    # green
}

OPTYPE_COLORS = {
    "conv2d": "#2563eb",
    "matmul": "#dc2626",
    "generic": "#16a34a",
    "pooling": "#9333ea",
}
