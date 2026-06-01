"""Data loading and benchmark utilities for the dashboard."""

import pandas as pd
from pathlib import Path
from typing import Optional

_CSV_DIR: Optional[Path] = None


def set_csv_dir(path: Path):
    global _CSV_DIR
    _CSV_DIR = path


def get_csv_dir() -> Path:
    global _CSV_DIR
    if _CSV_DIR is None:
        # Auto-resolve: project_root/results/new_dataset_results/dashboard/
        p = Path(__file__).resolve().parent.parent.parent / "results" / "new_dataset_results" / "dashboard"
        _CSV_DIR = p
    return _CSV_DIR


def load_csv(name: str) -> pd.DataFrame:
    path = get_csv_dir() / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_benchmarks_df() -> pd.DataFrame:
    """Load all agent eval CSVs stacked together with an 'agent' column."""
    registry = load_csv("agent_registry.csv")
    frames = []
    for _, row in registry.iterrows():
        csv_file = row["csv_file"]
        df = load_csv(csv_file)
        if df.empty:
            continue
        df["agent"] = row["agent_key"]
        df["agent_display"] = row["display_name"]
        df["category"] = row["category"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_version_comparison() -> pd.DataFrame:
    return load_csv("version_comparison.csv")


def load_ablation_summary() -> pd.DataFrame:
    return load_csv("ablation_summary.csv")


def load_agent_registry() -> pd.DataFrame:
    return load_csv("agent_registry.csv")


def get_family_colors() -> dict:
    return {
        "cnn": "#2563eb",
        "encoder_transformer": "#dc2626",
        "decoder_seq2seq": "#16a34a",
        "llm": "#9333ea",
        "vision_transformer": "#ea580c",
        "gnn": "#0891b2",
        "audio": "#4f46e5",
        "detection": "#ca8a04",
        "legacy_synthetic": "#94a3b8",
        "legacy_single": "#64748b",
        "other": "#cbd5e1",
    }


def get_agent_colors() -> dict:
    return {
        "v0": "#94a3b8",
        "v4_5": "#2563eb",
        "no_transformer": "#16a34a",
        "no_hw": "#dc2626",
        "no_reward": "#f59e0b",
    }


def get_agent_display_name(key: str) -> str:
    names = {
        "v0": "V0 (Baseline RL)",
        "v4_5": "V4.5 (Robust)",
        "no_transformer": "No Transformer",
        "no_hw": "No HW",
        "no_reward": "No Reward",
    }
    return names.get(key, key)


def build_benchmark_index(df: pd.DataFrame) -> dict:
    """Return {benchmark_name: family} and {family: [benchmark_names]}"""
    bench_family = {}
    family_benches = {}
    for _, row in df.iterrows():
        b, f = row["benchmark"], row["family"]
        bench_family[b] = f
        family_benches.setdefault(f, []).append(b)
    return bench_family, family_benches


def parse_model_name(benchmark: str) -> str:
    """Extract model name from benchmark, e.g. 'albert_block_1004' -> 'albert'"""
    for prefix in ["albert", "bert", "distilbert", "bart", "gpt2", "t5",
                   "llama3", "convnext_tiny", "convnext", "efficientnet",
                   "mobilenet", "resnet50", "resnext50", "resnet", "resnext",
                   "vgg16", "vgg", "vit", "whisper", "yolov8m", "yolo",
                   "gat", "gin", "gcn", "densenet", "bench", "single"]:
        if benchmark.lower().startswith(prefix):
            return prefix
    return "other"
