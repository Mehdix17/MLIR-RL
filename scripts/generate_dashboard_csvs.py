"""
Generate dashboard CSVs from eval markers and baselines.

Reads eval_marker_mapping.csv to find the best checkpoint (highest avg speedup)
for each agent, extracts speedup/exec_time from marker files,
joins with baselines, and writes clean CSVs to results/new_dataset_results/dashboard/.

Usage:
    python scripts/generate_dashboard_csvs.py
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "new_dataset_results"
MAPPING_CSV = RESULTS_DIR / "eval_marker_mapping.csv"
BASELINE_CSV = RESULTS_DIR / "baselines" / "eval_merged.csv"
OUTPUT_DIR = RESULTS_DIR / "dashboard"

FAMILY_MAP = {
    "albert": "encoder_transformer",
    "bert": "encoder_transformer",
    "distilbert": "encoder_transformer",
    "bart": "decoder_seq2seq",
    "gpt2": "decoder_seq2seq",
    "t5": "decoder_seq2seq",
    "llama3": "llm",
    "convnext_tiny": "cnn",
    "convnext": "cnn",
    "efficientnet": "cnn",
    "mobilenet": "cnn",
    "resnet50": "cnn",
    "resnet": "cnn",
    "resnext50": "cnn",
    "resnext": "cnn",
    "vgg16": "cnn",
    "vgg": "cnn",
    "densenet": "cnn",
    "vit": "vision_transformer",
    "whisper": "audio",
    "yolov8m": "detection",
    "yolo": "detection",
    "gat": "gnn",
    "gin": "gnn",
    "gcn": "gnn",
    "bench": "legacy_synthetic",
    "single": "legacy_single",
}


def assign_family(benchmark: str) -> str:
    for prefix, family in FAMILY_MAP.items():
        if benchmark.lower().startswith(prefix):
            return family
    return "other"


def get_best_checkpoint_for_agent(model_key: str) -> tuple[str, int, Path]:
    """Scan all checkpoints, pick the one with highest average speedup.
    Returns (display_name, best_checkpoint_number, markers_dir_path)."""
    candidates = []
    with open(MAPPING_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model_key and row["status"] == "complete":
                candidates.append(row)

    if not candidates:
        return None, None, None

    best_ckpt = None
    best_avg = -1
    best_dir = None
    best_display = None

    for row in candidates:
        markers_dir = Path(row["markers_dir"])
        if not markers_dir.exists():
            continue
        speeds = []
        for marker_file in markers_dir.iterdir():
            if marker_file.is_dir():
                continue
            try:
                data = json.loads(marker_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sp = data.get("speedup") or 0
            if sp and sp > 0:
                speeds.append(sp)
        if not speeds:
            continue
        avg_sp = sum(speeds) / len(speeds)
        if avg_sp > best_avg:
            best_avg = avg_sp
            best_ckpt = int(row["checkpoint"])
            best_dir = markers_dir
            best_display = row["display_name"]

    return best_display, best_ckpt, best_dir


def load_baselines() -> dict[str, dict]:
    """Return {benchmark: {'mlir_baseline': int, 'pytorch_eager': int, 'pytorch_jit': int}}"""
    baselines = {}
    with open(BASELINE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            baselines[row["benchmark"]] = {
                "mlir_baseline": int(float(row["mlir_baseline"])),
                "pytorch_eager": int(float(row["pytorch_eager"])),
                "pytorch_jit": int(float(row["pytorch_jit"])),
            }
    return baselines


def extract_agent_data(agent_key: str, agent_display: str, markers_dir: Path,
                       baselines: dict) -> list[dict]:
    """Read marker files and produce per-benchmark rows."""
    rows = []
    for marker_file in sorted(markers_dir.iterdir()):
        if marker_file.is_dir():
            continue
        bench = marker_file.name
        try:
            data = json.loads(marker_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        speedup = data.get("speedup") or 0
        exec_time = data.get("exec_time") or 0
        if not speedup or not exec_time or speedup <= 0 or exec_time <= 0:
            continue
        baseline = baselines.get(bench, {})
        family = assign_family(bench)
        rows.append({
            "benchmark": bench,
            "family": family,
            "mlir_baseline": baseline.get("mlir_baseline", ""),
            "pytorch_eager": baseline.get("pytorch_eager", ""),
            "pytorch_jit": baseline.get("pytorch_jit", ""),
            "mlir_rl_exec_time": int(exec_time),
            "speedup": round(speedup, 4),
        })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baselines = load_baselines()
    print(f"Loaded {len(baselines)} baselines")

    agents = {
        "V0": ("v0", "V0"),
        "V4.5": ("v4_5", "V4.5"),
        "No-Transformer": ("no_transformer", "No-Transformer"),
        "No-HW": ("no_hw", "No-HW"),
        "No-Reward": ("no_reward", "No-Reward"),
    }
    fieldnames = ["benchmark", "family", "mlir_baseline", "pytorch_eager",
                   "pytorch_jit", "mlir_rl_exec_time", "speedup"]

    all_agent_data = {}
    for model_key, (csv_prefix, _) in agents.items():
        display_name, best_ckpt, markers_dir = get_best_checkpoint_for_agent(model_key)
        if markers_dir is None:
            print(f"  {model_key}: no complete checkpoint found, skipping")
            continue
        rows = extract_agent_data(model_key, display_name, markers_dir, baselines)
        out_path = OUTPUT_DIR / f"{csv_prefix}_eval.csv"
        write_csv(out_path, rows, fieldnames)
        all_agent_data[csv_prefix] = {r["benchmark"]: r for r in rows}
        print(f"  {model_key} ({display_name}) ckpt={best_ckpt}: {len(rows)} benchmarks → {out_path}")

    # Combined version comparison: V0 vs V4.5
    if "v0" in all_agent_data and "v4_5" in all_agent_data:
        v0_data = all_agent_data["v0"]
        v45_data = all_agent_data["v4_5"]
        comp_rows = []
        comp_fields = ["benchmark", "family", "mlir_baseline", "pytorch_eager",
                       "pytorch_jit", "v0_exec_time", "v0_speedup",
                       "v45_exec_time", "v45_speedup", "v45_vs_v0_ratio"]
        common = set(v0_data) & set(v45_data)
        for bench in sorted(common):
            v0 = v0_data[bench]
            v45 = v45_data[bench]
            ratio = round(v45["speedup"] / v0["speedup"], 4) if v0["speedup"] > 0 else 0
            comp_rows.append({
                "benchmark": bench,
                "family": v0["family"],
                "mlir_baseline": v0["mlir_baseline"],
                "pytorch_eager": v0["pytorch_eager"],
                "pytorch_jit": v0["pytorch_jit"],
                "v0_exec_time": v0["mlir_rl_exec_time"],
                "v0_speedup": v0["speedup"],
                "v45_exec_time": v45["mlir_rl_exec_time"],
                "v45_speedup": v45["speedup"],
                "v45_vs_v0_ratio": ratio,
            })
        out_path = OUTPUT_DIR / "version_comparison.csv"
        write_csv(out_path, comp_rows, comp_fields)
        print(f"  V0 vs V4.5: {len(comp_rows)} common benchmarks → {out_path}")

    # Ablation summary: per-family averages for V4.5 + 3 ablations
    ablation_keys = ["v4_5", "no_transformer", "no_hw", "no_reward"]
    available_ablation = [k for k in ablation_keys if k in all_agent_data]
    if len(available_ablation) >= 2:
        families = sorted(set(f for k in available_ablation for f in
                              set(r["family"] for r in all_agent_data[k].values())))
        sum_rows = []
        for family in families:
            row = {"family": family}
            for key in available_ablation:
                f_benchmarks = [r for r in all_agent_data[key].values() if r["family"] == family]
                if f_benchmarks:
                    avg = round(sum(r["speedup"] for r in f_benchmarks) / len(f_benchmarks), 4)
                    row[f"{key}_speedup"] = avg
                    row[f"{key}_n"] = len(f_benchmarks)
            sum_rows.append(row)

        sum_fields = ["family"] + [f"{k}_{s}" for k in available_ablation
                                   for s in ["speedup", "n"]]
        out_path = OUTPUT_DIR / "ablation_summary.csv"
        write_csv(out_path, sum_rows, sum_fields)
        print(f"  Ablation summary: {len(sum_rows)} families → {out_path}")

    # Agent registry for the dashboard home page
    agent_registry = [
        {"agent_key": "v0", "display_name": "V0 (Baseline RL)",
         "description": "LSTM-based policy, no hardware awareness, flat reward",
         "category": "baseline", "csv_file": "v0_eval.csv"},
        {"agent_key": "v4_5", "display_name": "V4.5 (Robust Integration)",
         "description": "Transformer encoder + hardware awareness + shaped reward + process isolation",
         "category": "main", "csv_file": "v4_5_eval.csv"},
        {"agent_key": "no_transformer", "display_name": "No Transformer",
         "description": "Ablation: LSTM policy instead of Transformer (V4.5 minus transformer)",
         "category": "ablation", "csv_file": "no_transformer_eval.csv"},
        {"agent_key": "no_hw", "display_name": "No Hardware Observation",
         "description": "Ablation: hardware features disabled (V4.5 minus HW awareness)",
         "category": "ablation", "csv_file": "no_hw_eval.csv"},
        {"agent_key": "no_reward", "display_name": "No Shaped Reward",
         "description": "Ablation: reward shaping disabled (V4.5 minus shaped reward)",
         "category": "ablation", "csv_file": "no_reward_eval.csv"},
    ]
    reg_fields = ["agent_key", "display_name", "description", "category", "csv_file"]
    out_path = OUTPUT_DIR / "agent_registry.csv"
    write_csv(out_path, agent_registry, reg_fields)
    print(f"  Agent registry: {len(agent_registry)} agents → {out_path}")

    print(f"\nDone. Dashboard CSVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
