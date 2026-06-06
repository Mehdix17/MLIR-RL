#!/usr/bin/env python3
"""
Build CSV files for all 3 graphs.
Reads eval results + benchmark classification, outputs aggregated CSVs.
"""

import os, json, csv, math
from collections import defaultdict

CLASSIFICATION = "plots/benchmark_classification.csv"
BASE_EVAL = "results/new_dataset_results/baselines/mlir/eval_base.json"
BASE_FULL = "results/new_dataset_results/baselines/mlir/eval_full_base.json"
OUT_DIR = "plots"

# ---- Data sources ----
SOURCES = {
    # Graph 1 & 2: eval_base
    "V0_bergamo": {
        "type": "markers",
        "path": "results/new_dataset_results/v0_agent/old_agent/run_30/eval/markers",
        "checkpoint": 600,
        "cluster": "bergamo",
    },
    "NoReward_bergamo": {
        "type": "markers",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/global_markers/110",
        "checkpoint": 1100,
        "cluster": "bergamo",
    },
    "NoReward_dalma": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/rl_autoschedular_v45_no_shaped_reward_agent/run_129/logs/eval/eval_exec_times.json",
        "checkpoint": 1100,
        "cluster": "dalma",
    },
    "NoReward_jubail": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/rl_autoschedular_v45_no_shaped_reward_agent/run_130/logs/eval/eval_exec_times.json",
        "checkpoint": 1100,
        "cluster": "jubail",
    },
    "NoHW_NoReward_bergamo": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/eval/checkpoint_1100_bergamo.json",
        "checkpoint": 1100,
        "cluster": "bergamo",
    },
    "NoHW_NoReward_dalma": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/eval/checkpoint_1100_dalma.json",
        "checkpoint": 1100,
        "cluster": "dalma",
    },
    "NoHW_NoReward_jubail": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/eval/checkpoint_1100_jubail.json",
        "checkpoint": 1100,
        "cluster": "jubail",
    },
    # Graph 3: eval_full
    "V0_full": {
        "type": "eval_json",
        "path": "results/new_dataset_results/v0_agent/old_agent/run_45/logs/eval/eval_exec_times.json",
        "checkpoint": 600,
        "cluster": "bergamo",
    },
    "NoReward_full": {
        "type": "eval_json",
        "path": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent/rl_autoschedular_v45_no_shaped_reward_agent/run_124/logs/eval/eval_exec_times.json",
        "checkpoint": 1100,
        "cluster": "bergamo",
    },
}


def load_classification():
    """Return {benchmark_name: row_dict}."""
    lookup = {}
    with open(CLASSIFICATION) as f:
        for row in csv.DictReader(f):
            lookup[row["benchmark"]] = row
    return lookup


def load_speedups(source_key):
    """Load {benchmark_name: speedup} from a source. Returns empty dict if not available."""
    src = SOURCES[source_key]
    path = src["path"]
    stype = src["type"]

    if not os.path.exists(path):
        return {}

    speedups = {}
    if stype == "markers":
        for fname in os.listdir(path):
            if fname == "_eval_meta.json":
                continue
            try:
                with open(os.path.join(path, fname)) as f:
                    m = json.load(f)
            except:
                continue
            bench = fname
            sp = m.get("speedup")
            if sp is not None and sp > 0:
                speedups[bench] = sp
    elif stype == "eval_json":
        with open(path) as f:
            d = json.load(f)
        # Need baseline to compute speedup
        eval_base = None
        with open(BASE_EVAL) as f:
            eval_base = json.load(f)
        with open(BASE_FULL) as fb:
            eval_base.update(json.load(fb))
        for bench, opt_time in d.items():
            if opt_time is None or opt_time <= 0:
                continue
            bl = eval_base.get(bench)
            if bl and bl > 0:
                speedups[bench] = bl / opt_time
    return speedups


def geo_mean(values):
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


def arithmetic_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


# ============================================================
# GRAPH 1: Overall Performance (V0 vs No-Reward, by op_type)
# ============================================================
def build_graph1():
    cls = load_classification()
    v0_spd = load_speedups("V0_bergamo")
    nw_spd = load_speedups("NoReward_bergamo")

    groupings = {
        "matmul": {"V0": [], "No-Reward": []},
        "batch_matmul": {"V0": [], "No-Reward": []},
        "conv2d": {"V0": [], "No-Reward": []},
        "generic": {"V0": [], "No-Reward": []},
        "pooling": {"V0": [], "No-Reward": []},
        "block": {"V0": [], "No-Reward": []},
    }

    for bench, row in cls.items():
        if row["eval_set"] != "eval_base":
            continue
        op = row["op_type"]
        # Graph 1: exclude model blocks — only single ops (legacy + model)
        if row["category"] == "model_block":
            continue
        if op not in groupings:
            groupings[op] = {"V0": [], "No-Reward": []}
        if bench in v0_spd:
            groupings[op]["V0"].append(v0_spd[bench])
        if bench in nw_spd:
            groupings[op]["No-Reward"].append(nw_spd[bench])

    with open(os.path.join(OUT_DIR, "graph1_performance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_type", "V0_geo_mean", "V0_avg", "V0_count", "NoReward_geo_mean", "NoReward_avg", "NoReward_count"])
        for op in sorted(groupings):
            g = groupings[op]
            if not g["V0"] and not g["No-Reward"]:
                continue
            w.writerow([
                op,
                round(geo_mean(g["V0"]), 4) if g["V0"] else "",
                round(arithmetic_mean(g["V0"]), 4) if g["V0"] else "",
                len(g["V0"]),
                round(geo_mean(g["No-Reward"]), 4) if g["No-Reward"] else "",
                round(arithmetic_mean(g["No-Reward"]), 4) if g["No-Reward"] else "",
                len(g["No-Reward"]),
            ])

    print("graph1_performance.csv: done")


# ============================================================
# GRAPH 2: Multi-Hardware
# ============================================================
def build_graph2():
    cls = load_classification()

    agents = [
        ("NoReward", "No-Reward (HW)"),
        ("NoHW_NoReward", "No-Reward (No-HW)"),
    ]
    clusters = ["bergamo", "dalma", "jubail"]

    for cluster in clusters:
        rows_op = []
        rows_model = []

        for src_key, display in agents:
            full_key = f"{src_key}_{cluster}"
            if full_key not in SOURCES:
                continue
            spd = load_speedups(full_key)
            if not spd:
                continue

            # By op_type
            ops = defaultdict(list)
            for bench, s in spd.items():
                if bench not in cls:
                    continue
                row = cls[bench]
                if row["eval_set"] != "eval_base":
                    continue
                ops[row["op_type"]].append(s)

            for op in sorted(ops):
                vals = ops[op]
                rows_op.append({
                    "cluster": cluster,
                    "agent": display,
                    "group": op,
                    "group_type": "op_type",
                    "geo_mean": round(geo_mean(vals), 4),
                    "avg": round(arithmetic_mean(vals), 4),
                    "count": len(vals),
                })

            # By model
            models = defaultdict(list)
            for bench, s in spd.items():
                if bench not in cls:
                    continue
                row = cls[bench]
                if row["eval_set"] != "eval_base":
                    continue
                models[row["full_model"]].append(s)

            for mdl in sorted(models):
                vals = models[mdl]
                rows_model.append({
                    "cluster": cluster,
                    "agent": display,
                    "group": mdl,
                    "group_type": "model",
                    "geo_mean": round(geo_mean(vals), 4),
                    "avg": round(arithmetic_mean(vals), 4),
                    "count": len(vals),
                })

        # Write per-cluster CSV
        if rows_op:
            with open(os.path.join(OUT_DIR, f"graph2_{cluster}_optype.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["cluster", "agent", "group", "group_type", "geo_mean", "avg", "count"])
                w.writeheader()
                w.writerows(rows_op)

        if rows_model:
            with open(os.path.join(OUT_DIR, f"graph2_{cluster}_model.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["cluster", "agent", "group", "group_type", "geo_mean", "avg", "count"])
                w.writeheader()
                w.writerows(rows_model)

    print("graph2 CSVs: done")


# ============================================================
# GRAPH 3: Full Model Support
# ============================================================
def build_graph3():
    cls = load_classification()

    with open(BASE_FULL) as f:
        base_full = json.load(f)

    agents_full = {
        "V0": load_speedups("V0_full"),
        "No-Reward": load_speedups("NoReward_full"),
    }

    rows = []
    for model in sorted(set(r["full_model"] for r in cls.values() if r["eval_set"] == "eval_full")):
        # Sum baseline and optimized times for all benchmarks of this model
        model_benches = [b for b, r in cls.items() if r["full_model"] == model and r["eval_set"] == "eval_full"]
        if not model_benches:
            continue

        sum_baseline = sum(base_full.get(b, 0) for b in model_benches if base_full.get(b, 0) > 0)
        if sum_baseline <= 0:
            continue

        row = {"model": model, "bench_count": len(model_benches), "sum_baseline_ns": sum_baseline}
        for agent_name, spd_map in agents_full.items():
            sum_opt = 0
            valid = 0
            for b in model_benches:
                bl = base_full.get(b, 0)
                if bl <= 0:
                    continue
                sp = spd_map.get(b)
                if sp and sp > 0:
                    sum_opt += bl / sp
                    valid += 1
            row[f"{agent_name}_valid"] = valid
            row[f"{agent_name}_total_speedup"] = round(sum_baseline / sum_opt, 4) if sum_opt > 0 else ""
            row[f"{agent_name}_sum_opt_ns"] = round(sum_opt, 2)
        rows.append(row)

    with open(os.path.join(OUT_DIR, "graph3_fullmodel.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "bench_count", "sum_baseline_ns",
            "V0_valid", "V0_total_speedup", "V0_sum_opt_ns",
            "No-Reward_valid", "No-Reward_total_speedup", "No-Reward_sum_opt_ns",
        ])
        w.writeheader()
        w.writerows(rows)

    print("graph3_fullmodel.csv: done")


if __name__ == "__main__":
    build_graph1()
    build_graph2()
    build_graph3()
