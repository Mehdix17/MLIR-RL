#!/usr/bin/env python3
"""Keep only the top 9 checkpoints of an experiment: 3 best (by CSV speedup) + their ±50 neighbors.

DANGER: this deletes model files. It never touches anything that isn't model_<int>.pt,
never uses rm -rf, and refuses to do anything without --confirm. Run it WITHOUT --confirm
first to see the exact keep/delete lists, then pass --confirm after explicit approval.

Usage:
  python scripts/utils/cleanup_checkpoints.py \
      --experiment <name> --agent <agent_version>                     # dry run (csv auto-resolved)
  ... same + --confirm                                                # actually delete

  Legacy experiments not in experiments.json: pass --csv <path> and --models <dir> explicitly.
"""
import argparse
import json
import os
import re
import statistics
import sys
import tempfile

MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
NEIGHBOR_STEP = 50
TOP_N = 3


def resolve_results_dir(experiment: str) -> str:
    exp_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "experiments.json")
    try:
        data = json.load(open(exp_file))
        for e in data.get("experiments", []):
            if e["name"] == experiment:
                return e["results_dir"]
    except Exception as exc:
        sys.exit(f"ERROR: cannot read experiments.json: {exc}")
    sys.exit(f"ERROR: experiment '{experiment}' not found in experiments.json")


def resolve_models_dir(models_dir: str | None, experiment: str | None) -> str:
    if models_dir:
        return models_dir
    if not experiment:
        sys.exit("ERROR: provide --models <dir> or --experiment <name> (from experiments.json)")
    return os.path.join(resolve_results_dir(experiment), "models")


def resolve_csv_path(csv_path: str | None, experiment: str | None) -> str:
    if csv_path:
        return csv_path
    if not experiment:
        sys.exit("ERROR: provide --csv or --experiment <name>")
    # Ranking CSV lives in the experiment's csvs/ folder (auto-updated by eval.py)
    return os.path.join(resolve_results_dir(experiment), "csvs", "checkpoint_speedups.csv")


def read_rankings(csv_path: str, agent: str) -> dict[int, float]:
    """Return {checkpoint: mean speedup} for the agent from the CSV.

    The CSV is per-experiment (checkpoint,speedup). Legacy aggregate CSVs with an
    agent_version column are filtered by agent.
    """
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: CSV not found: {csv_path}")
    rows: dict[int, list[float]] = {}
    with open(csv_path) as f:
        header = f.readline().strip().lower().split(",")
        try:
            i_ckpt, i_sp = header.index("checkpoint"), header.index("speedup")
        except ValueError:
            sys.exit(f"ERROR: CSV must have checkpoint and speedup columns — got: {header}")
        i_agent = header.index("agent_version") if "agent_version" in header else None
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(i_ckpt, i_sp):
                continue
            if i_agent is not None and (len(parts) <= i_agent or parts[i_agent] != agent):
                continue
            try:
                rows.setdefault(int(parts[i_ckpt]), []).append(float(parts[i_sp]))
            except ValueError:
                continue
    if not rows:
        sys.exit(f"ERROR: agent '{agent}' has no rows in {csv_path}")
    return {ckpt: statistics.mean(v) for ckpt, v in rows.items()}


def select_keep(disk_models: set[int], rankings: dict[int, float]) -> set[int]:
    """top-3 by speedup (only checkpoints that exist on disk) + their ±50 neighbors.

    The highest-numbered checkpoint is ALWAYS kept: --resume loads the latest
    model_<n>.pt, so deleting it breaks resumption.
    """
    ranked = sorted((c for c in disk_models if c in rankings), key=lambda c: rankings[c], reverse=True)
    top3 = ranked[:TOP_N]
    keep = set(top3)
    for c in top3:
        keep.add(c - NEIGHBOR_STEP)
        keep.add(c + NEIGHBOR_STEP)
    keep.add(max(disk_models))  # resume anchor
    return {c for c in keep if c in disk_models}


def plan(models_dir: str, csv_path: str, agent: str):
    disk_models = {int(m.group(1)) for f in os.listdir(models_dir) if (m := MODEL_RE.match(f))}
    if not disk_models:
        sys.exit(f"ERROR: no model_<n>.pt files in {models_dir}")
    rankings = read_rankings(csv_path, agent)
    ranked = sorted((c for c in disk_models if c in rankings), key=lambda c: rankings[c], reverse=True)
    keep = select_keep(disk_models, rankings)
    delete = disk_models - keep
    missing = sorted(set(rankings) - disk_models)
    return disk_models, ranked, keep, delete, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep only top-9 checkpoints of an experiment (3 best +-50 neighbors)")
    ap.add_argument("--csv", required=False,
                    help="path to checkpoint_speedups.csv (default: <results_dir>/csvs/checkpoint_speedups.csv "
                         "when --experiment is given; required otherwise)")
    ap.add_argument("--models", default=None, help="models/ dir (or use --experiment)")
    ap.add_argument("--experiment", default=None, help="experiment name in experiments.json")
    ap.add_argument("--agent", required=False, help="agent_version column value in the CSV (required unless --self-test)")
    ap.add_argument("--confirm", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--self-test", action="store_true", help="run internal sanity check and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.agent:
        ap.error("--agent is required (or use --self-test)")
    if not args.csv and not args.experiment:
        ap.error("provide --csv or --experiment")

    models_dir = os.path.abspath(resolve_models_dir(args.models, args.experiment))
    csv_path = resolve_csv_path(args.csv, args.experiment)
    if not os.path.isdir(models_dir):
        sys.exit(f"ERROR: models dir not found: {models_dir}")
    disk_models, ranked, keep, delete, missing = plan(models_dir, csv_path, args.agent)

    print(f"models dir : {models_dir}")
    print(f"csv        : {csv_path} (agent '{args.agent}')")
    print(f"on disk    : {len(disk_models)} checkpoints")
    print(f"resume anchor: model_{max(disk_models)}.pt (always kept — --resume loads the latest)")
    rankings = read_rankings(csv_path, args.agent)
    print(f"ranked top3: {[(c, round(rankings[c], 3)) for c in ranked[:TOP_N]]}")
    print(f"KEEP ({len(keep)}): {sorted(keep)}")
    print(f"DELETE ({len(delete)}): {sorted(delete)}")
    if missing:
        print(f"NOTE: {len(missing)} CSV checkpoints not on disk (ignored): {missing}")
    if not delete:
        print("Nothing to delete — models/ already within the keep set.")
        return 0
    if not args.confirm:
        print("\nDRY RUN — nothing deleted. Re-run with --confirm to apply.")
        return 0
    print(f"\nCONFIRMED — deleting {len(delete)} file(s):")
    for c in sorted(delete):
        path = os.path.join(models_dir, f"model_{c}.pt")
        os.remove(path)
        print(f"  removed model_{c}.pt")
    print(f"Done. {len(os.listdir(models_dir))} files remain in models/.")
    return 0


def self_test() -> int:
    """Verify keep/delete selection against a synthetic experiment."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        for n in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800]:
            open(os.path.join(tmp, "models", f"model_{n}.pt"), "w").close()
        csv = os.path.join(tmp, "ck.csv")
        # best speedups at 400, 200, 600 → keep {200,250,350,400,450,550,600} + neighbors {150,650} = 9
        with open(csv, "w") as f:
            f.write("agent_version,checkpoint,speedup\r\n")
            for n, sp in [(100, 1.0), (150, 1.1), (200, 2.0), (250, 1.2), (300, 1.3),
                          (350, 1.4), (400, 2.5), (450, 1.5), (500, 1.6), (550, 1.7),
                          (600, 1.9), (650, 1.8), (700, 1.2), (800, 1.1)]:
                f.write(f"my_agent,{n},{sp}\r\n")
        disk, ranked, keep, delete, _ = plan(os.path.join(tmp, "models"), csv, "my_agent")
        # 800 is the latest checkpoint → resume anchor, kept even though its speedup (1.1) is low
        expected_keep = {150, 200, 250, 350, 400, 450, 550, 600, 650, 800}
        assert keep == expected_keep, f"keep={sorted(keep)} expected={sorted(expected_keep)}"
        assert delete == disk - expected_keep
        assert len(keep) == 10
        assert max(disk) in keep  # resume anchor invariant
        print(f"self-test PASS: keep={sorted(keep)} delete={sorted(delete)}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
