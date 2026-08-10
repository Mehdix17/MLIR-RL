#!/usr/bin/env python3
"""Post-hoc analysis of Optuna HPO study results.

Usage:
    python scripts/hpo/analyze.py
    python scripts/hpo/analyze.py --study-db scripts/hpo/study.db
    python scripts/hpo/analyze.py --top 10
"""

import argparse
import json
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HPO_DIR = PROJECT_ROOT / "scripts" / "hpo"


def main():
    parser = argparse.ArgumentParser(description="Analyze HPO study results")
    parser.add_argument("--study-db", default=str(HPO_DIR / "study.db"), help="Path to Optuna SQLite DB")
    parser.add_argument("--top", type=int, default=10, help="Number of top trials to show")
    parser.add_argument("--save-best", default=str(HPO_DIR / "best_config.json"), help="Path to save best config")
    args = parser.parse_args()

    study = optuna.load_study(study_name="paper_transformer_hpo", storage=f"sqlite:///{args.study_db}")

    # Filter completed trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"{'='*60}")
    print(f"HPO STUDY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total trials:    {len(study.trials)}")
    print(f"Completed:       {len(completed)}")
    print(f"Pruned:          {len(pruned)}")
    print(f"Failed:          {len(failed)}")

    if not completed:
        print("\nNo completed trials to analyze.")
        return

    # Best trial
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL")
    print(f"{'='*60}")
    print(f"Trial:    {best.number}")
    print(f"Speedup:  {best.value:.4f}")
    print(f"Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Top N trials
    sorted_trials = sorted(completed, key=lambda t: t.value if t.value else 0, reverse=True)
    print(f"\n{'='*60}")
    print(f"TOP {min(args.top, len(sorted_trials))} TRIALS")
    print(f"{'='*60}")
    for i, trial in enumerate(sorted_trials[:args.top]):
        print(f"\n#{i+1} trial={trial.number} speedup={trial.value:.4f}")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")

    # Parameter importance (if enough trials)
    if len(completed) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n{'='*60}")
            print(f"PARAMETER IMPORTANCE")
            print(f"{'='*60}")
            for param, score in importance.items():
                bar = "#" * int(score * 50)
                print(f"  {param:35s} {score:.3f} {bar}")
        except Exception as e:
            print(f"\nCould not compute parameter importance: {e}")

    # Stats summary
    values = [t.value for t in completed if t.value is not None]
    if values:
        print(f"\n{'='*60}")
        print(f"SPEEDUP STATISTICS")
        print(f"{'='*60}")
        print(f"  Mean:    {sum(values)/len(values):.4f}")
        print(f"  Median:  {sorted(values)[len(values)//2]:.4f}")
        print(f"  Best:    {max(values):.4f}")
        print(f"  Worst:   {min(values):.4f}")
        print(f"  Std:     {(sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5:.4f}")

    # Save best config as a real config
    if best.params:
        base_config = json.loads((HPO_DIR / "base_config.json").read_text())
        base_config.update(best.params)
        base_config["results_dir"] = f"results/hpo/trial_{best.number}"
        base_config["tags"] = ["ops_and_blocks", "paper", "transformer", "hpo", "best"]
        Path(args.save_best).write_text(json.dumps(base_config, indent=2))
        print(f"\nBest config saved to: {args.save_best}")

    # Try to generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("HPO Study Analysis", fontsize=14)

        # 1. Optimization history
        ax = axes[0, 0]
        trials = [t.number for t in completed]
        values = [t.value for t in completed]
        ax.plot(trials, values, "o-", markersize=3)
        best_so_far = []
        current_best = float("-inf")
        for v in values:
            current_best = max(current_best, v)
            best_so_far.append(current_best)
        ax.plot(trials, best_so_far, "r-", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Speedup")
        ax.set_title("Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Parameter importance (if available)
        ax = axes[0, 1]
        if len(completed) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                scores = list(importance.values())
                y_pos = range(len(params))
                ax.barh(y_pos, scores)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([p.replace("transformer_", "") for p in params], fontsize=8)
                ax.set_xlabel("Importance")
                ax.set_title("Parameter Importance")
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "Need >= 10 trials", ha="center", va="center")

        # 3. Speedup distribution
        ax = axes[1, 0]
        ax.hist(values, bins=min(20, len(values)), edgecolor="black", alpha=0.7)
        ax.axvline(sum(values)/len(values), color="red", linestyle="--", label="Mean")
        ax.set_xlabel("Speedup")
        ax.set_ylabel("Count")
        ax.set_title("Speedup Distribution")
        ax.legend()

        # 4. Parallel coordinate (top params)
        ax = axes[1, 1]
        if len(completed) >= 5:
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=ax)
                ax.set_title("Parallel Coordinate")
            except Exception:
                ax.text(0.5, 0.5, "Could not generate", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "Need >= 5 trials", ha="center", va="center")

        plt.tight_layout()
        plot_path = HPO_DIR / "analysis_plots.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plots saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping plots")
    except Exception as e:
        print(f"\nPlot generation failed: {e}")


if __name__ == "__main__":
    main()
