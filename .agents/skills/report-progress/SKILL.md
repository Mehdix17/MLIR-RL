---
name: report-progress
description: Session kickoff hub for MLIR-RL — reports training/eval/Lustre progress and suggests concrete next actions (resume failed jobs, evaluate new checkpoints, run final median eval, generate plots). Use at the start of every session or when the user asks to report progress, monitor runs, or check resources.
---

# MLIR-RL Progress Reporting — Session Kickoff

**Role:** This is the **kickoff skill for every session**. Run it first, read the 4 tables, then decide the next action. It connects the full lifecycle — training → checkpoint eval → cleanup → final median eval → plots — and routes you to the right follow-up skill.

> Companion skills: `cleanup-checkpoints` (prune after 20k), `final-checkpoint-eval` (N-run median on survivors), `plot-experimentation-results` (evolution + family/op comparison PNGs).

---

## Step 1. Resolve scope & run unified fast report

`experiments.json` (repo root) is the **single source of truth** — `fast_report.py` reads it to decide what to report. Never hardcode a dataset.

1. **Read `experiments.json`** and derive the active scope:

```bash
python -c "
import json, collections
data = json.load(open('experiments.json'))
active = [e for e in data.get('experiments', []) if not e.get('archived')]
by_ds = collections.Counter(e['dataset'] for e in active)
print('Active experiments:', len(active))
for ds, n in sorted(by_ds.items()):
    print(f'  {ds}: {n} —', ', '.join(e['name'] for e in active if e['dataset']==ds))
print('Datasets:', sorted(by_ds))
"
```

2. **Run the report for that scope** — default is `-d all` (every non-archived experimentation). Use a dataset filter only if the user explicitly asks:

```bash
python scripts/utils/fast_report.py -d all
# filtered (only when user scopes the session): -d ops_and_blocks | -d legacy_paper | -d new
```

*Redirect output to a temp file and read it to avoid truncation of wide tables.*
*Distributed (V5.1) runs show their dask worker fleet (count + nodes) in Table 1; Table 2 no longer has a Workers column.*
*Each call auto-updates `experiments.json:state` (pending → running → done/failed/stopped). Set `"archived": true` to hide a concluded experiment (e.g. `v5_single_node`). Before launching a new experiment, register it there (name, config, results_dir, dataset, mode, seed, description). If `experiments.json` is empty/missing, report that and fall back to `-d ops_and_blocks` with a warning.*

*Example of current registry (datasets adapt — do not hardcode): `ops_and_blocks: v5, v5_no_transformer` + `legacy_paper: v5, v5_no_transformer` (4 active; `v5_single_node` archived). Your run must reflect whatever `experiments.json` contains at call time.*

---

## Step 2. Present the Output Report

Print the script output **verbatim** — all 4 Markdown tables, no substitutions:

1. **Active Slurm Jobs** — `| Job ID | Dataset | Agent Version | Job Type | State | Compute Node |` — interactive sessions (`interact`/`salloc`/`srun`) are hidden; only real Train/Eval + dask workers appear. Empty → `*No active Slurm jobs found*`.
2. **Training Progress** — `| Dataset | Version | Iteration | Progress % | Status | Failures |` — `Status` is precise: `Running (STATE)` | `Finished` | `Timeout` | `Cancelled` | `Failed` | `Not Started` — no `Workers` or `Latest Ckpt` columns (workers live in Table 1, ckpt depth in Table 3).
3. **Evaluation Progress** — `| Dataset | Agent Version | Evaluated | Evaluating | Pending | Last Evaluated |` — `Max Trained Checkpoint` removed, `Last Evaluated` is last column.
4. **Lustre Storage Quota** — `| Metric | Used | Soft Limit | Hard Limit | Utilized % |`

**Mandatory shape:** every invocation MUST emit all 4 tables (header + rows, even when empty). NEVER replace a table — especially Training Progress — with prose/bullets. Keep at most one short commentary line *after* the 4 tables. **This applies to any follow-up question about train/eval state** ("what's evaluated?", "what will jobs cover?") — answer as Markdown table(s), not bullets.

---

## Step 3. Quick actions (bullet points, no table, no redundancy)

Immediately after the tables, emit a single `### Suggested Next Actions` bullet list — **do NOT emit a table and do NOT add a second `Next — pick one` section** (they were redundant). This one list is the quick actions the user asked for.

Derive bullets **from the tables you just printed** — do not hallucinate. Check each experiment in order:

| # | Trigger (read from tables) | Short bullet to emit |
|---|---|---|
| A | `Status` = `Timeout`/`Cancelled`/`Failed` and `Progress %` < 100% and no active Train job | Resume (timeout/cancelled/failed) |
| B | `Pending` > 0 and `Evaluating` = 0 | Evaluate new checkpoints (pending batch) |
| D | Training ≈100% (`Finished`/`20000`) and `models/` unpruned (>15 files) | Prune checkpoints (9+anchor) |
| E | Pruned + `eval_final/` missing/stale (<9 jsons) | Final median eval (5×) |
| F | `Finished` + `csvs/best_*` exists and `plots/…` missing/stale | Generate plots (evolution + comparison) |
| G | `Lustre` >85% inodes or >90% space | Free quota |

### Formatting (mandatory)

- Output **only bullets**, max 6, ordered 🔴 **A/G** → 🟡 **B/D/E** → 🟢 **F**, grouped by dataset.
- Each bullet = one line, **bold label**, leading color emoji, **short** (<20 words + one copy-paste command). Add a **blank line between bullets** for vertical spacing.
- Use colors: 🔴 for resume/quota (urgent), 🟡 for eval/prune/final-eval (next step), 🟢 for plots (polish), ⚪ for no-op.
- Never assume `ops_and_blocks` only — use dataset from `experiments.json`. Never invent paths.

Example (adapt to real triggers, keep short + spaced):

```markdown
### Suggested Next Actions

- 🔴 **Resume `legacy_paper/v5_no_transformer` (Failed at 62%)** — `sbatch scripts/train/train.sh config/v5/v5_no_transformer_legacy_paper.json --resume results/legacy_paper_results/v5_no_transformer_legacy_paper_agent`

- 🟡 **Eval `legacy_paper/v5_no_transformer` (167 pending)** — `python scripts/eval/submit_eval.py v5_no_transformer_legacy_paper 14100 16700 100`

- 🟡 **Prune `ops_and_blocks/v5` (42 → 9+anchor)** — `python scripts/utils/cleanup_checkpoints.py --experiment v5 --agent v5`

- 🟢 **Plots `ops_and_blocks` (v5 + v5_no_transformer)** — `python scripts/plots/generate_plots.py -d ops_and_blocks -m evolution --out-dir plots/experimentation_plots/ops_and_blocks/v5_v5_no_transformer -a v5 v5_no_transformer`
```

Rules:
- Keep bullets **short** — label + one command/skill. Details belong in the follow-up skill, not the bullet.
- Always separate bullets with a blank line.
- If no trigger fires, emit one bullet: `- ⚪ **No action needed — all caught up.**`
- After the list, wait for user reply — do not add another heading. Dispatch the chosen skill/`sbatch`; walk `D→E→F` in order and confirm before `cleanup --confirm`.

---

## Hard rules

- The 4 tables are **never optional**, never summarized as prose — even when answering follow-ups.
- Suggested actions are **derived from the just-printed tables + `experiments.json`** — never from memory or assumption. The dataset list comes from `experiments.json` at runtime; never hardcode `ops_and_blocks`.
- Training is typically **20 000 iterations** (`nb_iterations` in the train config) — treat ≥ 20k or `FINISHED`/`done` as "training finished".
- `final-checkpoint-eval` runs **only after** `cleanup-checkpoints` (eats the 9 survivors in `models/`). If `models/` is unpruned (> ~15 files) or `eval_final/` already has 9 jsons, call that out.
- Plot comparison (`best_checkpoint_*`) is **only meaningful after** `final-checkpoint-eval` (median, not single-run); evolution line can be plotted anytime.
