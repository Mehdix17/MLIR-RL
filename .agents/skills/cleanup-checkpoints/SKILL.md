---
name: cleanup-checkpoints
description: Keep only the top 9 checkpoints of an experiment's models/ folder (3 best by CSV speedup + their ±50 neighbors), deleting the rest. DANGEROUS, irreversible operation — must follow the dry-run → user-confirmation → apply flow exactly. Use when the user asks to clean/prune/trim checkpoints or free space in a models/ folder.
---

# Checkpoint Cleanup — Top-9 Retention

This skill deletes model checkpoint files. **A wrong deletion is irreversible.**
Follow the flow below EXACTLY. Never skip a step, never improvise the keep/delete
selection, never delete without explicit user confirmation.

## What gets kept

1. **Top 3 checkpoints** by mean speedup for the agent, read from
   `<results_dir>/csvs/checkpoint_speedups.csv` — the experiment's ranking CSV,
   auto-updated by the eval script and regenerable via `utils/csvs.py`
   (columns: `checkpoint, speedup` — the CSV is per-experiment).
2. **Neighbors** of each top checkpoint: `±50` (e.g. top = 200 → also 150 and 250).
3. Final keep set = (top3 ∪ neighbors) **∩ checkpoints that actually exist on disk**,
   **plus the highest-numbered checkpoint (the `--resume` anchor — resume loads the
   latest `model_<n>.pt`, so it must never be deleted)**.
   Everything else in `models/` is deleted. Nothing outside `model_<n>.pt` files is
   ever touched.

## Flow (mandatory)

### Step 1 — Locate inputs
- Resolve the experiment's `models/` dir from `experiments.json` (repo root):
  `<results_dir>/models`. If the experiment is not registered, use the explicit
  `--models <dir>`.
- The ranking CSV is `<results_dir>/csvs/checkpoint_speedups.csv` (auto-resolved
  from `--experiment`; auto-updated after each eval, regenerable with
  `python utils/csvs.py --results-dir <dir> --agent <name>`). If it doesn't
  exist or the agent has no rows there → **STOP**, tell the user the ranking data is
  missing. Do not delete anything without ranking data.

### Step 2 — Dry run (no deletion)
```bash
python scripts/utils/cleanup_checkpoints.py \
  --experiment <name> \
  --agent <agent_version>
```
(For legacy experiments not in `experiments.json`: pass `--csv <path>` and
`--models <dir>` explicitly.)
The script prints: checkpoints on disk, ranked top-3 (with speedups), the exact
KEEP list, the exact DELETE list, and any CSV checkpoints missing from disk.

### Step 3 — Present to the user
Show the user the KEEP and DELETE lists verbatim. State clearly: `N` files will be
permanently deleted, `M` kept. **Wait for an explicit "yes/proceed"** — never infer
approval from silence.

### Step 4 — Apply (only after explicit confirmation)
```bash
python scripts/utils/cleanup_checkpoints.py \
  --experiment <name> \
  --agent <agent_version> \
  --confirm
```
The script re-prints the delete list and removes only those files.

### Step 5 — Verify
- Confirm the kept files are all present: `ls <models_dir>`
- Confirm the count matches the KEEP list size from Step 2.

## Hard rules (do not violate)

- **Never** delete without `--confirm` AND the user's explicit approval.
- **Never** deviate from the script's keep/delete selection — the script is the
  single source of truth; it only touches files matching `model_<n>.pt`.
- **Never** delete the highest-numbered checkpoint (the `--resume` anchor) — the
  script always keeps it; verify the dry-run KEEP list contains it before applying.
- **Never** delete checkpoints not ranked in the CSV. If the CSV is stale/missing
  rows for the agent, STOP and report — do not guess.
- **Never** use `rm`/`find -delete`/shell wildcards for this. Use the script only.
- If the dry-run shows "Nothing to delete" → done, no confirmation needed.
- If anything is ambiguous (agent name mismatch, CSV columns unexpected, models
  dir empty) → stop and ask, don't proceed.

## Self-check

The script carries its own sanity test (synthetic experiment, verifies the exact
top-3 + neighbors selection):
```bash
python scripts/utils/cleanup_checkpoints.py --self-test
```
Run it before first use and after any modification to the script.
