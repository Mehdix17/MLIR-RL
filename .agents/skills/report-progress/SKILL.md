---
name: report-progress
description: Unified runbook for reporting MLIR-RL training progress, evaluation progress, and Lustre quota usage. Use this skill when the user asks to report progress, monitor runs, or check resources.
---

# MLIR-RL Progress Reporting Runbook (Unified)

Use this skill to fetch, synchronize, and report the current status of all training jobs, evaluation batches, and storage quotas.

---

## 🛠️ Step 1. Fetch Training Progress
Run the training report script to get the exact iteration counts, trend lines, and status of running/completed training jobs:
```bash
python scripts/utils/report_training.py -d ops_and_blocks
```
*Note: Save the output to a temporary file and read it to prevent truncation of wide tables.*

---

## 🛠️ Step 2. Synchronize and Read Evaluation Progress
Sync the active Slurm evaluation jobs and read the updated evaluation progress tracker:
```bash
python scripts/eval/sync_progress.py
```
After running the sync script, read the contents of the tracker file:
[docs/results/eval_progress.md](file:///scratch/mb10856/MLIR-RL/docs/results/eval_progress.md)

---

## 🛠️ Step 3. Check Lustre Storage Quota
Lustre has a soft limit of 500K files and a hard limit of 1M files. Check the quota to ensure training/eval jobs don't crash:
```bash
lfs quota -u $USER /scratch
```

---

## 📊 Step 4. Format and Present the Report
Present a clean, concise, unified report to the user using Markdown tables:

1. **Active Slurm Jobs Table**: Show job ID, agent version, type (Train/Eval), state, and compute node.
2. **Training Progress Table**: Show version, iteration, progress %, status, latest checkpoint, and trend.
3. **Evaluation Progress Table**: Show version, evaluated count, currently evaluating count, pending count, and max trained checkpoint.
4. **Lustre Storage Quota Table**: Show space used/quota/limit, files used/quota/limit, and the percentage utilized. Highlight warning alerts if file count is above 85% (>425K files).
