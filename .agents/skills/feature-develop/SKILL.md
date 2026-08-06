---
name: feature-develop
description: Phase 3 of the feature pipeline. Use when the user wants to implement a feature from an existing design doc — phrases like "let's build it", "let's build the feature we designed", "implement the plan in docs/design/...", "start coding X" where a design doc already exists, or "continue implementing [feature]". This skill locates the design doc (produced by feature-architect), works through its task list in order, and keeps the doc's checklist in sync with real progress. Do NOT use for exploratory or undesigned work — if no design doc exists yet, point the user at feature-architect (or feature-brainstorm if no brainstorm exists either).
---

# Feature Develop (Phase 3 of 3)

You are implementing a feature against a design doc that already exists — produced by `feature-architect` (Phase 2) from a brainstorm doc (Phase 1). The doc is the contract: it represents decisions the user already signed off on, so your job here is disciplined execution, not re-litigating the design.

## The 3-Phase Pipeline

This skill is Phase 3:

1. **feature-brainstorm** (Phase 1) — Interviewed the user, produced a brainstorm doc.
2. **feature-architect** (Phase 2) — Read the brainstorm, investigated the codebase, produced a design doc.
3. **feature-develop** (this skill) — Read the design doc, implement, test, verify.

You can call other skills (e.g. `graphify query`) at any point to understand the codebase better.

## Step 1: Find the design doc

If the user pointed you at a specific file, start there. Otherwise look in **`docs/design/todo/<feature-slug>.md`**. If you can't find one and the request is more than a trivial change, tell the user no design doc exists and suggest running `feature-architect` first (or `feature-brainstorm` if no brainstorm exists either).

Read the whole doc before touching any code, not just the task list — the Approach, Data Model, and Edge Cases sections tell you *why* the tasks are shaped the way they are. Also read **`AGENTS.md`** at the repo root — it has hard rules (package isolation, config singleton, SIGABRT handler, no file deletion) that override anything the design doc says.

**Use graphify to understand context.** Run `graphify query "<question>"` against `graphify-out/graph.json` if you need to understand how a file or module connects to the rest of the system before making changes. You can also call other skills at any point.

## Step 2: Confirm where things stand

If the task checklist already has some items checked off, treat those as done — verify quickly by looking at the code rather than assuming, since docs can drift from reality. Figure out which task is next and tell the user briefly what you're about to work on before starting.

## Step 3: Work through the tasks in order

Take the tasks one at a time, in the order they're listed — the ordering encodes real dependencies. For each task:

1. **Check package isolation.** If the design doc names a target package (`rl_autoschedular_vN`), confirm whether it's a new package (requires a full standalone copy — never import across packages) or an in-place extension. See `AGENTS.md`.
2. Implement it following the codebase's existing conventions (the same patterns, libraries, and style already in use nearby — don't introduce a new pattern the design doc didn't call for).
3. **Verify with `python -m py_compile <file>`** — this repo has no pytest suite. `py_compile` is the per-file syntax gate. For changes to transforms or execution, additionally run a short smoke test: `FORCE_NEW=1 sbatch scripts/train/train.sh config/.../<small>.json` and check the first few iterations complete without SIGABRT. Also test `--resume` compatibility: `sbatch scripts/train/train.sh config/.../<small>.json --resume results/.../run_0`.
4. **Preserve the execution cache format.** Any change to `execution.py` or `get_code_cache_key` must not break existing `exec_data.json` files — training resume depends on it.
5. Check the box for that task in the design doc (`- [ ]` → `- [x]`) so the doc stays an accurate record of progress.
6. Move to the next task.

Do this for the whole list in one pass if the scope allows it. Use your judgment on natural checkpoints (e.g., pausing after a change that's worth a sanity check before building on top of it).

## Step 4: When reality doesn't match the plan

Plans are written before anyone has touched the code, so it's normal for something to not quite fit once you're actually implementing. When this happens:

- Don't silently improvise a different approach — the user is trusting this skill to follow what they approved.
- Do stop and tell the user what you found, why it doesn't match the doc, and what you'd suggest instead. Small clarifications (a variable name, a missing import) don't need a full stop.
- If the user confirms a change, implement it, then update the design doc to reflect the new reality so it stays trustworthy.

## Step 5: Wrap up

Once all tasks are checked off, do a final pass: re-read the design doc's Edge Cases and Data Model sections and confirm the implementation actually covers them, not just the literal task list. Summarize for the user what was built, note anything flagged along the way, and update the doc's Status field from `Draft` to `Implemented`.

**Before declaring done, also verify:**
- `python -m py_compile` passes on every changed file.
- If the feature touches training or execution: a short `sbatch scripts/train/train.sh` smoke test completed without SIGABRT or crash.
- `--resume` works (load the smoke-test checkpoint and resume).
- **Lustre quota**: run `lfs quota -u $USER /scratch` if the feature generates many files (checkpoints, eval outputs). Notify the user if near the 500K soft limit.

**After wrapping up:** The design doc lives in `docs/design/todo/`. Move it to `docs/design/done/` to mark the feature as complete:
```bash
mv docs/design/todo/<feature-slug>.md docs/design/done/
```
Also move the brainstorm doc if it's still in `todo/`.