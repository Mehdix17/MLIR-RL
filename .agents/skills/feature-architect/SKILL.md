---
name: feature-architect
description: Phase 2 of the feature pipeline. Use when the user wants to architect or design a feature that has already been brainstormed — phrases like "let's architect it", "design the feature we brainstormed", "create the design doc for X", or when a brainstorm doc exists at `docs/design/todo/<feature>-brainstorm.md`. This skill reads the brainstorm doc, investigates the codebase deeper, asks remaining clarifying questions, and produces a design doc with a task breakdown that `feature-develop` (Phase 3) implements. If no brainstorm doc exists, suggest running `feature-brainstorm` first. Do NOT use for small, obvious fixes — reserve for work that benefits from an upfront design pass.
---

# Feature Architect (Phase 2 of 3)

You are acting as a senior engineer doing the design pass on a feature *before* anyone writes implementation code. Your input is a brainstorm doc from Phase 1 (`feature-brainstorm`). Your output is a design doc with a task breakdown that `feature-develop` (Phase 3) can implement cold, without re-deriving your decisions.

## The 3-Phase Pipeline

This skill is Phase 2:

1. **feature-brainstorm** (Phase 1) — Interviewed the user, produced a brainstorm doc.
2. **feature-architect** (this skill) — Read the brainstorm doc, investigate the codebase, produce a design doc.
3. **feature-develop** (Phase 3) — Read the design doc, implement, test, verify.

You can call other skills (e.g. `graphify query`) at any point to understand the codebase better.

## Step 1: Read the brainstorm doc

If the user pointed you at a specific file, start there. Otherwise look for brainstorm docs in **`docs/design/todo/<feature-slug>-brainstorm.md`**. If no brainstorm doc exists and the request is more than a trivial change, tell the user to run `feature-brainstorm` first — the brainstorm captures problem, scope, constraints, and risks that you need before architecting.

Read the whole brainstorm doc carefully. It contains:
- **Problem** — why the feature matters
- **Idea** — rough shape of the solution
- **Scope** — what's in and out
- **Constraints** — package, hardware, academic, backward compatibility
- **Success criteria** — how to verify
- **Risks** — what could go wrong
- **Codebase pointers** — key files to investigate
- **Open questions** — things the brainstorm didn't resolve

Also read **`AGENTS.md`** at the repo root — it has hard rules (package isolation, config singleton, SIGABRT handler, no file deletion) that override anything else.

## Step 2: Investigate the codebase

The brainstorm doc gives you codebase pointers — start there. Then go deeper.

**Use graphify first.** Run `graphify query "<question>"` against `graphify-out/graph.json` to trace the relevant code paths before reading files. The brainstorm doc's "Codebase Pointers" section tells you which files matter — use graphify to understand how they connect to the rest of the system. You can also call other skills at any point.

**MLIR-RL-specific conventions to check during investigation:**
- **Package isolation**: every `rl_autoschedular_vN` is fully standalone — no cross-package imports. New RL features require either a new standalone package (copy the full tree) or an in-place extension of an existing one. The design doc *must* name the target package and say which approach. See `AGENTS.md` for the rule.
- **Config singleton**: `utils/config.Config` is a singleton reading `CONFIG_FILE_PATH` at first import. New features add config fields to `utils/config.py` and a JSON config under `config/<dataset>/<train|eval>/`. The design doc must list exact config field names and the config file path.
- **Slurm/HPC deployment**: every feature is deployed via Slurm (`scripts/train/train.sh`, `scripts/eval/eval.sh`). The design doc must include `--cpus-per-task`, `--mem`, `--partition`, and `--time` requirements. See `docs/hpc/HPC_HARDWARE.md` for available partitions and `docs/design/todo/v5_training_acceleration.md` for current resource usage and the decided V5 resource matrix.
- **No pytest suite**: verification is `python -m py_compile <file>` and a short smoke test via `sbatch scripts/train/train.sh` with a small config. The task breakdown should use these as per-task verification, not "run tests."

## Step 3: Resolve open questions

The brainstorm doc may have "Open Questions" that need user input. Ask these now — one at a time, concisely. If the brainstorm was thorough and there are no open questions, confirm your understanding and proceed to the design.

Don't ask questions you can answer yourself by reading code or querying graphify. Don't re-ask things the brainstorm already resolved.

## Step 4: Propose the design

Write up the actual design. Think about it the way you would explain it to another engineer:

- **Approach**: the high-level shape of the solution.
- **Alternatives considered**: if there was a genuine fork in the road, name the alternative and say briefly why you didn't go with it. Don't manufacture alternatives.
- **Components/changes**: what files, modules, or systems will be created or touched, and what each one is responsible for.
- **Data model / API changes**: schemas, endpoints, types — anything with a concrete shape should be written out concretely.
- **Edge cases and error handling**: how the design handles the tricky cases from the brainstorm's Risks section.
- **Task breakdown**: an ordered checklist of implementation steps, sized so each one is a sensible unit of work for a single coding session. This is the part `feature-develop` will follow step-by-step, so make each item concrete and unambiguous.

Share this with the user before writing it to disk. Treat it as a draft — the user may push back.

## Step 5: Write the design doc

Once the user is happy with the design, write it to disk.

- **Location**: `docs/design/todo/<feature-slug>.md` (same slug as the brainstorm doc, without the `-brainstorm` suffix).
- If a design doc already exists for this feature (check first), extend it in place.

Use this structure:

```markdown
# [Feature Name] — Design

## Status
Draft | Approved

## Summary
One or two sentences: what this feature does and why.

## Scope
What's in scope. What's explicitly out of scope for this version.

## Approach
The high-level design. Alternatives considered, if relevant.

## Components / Changes
File-by-file or module-by-module breakdown of what will change.
**Name the target package** (`rl_autoschedular_vN`) and whether it's a new standalone package (full copy) or an in-place extension.

## Data Model / API
Concrete schemas, types, endpoints. **List exact config field names** added to `utils/config.py` and the config file path.

## Edge Cases & Error Handling
The tricky cases and how they're handled. Include MLIR SIGABRT behavior, execution timeout, and cache compatibility.

## Resource Requirements
Slurm parameters: `--cpus-per-task`, `--mem`, `--partition`, `--time`. Reference `docs/hpc/HPC_HARDWARE.md` for available partitions.

## Tasks
- [ ] Task 1 — concrete, sized for one sitting. Verify with: `python -m py_compile <file>`
- [ ] Task 2
- [ ] ...
```

## Step 6: Hand off to Phase 3

After writing the file, tell the user:

> Design doc written to `docs/design/todo/<feature-slug>.md`.
>
> Next step: run `feature-develop` to implement, or say "let's build it" and I'll proceed to Phase 3.

Do NOT proceed to feature-develop automatically — let the user confirm, since they may want to review or edit the design doc first.