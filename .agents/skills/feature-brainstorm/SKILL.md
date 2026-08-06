---
name: feature-brainstorm
description: Phase 1 of the feature pipeline. Use when the user wants to brainstorm a new feature idea — phrases like "I have an idea for X", "let's brainstorm Y", "I want to add Z but I'm not sure how", or when the user has a rough concept and wants to explore it before committing to a design. This skill interviews the user to sharpen a vague idea into a structured brainstorm doc. It outputs a markdown file that feeds directly into the feature-architect skill (Phase 2). Do NOT use for features that already have a design doc — go straight to feature-architect or feature-develop.
---

# Feature Brainstorm (Phase 1 of 3)

You are acting as a senior engineer helping the user sharpen a rough idea into a structured brief that the `feature-architect` skill can turn into a proper design doc. Your value is in asking the right questions, not in writing code or producing a design — that's Phase 2.

## The 3-Phase Pipeline

This skill is Phase 1 of a 3-phase feature workflow:

1. **feature-brainstorm** (this skill) — Interview the user, explore the codebase, produce a brainstorm doc.
2. **feature-architect** — Read the brainstorm doc, investigate the codebase deeper, produce a design doc with task breakdown.
3. **feature-develop** — Read the design doc, implement the tasks, test, and verify.

Each phase feeds the next via a markdown file. You can call other skills (e.g. `graphify query`) at any point to understand the codebase better.

## Step 1: Understand where the user is

The user arrives in one of three states:

1. **Has nothing yet** — "I want to add something to the project but I'm not sure what." Start broad: ask what problem they're trying to solve, what's painful about the current workflow, what they wish the system could do.
2. **Has a rough idea** — "I want to support full model input" or "I want to accelerate training." They know the direction but not the shape. Ask targeted questions about scope, constraints, and what success looks like.
3. **Has a detailed concept** — "I want a persistent MLIR worker pool that reuses the Context across benchmarks." They've already thought it through. Skip the exploratory questions, confirm your understanding, and go straight to structuring the brief.

Adapt your questioning to where they are. Don't ask "what problem are you trying to solve?" if they just told you.

## Step 2: Investigate the codebase

Before or during the interview, get informed about the parts of the codebase this feature would touch. This makes your questions sharper and prevents you from asking things the code already answers.

**Use graphify first.** This repo has a prebuilt knowledge graph at `graphify-out/graph.json`. Run `graphify query "<question>"` to trace relevant code paths before reading files. Example queries:

- `graphify query "how does the training loop work"` 
- `graphify query "how are MLIR transforms applied and executed"`
- `graphify query "what actions are available in the action space"`
- `graphify query "how does the config singleton work"`

You can also call other skills at any point. The goal is to walk into the interview already knowing the codebase shape.

**MLIR-RL context to keep in mind:**
- The project is an RL auto-scheduler for MLIR loop nests. Python 3.11+, Slurm HPC, Conda env at `~/envs/mlir`.
- Each `rl_autoschedular_vN` package is fully standalone — no cross-package imports.
- `utils/config.Config` is a singleton reading `CONFIG_FILE_PATH` at first import.
- Training runs on CPU-only Bergamo nodes (256 cores, no GPU). GPU nodes available via C2 QOS.
- MLIR compilation is always CPU. The model (Transformer, tiny) can run on GPU but the bottleneck is compilation.
- No pytest suite — verification is `python -m py_compile`.
- See `AGENTS.md` for full project rules.

## Step 3: Interview the user

Ask questions one at a time, building on previous answers. The goal is to surface:

1. **Problem statement** — What's broken, missing, or suboptimal? What does the user want the system to do that it can't do now?
2. **Scope** — What's in and out of scope? Is this a new package, an extension of an existing one, or infrastructure (scripts, configs, docs)?
3. **Constraints** — Performance requirements, backward compatibility, academic reproducibility concerns, hardware limits.
4. **Success criteria** — How will we know it works? What metric improves? What test demonstrates it?
5. **Risks** — What could go wrong? MLIR crashes, entropy collapse, cache format changes, Lustre quota, training time explosion.
6. **Dependencies** — Does this depend on other features? Does it block future work?

**Interview rules:**
- Ask one question at a time. Wait for the answer before asking the next.
- If the user's answer reveals they've already thought about something deeply, don't re-ask it — move on.
- If you can answer a question yourself by querying graphify or reading code, do that instead of asking the user.
- Keep the interview to 5-10 exchanges. If it's going longer, the idea is probably too big and should be split.
- Use `graphify query` during the interview if the user says something that touches code you haven't explored yet.

## Step 4: Write the brainstorm doc

Once the interview converges (the user has answered the key questions and there's no major ambiguity left), write a structured brief. This is NOT a design doc — it doesn't include file-by-file changes, task lists, or implementation details. It's the *input* to the design phase.

Write to: **`docs/design/todo/<feature-slug>-brainstorm.md`**

Use this template:

```markdown
# [Feature Name] — Brainstorm

## Status
Brainstorm complete — ready for feature-architect

## Problem
What's broken, missing, or suboptimal. Why it matters.

## Idea
The rough shape of the solution. 2-4 sentences, not a design.

## Scope
- In scope: ...
- Out of scope: ...

## Constraints
- Package: new `rl_autoschedular_vN` or extension of existing (which one?)
- Hardware: CPU-only (Bergamo), GPU (C2 QOS), or both
- Academic: does this change the reward signal? Does it affect reproducibility?
- Backward compatibility: must existing checkpoints/configs still work?

## Success Criteria
How to verify the feature works. Concrete and measurable.

## Risks
What could go wrong. MLIR crashes, training instability, resource limits.

## Dependencies
What this depends on, what depends on this.

## Open Questions
Anything the architect phase still needs to resolve. Can be empty if the interview covered everything.

## Codebase Pointers
Key files/modules the architect should investigate. Identified during graphify queries or file reads.
- `path/to/file.py` — what it does and why it's relevant
- ...
```

## Step 5: Hand off to Phase 2

After writing the file, tell the user:

> Brainstorm doc written to `docs/design/todo/<feature-slug>-brainstorm.md`.
> 
> Next step: run `feature-architect` to produce the design doc, or say "let's architect it" and I'll proceed to Phase 2.

Do NOT proceed to feature-architect automatically — let the user confirm, since they may want to review or edit the brainstorm doc first.