---
name: feature-develop
description: Use this skill whenever the user wants to implement a feature from an existing design doc — phrases like "let's build the feature we designed", "implement the plan in specs/...", "start coding X" where a design doc already exists, or "continue implementing [feature]". Also trigger if the user references a design doc, spec file, or task checklist and wants code written against it. This skill locates the design doc, works through its task list in order, and keeps the doc's checklist in sync with real progress. Do NOT use this for exploratory or undesigned work — if no design doc exists yet and the request is substantial enough to need one, point the user at the feature-architect skill first instead of improvising a plan on the spot.
---

# Feature Develop

You are implementing a feature against a design doc that already exists — produced by `feature-architect` or written by hand. The doc is the contract: it represents decisions the user already signed off on, so your job here is disciplined execution, not re-litigating the design. If you catch yourself second-guessing the plan or wanting to solve a different problem than the one it describes, that's a signal to stop and flag it, not to quietly go off-script (see "When reality doesn't match the plan" below).

## Step 1: Find the design doc

If the user pointed you at a specific file, start there. Otherwise look in the conventional locations: `specs/<feature-slug>/design.md`, `docs/design/`, `docs/features/`, or ask the user which doc they mean. If you can't find one and the request is more than a trivial change, tell the user no design doc exists and suggest running `feature-architect` first rather than inventing a plan yourself — the whole point of this skill is executing a plan someone already thought through, not thinking it through yourself under a different name.

Read the whole doc before touching any code, not just the task list — the Approach, Data Model, and Edge Cases sections tell you *why* the tasks are shaped the way they are, which you'll need when a task is underspecified in the moment.

## Step 2: Confirm where things stand

If the task checklist already has some items checked off, treat those as done — verify quickly by looking at the code rather than assuming, since docs can drift from reality, but don't redo completed work. Figure out which task is next and tell the user briefly what you're about to work on before starting, especially at the beginning of a session.

## Step 3: Work through the tasks in order

Take the tasks one at a time, in the order they're listed — the ordering usually encodes real dependencies (e.g. the data model needs to exist before the API that uses it). For each task:

1. Implement it following the codebase's existing conventions (the same patterns, libraries, and style already in use nearby — don't introduce a new pattern the design doc didn't call for).
2. Verify it works — run tests, a build, a linter, or whatever the repo already uses for that. If the repo has no test setup for the relevant area, at least sanity-check the change manually before moving on.
3. Check the box for that task in the design doc (`- [ ]` → `- [x]`) so the doc stays an accurate record of progress. This matters if implementation spans multiple sessions or hands off to someone else.
4. Move to the next task.

Do this for the whole list in one pass if the scope allows it, rather than stopping after every single task to ask "should I continue?" — the user already approved the plan, so grinding through it is the expected behavior. Use your judgment on natural checkpoints (e.g., pausing after a schema migration that's worth a sanity check before building on top of it).

## Step 4: When reality doesn't match the plan

Plans are written before anyone has touched the code, so it's normal for something to not quite fit once you're actually implementing — a function that doesn't exist where the doc assumed it would, a task that turns out to be two tasks, a dependency the design missed. When this happens:

- Don't silently improvise a different approach and keep going as if nothing changed — the user is trusting this skill to follow what they approved, and a silent deviation defeats that.
- Do stop and tell the user what you found, why it doesn't match the doc, and what you'd suggest instead. Small, obvious clarifications (a variable name, a missing import) don't need a full stop — use judgment on what's actually a deviation from the *design* versus a normal implementation detail.
- If the user confirms a change, implement it, then update the design doc itself to reflect the new reality (edit the relevant section, not just the checklist) so it stays a trustworthy record for next time.

## Step 5: Wrap up

Once all tasks are checked off, do a final pass: re-read the design doc's Edge Cases and Data Model sections and confirm the implementation actually covers them, not just the literal task list. Summarize for the user what was built, note anything you flagged along the way, and mention the doc's Status field — update it from `Draft` to `Implemented` (or whatever the doc's convention is) if everything landed.
