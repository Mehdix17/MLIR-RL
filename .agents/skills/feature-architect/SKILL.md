---
name: feature-architect
description: Use this skill whenever the user wants to plan, design, or architect a new feature before writing any code — phrases like "let's plan out X", "design a feature for Y", "how should we build Z", "I want to add [feature] to the app", or "before we code this, let's think it through". Also trigger when the user hands over a rough idea, a ticket, or a one-line request for new functionality and hasn't already specified an implementation approach. This skill investigates the existing codebase, asks the clarifying questions needed to remove ambiguity, and produces a written design doc that the feature-develop skill (or any coding agent) can implement from. Do NOT use this for small, obvious fixes (typos, one-line bug fixes, config tweaks) — reserve it for work substantial enough to benefit from an upfront design pass.
---

# Feature Architect

You are acting as a senior engineer doing the design pass on a feature *before* anyone writes implementation code. Your output is a design doc that a developer (human or agent) can pick up cold and implement without having to re-derive the decisions you made here. The value of this skill comes entirely from catching ambiguity and bad assumptions now, while they're cheap to fix, instead of after code exists.

Don't start writing code during this skill. If you notice yourself about to edit application source files, stop — that's the job of the `feature-develop` skill, and doing it here defeats the point of separating planning from implementation.

## Step 1: Understand the request

Read what the user asked for. Don't assume you understand the full scope from a one-liner — feature requests are almost always underspecified in ways that matter for implementation (edge cases, who can access what, how it interacts with existing features, performance/scale expectations).

## Step 2: Investigate the codebase

Before asking the user anything, get informed about the part of the codebase this feature touches. You want to walk in already knowing the answer to anything you could reasonably discover yourself, not make the user explain it to you:

- What's the relevant existing code for this area (similar features, adjacent modules, the parts of the codebase this will touch)?
- What conventions does this codebase already follow (naming, file layout, state management, API patterns, error handling, testing style)?
- Are there existing abstractions this feature should reuse rather than duplicate?
- Is there an existing docs/specs folder with prior design docs to match the format of? Look for common locations: `specs/`, `docs/design/`, `docs/features/`, `.claude/specs/`, or similar. If one exists, follow its convention. If none exists, you'll create `specs/<feature-slug>/design.md` (see Step 5).

**Check for existing exploration before doing your own.** On a large or unfamiliar codebase, don't blindly re-walk the whole repo if the groundwork has already been done. If the conversation already contains a codebase map, architecture summary, or module breakdown (e.g. from a codebase-exploration skill the user ran earlier, or from earlier turns in this same conversation), read and rely on that instead of re-deriving it. Only fall back to exploring the repo yourself for the specific area this feature touches when no such context exists — and in that case, keep it targeted to what's relevant rather than mapping the entire codebase, since that's a different, more expensive job than architecting one feature. If the user mentions they typically map the codebase with a separate tool/skill first, treat that as this feature's starting context whenever it's present in the conversation.

This research should shrink the list of questions you need to ask — if you can find the answer by reading the code (or the existing map of it), don't make the user answer it.

## Step 3: Ask clarifying questions

Once you've done your homework, ask the user the questions that actually require their input — product decisions, priorities, scope boundaries, and anything the code genuinely can't tell you. Good candidates:

- Scope: what's explicitly in and out of scope for a first version?
- Edge cases and error states: what should happen when things go wrong?
- Who/what this affects: users, permissions, other features, data migrations?
- Non-functional constraints: performance expectations, backward compatibility, rollout plan?

Don't ask questions you already answered yourself in Step 2, and don't pad the list — a handful of sharp questions beats twenty boilerplate ones. If the request is genuinely simple and your investigation answered everything, it's fine to confirm your understanding and skip straight to proposing a design, but for most real features, some back-and-forth here is what makes the rest of this skill worth using.

## Step 4: Propose the design

Once scope is clear, write up the actual design. Think about it the way you would explain it to another engineer:

- **Approach**: the high-level shape of the solution.
- **Alternatives considered**: if there was a genuine fork in the road (e.g., two reasonable architectures), name the alternative and say briefly why you didn't go with it. Don't manufacture alternatives that aren't real — only include this when there was a real decision.
- **Components/changes**: what files, modules, or systems will be created or touched, and what each one is responsible for.
- **Data model / API changes**: schemas, endpoints, types — anything with a concrete shape should be written out concretely, not described vaguely.
- **Edge cases and error handling**: how the design handles the tricky cases surfaced in Step 3.
- **Task breakdown**: an ordered checklist of implementation steps, sized so each one is a sensible unit of work for a single coding session. This is the part `feature-develop` will follow step-by-step, so make each item concrete and unambiguous — "Add `UserPreferences` model with fields X, Y, Z" rather than "Set up data layer."

Share this with the user before writing it to disk. Treat it as a draft — the user may push back on the approach, and it's much cheaper to revise a proposal in conversation than to rewrite a committed doc.

## Step 5: Write the design doc

Once the user is happy with the design, write it to a markdown file in the repo so it persists and can be handed to `feature-develop` (or any other agent/person) later.

- If the repo already has a design-doc convention (see Step 2), follow it.
- Otherwise, create `specs/<feature-slug>/design.md`, where `<feature-slug>` is a short kebab-case name for the feature (e.g. `specs/user-notification-preferences/design.md`).

Use this structure for the doc:

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

## Data Model / API
Concrete schemas, types, endpoints.

## Edge Cases & Error Handling
The tricky cases and how they're handled.

## Tasks
- [ ] Task 1 — concrete, sized for one sitting
- [ ] Task 2
- [ ] ...
```

Tell the user where the file is, and let them know `feature-develop` (or any coding agent, since this is a plain markdown file) can pick it up from there to start implementation.
