---
name: commit
description: Automated git workflow for staging, formatting conventional commits, branch safety validation, pushing, and PR suggestions. Use this when the user asks to commit, push, or manage repository changes.
---

# Git Workflow Automation (Staging, Committing, Pushing, & PRs)

When this skill is invoked, follow this step-by-step workflow to safely and conventionally manage code changes.

---

## 🛠️ Step 1. Branch Safety Check
1. Query the active branch:
   ```bash
   git branch --show-current
   ```
2. **Safety Rule:** If the current branch is `main` or `master`, **DO NOT COMMIT**. Print a warning warning the user and advising them to create or switch to a development or feature branch.

---

## 🛠️ Step 2. Analyze & Stage Changes
1. Run `git status --short` to check for unstaged modified/deleted files and untracked files.
2. Stage the modified files:
   ```bash
   git add <files>
   ```
3. If there are untracked files, list them clearly and ask the user if they should be included before staging.

---

## 🛠️ Step 3. Generate Conventional Commit Message
1. Inspect the staged diff to understand the changes:
   ```bash
   git diff --cached --name-status
   ```
2. Draft a commit message following the **Conventional Commits** specification:
   `<type>[optional scope]: <description>`
   
   **Allowed Types:**
   - `feat`: A new feature / script
   - `fix`: A bug fix or timing fallback
   - `docs`: Documentation updates
   - `style`: Whitespace, formatting, etc.
   - `refactor`: Restructuring code without changing behavior
   - `perf`: Performance improvements
   - `test`: Adding/modifying tests
   - `chore`: Maintenance, updates, package configs

3. Ensure the description uses the imperative mood (e.g., "implement fast report" instead of "implemented fast report").

---

## 🛠️ Step 4. Confirm Branch & Push
1. Present a summary of the changes and the proposed conventional commit message to the user.
2. Ask the user:
   - "Which branch would you like to commit/push to?" (Default: current branch).
   - Confirm before committing.
3. Once confirmed, execute the commit:
   ```bash
   git commit -m "<conventional-message>"
   ```

---

## 🛠️ Step 5. Push to Remote & Suggest PR
1. Prompt the user: "Should I push this commit to remote branch `<branch>`?".
2. If confirmed, push the commit:
   ```bash
   git push origin <branch>
   ```
3. If pushed successfully to a non-main branch (e.g. `dev`), ask the user if they would like to create a Pull Request to merge into `main`, and suggest a PR title and description based on the commit.
