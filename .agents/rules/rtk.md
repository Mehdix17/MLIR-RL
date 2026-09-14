---
trigger: always_on
description: Use the RTK token-optimizing CLI proxy (rtk) for shell commands that produce large output, to reduce token usage.
---

## rtk

RTK (Rust Token Killer) is a token-optimizing CLI proxy installed at
`~/.local/bin/rtk` (v0.44.2). It filters and compresses command output
before it reaches the model context (up to ~90% token savings).

Rule:
- For output-heavy commands, invoke them through rtk instead of the raw
  command: `rtk ls`, `rtk git status`, `rtk tree -L 2`, `rtk err <cmd>`,
  `rtk json < file.json`, `rtk test`, `rtk read <file>`.
- `rtk proxy <cmd>` runs the raw unfiltered command (debugging bypass).
- `rtk gain` shows token-savings analytics; `rtk gain --history` shows
  command history; `rtk discover` finds missed opportunities.
- Note: RTK has no Hermes hook (only Claude Code/Cursor/Gemini/Copilot);
  usage here is by convention — prefix heavy commands with `rtk`.
- If rtk output looks wrong/truncated, use `rtk proxy <cmd>` to see raw.
- `read_file` / `search_files` remain the primary file-reading tools; use
  rtk for shell-level output compression (ls, git, tree, test output).
