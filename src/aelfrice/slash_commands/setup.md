---
name: aelf:setup
description: Install the aelfrice UserPromptSubmit hook in Claude Code's settings.json.
allowed-tools:
  - Bash
---
<objective>
Wire aelfrice's retrieval into Claude Code so that every user prompt is
augmented with the most relevant locked beliefs and FTS5 hits from the
local memory store. This adds an entry under
`hooks.UserPromptSubmit` in Claude Code's `settings.json`.
</objective>

<process>
Run: `uv run aelf setup`

By default the hook is installed user-wide
(`~/.claude/settings.json`) and the recorded command is
`aelf-hook` (the script entry-point exposed by aelfrice's
pyproject). Pass `--scope project` for a project-scoped install, or
`--settings-path PATH` to write to an explicit location. The
command is idempotent: running it twice results in exactly one
matching hook entry.

Display the output verbatim. Do not add commentary.
</process>
