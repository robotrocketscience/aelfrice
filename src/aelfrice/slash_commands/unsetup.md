---
name: aelf:unsetup
description: Remove the aelfrice UserPromptSubmit hook from Claude Code's settings.json.
allowed-tools:
  - Bash
---
<objective>
Reverse `aelf:setup`. Strip the matching `hooks.UserPromptSubmit`
entry from Claude Code's `settings.json` so prompts are no longer
augmented with aelfrice retrieval. Other hook events and any
unrelated `UserPromptSubmit` entries are preserved.
</objective>

<process>
Run: `uv run aelf unsetup`

Defaults match `aelf setup`: user scope and command
`python -m aelfrice.hook`. Pass the same `--scope` /
`--settings-path` / `--command` flags you used at install time so
the matching entry is found. The command is idempotent: a second
invocation is a no-op and reports `no matching hook`.

Display the output verbatim. Do not add commentary.
</process>
