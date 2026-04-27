---
name: aelf:doctor
description: Verify that hook + statusline commands in Claude Code settings.json actually resolve to executables.
allowed-tools:
  - Bash
---
<objective>
Diagnose Claude Code wiring: scan ~/.claude/settings.json (and the
project-scope .claude/settings.json under cwd if present) for hook /
statusline commands whose program token does not resolve. Catches
dangling absolute paths and bare names that aren't on $PATH.
</objective>

<process>
Run: `uv run aelf doctor`
Display the output verbatim. Do not add commentary. Exit code is 1
when at least one broken command is found, so a wrapping CI step can
gate on it.
</process>
