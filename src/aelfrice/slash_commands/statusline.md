---
name: aelf:statusline
description: Emit the orange update-banner statusline snippet for Claude Code.
allowed-tools:
  - Bash
---
<objective>
Print a one-line statusline prefix snippet. When an update to aelfrice
is available, the snippet is an orange-coloured "⬆ aelfrice X.Y.Z
available, run: aelf upgrade │ ". When no update is pending, the
output is empty so the user's existing statusline is unaffected.

This is normally invoked by Claude Code's `statusLine` configuration
(installed automatically by `aelf setup`); the slash command lets the
user invoke it manually for diagnostics.
</objective>

<process>
Run: `uv run aelf statusline`

Display the output verbatim. Do not add commentary.
</process>
