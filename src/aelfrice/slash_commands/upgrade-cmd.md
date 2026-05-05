---
name: aelf:upgrade-cmd
description: Print the right upgrade command for this aelfrice install context.
allowed-tools:
  - Bash
---
<objective>
Tell the user how to upgrade aelfrice. Detects the install method
(uv tool / pipx / venv / system pip) and prints the matching command.
Does NOT execute the upgrade itself: replacing the running package
mid-process is unreliable on Windows and can leave a broken interpreter.
The user runs the printed line.

The advisory `-cmd` suffix in the slash-command name is deliberate.
The previous name `/aelf:upgrade` read as an imperative ("upgrade
now") even though invocation only prints the shell line. New name
signals that this surface emits the *command*; the user runs it.

If an update is available, also surfaces the published wheel SHA-256
and PyPI release URL so the user can hash-pin the install if they want.
</objective>

<process>
Run: `uv run aelf upgrade-cmd`

Display the output verbatim. Do not add commentary.
</process>
