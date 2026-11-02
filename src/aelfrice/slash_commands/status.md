---
name: aelf:status
description: Show the store in use, the rule that chose it, and counts of beliefs, threads, locks, feedback events, and phantoms.
allowed-tools:
  - Bash
---
<objective>
Quick snapshot: how much memory has aelfrice accumulated?
For the structural auditor (orphan threads, FTS5 sync, locked
contradictions) use `aelf:doctor` with the `graph` subcommand.
</objective>

<process>
Run: `uv run aelf status`
Display the output verbatim. Do not add commentary.
</process>
