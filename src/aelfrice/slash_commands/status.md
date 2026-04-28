---
name: aelf:status
description: Show summary counts — beliefs, threads, locks, and feedback events.
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
