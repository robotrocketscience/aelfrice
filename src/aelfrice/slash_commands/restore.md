---
name: aelf:restore
description: Restore a retired (soft-deleted) belief to active — the inverse of aelf retire. Clears valid_to and re-indexes the belief for search. No-op on an already-active or unknown id.
argument-hint: The belief ID to restore
allowed-tools:
  - Bash
---
<objective>
Bring a retired (soft-deleted) belief back to active. `restore` clears the
belief's `valid_to` timestamp and re-inserts its FTS index row, so it
re-enters keyword search and retrieval. This is the inverse of
`aelf retire`.

One audit row is written to feedback_history (valence=+1.0,
source=user_restored) on a successful restore. Restoring a belief that is
already active, or an id that does not exist, is a no-op that reports
"not restorable" and exits non-zero.
</objective>

<process>
Run: `uv run aelf restore $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
