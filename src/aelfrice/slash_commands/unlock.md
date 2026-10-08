---
name: aelf:unlock
description: Drop the user-lock on a belief without changing its origin. Writes a lock:unlock audit row. Idempotent.
argument-hint: The belief ID to unlock, optionally preceded by --user
allowed-tools:
  - Bash
---
<objective>
Clear the user-lock on a belief. The belief remains in the store with its
origin tier intact — only the lock is removed. Idempotent: calling unlock
on an already-unlocked belief exits 0 with an "already unlocked" message
and writes no audit row.
</objective>

<process>
If `$ARGUMENTS` starts with `--user` followed by a space, the lock is in
the user-scope store. Pass `--user` as its own flag and run
`uv run aelf unlock --user "<the id after --user>"`.
Otherwise run: `uv run aelf unlock "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
