---
name: aelf:lock
description: Lock a statement as user-asserted ground truth (L0). Re-locking the same text refreshes the lock.
argument-hint: The statement text to lock as ground truth
allowed-tools:
  - Bash
---
<objective>
Lock a statement as user-asserted ground truth. Locked beliefs auto-load
above keyword search results (L0 layer) and resist passive feedback by
design — change a wrong lock via `aelf unlock` / `aelf delete`, or
re-lock the corrected statement (per #605, #814).
</objective>

<process>
Run: `uv run aelf lock "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
