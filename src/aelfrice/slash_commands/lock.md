---
name: aelf:lock
description: Lock a statement as user-asserted ground truth (L0). Re-locking the same text refreshes the lock.
argument-hint: The statement text to lock as ground truth
allowed-tools:
  - Bash
---
<objective>
Lock a statement as user-asserted ground truth. Locked beliefs auto-load
above keyword search results (L0 layer) and resist demotion until 5+
contradicting feedback events arrive.
</objective>

<process>
Run: `uv run aelf lock "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
