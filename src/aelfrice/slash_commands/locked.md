---
name: aelf:locked
description: List user-locked beliefs.
allowed-tools:
  - Bash
---
<objective>
Inspect the locked-belief tier — every user-asserted ground-truth
statement in this repository's store, then every lock in the user-scope
store, tagged `[user]`.
</objective>

<process>
Run: `uv run aelf locked`
Display the output verbatim. Do not add commentary.
</process>
