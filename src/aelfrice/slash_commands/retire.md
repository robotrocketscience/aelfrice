---
name: aelf:retire
description: Soft-delete (retire) a belief — the reversible sibling of delete. Drops it out of retrieval and search while preserving its evidence trail; undo with aelf restore. Locked beliefs require --force.
argument-hint: The belief ID to retire
allowed-tools:
  - Bash
---
<objective>
Retire one belief without destroying it. `retire` sets the belief's
`valid_to` timestamp so it stops matching keyword search and drops out of
retrieval, but its edges, entity index, and corroboration rows are left
intact. This is the gentle counterpart to `aelf delete` (which hard-deletes
and cascades): a retired belief can be brought back at any time with
`aelf restore <id>`.

One audit row is written to feedback_history (valence=-1.0,
source=user_retired, or user_retired_force when --force is used) before the
soft-delete, so the forensic record survives.

Locked beliefs (lock_level=user) are protected: pass `--force` to retire one.
Retiring an already-retired belief is a no-op.
</objective>

<process>
Run: `uv run aelf retire $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
