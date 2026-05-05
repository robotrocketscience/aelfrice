---
name: aelf:delete
description: Hard-delete a belief from the store. Writes an audit row before the cascade. Confirmation prompt required unless --yes is passed; locked beliefs require --force.
argument-hint: The belief ID to delete
allowed-tools:
  - Bash
---
<objective>
Permanently remove one belief from the store, including its FTS entry,
all edges (both src and dst), and entity index rows. One audit row is
written to feedback_history (valence=-1.0, source=user_deleted) before
the cascade, so the forensic record of "belief X existed and was deleted
at T" survives.

This is a hard-delete. There is no recovery. Use `aelf unlock` if you
only want to remove the user-lock, or `aelf feedback <id> harmful` if
you want the belief to decay gradually rather than disappear outright.

Safety note: `/aelf:delete` does NOT imply `--yes`. The invoking surface
must let the user respond to the confirmation prompt (type the first 8
characters of the belief id) before the delete proceeds. To skip the
prompt the user must explicitly pass `--yes`.
</objective>

<process>
Run: `uv run aelf delete "$ARGUMENTS"`
Display the output verbatim. Do not add commentary.
</process>
