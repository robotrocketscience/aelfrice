---
name: aelf:delete
description: Hard-delete a belief from the store. Writes an audit row before the cascade. Confirmation prompt required unless --yes is passed; locked beliefs require --force.
argument-hint: The belief ID to delete, plus optional --force, --yes, and --user
allowed-tools:
  - Bash
---
<objective>
Permanently remove one belief from the store, including its FTS entry,
all edges (both src and dst), and entity index rows. One audit row is
written to feedback_history (valence=-1.0, source=user_deleted, or user_deleted_force when --force is used) before
the cascade, so the forensic record of "belief X existed and was deleted
at T" survives.

The delete also clears the text of each ingest log row once no belief
that row derived exists; the row keeps only the text's sha256 and the
time it was cleared. A backup or copy of the store made before the
delete keeps the text. Pass `--user` to delete from the user store.

This is a hard-delete. There is no recovery. Use `aelf unlock` if you
only want to remove the user-lock, or `aelf feedback <id> harmful` if
you want the belief to decay gradually rather than disappear outright.

Safety note: `/aelf:delete` does NOT imply `--yes`. The invoking surface
must let the user respond to the confirmation prompt (type the first 8
characters of the belief id) before the delete proceeds. The prompt
names the belief by id, origin, and creation date, never by its text.
The prompt reads one line from stdin; if stdin is at end of file, the
delete aborts. To skip the prompt the user must explicitly pass `--yes`.
</objective>

<process>
Run: `aelf delete $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
