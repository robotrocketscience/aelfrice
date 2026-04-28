---
name: aelf:ingest-transcript
description: Ingest a turns.jsonl file produced by the transcript-logger into the active project's brain-graph DB.
allowed-tools:
  - Bash
---
<objective>
Read a turns.jsonl file (typically produced by the v1.2.0 transcript-
logger PreCompact hook on rotation, archived under
.git/aelfrice/transcripts/archive/) and ingest each conversation turn
into the active project's brain-graph DB. Within a session, consecutive
turns are linked by DERIVED_FROM edges so the v1.4.0 context rebuilder
can reconstruct conversation structure.

Idempotent: re-running on the same file produces zero new beliefs and
zero new edges. Compaction markers and malformed lines are counted
under skipped_lines and ignored without raising.
</objective>

<process>
Run: `uv run aelf ingest-transcript $ARGUMENTS`
Display the lines/turns/beliefs/edges/skipped summary verbatim.
</process>
