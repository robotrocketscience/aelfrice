# Transcript ingest

**Status:** spec.
**Target milestone:** v1.2.0 (rides alongside the commit-ingest hook
and triple-extraction port; same machinery, different producer).
**Dependencies:** the v1.2.0 ingest enrichment work — needs
`Belief.session_id`, `Edge.anchor_text`, and the `DERIVED_FROM`
edge type added before this is useful. Otherwise stdlib only.
**Risk:** medium. Hook integration into the Claude Code transcript
runtime; per-turn budget must stay sub-100ms or the hook degrades the
user-facing latency of every prompt and every assistant turn.
**Prior art:** the earlier research line shipped a working version
of this in production, with a per-turn JSONL writer, a `PreCompact`
rotation step, and a batch `ingest_jsonl()` reader. This spec ports
that design forward to the v1.0+ aelfrice schema rather than
copying — the implementation is being re-validated, not lifted.

## Summary

Two ingest paths that turn live conversation transcripts into beliefs
and edges:

1. **Live per-turn path.** A `UserPromptSubmit` hook + a `Stop` hook
   that each append one JSONL line per turn to a project-scoped
   `turns.jsonl`. Each line carries the turn text, role
   (`user`|`assistant`), session id, and timestamp. Append-only,
   sub-10ms, never blocks the model.

2. **Batch ingest path.** A new `aelfrice.ingest.ingest_jsonl()`
   function reads a `turns.jsonl` file, runs the existing
   `ingest_turn` per line, and produces beliefs and edges with
   `session_id` populated and `source='transcript'`. Invoked by:
   (a) the `PreCompact` hook on rotation (see "Compaction
   coupling"); (b) a CLI command `aelf ingest-transcript PATH` for
   manual runs and CI tests; (c) any caller that wants to ingest a
   captured session.

These two paths together give every later v1.x retrieval feature a
real production data source — the actual conversations the user is
having with the agent, captured as they happen, ingested at
compaction boundaries.

```
turn fires (user prompt or assistant stop)
        ↓
  hook script appends one JSONL line to turns.jsonl  (~5ms)
        ↓
  [eventually] PreCompact hook fires
        ↓
  rotate turns.jsonl → archive/turns-<ts>.jsonl
  trigger ingest_jsonl(archive/turns-<ts>.jsonl)
        ↓
  one ingest_turn per line:
     start_ingest_session(session_id=<derived>)
     classify + emit beliefs
     emit edges (DERIVED_FROM linking turns within a session,
                 anchor_text = the literal turn text)
     complete_ingest_session
```

## Motivation

Three independent v1.x asks all bottleneck on this:

1. **Closing the harness-conflict gap** documented at
   [LIMITATIONS.md § harness conflict](../user/LIMITATIONS.md). Today the
   MCP receives no new beliefs from normal session activity; the
   harness's auto-memory directive routes "save a memory" intents
   elsewhere. Transcript ingest is a parallel write path that does
   not depend on the harness directive at all — it captures
   regardless.

2. **Densifying production data for v1.3+ retrieval techniques.**
   Production stores observed in development today have `session_id`
   populated on a small fraction of beliefs and zero `DERIVED_FROM`
   edges. Most v1.3+ retrieval techniques (entity-index, BFS
   multi-hop, posterior-weighted ranking) read from edge structure
   that v1.0 ingest paths do not produce densely. Transcript ingest
   is a producer of both `session_id` and `DERIVED_FROM`, on the
   densest data source aelfrice has access to (every conversation
   turn).

3. **Enabling the v1.4.0 context rebuilder**
   ([context_rebuilder.md](context_rebuilder.md)). The rebuilder's
   premise is that the full conversation log is recoverable from
   aelfrice on demand. That premise requires this spec to ship
   first.

## Where transcripts come from

Claude Code already writes per-session transcripts to
`~/.claude/projects/<cwd-hash>/conversation_*.jsonl`. We do **not**
read those directly because:

- The path is undocumented and subject to change between Claude Code
  releases.
- The format is internal-harness JSON, not aelfrice's ingest schema.
- Per-project `cwd-hash` collides with the same `cwd-hash` orphan
  problem v1.1.0 solves for the memory DB.

Instead, this spec writes its own canonical log via hook
instrumentation. The Claude-Code-internal log stays untouched and
remains the harness's ground truth for replay / audit.

## Design

### Per-turn hook script

A new entry point `aelfrice.transcript_logger:main` registered as
two hook events:

- `UserPromptSubmit` — appends `{"role": "user", ...}`.
- `Stop` — appends `{"role": "assistant", ...}`.

Both hooks read the Claude Code hook JSON payload from stdin, extract
the relevant text field (`prompt` for `UserPromptSubmit`, the most
recent assistant turn for `Stop`), and append a single line to the
project-scoped `turns.jsonl`. Both follow the v1.0 hook
non-blocking contract: any exception writes to stderr and returns
exit 0; the conversation must never stall on logger failure.

Storage location: `<project-root>/.git/aelfrice/transcripts/turns.jsonl`
(matches v1.1.0's in-repo DB location at
`<root>/.git/aelfrice/memory.db`). Living under `.git/` means the
file is structurally uncommittable — git does not track its own
internals, so users do not need a gitignore entry per project. One
file per project; rotated by PreCompact into a sibling
`<root>/.git/aelfrice/transcripts/archive/` directory.

This also matches the data-flow story: `turns.jsonl` feeds the
brain-graph DB (which is itself under `.git/`), the brain graph
serves retrieval at hook time, and nothing on this path is ever
committed. The original conversation prose lives on disk locally and
on the brain graph in extracted form locally, and that's it.

### JSONL line schema

```json
{
  "schema_version": 1,
  "ts": "2026-04-27T15:42:11.337Z",
  "role": "user",
  "text": "What database does this project use?",
  "session_id": "20260427T154010Z-3f8a",
  "turn_id": "20260427T154211Z-7c2e",
  "context": {
    "cwd": "/abs/path/to/project",
    "git_branch": "main",
    "git_head": "abc1234"
  }
}
```

Notes:

- `session_id` is derived once per Claude-Code session start and
  stays stable for the duration. Source: a session-start hook that
  writes a `session_id` env var the per-turn hooks read.
- `turn_id` is unique per line; used by the rebuilder to thread
  user→assistant pairs.
- `context.git_*` is populated cheaply from `git rev-parse` and
  `git symbolic-ref --short HEAD`. Failures (non-git directories)
  set the fields to `null`; never block the hook.
- `schema_version` lets future readers detect format drift without
  breaking existing logs.

### Batch ingest

```python
def ingest_jsonl(
    store: MemoryStore,
    jsonl_path: Path,
    *,
    source_label: str = "transcript",
) -> IngestResult:
    """Read a turns.jsonl file and ingest one belief per turn.

    Each line: {"role": str, "text": str, "session_id": str, ...}.
    Calls ingest_turn(text=..., source=source_label,
    session_id=...) for each. Aggregates the IngestResults.

    Backward-compat with v1.0 stores: if session_id is None or the
    store doesn't carry it (legacy schema), still ingests the
    belief; the session field is silently dropped.

    Edges:
      - Within a session, consecutive turns get a DERIVED_FROM edge
        linking turn N+1's belief to turn N's belief, with
        anchor_text = the prior turn's literal text.
      - The triple extractor (separate v1.2.0 work) optionally adds
        typed edges (CITES, SUPPORTS, CONTRADICTS) discovered in the
        prose.

    Idempotent: re-ingesting the same JSONL line is a no-op (deduped
    on (session_id, turn_id)).
    """
```

The implementation lifts the earlier research line's batch JSONL
ingest design with these v1.x deltas:

- v1.x carries `session_id` end-to-end (the predecessor implementation
  accepted it as an arg-parity placeholder but did not persist it).
- v1.x emits `DERIVED_FROM` edges explicitly; the predecessor left
  edge emission to a separate relationship-detection pass.
- v1.x sets `anchor_text` on every edge; predecessor edges carried
  no anchor text.

### Compaction coupling

`PreCompact` hook fires when Claude Code is about to compact. The
hook script:

1. Writes a `{"event": "compaction_start", "ts": ...}` marker to
   `turns.jsonl`.
2. Renames `turns.jsonl` → `archive/turns-<iso8601-ts>.jsonl`.
3. Spawns `aelf ingest-transcript archive/turns-<iso8601-ts>.jsonl`
   in the background. (Background-spawn is critical: the PreCompact
   path is on the user-facing latency budget.)
4. Returns exit 0 immediately. New turns start writing to a fresh
   `turns.jsonl`.

`PostCompact` hook fires after compaction completes. Writes a
`{"event": "compaction_complete", "ts": ...}` marker to the new
`turns.jsonl` so the next archived segment carries an unambiguous
boundary.

These two markers let the rebuilder (separate spec) reason about
session boundaries without needing the harness's internal compaction
state.

### `aelf setup` integration

`aelf setup --transcript-ingest` adds the four hook entries
(`UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact`) to
`~/.claude/settings.json` or the project-scoped `.claude/settings.json`.
Idempotent per the same contract as the existing v1.0 setup flow.

`aelf unsetup --transcript-ingest` removes them.

The flag is opt-in at v1.2.0. v1.3.0 may flip it on by default once
the disk-footprint and latency telemetry on opt-in users is
acceptable.

## Acceptance criteria

1. `aelf setup --transcript-ingest` writes four hook entries
   idempotently to the appropriate `settings.json`.
2. A round-trip test (manual): in a fresh project, run
   `aelf setup --transcript-ingest`, hold a 5-turn conversation,
   trigger a manual compact, verify `turns.jsonl` rotated to
   `archive/`, verify the archived file ingested into the DB,
   verify `aelf search` against a query about the conversation
   surfaces the ingested beliefs.
3. The per-turn hook stays sub-10ms p99 on a 50-turn conversation.
4. The PreCompact hook returns exit 0 within 50ms regardless of
   ingest progress (background-spawn requirement).
5. `ingest_jsonl` is idempotent: re-running it on the same file
   produces 0 new beliefs and 0 new edges.
6. The schema migration is forward-compatible per the v1.x
   convention: v1.0 stores ingest transcripts (silently dropping
   `session_id`), v1.x stores ingest legacy logs (silently
   dropping unknown fields).

## Test plan

- `tests/test_transcript_logger.py` — unit tests on the per-turn
  JSONL write path, including non-blocking exception handling.
- `tests/test_ingest_jsonl.py` — `ingest_jsonl` correctness, edge
  emission, idempotency, schema-version handling.
- `tests/test_compaction_rotation.py` — PreCompact rotation, marker
  emission, background-spawn budget.
- `tests/integration/test_round_trip.py` — full round-trip with a
  fixture transcript.
- `tests/test_setup_transcript_ingest.py` — `aelf setup
  --transcript-ingest` idempotency, conflict detection with existing
  hook entries.

Per the project test policy, every test is deterministic and
short-running.

## Out of scope

- The rebuilder itself. That's a separate spec
  ([context_rebuilder.md](context_rebuilder.md)).
- Codex transcript ingest. Codex doesn't have hooks the way Claude
  Code does. A v1.x slice for Codex would need a different mechanism
  — likely an `aelf log-turn` MCP tool the Codex side calls
  explicitly. Out of scope for this spec.
- Real-time ingest (per turn into the brain graph, rather than
  per compaction). The earlier research line tested this and found
  it added latency on the user-facing prompt path. Defer until the
  rebuilder eval shows real-time is needed.
- PII scrubbing on ingest. `turns.jsonl` lives under
  `<root>/.git/aelfrice/transcripts/` and is structurally
  uncommittable. Same for the brain-graph DB it feeds. Both stay
  local. No scrub on the live ingest path. If a user ever
  exports a transcript out-of-tree (eval corpus contribution,
  bug report attachment), scrub at the export boundary — that's
  a separate concern from this spec.

## Roadmap

Listed under v1.2.0 in [ROADMAP.md](../concepts/ROADMAP.md) as the
"Transcript-ingest hooks" bullet. v1.4.0 declares this spec as a
hard prerequisite.
