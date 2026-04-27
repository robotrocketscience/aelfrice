# Commit-ingest hook

**Status:** spec.
**Target milestone:** v1.2.0 (named on the public roadmap as
"commit-ingest `PostToolUse` hook").
**Dependencies:** stdlib only. Consumes the
[triple extractor](triple_extractor.md) and the
[ingest enrichment](ingest_enrichment.md) schema.
**Risk:** medium. Hook integration into the Claude Code transcript
runtime; runtime budget must stay tight or the hook degrades the
user-facing latency of every commit.

## Summary

A Claude Code `PostToolUse` hook that fires after a successful
`git commit` Bash invocation, parses the commit message, runs the
triple extractor on the prose, and inserts the resulting beliefs
and edges under a session derived from the git context. Closes the
v1.0 limitation that the belief graph only grows on explicit
`aelf onboard` / `aelf remember` calls.

```
git commit -m "extends triple-extractor to handle DERIVED_FROM"
        ↓
  PostToolUse hook fires (cmd matched ^git commit\b)
        ↓
  read commit message + git show --stat HEAD
        ↓
  start_ingest_session(model="commit-ingest", project_context=<repo>)
  triples = extract_triples(commit_message)
  ingest_triples(store, triples, session_id=...)
  complete_ingest_session(session_id)
```

## Motivation

Every reasonable v1.x retrieval technique reads from edge structure
(typed edges, anchor text, session groupings). Today's ingest paths
do not produce that structure during normal session activity — the
graph grows only when a user explicitly runs `aelf remember` or
`aelf onboard`. The user shouldn't have to remember to hand-write
relationships; commit messages already encode them in prose. This
hook turns every commit into an ingest event.

This is also the v1.x mechanism through which production data
finally starts populating `Edge.anchor_text`, `Belief.session_id`,
and `DERIVED_FROM` edges — the schema fields the
[ingest enrichment](ingest_enrichment.md) spec adds. Without this
hook (or an equivalent), those fields would stay scarce in
production even after the schema migrations land.

## Design

### Hook registration

A new entry point `aelfrice.hook_commit_ingest:main` registered as
a Claude Code `PostToolUse` hook. The hook configuration matches
on `Bash` tool calls whose command starts with `git commit` and
exited successfully (non-zero exit codes mean no commit happened).

Configuration lives under the user's `~/.claude/settings.json` (the
existing hook-config surface). `aelf setup` writes the entry on
install if the user opts in.

### Hook body

```python
def main() -> None:
    """PostToolUse handler for git commit."""
    transcript = read_transcript_input()
    if not _is_successful_git_commit(transcript):
        return
    commit_message = _extract_commit_message_from_diff(transcript)
    if not commit_message.strip():
        return

    store = _open_default_store()
    session_id = _derive_session_id_from_git_context()
    try:
        store.start_ingest_session(
            session_id=session_id,
            model="commit-ingest",
            project_context=_repo_relative_cwd(),
        )
        triples = extract_triples(commit_message)
        result = ingest_triples(store, triples, session_id=session_id)
        _audit_log(store, session_id, result)
    finally:
        store.complete_ingest_session(session_id)
```

### Session id derivation

The hook derives `session_id` from git context rather than asking
the user or generating a random uuid:

```
session_id = sha256(branch_name + ":" + commit_hash)[:16]
```

Stable across hook invocations on the same commit (idempotent if
the hook fires twice). Distinct per commit. Surface-stable across
machines (the same commit on two clones produces the same id),
which makes future cross-machine session deduplication easier.

The hook is self-contained: start session, extract, insert, complete
session, all in one fire.

### Latency budget

The hook runs synchronously after every commit. User-facing latency
matters — the hook must not add perceptible delay to a `git commit`
followed by a `git status`. Budget: **median ≤ 30 ms, p95 ≤ 100 ms**
on a 200-character commit message.

Tactics:

1. **Lazy import** of `aelfrice.triple_extractor` and
   `aelfrice.store`. Cold-start a Python interpreter is the
   dominant cost on most systems.
2. **Skip work on empty messages.** Merge commits, automated
   commits with empty bodies, etc. exit immediately.
3. **Cap message size.** Truncate commit messages above 4 KB
   before extraction. Long commit messages are rare; the cap
   bounds the worst case.
4. **Skip the diff.** v1.2.0 ships extracting from the commit
   message only. Diff-aware extraction is a v1.3+ candidate once
   we know whether commit-message-only catches enough relations
   in practice.
5. **Read-write to the SQLite store with WAL.** Concurrent reads
   from other aelfrice consumers are not blocked.

A regression test verifies the budget on a representative fixture.

### What gets ingested

- **Beliefs.** Each unique noun phrase resolved by the triple
  extractor's content-hash lookup.
- **Edges.** One per extracted triple, typed by the relation
  template, `anchor_text` populated from the citing commit-message
  substring.
- **Session.** One row per commit; `session_id` derived as above.

What does NOT get ingested:

- The diff itself. Out of scope at v1.2.0.
- File metadata (paths touched, file count, etc.). The v1.0
  filesystem-walk scanner covers that surface.
- Author / timestamp metadata. The store records `created_at` per
  belief; nothing more is needed.

### Failure modes

- **Hook crashes mid-extraction.** The session is left open. The
  store's `complete_ingest_session` is idempotent on re-entry, and
  a follow-up sweep can close orphans (out of scope for v1.2.0).
- **Store is locked by another process.** Hook returns silently;
  no commit is degraded by an aelfrice problem. Logs a single line
  to a per-project log file for post-hoc debugging.
- **Triple extractor produces zero triples.** Normal case for
  one-word commit messages; hook completes the session with no
  inserts and exits.
- **Disk full / permission error.** Same as locked: silent skip,
  one-line log.

The principle: the hook may NEVER cause a `git commit` to feel
broken. Worst case is "memory didn't grow this commit," which is
the same as v1.0 behavior. Brain-graph writes are local-only — the
hook never crosses the git boundary or any network boundary.

### Opt-in surface

`aelf setup --commit-ingest` writes the hook configuration. Default
on fresh install: opt-in (the v1.0.1 hook→retrieval wiring is
opt-in too; consistency).

`aelf setup --no-commit-ingest` removes it. Same shape as the
existing `UserPromptSubmit` hook surface.

## Acceptance criteria

1. The hook fires on a successful `git commit` Bash invocation and
   does NOT fire on a failed one (exit code != 0).
2. Empty / whitespace-only commit messages are no-ops; the hook
   exits without opening the store.
3. A commit message containing one extractable triple produces:
   one new session row, two new beliefs (if both noun phrases are
   new), one new edge with non-empty `anchor_text`, all stamped
   with the derived `session_id`.
4. Re-running the same hook on the same commit (idempotency) does
   not produce new beliefs or duplicate edges.
5. The same commit on two different clones produces the same
   `session_id` (cross-machine session stability).
6. Median latency on a 200-char commit message ≤ 30 ms; p95 ≤ 100 ms
   on commodity hardware. Verified by a regression test in
   `tests/regression/`.
7. Store-locked / disk-full / permission-error conditions cause
   the hook to exit silently with one log line; the `git commit`
   itself is not affected.
8. `aelf setup --commit-ingest` writes the hook config and
   `aelf setup --no-commit-ingest` removes it. Idempotent in both
   directions.

## Test plan

- `tests/test_commit_ingest_hook.py` covering criteria 1–5 + 7–8.
- `tests/regression/test_commit_ingest_latency.py` for criterion 6.
  Uses `time.perf_counter`, runs N=20, asserts median + p95.
- All deterministic. Subprocess git tests use a fresh `:memory:`
  store and a temporary git repo per test.
- Wall-clock budget: < 2 s per test (the regression test is the
  longest at ~1 s for N=20 commits).

## Out of scope

- Diff-based extraction. Commit message only at v1.2.0.
- LLM-augmented extraction. Mechanical only.
- Cross-repo session linking. Each repo's hook produces its own
  session ids.
- Pre-commit ingest. Only PostToolUse on successful commit; pre-
  commit ingest could double-write on amended commits.
- Push-time ingest. Out of scope; commits, not pushes, are the
  natural ingest unit for "what did this work establish."
- Squash / rebase handling. The hook fires once per resulting
  commit. Old session ids are garbage; cleanup is a follow-up.

## What unblocks when this lands

This is the first ingest path that:

- Populates `Edge.anchor_text` densely on real corpora — the
  prereq for the v1.3.0 augmented-BM25F retrieval work.
- Populates `Belief.session_id` densely — the prereq for
  session-coherent supersession work in v1.3.0+.
- Emits `DERIVED_FROM` edges on commits that mention "derived from
  / based on / extends" — the second prereq for that work.

Combined with the v1.0.1 hook→retrieval wiring (which produces
feedback events on every retrieval) and the v1.2.0 transcript-ingest
hooks, v1.2.0 is the milestone where production data finally starts
looking like the data the v1.3+ techniques expect.

## Open questions

- Should `aelf setup --commit-ingest` be the default on install at
  v1.2.0, or stay opt-in like the v1.0.1 hook? Recommendation:
  opt-in at v1.2.0; flip to default-on at v1.3.0 once a
  representative corpus shows the latency budget holds in the wild.
- Diff-based extraction is the natural v1.3.0 extension. Worth
  reserving a `--diff` flag on the hook config now so adding it
  later doesn't require a config rewrite. Recommendation: yes,
  reserve the flag with default `False`.
- Squash / rebase produces stale session ids whose commits no
  longer exist. A garbage-collection pass to remove orphan
  sessions is a v1.x housekeeping item — flagged here, not
  addressed.
