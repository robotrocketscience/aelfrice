# v2.x spec: derivation worker — beliefs become materialized state

Spec for issue [#264](https://github.com/robotrocketscience/aelfrice/issues/264). Cascade addendum to [`design/write-log-as-truth.md`](design/write-log-as-truth.md). The actual refactor; #261/#262/#263 are scaffolding for this.

Status: spec, no implementation. Recommendation included; decision is the user's.

## What's being decided

Four contract calls left open by the issue body:

1. **Order of operations** in the new ingest path.
2. **Idempotency check** on re-runs.
3. **Crash-recovery semantics** when the worker dies mid-batch.
4. **Edge derivation timing** — same pass as beliefs or a follow-up pass.

After this issue ships, every ingest entry point (`scan_repo`, `ingest_turn`, `triple_extractor.ingest_triples`, MCP `tool_lock`, CLI lock, `accept_classifications`) writes to `ingest_log` only and invokes the worker. Direct `Belief()` construction outside the worker is forbidden by convention; the next issue (#265) makes it forbidden by assertion.

## Recommendation

**Ship at v2.x with synchronous in-process invocation, single-pass derivation (beliefs + deterministic edges together), and recover-by-replay crash semantics.**

### Order of operations

```
ingest entry point:
  1. record_ingest(log_row)         -> ingest_log INSERT (derived_*_ids = [])
  2. derived = derive(log_row)      -> pure function, no side effects
  3. INSERT/UPDATE beliefs from derived.beliefs
  4. INSERT/UPDATE edges from derived.deterministic_edges
  5. UPDATE ingest_log SET derived_belief_ids = ..., derived_edge_ids = ...
       WHERE id = log_row.id
  All in a single SQLite transaction.
```

Single transaction is the right call: SQLite's WAL makes this cheap, and it eliminates the "log row exists but beliefs don't" failure window. The crash-recovery story below is for the rarer case where the *transaction itself* succeeds but the worker dies before the next batch.

### Idempotency check

The composite check the issue body proposes is correct: a log row is **already derived** iff `derived_belief_ids` is non-empty AND every id in that list exists in `beliefs`. Worker MUST re-derive (and re-stamp) if either half fails — that is what catches the "belief was deleted post-ingest" case described in #262 as `derived_orphan`.

```python
def needs_derivation(log_row, store):
    if not log_row.derived_belief_ids:
        return True
    return not all(store.belief_exists(bid) for bid in log_row.derived_belief_ids)
```

### Crash recovery

If the worker dies between the SQLite transaction commit and returning control to the entry point: nothing to recover, the database state is consistent. If the worker dies *between batches* (e.g. an entry point ingests 100 rows in a loop and dies after row 50): on next worker invocation, sweep all rows where `needs_derivation(row) == True` and derive them. This is `aelf doctor --derive-pending` as a manual escape hatch and a startup hook for the next entry point.

The pathological case the issue body raises ("worker dies between INSERT belief and UPDATE log row") is impossible under single-transaction operation. We adopt the constraint to make it impossible rather than recovering from it.

### Edge derivation timing

Same pass as beliefs. Deterministic edges (from `triple_extractor`) are a function of the same raw_text the belief is derived from; computing them in the same `derive()` call costs nothing extra and removes a class of "edges lag beliefs" bugs. Feedback-driven edges (`propagate_valence`) are NOT computed by the worker — they are written by the feedback path and live in a separate cohort per the equality contract in #262.

### Concurrency

Two-process race producing the same belief: `INSERT OR IGNORE` on `(content_hash)` lets one win; both ingest_log rows stamp `derived_belief_ids = [winner_bid]`. The existing `tests/test_concurrency.py` shape covers this; the new test extends it to assert both log rows converge to the same belief id.

## Decision asks

- [ ] **Confirm single-transaction operation.** The alternative is multi-transaction with a recovery sweep on crash. Single-transaction is simpler and SQLite-WAL-cheap; the only reason to reject is if a future async/daemon worker needs to be designed-in now (recommendation: defer to v3).
- [ ] **Confirm "worker dies between batches" recovery semantics.** `aelf doctor --derive-pending` as both a CLI escape hatch and an automatic startup hook on next entry-point invocation. Default: automatic; `--no-auto-derive-on-startup` opt-out for diagnosing stuck states.
- [ ] **Confirm edges in the same pass.** Alternative is a separate edge-derivation pass for parallelism / locality. Single-pass is simpler; defer the split until benchmarks force it.
- [ ] **Worker-only insert enforcement timing.** This issue makes direct `insert_belief()` *unused* outside the worker. #265 makes it *forbidden* (raises). Confirm the split — keeping the assertion gated on #265's feature flag means this issue stays additive and bisectable.

## Why this is judgment-scope

The four decisions above are the design work. The body the issue lists at ~400 LOC code + ~300 LOC tests is achievable only if these are settled before implementation; otherwise the worker grows a recovery-mode flag matrix and the test suite balloons.

## Downstream impact

- Every ingest entry point's call shape changes. Hooks, MCP tools, CLI commands, scanner — all must route through the worker. The patch is wide but mechanical.
- `tests/test_concurrency.py` extends with worker-race coverage.
- `LIMITATIONS.md`: drops the "re-classification requires re-onboard" caveat (the worker handles the re-derivation path; #265 surfaces it as `aelf rebuild`).
- `aelf doctor`: gains `--derive-pending` to manually trigger the sweep.

## Out of scope (deferred)

- **Async / daemon-mode worker.** v3.
- **Multi-rule-set re-derivation.** Held for #265's `aelf rebuild --rule-set <hash>` surface.
- **Performance work beyond the latency alarm in #205.** If the worker becomes a hotspot under realistic ingest sizes, that is a follow-up.

## Provenance

- Source-of-truth: [`docs/design/write-log-as-truth.md`](design/write-log-as-truth.md) §§ "What changes under the proposed contract", "Costs and risks".
- Upstream chain: #205 → #261 → #262 → **#264 this issue** → #265.
- Equality contract that the worker must satisfy: [`v2_replay.md`](v2_replay.md).
