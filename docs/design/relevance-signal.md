# Relevance signal — close-the-loop infrastructure

**Issue:** [#779](https://github.com/robotrocketscience/aelfrice/issues/779).
**Umbrella:** [#480](https://github.com/robotrocketscience/aelfrice/issues/480) (adaptive meta-belief layer).
**Substrate prereq:** [#755](https://github.com/robotrocketscience/aelfrice/issues/755) (meta-belief tables + `update_meta_belief` API).

## Why this layer exists

The umbrella #480 adaptive-meta-belief substrate (#755) ships four
signal classes: `relevance`, `latency`, `bfs_depth`, `bm25_l0_ratio`.
The first two consumers shipped (#756 half-life, #757 BM25F anchor
weight) wired the **non-relevance** signals only — `latency` and
`bm25_l0_ratio` respectively — because the production retrieval path
had no live source of `referenced ∈ {0, 1}` evidence per injected
belief.

#365 had shipped offline calibration metrics (`precision_at_k`,
`roc_auc`, `spearman_rho`), but those are bench-time scoring against
a labeled JSONL fixture. Nothing in the production hook path
recorded *which beliefs were injected* and detected *did the agent
reference them next turn*.

#779 closes that loop.

## Three layers, three files

| Layer | Owner | What it does |
|---|---|---|
| 1. Injection log | `MemoryStore` (`store.py`) + UPS hook (`hook.py`) | One `injection_events` row per (UPS turn × injected belief), audit-friendly. |
| 2. Reference detection | `relevance_detection.py` | Pure-function scoring of belief content against assistant response. Exact-substring v1; n-gram is a future opt-in. |
| 3. Sweeper | `hook.py:_sweep_relevance_signal` | At the start of every UPS hook, score the prior turn's pending events and push `relevance` evidence into each event's active consumers. |

## Data flow

```
UPS turn N (user prompt)
├── hits = retrieve(prompt)
├── _emit_user_prompt_submit_rebuild_log(hits)         # #288 diagnostic JSONL
├── _record_injection_events(hits, source='ups',       # #779 Layer 1 audit row
│       active_consumers=get_active_meta_belief_consumers())
└── render <aelfrice-rebuild> block
       │
       ▼  (claude generates response)
       │
Stop hook → transcript_logger appends to turns.jsonl
       │      {role:'assistant', text:..., session_id:..., ts:...}
       │
       ▼
UPS turn N+1 (next user prompt)
├── apply_sentiment_feedback(...)
├── _sweep_relevance_signal(session_id)                # #779 Layer 3
│     ├── list_pending_injection_events(session_id)    # rows with referenced IS NULL
│     ├── _read_assistant_text_since(session_id, oldest.injected_at)
│     ├── join event.belief_id → belief.content
│     ├── score_references(pairs, response_text)       # Layer 2
│     ├── for each (event_id, referenced):
│     │     for consumer_key in event.active_consumers:
│     │         update_meta_belief(consumer_key, SIGNAL_RELEVANCE,
│     │             evidence=float(referenced), ...)
│     │     update_injection_referenced(event_id, referenced, ...)
└── hits = retrieve(prompt)    # consumers see shifted posteriors
```

## Schema (Layer 1)

```sql
CREATE TABLE injection_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    turn_id          TEXT    NOT NULL,
    belief_id        TEXT    NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    injected_at      TEXT    NOT NULL,            -- ISO-8601 UTC
    source           TEXT    NOT NULL,            -- 'ups' (v1); 'pre_compact' deferred
    active_consumers TEXT    NOT NULL DEFAULT '[]',
                                                  -- canonical-sorted JSON
                                                  -- array of meta-belief keys
    referenced       INTEGER,                     -- NULL / 0 / 1 (tri-state)
    referenced_at    TEXT
);
```

Indexes: `(session_id, turn_id)`, `(belief_id)`, partial
`(session_id, referenced) WHERE referenced IS NULL` for the sweeper's
hot path.

## Determinism contract (#605 / `c06f8d575fad71fb`)

- `normalize_text(s)` is a fixed point: NFC + casefold + whitespace
  collapse. Running it twice yields the same bytes.
- `score_references(pairs, response_text)` is a pure function: same
  inputs → byte-identical output list, in input order.
- The sweeper's wall-clock dependence is bounded to:
  - `update_meta_belief`'s `now_ts` (caller-supplied; the substrate's
    decay math is wall-clock-independent at the function level).
  - `referenced_at` ISO timestamp (audit-only; never re-read).

## Why `active_consumers` is a JSON column

Two design questions ratified in
[#779#issuecomment-4448107904](https://github.com/robotrocketscience/aelfrice/issues/779#issuecomment-4448107904):

- **Q1 — JSON column vs sidecar table.** Every read path is "one
  event, all its consumers." No query asks "find all events where
  consumer X was active." JSON is the cheapest representation; a
  sidecar `injection_event_consumers(event_id, meta_key)` would cost
  one extra row per (event × consumer) with no observable benefit.
- **Q2 — Don't reuse `rebuild_log`.** `rebuild_log` is JSONL on disk
  (`~/.aelfrice/logs/rebuild/<session>.jsonl`), not SQL. Appending
  `referenced` to existing lines is a write-in-place footgun; the
  audit-table-in-SQL precedent (`belief_corroborations`,
  `deferred_feedback_queue`, `meta_belief_signal_posteriors`) is the
  right substrate for queryable lifecycle state.

## Deferred (not in v1)

- **GC.** No retention policy ships in v1. The Beta-Bernoulli
  posteriors decay over time (30d half-life on every consumer), so
  ancient injection_events rows contribute negligible signal. Add
  GC as a sub-issue when the storage-vs-evidence trade-off is
  measured.
- **`source='pre_compact'`.** The PreCompact rebuilder injects via
  the `<aelfrice-rebuild>` block but the "next user turn" reference-
  detection semantics get fuzzy when injection happens *after* a
  user prompt. v1 ships UPS only; the schema's TEXT-not-CHECK
  `source` column accommodates the second source without migration.
- **N-gram overlap detection.** `STRATEGY_NGRAM_OVERLAP` is reserved
  in `relevance_detection.py` but the dispatch raises `ValueError`
  in v1. Add as opt-in via `[retrieval] relevance_detection =
  "ngram"` in a follow-up sub-issue.
- **Embedding / LLM-judge detection.** Out of scope per locked
  PHILOSOPHY (#605, `c06f8d575fad71fb`). Not a sub-issue path.

## What this PR enables for siblings

The half-life consumer (#756) subscribes to `latency` only today;
the sweeper will already push `relevance` evidence into it via
`update_meta_belief(META_HALF_LIFE_KEY, SIGNAL_RELEVANCE, ...)` if
the env flag is on — the substrate silently no-ops because the
consumer doesn't subscribe. Adding `relevance` to a consumer's
subscription is a separate config decision (the install signature
is immutable per `meta_beliefs.install_meta_belief`'s "config rows
are never silently mutated" contract); a future sub-issue can
ship a migration to re-install consumers with both signal
subscriptions.

Same applies to #757 BM25F anchor weight (`bm25_l0_ratio` only
today). And to #758 / #759 / #760 as they ship — `active_consumers`
in `get_active_meta_belief_consumers()` lights them up automatically.
