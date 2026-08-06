# Design Memo: Write Log as Source of Truth

**Status:** historical design memo (pre-v2.0), with one current section — the edge contract ratified 2026-08-01 under [#1283](https://github.com/robotrocketscience/aelfrice/issues/1283) is live, not historical, and is recorded in the implementation-status note below. The `ingest_log` table shipped at v1.5 (#205); replay/validation (#262), derivation-worker call-site migration (#264), and the view-flip (#265, gated default-off behind `AELFRICE_WRITE_LOG_AUTHORITATIVE`) have all since landed.

---

## The contract

The append-only log of writes is the canonical state. The queryable structures — the FTS5 index, the typed graph, the alpha/beta posteriors — are *materialized views* over the log. To change a derived view (new extraction rule, new edge inference, schema migration), replay the log against the new derivation function. The log is immutable record; everything else is derived.

> That paragraph is the **ratified contract**, stated in the present tense throughout this memo because it is what the design commits to. It is not a description of the shipped system: *the typed graph* is the one queryable structure not covered by the log today, so a replay rebuilds the index and the posteriors but not the edges. See *Implementation status* under **The contract** below ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)).

This is the standard storage-engine pattern (Postgres WAL, Kafka log, Datomic facts). Stating it explicitly as aelfrice's contract has practical consequences: rule-set evolution stops requiring re-onboard, historical state becomes reproducible by construction, and the federation story (cross-project replication in v3) reduces to log shipping.

---

## Where aelfrice stands today

aelfrice runs on SQLite with `journal_mode=WAL`. This is **the engine's** WAL — a durability mechanism that gets checkpointed back into data pages. It is not an application-level write log. It cannot be replayed against new derivation logic.

The application-level picture, per `src/aelfrice/store.py`:

| Table | Append-only? | Replay-capable? |
|---|---|---|
| `beliefs` | No (mutated on feedback / decay) | No — current values only |
| `edges` | Mostly insert; weight can update | No — current values only |
| `beliefs_fts` | Derived (FTS5 virtual) | Yes — rebuildable from `beliefs.content` |
| `feedback_history` | **Yes** (INSERT-only) | Yes — posterior math reproducible |
| `onboard_sessions` | Insert + state UPDATE | Partially — keeps parsed output, not raw input |

Feedback math is replay-capable today: the conjugate Beta-Bernoulli update is closed-form, and `feedback_history` records every event with its source. Drop `alpha` / `beta` from `beliefs`, recompute from history, get bit-identical numbers back.

What is **not** replay-capable: the set of beliefs themselves. Once `scan_repo` parses a file and `classify_sentence` assigns a type, the resulting `Belief` row is the only record. The raw input is preserved in `belief.content`, but the source path, line number, git commit at ingest time, and classifier inputs are not separately logged. Change the extraction rule (new doc-format support, new AST visitor, future LLM-classifier prompt change) and you must re-onboard from scratch — losing any beliefs that came from sources that no longer exist or have changed.

---

## What changes under the proposed contract

A new `ingest_log` table captures, append-only, every raw input that produced a belief or edge:

```
ingest_log
├─ id                ULID PK            (monotone, sortable)
├─ ts                TIMESTAMP NOT NULL
├─ source_kind       TEXT NOT NULL       (filesystem | git | python_ast |
│                                         mcp_remember | cli_remember |
│                                         feedback_loop_synthesis)
├─ source_path       TEXT                (file path / commit SHA /
│                                         MCP session id)
├─ raw_text          TEXT NOT NULL       (exact bytes presented to the
│                                         classifier)
├─ raw_meta          JSON                (line number, AST node type,
│                                         commit author, etc.)
├─ derived_belief_ids JSON               (post-classification)
├─ derived_edge_ids  JSON
├─ classifier_version TEXT               (semver of classify_sentence at
│                                         ingest time)
└─ rule_set_hash     TEXT                (sha256 of regex pattern set +
                                          LLM-classifier prompt template,
                                          if applicable)
```

The contract:

1. Every belief and every edge has at least one `ingest_log` row pointing at its origin. Beliefs from later synthesis (e.g., feedback-driven re-classification) get a row of `source_kind=feedback_loop_synthesis`.
2. `beliefs` and `edges` become materialized views over `ingest_log` under the current rule set + the feedback log. Their values are computable, not authoritative.

> **Implementation status (2026-08-01, [#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)).** Points 1 and 2 hold for `beliefs` and, since [#1354](https://github.com/robotrocketscience/aelfrice/issues/1354), for a small slice of `edges`. Before it, all six `derive()` return paths emitted `edges=[]`, so `derived_edge_ids` was NULL on every `ingest_log` row and no edge had a log row pointing at its origin. `derive()` now emits the intra-turn `DERIVED_FROM` edges from the row's own `raw_meta` and the column populates forward-only (NULL still means the row predates that writer, and is exempt from the replay comparison) — at most 1.93% of the live edge set. The rest is still written outside the log by `ingest.py` (the inter-turn `DERIVED_FROM` writer, whose `src` is not knowable at `record_ingest` time), `temporal_spine.py` and the relationship / contradiction detectors. The contract above is therefore the **ratified target** for edges, not a description of the shipped system — the operator ruling on #1283 keeps it (edges are log-derived) and fixes the recompute key as `(created_at, ingest_log ULID)`, chosen because it reproduces 93.7% of the live `TEMPORAL_NEXT` edge set against 7.4% for the belief-table key it was measured against (`(created_at, belief_id)`). The ordering the writer actually uses is `(created_at, rowid)`, and that `rowid` is implicit — `beliefs` is declared `id TEXT PRIMARY KEY`, so VACUUM may renumber it and it survives no rebuild. That is why the ratified key is the log's ULID rather than anything read off the belief table. A **read-only** recompute has since shipped as the hidden `aelf spine verify` ([#1336](https://github.com/robotrocketscience/aelfrice/issues/1336)): it covers `TEMPORAL_NEXT` only, reproduces ~93.7% of the shipped spine against a ~95.0% ceiling under the current writer, and **reports** that gap rather than closing it — a gap meter, not a rebuild. That 93.68% was measured against a denominator that included the fan-in surplus, on a snapshot of 41,929 shipped edges; the same store now carries 41,984 and measures 93.69% under that old denominator, so the before/after pair below is that 93.69% rather than the #1336 figure. [#1356](https://github.com/robotrocketscience/aelfrice/issues/1356) has since corrected the denominator: the 546 fan-in > 1 successors leave the denominator entirely — both edges, not just the missed one — and the successor figure is **94.86%** (38,789 of 40,892 fan-in-1 eligible), re-derivable via `benchmarks/spine_fan_in_baseline.py` against the committed baseline in `benchmarks/spine_fan_in_baseline.json`. **The two are not comparable**: it is a denominator correction, not a movement in fidelity, and its direction is data-dependent (up here because the excluded edges were reproducing at 50% against an overall 93.7%; the same correction moves the module's CLI fixture 40% → 33.33%). The same issue added the recomputed-only direction of the diff — 2,102 links the recompute produces that the shipped spine lacks — which the meter previously could not see at all. The 2,103 misses left inside the eligible set are the no-log (2,100) and other (3) buckets, not fan-in. Closing them needs the writer to order on the same durable key, which is not funded. **The 98.70% structural ceiling quoted elsewhere is on the uncorrected denominator** — it was defined by the very fan-in misses this correction removes, so it is not the ceiling for the 94.86% figure and the two must not be subtracted. Points 3 and 4 below therefore remain edge-incomplete: a rebuild from log restores beliefs but not the graph.
3. **Re-onboarding is a no-op** when `(source_path, raw_text)` pairs match. New extraction rules become a "rebuild from log against version N rule set" operation — your feedback history is preserved.
4. **Historical reproducibility falls out.** *"What would the agent have retrieved on this query last March, before the user gave that correction?"* is answered by selecting `ingest_log` rows up to that timestamp, applying the rule set in effect at that time (`classifier_version` + `rule_set_hash`), and running retrieval against the resulting derived state.
5. The classifier-version provenance makes the future LLM-classifier path defensible in the determinism frame. The write of the *derived belief* is bounded-non-deterministic across classifier versions; the *ingest log* is deterministic; the derived belief is a function of `(log row, classifier version, rule-set hash)`. The boundary is visible.

---

## Costs and risks

- **Storage.** Doubling the storage footprint on a 10k-belief project is fine; on a 1M-belief project the metadata is a real number. The raw text overlaps heavily with `belief.content` — dedupe via content hash; net cost is the metadata overhead.
- **Migration burden.** Existing v1.x stores have no ingest log. Migration synthesizes `source_kind=legacy_unknown` rows per existing belief at the original `belief.created_at`. Acceptable but lossy.
- **Replay cost.** On a 1M-belief project, replay is minutes to hours. Mitigation: replay only on rule-set bump, which is rare. The materialized state is the day-to-day; replay is the migration tool.
- **Architectural shift.** Today `scan_repo` writes directly to `beliefs`. The proposed shape is `scan_repo` writes to `ingest_log`, then a derivation worker materializes `beliefs`. This is a real refactor of the ingest path.

---

## Smallest first step

Don't refactor `scan_repo`. Add `ingest_log` as a parallel table, populated alongside existing writes. No materialized-view contract yet. This gives:

- The data needed to replay later, without committing to a derivation-worker architecture.
- A live log to validate against — at any point, derive beliefs from the log and assert equality against the canonical `beliefs` table.
- A migration target — at the version where derivation becomes authoritative, we have months of log to validate the derivation function against.

This is a v2.0 candidate. It is foundational work; it does not move user-visible numbers; shipping it half-done is worse than not shipping it.

---

## What this memo does not propose

- Moving away from SQLite. SQLite's WAL is fine as the durability mechanism. The proposal is an application-level log living in a SQLite table.
- Event sourcing in the full CQRS sense. The derived state stays a SQLite store; no separate read model.
- Removing the `beliefs` table. It stays as the materialized view. The contract change is which one is authoritative.
- Externalizing the log (e.g., to a Kafka-style stream). All-local single-file SQLite remains the project's stance.

---

## Cross-references

- Determinism contract: [PHILOSOPHY § Determinism is the property](../concepts/PHILOSOPHY.md#determinism-is-the-property)
- Current schema: `src/aelfrice/store.py`
- Current ingest path: `src/aelfrice/scanner.py`
- v3 federation: [federation-primitives.md](federation-primitives.md) — the ingest log is the natural unit of inter-scope replication.
