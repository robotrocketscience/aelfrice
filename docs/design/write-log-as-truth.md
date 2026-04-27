# Design Memo: Write Log as Source of Truth

**Status:** v2.0 architectural direction. Documents the proposed contract change; not yet implemented.

---

## The contract

The append-only log of writes is the canonical state. The queryable structures — the FTS5 index, the typed graph, the alpha/beta posteriors — are *materialized views* over the log. To change a derived view (new extraction rule, new edge inference, schema migration), replay the log against the new derivation function. The log is immutable record; everything else is derived.

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

- Determinism contract: [PHILOSOPHY § Determinism is the property](../PHILOSOPHY.md#determinism-is-the-property)
- Current schema: `src/aelfrice/store.py`
- Current ingest path: `src/aelfrice/scanner.py`
- v3 federation: [federation-primitives.md](federation-primitives.md) — the ingest log is the natural unit of inter-scope replication.
