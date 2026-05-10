# Feature spec: Document / semantic linker (#435)

**Status:** implementation shipped (PR for #435), bench-gate pending lab-side corpus
**Issue:** #435
**Recovery-inventory line:** [`docs/ROADMAP.md`](ROADMAP.md) — *"Doc / semantic linker | v2.0.0"*
**Substrate prereqs:** belief schema (foundation), `ingest_log` (#205, v1.6.0), `belief_corroborations` (#190, v1.5.0), `DerivationInput.source_path` (`src/aelfrice/derivation.py:64`)

---

## Purpose

Connect a belief to the **document anchor** it describes — a file path, a section, a URL — so retrieval can return the canonical reference alongside the bare belief snippet. Distinct from `EDGE_CITES` (`models.py:27`), which is belief→belief: this is **belief↔document**.

The research line shipped this so a query like *"how does X work?"* surfaces the belief plus a pointer to the source file or doc section, letting the consumer route to the document for context the snippet doesn't carry. Today aelfrice records `source_path` on `ingest_log` rows (`store.py:242`) and on `DerivationInput` during a single ingest pass, but it does not persist a queryable mapping from `belief_id` to its canonical document.

---

## Contract

```python
from aelfrice.doc_linker import DocAnchor, link_belief_to_document

@dataclass(frozen=True)
class DocAnchor:
    belief_id: str
    doc_uri: str            # see "Doc URI scheme" below
    anchor_type: str        # "ingest" | "manual" | "derived"
    position_hint: str | None  # e.g. "L42-L60", "#section-name"; nullable
    created_at: float       # unix timestamp

# Ingest-time: invoked by onboard / commit-ingest when source_path is known.
def link_belief_to_document(
    store: MemoryStore,
    belief_id: str,
    doc_uri: str,
    *,
    anchor_type: str = "ingest",
    position_hint: str | None = None,
) -> DocAnchor: ...

# Retrieval-time: optional projection onto retrieve() output.
def get_doc_anchors(store: MemoryStore, belief_id: str) -> list[DocAnchor]: ...
```

`MemoryStore` gains two methods (no public surface change beyond):

- `store.link_belief_to_document(...)` — INSERT into the new `belief_documents` table (schema below). Idempotent on `(belief_id, doc_uri)` via `INSERT OR IGNORE`.
- `store.get_doc_anchors(belief_id)` — SELECT all anchors for a belief, ordered by `created_at`.

`retrieve()` / `retrieve_v2()` gain an opt-in kwarg:

```python
def retrieve(..., with_doc_anchors: bool = False) -> RetrievalResult: ...
```

When `with_doc_anchors=True`, each entry in `RetrievalResult.beliefs` is paired with `RetrievalResult.doc_anchors[i]: list[DocAnchor]` (parallel list, same length). Default `False` keeps the existing pack contract byte-stable.

---

## Doc URI scheme

The linker stores opaque strings; it does not parse the URI beyond rejecting empty input. The recommended encoding at v2.0.0 is one of two forms:

| Form | Shape | When |
|---|---|---|
| **File URI** | `file:///abs/path/to/source.py#Lstart-Lend` | local-source ingest (onboard, commit-ingest); `position_hint` carries the line range, mirrored in the URI fragment |
| **Web URL** | `https://host/path#fragment` | external-doc ingest (web-onboard, manual `aelf remember --doc=URL`) |

`doc_uri` is `TEXT NOT NULL` in the schema below. Beyond non-empty, the linker enforces no validation at v2.0.0 — the parsing burden lives with the consumer. A future revision may add a URI validator if a class of malformed inputs causes operator pain.

---

## Schema

New table:

```sql
CREATE TABLE belief_documents (
    belief_id     TEXT NOT NULL,
    doc_uri       TEXT NOT NULL,
    anchor_type   TEXT NOT NULL CHECK (anchor_type IN ('ingest', 'manual', 'derived')),
    position_hint TEXT,
    created_at    REAL NOT NULL,
    PRIMARY KEY (belief_id, doc_uri),
    FOREIGN KEY (belief_id) REFERENCES beliefs(id) ON DELETE CASCADE
);
CREATE INDEX idx_belief_documents_belief_id ON belief_documents (belief_id);
CREATE INDEX idx_belief_documents_doc_uri   ON belief_documents (doc_uri);
```

Notes:

- `(belief_id, doc_uri)` PK gives idempotency on re-ingest of the same belief from the same source.
- `ON DELETE CASCADE` on `belief_id` — when a belief is hard-deleted (#440 `aelf delete`), its anchors disappear with it. The linker is a derived projection of belief origin, not an audit trail; `belief_corroborations` (#190) is the audit-trail sibling for re-ingest events.
- `anchor_type` enum:
  - `ingest` — written at onboard / commit-ingest time when `source_path` is known.
  - `manual` — written by `aelf remember --doc=URI` (operator-supplied; ships with this spec only as a CLI surface, not a core requirement).
  - `derived` — reserved for a future revision that infers anchors from belief content (e.g. parsing a belief that contains `file:src/foo.py` and writing the anchor automatically). **Out of scope at v2.0.0.**
- The reverse direction (`doc_uri` → list of beliefs) is the `idx_belief_documents_doc_uri` query — useful for "what beliefs reference this file?" without scanning.
- Migration is forward-only (`migrate.py` adds the table on first open after upgrade); empty table populated by future ingests.

### Why a sibling table, not a column on `beliefs`

A belief can have multiple anchors:

- A belief like *"the WAL discussion lives in `docs/ARCHITECTURE.md` § Storage"* may also be cited at `docs/PHILOSOPHY.md` § Why-SQLite.
- An onboard-extracted belief carries one ingest anchor; an operator may add a `manual` anchor pointing to a related doc.

A scalar column on `beliefs` collapses this to one anchor per belief. A sibling table is the standard schema for one-to-many; `belief_corroborations` (#190) and `belief_neighbors` (#227) are the established precedents.

### Why not extend `belief_corroborations`

`belief_corroborations` (`store.py:142-160`) records each **re-ingest** of an existing belief — when the same content shows up again under a new `source_kind` / `source_path`. Doc-linker writes apply on **first ingest** as well, when no corroboration row exists. The two cardinalities differ: corroborations are a 1+ event log; doc anchors are a 1+ membership relation. Separating them keeps the corroboration recorder's posterior-update semantics (#190) untouched.

---

## Linker invocation point

### Recommended: ingest-time only at v2.0.0

Hook the linker into the existing ingest paths where `source_path` is materialised:

- **Onboard** — `derivation.py:223` (`source = inp.source_path or inp.source_kind`). When `source_path` is set, the post-derive code calls `link_belief_to_document(belief_id, file_uri_from(source_path), anchor_type="ingest")`.
- **Commit-ingest** — same hook; the commit-ingest path also flows through `derivation.py`.
- **Transcript-ingest** — `source_path` is typically `None` (transcripts have no canonical doc URI). Skip the linker call when `source_path is None`.
- **`aelf remember`** — when invoked with the `--doc` flag (new at v2.0; one-line CLI addition), call `link_belief_to_document(..., anchor_type="manual")`.

### Deferred: retrieval-time projection

The research-line surface also supported retrieval-time inference: if a belief has no stored anchor but its content contains a recognisable file-path/URL token, project that as a tentative anchor on the way out. This is **out of scope at v2.0.0** because:

1. It requires a token-extraction pass on every retrieval, on the hot path.
2. The output is non-canonical (an inferred anchor isn't the same kind as a stored one), so the `anchor_type` enum would need a new value; that interacts with the bench-gate.

The deferred mechanism is `anchor_type="derived"` above; the table reserves the value but the writer is not in this spec.

---

## Where the linker sits in retrieval

The linker is a **post-rank, pre-pack projection**. After lane fan-out and rank, but before the budget pack, retrieval optionally fetches doc anchors per belief:

```
query
  -> lane fan-out + rank
  -> [for each ranked belief, fetch doc_anchors] if with_doc_anchors=True
  -> compose
  -> pack
```

The fetch is a single batched query (`SELECT ... WHERE belief_id IN (...)`); cost is one indexed read per retrieval call, dominated by the existing pack-loop cost. The token-budget pack does **not** count `doc_anchors` against `token_budget` — anchors are metadata for the consumer, not body text. If a future revision wants to render anchors inline in the pack, a `with_doc_anchors_inline=True` kwarg owns that behaviour separately.

`with_doc_anchors=False` (the default) skips the fetch entirely. No flag in `[retrieval]` TOML — this is a per-call kwarg, not a process-wide gate.

---

## Configuration

No new TOML knobs at v2.0.0. The kwargs (`with_doc_anchors`) and CLI flags (`aelf remember --doc=URI`) are the only surface.

If a future revision wants a process-wide default for anchor projection, the convention is the same as `use_bm25f_anchors` (`retrieval.py:118-131`): kwarg → env var → `[retrieval] with_doc_anchors_default = true` in `.aelfrice.toml`. **Deferred.**

---

## Reconciliation

### vs. #148 (BM25F anchor text)

#148 stores **belief-incoming-edge text** (the words a citing belief used to describe a target belief). It augments the indexed-document field for retrieval scoring. The doc linker stores **belief-outgoing document references** (a pointer from a belief to the source it describes). Different direction, different consumer:

| | #148 (BM25F anchor text) | #435 (doc linker) |
|---|---|---|
| direction | belief A's incoming edges (citers) | belief A's outgoing reference (source) |
| storage | `edges.anchor_text` (existing) | new `belief_documents` table |
| consumer | BM25F scoring | retrieval output projection |
| retrieval impact | augments ranking | augments output metadata |

Neither replaces the other. They compose with no shared code path.

### vs. `EDGE_CITES` (`models.py:27`)

`EDGE_CITES` is belief→belief: belief A cites belief B (both in-store). Doc linker is belief→document: belief A references doc D (out-of-store). The two cardinalities differ — a CITES edge requires both endpoints to be beliefs; a doc anchor requires only the belief to exist.

A future revision could synthesise CITES edges from doc anchors (when two beliefs anchor to the same doc, infer a relationship) — that is out of scope at v2.0.0. The bench-gate (below) measures the linker's impact on retrieval, not its impact on the graph.

### vs. `ingest_log` source_path (#205)

`ingest_log` (`store.py:242`) records the per-turn `source_path` of every ingest event. It is the **event log**; doc anchors are the **materialised state**. The relationship parallels #264 (the v2.x derivation worker) — `ingest_log` is the source of truth, materialised tables (beliefs, edges, and now `belief_documents`) are derived.

The linker writer is a sibling of the derivation worker: both consume `ingest_log` rows and write into derived tables. The implementation PR may build the linker as a derivation-worker output node, or as a standalone writer wired into the same paths. Either choice keeps the source-of-truth invariant intact.

### vs. #227 (`belief_neighbors`)

#227 ships a `belief_neighbors` table. The schema for `belief_documents` follows the same conventions: `ON DELETE CASCADE`, `(belief_id, ...)` composite PK, `created_at` for audit, dedicated index for the reverse-direction query. `belief_neighbors` is the precedent; `belief_documents` is the second derived-table sibling.

---

## Acceptance

### A1 — corpus

A labeled `doc_linker` corpus lives under `tests/corpus/v2_0/doc_linker/`, mirroring the v2.0 corpus scaffold. Rows encode `(query, expected_belief_ids, expected_doc_uris)`. Public CI runs with `AELFRICE_CORPUS_ROOT` unset and skips via the autouse `bench_gated` marker; labeled content lives in the lab repo only, per the published corpus policy.

### A2 — retrieval uplift

On the `doc_linker` fixture:

```
NDCG@k(with_doc_anchors=ON, anchors_populated=ON) > NDCG@k(with_doc_anchors=ON, anchors_populated=OFF)
```

This measures the **utility of the anchor data**, not the cost of the projection. The threshold is **strictly positive uplift** over a baseline that ranks the same beliefs without consulting the anchor table. If consumers can route on doc-anchor presence (e.g. boost beliefs whose anchors match a query-extracted file path), the lift should be measurable.

### A3 — idempotency

`store.link_belief_to_document(b, uri)` called N times produces exactly one row in `belief_documents`. Property test asserts the count is independent of call frequency.

### A4 — schema migration

`migrate.py` creates `belief_documents` on first open of an existing v1.7-era store. A round-trip test opens, migrates, writes one anchor, reads it back, asserts row equality.

### A5 — composition tracker

The #154 composition tracker doc gains a row for `with_doc_anchors`: input shape, output shape, where it sits, bench verdict. Like type-aware compression (#434), this is **not a lane** — it is an output-projection. The tracker row is present for operator clarity.

---

## Bench-gate / ship-or-defer policy

`needs-spec` → `bench-gated` once this spec lands. Implementation is the next gate. **The implementation PR ships only on positive bench evidence per A2.** The schema migration (A4) and idempotency (A3) are mechanical correctness checks; A2 is the ship gate.

The schema migration is **forward-only and additive** — once the table is created, removing it is a destructive operation. If A2 fails, the table can stay (no rows == no impact); the writer call sites are reverted instead.

---

## Out of scope at v2.0.0

- **`anchor_type="derived"` writers** (retrieval-time inference from belief content). Reserved enum value, no writer. Deferred to v2.x.
- **URI validation.** `doc_uri` is opaque `TEXT`. A validator may land in v2.x if pain accumulates.
- **CITES synthesis.** Inferring belief↔belief edges from shared doc anchors. Separate spec.
- **Process-wide anchor-projection default.** No `[retrieval] with_doc_anchors_default` knob. Per-call kwarg only.
- **Document-content embedding / semantic linking.** The "semantic" half of the issue title is **lexical** at v2.0.0 (URI-equality semantics). Genuine semantic linking (anchor a belief to a doc whose *content* matches) requires an embedding model and is excluded by the project's no-embedding-in-retrieval posture. If it ships, it is a v3.x candidate.
- **`aelf doctor` integration.** Pruning anchors that point to vanished files, rewriting anchors after a rename, etc. Sibling work; deferred.

---

## Implementation prereqs

- `src/aelfrice/store.py:142-162` — `belief_corroborations` table as the schema-pattern precedent.
- `src/aelfrice/store.py:316-330` — migration block; new `ALTER`/`CREATE` lines added here.
- `src/aelfrice/derivation.py:64, :223` — `DerivationInput.source_path` is the input the ingest-time writer consumes.
- `src/aelfrice/derivation_worker.py` — sibling writer; the doc-linker writer can attach as a derivation-worker output node or run alongside it.
- `src/aelfrice/retrieval.py` — output-shape change for `with_doc_anchors=True` plumbing.
- `src/aelfrice/cli.py` — `aelf remember --doc=URI` flag.
- `tests/corpus/v2_0/`, `tests/bench_gate/` — corpus + harness.

All substrate is on `main` as of `68dafc0`. No new dependencies. One additive schema migration.

---

## Open questions for review

1. **`anchor_type="derived"` reservation.** Reserve the value now (clean enum at v2.0.0) vs add it when the writer ships (no migration needed)? The spec defaults to reserve, on the grounds that schema additions are cheap and enum-extending later is more annoying than enum-pruning never. Pick at impl-PR review.
2. **`with_doc_anchors_inline`.** Mentioned in §"Where the linker sits" as a deferred kwarg for inline rendering. Is the use case real, or is the consumer-side projection (kwarg returns metadata, consumer formats) sufficient? Settle once a consumer asks.
3. **Migration of existing beliefs.** New ingests populate `belief_documents` going forward; existing beliefs have no anchors until re-ingested. Is there a one-shot migration that walks `ingest_log` and back-fills `belief_documents` from historical `source_path` rows? Probably yes — the data is there. Spec leaves this as a follow-up, not a ship gate.
4. **`source_path` normalisation.** Should the writer normalise `/Users/x/projects/aelfrice/docs/foo.md` to `docs/foo.md` (relative-to-repo-root) before storing? Absolute paths leak local filesystem layout into the store. Recommended default: relative-to-repo-root, with `aelf doctor --normalize-doc-uris` for cleanup. Confirm at impl-PR review.
