# Entity-index retrieval (L2.5)

**Status:** spec.
**Target milestone:** v1.3.0 (named on the public roadmap as "entity-index
retrieval"; ports forward from the earlier research line).
**Dependencies:** v1.2.0 triple-extraction port (already shipped). Stdlib
only — no new third-party deps. Consumes the v1.2.0 [`Belief.session_id`](../src/aelfrice/models.py),
the v1.2.0 [`Edge.anchor_text`](../src/aelfrice/models.py), and the
existing [`MemoryStore`](../src/aelfrice/store.py) callback registry.
**Risk:** medium. New retrieval tier sitting between L0 + L1 (FTS5/BM25)
and the eventual BFS multi-hop layer. Correctness depends on the entity
patterns generalising to real prose; latency depends on the on-write
extraction cost staying off the hot path.
**Prior art:** the earlier research line shipped `entity_extractor.py`
producing an entity → belief inverted index against the same SQLite
substrate. This spec reasons from
[`src/aelfrice/triple_extractor.py`](../src/aelfrice/triple_extractor.py)
(noun-phrase regexes already in tree) plus the new identifier patterns
named below; the predecessor module is referenced for design intent
only and is not lifted verbatim.

## Summary

A new retrieval tier — **L2.5** — that sits between L1 (FTS5/BM25 keyword
search) and the future L3 (BFS multi-hop graph traversal). Mechanism:

1. **Extraction.** A regex bank pulls structured entities (file paths,
   identifiers, branch names, version strings, URLs, error codes,
   noun phrases) out of every belief at insert / update time and out
   of the live query at retrieve time.
2. **Index.** A new `belief_entities` table stores
   `(entity_lower, belief_id, kind, span_start, span_end)`. Indexed on
   `entity_lower` so entity → beliefs is a single SQL lookup.
3. **Refresh.** On-write trigger — every `insert_belief` /
   `update_belief` extracts entities and rewrites the per-belief rows
   inside the same transaction. Cache invalidation reuses the existing
   `add_invalidation_callback` registry.
4. **Query.** L2.5 extracts entities from the user's query, looks them
   up in the inverted index, and returns matched beliefs ranked by
   entity-overlap count then BM25 (when also returned by L1).
5. **Budget.** L2.5 is **additive** to L1 within a slightly expanded
   default budget. L0 is still untouchable. Defaults below.

```
L0:    store.list_locked_beliefs()         always loaded; never trimmed
            ↓
L2.5:  entity_index_lookup(query_entities) deterministic, exact-match
            ↓
L1:    FTS5 BM25 (limit l1_limit)          existing v1.0 path
            ↓
Dedupe L2.5 / L1 against L0 ids and against each other (L2.5 wins)
            ↓
Trim L1 from tail until sum(estimated_tokens) ≤ token_budget
```

L2.5 lands ahead of L1 in the output ordering: an entity hit is more
specific than a bag-of-words hit and more likely to satisfy a precise
query (a file path, a function name, an error code) before a fuzzier
keyword match.

## Motivation

Three independent v1.3 asks bottleneck on this:

1. **Closing the multi-hop chain-valid gap on MAB.** Per
   [ROADMAP.md § v1.3.0](ROADMAP.md#v130--retrieval-wave), aelfrice's
   v1.0–v1.2 retrieval is L0 + BM25 only. The MAB (MemoryAgentBench)
   chain-valid baseline is the v1.3.0 milestone target the entity index
   plus the v1.3.0 BFS layer are sized to reach. BFS layers on top of
   entity hits — without a precise-match retrieval tier underneath, the
   BFS frontier is BM25's noisy top-K, which is already known to dilute
   chain-valid recall on the predecessor research line.
2. **Identifier-shaped queries.** The agent's own search queries
   (already surfaced through the v1.2.x search-tool hook —
   [search_tool_hook.md](search_tool_hook.md)) are densely identifier-
   shaped: file paths, function names, branch names. BM25's bag-of-
   words tokenizer chops `aelfrice.retrieval` into `aelfrice` and
   `retrieval` and ranks them as ordinary tokens. An entity index
   keyed on the literal identifier preserves the precision the agent
   asked for.
3. **Producer side already shipped.** v1.2.0's triple-extraction port
   stamped every triple-derived belief with a noun-phrase content
   hash and populated `Edge.anchor_text` densely. The same prose that
   produced those triples will produce entities. L2.5 is the *reader
   side* the v1.2.0 producer was always sized for.

This is also the first retrieval tier where ranking is **not** BM25.
Locking the slot in before posterior-weighted ranking lands (v1.3.0
partial / v2.0.0 full) keeps the eventual ranking surface composable —
posterior weighting can apply uniformly across L1 + L2.5 without the
two tiers having to share an underlying scorer.

## Scope

### In

- `src/aelfrice/entity_extractor.py`: pure regex extractor (no store).
  Mirrors the `triple_extractor.py` shape — `extract_entities(text) ->
  list[Entity]`.
- `src/aelfrice/entity_index.py`: store-aware refresh + lookup. Wires
  to `MemoryStore` mutators via the existing invalidation callback
  registry.
- New `belief_entities` table + supporting indexes (forward-compatible
  per v1.x semver; additive only).
- L2.5 wired into `aelfrice.retrieval.retrieve()` behind a config
  flag (default-on at v1.3.0 once the regression tests below pass;
  default-off if the latency budget regresses).
- `mab_entity_index_adapter` in `benchmarks/` — a thin variant of
  `mab_adapter.py` that flips the L2.5 toggle on and reports both
  the headline scoring metrics and the L0/L1/L2.5 counts.

### Out

- BFS multi-hop traversal. That is a separate v1.3.0 deliverable
  (issue tracked under "BFS multi-hop graph traversal"). L2.5
  surfaces the *seed beliefs* BFS will walk from; the walk itself
  is not in this spec.
- LLM-based entity extraction. v1.3.0 ships mechanical (regex-only)
  extraction. The Haiku-backed onboard classifier is a parallel
  v1.3.0 track — see ROADMAP — and may call the entity extractor
  but does not replace it.
- HRR / sentence embeddings (deferred to v2.0.0 per ROADMAP non-goals).
- Co-reference resolution. "the parser" and "our parser" remain
  separate entities. Same stance as `triple_extractor.py` § Resolution
  policy.
- Cross-language entity extraction. English only. v1.3.0 does not
  attempt to recognise non-English noun phrases — identifiers are
  language-neutral and dominate the agent's query mix.

## Schema

### Decision: new `belief_entities` table

**Recommendation: a new table.** Reasoning:

| Concern | New table (recommended) | Column on `beliefs` |
|---|---|---|
| Cardinality | One belief → many entities (typical: 5–20). Natural one-to-many relation. | Forces a delimited list (`,`-joined) or JSON blob — neither is queryable in SQLite without `instr` / `json_each`. |
| Insert cost | One `INSERT … VALUES …` per entity, batched in the same transaction as the belief. | One column write — but the encoding work (sort, join, dedup) lives in Python. Net wash on insert. |
| Lookup cost | `SELECT belief_id FROM belief_entities WHERE entity_lower = ?` is index-backed and runs in O(log N + K). | Requires `WHERE beliefs.entities LIKE '%entity%'` — full table scan, no index. Disqualifying. |
| FTS5 interaction | Independent table. FTS5 (`beliefs_fts`) keeps its current shape. No reindex on entity churn. | Would either bloat FTS5's `content` column or fragment it across two FTS5 indexes. |
| Migration | Pure additive: `CREATE TABLE` + `CREATE INDEX`. v1.0/v1.1/v1.2 stores open at v1.3 unchanged. | Adds a column to `beliefs` — also additive, but couples entity storage to `beliefs` row size and slows full-row scans for unrelated queries. |
| Debug / introspect | `aelf entities <query>` (future CLI) reads one table. | Forces the CLI to parse a delimited blob from inside `beliefs`. |

The new-table option is materially better on the lookup axis (the only
one that runs on the hot path) and at worst comparable on every other
axis. The table:

```sql
CREATE TABLE IF NOT EXISTS belief_entities (
    belief_id    TEXT NOT NULL,
    entity_lower TEXT NOT NULL,
    entity_raw   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    span_start   INTEGER NOT NULL,
    span_end     INTEGER NOT NULL,
    PRIMARY KEY (belief_id, entity_lower, span_start)
);

CREATE INDEX IF NOT EXISTS idx_belief_entities_lower
    ON belief_entities(entity_lower);
CREATE INDEX IF NOT EXISTS idx_belief_entities_belief
    ON belief_entities(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_entities_kind
    ON belief_entities(kind);
```

Field notes:

- `entity_lower` is the lookup key. Lowercasing makes case-insensitive
  matches O(log N) without a separate functional index.
- `entity_raw` preserves the original spelling for display
  (`Belief.content` already has it embedded; the column is here so the
  index can serve simple "list entities" queries without re-reading
  the parent belief).
- `kind ∈ {file_path, identifier, branch, version, url, error_code,
  noun_phrase}` — see § Algorithm. Exposed as a column so a future
  ranker can weight by kind without re-extracting.
- `span_start` / `span_end` are byte offsets into `Belief.content`.
  Used by the future highlighting / anchor-aware ranker; currently
  written for free since the regex `Match` object has them.
- The composite PK `(belief_id, entity_lower, span_start)` permits
  the same entity to appear multiple times in one belief (different
  spans) without violating uniqueness.

### Migration story

Per the v1.x semver commitment ([ROADMAP § Compatibility](ROADMAP.md#compatibility)):

- v1.3.0 adds the `CREATE TABLE` + three `CREATE INDEX` statements to
  `_SCHEMA` in `store.py`. They are `IF NOT EXISTS`, so a v1.3 store
  opens cleanly on a v1.0/v1.1/v1.2 DB.
- A one-shot **backfill** runs on first open of a v1.3+ binary against
  a pre-v1.3 store: extract entities for every existing belief and
  insert rows. Backfill is idempotent (PK is `belief_id, entity_lower,
  span_start`; re-runs no-op). Tracked via a row in `schema_meta`
  (`entity_index_backfilled_at`).
- No destructive change. Downgrading to v1.2.x reads the table as
  unknown and ignores it — `MemoryStore.__init__` does not depend on
  it.

The backfill cost is bounded: extraction is pure-regex over the
existing `Belief.content` (no I/O), parallelisable per-belief, and the
row count is roughly equal to (# beliefs) × (mean entities/belief ≈ 8).
A 10k-belief store materialises ~80k `belief_entities` rows; on
commodity hardware this is sub-second. See § Telemetry for the
target.

## Algorithm

### Pattern bank

The extractor ports the noun-phrase pattern from
[`triple_extractor.py`](../src/aelfrice/triple_extractor.py) (already
in tree) and adds six identifier-shaped kinds. Each is listed here
with its regex and a one-line rationale.

| Kind | Regex (Python `re`, IGNORECASE where noted) | Rationale |
|---|---|---|
| `file_path` | `(?<![\w/])(?:[\w.-]+/)+[\w.-]+\.[\w]+(?![\w/])` | POSIX-ish path with at least one `/` and a dotted extension. Excludes URL-internal paths via the URL pattern's earlier match. |
| `file_path` (Windows) | `(?<![\w])[A-Za-z]:\\(?:[\w. -]+\\)+[\w.-]+\.[\w]+` | Drive-letter Windows path. Documented but optional at v1.3.0 — the agent rarely emits these on macOS / Linux. |
| `identifier` | `\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b` | Dotted Python identifier (`aelfrice.retrieval`, `store.list_locked_beliefs`). Disambiguated from sentence prose by requiring at least one dot AND lowercase head. |
| `identifier` (snake/camel) | `\b(?:[a-z]+_[a-z0-9_]+\|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b` | snake_case (`session_id`, `apply_feedback`) or CamelCase (`MemoryStore`, `RetrievalCache`) with at least two parts. The two-part requirement keeps single English words out. |
| `branch` | `\b(?:feat\|fix\|docs\|refactor\|chore\|exp\|test\|ci\|build\|style\|perf\|gate\|audit\|release\|revert)/[A-Za-z0-9._/-]+\b` | Conventional-branch prefix `<type>/<rest>` per the project commit-message rules. Catches `feat/invisibility-reframe`, `docs/entity-index-spec`, etc. |
| `version` | `\bv?\d+\.\d+\.\d+(?:[.-][A-Za-z0-9.-]+)?\b` | Semver-shaped. `v1.2.0`, `1.2.0a0`, `1.2.0-rc1`. The optional leading `v` is mandatory in product copy and required by `release/v*` tags. |
| `url` | `\bhttps?://[^\s<>"')\]]+` (IGNORECASE) | HTTP(S) URL up to the first whitespace / closing bracket. Conservative on the right boundary so trailing punctuation in prose (`see https://x.y/z.`) doesn't get pulled into the entity. |
| `error_code` | `\b(?:HTTP\|HTTPS)\s+\d{3}\b\|\bE[0-9]{3,5}\b\|\b(?:OSError\|ValueError\|RuntimeError\|TypeError\|KeyError\|IndexError\|AttributeError\|FileNotFoundError\|PermissionError\|TimeoutError\|sqlite3\.[A-Z][a-zA-Z]+)\b` | Three families: HTTP status (`HTTP 503`), Unix-style error codes (`E404`, `E1001`), and the named Python exception classes the agent's tracebacks emit. Extending this list is forward-compatible. |
| `noun_phrase` | `(?:(?:the\|a\|an\|our\|their\|its\|this\|that)\s+)?[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,4}` | Re-used from `triple_extractor._NP`. Up to 5 word-tokens, optionally led by an article / possessive. Captures multi-word noun phrases (`the search-tool hook`, `triple extractor`). |

Notes:

- Patterns run in declared order. Earlier-listed kinds **win on
  overlap** — a span matched as `file_path` is removed from the input
  before `identifier` runs over it, so `aelfrice/retrieval.py` produces
  one `file_path` entity, not three (`aelfrice/retrieval.py`,
  `aelfrice.retrieval`, `retrieval`). Implementation: a "consumed"
  interval list, mirroring the `triple_extractor` overlap policy.
- `noun_phrase` runs **last** and is the most permissive. It catches
  the long tail of multi-word phrases the structured kinds miss.
  Sentence-prose noise from this pattern is the dominant source of
  index bloat — the per-belief cap in § Refresh strategy is sized
  for this.
- The bank is **English-only** at v1.3.0. The structured kinds
  (file_path, identifier, branch, version, url, error_code) are
  language-neutral; `noun_phrase` is not. Cross-language is deferred
  to v2.x.

### Extractor API

```python
@dataclass(frozen=True)
class Entity:
    raw: str            # the literal matched substring
    lower: str          # raw.lower(), used as the index key
    kind: str           # one of the kinds in the table above
    span_start: int     # byte offset in source text
    span_end: int       # byte offset in source text


def extract_entities(
    text: str,
    *,
    max_entities: int = 64,
) -> list[Entity]:
    """Return entities in left-to-right order of match start.

    `max_entities` is a hard ceiling per call to bound the cost of a
    pathological belief (e.g. a license blob). Past the cap, the
    overflow is dropped and a single counter increments — the
    `aelf health` informational output exposes overflow count so
    operators can see when the cap is biting.
    """
```

### Index API

```python
class EntityIndex:
    """Store-aware refresh + lookup.

    Subscribes to MemoryStore.add_invalidation_callback() so the
    in-memory query cache (separate concern, lives in retrieval.py)
    flushes on the same schedule.
    """

    def __init__(self, store: MemoryStore) -> None: ...

    def refresh_belief(self, belief_id: str) -> None:
        """Re-extract entities for one belief; replace its rows in
        belief_entities atomically."""

    def lookup(
        self,
        entities: Iterable[Entity],
        *,
        limit: int = DEFAULT_L25_LIMIT,
    ) -> list[tuple[str, int]]:
        """Return [(belief_id, overlap_count)] sorted by overlap
        desc then belief_id asc, capped at `limit`."""

    def backfill_all(self) -> int:
        """One-shot backfill for pre-v1.3 stores. Returns row count.
        Idempotent."""
```

### Hot-path flow inside `retrieve()`

```python
def retrieve(store, query, token_budget=DEFAULT_TOKEN_BUDGET, ...):
    locked = store.list_locked_beliefs()
    locked_ids = {b.id for b in locked}

    l25: list[Belief] = []
    if query.strip() and CONFIG.entity_index_enabled:
        q_entities = extract_entities(query, max_entities=16)
        if q_entities:
            hits = store.entity_index.lookup(q_entities, limit=l25_limit)
            l25 = [
                store.get_belief(bid)
                for bid, _overlap in hits
                if bid not in locked_ids
            ]

    l1: list[Belief] = []
    if query.strip():
        l25_ids = {b.id for b in l25}
        raw_l1 = store.search_beliefs(query, limit=l1_limit)
        l1 = [b for b in raw_l1 if b.id not in locked_ids and b.id not in l25_ids]

    # L0 always survives. L2.5 lands above L1. L1 trims from tail.
    out = list(locked) + list(l25)
    used = sum(_belief_tokens(b) for b in out)
    for b in l1:
        cost = _belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
    return out
```

### Ranking

L2.5 ranks by **entity overlap count** (number of distinct query
entities that match the belief), tie-broken by `belief_id` ASC for
determinism. Posterior weighting (the v1.3.0 partial / v2.0.0 full
ranker) applies after this initial ordering and is layered on at the
ranking-PR boundary, not here.

Higher-order ideas explicitly **not** in v1.3.0 entity-index PR:

- Kind-weighted ranking (file_path > identifier > noun_phrase).
  Reserved for the v1.3.0 ranking PR after the synthetic eval shows
  the lift.
- Span-proximity boosts. The `span_start` / `span_end` columns are
  written but unused by the v1.3.0 ranker.
- IDF-style entity weighting. Common entities (`the`-led noun phrases)
  add noise; the per-belief cap and `noun_phrase`-runs-last policy
  handle the worst cases. IDF is a v2.0.0 candidate.

## Refresh strategy

### Decision: on-write trigger

**Recommendation: on-write.** Reasoning:

| Strategy | Pros | Cons |
|---|---|---|
| **On-write (recommended)** | Index is always consistent. No background daemon. No extra state machine. Cache invalidation is already wired through `add_invalidation_callback`. | Inflates `insert_belief` / `update_belief` cost. |
| Background rebuild | Insert path stays cheap. Can amortise extraction across N beliefs. | Two phases of consistency (write-time, post-rebuild); retrieval may miss recently-inserted beliefs until rebuild fires. Adds a watchdog the v1.x line has carefully avoided. |
| Lazy on retrieval | Insert path stays cheap. Cold-start cost amortises naturally. | Retrieval gets unpredictable: one query hits a populated index, the next sees a stale slice. Hard to reason about in tests. |

The on-write cost is bounded and runs synchronously inside the same
transaction as the parent `INSERT INTO beliefs`, so failure semantics
are clean (either both rows commit or neither does).

### Cost model

Per-belief overhead, back-of-envelope:

- Pattern count: ~9 regex passes over `Belief.content`.
- Mean content length: 200 chars (observed on the v1.2 corpus —
  triple-derived beliefs are short noun phrases; commit-ingested
  beliefs are commit-message paragraphs).
- Python `re.finditer` over 200 chars at compiled-regex speed:
  ~5 µs per pattern × 9 patterns = ~45 µs CPU.
- One `INSERT INTO belief_entities` per entity. Mean 8 entities per
  belief; SQLite write at ~10 µs/row inside a single transaction =
  ~80 µs.
- **Total: ~125 µs per belief insert / update overhead.**

Compared against the existing `insert_belief` cost (FTS5 insert plus
the `beliefs` row write — measured at ~500 µs–1 ms on commodity
hardware with WAL), this is a 12–25 % overhead per insert. Acceptable
on the producer side; the user-visible insert path (`aelf lock`,
`aelf remember`, the commit-ingest hook, transcript-ingest) is not
latency-bound at this granularity.

The hot path (`retrieve()`) does **not** incur extraction cost on
existing beliefs — those rows are already populated. It pays only
the query-side extraction (one regex pass over a typical query of
~50 chars: ~5 µs × 9 patterns ≈ 45 µs).

Tentative pending implementation experience: the 125 µs estimate
assumes Python's compiled regex cache stays warm. The implementation
PR should re-measure on a representative store before flipping the
config flag default.

### Failure mode

If extraction raises (which it should not — pure regex over a
string), the parent `insert_belief` / `update_belief` transaction
aborts and the belief itself is not persisted. This is stricter than
v1.0's behaviour where the belief commits unconditionally, but it
preserves the index ↔ store invariant we'd otherwise have to police
in `aelf health`. The implementation should catch only the specific
exceptions that can come from a malformed regex pattern (which would
be a code bug discoverable in CI), not arbitrary `Exception`.

## Budget split

### Decision: additive within an expanded cap

L0 is fixed at "all locked beliefs, never trimmed" — that is the v1.0
contract and v1.3 does not touch it.

L1 (BM25) and L2.5 (entity hits) compete for the remaining budget.
Two structural choices, ruled out, and the recommended third:

| Option | Behaviour | Reject because |
|---|---|---|
| **Subtractive** (L2.5 steals from L1) | Default `token_budget` stays at 2000. L2.5 runs first; whatever budget remains feeds L1. | Underuses retrieval surface. The agent ends up with strictly *less* BM25 context on identifier-heavy queries — the exact case the entity index is supposed to *improve*. |
| **Gating** (L2.5 replaces L1 when entities present) | If extracted-from-query entities is non-empty, return L0 + L2.5 only. Else fall back to L0 + L1. | All-or-nothing. A query with one weak entity (`the parser`) collapses BM25's keyword recall to nothing. The two tiers are complementary, not competing. |
| **Additive (recommended)** | Default `token_budget` increases from 2000 to 2400. L2.5 fills against a sub-budget of 400 tokens; L1 fills against the remaining 2000. | Costs 20 % more output tokens in the worst case. Worth it: this is auxiliary context we already trim, and the v1.0 default of 2000 was sized when L1 was the only non-locked source. |

**Recommended defaults at v1.3.0:**

| Constant | v1.0 / v1.2 | v1.3.0 (proposed) |
|---|---|---|
| `DEFAULT_TOKEN_BUDGET` | 2000 | 2400 |
| `DEFAULT_L1_LIMIT` | 50 | 50 (unchanged) |
| `DEFAULT_L25_LIMIT` (new) | — | 20 |
| `DEFAULT_L25_TOKEN_SUBBUDGET` (new) | — | 400 |
| `entity_index_enabled` (config) | n/a | `True` (default-on if regression tests pass) |

Rationale for 400 / 2000 split: 400 tokens is ~10 entity-matched
beliefs at a typical 40-token-per-belief content length. That is a
plausible upper bound on "how many distinct prior beliefs match the
identifiers in this query"; beyond 10, the agent is asking a vague
question and L1 (BM25) is the right tier. The 2000-token L1 sub-
budget preserves the v1.0 behaviour byte-for-byte on queries where
L2.5 returns nothing.

### Empty-query behaviour

Empty query: L0 only, exactly as v1.0. L2.5 doesn't fire on empty
queries (no entities to extract). This is required to preserve the
v1.0 hook contract — the SessionStart "baseline" injection runs
with an empty query and must remain L0-only.

### Backwards-compat retrieval

Callers that pass an explicit `token_budget=2000` still work; the
default expansion is a default-arg change, not a hard floor.
Adapters that pin `token_budget` (notably the MAB adapter, which
accepts `--budget`) get exactly what they ask for. This satisfies
the "minor releases preserve API" rule from
[ROADMAP § Compatibility](ROADMAP.md#compatibility).

## Activation in `benchmarks/`

### Adapter

`benchmarks/mab_entity_index_adapter.py` — a thin wrapper over
`mab_adapter.py`. Differences:

1. Constructs `MemoryStore` and explicitly enables L2.5 by passing
   `entity_index_enabled=True` to retrieval.
2. Uses `retrieve_v2(store, query, budget=..., use_entity_index=True)`
   (a new kwarg added on `retrieve_v2`, no-op outside v1.3.0+).
3. Reports an extended results block: per-question L0 / L1 / L2.5
   counts in the `per_question` JSON, plus an aggregate L2.5 hit
   rate in the summary.
4. Same MAB metrics (`exact_match`, `substring_exact_match`, `f1`).
   Pass-through to the same scorer.

### Definition of "activates"

For this spec, **"activates" means a config-level feature flag, not a
separate retrieval entry point.** Reasoning:

- The retrieval surface stays as one function (`retrieve()`). Easier
  to reason about than two parallel functions; no risk of an adapter
  drifting onto the wrong code path.
- The flag lives in `aelfrice.config` (see [CONFIG.md](CONFIG.md)) and
  defaults to `True` at v1.3.0. The benchmark adapter sets it
  explicitly so the result is reproducible regardless of the user's
  `.aelfrice.toml`.
- A separate retrieval entry point was considered and rejected: it
  would force every consumer (MCP, hook, CLI, benchmark adapter) to
  pick the right one, and the wrong choice silently disables the
  index. A flag is a single switch.
- Runtime opt-in via per-call kwarg is also rejected for the same
  reason — every caller would need to pass it, and missed call sites
  would silently miss the index.

The flag is exposed three ways:

1. `[retrieval] entity_index_enabled = true` in `.aelfrice.toml`.
2. `entity_index_enabled: bool` kwarg on `retrieve()` and
   `retrieve_v2()` (overrides config when set).
3. Environment variable `AELFRICE_ENTITY_INDEX=0` for emergency
   disable. Same convention as the v1.2.x `AELFRICE_SEARCH_TOOL=0`
   off-switch.

### What gets measured

The benchmark adapter exists to demonstrate the v1.3.0 acceptance
criterion that L2.5 reaches the MAB chain-valid baseline target.
Specifically:

- **Headline metric:** MAB Conflict_Resolution (factconsolidation_mh
  source) F1 vs. the L1-only baseline. Goal: reproduce the chain-
  valid lift sized in the v1.3.0 milestone framing. Exact target
  number is set by the v1.3.0 milestone gate, not this spec.
- **Latency band:** P50 retrieval latency on the same corpus. Must
  not regress beyond the documented band (see § Validation).
- **Index size:** total `belief_entities` row count vs. belief
  count. Diagnostic only — surfaces in the report, doesn't gate.

## Telemetry

`aelf health` (and `aelf status`) gain three informational counters:

| Counter | Meaning |
|---|---|
| `entity_rows` | `SELECT COUNT(*) FROM belief_entities`. Health if ~5–20× `count_beliefs()`. Below 1× signals extraction is misfiring; above 50× signals the per-belief cap is biting (see `entity_overflow`). |
| `entity_overflow_total` | Counter on the in-process extractor — cumulative beliefs that hit `max_entities=64`. Surfaced via the `aelf health` output as a warning above a configurable threshold. |
| `entity_index_backfilled_at` | ISO timestamp from `schema_meta`. Empty on v1.3-fresh stores; set on first open of a v1.3+ binary against a pre-v1.3 store. Diagnostic. |

These mirror the triple-extractor's "X commit messages parsed; Y
triples produced" health surface (per
[triple_extractor.md § Open questions](triple_extractor.md#open-questions))
so operators can see when the corpus needs broader patterns.

No outbound network calls, per [ROADMAP § Non-goals](ROADMAP.md#non-goals).

## Validation / acceptance

The implementation PR (separate from this spec) must demonstrate:

1. **Pattern coverage.** Per-kind unit tests with at least three
   positive and three negative fixtures. Each kind documented above
   has a dedicated test file
   (`tests/test_entity_extractor_<kind>.py`) so failing patterns are
   easy to localise.
2. **Idempotency.** `EntityIndex.refresh_belief(b.id)` called twice
   produces the same row set. `EntityIndex.backfill_all()` called
   twice produces zero new rows on the second call.
3. **On-write trigger fires.** A test inserts a belief through
   `MemoryStore.insert_belief` and verifies `belief_entities` rows
   exist for it without the test calling the index directly.
4. **Cache invalidation.** A `RetrievalCache` fronting an L2.5-aware
   `retrieve()` invalidates on belief mutations exactly as v1.0
   already does (regression coverage; the L2.5 work must not break
   existing cache semantics).
5. **Budget regression.** A new
   `tests/regression/test_l25_latency.py` runs the full L0 + L1 +
   L2.5 pipeline on a 10k-belief fixture store. Asserts:
   - Median end-to-end retrieve latency ≤ 50 ms.
   - p95 ≤ 200 ms.
   - Documented regression band: no more than +20 % over the v1.2.0
     L0 + L1 median on the same fixture. (The v1.2.0 baseline is
     captured in `benchmarks/results/v1.0.0.json` and fixtured into
     the test as a frozen number.)
6. **MAB chain-valid regression.** `mab_entity_index_adapter` runs
   on the Conflict_Resolution split (factconsolidation_mh source,
   `--rows 5 --subset 5` for CI speed) and produces F1 ≥ the
   committed baseline value. The full sweep (all rows) runs offline
   for the milestone gate; CI runs the smoke variant.
7. **Forward compatibility.** A v1.0 fixture store opens cleanly on
   v1.3.0 binary, the backfill runs, and existing `retrieve()`
   results for queries that don't trigger entities are byte-
   identical to the v1.0 result. (Locks the additive-only schema
   commitment.)
8. **Default-off regression.** `AELFRICE_ENTITY_INDEX=0` reverts
   `retrieve()` to byte-identical v1.2 output on the synthetic
   benchmark harness — proves the off-switch is real.

## Open questions resolved

| Issue body asks | Recommendation in this spec | Rationale |
|---|---|---|
| Exact regex patterns | Nine kinds tabulated above (`file_path` × 2, `identifier` × 2, `branch`, `version`, `url`, `error_code`, `noun_phrase`). | Concrete patterns; rationale for each; overlap policy specified. |
| Schema: new table vs. column | New `belief_entities` table. | Index-backed lookup is the only viable hot-path option; column-encoded list disqualifies on lookup cost. |
| Refresh: on-write vs. background | On-write. | Bounded ~125 µs/insert overhead; preserves single-transaction consistency; no daemon. |
| Budget: subtractive vs. additive vs. gating | Additive. Default budget 2000 → 2400. L2.5 sub-budget 400; L1 sub-budget 2000. | Avoids cannibalising L1 on identifier-heavy queries; preserves v1.0 behaviour on entity-empty queries. |
| Activation in `benchmarks/` | Config flag (`entity_index_enabled`); single retrieval entry point; benchmark adapter flips the flag explicitly. | Avoids parallel retrieval surfaces and silent miss-by-default failures. |

Tentative pending implementation experience (called out for re-
measurement at PR time, not punted):

- **Per-belief overhead estimate (~125 µs).** Re-measure against the
  realistic v1.2 corpus; if the on-write cost regresses past the
  +20 % insert-latency band, the implementation PR can defer the
  default-on flip without changing this spec's structure.
- **`max_entities=64` cap.** May be tuned up or down based on the
  observed `entity_overflow_total` rate on a v1.2 fixture store. The
  cap exists to protect against pathological beliefs (license blobs,
  serialised JSON pasted into a transcript); the right value is
  empirical.
- **`DEFAULT_L25_LIMIT=20` and `DEFAULT_L25_TOKEN_SUBBUDGET=400`.**
  Sized from a typical entity-overlap distribution; the implementation
  PR should report the L2.5 hit-count distribution on the MAB
  factconsolidation_mh corpus so the defaults can be confirmed or
  retuned before flipping the config flag default-on.

These are bounds on the *defaults*, not the *design*. The design
holds either way.

## Dependencies

Required at v1.3.0 implementation time:

- **v1.2.0 triple-extraction port** — already shipped. Re-uses the
  noun-phrase regex (`_NP` in `triple_extractor.py`) and the same
  regex-only stance.
- **v1.1.0 per-project DB resolution** — already shipped. The
  `belief_entities` table lives in the same per-project DB as
  `beliefs`, so per-project isolation flows through unchanged.
- **v1.0.1 retrieval cache invalidation registry** — already shipped.
  The new index hooks into the same `add_invalidation_callback`
  surface.

No new third-party dependencies. Pure stdlib (`re`, `sqlite3`).

## Out of scope (explicit)

- BFS multi-hop traversal (separate v1.3.0 PR; layers on L2.5 hits).
- LLM-classification onboard (parallel v1.3.0 track).
- Posterior-weighted ranking (v1.3.0 partial / v2.0.0 full; layers
  uniformly across L1 + L2.5).
- HRR / sentence embeddings / vector retrieval (v2.0.0).
- Cross-language entity extraction.
- Co-reference resolution.
- Kind-weighted ranking, span-proximity boosts, IDF weighting (all
  reserved for the v1.3.0 ranker PR or later).
- A user-facing `aelf entities <query>` CLI. The data is in the table;
  surfacing it as a command is a v1.3.x patch candidate.

## What unblocks when this lands

L2.5 is the seed-belief tier the v1.3.0 BFS multi-hop layer walks
from. With L2.5 in place, BFS can frontier-expand from a handful of
high-precision entity matches rather than from BM25's noisy top-K.

It is also the first retrieval tier that will admit posterior-
weighted ranking on day one without re-architecting — the entity
overlap score and the BM25 score sit alongside `α / (α + β)` in the
final combiner, rather than the combiner having to invent the L2.5
score from scratch.

For the v1.4.0 context rebuilder, L2.5 means the rebuilder can recover
recent beliefs that share identifiers with the session tail — the
rebuilder's "highest-value beliefs against the session tail" query is
exactly the kind of identifier-heavy query L2.5 handles best.
