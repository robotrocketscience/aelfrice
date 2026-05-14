# Triple extractor

**Status:** spec.
**Target milestone:** v1.2.0 (named on the public roadmap as
"triple-extraction port").
**Dependencies:** stdlib only (regex). No LLM. Consumes the
[ingest enrichment](ingest_enrichment.md) schema.
**Risk:** medium. New module surface; correctness depends on the
extraction patterns generalizing beyond the test fixtures.

## Summary

A pure function that reads prose and emits zero or more
`(subject, relation, object)` triples. Each triple becomes an `Edge`
between two beliefs (creating either or both as needed) with the
relation mapped to an `EDGE_TYPES` member and `anchor_text`
populated from the citing prose. Reusable by every v1.x ingest
caller that has prose to extract from — the commit-ingest hook,
transcript-ingest, manual `aelf remember` calls, the v1.3.0
entity-index path.

```python
def extract_triples(text: str) -> list[Triple]:
    """Pure extraction. No store side-effects."""

def ingest_triples(
    store: MemoryStore,
    triples: list[Triple],
    session_id: str | None = None,
) -> IngestResult:
    """Apply extracted triples to the store."""
```

The split keeps extraction testable in isolation and lets callers
inspect / filter triples before they hit the store.

## Motivation

The v1.0 ingest paths (filesystem walk, git log scanner, Python AST)
produce beliefs with content but no relational edges between them.
The retrieval-side techniques planned for v1.3+ (BFS multi-hop,
entity index, augmented BM25F, supersession cascade, posterior-
weighted ranking) all benefit from — or require — typed edges with
non-empty `anchor_text`. The triple extractor is the first ingest
surface that produces those edges from real prose.

The v1.2.0 commit-ingest hook and transcript-ingest are the named
consumers at this milestone, but the extractor is positioned to serve
every prose-ingesting caller that lands later. Building it as a
reusable library rather than embedding it in any one hook is what
keeps that promise.

## Design

### Module: `src/aelfrice/triple_extractor.py`

Two pieces, each independently testable.

#### `extract_triples(text: str) -> list[Triple]`

Pure function. Input is a free-text string (commit message, prose
paragraph, sentence). Output is a list of `Triple` dataclass
instances. No store reference; no side effects.

```python
@dataclass(frozen=True)
class Triple:
    subject: str        # short noun phrase
    relation: str       # one of EDGE_TYPES
    object: str         # short noun phrase
    anchor_text: str    # the substring of `text` the triple was extracted from
```

The extractor matches a fixed set of relational templates and emits
one `Triple` per match. Initial template set:

| pattern | edge type | example |
| --- | --- | --- |
| `X supports Y` / `Y is supported by X` | `SUPPORTS` | "the new index supports faster queries" |
| `X cites Y` / `X mentions Y` | `CITES` | "the proposal cites RFC 8259" |
| `X contradicts Y` / `X disagrees with Y` | `CONTRADICTS` | "the new finding contradicts the earlier paper" |
| `X supersedes Y` / `X replaces Y` | `SUPERSEDES` | "this commit replaces the legacy parser" |
| `X relates to Y` / `X is related to Y` | `RELATES_TO` | "the cache layer relates to retrieval" |
| `X is derived from Y` / `X is based on Y` / `X extends Y` | `DERIVED_FROM` | "the spec is derived from the prior memo" |

Patterns are mechanical (regex + light noun-phrase heuristics).
Implementation can use a small grammar of capitalized words /
dashes / underscores for noun phrases; punctuation and conjunctions
delimit candidate boundaries. No POS tagger, no embedding, no LLM.

`anchor_text` is the literal matched substring (with surrounding
context up to ~80 chars), preserving the citing prose's own
phrasing.

#### `ingest_triples(store, triples, session_id=None) -> IngestResult`

Side-effecting. For each `Triple`:

1. Resolve / create the subject belief (look up by `content_hash`;
   create a new belief if absent).
2. Resolve / create the object belief.
3. Insert an `Edge(src=subject_id, dst=object_id, type=relation,
   weight=1.0, anchor_text=triple.anchor_text)`.

`session_id` (if provided) is written to every newly-created belief
per the [ingest enrichment](ingest_enrichment.md) spec.

```python
@dataclass(frozen=True)
class IngestResult:
    new_beliefs: list[str]      # ids
    new_edges: list[tuple[str, str, str]]
    skipped_duplicate_edges: int
    skipped_no_subject_or_object: int
```

### Resolution policy

- **Subject / object resolution:** content-hash lookup. Same noun
  phrase → same belief id. If the lookup misses, create a new
  belief with `type=BELIEF_FACTUAL`, `lock_level=LOCK_NONE`, and
  the noun phrase as content. Future v1.x callers can override
  the type via a kwarg if context demands it.
- **Edge dedup:** if the `(src, dst, type)` tuple already exists,
  do not insert a second edge. Update `anchor_text` to the new
  value if provided (last-write-wins, simple).
- **Self-edges:** subject == object after resolution → skip.

### Confidence stance

Newly-created beliefs from extraction get the project default
prior, NOT a boosted prior. The triple extractor is mechanical,
non-authoritative, and may extract noise. Downstream `apply_feedback`
calls (or user `aelf lock`) are how a triple-derived belief earns
posterior confidence.

### What this spec does NOT do

- It does NOT decide *when* to extract — that is the caller's
  responsibility (the commit-ingest hook fires on commits;
  transcript-ingest fires per turn; seed-file ingest fires on
  `aelf onboard`).
- It does NOT do co-reference resolution. "the parser" in one
  triple and "the parser" in another both resolve to the same
  belief by content hash; "the parser" and "our parser" do not.
  Coref is a v2.x consideration if production data shows the gap.
- It does NOT extract from structured input (JSON, AST). Those
  callers should produce edges directly via `store.insert_edge()`
  with the relevant `anchor_text`.

## Acceptance criteria

1. `extract_triples("the new index supports faster queries")`
   returns at least one triple with relation `SUPPORTS`.
2. `extract_triples("")` and `extract_triples("no relational
   structure here")` return empty lists.
3. `extract_triples` produces a `Triple` whose `anchor_text` is a
   substring of the input.
4. Each triple's `relation` field is in `EDGE_TYPES` (no extractor
   ever emits an unknown edge type).
5. `ingest_triples` is idempotent: calling twice with the same
   triples produces zero new edges on the second call.
6. `ingest_triples` writes `session_id` on every new belief when
   the kwarg is provided; leaves NULL when omitted.
7. A triple whose subject and object resolve to the same belief id
   is dropped; the count appears in
   `IngestResult.skipped_no_subject_or_object`.
8. `extract_triples("X is derived from Y")` produces a triple with
   relation `DERIVED_FROM`. (Validates that ingest-enrichment's
   `DERIVED_FROM` re-add has a real producer.)

## Test plan

- `tests/test_triple_extractor_patterns.py` — one test per relation
  template (criteria 1, 4, 8). Each fixture is a single sentence
  that triggers exactly one pattern.
- `tests/test_triple_extractor_negatives.py` — empty / non-relational
  / ambiguous inputs (criterion 2, plus "extracts nothing rather
  than something wrong" cases).
- `tests/test_triple_extractor_anchor_text.py` — anchor text is a
  substring of the input (criterion 3).
- `tests/test_ingest_triples_idempotency.py` — idempotency +
  session_id propagation + self-edge handling (criteria 5, 6, 7).
- All deterministic, in-memory `:memory:` store, < 200 ms per test.

## Out of scope

- LLM-based extraction. The v1.3.0 LLM-classification onboard path
  is a separate track; v1.2.0 is mechanical-only.
- Coreference resolution.
- Multi-sentence relations ("X. It supersedes Y.")
- Structured-input extraction (handled by callers via direct
  `insert_edge`).
- Confidence-weighted extraction (every triple is uniformly weighted
  for now; a downstream v1.x patch could weight by pattern
  reliability).
- Cross-language extraction. English only at v1.2.0.

## Open questions

- The pattern set above is a starting point. Real commit messages
  and prose have far more varied phrasings. Should the extractor
  ship with the minimal set and grow via observed false negatives,
  or should the v1.2.0 release include a wider pattern bank?
  Recommendation: minimal set + an `aelf health` check that
  surfaces "X commit messages parsed; Y triples produced" so users
  can see when their corpus needs more patterns.
- Anchor text length cap. The ingest-enrichment spec proposes 1000
  chars; this spec produces ~80 chars per triple. Consistent with
  that cap.
- Should `ingest_triples` create a new session if `session_id` is
  None, or leave it NULL? Recommendation: leave NULL. Session
  creation is the caller's responsibility (the commit-ingest hook
  spec handles that explicitly).
