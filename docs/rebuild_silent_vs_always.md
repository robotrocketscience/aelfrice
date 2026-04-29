# Rebuild silent-vs-always contract + relevance floor (#289)

## Status

**Spec — proposing for ratification.** Doc-only. Ratification unblocks
the `rebuild_v14()` floor + caller contract change. Calibration of the
threshold value `T` blocks on #288 (eval harness — spec landed in
#294); this memo specifies the *shape* of the floor, not its value.

## Why this comes before the floor itself

`context_rebuilder.rebuild_v14()` today packs top-K beliefs to the
token budget regardless of relevance. The #281 reproduction surfaced
14 unrelated beliefs (10 "Graceful degradation", 4 "continuation
fidelity") for a session about PR review and `insert_belief`
content-hash dedup. The packing logic worked perfectly; the contract
forced it to.

Two separate decisions are entangled in the failing behaviour:

1. **Caller contract.** Are non-L0 callers prepared for an empty (or
   near-empty) `<retrieved-beliefs>` section? If not, any floor that
   could produce empty output is unshippable.
2. **Floor scope.** Does the floor apply uniformly to L0 / session-
   scoped / L2.5 / L1, or are the locked tiers exempt?

The floor's *value* (`T = 0.4 BM25` vs `T = 0.6 reranker score` vs
something learned) is calibration work and can't be settled until the
eval harness from #288 collects a baseline. But the contract and the
tier-by-tier scope can be settled now and let calibration proceed
against a fixed shape.

## What the rebuilder does today

`src/aelfrice/context_rebuilder.py:240` — `rebuild_v14()`:

1. Pulls L0 locked via `store.list_locked_beliefs()`. Always
   prepended whole, never trimmed.
2. Computes `query` from recent turns
   (`_query_for_recent_turns`, line 546).
3. Calls `retrieve(store, query, token_budget=…)` (line 276) which
   returns L0 + L2.5 + L1 + L3 in a single ordered `list[Belief]`.
   **No per-belief scores are exposed** — `retrieve()` returns
   `list[Belief]`, not `list[(Belief, float)]`.
4. Drops L0 from the retrieved list (it's prepended manually).
5. Appends session-scoped beliefs whose `session_id` matches the
   latest recent turn.
6. Appends remaining non-locked hits in `retrieve()`'s native order,
   deduped by `id` and by `content_hash` (output dedup landed in
   #281's partial fix).
7. Stops when `token_budget` is exhausted.

Empty input today: `retrieve()` returns L0 only when query is empty
or whitespace-only (`retrieval.py:852`). The output block is still
well-formed — `_format_block` (line 667) skips the `<retrieved-
beliefs>` element when `hits` is empty.

The shape of "empty output is well-formed" already exists. What
changes with a floor is the *frequency* and *cause* of empty output:
today empty means "no query"; tomorrow empty means "no candidate
cleared the floor."

## Recommendation summary

1. **Caller contract: empty-output is part of the contract.**
   Today's three callers (`UserPromptSubmit` hook, PreCompact hook,
   `aelf rebuild` CLI) already tolerate empty `<retrieved-beliefs>`
   because `_format_block` already emits it that way for the
   no-query case. No behavioural change needed in the callers; the
   spec change is making this load-bearing.
2. **Floor scope: BM25 / L1 only. L0 and session-scoped exempt.**
   Locked beliefs are user-asserted ground truth and inject whenever
   matched; session-scoped beliefs are recency-driven and exempt for
   different reasons (see § Tier exemptions). L2.5 is a closer call
   and is treated case-by-case.
3. **Score-aware retrieval contract.** `retrieve()` becomes
   `retrieve_scored()` (or grows a `return_scores=True` kwarg)
   returning `list[ScoredBelief]` with the final composite score
   (BM25 + posterior weight already log-additive at
   `retrieval.py:843`). Existing `retrieve()` keeps its signature
   for backwards compatibility, layered on top of the scored variant.
4. **Empty-marker attribute, not a separate element.** The empty
   case is signalled by `<aelfrice-memory floor_active="true"
   n_below_floor="N">` (or similar) so the consuming agent can tell
   "the system tried and rejected" from "the system didn't try". No
   new element shape; just an attribute.
5. **The `last_retrieved_at` stamp (#266) fires on candidates,
   not survivors.** Querying a belief and rejecting it as
   below-floor is still evidence the belief was *considered*; the
   stamp is "did this belief enter the candidate set," not "did it
   survive packing."

Mitigations declined:

- **Adaptive per-query thresholds** ("normalize by mean score in this
  query"). Adds a hidden coupling between the floor and the rest of
  the candidate set — a single dominant high-scoring candidate
  raises the bar for everything below, which is the opposite of the
  fairness property we want. Single absolute threshold is the right
  shape for v1.x; adaptive can be re-evaluated in v2.x.
- **Different floors per tier (L1 vs L2.5).** Out of scope for this
  memo — locks one knob in for now (L1 only). If L2.5 turns out to
  produce off-topic injections in the field, that's a follow-up with
  its own calibration pass.

## Detailed proposal

### 1. Caller contract — what "empty is OK" actually means

Three call sites today:

- `src/aelfrice/hook.py:592` — UserPromptSubmit hook. Calls
  `rebuild_v14`, captures the returned block, writes it to stdout.
  An empty `<retrieved-beliefs>` section already passes through
  unchanged.
- PreCompact hook path (same module) — wraps the block in the
  `emit_pre_compact_envelope` (line 329). Envelope is independent
  of belief count; empty inner section round-trips fine.
- `src/aelfrice/cli.py:779` — `aelf rebuild` CLI. Prints the block
  to stdout. Operator-facing; empty section is a legitimate "the
  store didn't have anything relevant" answer.

**Decision ask:** confirm that callers' obligation under the new
contract is *"do not assume `<retrieved-beliefs>` is non-empty,
including under non-empty queries."* This is already true today by
inspection of `_format_block`; the spec change is promoting it from
incidental to load-bearing.

The corresponding test obligation: `tests/test_context_rebuilder.py`
gains a "non-empty query, all candidates below floor → block has no
`<retrieved-beliefs>` element" case. Existing tests that assert on
specific belief content do not need to change; they're about the
non-empty path.

### 2. Floor scope — tier exemptions

| Tier | Floor applied? | Reason |
|---|---|---|
| L0 (locked) | No | User-asserted ground truth. If a lock matches the query at all, it ships. |
| Session-scoped | No | Recency-driven, not relevance-driven. The session-scoping invariant is "what was just discussed should stay visible," even if BM25 doesn't love it. |
| L2.5 (entity index) | Conditional | If the entity match is direct (token equality with a query entity), exempt. If the entity comes from BFS expansion or surface-form fuzzy match, apply the floor. |
| L1 (BM25 / FTS5) | **Yes — primary target** | The duplicate-graceful-degradation reproduction in #281 was an L1 failure: BM25 had no other candidates with high scores, so weak hits filled the budget. |
| L3 (BFS expansion) | Yes — same floor as L1 | BFS scores are already comparable in scale (path-product over the same posterior-weighted edges); reuses L1's `T`. |

Decision ask: confirm L0 + session-scoped exempt; L1 + L3 floored;
L2.5 floored only when the entity match is non-direct. Reject if a
flatter scope (floor-everything) is preferred.

### 3. Score-aware retrieval contract

Today `retrieve()` returns `list[Belief]`, no scores. The floor
needs scores. Two options:

| Option | Shape | Tradeoff |
|---|---|---|
| **A — `return_scores=True` kwarg** | `retrieve(..., return_scores=True) -> list[ScoredBelief]` (`Belief` + `final_score: float` + `tier: Literal["L0","L25","L1","L3"]` + `dropped_reason: str \| None`). Default `False` returns existing shape. | Backward-compatible; one entry point; kwarg-coupled return type is mildly ugly. |
| **B — separate `retrieve_scored()`** | New module-level function; existing `retrieve()` becomes a one-liner over it that strips scores. | Cleaner type signature; extra public symbol; everyone migrating to scored adds an import. |

**Recommendation: Option B.** Cleaner type story matters more than
one-extra-import. `retrieve()` keeps its existing signature; the
rebuilder migrates to `retrieve_scored()`; future callers (eval
harness #288, audit script #294 phase 1c) likely also want scores
and benefit from the dedicated entry point.

`ScoredBelief`:

```python
@dataclass(frozen=True, slots=True)
class ScoredBelief:
    belief: Belief
    tier: Literal["L0", "L25", "L1", "L3"]
    final_score: float
    components: dict[str, float]  # {"bm25": …, "posterior": …, …}
    floor_decision: Literal["packed", "below_floor"] | None
```

`floor_decision` is filled in by the rebuilder, not by
`retrieve_scored()` — the rebuilder owns the threshold choice.
Returning the components separately matters for the audit script
in #294 phase 1c (drop-reason distribution).

Decision ask: confirm Option B.

### 4. Empty-marker attribute

When the rebuilder produces a block whose `<retrieved-beliefs>`
section would be empty *because of the floor* (as distinct from
empty-because-no-query), the outer envelope grows two attributes:

```
<aelfrice-memory floor_active="true" n_below_floor="3">
  <recent-turns>…</recent-turns>
  <continue/>
</aelfrice-memory>
```

`floor_active="true"` says "the system retrieved candidates and
rejected all of them as below-floor." `n_below_floor` is the
count of candidates that scored but didn't make the cut. The
combination tells a downstream agent "memory was searched and
came up dry on relevance," which is meaningfully different from
"memory wasn't searched."

Cases:

- Empty query: no attributes (existing behaviour).
- Non-empty query, store empty: no attributes.
- Non-empty query, all candidates below floor:
  `floor_active="true" n_below_floor="N"`.
- Non-empty query, ≥1 candidate above floor:
  `<retrieved-beliefs>` element present, no attributes on outer
  tag (existing behaviour).

Decision ask: confirm the attribute names + the four-case truth
table. Reject if a separate sentinel element
(`<aelfrice-memory><below-floor n="3"/></aelfrice-memory>`) is
preferred — that has a slightly different parse story for
downstream agents.

### 5. `last_retrieved_at` interaction (#222 / #266)

The hook-driven retrieval `last_retrieved_at` stamp (#266) currently
fires on every belief that comes back from `retrieve()`. Under the
floor, "comes back from retrieve_scored()" = "entered the candidate
set," which includes below-floor candidates.

Option: stamp on candidates (current semantic) vs stamp only on
packed (post-floor survivors).

**Recommendation: stamp on candidates.** A below-floor hit is still
evidence the belief was considered relevant enough to score; that's
the signal `last_retrieved_at` is for. Stamping only on survivors
would make the stamp track "what made it past the floor" rather than
"what was searched," which is a meaningful semantic shift the
downstream consumers (decay, classify-orphans #253) probably don't
want.

Decision ask: confirm "stamp on candidates, not on packed." Reject
if a stricter "packed-only" stamp is preferred (and accept the
decay-pressure shift that implies).

### Out of scope

- **The actual `T` value.** Calibration work; depends on the eval
  harness from #288 / #294 phase 1b collecting a week of operator
  data. This memo specifies the floor's *shape*; ratification of
  the value is a separate decision.
- **Adaptive / per-query / learned floors.** Single absolute
  threshold for v1.x; adaptive variants are a v2.x evaluation
  question.
- **Different floors per tier within L1.** L1 + L3 share `T`. If
  the data shows L3-from-BFS produces qualitatively different
  off-topic injections, that's a follow-up.
- **Output-stage content_hash dedup.** Already in flight via #281
  partial / PR #293.
- **Reranking with richer signal.** Stacks on top of a floor in a
  later PR; not a substitute for the floor.
- **`<aelfrice-baseline>` (SessionStart) treatment.** The baseline
  block doesn't run the v14 retrieval path; it's a separate code
  surface in `hook.py`. Out of scope here; same floor logic *could*
  apply but is a follow-up if `<aelfrice-baseline>` shows the same
  off-topic-flood failure mode in the field.

## Decision asks (consolidated)

- [ ] **Caller contract.** Confirm callers tolerate empty
  `<retrieved-beliefs>` under non-empty queries; spec promotes
  this from incidental to load-bearing.
- [ ] **Tier exemptions.** Confirm L0 + session-scoped exempt; L1
  + L3 floored; L2.5 floored conditionally on non-direct entity
  match.
- [ ] **Scored retrieval shape.** Confirm Option B — separate
  `retrieve_scored()` function returning `list[ScoredBelief]`.
- [ ] **Empty-marker attribute.** Confirm `floor_active="true"
  n_below_floor="N"` on the outer `<aelfrice-memory>` element.
- [ ] **`last_retrieved_at` stamp scope.** Confirm "stamp on
  candidates" — below-floor hits still bump the stamp.

## Implementation tracker (post-ratification)

Roughly two PRs:

1. **`retrieve_scored()` extraction.** New function alongside
   `retrieve()`; existing `retrieve()` reduced to a thin wrapper.
   New `ScoredBelief` dataclass. Tests covering score parity with
   the unscored path. ~250 lines net incl. tests. No behavioural
   change for any caller until phase 2 lands.
2. **Floor + empty-marker.** `rebuild_v14()` migrated to
   `retrieve_scored()`; floor applied per § Floor scope; empty-
   marker attributes added; tests for the four-case truth table +
   tier exemption matrix. `T` defaulted to a placeholder
   (`AELFRICE_REBUILD_FLOOR=…` env var, `[rebuilder]
   relevance_floor` TOML key) with the calibration-driven value
   landing in a follow-up after #288 phase 1b. ~350 lines net.

`docs/context_rebuilder.md` gains a "Relevance floor" section
referencing this memo. `LIMITATIONS.md` notes the residual risk:
a calibration error in `T` either over-suppresses (legitimate
beliefs dropped) or under-suppresses (preserves the #281 failure).
The eval harness is the corrective loop.

## Provenance

- Parent: #286 (rebuild redesign scoping).
- Sibling specs:
  - `docs/rebuild_eval_harness.md` (#288 / merged via #294).
  - `docs/belief_retention_class.md` (#290 / merged via #296).
- Symptom evidence: #281 (10 + 4 duplicate off-topic hits).
- Calibration dependency: #288 — `T` is unfalsifiable without it.
- Code references:
  - `src/aelfrice/context_rebuilder.py:240` — `rebuild_v14()`.
  - `src/aelfrice/context_rebuilder.py:667` — `_format_block()`.
  - `src/aelfrice/retrieval.py:805` — `retrieve()`.
  - `src/aelfrice/retrieval.py:843-850` — posterior-weighted
    composite scoring (the score this floor consumes).
  - `src/aelfrice/hook.py:592` — UserPromptSubmit caller.
  - `src/aelfrice/cli.py:779` — `aelf rebuild` CLI caller.
