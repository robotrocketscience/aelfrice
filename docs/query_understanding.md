# Query understanding for rebuild (#291)

## Status

**Spec — proposing for ratification.** Doc-only. Implementation
blocks on this memo + #288 logs (for query-strategy comparison
data) before any default change ships.

## Problem

`context_rebuilder._query_for_recent_turns()` (line 534) builds the
`retrieve()` query from concatenated recent-turn text via:

1. Entity extraction (file paths, identifiers, error codes).
2. Triple-subject/object extraction.
3. Stopword-token fallback when neither extractor finds structure.

All three steps treat every turn equally. A 10-turn window includes:

- The actual operator question (1-2 turns).
- System-reminder text from the harness.
- Tool-output dumps (file contents, search results).
- Prior assistant prose (which may discuss tangentially-related
  topics).

#281 caught the consequence: an operator session focused on PR
review and `insert_belief` content_hash dedup surfaced beliefs about
PreCompact hooks, ancillary transcript paths, and vsa-logic R6/R10.
Those topics appeared in *adjacent context*, not in what the
operator was actually working on.

The same bag-of-entities approach is what the L2.5 entity-index
lookup consumes too — fixing this fixes both retrieval lanes
simultaneously.

## Recommendation summary

- **Ship a deterministic-only redesign first; defer LLM-in-the-loop.**
  Three deterministic improvements stack: turn-recency weighting,
  role-source weighting, intent classification by surface-pattern.
  All run in <5 ms, no model dep, no API call.
- **LLM-assisted query summarisation is parked.** The
  determinism / zero-dep / latency commitments are load-bearing for
  the hook surface (#280) and the doctor pipeline. A local-LLM lane
  can reopen as a v2.x spec memo if deterministic improvements
  plateau on #288 calibration data.
- **Filter-then-rank, not rank-only.** Adjacent-context filters
  (system-reminder turns, tool-output blocks) drop *before* the
  query is constructed, not after BM25. Saves the floor (#289) from
  having to reject everything; saves the ranker from poisoning by
  irrelevant tokens.
- **Negative signals deferred.** "I'm not working on X anymore" is a
  v2.x feature that needs operator-driven UX (CLI flag, slash
  command). Out of scope here.
- **All three improvements ship behind feature flags.** Lets the
  #288 harness compare strategy-on vs strategy-off cleanly.

## Detailed proposal

### 1. Turn-recency weighting

Split recent turns into three buckets by recency:

| Bucket | Range | Weight |
|---|---|---|
| `latest` | last 1-2 turns | 1.0 |
| `mid` | turns 3-5 | 0.5 |
| `tail` | turns 6+ | 0.2 |

Entities and triple phrases extracted from each bucket are tagged
with their bucket weight. Duplicates across buckets keep the *max*
weight (an entity that appears in both `latest` and `tail` is
treated as `latest`).

The query passed to `retrieve()` is unchanged in shape (still a
space-joined string), but extracted with weights applied to
post-extraction deduplication: ties between equally-novel terms
break toward the higher-weight bucket. For BM25, weight is
expressed by *repetition* — terms with weight 1.0 appear once,
terms with weight 0.5 appear in the query string only if no
weight-1.0 terms cover the same retrieval target, terms with weight
0.2 appear only as fallback.

Concretely, the new helper signature:

```python
def _query_for_recent_turns_v2(
    recent_turns: list[RecentTurn],
    *,
    max_terms: int = DEFAULT_QUERY_ENTITY_CAP,
) -> str:
    weighted = _extract_weighted(recent_turns)  # list[(term, weight)]
    weighted.sort(key=lambda x: -x[1])
    selected = _dedup_keep_first(weighted)[:max_terms]
    return " ".join(term for term, _ in selected)
```

Determinism preserved: same `recent_turns` → same output.

**Decision asks:**

- [ ] **Three-bucket recency weighting.** Confirm bucket
  boundaries (1-2 / 3-5 / 6+) and weights (1.0 / 0.5 / 0.2).
  Reject if a continuous decay is preferred (and accept the
  parameter tuning surface).
- [ ] **Max-of-bucket on duplicates.** Confirm.

### 2. Role / source filtering

`RecentTurn.role` is `"user"` or `"assistant"` (line 203). Today
both contribute equally to query construction. The proposal:

| Role | Behavior |
|---|---|
| `user` (operator-typed) | Always included, full weight. |
| `assistant` (model output) | Included at half weight. Assistant prose tracks the operator's topic but adds noise. |

Filter *out* of query construction entirely:

- **System-reminder spans** within user-role turns. The harness
  prepends `<system-reminder>...</system-reminder>` blocks. Strip
  these from the text passed to extract_entities. They're not
  operator-authored.
- **Tool-output blocks** within assistant-role turns.
  `<tool_use_result>...</tool_use_result>` and the file/grep dumps
  inside them. These contain code identifiers that aren't the
  operator's topic — they're the *means* of investigating the
  topic.
- **Hook-output blocks** within user-role turns.
  `<aelfrice-memory>...</aelfrice-memory>` spans (whether well-formed
  or escaped per #280). The hook's own retrieved beliefs are
  feedback, not query input. Self-referential inclusion is the
  bug.

Implementation: a single pre-extraction strip pass over each turn's
text:

```python
_STRIP_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<aelfrice-memory>.*?</aelfrice-memory>", re.DOTALL),
    re.compile(r"<aelfrice-baseline>.*?</aelfrice-baseline>", re.DOTALL),
    re.compile(r"<tool_use_result>.*?</tool_use_result>", re.DOTALL),
)

def _strip_adjacent_context(text: str) -> str:
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text)
    return text
```

Closed pattern list (same approach as the hook escape in #280).
Adding a fifth pattern is a one-line code change once a new
adjacent-context tag earns inclusion.

**Decision asks:**

- [ ] **Assistant role at half weight.** Confirm. Reject if
  assistant turns should be fully included (and accept the
  noise) or fully excluded (and accept losing context where the
  assistant correctly summarised the operator's intent).
- [ ] **Strip pattern list.** Confirm the four patterns above.
  Reject specific patterns or add to the list.

### 3. Intent classification (deterministic)

Three intent classes inferred from surface patterns in the
`latest` bucket only:

| Intent | Surface pattern | Retrieval bias |
|---|---|---|
| `question` | Latest user turn ends with `?` OR contains `how`, `why`, `what`, `where`, `when` as a leading word | Boost L0 + L2.5; broader BM25 |
| `action` | Latest user turn contains imperative verbs (`fix`, `add`, `update`, `remove`, `refactor`, `rename`) | Boost L1 BM25; tighter floor |
| `context-switch` | Latest user turn contains `now`, `next`, `instead`, `actually` as opening | Down-weight `mid` and `tail` buckets to 0.0; effectively ignore older turns |
| `unknown` | None of the above match | Defaults: equal weighting per §1 |

Intent affects:

- **Bucket weights.** `context-switch` zeroes `mid` and `tail`.
- **Floor (#289) per-class.** `question` uses `T_l1 = 0.30`
  (broader); `action` uses `T_l1 = 0.50` (tighter); others use the
  default `0.40`.

Closed pattern list. Surface-pattern classification means a
deterministic 3-line regex match — no model call, no embedding,
sub-millisecond cost.

The intent label is included in the rebuild log (#288 schema)
under the `extracted_intent` field so the harness can audit
classification quality.

**Decision asks:**

- [ ] **Three intent classes plus `unknown`.** Confirm.
- [ ] **Surface-pattern classification only.** Confirm
  deterministic regex match. Reject if a learned classifier or
  LLM judge is preferred (and accept the determinism / dep cost).
- [ ] **Per-intent floor adjustments.** Confirm `question = 0.30`,
  `action = 0.50`, `context-switch / unknown = default`. Reject
  specific values; the *shape* (per-intent floor multiplier on
  #289's base T) is what's being ratified.

### 4. Feature flags

Each improvement ships behind a `[query_understanding]` config
block:

```toml
[query_understanding]
recency_weighting = true   # §1
adjacent_strip = true      # §2 (strip patterns)
assistant_half_weight = true  # §2 (role weighting)
intent_classify = true     # §3
```

Defaults: all `true`. Operators can flip individually. The #288
harness uses these flags to A/B compare strategy-on vs
strategy-off on the same query corpus, producing the
quantitative evidence #291's calibration block calls for.

**Decision ask:**

- [ ] **Default-on with operator-tunable flags.** Confirm. Reject
  if any flag should be opt-in (note which + why).

### 5. LLM-assisted lane (parked, not deferred-with-shape)

The issue body raises "small local LLM call to summarise the topic
of recent turns in 1 sentence then BM25 against that sentence."
This memo *parks* that lane — does not specify it.

Reasons:

1. **Determinism load-bearing.** The hook hardening memo (#280)
   relies on the rebuild output being a deterministic function of
   inputs for audit-log replay. An LLM-summarisation step breaks
   reproducibility, which broadens the hook-surface threat model
   in ways out of scope here.
2. **Local LLM dep is non-trivial.** The closest existing dep is
   the noise filter (`noise_filter.py`) which uses lightweight
   heuristics, not a model. Adding a local-model dep changes the
   install profile (model download, memory budget, model version
   matrix). That's a project-level decision, not a query-strategy
   decision.
3. **Deterministic improvements should be benchmarked first.**
   #288 logs let the deterministic three-stack be measured before
   the LLM lane is opened. If §1-§3 plateau below the precision
   target the operator wants, an LLM-assist memo opens with the
   evidence in hand.

The deterministic stack does not preclude the LLM lane; it
complements it. An LLM call could, in a future memo, run *on the
deterministic-stack output* to refine the query, rather than
replacing the stack.

### Out of scope

- **Reranking.** Stacks on top of any query strategy; separate
  spec.
- **Negative signals** ("I'm not working on X anymore").
  Operator-driven UX work, defer until a concrete trigger
  surfaces.
- **G6 / vocabulary-bridge integration (#227).** Different query
  lane (find-similar-to-X) — distinct from prompt-driven rebuild.
- **The query string format passed to `retrieve()`.** Still
  space-joined tokens. Changing the contract between rebuild and
  retrieve is a v2.0 substrate concern.
- **Calibration of intent thresholds and per-intent floor
  multipliers.** Lives in a follow-up PR after #288 collects
  enough labeled data.

## Decision asks (consolidated)

- [ ] **Three-bucket recency weighting** (1-2 / 3-5 / 6+ → 1.0 / 0.5
  / 0.2) with max-of-bucket dedup.
- [ ] **Assistant-role half-weight.**
- [ ] **Adjacent-context strip pattern list** (4 patterns above).
- [ ] **Three intent classes** plus `unknown`, surface-pattern
  classification only.
- [ ] **Per-intent floor adjustments** (shape, not values).
- [ ] **Feature-flag rollout, default-on, operator-tunable.**
- [ ] **LLM-assist lane parked**, deterministic-stack-first.

## Implementation tracker (post-ratification)

Roughly two PRs.

1. **`_query_for_recent_turns_v2`** — recency weighting + adjacent
   strip + role weighting. Behind `[query_understanding]` flags.
   Tests covering: bucket dedup, strip-pattern coverage,
   assistant half-weight, all-flags-off → identical to v1
   behavior. ~350 lines net.
2. **Intent classifier + per-intent floor wiring** — surface-
   pattern regex set + intent → floor-multiplier table +
   integration with #289's floor. Tests covering: each pattern,
   `unknown` fallback, floor adjustment per class. ~250 lines net.

Both PRs land before #289's calibration PR so the harness has all
three strategy components live to compare against. Ratifying
#288, #289, #290, #291 in sequence (any order, but all four before
the calibration follow-ups) is the cleanest path.

## Provenance

- Parent: #286.
- Symptom evidence: #281 (PreCompact / vsa-logic / ancillary
  paths surfacing in unrelated session).
- Adjacent: #288 (eval harness — calibration), #289 (relevance
  floor — consumes per-intent multipliers), #290 (retention class
  — orthogonal axis), #280 (hook hardening — strip patterns reuse
  the same closed list approach), #227 (different query lane).
- Code touchpoints:
  - `src/aelfrice/context_rebuilder.py:534`
    `_query_for_recent_turns`.
  - `src/aelfrice/triple_extractor.py`, `extraction.py` (entity /
    triple extractors — unchanged).
  - `src/aelfrice/retrieval.py` (consumes the query string —
    unchanged).
- Constants: `DEFAULT_QUERY_ENTITY_CAP` (`context_rebuilder.py`)
  remains the cap on max output terms.
