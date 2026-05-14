# Feature spec: Ingest speaker-gate + sentiment routing + SVO length floor (#785)

**Status:** spec (no implementation yet)
**Issue:** #785
**Substrate prereqs:** #606 (sentiment hook, shipped), #290 (retention class column, shipped v1.6.0)

---

## Purpose

Belief ingest currently accepts every archived message as a candidate for belief creation, regardless of speaker, content length, or sentiment classification. The result is a self-reinforcing feedback loop that pollutes the most-retrieved stratum with agent self-narration, user feedback utterances, and prose fragments — none of which carry the load BM25 + posterior reranking expects from a "belief."

This spec closes the loop with three coordinated additions across `transcript_logger.py`, `ingest.py`, and `hook.py`. The fix is purely subtractive on the ingest path; no retrieval-side or scoring-side change is required.

---

## The feedback loop being closed

```
agent narrates progress
    ↓
transcript_logger ingests narration as a belief row
    ↓
next user turn says "good" / "yeah keep going"
    ↓
sentiment hook (#606) bumps α on the recently-retrieved narration belief
    ↓
narration posterior rises; BM25 ranks it higher next turn
    ↓
agent sees its own old narration as "context" and continues the pattern
```

Empirical attribution of short, reinforced beliefs (length < 80 chars, α+β ≥ 10):

| Source | Share | Closed by |
|---|---|---|
| Agent self-narration (status / tool output / progress lines) | ~51% | §1 speaker-attribution gate |
| User direction / sentiment utterances | ~30% | §2 sentiment → feedback_history routing |
| SVO-extractor sub-floor prose fragments | ~19% | §3 min-length floor + edge-anchor demotion |

---

## §1. Speaker-attribution gate (`transcript_logger.py`)

### Contract

At the boundary between hook-payload reception and the spawned ingest call:

- For each archived message, inspect `role`.
- If `role == "assistant"`, **do not pass the message to belief creation.** The message may still feed edge construction (citation / reference links between user-turn beliefs).
- If `role == "user"`, behavior is unchanged.

The decision is per-message, not per-archive. A mixed-role archive routes user messages normally and skips assistant messages.

### Why this is load-bearing

Agent narration dominates the short-reinforced-belief leak. The moment the agent stops ingesting its own narration, ~51% of the leak disappears with zero false positives — assistant role is a structural attribute of the payload, not a heuristic.

### Pre-existing rows

This change does NOT retroactively delete pre-existing assistant-role belief rows. A separate stratum-aware cleanup campaign handles the back-purge once this gate is in place upstream (so cleanup doesn't refill).

### Tests

- `test_transcript_logger_skips_assistant_role_for_belief_creation` — archive with mixed user/assistant turns; assert only user-role content reaches the ingest entrypoint.
- `test_transcript_logger_assistant_role_edges_still_construct` — assert citation edges between user-turn beliefs are still built when assistant content references them.

---

## §2. Sentiment → `feedback_history` routing (`hook.py` ↔ ingest)

### Contract

When a UserPromptSubmit's text matches a sentiment pattern (per the #606 classifier) *and* the same text would otherwise become a belief via the post-compaction ingest path:

- The sentiment match continues to bump `feedback_history` on prior beliefs (existing #606 behavior, unchanged).
- Belief creation is suppressed for that message text.

### Implementation choice — single source of truth

Two structural options were considered:

| Option | Description | Trade-off |
|---|---|---|
| **A. Sentiment-first (recommended)** | Sentiment classifier runs before ingest archives the user message. On match, mark the archived entry as feedback-only. Ingest reads the mark and skips belief creation. | Classifier stays authoritative in one place. |
| B. Ingest-side mirror | Re-run the sentiment classifier inside `ingest.py` on each user-role message and skip belief creation on match. | Simpler wiring, but duplicates the classifier. |

**Adopt Option A.** A single source of truth on sentiment classification keeps classifier-update churn contained to the hook surface.

### Why this is needed

After §1 lands, the user-side `"yeah keep going"` / `"good"` / `"no that's wrong"` utterances still enter both lanes: feedback_history (correct) and beliefs (wrong). §2 closes the second leak. Without §2, fixing §1 alone leaves ~30% of the bloat in place.

### Tests

- `test_sentiment_match_suppresses_belief_creation` — UPS `"yeah keep going"` with a prior belief in scope; assert `feedback_history` bumped on the prior belief, no new belief row created.
- `test_sentiment_nonmatch_creates_belief_as_before` — UPS `"the auth refactor needs to handle PKCE flow"`; assert belief created (no sentiment match, behavior unchanged).
- `test_sentiment_classifier_authoritative_at_hook` — assert ingest does not invoke the sentiment classifier directly.

---

## §3. SVO-extractor min-length floor + edge-anchor demotion (`ingest.py`)

### Contract

On the triple-extraction emission path, introduce a content-length floor:

```python
MIN_BELIEF_CONTENT_CHARS: Final[int] = 80
```

For each candidate belief produced from a triple's subject or object slot:

- If `len(candidate.content.strip()) >= MIN_BELIEF_CONTENT_CHARS`, the candidate becomes a belief row as before.
- If `len(candidate.content.strip()) < MIN_BELIEF_CONTENT_CHARS`, the candidate does NOT become a freestanding belief row. Instead, the clause attaches as `anchor_text` on an edge between the surrounding full-length beliefs.
- If neither the subject nor the object slot resolves to a ≥ `MIN_BELIEF_CONTENT_CHARS` belief, the entire triple is rejected. Unanchored sub-floor prose has no graph place.

### Why 80

Empirical: the short-reinforced stratum bottoms out around 80 characters. Below this, content is almost exclusively code-fence prefixes (`"```bash"`), bullet stubs (`"- "`-prefixed fragments), section headers ending in `:`, and tool-output echoes. Above 80, content is overwhelmingly real docstring / statement / claim.

The floor is a module constant, not a config knob — there is no expected operator override and the value should move only by re-measurement.

### Why edge-anchor demotion rather than full reject

A sub-floor clause linking two full-length beliefs *does* carry relational meaning ("X → Y"). Attaching it as edge `anchor_text` preserves the relation without inflating the belief-row count. This keeps BFS traversal informative on the relation while removing the clause from BM25 / posterior reranking.

### Tests

- `test_ingest_minlen_demotes_short_to_edge_anchor` — paragraph whose triple-extraction produces `(headerA, "→", bulletStart)` where `bulletStart` is sub-floor; assert no new belief row created; edge between two full-length parents carries the clause as `anchor_text`.
- `test_ingest_minlen_rejects_unanchored_triple` — sub-floor content in both subject and object slots; assert no belief rows created, no edges created, triple silently dropped.
- `test_ingest_above_floor_unchanged` — full-length content; assert belief row created as before, edge construction unchanged.

---

## Determinism property

All three changes preserve the contract that *same payload → same belief set, byte-for-byte*. None of them introduce non-determinism, embedding lookups, LLM calls, or wall-clock-dependent behavior. PHILOSOPHY (#605) is preserved.

---

## Non-decisions (intentionally out of scope)

- **No SQL DELETE on the existing belief corpus.** Blind purge of pre-existing short-reinforced rows is wrong — separate evidence DROPPED several purge filters during the analysis that led to this spec. The back-purge is a stratum-aware cleanup campaign once §1–§3 are in place upstream (so cleanup doesn't refill).
- **No ambiguity classifier yet** for the "no yeah that makes sense" case (affirmation prefixed by negation). 0/100 in the analyzed sample. Defer until production data post-fix shows it matters.
- **No retrieval-side top-K cap change.** Originally proposed in an earlier conversation; out of scope here. Fix the input side first, re-measure top-K on the cleaner data, then decide whether to change it.
- **No retention-class re-classification for assistant-role rows.** The pre-existing assistant rows keep their current retention_class until the back-purge campaign runs.

---

## Acceptance (closes #785)

- [ ] `transcript_logger.py` skips assistant-role messages for belief creation; edge construction path unchanged.
- [ ] `hook.py` marks sentiment-matched UPS text as feedback-only before ingest archives the message; ingest reads the mark.
- [ ] `ingest.py` defines `MIN_BELIEF_CONTENT_CHARS = 80` as a module constant; sub-floor candidates demote to edge `anchor_text` or are rejected per §3 contract.
- [ ] All six unit tests in §1, §2, §3 pass.
- [ ] Full pytest suite green (no regression in unrelated ingest paths).
- [ ] Determinism unchanged: same payload → same belief set across two ingest runs.
