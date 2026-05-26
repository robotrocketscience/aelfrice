# v2.0 evaluation: semantic contradiction detector

Spec for issue [#201](https://github.com/robotrocketscience/aelfrice/issues/201). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B).

Status: spec, no implementation. **Recommendation: ship at v2.0.**

## What's being decided

Whether to port the research-line `relationship_detector.py` (~147 LOC) to v2.0. The module detects contradictions between beliefs by **negation/quantifier signal divergence** rather than by explicit-prose extraction. v1.x has explicit-only via `triple_extractor.py` (catches "X contradicts Y"); the implicit-divergence path is missing, and `contradiction.py`'s docstring acknowledges the gap.

## Substrate dependency

None. Operates on text fields and belief-pair similarity. Output is a relationship verdict + scalar confidence — substrate Option B handles this directly.

## Recommendation

**Ship at v2.0.**

Four reasons:

1. **Closes a public README promise.** Research-line README claimed *"It catches contradictions ... flags them: 'these contradict each other — resolve before proceeding.'"* Public LIMITATIONS.md § Sharp edges (line 36) walks part of that back: *"`CONTRADICTS` edges drive demotion pressure. The regex classifier rarely produces them."* That's a real gap between advertised behavior and shipped behavior. Porting closes it.
2. **Stdlib-only.** Regex over modality (negation, hedging, certainty markers) and quantifiers (always/never/sometimes/often). No LLM, no embedding. Compatible with PHILOSOPHY's determinism property.
3. **Edge type already exists.** `CONTRADICTS` is one of v1.x's four edge types (`models.py`). Downstream consumers (`aelf resolve`, demotion-pressure scoring) are already wired. The detector ships as a write-path hook; no schema migration.
4. **Pairs cleanly with #197 dedup.** Both modules examine candidate pairs after writes. Reuse the same candidate-generation pass (token-overlap on subject + predicate) and split the verdict path: high overlap + agreeing modality → SUPERSEDES (dedup); high overlap + disagreeing modality → CONTRADICTS (this module). Reduces total cost vs porting them as independent passes.

## Decision asks

- [ ] **Confirm port at v2.0.** If yes, scope is ~147 LOC + write-path hook (after `ingest_turn`, `apply_feedback`).
- [ ] **High-confidence threshold for auto-insert.** Research line shipped 0.85. Defaults to 0.85 in v2.0? Lower threshold flags more, requires more `aelf resolve` traffic; higher threshold misses softer contradictions.
- [ ] **Lower-confidence flagging path.** Should low-confidence pairs (verdict = "contradicts" but confidence < threshold) get a separate `POTENTIALLY_STALE` edge or just sit unflagged? `V2_REENTRY_QUEUE.md` has `POTENTIALLY_STALE` listed as a candidate new edge type; this is the natural use case.
- [ ] **Pair with #197 dedup.** Recommendation: ship as a unified "near-pair classifier" pass that produces SUPERSEDES, CONTRADICTS, or REFINES verdicts from the same candidate pool. If preferred separately, both ship at v2.0 with overlapping but distinct write-path hooks.
- [ ] **Quantifier vocabulary.** Research line: always / never / sometimes / often / usually / rarely. Ship as-is? Extend to compound forms ("almost never", "in most cases")?

## Edge case worth flagging

The detector compares modality signals between beliefs that share subject + predicate. False positives surface when two beliefs about the same subject describe *different aspects*: "use uv for Python deps" + "never use pip in this project" both have negation/imperative signals on overlapping tokens but aren't contradicting. Mitigation in the research line: subject-extraction granularity (the predicate-similarity check requires high overlap on the *object*, not just the subject). Ship that, or document the limitation in LIMITATIONS.md and let `aelf resolve` adjudicate?

Recommendation: ship the research-line subject-extraction granularity. The `aelf resolve` surface is the safety net, not the primary defense.

## Downstream impact

- New write-path hook after `ingest_turn` and `apply_feedback`. Cost: O(k) per write where k is the candidate-pair pool size. Same cost shape as the recommended dedup pass; can share the candidate pool.
- LIMITATIONS.md § Sharp edges line 36 ("regex classifier rarely produces CONTRADICTS edges") gets rewritten or removed.
- Closes the open gap in `contradiction.py`'s docstring.
- Pairs with #197 (dedup); both modules can share candidate-generation if shipped together.

## Provenance

Research-line module: `agentmemory/relationship_detector.py` (~147 LOC, stdlib-only).
Lab parity audit: `aelfrice-lab/docs/agentmemory-parity-audit-2026-04-28.md` § 16.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B). Substrate-neutral.
