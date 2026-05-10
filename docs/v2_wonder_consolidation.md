# v2.0 research spec: wonder-consolidation generation strategy bake-off

Spec for issue [#228](https://github.com/robotrocketscience/aelfrice/issues/228). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B). Companion to [`v2_phantom_promotion_trigger.md`](v2_phantom_promotion_trigger.md) (#229) — that issue specifies the rule applied to the phantoms this issue's strategy generates.

Status: lifecycle and dispatch shipped at v2.1+. `src/aelfrice/wonder_consolidation.py` is wired; the `aelf:wonder` MCP tool and the `aelf wonder --axes` CLI verb both ship. The generation-strategy bake-off (RW / TC / STS / ensemble) below is still the open question — gated on lab pre-registration close.

## What's being decided

Which `wonder` offline-generation strategy ships at v2.0 — random walk (RW), triangle closure (TC), span-topic sampling (STS), or a budget-split ensemble. v1.x has no `wonder` surface; v2.0 introduces the speculative-belief lifecycle. The strategy choice is upstream of every other `wonder` decision (lifecycle, retrieval, gc).

## Status of upstream work

**Lab pre-registration:** `aelfrice-lab/experiments/wonder-consolidation/HYPOTHESES.md`, dated 2026-04-28. Predictions H0–H6 written before any code runs. R0 (synthetic 200-atom corpus + feedback simulator + evaluator) is defined but not built.

**Substrate context:** #196 ratified Option B (single-axis Beta-Bernoulli). The lab pre-registration uses scalar `α/β` confirmation thresholds (`α ≥ 12` promotion gate). Substrate-compatible.

**Dependencies:**

- #195 `scoring.uncertainty_score(α, β)` — input to RW seed selection ("uncertainty score > 0.7 atoms"). Spec exists; ports as scalar per #196.
- #194 `bulk=` parameter to `ingest_turn` — needed to load the 200-atom synthetic corpus efficiently. Mechanical add.

Both are unblocked by #196 ratification.

## The three candidate strategies

Verbatim from lab HYPOTHESES.md § Question:

1. **Random walk (RW).** Start from a high-uncertainty atom, walk N hops on the typed-edge graph, bundle visited atoms.
2. **Triangle closure (TC).** Find pairs (A, B) where A→C and B→C both exist with edge type ∈ {SUPPORTS, CITES, RELATES_TO}; propose (A, B) keyed on shared target C.
3. **Span-topic sampling (STS).** Sample compositions whose constituents span topics or sessions (max session-id diversity in the no-embedding form).

Lab predictions, summarized:

| Strategy | H confirmation rate | Predicted strength |
|----------|---------------------|--------------------|
| H0 null (uniform random pair) | 5–8% | floor every strategy must beat |
| H1 RW depth 2 | 15–20% | wins on diversity, loses on rate |
| H2 TC | 25–35% | highest density on small corpora |
| H3 STS | 10–15% (predicted) | wins cross-topic, loses raw rate |

These are predictions, not measured results. The campaign exists to falsify or confirm them.

## Adoption criteria for v2.0 ship

Per lab § What success looks like, with the public-side decision rules made explicit:

1. **Single-strategy ship if:** one strategy beats the H0 null floor by ≥10pp on confirmation rate AND has retrieval-frequency-per-construction-cost within 25% of the best, AND the others are not complementary on phantom-overlap (Jaccard ≥ 0.6 between any two strategies' phantom sets).
2. **Budget-split ensemble ship if:** the top two strategies have phantom-set Jaccard < 0.3 (genuinely complementary) AND each clears the H0+10pp floor independently.
3. **Defer the line item if:** no strategy clears H0+10pp by R<final>, or junk rate > 60% (wonder_gc cleanup dominates the run). The `wonder` v2.0 line item then ships without offline generation; on-line wonder-prompted generation only.
4. **Drop the line item if:** all three strategies fall below the H0 floor. This would falsify the entire offline-generation premise.

## Decision asks

These commit the public side regardless of campaign outcome:

- [ ] **Adopt the lab pre-registration as the public research spec.** No re-pre-registration; the campaign closes against `aelfrice-lab/experiments/wonder-consolidation/HYPOTHESES.md`.
- [ ] **Confirm the four adoption criteria above.** Specifically the H0+10pp floor and the Jaccard thresholds. These are public commitments — relax them later only with a written rationale.
- [ ] **Confirm the campaign owner is Gylf or the next claimant of #228.** Mutex via the `Gylf` label is in place.
- [ ] **Confirm scope: this issue is generation strategy only.** Promotion rule lives in #229. Lifecycle (`wonder_ingest`, `wonder_gc`) is a separate sub-issue not yet filed. Retrieval surface for phantoms is a separate sub-issue not yet filed.
- [ ] **`junk_rate` definition.** Lab specifies "wonder_gc cleanup dominates" qualitatively. Public spec needs a quantitative definition: junk_rate = (phantoms gc'd within 14d) / (phantoms generated). Threshold 60% per criterion 3 above.

## Out of scope

- Promotion rule (#229).
- Lifecycle pieces (`wonder_ingest`, `wonder_gc`) — separate v2.0 sub-issue.
- Retrieval-side surfacing of phantoms — separate.
- The `reason` line item — operates on accepted beliefs, not phantoms.

## Provenance

Lab pre-registration: `aelfrice-lab/experiments/wonder-consolidation/HYPOTHESES.md` (2026-04-28).
Lab beliefs cited in the issue body: `c7202e80`, `85951e1d`, `2677f4f2`, `0ebc73d68a`.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B). Substrate-compatible; `α ≥ 12` promotion threshold uses scalar form.
