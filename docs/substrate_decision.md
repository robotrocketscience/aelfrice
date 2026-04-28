# v2.0 substrate decision: multi-axis vs single-axis uncertainty

Spec for issue [#196](https://github.com/robotrocketscience/aelfrice/issues/196). Blocks [#195](https://github.com/robotrocketscience/aelfrice/issues/195) (`scoring.uncertainty_score`); informs [#199](https://github.com/robotrocketscience/aelfrice/issues/199) (enforcement triad), [#193](https://github.com/robotrocketscience/aelfrice/issues/193), [#197](https://github.com/robotrocketscience/aelfrice/issues/197), [#198](https://github.com/robotrocketscience/aelfrice/issues/198), [#201](https://github.com/robotrocketscience/aelfrice/issues/201).

Status: spec, no implementation. **Recommendation included** in § Recommendation; decision is the user's. Implementation paths for both options are spelled out so this memo is the only document that needs to merge before downstream issues unblock.

## What's being decided

aelfrice v1.x scores every belief with a **single-axis** Beta-Bernoulli posterior:

```
posterior_mean = α / (α + β)
```

One `(α, β)` pair per belief. A wrong belief is wrong overall; a useful belief is useful overall. This was made explicit in [`docs/PHILOSOPHY.md § Bayesian, not vector`](PHILOSOPHY.md#bayesian-not-vector) at commit `1afdc04`.

The research-line codebase shipped a **multi-axis** `UncertaintyVector` — per-aspect `(α_i, β_i)` across four dimensions (existence / semantics / mechanism / cost) — in `uncertainty.py` (~242 LOC, stdlib-only). The vector backed `wonder.analyze_gaps`, `voi`, `hibernate`, and `propagate`.

v2.0 must commit one way before `wonder`, `reason`, and the speculative-belief surface can land. This memo locks that commitment.

## The two options

### Option A — port the multi-axis substrate

- **Schema migration.** Add three columns to `beliefs`:
    - `uncertainty_vector TEXT` — JSON-serialised `[[α₁,β₁], [α₂,β₂], [α₃,β₃], [α₄,β₄]]`.
    - `hibernation_score REAL` — soft-suspend score; `NULL` for active beliefs.
    - `activation_condition TEXT` — JSON predicate that re-activates a hibernated belief.
- **Code port.** ~242 LOC from agentmemory `uncertainty.py`. Clean stdlib (`json`, `math`, `dataclasses`); zero coupling to other research-line modules (verified — no imports from `multimodel`, `intention`, `semantic_linker`).
- **Capabilities unlocked.**
    - `wonder.analyze_gaps` answers *"this aspect of this belief is uncertain — research it"* rather than *"this belief is uncertain overall."*
    - `voi(prior, observation)` — value-of-information for a proposed evidence-gathering action. Lets `wonder` rank what's worth investigating next.
    - `hibernate(belief, threshold)` — soft-suspend beliefs whose confidence has decayed below a per-axis threshold; lifecycle distinct from delete.
    - `propagate(source_vector, edge_type)` — uncertainty propagation across graph edges with edge-type-specific decay rates.
- **Backwards compatibility.** `Belief.alpha` and `Belief.beta_param` remain on the model as the *aggregate* projection of the vector — `α = sum(α_i)`, `β = sum(β_i)` — so single-axis consumers (current `partial_bayesian_score`, BFS broker dampening, `apply_feedback` arithmetic) keep working unchanged. Multi-axis is purely additive at the consumption surface.
- **Costs.**
    - Schema migration discipline: new `store.migrate()` step, version-bump to `schema_version = 2`, idempotent re-run.
    - Documentation surface: `PHILOSOPHY.md § Bayesian, not vector` flips from "single-axis only" to "single-axis is the projection of the vector", which is a substantive narrative change. `ARCHITECTURE.md` gains an `uncertainty_vector` section.
    - Benchmark complication: the synthetic-graph harness scores beliefs as scalars; either it stays scalar (vector projects to scalar for benchmarks) or each benchmark grows a per-axis variant.
    - User-visible vector. `aelf show <id>` either hides the vector (then `wonder` becomes a hidden mechanism) or surfaces it (then users must learn the four axes). Neither is free.
    - Defining "uncertain about cost" requires a feedback intake change: `aelf feedback <id> harmful --aspect cost`. That's a new CLI surface and a new MCP tool surface. The research line did this implicitly via LLM classification — bringing that back means re-opening the multimodel question (#198).

### Option B — keep single-axis Beta-Bernoulli, add scalar hibernation

- **Schema migration.** Add one column:
    - `hibernation_score REAL` — soft-suspend score on the existing scalar posterior; `NULL` for active beliefs.
- **Code port.** Zero from `uncertainty.py`. `scoring.uncertainty_score(α, β)` (#195) ports as a scalar Beta-entropy function, ~15 LOC, stdlib `math.lgamma`. `wonder` operates on scalar entropy. No `voi`. No `propagate`.
- **Capabilities lost vs research line.**
    - `wonder.analyze_gaps` becomes *"these beliefs are uncertain overall"* — a coarser filter than per-aspect. Mechanically still useful; semantically less informative.
    - VOI is computable on the scalar (entropy reduction is well-defined for a single Beta), but the comparison axis is lost — VOI tells you *which belief* to investigate, not *which aspect of which belief*.
    - Hibernation is binary (active / hibernated) not per-aspect.
    - Propagation across edges is a single decay rate per edge type, not per-aspect.
- **Costs.**
    - One-paragraph `LIMITATIONS.md` update calling out the deliberate divergence from the research line.
    - `wonder` ships at v2.0 as a coarser product than the research line. Some users perceive this as "wonder is BFS over uncertain beliefs," which is true mechanically.
- **Reversibility.** If v2.x evidence (a benchmark, not a hunch) shows per-aspect uncertainty materially improves `wonder` output, the multi-axis schema migration is purely additive — `uncertainty_vector` and `activation_condition` become new columns, the existing `(alpha, beta)` projection stays as a fallback, and the consumption surface gains a new code path. Going from B to A later costs one migration. Going from A to B later costs deprecating a user-facing concept.

## Recommendation

**Option B.**

Four reasons, ordered by weight:

1. **No benchmark backs the multi-axis claim.** The research line shipped four named axes (existence / semantics / mechanism / cost) but produced no benchmark showing the decomposition added measurable retrieval or reasoning quality over a scalar projection. `wonder.analyze_gaps` ranked by entropy — entropy is computable on the scalar form. The user-visible value of *"the cost dimension is uncertain"* over *"this trade-off belief is uncertain"* was an editorial intuition, not an evaluated property. Shipping a four-axis vector at v2.0 commits the documentation surface, the user-facing CLI surface, and the schema to a structure that hasn't earned a published number. aelfrice's compatibility rule (`docs/ROADMAP.md § Compatibility`) is "v2.0 may break v1 API only where a benchmark or eval justifies the break." The vector doesn't clear that bar today.

2. **PHILOSOPHY already committed direction.** Commit `1afdc04` declared *"A wrong belief is wrong overall; a useful belief is useful overall."* That's not a v1-only statement — it's a stance about how this system models confidence. Flipping to Option A at v2.0 means rewriting that paragraph from "this is what aelfrice chose" to "this was the v1 narrowing." That's a substantive narrative inversion to absorb in a release whose advertised goal is *parity*, not *identity reformation*.

3. **The four axes don't survive scrutiny.** *Existence* (does the thing exist?), *semantics* (does it mean what we think?), *mechanism* (does it work the claimed way?), and *cost* (is the price right?) are plausible but underjustified. They're orthogonal in some belief domains and overlapping in others. *"npm install is fast"* — is that a mechanism uncertainty (does the install succeed?) or a cost uncertainty (is *fast* the right price?). The research line didn't publish a calibration showing humans agree on which axis a given belief loads onto. Without that, the four-axis decomposition imposes structure that may not correspond to anything users can articulate.

4. **Reversibility is asymmetric.** B → A is a clean additive migration. A → B is deprecation of a user-facing concept. Choose the path whose reversal is cheap.

## Decision asks

This memo proposes Option B. Ratify or override:

- [ ] **Confirm Option B for v2.0.** If yes, the LIMITATIONS update and the scalar-only port path proceed.
- [ ] **If Option A:** override this memo with a written rationale. Specifically — what evidence (benchmark, user study, mechanical advantage) justifies the four-axis schema cost? Are the four axes (existence / semantics / mechanism / cost) the right ones, or does v2.0 ship a different decomposition?
- [ ] **Hibernation specifically.** Independent of the axis count, ship `hibernation_score` at v2.0? It's a separable lifecycle feature; the only schema cost is one nullable column. Recommend yes.
- [ ] **`activation_condition`.** Only meaningful with hibernation. Predicate language — JSON, Python expression, or string match? If hibernation ships, this lands with it.

## Downstream impact

If Option B is ratified:

- **#195** (`scoring.uncertainty_score`) ports as scalar — `uncertainty_score(alpha: float, beta: float) -> float`, returning Beta differential entropy via `math.lgamma`. Tier stays `rook`; spec becomes a one-paragraph addendum to this memo.
- **#199** (enforcement) is unaffected — the directive/compliance/selective-injection triad doesn't depend on per-axis uncertainty.
- **#193** (sentiment-from-prose), **#197** (dedup), **#198** (multimodel), **#201** (relationship_detector) are all unaffected by the substrate axis count. They ship or defer on their own merits.
- **`wonder` and `reason`** ship as v2.0 line items operating on scalar entropy. Their PRs reference this memo's § Capabilities lost section for the deliberate scope-narrowing.
- **`LIMITATIONS.md § Sharp edges`** gains one entry: *"v2.0 `wonder` ranks beliefs by scalar Beta entropy. The research line ranked by per-aspect entropy across four axes (existence / semantics / mechanism / cost); aelfrice v2.0 deliberately ships the scalar form. See [substrate_decision.md](substrate_decision.md)."*

If Option A is ratified instead:

- **#195** ports as vector — `uncertainty_score(vector: UncertaintyVector) -> dict[str, float]`. Tier upgrades to `queen` (CLI/MCP feedback surface needs to grow `--aspect`).
- **#196 follow-on:** schema migration spec — `schema_version = 2`, the three new columns, `Belief.uncertainty_vector` model field, `Belief.alpha`/`Belief.beta_param` redefined as vector projections.
- **`PHILOSOPHY.md § Bayesian, not vector`** rewrites to lead with the vector form; the scalar projection becomes the "aggregate view" paragraph.
- **`#198` (multimodel)** moves from "open evaluation" to *"likely required"* — populating per-aspect `(α_i, β_i)` from prose without an LLM is hard, and the research line used cross-model classification to do it.

## Provenance

Surfaced by parity-audit verify-deeper pass on `wonder.py` and `uncertainty.py` (agentmemory archive). Audit doc lives in the private aelfrice-lab workspace; this memo is the public ratification surface.

PHILOSOPHY single-axis declaration: commit `1afdc04` (`docs(philosophy): declare single-axis posterior; cross-link multi-axis substrate decision`).

Companion issues: [#193](https://github.com/robotrocketscience/aelfrice/issues/193), [#194](https://github.com/robotrocketscience/aelfrice/issues/194), [#195](https://github.com/robotrocketscience/aelfrice/issues/195), [#197](https://github.com/robotrocketscience/aelfrice/issues/197), [#198](https://github.com/robotrocketscience/aelfrice/issues/198), [#199](https://github.com/robotrocketscience/aelfrice/issues/199), [#201](https://github.com/robotrocketscience/aelfrice/issues/201).
