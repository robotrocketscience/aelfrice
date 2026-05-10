# v3.0 PHILOSOPHY: natural-language-relatedness gate

Spec for issue [#605](https://github.com/robotrocketscience/aelfrice/issues/605). Ratifies one of the three options the issue surfaces.

## Decision

**Option 1 — stay deterministic, narrow the surface.**

Dedup / contradiction / relatedness gates that require resolving paraphrase or synonymy are out of scope at the retrieval and ingest layers. They live in the consuming agent. PHILOSOPHY's determinism property holds end-to-end; the known capability ceiling on multi-fact coherence and stale-belief surfacing is accepted, not paid down.

## Why

[`PHILOSOPHY.md`](PHILOSOPHY.md) § *Determinism is the property* already commits to this:

> No embeddings, no learned re-rankers, no LLM in the retrieval path. […] These hold compositionally. A single non-deterministic step in the retrieval path destroys the property for the whole pipeline. There is no "mostly deterministic" — either it holds end-to-end or it does not.

> The trade-off is real. Embedding systems beat aelfrice on fuzzy semantic recall and multi-session aggregation. We treat that as a clarification of what aelfrice is for, not a gap to close at the cost of the property.

Option 2 (admit a sentence-embedding lane, fenced) and Option 3 (deterministic primary, embedding fallback) both violate the compositionality clause: any embedding lane in the retrieval path — even fenced behind a flag, even consulted only on miss — is a non-deterministic step in the retrieval path. The property cannot be partitioned per-lane without rewriting `PHILOSOPHY.md` § *Determinism* itself, and the costs cited in that section (debugging boundedness, provenance composition, counterfactual evaluation, audit comprehensible to a non-technical reviewer) are downstream of the global property holding.

The bench evidence the issue cites confirms the ceiling is real, not the conclusion that the ceiling should be paid for inside aelfrice:

- [#197](https://github.com/robotrocketscience/aelfrice/issues/197) dedup R2 (WONTFIX 2026-05-08): per-label recall 0.000 against `prefix-overlap T=0.70 OR cosine-TF/IDF T=0.92` on the 57-pair labelled corpus. Score distributions inverted, not attenuated. Token-overlap cannot bridge synonym substitution.
- [#422](https://github.com/robotrocketscience/aelfrice/issues/422) contradiction-detector (WONTFIX 2026-05-05): `contradicts` recall 0.033 for the same reason — paraphrases substitute synonyms.
- [#201](https://github.com/robotrocketscience/aelfrice/issues/201) R2: same wall, closed earlier.

Three independent corpora, three closures on the same boundary. That's evidence that the *retrieval-layer relatedness gate* shape doesn't work on real natural-language pairs without semantic similarity. It is not evidence that aelfrice should ship semantic similarity.

## Where the capability lives instead

Pushed out of aelfrice's retrieval / ingest path and into the consuming agent's responsibility:

- **Dedup-by-paraphrase.** The agent that ingests two beliefs that are paraphrases of each other is the right layer to decide they say the same thing. The agent has the LLM call available; aelfrice does not. `aelf doctor dedup` (R1 surface, hash-collision dups) stays shipped — it catches what determinism *can* catch, and explicitly does not claim near-duplicate detection ([`v2_dedup.md`](v2_dedup.md)).
- **Contradiction detection.** Same pattern. The typed-slot value-comparison gate from [#422](https://github.com/robotrocketscience/aelfrice/issues/422) (`feat(value_compare): typed-slot extractor + mutual-exclusion comparator`) stays shipped because it operates on structured, deterministic comparisons (numeric / enum mutual exclusion). Free-form contradiction is the agent's call.
- **Relatedness for retrieval ranking.** BM25F + posterior + heat-kernel + HRR-structural already cover the deterministic surface. Synonym-bridging recall is the model's job at consumption time, not aelfrice's at retrieval time.

## What this rules out, explicitly

- A `sentence-transformers` (or any embedding-model) dependency in the runtime install footprint. Not in `[mcp]`, not in a new `[similarity]` extra, not in a "deterministic by default, embedding by flag" lane.
- An "LLM-judged similarity" path inside `retrieve_v2`, `relationship_detector`, `dedup`, or any successor module on the retrieval / ingest path.
- A v3.x successor `feat/issue-NNN-relatedness-gate-with-embedding-fallback` ticket. Filed as a non-issue: any such ticket lands at the same WONTFIX boundary unless this memo is first amended.

## What this leaves open

- **Lab-side experiments** (in `~/projects/aelfrice-lab`, gitea origin only) that *evaluate* embedding-based or LLM-judged similarity on real corpora are not scoped out — the lab is the right place to measure the ceiling, characterize it, and produce evidence for any future amendment to this memo. The constraint here is on what lands on the public retrieval path, not on what gets investigated.
- **Re-opening this memo at v3.x.** If a future bench shows the deterministic ceiling is hard-blocking a stated user-visible capability — i.e., a real reported pain point that cannot be addressed at the consuming-agent layer and that materially impacts adoption — file a new issue citing this memo, naming the specific capability gap, and proposing the minimum amendment to `PHILOSOPHY.md` § *Determinism* required to admit the change. The amendment is the gate, not the embedding library.

## Acceptance

- [x] Decision recorded: Option 1.
- [x] Rationale grounded in `PHILOSOPHY.md` § *Determinism is the property*.
- [x] Successor pattern ("capability lives in the consuming agent") documented.
- [x] Explicit exclusions listed so future tickets don't have to re-litigate the boundary.

## Out of scope

- Implementing any change to existing modules. R1 dedup surface and the typed-slot value-comparator stay as-is.
- Lab-side bench evidence collection. (See "What this leaves open".)
- A `PHILOSOPHY.md` § *Determinism* edit. The decision is to ratify the existing wording; no doc surgery needed beyond a one-line cross-reference.

## Refs

- [`PHILOSOPHY.md`](PHILOSOPHY.md) § *Determinism is the property*
- [`v2_dedup.md`](v2_dedup.md) — the dedup R1 ship-decision; bounds what determinism *can* do at the dedup gate
- [#197](https://github.com/robotrocketscience/aelfrice/issues/197) WONTFIX terminal comment — the diagnostic that prompted this memo
- [#422](https://github.com/robotrocketscience/aelfrice/issues/422) — typed-slot value-comparator (shipped) + free-form contradiction detector (WONTFIX)
- [#201](https://github.com/robotrocketscience/aelfrice/issues/201) — earlier closure on the same boundary
- [#608](https://github.com/robotrocketscience/aelfrice/issues/608) — v3.0 milestone tracker
