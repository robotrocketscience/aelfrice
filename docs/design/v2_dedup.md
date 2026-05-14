# v2.0 evaluation: dedup module

Spec for issue [#197](https://github.com/robotrocketscience/aelfrice/issues/197). Substrate-cascade addendum to [`substrate_decision.md`](substrate_decision.md) (#196 ratified Option B).

Status: read-path shipped. `src/aelfrice/dedup.py` is wired and exposed via `aelf doctor dedup` (audit-only mode); it's also imported by `relationship_detector.py`. The write-path `SUPERSEDES` hook (collapse-on-ingest) is bench-gated behind the corpus benchmark and remains deferred per V2_REENTRY_QUEUE.

## What's being decided

Whether to port the research-line `dedup.py` (~254 LOC, stdlib-only) to v2.0. The module finds near-duplicate beliefs after ingest by token-overlap + edit-distance, then collapses duplicates by inserting `SUPERSEDES` edges (not deletes) to preserve audit trail. v1.x has only `INSERT OR IGNORE` on `(source, sentence)` content_hash — exact-match only, paraphrases survive.

## Substrate dependency

None. dedup operates on text fields only. Substrate ratification (Option B, scalar Beta-Bernoulli) does not affect the port shape.

## Recommendation

**Ship at v2.0.** Port the research-line module as-is.

Three reasons:

1. **The capability fills a gap users will hit.** "don't push to main" inserted via lock plus "never push directly to main" inserted via onboard scanner produce two beliefs with the same intent. Both surface in retrieval. This is observable today; #219's 5.3× re-ingest row inflation makes it worse, not better.
2. **Stdlib-only port preserves determinism.** Token-overlap + Levenshtein is regex/integer-arithmetic. No learned similarity model, no LLM call, no embedding. Compatible with PHILOSOPHY's determinism property.
3. **Audit-trail discipline already exists.** The research line uses `SUPERSEDES` edges, which v1.x already has (one of the four edge types in `models.py`). No edge-type expansion needed for the port itself; downstream consumers of `SUPERSEDES` already exist (`aelf resolve`, demotion-pressure scoring).

## Decision asks

- [ ] **Confirm port at v2.0.** If yes, scope is ~254 LOC + write-path hooks (after `ingest_turn`, `onboard`, `apply_feedback`).
- [ ] **Threshold defaults.** Research line shipped Jaccard ≥ 0.8 + Levenshtein ratio ≥ 0.85. Ship those as defaults? Configurable via `.aelfrice.toml`?
- [ ] **Sample size.** Research line sampled 5000 candidate pairs to keep dedup O(n) per write. Ship that, or full-pairwise on small corpora?
- [ ] **Onboard interaction.** Currently `aelf onboard` is non-incremental (LIMITATIONS § Sharp edges). Dedup-on-onboard would partially fix that. Worth advertising as such, or ship dedup quietly and leave the onboard-incrementality issue separate?

## Downstream impact

- `LIMITATIONS.md § Sharp edges`: the "INSERT OR IGNORE only catches exact matches" caveat shrinks (or disappears, depending on dedup confidence).
- `docs/concepts/PHILOSOPHY.md`: one paragraph noting that v2.0 dedup is deterministic (token + edit distance, no embedding).
- New CLI surface: `aelf doctor dedup` audit command, returning candidate clusters without auto-merging. Lets users review before committing.
- `aelf:dedup` MCP tool would be the natural pairing; defer to per-tool bench-impact gate per `V2_REENTRY_QUEUE.md` decision #1.

## Provenance

Research-line module: `agentmemory/dedup.py` (254 LOC, stdlib-only).
Lab parity audit: `aelfrice-lab/docs/agentmemory-parity-audit-2026-04-28.md` § 2.
Substrate ratification: [substrate_decision.md](substrate_decision.md) (Option B).
