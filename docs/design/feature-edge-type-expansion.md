# Edge-type expansion: SUPPORTS + SUPERSEDES at ingest (#999)

Status: spec (needs-spec → spec-ready)
Issue: #999
Depends on: #988 (CONTRADICTS substrate), #201 (detector ratification), #197 (dedup bench gate)
Blocks: #1001 (`+HRR-expand` ablation arm)

## Problem

The deterministic HRR vocabulary-bridge expansion lane (`use_hrr_expand`,
#981/PR #997) finds semantic neighbors by traversing typed edges. The ingest
substrate built in #988 emits **CONTRADICTS only**
(`relationship_detector.py:905`), so the lane traverses a graph missing every
other edge class. The #981 ceiling measurement (2,098 edges on LoCoMo10, all
CONTRADICTS) is the binding constraint, not the lane logic. Until the substrate
carries more edge types, the `#1001` ablation measures a starved graph and a
flat F1 delta would be **misread** as "the lane does not help."

This spec adds the two highest-value semantic edge types for a conversational
corpus — **SUPPORTS** and **SUPERSEDES** — to the ingest substrate.
Code-edge types (CALLS / CITES / TESTS / IMPLEMENTS) are out of scope: they
are source-graph relations with no deterministic lexical signal in
conversational belief text.

## Key finding: both writers already exist, deferred

Neither edge type needs a new detector. The detection already exists; only the
write-path is gated off.

- **SUPPORTS** — `relationship_detector.analyze()` already returns a `REFINES`
  verdict for pairs with residual-content overlap above the relatedness floor
  **and** agreeing modality (`relationship_detector.py:335-340`). REFINES is
  counted in audits (`n_refines`) but never written: `write_semantic_edges`
  is `CONTRADICTS only` by design (#988). `EDGE_SUPPORTS` is a defined model
  type (`models.py:32`, valence `+1.0`).
- **SUPERSEDES** — `dedup.dedup_audit()` already clusters near-duplicate
  beliefs (union-find) and documents the write-path: "will use the *oldest*
  member as the SUPERSEDES target" (`dedup.py:160`), "deferred behind the #197
  bench gate" (`dedup.py:329`). `EDGE_SUPERSEDES` is a defined model type
  (`models.py:35`, valence `0.0`).

So #999 = **activate two already-designed, deferred write-paths**, gated
default-off under the existing `AELFRICE_AUTO_RELATIONSHIPS` flag. This mirrors
exactly how #988 activated the #201-ratified CONTRADICTS writer default-off,
and it keeps the #605/#897 determinism-and-narrow-surface decision intact: no
LLM, no embedding, no new default-on behavior.

## Design

### SUPPORTS (relationship_detector.py)

Add `write_supports_edges(store, *, jaccard_min, residual_overlap_min,
max_candidate_pairs, max_edges_per_belief) -> SemanticEdgeWriteReport`,
structurally identical to `write_semantic_edges` but acting on the `REFINES`
slice of the audit instead of the `CONTRADICTS` slice.

- Candidate set: `report.pairs` where `label == LABEL_REFINES`.
- REFINES carries no confidence gradient (score is fixed `0.0`,
  `relationship_detector.py:337`); the residual-overlap floor already cleared
  in `analyze()` is the relatedness gate. So **all** REFINES pairs are
  eligible — no `confidence_min` partition (unlike CONTRADICTS, which splits
  high/sub-confidence between `write_semantic_edges` and
  `write_potentially_stale_edges`).
- Edge is **symmetric** (mutual support): canonicalize `src = min(id)`,
  `dst = max(id)` — same as the CONTRADICTS writer.
- **Idempotent**: skip if `(src, dst, SUPPORTS)` exists.
- **Write-gated**: reuse `max_edges_per_belief` (Exp-48 coverage-dilution
  guard); pre-existing edges count toward the per-belief budget across re-runs.
- **Deterministic**: pairs processed in the audit's `(belief_a_id,
  belief_b_id)` order, so which edges survive the cap is deterministic.

### SUPERSEDES (dedup.py)

Add `write_supersedes_edges(store, *, jaccard_min, levenshtein_min,
max_candidate_pairs, max_edges_per_belief) -> SupersedesWriteReport`, acting on
`dedup_audit()` clusters.

- For each cluster, the **oldest** member (min `created_at`, tie-break min
  `id`) is the SUPERSEDES *target*. Every other member emits one edge
  `src = member` → `dst = oldest` (directional: the newer belief supersedes the
  older).
- **Idempotent**: skip if `(member, oldest, SUPERSEDES)` exists.
- **Write-gated**: `max_edges_per_belief`, same semantics.
- **Deterministic**: members are already lexicographically sorted in
  `DuplicateCluster.member_ids`; clusters sorted by `representative_id`.
- Direction rationale: SUPERSEDES is the one asymmetric type here. Newer→older
  matches the lane's intent (a query hitting the current belief should reach
  the ones it replaced, and vice-versa, via the directed edge the lane probes).

### Ingest wiring (ingest.py)

The CONTRADICTS writer is already invoked behind
`is_auto_relationship_detection_enabled()` (ingest.py:307-311). Add the two new
writers in the same guarded block, same order every run:
`write_semantic_edges` → `write_supports_edges` → `write_supersedes_edges`.
When the flag is off, ingest stays byte-identical to today.

## Determinism & test plan

Mirror the #988 CONTRADICTS tests for each new writer:

1. Flag-off no-op: ingest writes zero SUPPORTS/SUPERSEDES edges.
2. Byte-equal edge table across two runs on the same store snapshot
   (tie-break by `(src, dst)` ASC).
3. Idempotency: second `write_*` run writes 0 new edges, skips N existing.
4. Write-gate: a belief at `max_edges_per_belief` drops further pairs
   deterministically.
5. SUPPORTS: a REFINES pair produces exactly one symmetric SUPPORTS edge; a
   CONTRADICTS pair produces none.
6. SUPERSEDES: a 3-member duplicate cluster produces 2 edges both pointing at
   the oldest member; direction is newer→oldest.
7. No regression: existing CONTRADICTS / POTENTIALLY_STALE writers unchanged.

## Out of scope

- CALLS / CITES / TESTS / IMPLEMENTS edge types (no deterministic conversational
  signal).
- Flipping `AELFRICE_AUTO_RELATIONSHIPS` to default-on — that reverses #605 and
  is explicitly routed elsewhere.
- The incremental per-turn detection rewrite — tracked separately (#1000); this
  spec keeps the existing full-audit candidate-pair path.
- Running the `+HRR-expand` ablation — #1001, blocked on this landing.
