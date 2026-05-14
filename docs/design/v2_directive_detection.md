# v2.x re-entry: directive detection (enforcement H1)

Iteration spec for issue [#374](https://github.com/robotrocketscience/aelfrice/issues/374). Successor memo to [`v2_enforcement.md` § H1](v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate); does not re-decide the deferral or the gate, only the path to clearing it.

Status: deferred. Harness shipped (PR [#377](https://github.com/robotrocketscience/aelfrice/pull/377)). Path A intent-prefix filter shipped at PR [#374](https://github.com/robotrocketscience/aelfrice/pull/374) — `src/aelfrice/directive_detector.py` is in-tree. Candidate detector measured at P=0.664 / R=0.937 against the lab corpus v0.1 (285 rows, ≥200 floor met). Below the P≥0.80 floor, so H1 stays deferred per spec; iteration target is precision, not recall.

## What's being decided

Which detector-iteration path to commit to before the next implementation attempt at #374. The harness is in place; what's missing is a chosen direction for raising precision from 0.664 to ≥0.80 without dropping recall below 0.60. This memo proposes three concrete paths, recommends one, and defines the public-tree vs lab-tree work split so the issue stops bouncing.

## Substrate dependency

None. Directive detection operates on prompt text only; no belief schema, posterior, or edge-type interaction. The detector is a pure function `str → bool`.

## Failure-mode analysis (from the gate run on lab corpus v0.1)

The 45 false positives that drag precision to 0.664 cluster as a single dominant pattern: **imperative-grammar one-shot coding tasks**. Examples documented publicly on the issue thread:

- "Refactor X so it never blocks"
- "Add a test that ensures …"

These are imperatives the user issues to the agent for the immediate session — task instructions, not durable rules to remember. The 29-verb regex correctly fires on `never` / `ensure`; the spec's three filters (wh-question, hedge, "I never X when Y" narration) do not catch this class because the surface form is a clean second-person imperative.

The 6 false negatives are not the load-bearing problem — recall at 0.937 is comfortably above the 0.60 floor and has 23 percentage points of headroom for a more conservative detector.

## Iteration paths

### Path A: intent-prefix filter (recommended)

Add a pre-filter that classifies the leading clause. Imperative coding-task prefixes (`refactor`, `add`, `implement`, `write`, `create`, `update`, `fix`, `make`, `build`, `remove`, `rename`, `extract`, `merge`, `split`, `move`, `delete`) issued as the sentence head — without any rule-marker connective ("so that", "as a rule", "from now on") elsewhere in the sentence — are treated as one-shot tasks and short-circuit to `False` regardless of downstream imperative-verb hits.

- **Why this fits the failure mode:** the FP cluster all begin with a coding-task verb. The classification is a head-position lexical check, not semantic.
- **Precision impact:** removes the ~45 documented FPs (estimated; needs corpus re-run to confirm).
- **Recall impact:** small. The pattern "Refactor X so it never blocks as a rule" — coding task with embedded durable directive — is rare and can be opt-in via the rule-marker connective, which the filter detects.
- **Implementation surface:** ~30 LOC in `src/aelfrice/directive_detector.py`. Pure stdlib, deterministic. No corpus dependency for the public unit tests beyond what already lives in `tests/test_directive_detector.py`.
- **Determinism:** preserved.

### Path B: deontic-anchor requirement

Require an explicit deontic anchor (`always`, `never`, `must`, `must not`, `should`, `shall`, `forbidden`, `mandatory`, `prohibited`) — a strict subset of the current 29 verbs — and demote the remaining markers (`avoid`, `prefer`, `only`, `before`, `after`, `unless`, `whenever`, `need to`, `ensure`, `require`, …) to "supporting" status that does not fire on its own.

- **Precision impact:** large. Imperatives without a deontic anchor (the FP cluster) all return `False`.
- **Recall impact:** unknown but plausibly significant. Many durable rules are stated without strong deontic markers ("only push from main", "before merging, run the gate"). The 0.937 → ? drop could blow the 0.60 recall floor.
- **Implementation surface:** ~10 LOC change to `_IMPERATIVE_VERBS` partition.
- **Risk:** the recall hit may be larger than the headroom allows.

### Path C: lightweight LLM classifier

Replace the regex with an LLM call ("does this prompt encode a durable rule?"). Cached per-prompt to amortize cost; falls back to regex on cache miss + budget exhaustion.

- **Precision impact:** likely high; LLMs disambiguate this class easily.
- **Recall impact:** likely high.
- **Implementation surface:** large. Requires model selection, prompt design, cache schema, latency/cost budget, and a new dependency posture.
- **Determinism:** **broken.** Conflicts with `PHILOSOPHY.md` determinism property and with the `feedback_avoid_embeddings_nondeterminism` posture. Listed for completeness; not a serious option without an explicit ratification of a non-deterministic component in the rebuild path.

## Recommendation

**Path A (intent-prefix filter)** at v2.x re-entry. Reasons:

1. The failure mode is monolithic — one cluster, one structural property. A targeted filter beats a coarse partition (Path B) or a model swap (Path C).
2. Determinism preserved. No new dependency. ~30 LOC.
3. Failure cost is bounded: if Path A under-performs against an updated corpus, Path B is the natural next step (already a strict subset of A's verb bank); the iteration order does not lock anything in.

## Decision asks

- [ ] **Confirm Path A as the iteration target.** If no, name an alternative (B, C, or a new path) before any code change.
- [ ] **Confirm head-position lexical anchors for the prefix filter.** The 16 verbs listed are reconstructed from typical session-task prefixes; the gate scores whatever lands in `directive_detector.py`, but the choice is worth ratifying so the verb bank does not drift quietly.
- [ ] **Confirm the rule-marker connective list** (`so that`, `as a rule`, `from now on`, …) that re-enables directive classification when an imperative coding-task prefix is present. Default: empty (i.e., coding-task prefix always wins). Conservative; revisit after first corpus run.
- [ ] **Lab-side action: open `tests/corpus/v2_0/directive_detection/v0_1.jsonl` PR against lab `main`** (currently on branch `exp/issue-374-directive-corpus-v0_1`). The gate harness in PR #377 cannot fire end-to-end until the corpus is on the canonical lab path (`AELFRICE_CORPUS_ROOT/directive_detection/v0_1.jsonl`). Without that merge, every public-tree session that reads `aelf-scan` and lands on #374 will continue to bounce.

## Out of scope

- `process_directive`, the TODO lifecycle, the repetition counter, the escalation table, hook wiring. All gated on the bench gate passing per [`v2_enforcement.md` § H1](v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate). Implementation work on those starts only after a detector revision lands a passing gate run.
- Corpus authoring. Per directory-of-origin rules and `tests/corpus/v2_0/README.md`, the labeled rows live under `AELFRICE_CORPUS_ROOT` (lab) and never ship to the public tree. This memo does not propose corpus changes; it assumes lab corpus v0.1 is the immediate evaluation target and that v0.2 (with hard-negative imperative-task examples added) is a natural follow-up.
- Verb-bank expansion in the existing 29-imperative regex. Path A composes with the current regex; it does not modify it.

## Public-tree vs lab-tree work split

To stop the 10-bounce cycle on #374, the actionable work split is:

- **Public-tree (this repo):** detector source change in `src/aelfrice/directive_detector.py`; public-CI sanity tests in `tests/test_directive_detector.py` for the new filter behaviors; the bench-gate harness already in `tests/bench_gate/test_directive_detection.py` does not need changes.
- **Lab-tree:** corpus authoring (v0.1 → v0.2 with added hard-negatives); bench-gate run with `AELFRICE_CORPUS_ROOT` mounted to verify P/R against the gate. Lab-side P/R numbers are reported back as a comment on #374 but the rows themselves do not cross the boundary.

A public-tree session can land Path A as a doc-only commit that preserves H1's deferred status (no claim that the gate is passing). Closing #374 still requires a lab-side run that demonstrates P≥0.80 ∧ R≥0.60 ∧ n≥200; that closing event lands as the same PR that strikes the row from `V2_REENTRY_QUEUE.md`.

## Re-entry trigger (unchanged from § H1)

H1 reopens for implementation when:

- ≥0.80 precision on the lab corpus (currently `aelfrice-lab/tests/corpus/v2_0/directive_detection/`, ≥200 rows).
- ≥0.60 recall on the same corpus.
- A reproducible bench-gate run is recorded on the closing PR (lab `pytest -q tests/bench_gate/test_directive_detection.py` output; numbers cited in PR body).

If those numbers are not met after Path A, this memo's recommendation is to revisit Path B (deontic-anchor partition) before any model-based approach. Path C stays out of scope until the determinism property is explicitly re-decided.

## Provenance

- Parent spec: [`docs/design/v2_enforcement.md` § H1](v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate) (PR [#257](https://github.com/robotrocketscience/aelfrice/pull/257), merged 2026-04-28).
- Re-entry queue row: [`docs/design/V2_REENTRY_QUEUE.md`](V2_REENTRY_QUEUE.md) § "Directive detection (enforcement H1) — issue #374".
- Harness PR: [#377](https://github.com/robotrocketscience/aelfrice/pull/377), merged 2026-05-03.
- Lab corpus v0.1 reference: issue #374 comment 2026-05-03T17:16:22Z (285 rows, P=0.664 / R=0.937 against the candidate detector).
- Failure-mode examples ("Refactor X so it never blocks", "Add a test that ensures …"): same comment.
- Umbrella: [#199](https://github.com/robotrocketscience/aelfrice/issues/199).
