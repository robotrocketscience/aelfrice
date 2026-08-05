# v2.x re-entry: directive detection (enforcement H1)

Iteration spec for issue [#374](https://github.com/robotrocketscience/aelfrice/issues/374). Successor memo to [`v2_enforcement.md` § H1](v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate); does not re-decide the deferral or the gate, only the path to clearing it.

Status: deferred. The gate was **not a valid measurement** against corpus v0.1 alone; corpus v0.2 (2026-08-05) fixes that and the deferral now rests on a real number — see § Gate validity. Harness shipped (PR [#377](https://github.com/robotrocketscience/aelfrice/pull/377)). Path A intent-prefix filter shipped at PR [#467](https://github.com/robotrocketscience/aelfrice/pull/467) (issue #374) — `src/aelfrice/directive_detector.py` is in-tree.

The shipped detector measures **P=0.706 / R=0.937** against lab corpus v0.1 (285 rows; TP=89, FP=37, FN=6, TN=153), re-run 2026-08-05 under [#1341](https://github.com/robotrocketscience/aelfrice/issues/1341). Below the P≥0.80 floor, so H1 stays deferred per spec.

Two corrections to what this memo previously published, both from that re-run:

- The status line carried **P=0.664** for three months. That was the *pre-*Path-A number. Path A shipped in May and the confirming re-run this memo asked for (§ Path A, "needs corpus re-run to confirm") never happened until now.
- Path A removed **8** false positives, not the "~45" § Path A estimated — the estimate was off by roughly 5x. 37 false positives remain, and they are not the single monolithic cluster § Failure-mode analysis claimed.

## What's being decided

**Superseded — read § Gate validity first.** As originally written: which detector-iteration path to commit to before the next implementation attempt at #374; the harness is in place, and what was missing was a chosen direction for raising precision from 0.664 to ≥0.80 without dropping recall below 0.60. That framing no longer holds. The measured figure is **0.706**, not 0.664, and § Gate validity shows corpus v0.1 cannot distinguish a detector that generalises from one that has memorised opening vocabulary — so no measured preference among the three paths below was interpretable at the time. Corpus v0.2 landed 2026-08-05 and closed that gap; the live decision is again a detector path, but chosen against the union corpus and against recall, not precision — see § Recommendation.

## Substrate dependency

None. Directive detection operates on prompt text only; no belief schema, posterior, or edge-type interaction. The detector is a pure function `str → bool`.

## Failure-mode analysis (re-run 2026-08-05, lab corpus v0.1)

The original analysis called the false positives "a single dominant pattern: imperative-grammar one-shot coding tasks", and Path A was built for that one cluster. The measured error set does not support that. The 37 surviving false positives fall into at least six structurally distinct families:

1. **Interrogatives the filters miss.** The detector tests for a trailing `?` on the whole string and a wh-word at position 0. It does not see a question embedded mid-message, an auxiliary-leading clause (`should we …`, `is there …`), or a question that trails off unpunctuated.
2. **Reported speech / attribution.** A deontic marker governed by a speech or claim verb belongs to the cited source, not to the speaker.
3. **Third-person descriptive statements and past tense** — describing how a system or a person behaves, rather than instructing.
4. **Use/mention.** The imperative appears only inside quotation marks, or as a cited example or maxim.
5. **One-shot task imperatives outside the 16-verb prefix bank.** The only family Path A targets; the bank is simply too small.
6. **Affirmations, stated intent, and observations** — assent to something already proposed, or an evidential frame, carrying a real modal.

The 6 false negatives are not the load-bearing problem — recall at 0.937 is comfortably above the 0.60 floor and has 34 percentage points of headroom for a more conservative detector.

## Gate validity — the corpus admits a one-token solution

**Do not iterate the detector against corpus v0.1. It cannot tell a working detector from a memorised one.**

Measured 2026-08-05 (#1341, swept under #1349). Build the weakest classifier that can be written: read the **first word** of the prompt, look up the majority label for that word among the training rows, answer with it. It has no representation of mood, grammar, attribution, durability, or task-versus-rule. Partition the 285 rows 60/40 by `sha1(salt + id)` and score it on the held-out half, over **K=200 salted partitions**:

```
partitions where it CLEARS P>=0.80 / R>=0.60 : 196 / 200  (98.0%)
precision   min 0.833   median 0.941   max 1.000
recall      min 0.529   median 0.729   max 0.875
pooled      P=0.9381 (n=5,978 positive predictions, Wilson 95% [0.9317, 0.9439])
```

It clears the H1 gate on 98% of partitions. The real detector scores P=0.706 and does not clear it on any.

The sweep is not decoration. A single partition decides precision on ~34 positive predictions, and that interval straddles the gate — the first published figure for this finding was one such draw (P=0.912, R=0.795), which is a sample from the distribution above rather than a property of the corpus. The claim that survives is the swept one. The **partition-independent** statistics below are what the conclusion actually rests on, and they do not depend on any split at all.

The cause is in how the corpus separates its classes. Of 114 distinct opening words, only **7** appear in both classes; 87.9% of rows are perfectly classified by their first word alone. The positive class opens overwhelmingly with deontic or policy vocabulary (`always`, `never`, `don't`, `avoid`, `prefer`, `use`) and the negative class with task, question, and discourse vocabulary (`write`, `run`, `check`, `what`, `should`, `can`, `ok`, `please`). No row labeled `directive` opens with a one-shot task verb, and none labeled `not-directive` opens with a bare deontic.

The consequence for detector work: **any rule keyed on head position buys precision for free.** It cannot be charged for the durable directives it would wrongly suppress, because the corpus contains none of them — no "Check every PR for a changelog entry before you approve it", no "Run the full suite before pushing to a shared branch". Those are ordinary standing rules that open with a task verb, and a head-verb filter eats them silently in production while scoring a clean sweep in-corpus.

This was found by building six family-scoped suppression rules against the train split and putting each through independent adversarial review. Composed, they took the corpus to P=0.953 / R=0.853 — a comfortable pass. All six were then judged both overfit and over-reaching, unanimously, with concrete minimal pairs; the one-token baseline above explains why. **None of them shipped.** A gate that a one-token lookup table passes will bless an overfit detector, and did.

`tests/bench_gate/test_directive_detection.py::test_directive_corpus_defeats_a_first_token_baseline` asserts this property over the same K=200 sweep, failing if the baseline clears the gate on **any** partition, so the condition cannot silently return. It is red against v0.1 alone, by design.

### Corpus v0.2 — delivered 2026-08-05

Not more rows — **minimal pairs that break the head-word/class correlation**: the same opening word carrying both labels, so no head-position rule can buy precision for free.

`v0_2.jsonl` adds **225 rows** across six head-word buckets — durable rules opening with task verbs (`check`, `run`, `review`, `update`, `remove`, `write`, `add`), which v0.1 lacked entirely, and one-shot requests opening with policy verbs (`always`, `never`, `avoid`, `use`, `prefer`, `ensure`). Effect on the union (510 rows):

| | v0.1 | v0.1 + v0.2 |
|---|---|---|
| rows whose first word is class-ambiguous | 12.1% | **71.6%** |
| partitions where the one-token baseline clears the gate | 196/200 | **0/200** |
| baseline precision, max over K=200 | 1.000 | **0.754** |
| baseline pooled precision | 0.9381 | 0.6108 |
| real detector | P=0.706 / R=0.937 | **P=0.665 / R=0.636** |

The validity guard goes green on the union. The gate itself stays red, now for a real reason: the detector genuinely does not clear P≥0.80 on a corpus that cannot be solved by memorising opening vocabulary. **H1 remains deferred, and its number now means what § H1 intends.**

Labels were authored per bucket, then reproduced by two independent passes with labels stripped, row ids replaced by opaque hashes, and rows reshuffled — 225/225 agreement, κ=1.000, zero unclear. Two caveats bound that: both passes are the same model family, and minimal pairs stay recognisable by topic however the ids are scrambled. It bounds label noise; it is not proof of independence.

Remaining known weakness, self-reported by the authors: one-shot rows still carry concrete referents (a file, a PR number, a version) more often than durable rows do. That is arguably the real semantics of durability rather than an artifact, but it is the next shortcut a classifier would find, and v0.3 should be sized against it rather than against head words.

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

**Path B (deontic-anchor partition), measured against the v0.1+v0.2 union.**

Steps 1 and 2 of this memo's previous ordering are done: corpus v0.2 landed 2026-08-05 and the validity guard is green on the union, so a P/R number from this corpus is now worth quoting. Detector iteration is unblocked.

Path A was this memo's original recommendation and it shipped (PR #467), moving precision 0.664 → 0.706 on v0.1. That preference was never interpretable — it was measured on a corpus a lookup table could solve — so it should not be read as evidence for A over B.

Against the union the detector sits at **P=0.665 / R=0.636**, and the shape of the problem has changed: recall is no longer comfortable. On v0.1 recall was 0.937 with 34 points of headroom, which is what licensed "trade recall for precision freely". On the union it is 0.636, barely above the 0.60 floor, because v0.2's hard positives are durable rules the 29-verb regex never fires on. **Any further precision work now has to pay for its recall**, which is exactly the constraint § Path B's risk note anticipated and could not previously measure.

Ordering from here:

1. Re-derive the failure families against the union — the six in § Failure-mode analysis were read off v0.1 and their relative weights will have moved.
2. Attack recall first, not precision. The floor is the binding constraint now.
3. Path B on the union, with its recall cost finally observable.

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
- **The corpus defeats the one-token baseline** — `test_directive_corpus_defeats_a_first_token_baseline` green. Added 2026-08-05 (#1341); without it the two thresholds above can be met by a lookup table. This condition is new and is not a re-decision of the gate, which § H1 fixes; it is the precondition under which the gate's numbers mean what § H1 intends.
- A reproducible bench-gate run is recorded on the closing PR (lab `pytest -q tests/bench_gate/test_directive_detection.py` output; numbers cited in PR body).

Path B (deontic-anchor partition) is the next candidate once the corpus is valid. Path C stays out of scope until the determinism property is explicitly re-decided.

## Provenance

- Parent spec: [`docs/design/v2_enforcement.md` § H1](v2_enforcement.md#h1-directive-detection--defer-to-v2x-with-benchmark-gate) (PR [#257](https://github.com/robotrocketscience/aelfrice/pull/257), merged 2026-04-28).
- Re-entry queue row: [`docs/design/V2_REENTRY_QUEUE.md`](V2_REENTRY_QUEUE.md) § "Directive detection (enforcement H1) — issue #374".
- Harness PR: [#377](https://github.com/robotrocketscience/aelfrice/pull/377), merged 2026-05-03.
- Lab corpus v0.1 reference: issue #374 comment 2026-05-03T17:16:22Z (285 rows, P=0.664 / R=0.937 against the candidate detector).
- Failure-mode examples ("Refactor X so it never blocks", "Add a test that ensures …"): same comment.
- Umbrella: [#199](https://github.com/robotrocketscience/aelfrice/issues/199).
