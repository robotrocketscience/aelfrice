# Feature spec: Correction-detection eval (#438)

**Status:** spec, no implementation
**Issue:** #438
**Recovery-inventory line:** [`docs/ROADMAP.md`](ROADMAP.md) — *"Correction-detection eval — five-codebase labeled fixture, scored by both the zero-LLM detector and the LLM-judge path"*
**Substrate prereqs:** [`relationship_detector.py`](../src/aelfrice/relationship_detector.py) (`LABEL_CONTRADICTS`/`LABEL_REFINES`/`LABEL_UNRELATED`, #201/#422), [`value_compare.py`](../src/aelfrice/value_compare.py) (slot conflict, #422), [`correction.py`](../src/aelfrice/correction.py) (utterance-level detector, distinct surface), v2.0 corpus scaffold + bench-gate harness (#307, #311), `bench_gated` autouse marker

---

## Purpose

Score how well the existing relationship-detector substrate identifies **historical corrections** — pairs `(A, B)` where belief `B` replaced belief `A` because `A` was wrong. Distinct from:

- `correction.py` (utterance-level): "is *this input text* a correction directive?" Single-string classifier.
- `contradiction` bench-gate (already shipped): "do these two beliefs about the same subject mutually exclude?" Pair classifier, no temporal evidence required.

The correction-detection eval is the **subset of contradictions** for which commit history (or equivalent provenance) shows `B` superseded `A` after `A` was identified as wrong. Five-codebase, dual-scored: zero-LLM (regex / structural) vs an opt-in external LLM judge (model selected by env var, see Path B).

This memo is a **decision memo + acceptance contract**. There is no detector code change in the implementation PR that follows — the eval scores existing substrate. If the regex path falls below the bench-gate floor, that is a signal to either improve substrate or relax the floor; this memo does not pre-commit either response.

---

## Why a correction eval is distinct from contradiction

The contradiction tracks (`tests/bench_gate/test_contradiction.py`, `tests/bench_gate/test_contradiction_v3.py`) score *pair-classification* — given `(A, B)`, is the verdict `contradicts`? They do not consult provenance. A pair labeled `contradicts` may be:

1. **A correction** — `B` replaced `A` in history because `A` was wrong (commit edited the docstring; new release supersedes old; bugfix replaced the buggy claim).
2. **A standing disagreement** — two beliefs from different sources that happen to disagree, neither having "replaced" the other.

The research-line surface treated (1) as the *high-value* contradiction class: corrections carry a strong signal that `A` should be retired and `B` retained. Standing disagreements are the lower-value class — both beliefs may be live, and the system does not know which is right.

`#438` measures the detector's ability to identify class (1) specifically, using **commit-history provenance** as ground truth. The class distinction is a property of the labeled pair, not a new verdict the detector produces.

---

## Definition of a correction pair

```
A pair (A, B) is a CORRECTION iff:
  1. contradicts(A, B) holds — A and B make mutually-exclusive claims about
     the same subject (per relationship_detector.classify).
  2. There exists a commit C such that C's diff replaces text expressing A
     with text expressing B, and C's commit-message intent is one of
     {fix, correction, revert-of-error}.
```

Condition (2) is the **historical-supersession** evidence that distinguishes corrections from standing disagreements. It is provenance, not detector output — the labeled fixture carries the commit URL in `provenance` and the labeller's reasoning in `labeller_note` (matching the Django pilot's existing schema in `tests/corpus/v2_0/contradiction/django_v0_1.jsonl`).

The eval **does not** ask the detector to identify condition (2) from belief content alone. It asks: given a pair where (2) holds (per fixture), does the detector's `classify(A, B)` verdict cluster correctly?

---

## Corpus

### Codebases (5)

| codebase | category | primary language | source |
|---|---|---|---|
| `mlflow` | aiml | Python | archon `~/agentmemory-corpus/public/aiml/mlflow` |
| `cockroach` | database | Go | archon `~/agentmemory-corpus/public/database/cockroach` |
| `terraform` | devops | Go | archon `~/agentmemory-corpus/public/devops/terraform` |
| `rustls` | security | Rust | archon `~/agentmemory-corpus/public/security/rustls` |
| `micropython` | embedded | C + Python | archon `~/agentmemory-corpus/public/embedded/micropython` |

Picked for language and domain diversity across the cloned-and-mined public repos. All five are well-known public OSS projects; provenance URLs cite their public GitHub mirrors. The corpus authoring tooling lives lab-side and never reads from any non-public corpus tree — see "Public-safe corpus boundary" below.

The pilot Django corpus at `corpus/issue-438-correction-detection` (lab branch, 30 rows, currently labeled `contradicts`) **belongs to this eval**: those rows fit the correction definition above (each `labeller_note` describes a pre-correction → post-correction docstring change). The implementation PR relabels the Django rows from `contradicts` to `corrects` under the new schema, retains them, and adds them to the corpus as a sixth codebase footnote — they do not count against the per-codebase 50-pair target but do exercise the schema migration.

### Pair count

**50 labeled pairs per codebase × 5 codebases = 250 total.** The 95% Wilson CI on a P=0.85 estimate from N=50 is roughly ±10pp; aggregated to N=250 it tightens to ~±4pp. The bench-gate floor (P≥0.80, R≥0.70 — Decision 3) is calibrated against the aggregated number, not per-codebase. Per-codebase numbers are reported but do not gate.

### Schema

```
{
  "id": "<codebase>-<n:03d>",         e.g. "mlflow-001"
  "belief_a": "...",                   pre-correction belief content
  "belief_b": "...",                   post-correction belief content
  "label": "corrects",                 enum: {corrects, refines, unrelated}
  "provenance": "https://github.com/<org>/<repo>/commit/<sha>",
  "labeller_note": "...",              one-sentence labelling rationale
  "commit_intent": "fix|correction|revert-of-error"   commit-msg classification
}
```

The label enum here (`corrects`/`refines`/`unrelated`) is the **eval-fixture vocabulary**. The detector substrate's verdicts (`contradicts`/`refines`/`unrelated`, see `relationship_detector.LABEL_CONTRADICTS` etc.) are mapped into this vocabulary by the scoring pass:

- detector `contradicts` ∧ fixture `corrects` → true positive (regex caught the correction).
- detector `contradicts` ∧ fixture `unrelated` → false positive (regex over-fired).
- detector `refines` ∧ fixture `corrects` → false negative (regex misclassified as benign).
- detector `unrelated` ∧ fixture `corrects` → false negative (regex missed entirely).

This mapping is the bridge between substrate verdict and eval verdict; **no `LABEL_CORRECTS` is added to substrate at v2.0.0** (out-of-scope below).

### Public-safe corpus boundary

Per `docs/eval_fixture_policy.md` and the campaign-scoped corpus policy (#307), the labeled `pairs.jsonl` files live in **lab repo only** under `tests/corpus/v2_0/correction/{codebase}/pairs.jsonl`. The public repo's CI runs the bench-gate test with `AELFRICE_CORPUS_ROOT` unset; the autouse `bench_gated` marker skips the test, keeping public CI green.

Sourcing rules for the labellers:

1. Source repos are limited to the public corpus tree (`~/agentmemory-corpus/public/`). Private/personal corpus trees are never sourced. The boundary is the `public/` directory, not a per-pair filter.
2. Belief content is **paraphrased from public commit history** (per #307 policy — paraphrase real OSS content into corpus rows). Direct quotation of >5 consecutive words from the source is avoided to keep the corpus well clear of any republication question.
3. Provenance URLs cite the public GitHub mirror, not archon. The archon paths are an authoring convenience; published rows do not name them.
4. The labeller_note is the labeller's own prose — not lifted from the commit message.

---

## Detector paths

### Path A — zero-LLM (regex / structural)

Composes existing substrate, **no new ML**:

1. `relationship_detector.classify(belief_a, belief_b)` — produces `{contradicts, refines, unrelated}`.
2. `value_compare.find_conflicts(belief_a, belief_b)` — produces slot-level disagreements (numerics, enums); a non-empty conflict set is a strong correction signal independent of (1).
3. **Composite verdict:** `corrects` if `classify == contradicts` OR `find_conflicts` non-empty AND commit_intent ∈ {fix, correction, revert-of-error}; otherwise `refines`/`unrelated` per `classify`.

The composite is the eval's regex-path verdict. The bench-gate test in `tests/bench_gate/test_correction_detection.py` runs this path against the labeled fixture and reports `(P, R, F1)` per-codebase + aggregated.

The **commit-intent classifier** is the one piece that is not currently in substrate. Two options:

- **A1 (recommended):** simple regex over the commit message — keyword bank `{fix, fixes, fixed, correction, corrects, revert, wrong, incorrect, regression}`. Deterministic. Lives in `tests/bench_gate/_commit_intent.py` (test-utility scope, not shipped library code).
- **A2 (deferred):** promote `extract_commit_intent.py` (currently on archon at `~/agentmemory-corpus/scripts/`) into `src/aelfrice/`. Out of scope for #438; tracked separately if A1's accuracy proves insufficient.

A1 is the path the spec ships. A2 is the upgrade path if A1 is the bottleneck.

### Path B — LLM-judge

Opt-in. Provider + model selected at runtime by env vars (`AELFRICE_JUDGE_PROVIDER`, `AELFRICE_JUDGE_MODEL`); when either is unset the path is skipped and only the regex column is reported. The judge receives `(belief_a, belief_b, provenance_url, commit_message_excerpt)` and returns one of `{corrects, refines, unrelated}` plus a one-sentence rationale.

**Tier choice.** The issue body suggested a fast / cheap tier as a default. The operator decision (captured in the #438 triage thread, 2026-05-07) chose a stronger tier so the judge column is defensible standalone — i.e., the judge verdict is itself a credible label, not a noisy boundary check. Cost is higher; the eval runs are infrequent (one per implementation PR + one per quarterly recalibration), so absolute cost is low. The specific model id is set in the env var, not pinned in this memo, so model upgrades do not require a spec amendment.

**Determinism caveat.** The LLM judge is **not deterministic**. The eval runs the judge with `temperature=0` and reports the verdict; bench-gate reproducibility relies on the regex path (Path A), not the judge. If the judge produces different verdicts on two runs against identical fixture rows, that is logged as judge-instability and is itself a metric.

---

## Disagreement metric

For every fixture pair `(A, B)` with fixture label `L_fix`, regex verdict `L_regex`, and judge verdict `L_judge`:

| Metric | Definition |
|---|---|
| Regex P/R/F1 | Standard, against `L_fix`. |
| Judge P/R/F1 | Standard, against `L_fix`. Reported but not gated. |
| **Cohen's κ** | Inter-rater agreement between regex and judge. Floor for "judges substantially agree" is **κ ≥ 0.6**. Below that, the bench-gate report flags the corpus or one of the detector paths for review — neither is automatically failed. |
| Raw disagreement count | Number of pairs where `L_regex ≠ L_judge`. Per-codebase + aggregated. |
| Three-way disagreement | Pairs where `L_fix`, `L_regex`, `L_judge` all differ. These are escalated as labeller-review candidates — they are the most informative signal that the corpus row is ambiguous. |

The κ floor is **diagnostic, not gating**: low κ does not fail the bench-gate. The bench-gate floor is the regex P/R per Decision 3 below. The κ + raw disagreement reporting is what makes the dual-scored framing pay rent — without it, the LLM-judge path adds cost with no downstream signal.

---

## Decision 3 — bench-gate floor

| Metric | Floor |
|---|---|
| Regex precision | **P ≥ 0.80** |
| Regex recall | **R ≥ 0.70** |
| Implied F1 | ~0.75 |

Calibration is **balanced** — neither precision-heavy ("never false-flag a correction") nor recall-heavy ("catch every correction, accept noise"). Operator decision captured 2026-05-07. The floor applies to the **aggregated** 250-pair number; per-codebase numbers are reported and may legitimately fall below it (e.g. one codebase that exercises hard cases).

The floor is **adjustable** — if the first calibration pass shows the regex is far from 0.80 P, the bench-gate tightens or loosens at impl-PR review. The number captured here is the operator's starting target.

LLM-judge is not gated. Reported only.

---

## Reconciliation with adjacent work

### vs. existing `tests/bench_gate/test_contradiction.py`

`test_contradiction.py` and `test_contradiction_v3.py` already score `relationship_detector.classify` against a contradiction fixture. The correction eval is the **subset** with historical-supersession evidence. Implementation strategy:

- The correction-eval fixture is its own directory: `tests/corpus/v2_0/correction/`.
- The bench-gate test is its own file: `tests/bench_gate/test_correction_detection.py`.
- Code reuse: shared scoring helpers in `tests/bench_gate/_correction_scoring.py` (test utilities, not library code) wrap `relationship_detector.classify` + `value_compare.find_conflicts` and produce the composite verdict.

The contradiction eval continues to score the broader pair-classification task; the correction eval scores the high-value subset.

### vs. `correction.py` (utterance detector)

`correction.py:121` (`detect_correction`) is a **single-string classifier**: "is this input text a correction directive?" (e.g. "no, do X instead of Y"). Different surface — operates on incoming user prose, not on belief pairs. The two share the word "correction" but no code path. The correction-detection eval does not consume `correction.py` output.

### vs. composition tracker (#154)

#154's tracker doc enumerates retrieval-time projections and lanes. The correction-detection eval is **not a lane** — it is a quality measurement on existing substrate. The tracker gains a row for the eval (input shape, output shape, bench verdict), parallel to the doc-linker entry (#435).

---

## Acceptance

### A1 — corpus

`tests/corpus/v2_0/correction/{mlflow,cockroach,terraform,rustls,micropython}/pairs.jsonl` lives in lab repo, 50 rows each (250 total). Schema per "Schema" above. Public CI runs with `AELFRICE_CORPUS_ROOT` unset and skips via `bench_gated`. Django pilot (lab branch `corpus/issue-438-correction-detection`) is rebased onto this schema and retained as a sixth-codebase footnote (relabel `contradicts` → `corrects`, add `commit_intent` column).

### A2 — bench-gate test

`tests/bench_gate/test_correction_detection.py` exists. It: loads each codebase's `pairs.jsonl`, runs the regex path's composite verdict over each pair, computes per-codebase + aggregated `(P, R, F1)`, asserts aggregated `P ≥ 0.80` AND `R ≥ 0.70`. Optionally — when the judge env vars (`AELFRICE_JUDGE_PROVIDER`, `AELFRICE_JUDGE_MODEL`, plus the provider's API key) are set — runs the LLM-judge path and emits the disagreement metric block.

### A3 — judge harness

`tests/bench_gate/_correction_judge.py` (test utility, not shipped library) implements the judge call with `temperature=0`, structured output (the three-class enum), and result caching keyed on `(provider, model, fixture-row-id, content-hash(belief_a, belief_b))` so reruns are free. Cache lives at `~/.cache/aelfrice/correction-judge-cache.json`; invalidated on model rev or content change.

### A4 — composition tracker entry

`docs/composition-tracker.md` (#154) gains a row for `correction-detection-eval`: input = (belief_a, belief_b, provenance), output = (regex_verdict, judge_verdict, P, R, F1, κ), bench verdict = `pass`/`fail`/`skipped`.

### A5 — `aelf bench all` integration

The bench-gate harness's `aelf bench all` enumerator picks up `test_correction_detection.py` automatically (existing pattern; no new wiring required). Manual verification step in the impl PR: run `uv run aelf bench all` with `AELFRICE_CORPUS_ROOT=/path/to/lab/corpus` set, confirm the suite includes the correction-detection numbers.

---

## Bench-gate / ship-or-defer policy

`needs-spec` → `bench-gated` once this memo lands. Implementation is the next gate. **The implementation PR ships only on the regex path meeting Decision 3 floors against the 250-pair corpus.** A4 (tracker entry) and A5 (harness integration) are mechanical wiring; A1 + A2 + Decision 3 are the ship gate.

If the regex path falls below the floor:

1. **First response: relabel + retest.** Inspect three-way-disagreement rows for fixture mislabels; corrected fixture re-runs.
2. **Second response: tighten the regex.** Specific failure modes drive substrate work in `relationship_detector.py` / `value_compare.py`. Each substrate change is its own PR; the eval is the regression check.
3. **Third response: lower the floor.** Operator decision; the spec memo is updated with the revised floor and the reasoning.

The corpus itself does not regress: rows added later strictly grow the fixture, never replace earlier rows (per the Django pilot's `_v0_1.jsonl` versioning convention).

---

## Out of scope at v2.0.0

- **`LABEL_CORRECTS` in `relationship_detector.py`.** Substrate ships `{contradicts, refines, unrelated}` only. The eval's `corrects` label is at the fixture level; promoting it to substrate is a separate spec.
- **Promoting archon's `extract_commit_intent.py` into `src/aelfrice/`.** Test-utility regex (Path A1 above) carries the eval; the production-grade extractor stays archon-side until a non-eval consumer needs it.
- **Per-codebase bench-gate floors.** Aggregated number gates; per-codebase reported only.
- **Cross-codebase pair-mining at fixture-authoring time.** Each codebase's pairs come from that codebase's own commit history; we do not synthesize cross-codebase corrections.
- **The judge column being deterministic.** `temperature=0` is best-effort; the judge cache (A3) provides reproducibility, but a fresh run on a new model rev will produce fresh verdicts. The regex path is the deterministic source of truth.
- **Embedding-based pair retrieval.** Pair selection is by commit-history walk + filter, not by similarity search. Embedding-free retrieval posture is preserved.

---

## Implementation prereqs

- `src/aelfrice/relationship_detector.py:73-78, :322` — verdict labels + `classify` entry point.
- `src/aelfrice/value_compare.py:194, :267` — `extract_values` + `find_conflicts`.
- `tests/corpus/v2_0/contradiction/django_v0_1.jsonl` — schema precedent (relabel target).
- `tests/bench_gate/test_contradiction.py`, `tests/bench_gate/test_contradiction_v3.py` — sibling bench-gate patterns to mirror.
- `tests/bench_gate/__init__.py` — autouse `bench_gated` marker.
- `tests/conftest.py` — `AELFRICE_CORPUS_ROOT` resolution.
- archon `~/agentmemory-corpus/public/{aiml/mlflow, database/cockroach, devops/terraform, security/rustls, embedded/micropython}` — corpus source repos (lab-side authoring; not consumed by public CI).
- `~/projects/aelfrice-lab/.claude/worktrees/438-corpus` — existing pilot branch for the Django seed; rebased into the new schema during the impl PR.

All substrate is on `main`. No new dependencies — the judge harness uses whichever LLM-provider SDK is already a dev dep when this lands; if a fresh provider is added later, that is its own dep PR. One new test file. No public-CI behavioural change.

---

## Open questions for review

1. **Judge cache location.** `~/.cache/aelfrice/correction-judge-cache.json` — single global cache vs per-corpus cache vs per-PR cache. The default is global because rerun cost is what the cache exists to absorb; per-corpus split is only needed if labellers want to run with cache disabled per pair. Confirm at impl-PR review.
2. **Sourcery / CodeRabbit on judge prompt text.** The judge prompt itself is content; if it lands in the public repo (in `_correction_judge.py`) it goes through the same scanners as any source. Flag if the prompt mentions the labeller's name or any internal codename. (Sanity-check; the prompt is tooling-level and should be clean by construction.)
3. **κ threshold.** 0.6 ("substantial agreement") is the Landis-Koch convention. Stricter (0.8 = "near-perfect") flags more rows for review and grows labeller workload; looser (0.4) catches less. 0.6 ships; revisit at first run.
4. **Django pilot — sixth codebase footnote vs full sixth codebase.** Pilot has 30 rows; the per-codebase target is 50. Treat as 30/50 or grow Django to 50? Default is the footnote treatment (30 rows, not gating per-codebase report); grow if the per-codebase number is the operator's primary signal. Confirm.
5. **Commit-message language coverage.** A1 regex bank is English-only. Codebases with non-English commit messages (none in the picked five, but a future addition) need either translation or a multi-lingual bank. Out of scope for v2.0.0; flag in the issue if a non-English codebase is added later.
