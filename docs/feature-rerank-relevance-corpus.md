# Feature spec: labeled rerank-relevance corpus (#819)

**Status:** scaffold landed; v0_1 row delivery pending (operator-time)
**Issue:** [#819](https://github.com/robotrocketscience/aelfrice/issues/819)
**Unblocks:** [#769](https://github.com/robotrocketscience/aelfrice/issues/769),
[#724](https://github.com/robotrocketscience/aelfrice/issues/724),
[#800 R5](https://github.com/robotrocketscience/aelfrice/issues/800),
[#817](https://github.com/robotrocketscience/aelfrice/issues/817) flip-default,
feature-ablation harness (lab Layer 2)
**Module path:** `tests/corpus/v2_0/rerank_relevance/`
**Bench-gate smoke:** `tests/bench_gate/test_rerank_relevance.py`
**Schema enforcement:** `tests/test_corpus_schema.py` (`MODULES["rerank_relevance"]`)

---

## Purpose

Carry a labeled set of `(query, candidate-belief-pool, gold-relevance)` triples
so the four downstream bench-gates above can be load-bearing. Each row asserts:
*for this `query` against this pool of beliefs, here is the labeller-judged
relevant top-K*. Consumers compute precision@k / recall@k / RBO / rank-changed
fraction against that gold set.

The four consumers each measure something different, but all of them need the
same row shape. Building one well-labeled corpus is cheaper than four
single-purpose corpora.

---

## Row shape

```jsonl
{
  "id": "rerank-relevance-<short>-<NNN>",
  "provenance": "synthetic-v0.1 | <transcript-hash> | <store-export-hash>",
  "labeller_note": "one-line rationale for the label",
  "label": "graded",
  "query": "natural language query",
  "beliefs": [
    {"id": "b-aaa", "text": "..."},
    {"id": "b-bbb", "text": "..."},
    ...
  ],
  "gold_top_k": ["b-aaa", "b-xxx"],
  "k": 10,
  "gold_ordering": ["b-aaa", "b-xxx", "b-yyy", ...]   // optional
}
```

Field contracts (also enforced by `tests/test_corpus_schema.py`):

- `id` — stable unique id within the module file. Conventional prefix
  `rerank-relevance-<short>-<NNN>`.
- `provenance` — non-empty. Synthetic rows must match `synthetic-vN.M`
  (e.g. `synthetic-v0.1`). Real-store exports use a content hash; never
  embed user-identifying strings or `~/.claude/`-sourced text (see
  *Discretion* below).
- `labeller_note` — non-empty, one-line. Justifies the label. Audited
  when the bench gate's outcome is contested.
- `label` — always `"graded"`. Existing convention for ranking modules.
- `query` — natural-language input to `retrieve_v2()`.
- `beliefs` — non-empty list of `{"id": str, "text": str}`. The candidate
  pool for this query. Belief ids must be unique within the row. Pool
  size 20–200 is the v0_1 target; >500 is operator-time-prohibitive to
  label.
- `gold_top_k` — non-empty list of belief ids labelled relevant. Set
  semantics (order ignored by the relevance gate). Every id must appear
  in `beliefs`. The labeller picks `gold_top_k` first; `k` follows.
- `k` — integer ≥ 1. The rank cutoff for the consumer. `len(gold_top_k)`
  ≤ k is the convention but not enforced (sparse labelling is allowed
  when fewer than k beliefs are relevant).
- `gold_ordering` — optional, ordered list. Present only when a
  *defensible* full preference order is assignable; absent (or `null`)
  when the labeller can rank the top-K but cannot order the remainder.
  Must contain every `gold_top_k` id. Consumers that need pairwise
  preferences (e.g. ζ rerank's RBO axis) read this; consumers that need
  only set-membership (`use_type_aware_compression` flip-default A2)
  ignore it.

The smoke harness `tests/bench_gate/test_rerank_relevance.py` enforces
membership + dedup invariants; downstream consumer tests add their own
threshold checks.

---

## Where rows live

Per the v2.0 corpus convention (`tests/corpus/v2_0/README.md` §
"Mounting on the lab side"):

- **Public CI** runs with `AELFRICE_CORPUS_ROOT` unset. The bench-gate
  smoke test and downstream consumers skip cleanly. The schema test
  in `tests/test_corpus_schema.py` skips when the module dir is empty.
- **Private-corpus runs** mount the corpus via `AELFRICE_CORPUS_ROOT` per
  the existing instructions in `tests/corpus/v2_0/README.md` § "Mounting
  on the lab side"; the bench harness then reads under
  `$AELFRICE_CORPUS_ROOT/rerank_relevance/`.

**Synthetic-only rows may live in the public tree** (`tests/corpus/v2_0/rerank_relevance/*.jsonl`)
when scrubbed per the checklist below. The issue body envisioned
`tests/corpus/v2_0/rerank_relevance/v0_1.jsonl` as a public file. That
is acceptable for purely synthetic rows; **any row touched by real
aelfrice-store content stays lab-side**.

---

## Labeling protocol (v0_1)

Target: **50–100 rows** for first batch. Estimated **~5 min/row** of
operator time → ~4–8 hours of labeller-time for v0_1 delivery.

1. **Select a query.** Draw from one of:
   - *Synthetic queries* with known ground-truth answers. These can
     live public-tree.
   - *Real aelfrice-usage queries* captured private-corpus-side from
     the operator's own store. Private-corpus-only — never enter the
     public tree.
2. **Assemble the candidate pool.** 20–200 beliefs. Mix in 3–5
   *intentional* distractors per row to keep precision-at-k informative.
   For real-store rows the pool comes from `retrieve_v2(query, k=200)`
   on the store; for synthetic rows the labeller writes them.
3. **Pick `gold_top_k`.** The beliefs the labeller would expect a
   well-ranked retrieval to surface in the top-K. Independent of any
   particular rerank function — this is the operator-truth.
4. **Optionally pick `gold_ordering`.** Only when a defensible full
   preference order exists (top-K plus tie-broken ordering of the
   rest). Omit when the labeller can only rank the top-K confidently.
5. **Write `labeller_note`.** One line. Future you re-reading this in
   six months will thank present you for naming the cluster, axis, or
   atom under test.

Subsequent batches may be agent-assisted (machine pre-labelling, operator
audits the rows in bulk). v0_2 may add inter-judge κ verification once
two labellers exist.

---

## Discretion / directory-of-origin checklist

Every row entering the public tree must pass:

- [ ] No content sourced from the operator's private-tools directory
      (memory files, handoff docs, session transcripts, lab notes). The
      boundary is the directory of origin, **not** transformation —
      paraphrasing a private-tools artifact still counts as derived
      content.
- [ ] No PII in belief `text` or `query` strings. Operator names, email
      addresses, internal hostnames, repo paths, ticket numbers from
      private trackers are all out.
- [ ] No identifier-vocabulary leaks from the project's internal
      tooling. The authoritative deny-list is in
      `.git/hooks/pre-push` (`BANNED_VOCAB` + `BANNED_PHRASES`); the
      pre-push hook will block the commit before it lands on the
      remote.
- [ ] Belief ids are synthetic or hashed; do not reuse production
      belief uuids.
- [ ] `provenance` does not embed identifying paths. Synthetic →
      `synthetic-vN.M`. Real-store → SHA256 prefix of the exported
      content.

Lab-side rows skip the public checklist but still must not embed
private-tools-directory content per the same locked boundary.

Run the same regex the pre-push hook uses, against the staged file,
before committing:

```bash
# Patterns live in .git/hooks/pre-push. Source them, then grep the file.
PATTERN=$(awk -F= '/^BANNED_VOCAB=/{print substr($0, index($0,"=")+2, length-index($0,"=")-2)}' \
    .git/hooks/pre-push)
grep -niE "$PATTERN" tests/corpus/v2_0/rerank_relevance/*.jsonl
```

Empty → safe to commit. Hits → sanitize and re-grep. The hook is the
source of truth; if you sanitize against an older copy of the
deny-list, the hook will still reject the push.

---

## Acceptance vs #819

The issue lists four boxes:

- [x] **Scaffold:** module dir, README row, schema validator entry,
      bench-gate smoke harness, protocol doc.
- [ ] **v0_1 corpus committed.** Operator-time work; this PR does not
      ship rows.
- [ ] **Downstream consumer wired.** Either #769 (A2 recall@k) or #817
      (ζ rerank R5) connects to the corpus rows and clears OR fails.
      Out of scope for the scaffold — lands with the consumer's own PR.
- [ ] **Schema validation in CI.** `tests/test_corpus_schema.py` already
      parametrises over `MODULES` and the `rerank_relevance` entry rides
      that machinery; the test skips while the module is empty and
      activates per-row when rows land.

---

## Why a separate module

Could this reuse `retrieve_uplift` (#154)? No — `retrieve_uplift` measures
flip-default NDCG@k uplift on a per-flag basis with an *ordered*
`expected_top_k`. The rerank-relevance consumers split into two camps:

- ζ rerank / γ rerank / hot-path consumers need RBO and rank-changed
  fraction, which want a *partial* gold ordering (top-K labelled,
  remainder ignored). `expected_top_k` as an ordered hard contract is
  the wrong fit.
- `use_type_aware_compression` A2 needs *set-membership* recall@k — no
  ordering required. `retrieve_uplift` is over-specified for that case.

`gold_top_k` (set) + optional `gold_ordering` (full order) covers both
camps with one row. The bench gates pick the field that matches their
axis.

---

## Open questions

- **Multi-rater inter-judge κ harness.** Deferred to v0_2 if the first
  bench-gate verdict is operator-contested. Single-labeller is the
  v0_1 posture; the audit trail lives in `labeller_note`.
- **Synthetic-generation pipeline.** Deferred to v0_2. v0_1 is
  hand-curated.
- **Full-corpus retrieval-quality measurement.** Out of scope — this
  module measures rerank-relevance on pre-filtered candidate pools, not
  end-to-end retrieval. Recall@k is BM25-bounded by assumption.
