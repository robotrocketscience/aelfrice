# EXP-001: BM25 parameter tuning (`k1`, `b`)

**Status:** shipped. Phase 1 landed at v1.5.0 as `src/aelfrice/bm25.py` — not the FTS5-candidate reranker sketched in Option 3 below, but a standalone BM25F sparse-matvec index (configurable `k1` and `b`, anchor-text support) that replaces the FTS5 lane entirely when `use_bm25f_anchors=True` (default-on since v1.7.0). Phase 2 (the 25-cell `(k1, b)` sweep + calibrated defaults) has not been run on the public corpus as a single experiment. This memo is retained as the original design + negative finding write-up.
**Origin:** v1 research-brief Section C — *"BM25 has tunable parameters (k1, b) that almost everyone leaves at defaults. The defaults are tuned for general web search (long documents, varied lengths). Behavioral directives are short; high `b` (length normalization) probably hurts you. This is concrete, low-risk experimentation worth doing."*
**Question:** Does retuning `k1` and `b` for aelfrice's typical short, behavioural-directive content improve MRR / hit@k on the synthetic harness without harming latency?

## Hypothesis (from the brief)

FTS5's BM25 defaults are `k1=1.2`, `b=0.75`, calibrated for general web search (variable-length documents, ~2 kB typical). aelfrice beliefs are short — typically a single behavioural directive of 50–250 characters. Aggressive length normalization (`b=0.75`) treats a 100-char belief as proportionally "longer" than the corpus average and penalizes its term-frequency contribution. Lower `b` (toward 0) should improve ranking on a short-document corpus.

## Empirical finding: SQLite FTS5 does not expose `k1` and `b`

Per the SQLite FTS5 documentation and verified empirically against `sqlite3 3.51.3`:

```python
sqlite3.connect(':memory:').execute(
    "SELECT c, bm25(t, 1.5, 0.25) FROM t WHERE t MATCH 'quick'"
)
```

Extra arguments to `bm25()` are interpreted as **per-column weights**, not as `k1` / `b`. From SQLite's docs:

> The BM25 algorithm includes a number of constants that are commonly known as "k1" and "b". Consult the documentation for these constants if necessary. … SQLite's FTS5 BM25 implementation hardcodes these constants. They are not configurable through the FTS5 API.

This makes the brief's "concrete, low-risk experimentation" infeasible *as the brief framed it*. To tune `k1` / `b`, aelfrice would need to:

1. Patch the SQLite source — incompatible with stdlib-only commitment.
2. Ship a custom SQLite build with FTS5 modifications — same incompatibility.
3. **Implement BM25 ranking in Python over the FTS5 candidate set** — feasible within the project's principles; non-trivial in implementation.
4. Replace BM25 with a different lexical scorer aelfrice maintains — large scope change, and the alternatives (TF-IDF, language-model scoring) have known weaknesses BM25 was designed to fix.

Option 3 is the only path consistent with aelfrice's architectural commitments.

## Proposed implementation: Python-side BM25 reranker (Option 3)

Sketch:

```python
# pseudocode — original sketch; shipped at src/aelfrice/bm25.py (v1.5.0)
@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75

def rerank_bm25(
    candidates: list[Belief],
    query_terms: list[str],
    corpus_stats: CorpusStats,   # avgdl, doc_freqs, n_docs
    params: BM25Params,
) -> list[tuple[Belief, float]]:
    """Score each candidate against the query under (k1, b).

    Score formula (Robertson & Walker 1994):
        sum over t in q ∩ d:
            idf(t) * (f(t,d) * (k1 + 1)) /
                    (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
    """
```

The retrieval path becomes:

1. FTS5 returns top-N candidates ordered by FTS5's hardcoded BM25.
2. Python reranker re-scores those candidates under `(k1, b)`.
3. Final ordering uses Python scores.

**Cost estimates:**

- Implementation: ~150–250 LOC + tests. One day of focused work.
- Latency: O(N · |query|) where N is the FTS5 candidate set size (default 20). For typical queries this is microseconds; the reranker itself is not the bottleneck.
- Correctness: requires fresh `avgdl`, `n_docs`, and per-term `doc_freqs` from the SQLite store. These are derivable from `beliefs_fts` directly via the FTS5 `*_meta` virtual tables, or maintained in a small derived stats table updated on writes.
- New principle violations: none. Stdlib only. No new dependencies. Determinism contract preserved (every score is a function of the corpus and the parameters; reproducible bit-for-bit).

**Risk:**

- Doubling the ranking work means two BM25 implementations in the codebase (FTS5's hardcoded one for candidate selection; aelfrice's Python one for reranking). Drift between them on edge cases (Unicode normalization, stopword behavior) is real.
- Mitigation: the Python reranker uses FTS5's tokenizer output as input where possible — same tokens, different scoring math. Tests pin the agreement between the two implementations on a fixed corpus.

## Sweep design (when implementation lands)

A small grid sweep against the synthetic harness (16-belief × 16-query corpus):

```
k1 ∈ {0.6, 0.9, 1.2, 1.5, 1.8}
b  ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

25 cells. Report `mrr`, `hit_at_1`, `hit_at_3`, `hit_at_5`, `p99_latency_ms` per cell. Heatmap output in `benchmarks/results/exp-001-bm25-params.json`. Baseline cell `(1.2, 0.75)` matches FTS5 defaults and serves as the reference.

Two outcomes considered acceptable:

1. **Win at lower `b`.** New defaults ship in `BM25Params`, regression test pins the synthetic-harness numbers, the parameters are a tunable in the retrieve API, and the SQLite-side ordering stays as candidate selection only. Documented in CHANGELOG and BENCHMARKS.md.
2. **No clear win.** New defaults match `(1.2, 0.75)`, the reranker still ships (because the brief's framing — that we should at minimum *be able* to control these — is sound), and the experiment closes with a documented null result.

Either outcome is publishable. The failure mode is implementing the reranker without the sweep, leaving `(k1, b)` exposed but uncalibrated.

## Outcome

Phase 1 shipped at v1.5.0: `src/aelfrice/bm25.py` is a BM25F module with configurable `k1` / `b`, anchor-text support, and sparse matvec. Default-on as part of the BM25F retrieval lane (`use_bm25f_anchors`). Phase 2 (the 25-cell `(k1, b)` sweep + calibrated defaults on the synthetic + academic corpora) has not been run as a single labeled experiment under this EXP number; the BM25F module ships with `(k1=1.5, b=0.75)` defaults (`DEFAULT_K1` / `DEFAULT_B` in `src/aelfrice/bm25.py`), unchanged since the module's introduction at v1.5.0 — note `k1` already differs from FTS5's hardcoded `(1.2, 0.75)`.

If you re-open this experiment, the Phase 2 work is: enable the existing knobs, run the sweep against `benchmarks/results/v3.0.1.json` baselines, and either pin a new default or close out with a documented null result.

## What this memo establishes

- The brief was wrong about FTS5 exposing `k1` and `b`. The empirical proof is captured at `python -c "..."` above, reproducible against any standard `sqlite3`.
- The experiment is still valuable; it is just bigger than the brief estimated.
- The path forward was consistent with the then-current stdlib-only commitment as sketched; the implementation that actually shipped at v1.5.0 added numpy + scipy runtime deps (snowballstemmer at v1.7.0) — see ARCHITECTURE § principles for the current dependency posture. Determinism holds.
- This memo is the formal write-up of a finding the brief did not anticipate; the file lives in `docs/experiments/` so future research-driven proposals have a home.

## References

- Robertson & Walker (1994), *Some simple effective approximations to the 2-Poisson model* — original BM25.
- SQLite FTS5 documentation: <https://www.sqlite.org/fts5.html> (search for `bm25` — the per-column weights API is documented; `k1` / `b` configurability is not).
- v1 research-brief §C, `aelfrice-lab/v1 feedback and survey/aelfrice_research_brief.md`.
- Synthetic harness: `src/aelfrice/benchmark.py`.
