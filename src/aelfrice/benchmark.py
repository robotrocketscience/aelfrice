"""Minimal v0.9.0-rc benchmark harness.

Runs aelfrice retrieval against a small deterministic synthetic
corpus (16 beliefs across 4 topics) and a deterministic query set
(16 queries with one known correct belief each). Produces a single
JSON document containing hit@1 / hit@3 / hit@5, MRR, and latency
percentiles.

This harness is the **measurement instrument**, not a proof of the
central claim. It does not currently differentiate
"with-feedback" vs "no-feedback" because retrieval ranking in
v1.0 is BM25-only (posterior alpha/beta does not factor into
search_beliefs ordering — see store.py:244). A v1.x retrieval
upgrade that consumes posterior is the precondition for using this
harness to claim feedback drives accuracy.

What this harness DOES validate at v0.9.0-rc:
- The full ingest -> retrieve pipeline runs end-to-end against a
  fresh on-disk store.
- A reproducible score is produced. Re-running the harness against
  an empty store yields identical numbers.
- Latency stays below a sane ceiling for the size-16 corpus
  (regression test asserts p99 < 100ms).

Score floor (regression-asserted):
- hit_at_5 >= 0.75 — at least 12 of 16 queries find their correct
  belief in the top 5 results. Anything below means BM25 escaping
  or scoring has regressed.

The corpus, queries, and expected-correct mappings are all
hand-authored. They live in this module rather than a separate
data file because (a) we want the harness to ship in the wheel
without needing package_data plumbing, and (b) review of changes
is just a regular code-review of this file.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Final

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import Store

BENCHMARK_NAME: Final[str] = "aelfrice-bench-v1"
"""Stable identifier for this benchmark configuration.

Bumped (v2, v3, ...) only when corpus / queries / expected mapping
change in a way that breaks score comparability across releases.
Within a v1 series, hit@K numbers are directly comparable.
"""

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_TOKEN_BUDGET: Final[int] = 4000
"""Higher than retrieval's default 2000 so the budget doesn't
cap the harness at fewer than top_k results when belief contents
are long."""


# --- Synthetic corpus -------------------------------------------------


@dataclass(frozen=True)
class _CorpusEntry:
    id: str
    content: str


# 4 topics * 4 beliefs each = 16 beliefs. Topic key embedded in id
# prefix so reviewers can sanity-check the expected-mapping table
# without rerunning the harness.
_CORPUS: Final[tuple[_CorpusEntry, ...]] = (
    # Cooking
    _CorpusEntry("cook_01", "the kitchen knife should be sharpened weekly to maintain a clean cutting edge"),
    _CorpusEntry("cook_02", "boiling water for pasta needs at least one tablespoon of salt per liter"),
    _CorpusEntry("cook_03", "cast iron skillets must be seasoned with oil after every wash to prevent rust"),
    _CorpusEntry("cook_04", "a sourdough starter ferments overnight at room temperature before baking"),
    # Gardening
    _CorpusEntry("gard_01", "tomato plants should be staked when the stem reaches knee height to prevent breakage"),
    _CorpusEntry("gard_02", "compost piles need turning every two weeks to aerate and accelerate decomposition"),
    _CorpusEntry("gard_03", "drip irrigation delivers water directly to plant roots with minimal evaporation loss"),
    _CorpusEntry("gard_04", "raised beds drain better than ground-level plots in clay-heavy soil"),
    # Networking
    _CorpusEntry("net_01", "TCP retransmits lost packets after a timeout based on the round-trip estimate"),
    _CorpusEntry("net_02", "BGP announces routes between autonomous systems across the public internet"),
    _CorpusEntry("net_03", "TLS handshake negotiates a symmetric key using asymmetric certificate exchange"),
    _CorpusEntry("net_04", "subnet masks partition an IP address space into smaller administrative groups"),
    # Programming
    _CorpusEntry("prog_01", "garbage collection reclaims memory from objects no longer reachable from a root set"),
    _CorpusEntry("prog_02", "B-trees keep database indexes balanced so disk page reads stay logarithmic in row count"),
    _CorpusEntry("prog_03", "compilers lower high-level code into machine instructions through a sequence of passes"),
    _CorpusEntry("prog_04", "merge sort recursively splits and combines lists to achieve guaranteed n log n time"),
)


# --- Queries with expected-correct mapping ----------------------------


@dataclass(frozen=True)
class _QueryEntry:
    query: str
    correct_id: str


# Queries are deliberately constrained to terms present (surface-form,
# case-insensitive) in their target belief, because FTS5 with the
# simple tokenizer does not stem and store._escape_fts5_query joins
# whitespace tokens with implicit AND. Single-keyword queries would
# match too many beliefs; 2–4 specific terms gives a sharp signal.
_QUERIES: Final[tuple[_QueryEntry, ...]] = (
    _QueryEntry("kitchen knife sharpened", "cook_01"),
    _QueryEntry("salt pasta water", "cook_02"),
    _QueryEntry("cast iron seasoned", "cook_03"),
    _QueryEntry("sourdough starter overnight", "cook_04"),
    _QueryEntry("tomato plants staked", "gard_01"),
    _QueryEntry("compost turning weeks", "gard_02"),
    _QueryEntry("drip irrigation roots", "gard_03"),
    _QueryEntry("raised beds clay", "gard_04"),
    _QueryEntry("TCP packets timeout", "net_01"),
    _QueryEntry("BGP routes autonomous", "net_02"),
    _QueryEntry("TLS handshake certificate", "net_03"),
    _QueryEntry("subnet masks IP", "net_04"),
    _QueryEntry("garbage collection memory", "prog_01"),
    _QueryEntry("database indexes balanced", "prog_02"),
    _QueryEntry("compilers machine instructions", "prog_03"),
    _QueryEntry("merge sort recursively", "prog_04"),
)


# --- Scored output ----------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    """Reproducible single-run summary."""

    benchmark_name: str
    aelfrice_version: str
    corpus_size: int
    query_count: int
    top_k: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    p50_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# --- Harness ----------------------------------------------------------


def seed_corpus(store: Store, *, created_at: str = "2026-04-26T00:00:00Z") -> int:
    """Insert the deterministic benchmark corpus into `store`.

    Inserts each `_CorpusEntry` as an unlocked factual belief with
    Jeffreys prior (alpha=1, beta=1). Returns the count inserted.
    """
    inserted = 0
    for entry in _CORPUS:
        store.insert_belief(
            Belief(
                id=entry.id,
                content=entry.content,
                content_hash=f"bench_{entry.id}",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                demotion_pressure=0,
                created_at=created_at,
                last_retrieved_at=None,
            )
        )
        inserted += 1
    return inserted


def run_benchmark(
    store: Store,
    *,
    aelfrice_version: str,
    top_k: int = DEFAULT_TOP_K,
) -> BenchmarkReport:
    """Run all queries against `store` and return scored results.

    Caller must have seeded the corpus (e.g. via `seed_corpus`) into
    `store` before calling. The harness does not seed implicitly to
    keep store ownership explicit and to let benchmark variants
    layer additional state (e.g. pre-locked beliefs) on top of the
    base corpus.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for q in _QUERIES:
        t0 = time.perf_counter()
        results = retrieve(
            store, q.query, token_budget=DEFAULT_TOKEN_BUDGET, l1_limit=top_k
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        rank = _rank_of(q.correct_id, results)
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(_QUERIES)
    return BenchmarkReport(
        benchmark_name=BENCHMARK_NAME,
        aelfrice_version=aelfrice_version,
        corpus_size=len(_CORPUS),
        query_count=n,
        top_k=top_k,
        hit_at_1=hits_at_1 / n,
        hit_at_3=hits_at_3 / n,
        hit_at_5=hits_at_5 / n,
        mrr=sum(reciprocal_ranks) / n,
        p50_latency_ms=_percentile(latencies_ms, 0.50),
        p99_latency_ms=_percentile(latencies_ms, 0.99),
    )


def _rank_of(correct_id: str, results: list[Belief]) -> int | None:
    """1-indexed position of `correct_id` in `results`, or None."""
    for i, b in enumerate(results, start=1):
        if b.id == correct_id:
            return i
    return None


def _percentile(values: list[float], q: float) -> float:
    """Inclusive percentile with linear interpolation. q in [0, 1]."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac
