"""Per-flag retrieve() uplift harness for #154 default-on flip.

Loads `tests/corpus/v2_0/retrieve_uplift/*.jsonl` rows. For each row:
build a transient `MemoryStore` from the row's beliefs + edges,
then run `retrieve(store, query, k=row["k"], ...)` once with each
v1.7 flag toggled on while the others stay at default-off, plus
once with all flags off (the baseline). Score each result list with
NDCG@k against `expected_top_k`. Per-flag uplift is the mean
delta over the corpus.

Five flags exercised:

- `use_bm25f_anchors` (#148)
- `use_signed_laplacian` (#149) — placeholder; warns if true
- `use_heat_kernel` (#150) — wired via `heat_kernel_enabled`
- `use_posterior_ranking` (#151) — wired via non-zero `posterior_weight`
- `use_hrr_structural` (#152) — placeholder; warns if true

The placeholders that haven't yet plumbed into `retrieve()` will
trivially produce uplift=0 — the harness reports that as evidence
that the flag is a no-op until the underlying lane lands.

Two consumers:

- `tests/bench_gate/test_retrieve_uplift.py` — bench-gate test
  asserting no per-flag uplift regression (uplift ≥ 0).
- Lab-side ad-hoc inspection via
  `python -m tests.retrieve_uplift_runner --corpus-root <path>`
  prints the per-flag NDCG table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    RETENTION_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.query_understanding import (
    LEGACY_STRATEGY,
    STACK_R1_R3_STRATEGY,
    transform_query,
)
from aelfrice.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    retrieve,
    retrieve_v2,
    retrieve_with_tiers,
)
from aelfrice.store import MemoryStore

_TS = "2026-05-05T00:00:00+00:00"

# Five v1.7 flags. Each entry maps the flag name to a `kwargs` lambda
# that produces the kwargs to pass to `retrieve()` when the flag is
# toggled ON. Flags not in this dict are kept at their default-off
# state. `use_signed_laplacian` and `use_hrr_structural` are
# placeholder-only in main today — listed here so the harness records
# their no-op status; once the lanes land in `retrieve()`, only this
# table needs the new wire.
FLAG_KWARGS: dict[str, Callable[[], dict]] = {  # type: ignore[type-arg]
    "use_bm25f_anchors": lambda: {"use_bm25f_anchors": True},
    "use_signed_laplacian": lambda: {},  # placeholder; warning-only flag
    "use_heat_kernel": lambda: {"heat_kernel_enabled": True},
    "use_posterior_ranking": lambda: {"posterior_weight": 0.5},
    "use_hrr_structural": lambda: {},  # placeholder; warning-only flag
}

# Baseline: all flags off. Heat-kernel and BFS off; posterior weight
# zero so the BM25-only ordering is the comparison floor.
BASELINE_KWARGS: dict = {  # type: ignore[type-arg]
    "use_bm25f_anchors": False,
    "heat_kernel_enabled": False,
    "posterior_weight": 0.0,
    "bfs_enabled": False,
    "entity_index_enabled": True,  # default-on already; not part of #154
}


def _default_k(row: dict) -> int:  # type: ignore[type-arg]
    return int(row.get("k", 10))


def _belief_from_row(b: dict) -> Belief:  # type: ignore[type-arg]
    """Build a Belief from a corpus row's belief dict.

    Required: `id`, `content`. Optional: `type`, `alpha`, `beta`.
    Defaults match the factual/agent-inferred shape.
    """
    return Belief(
        id=b["id"],
        content=b["content"],
        content_hash=f"corpus:{b['id']}",
        alpha=float(b.get("alpha", 1.0)),
        beta=float(b.get("beta", 1.0)),
        type=b.get("type", BELIEF_FACTUAL),
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=_TS,
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _edge_from_row(e: dict) -> Edge:  # type: ignore[type-arg]
    return Edge(
        src=e["src"],
        dst=e["dst"],
        type=e["type"],
        weight=float(e.get("weight", 0.5)),
        anchor_text=e.get("anchor_text"),
    )


def ndcg_at_k(
    result_ids: list[str],
    expected_top_k: list[str],
    k: int,
) -> float:
    """Graded NDCG@k.

    `expected_top_k` is the ground-truth ranking, top first. Relevance
    of the i-th expected belief (0-indexed) is `k - i` (so the top
    expected belief has the highest relevance, the k-th has rel=1,
    anything outside has rel=0).

    Returns 0.0 when expected_top_k is empty (NDCG undefined; treat
    as zero-uplift signal).
    """
    if not expected_top_k:
        return 0.0
    rel: dict[str, int] = {
        bid: max(0, k - i) for i, bid in enumerate(expected_top_k[:k])
    }
    dcg = 0.0
    for i, bid in enumerate(result_ids[:k], start=1):
        r = rel.get(bid, 0)
        if r:
            dcg += r / math.log2(i + 1)
    ideal_rels = sorted(rel.values(), reverse=True)
    idcg = sum(
        r / math.log2(i + 1) for i, r in enumerate(ideal_rels, start=1)
    )
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True)
class FlagUplift:
    """Per-flag NDCG@k summary."""
    flag: str
    n_rows: int
    mean_ndcg_off: float
    mean_ndcg_on: float

    @property
    def uplift(self) -> float:
        return self.mean_ndcg_on - self.mean_ndcg_off


def _seed_store(store: MemoryStore, row: dict) -> None:  # type: ignore[type-arg]
    for b in row.get("beliefs", []):
        store.insert_belief(_belief_from_row(b))
    for e in row.get("edges", []):
        store.insert_edge(_edge_from_row(e))


def _retrieve_ids(
    store: MemoryStore, query: str, k: int, **flag_kwargs,  # type: ignore[no-untyped-def]
) -> list[str]:
    kwargs = {**BASELINE_KWARGS, **flag_kwargs}
    results = retrieve(store, query, l1_limit=k, **kwargs)
    return [b.id for b in results[:k]]


_db_counter = [0]


def _row_ndcg(
    row: dict,  # type: ignore[type-arg]
    k: int,
    flag_kwargs: dict,  # type: ignore[type-arg]
    tmp_root: Path,
) -> float:
    """One retrieve() call against a fresh store; return NDCG@k.

    Each call gets its own SQLite path (monotonic counter) so adjacent
    rows can't bleed state across the comparison and identical
    `flag_kwargs` shapes (e.g. two placeholder flags) don't collide.
    """
    _db_counter[0] += 1
    db_path = tmp_root / f"row_{row['id']}_{_db_counter[0]}.db"
    store = MemoryStore(str(db_path))
    try:
        _seed_store(store, row)
        result_ids = _retrieve_ids(
            store, row["query"], k, **flag_kwargs,
        )
    finally:
        store.close()
    return ndcg_at_k(result_ids, row.get("expected_top_k", []), k)


def run_per_flag_uplift(
    rows: list[dict],  # type: ignore[type-arg]
) -> list[FlagUplift]:
    """Drive the corpus through retrieve() for each flag on/off.

    Per row, makes one baseline call (all flags off) plus one call per
    flag turned ON (others off). Mean NDCG@k computed across all
    rows, per arm.
    """
    out: list[FlagUplift] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for flag, kwargs_fn in FLAG_KWARGS.items():
            kwargs_on = kwargs_fn()
            off_total = 0.0
            on_total = 0.0
            for row in rows:
                k = _default_k(row)
                off_total += _row_ndcg(row, k, {}, tmp_root)
                on_total += _row_ndcg(row, k, kwargs_on, tmp_root)
            n = len(rows)
            out.append(FlagUplift(
                flag=flag,
                n_rows=n,
                mean_ndcg_off=off_total / n if n else 0.0,
                mean_ndcg_on=on_total / n if n else 0.0,
            ))
    return out


# ---------------------------------------------------------------------
# Intentional clustering (#436) — multi_fact bench gate (spec § A2).
#
# Consumed by tests/bench_gate/test_intentional_clustering.py via
# `pytest.importorskip("tests.retrieve_uplift_runner")` then
# `runner_mod.run_clustering_uplift(rows)`. Activates on any
# AELFRICE_CORPUS_ROOT pointing at a `multi_fact/*.jsonl` row set.
#
# Doc-linker (#435) symmetric runner lives further below
# (`run_doc_linker_uplift`). It implements the spec § A2 contract:
# anchors-OFF baseline (anchors NOT written) vs anchors-ON case
# (anchors written via `link_belief_to_document`), both with
# `with_doc_anchors=True`. Today the doc-anchor projection is read-only
# (it does not influence ordering), so the gate reports flat uplift
# until a downstream rerank lands; that is the correct gate-broken
# signal — strictly-positive uplift is the ship trigger.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClusteringUplift:
    """Spec § A2 result shape for #436.

    Bench-gate test asserts ``cluster_coverage_uplift > 0`` and
    formats the OFF/ON/uplift triple in its failure message; recall@k
    fields are reported alongside as the don't-degrade backstop.
    """

    n_rows: int
    mean_recall_off: float
    mean_recall_on: float
    cluster_coverage_off: float
    cluster_coverage_on: float

    @property
    def cluster_coverage_uplift(self) -> float:
        return self.cluster_coverage_on - self.cluster_coverage_off

    @property
    def recall_uplift(self) -> float:
        return self.mean_recall_on - self.mean_recall_off


def _cluster_coverage_at_k(
    returned: list[str],
    expected_clusters: list[list[str]],
    n_clusters_required: int,
) -> float:
    """Spec § A2: number of expected clusters represented in the top-K
    divided by ``n_clusters_required``. A cluster is represented if
    at least one of its expected member ids appears in the returned
    top-K. Returns 0.0 when ``n_clusters_required`` is non-positive
    (degenerate row; treated as zero-uplift signal)."""
    if n_clusters_required <= 0:
        return 0.0
    rs = set(returned)
    covered = sum(
        1 for cluster in expected_clusters
        if any(bid in rs for bid in cluster)
    )
    return covered / n_clusters_required


def _recall_at_k(returned: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.0
    return len(set(returned) & set(expected)) / len(expected)


# Tight enough that ~2 multi_fact beliefs at ~30-50 tokens each fit;
# spec § A2 calls for "fixed budget that forces the pack to choose
# between cluster representatives." Override via the CLI / programmatic
# kwarg if a corpus has a different per-belief size.
DEFAULT_MULTI_FACT_BUDGET: int = 100


def run_clustering_uplift(
    rows: list[dict],  # type: ignore[type-arg]
    *,
    budget: int = DEFAULT_MULTI_FACT_BUDGET,
    k: int | None = None,
) -> ClusteringUplift:
    """Spec § A2 multi-fact uplift driver.

    Per row: build a transient ``MemoryStore`` from the row's
    ``beliefs`` + ``edges``, run ``retrieve_v2`` once with
    ``use_intentional_clustering=False`` and once with ``=True`` at the
    same ``budget``. ``k`` defaults to ``row['n_clusters_required']``
    so the metric asks "did the top-K cover the required clusters."
    """
    n = len(rows)
    if n == 0:
        return ClusteringUplift(0, 0.0, 0.0, 0.0, 0.0)

    rec_off = rec_on = cov_off = cov_on = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            row_k = k if k is not None else int(row.get("n_clusters_required", 2))
            expected = list(row["expected_belief_ids"])
            clusters = list(row["expected_clusters"])
            required = int(row["n_clusters_required"])

            for clustering_on in (False, True):
                _db_counter[0] += 1
                db = tmp_root / f"clust_{row['id']}_{int(clustering_on)}_{_db_counter[0]}.db"
                store = MemoryStore(str(db))
                try:
                    _seed_store(store, row)
                    result = retrieve_v2(
                        store, row["query"],
                        budget=budget,
                        use_entity_index=False,
                        use_intentional_clustering=clustering_on,
                    )
                    returned = [b.id for b in result.beliefs[:row_k]]
                finally:
                    store.close()
                if clustering_on:
                    rec_on += _recall_at_k(returned, expected)
                    cov_on += _cluster_coverage_at_k(returned, clusters, required)
                else:
                    rec_off += _recall_at_k(returned, expected)
                    cov_off += _cluster_coverage_at_k(returned, clusters, required)

    return ClusteringUplift(
        n_rows=n,
        mean_recall_off=rec_off / n,
        mean_recall_on=rec_on / n,
        cluster_coverage_off=cov_off / n,
        cluster_coverage_on=cov_on / n,
    )


# ---------------------------------------------------------------------
# Doc / semantic linker (#435) — doc_linker bench gate (spec § A2).
#
# Consumed by tests/bench_gate/test_doc_linker.py.
# Row schema (`tests/corpus/v2_0/doc_linker/*.jsonl`):
#
#   {
#     "id": "row-id",
#     "query": "raw query string",
#     "k": 10,
#     "beliefs": [...],                     # seed beliefs (same shape as
#                                            # other modules in this file)
#     "edges": [...],                       # seed edges (optional)
#     "expected_top_k": ["b1", "b2", ...],  # ground-truth ranking
#     "anchors": [                          # written under anchors-ON
#       {
#         "belief_id": "b1",
#         "doc_uri":   "file:docs/X.md",
#         "anchor_type": "ingest",         # optional; defaults "ingest"
#         "position_hint": null             # optional
#       }
#     ]
#   }
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DocLinkerUpliftResults:
    """Spec § A2 result shape for #435.

    Field names mirror ``FlagUplift`` so the bench-gate failure-message
    formatter can read ``mean_ndcg_off`` / ``mean_ndcg_on`` / ``uplift``
    without per-runner branching.
    """

    n_rows: int
    mean_ndcg_off: float
    mean_ndcg_on: float

    @property
    def uplift(self) -> float:
        return self.mean_ndcg_on - self.mean_ndcg_off


def _seed_doc_anchors(store: MemoryStore, row: dict) -> None:  # type: ignore[type-arg]
    """Write each anchor in ``row['anchors']`` via the doc-linker module.

    Using the public ``link_belief_to_document`` (not the raw store call)
    keeps the gate honest: any ingest-time validation a future revision
    adds (URI shape, anchor-type checks) is exercised here too.
    """
    from aelfrice.doc_linker import link_belief_to_document

    for a in row.get("anchors", []):
        link_belief_to_document(
            store,
            belief_id=a["belief_id"],
            doc_uri=a["doc_uri"],
            anchor_type=a.get("anchor_type", "ingest"),
            position_hint=a.get("position_hint"),
        )


def run_doc_linker_uplift(
    rows: list[dict],  # type: ignore[type-arg]
) -> DocLinkerUpliftResults:
    """Spec § A2 doc-linker uplift driver.

    Per row, runs ``retrieve_v2`` twice with ``with_doc_anchors=True`` on
    fresh stores: once without anchors written (the OFF arm) and once
    with each row's ``anchors`` populated via ``link_belief_to_document``
    (the ON arm). Same query, same seed beliefs/edges, same ``k``. NDCG@k
    is scored against ``expected_top_k`` and averaged across rows.

    The doc-anchor projection is read-only at v2.0.0 — it surfaces
    ``DocAnchor`` rows alongside beliefs but does not influence ranking.
    On a corpus where ranking is invariant to anchor presence, ``uplift``
    will be ~0 and the bench gate's strict ``> 0`` assertion will fail.
    That is the correct gate-broken signal until a downstream rerank
    consumes the projection.
    """
    n = len(rows)
    if n == 0:
        return DocLinkerUpliftResults(0, 0.0, 0.0)

    off_total = 0.0
    on_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            k = _default_k(row)
            expected = list(row.get("expected_top_k", []))

            for anchors_on in (False, True):
                _db_counter[0] += 1
                db = (
                    tmp_root
                    / f"doc_{row['id']}_{int(anchors_on)}_{_db_counter[0]}.db"
                )
                store = MemoryStore(str(db))
                try:
                    _seed_store(store, row)
                    if anchors_on:
                        _seed_doc_anchors(store, row)
                    result = retrieve_v2(
                        store, row["query"],
                        budget=DEFAULT_TOKEN_BUDGET,
                        use_entity_index=False,
                        with_doc_anchors=True,
                    )
                    returned = [b.id for b in result.beliefs[:k]]
                finally:
                    store.close()
                ndcg = ndcg_at_k(returned, expected, k)
                if anchors_on:
                    on_total += ndcg
                else:
                    off_total += ndcg

    return DocLinkerUpliftResults(
        n_rows=n,
        mean_ndcg_off=off_total / n,
        mean_ndcg_on=on_total / n,
    )


# ---------------------------------------------------------------------
# Query-strategy (#291 / #527) — legacy-bm25 vs stack-r1-r3 bench gate.
#
# Consumed by tests/bench_gate/test_query_strategy.py.
# Row schema (`tests/corpus/v2_0/query_strategy/*.jsonl`):
#
#   {
#     "id": "row-id",
#     "query": "raw query string",         # FTS5 MATCH input pre-rewrite
#     "k": 10,
#     "beliefs": [...],                     # seed beliefs (shared shape)
#     "edges": [...],                       # seed edges (optional)
#     "expected_top_k": ["b1", "b2", ...]   # ground-truth ranking
#   }
#
# OFF arm: transform_query(raw, store, "legacy-bm25") -> raw query
#          unchanged, retrieve(store, raw, ...).
# ON arm:  transform_query(raw, store, "stack-r1-r3") -> R1 entity
#          expand + R3 per-store IDF-quantile clip, retrieve(store,
#          rewritten, ...).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class QueryStrategyUplift:
    """#291 / #527 result shape.

    Field names mirror ``FlagUplift`` / ``DocLinkerUpliftResults`` so the
    bench-gate failure formatter reads ``mean_ndcg_off`` / ``mean_ndcg_on``
    / ``uplift`` without per-runner branching.
    """

    n_rows: int
    mean_ndcg_off: float
    mean_ndcg_on: float

    @property
    def uplift(self) -> float:
        return self.mean_ndcg_on - self.mean_ndcg_off


def run_query_strategy_uplift(
    rows: list[dict],  # type: ignore[type-arg]
) -> QueryStrategyUplift:
    """#291 § Mechanism / Bench gates query-strategy uplift driver.

    Per row, runs ``retrieve()`` twice on a fresh seeded store:

    * OFF arm — ``legacy-bm25`` (``transform_query`` passthrough; the raw
      ``row['query']`` is the FTS5 MATCH input).
    * ON arm  — ``stack-r1-r3`` (R1 capitalised-token entity expand →
      R3 per-store IDF-quantile clip → joined whitespace term list).

    Same seed beliefs/edges, same ``k``, same ``BASELINE_KWARGS`` to
    ``retrieve()`` (so any future flag-default change in this module
    affects both arms identically). NDCG@k is scored against
    ``expected_top_k`` and averaged across rows.

    Degenerate case: a row whose query has no capitalised tokens AND
    whose tokens fall inside the per-store IDF quantile band produces
    an identical transformed query for both strategies, so the OFF and
    ON arms see identical retrieval state and ``uplift == 0`` by
    construction. Falsifiable in the unit test at
    ``tests/test_retrieve_uplift_runner.py``.
    """
    n = len(rows)
    if n == 0:
        return QueryStrategyUplift(0, 0.0, 0.0)

    off_total = 0.0
    on_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            k = _default_k(row)
            expected = list(row.get("expected_top_k", []))

            for strategy_on in (False, True):
                _db_counter[0] += 1
                strategy = (
                    STACK_R1_R3_STRATEGY if strategy_on else LEGACY_STRATEGY
                )
                db = (
                    tmp_root
                    / f"qs_{row['id']}_{int(strategy_on)}_{_db_counter[0]}.db"
                )
                store = MemoryStore(str(db))
                try:
                    _seed_store(store, row)
                    rewritten = transform_query(
                        row["query"], store, strategy,
                    )
                    result_ids = _retrieve_ids(store, rewritten, k)
                finally:
                    store.close()
                ndcg = ndcg_at_k(result_ids, expected, k)
                if strategy_on:
                    on_total += ndcg
                else:
                    off_total += ndcg

    return QueryStrategyUplift(
        n_rows=n,
        mean_ndcg_off=off_total / n,
        mean_ndcg_on=on_total / n,
    )


@dataclass(frozen=True)
class QueryStrategyLatency:
    """#291 § Bench gates per-rebuild p99 latency contract.

    Fields hold per-arm p99 latency in nanoseconds for the
    ``transform_query → retrieve`` path (the only span that differs
    between ``legacy-bm25`` and ``stack-r1-r3``; everything downstream
    in the rebuild path is strategy-invariant). The bench gate asserts
    ``p99_on_ns <= p99_off_ns + 5_000_000`` (5 ms).
    """

    n_rows: int
    reps_per_row: int
    p99_off_ns: int
    p99_on_ns: int

    @property
    def delta_ns(self) -> int:
        return self.p99_on_ns - self.p99_off_ns


def _p99_ns(samples: list[int]) -> int:
    """Deterministic p99 of an integer-ns sample list.

    Uses the "k-th largest" convention: with ``n`` samples sorted
    ascending, returns ``sorted[int(0.99 * n)]`` clamped into range.
    For ``n == 100`` this yields ``sorted[99]`` (the top 1 sample);
    for ``n == 600`` this yields ``sorted[594]`` (top 1%).
    """
    n = len(samples)
    if n == 0:
        return 0
    s = sorted(samples)
    idx = min(n - 1, int(0.99 * n))
    return s[idx]


def run_query_strategy_latency(
    rows: list[dict],  # type: ignore[type-arg]
    reps_per_row: int = 20,
) -> QueryStrategyLatency:
    """#291 § Bench gates per-rebuild p99 latency driver.

    Per row, builds the transient ``MemoryStore`` once, then times
    ``reps_per_row`` repetitions of ``transform_query → retrieve`` for
    each strategy. One warmup repetition per arm is discarded before
    the timed loop so JIT-free Python import + first-call caching does
    not skew the first sample.

    Aggregation: all ``n_rows * reps_per_row`` timed samples per arm
    feed into a single p99. Same-store/same-row pairing keeps the two
    arms measuring identical work modulo the strategy switch.
    """
    n = len(rows)
    if n == 0:
        return QueryStrategyLatency(0, reps_per_row, 0, 0)

    off_samples: list[int] = []
    on_samples: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            k = _default_k(row)
            for strategy_on in (False, True):
                _db_counter[0] += 1
                strategy = (
                    STACK_R1_R3_STRATEGY if strategy_on else LEGACY_STRATEGY
                )
                db = (
                    tmp_root
                    / f"qslat_{row['id']}_{int(strategy_on)}_{_db_counter[0]}.db"
                )
                store = MemoryStore(str(db))
                try:
                    _seed_store(store, row)
                    # Warmup (discard).
                    rewritten = transform_query(
                        row["query"], store, strategy,
                    )
                    _retrieve_ids(store, rewritten, k)
                    # Timed reps.
                    for _ in range(reps_per_row):
                        t0 = time.perf_counter_ns()
                        rewritten = transform_query(
                            row["query"], store, strategy,
                        )
                        _retrieve_ids(store, rewritten, k)
                        t1 = time.perf_counter_ns()
                        (on_samples if strategy_on else off_samples).append(
                            t1 - t0,
                        )
                finally:
                    store.close()

    return QueryStrategyLatency(
        n_rows=n,
        reps_per_row=reps_per_row,
        p99_off_ns=_p99_ns(off_samples),
        p99_on_ns=_p99_ns(on_samples),
    )


# ---------------------------------------------------------------------
# Type-aware compression A2 (#434) — compression_a2_recall bench gate.
#
# Consumed by tests/bench_gate/test_compression_a2_recall.py.
# Row schema (`tests/corpus/v2_0/compression_a2_recall/*.jsonl`):
#
#   {
#     "id": "row-id",
#     "query": "natural language query",
#     "k": 12,
#     "token_budget": 250,
#     "beliefs": [
#       {"id": "...", "content": "...",
#        "retention_class": "fact" | "snapshot" | "transient" | "unknown",
#        "lock_level": "none" | "user"}
#     ],
#     "expected_top_k": ["belief-id-1", ...]
#   }
#
# Differs from `compression_uplift` (the upstream-invariant gate that
# ships at tests/bench_gate/test_compression_uplift.py): this module's
# rows carry queries and expected ranking, the gate measures
# recall@k(ON) > recall@k(OFF) at fixed token_budget, and the runner
# drives full L0+L1 retrieval rather than calling compress_for_retrieval
# per row.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionA2Uplift:
    """Spec § A2 result shape for #434.

    Field names mirror ``DocLinkerUpliftResults`` / ``QueryStrategyUplift``
    so the bench-gate failure-message formatter can read ``mean_recall_off``
    / ``mean_recall_on`` / ``uplift`` without per-runner branching.
    """

    n_rows: int
    mean_recall_off: float
    mean_recall_on: float

    @property
    def uplift(self) -> float:
        return self.mean_recall_on - self.mean_recall_off


def _a2_belief_from_row(b: dict) -> Belief:  # type: ignore[type-arg]
    """Build a Belief from a compression_a2_recall row's belief dict.

    Differs from ``_belief_from_row`` by honouring the row's
    ``retention_class`` and ``lock_level`` — the two fields that
    determine compression strategy under the A2 arm.
    """
    lock = LOCK_USER if str(b.get("lock_level", "none")) == "user" else LOCK_NONE
    return Belief(
        id=b["id"],
        content=b["content"],
        content_hash=f"corpus:{b['id']}",
        alpha=float(b.get("alpha", 1.0)),
        beta=float(b.get("beta", 1.0)),
        type=b.get("type", BELIEF_FACTUAL),
        lock_level=lock,
        locked_at=_TS if lock == LOCK_USER else None,
        demotion_pressure=0,
        created_at=_TS,
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
        retention_class=str(b.get("retention_class", RETENTION_UNKNOWN)),
    )


def run_compression_a2_uplift(
    rows: list[dict],  # type: ignore[type-arg]
) -> CompressionA2Uplift:
    """Spec § A2 type-aware compression uplift driver.

    Per row: build a transient ``MemoryStore`` seeded with the row's
    beliefs (locks + retention classes preserved). Call
    ``retrieve_with_tiers`` twice on the same store — once with
    ``use_type_aware_compression=False`` and once with ``=True``. Same
    ``token_budget`` (per-row override or ``DEFAULT_TOKEN_BUDGET``),
    same query. Recall@k against ``expected_top_k`` is averaged
    across rows.

    Both arms disable L2.5 entity index and L3 BFS so the compression
    effect is isolated to the L1 pack-loop budget. ``intentional
    clustering`` is held off both arms (mutually exclusive with
    compression per ``retrieval.py``).

    Degenerate cases the gate-side commentary should expect:
    - All-``fact`` or all-locked rows produce identical OFF/ON arms
      (compression is a no-op on those classes) — uplift contribution
      is exactly zero.
    - Rows whose belief population fits the budget verbatim produce
      OFF == ON because there is no pack-trim under either arm.
    """
    n = len(rows)
    if n == 0:
        return CompressionA2Uplift(0, 0.0, 0.0)

    off_total = 0.0
    on_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for row in rows:
            k = _default_k(row)
            budget = int(row.get("token_budget", DEFAULT_TOKEN_BUDGET))
            expected = list(row.get("expected_top_k", []))

            for compress_on in (False, True):
                _db_counter[0] += 1
                db = (
                    tmp_root
                    / f"a2_{row['id']}_{int(compress_on)}_{_db_counter[0]}.db"
                )
                store = MemoryStore(str(db))
                try:
                    for b in row.get("beliefs", []):
                        store.insert_belief(_a2_belief_from_row(b))
                    result_tuple = retrieve_with_tiers(
                        store, row["query"],
                        token_budget=budget,
                        entity_index_enabled=False,
                        bfs_enabled=False,
                        use_type_aware_compression=compress_on,
                        use_intentional_clustering=False,
                    )
                    returned_ids = [b.id for b in result_tuple[0][:k]]
                finally:
                    store.close()
                recall = _recall_at_k(returned_ids, expected)
                if compress_on:
                    on_total += recall
                else:
                    off_total += recall

    return CompressionA2Uplift(
        n_rows=n,
        mean_recall_off=off_total / n,
        mean_recall_on=on_total / n,
    )


def _format_table(results: list[FlagUplift]) -> str:
    lines = [
        f"{'flag':<28} {'n':>4} {'NDCG_off':>10} {'NDCG_on':>10} {'uplift':>10}",
        "-" * 66,
    ]
    for r in results:
        lines.append(
            f"{r.flag:<28} {r.n_rows:>4} "
            f"{r.mean_ndcg_off:>10.4f} {r.mean_ndcg_on:>10.4f} "
            f"{r.uplift:>+10.4f}"
        )
    return "\n".join(lines)


def _load_corpus(corpus_root: Path) -> list[dict]:  # type: ignore[type-arg]
    mod_dir = corpus_root / "retrieve_uplift"
    if not mod_dir.is_dir():
        return []
    rows: list[dict] = []  # type: ignore[type-arg]
    for p in sorted(mod_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(os.environ.get("AELFRICE_CORPUS_ROOT", "")) or None,
        help="Path to a corpus root containing retrieve_uplift/*.jsonl.",
    )
    args = parser.parse_args()
    if args.corpus_root is None:
        print("AELFRICE_CORPUS_ROOT not set; use --corpus-root", file=sys.stderr)
        return 2
    rows = _load_corpus(args.corpus_root)
    if not rows:
        print(
            f"no rows under {args.corpus_root}/retrieve_uplift/",
            file=sys.stderr,
        )
        return 2
    results = run_per_flag_uplift(rows)
    print(_format_table(results))
    # Exit 1 if any flag is net-negative on average — that's the
    # ship-gate signal: don't flip a flag that regresses NDCG.
    any_regression = any(r.uplift < 0 for r in results)
    return 1 if any_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
