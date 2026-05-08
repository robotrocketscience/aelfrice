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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    retrieve,
    retrieve_v2,
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
# Doc-linker (#435) symmetric runner intentionally NOT here — its A2
# gate requires a downstream rerank that consumes the `with_doc_anchors`
# projection to influence belief order, and that rerank has not shipped.
# When it lands, the same shape applies (load rows → twin retrieve calls
# → mean-NDCG uplift). Filed for a follow-up PR.
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
