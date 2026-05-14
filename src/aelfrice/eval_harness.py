"""Relevance-calibration harness for the close-the-loop loop (#365 R4).

Lifts the harness logic that previously lived inline in
``scripts/audit_rebuild_log.py`` into a wheel-installable module, so
both the script and the ``aelf eval`` CLI subcommand can call it
without duplicating code.

Public API:

- ``DEFAULT_CALIBRATION_CORPUS`` — bundled synthetic corpus path.
- ``DEFAULT_K`` / ``DEFAULT_SEED`` — defaults.
- ``load_calibration_fixtures(path)`` — fail-soft JSONL loader.
- ``build_calibration_store(fixture, seed)`` — fresh in-memory store.
- ``run_calibration_on_fixtures(fixtures, k, seed)`` — returns a
  ``CalibrationReport``. Raises ``ValueError`` for invalid ``k`` or
  empty fixture list.
- ``format_calibration_report(report, *, corpus_path, seed)`` —
  deterministic human-readable text block.

Determinism contract (#365 ship gate): given the same
``(corpus, k, seed)``, the returned report's metric fields and the
formatted text are bytes-identical across reruns.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from aelfrice.calibration_metrics import (
    CalibrationReport,
    ordered_top_k_overlap,
    precision_at_k,
    rank_biased_overlap,
    roc_auc,
    spearman_rho,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

__all__ = (
    "DEFAULT_CALIBRATION_CORPUS",
    "DEFAULT_K",
    "DEFAULT_SEED",
    "DEFAULT_RBO_PERSISTENCE",
    "load_calibration_fixtures",
    "build_calibration_store",
    "run_calibration_on_fixtures",
    "format_calibration_report",
    "RankComparisonReport",
    "compare_ranking_panel",
    "format_ranking_comparison",
)

# #796 R4 default — top-weight that gives the top-K meaningful weight
# without ignoring tail rearrangements. Chosen so a single transposition
# at rank 1 shifts the score noticeably while a transposition at rank
# 10 does not. Held in this module so the γ-vs-log-additive A/B bench
# and any later harness see the same default.
DEFAULT_RBO_PERSISTENCE: float = 0.9

DEFAULT_CALIBRATION_CORPUS = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks"
    / "posterior_ranking"
    / "fixtures"
    / "default.jsonl"
)
DEFAULT_K = 10
DEFAULT_SEED = 0


def load_calibration_fixtures(path: Path) -> list[dict]:
    """Load a calibration JSONL corpus, fail-soft.

    Each row must be a JSON object with ``id``, ``query``,
    ``known_belief_content``, and a list-typed ``noise_belief_contents``.
    Malformed rows or rows missing required keys are silently skipped —
    same posture as the audit-mode reader.
    """
    out: list[dict] = []
    text = path.read_text(encoding="utf-8")
    required = ("id", "query", "known_belief_content", "noise_belief_contents")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if not all(key in row for key in required):
            continue
        if not isinstance(row["noise_belief_contents"], list):
            continue
        out.append(row)
    return out


def build_calibration_store(fixture: dict, seed: int) -> "MemoryStore":
    """Build a fresh in-memory store seeded with one fixture's beliefs.

    Inserts one belief per content row (one relevant + N noise). Noise
    order is shuffled with ``seed`` so AUC / ρ aggregates are
    deterministic across reruns at fixed seed.
    """
    from aelfrice.models import (  # noqa: PLC0415
        BELIEF_FACTUAL,
        LOCK_NONE,
        Belief,
    )
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(":memory:")
    fid = str(fixture["id"])
    known_content = str(fixture["known_belief_content"])
    noise_contents = list(fixture["noise_belief_contents"])

    def make_belief(bid: str, content: str) -> Belief:
        return Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=0.5,
            beta=0.5,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        )

    rng = random.Random(seed)
    rng.shuffle(noise_contents)

    store.insert_belief(make_belief(f"{fid}_known", known_content))
    for i, nc in enumerate(noise_contents):
        store.insert_belief(make_belief(f"{fid}_noise_{i}", nc))

    return store


def run_calibration_on_fixtures(
    fixtures: Sequence[dict],
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
) -> CalibrationReport:
    """Run the calibration harness over already-loaded fixtures.

    Same shape as the harness used by ``audit_rebuild_log.py
    --calibrate-corpus`` (#365 R1): rank-as-score, un-retrieved
    candidates pooled at score 0, BM25-with-default-posterior-blend
    posture (no L2.5/L3/heat-kernel/entity-index/BFS).

    Raises ``ValueError`` for non-positive ``k`` or empty fixtures.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not fixtures:
        raise ValueError("fixtures must be non-empty")

    from aelfrice.retrieval import retrieve  # noqa: PLC0415

    p_at_k_values: list[float] = []
    n_truncated = 0
    pooled_scores: list[float] = []
    pooled_labels: list[bool] = []

    for fx in fixtures:
        store = build_calibration_store(fx, seed)
        try:
            query = str(fx["query"])
            known_content = str(fx["known_belief_content"])
            noise_contents = list(fx["noise_belief_contents"])
            n_candidates = 1 + len(noise_contents)

            results = retrieve(
                store,
                query,
                l1_limit=max(k, n_candidates),
                entity_index_enabled=False,
                bfs_enabled=False,
                posterior_weight=None,
            )

            relevance_top_k = [b.content == known_content for b in results]
            if len(relevance_top_k) < k:
                n_truncated += 1
            p_at_k_values.append(precision_at_k(relevance_top_k, k))

            for rank_idx, belief in enumerate(results):
                pooled_scores.append(float(len(results) - rank_idx))
                pooled_labels.append(belief.content == known_content)
            retrieved_contents = {b.content for b in results}
            for noise_content in noise_contents:
                if noise_content not in retrieved_contents:
                    pooled_scores.append(0.0)
                    pooled_labels.append(False)
            if known_content not in retrieved_contents:
                pooled_scores.append(0.0)
                pooled_labels.append(True)
        finally:
            store.close()

    avg_p_at_k = sum(p_at_k_values) / len(p_at_k_values)
    auc = roc_auc(pooled_scores, pooled_labels)
    rho = spearman_rho(
        pooled_scores, [1.0 if x else 0.0 for x in pooled_labels],
    )
    return CalibrationReport(
        p_at_k=avg_p_at_k,
        k=k,
        n_queries=len(fixtures),
        n_truncated_queries=n_truncated,
        roc_auc=auc,
        spearman_rho=rho,
        n_observations=len(pooled_scores),
    )


def _format_optional_float(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a (undefined)"


def format_calibration_report(
    report: CalibrationReport, *, corpus_path: Path, seed: int,
) -> str:
    """Format a report as a deterministic human-readable text block."""
    lines = [
        f"calibration harness — corpus {corpus_path.name}",
        f"  n_queries:    {report.n_queries}",
        f"  n_obs:        {report.n_observations}",
        f"  seed:         {seed}",
    ]
    if report.n_truncated_queries:
        lines.append(
            f"  truncated:    {report.n_truncated_queries} "
            f"(query returned <{report.k} candidates)",
        )
    lines.append("")
    lines.append(f"P@{report.k}:        {report.p_at_k:.4f}")
    lines.append(f"ROC-AUC:      {_format_optional_float(report.roc_auc)}")
    lines.append(f"Spearman ρ:   {_format_optional_float(report.spearman_rho)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# #796 R4 rank-comparison panel
# ---------------------------------------------------------------------------
# When comparing two retrieval configurations (γ rerank vs log-additive
# baseline; bm25 vs bm25f; etc.) on the same query corpus, PR@K and
# Spearman ρ measure each config independently against the relevance
# label set. They cannot tell whether the configs produce the same
# top-K order or merely the same top-K set. The R4 campaign added
# ``ordered_top_k_overlap`` and ``rank_biased_overlap`` to discriminate
# top-of-list churn from middle-of-list reorderings; this panel
# averages those metrics across queries and exposes them in a stable
# report shape. The γ-vs-log-additive bench harness is the load-
# bearing consumer.


from dataclasses import dataclass  # noqa: E402 — keep grouping with #796 block


@dataclass(frozen=True)
class RankComparisonReport:
    """Per-query mean of rank-overlap metrics between two retrieval
    configurations on a shared query set.

    ``ordered_top_k`` is the mean of ``ordered_top_k_overlap(a_i, b_i, k)``
    across queries. ``rbo`` is the mean of
    ``rank_biased_overlap(a_i, b_i, p)``. Both are bounded in [0, 1];
    1.0 means the two configs agree at the measured granularity, 0.0
    means they disagree completely.

    ``n_queries`` is the number of (a, b) pairs aggregated. Queries
    where both rankings are empty contribute 1.0 to RBO (vacuous
    identity) but 0.0 to ordered_top_k (no positions to match).
    """

    k: int
    p: float
    n_queries: int
    ordered_top_k: float
    rbo: float


def compare_ranking_panel(
    pairs: Sequence[tuple[Sequence[object], Sequence[object]]],
    *,
    k: int = DEFAULT_K,
    p: float = DEFAULT_RBO_PERSISTENCE,
) -> RankComparisonReport:
    """Compute the #796 rank-comparison panel for paired rankings.

    Each pair ``(a_i, b_i)`` is the ranked-id list from configuration A
    and configuration B on the same query. The two metrics are computed
    per pair and arithmetic-mean-aggregated across pairs. An empty
    ``pairs`` list raises ``ValueError`` (no meaningful average).
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")
    otk_vals: list[float] = []
    rbo_vals: list[float] = []
    for a, b in pairs:
        otk_vals.append(ordered_top_k_overlap(a, b, k))
        rbo_vals.append(rank_biased_overlap(a, b, p=p))
    return RankComparisonReport(
        k=k,
        p=p,
        n_queries=len(pairs),
        ordered_top_k=sum(otk_vals) / len(otk_vals),
        rbo=sum(rbo_vals) / len(rbo_vals),
    )


def format_ranking_comparison(report: RankComparisonReport) -> str:
    """Format a ``RankComparisonReport`` as a deterministic text block.

    Mirrors ``format_calibration_report``'s shape so the two panels
    sit cleanly side-by-side in a combined bench run.
    """
    lines = [
        "rank-comparison panel — γ vs log-additive (#796 R4)",
        f"  n_queries:    {report.n_queries}",
        f"  k:            {report.k}",
        f"  p:            {report.p:.4f}",
        "",
        f"ordered_top_k@{report.k}:  {report.ordered_top_k:.4f}",
        f"RBO(p={report.p:.2f}):      {report.rbo:.4f}",
    ]
    return "\n".join(lines) + "\n"
