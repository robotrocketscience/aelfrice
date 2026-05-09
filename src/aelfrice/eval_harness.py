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
    precision_at_k,
    roc_auc,
    spearman_rho,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

__all__ = (
    "DEFAULT_CALIBRATION_CORPUS",
    "DEFAULT_K",
    "DEFAULT_SEED",
    "load_calibration_fixtures",
    "build_calibration_store",
    "run_calibration_on_fixtures",
    "format_calibration_report",
)

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
            demotion_pressure=0,
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
