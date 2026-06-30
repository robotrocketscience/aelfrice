"""Characterization guard for the pollution-recovery benchmark (#1011).

Pins the deterministic baseline finding from the #1011 doc-chunk R&D:
under a store flooded with keyword-overlapping document chunks, user
facts survive retrieval in the *lexical* and *entity* regimes but NOT in
the *weak* regime (no shared term or entity → BM25 never retrieves them).
This both documents the regimes and guards against silent regression in
the tractable ones.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.pollution_recovery.run import (
    evaluate,
    load_cases,
    regime_recall,
)

_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "pollution_recovery" / "fixtures" / "default.jsonl"
)


def _baseline():
    return evaluate(load_cases(_FIXTURES), k=5)


def test_entity_regime_survives_pollution() -> None:
    """Entity-match facts are recovered by the L2.5 entity index."""
    assert regime_recall(_baseline())["entity"] == 1.0


def test_weak_regime_drowns() -> None:
    """Weak-match facts (no shared term/entity) are not retrieved at all —
    the BM25 recall limit a rerank/recency lane cannot fix (#1011 R&D)."""
    rep = _baseline()
    assert regime_recall(rep)["weak"] == 0.0
    weak_ranks = [c.fact_rank for c in rep.cases if c.regime == "weak"]
    assert weak_ranks and all(r >= 999 for r in weak_ranks)


def test_top_k_is_mostly_document_chunks() -> None:
    """The drowning symptom: injected context is dominated by chunks."""
    assert _baseline().chunk_share_at_k >= 0.8


def test_evaluate_is_deterministic() -> None:
    cases = load_cases(_FIXTURES)
    a = evaluate(cases, k=5)
    b = evaluate(cases, k=5)
    assert a.fact_recall_at_k == b.fact_recall_at_k
    assert [c.fact_rank for c in a.cases] == [c.fact_rank for c in b.cases]


def test_fixture_covers_three_regimes() -> None:
    regimes = {c["regime"] for c in load_cases(_FIXTURES)}
    assert regimes == {"lexical", "entity", "weak"}
