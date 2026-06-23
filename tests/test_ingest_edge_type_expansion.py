"""Ingest-wiring tests for the #999 SUPPORTS/SUPERSEDES edge writers.

Verifies the default-off gate: with AELFRICE_AUTO_RELATIONSHIPS unset,
ingest writes no SUPPORTS/SUPERSEDES edges (byte-identical to today); with
it on, the two new writers run alongside the #988 CONTRADICTS writer.

All tests use a real ``MemoryStore(":memory:")`` and the real
``ingest_turn`` path — no mocks.
"""
from __future__ import annotations

import pytest

from aelfrice.ingest import ingest_turn
from aelfrice.models import EDGE_SUPERSEDES, EDGE_SUPPORTS
from aelfrice.store import MemoryStore

# REFINES (mutual-agreement) pair — same subject, agreeing modality, but
# too lexically distant for dedup's levenshtein floor (so SUPPORTS, not
# SUPERSEDES).
_ALWAYS = "the deployment script always runs the database migration step"
_ALWAYS_LAUNCH = _ALWAYS + " before the production launch window opens"
# Near-duplicate paraphrases — clear dedup thresholds → SUPERSEDES cluster.
_BASE = "never push commits directly to the main branch"
_BASE_NOW = _BASE + " now"
_BASE_HERE = _BASE + " here"


def _edge_count(store: MemoryStore, edge_type: str) -> int:
    row = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM edges WHERE type = ?", (edge_type,)
    ).fetchone()
    return int(row[0])


def test_ingest_off_writes_no_new_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_AUTO_RELATIONSHIPS", raising=False)
    s = MemoryStore(":memory:")
    for text in (_ALWAYS, _ALWAYS_LAUNCH, _BASE, _BASE_NOW, _BASE_HERE):
        ingest_turn(s, text, source="t", session_id="sess")
    assert _edge_count(s, EDGE_SUPPORTS) == 0
    assert _edge_count(s, EDGE_SUPERSEDES) == 0


def test_ingest_on_writes_supports_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_AUTO_RELATIONSHIPS", "1")
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _ALWAYS_LAUNCH, source="t", session_id="sess")
    assert _edge_count(s, EDGE_SUPPORTS) >= 1
    # Not a near-duplicate cluster → no SUPERSEDES.
    assert _edge_count(s, EDGE_SUPERSEDES) == 0


def test_ingest_on_writes_supersedes_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_AUTO_RELATIONSHIPS", "1")
    s = MemoryStore(":memory:")
    ingest_turn(s, _BASE, source="t", session_id="sess")
    ingest_turn(s, _BASE_NOW, source="t", session_id="sess")
    ingest_turn(s, _BASE_HERE, source="t", session_id="sess")
    assert _edge_count(s, EDGE_SUPERSEDES) >= 1
