"""Unit tests for the #1132 Q2 promotion-opportunity session-ring state.

Covers ``record_promotion_fire`` / ``read_promotion_state`` and the
``_normalize_for_session`` round-trip for the ``promotion_fires`` /
``promotion_dedup`` fields, mirroring ``test_session_ring_phantom.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.session_ring import (
    _normalize_for_session,
    read_promotion_state,
    record_promotion_fire,
)


@pytest.fixture
def db_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    aelf_dir = tmp_path / ".git" / "aelfrice"
    aelf_dir.mkdir(parents=True)
    db = aelf_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    from aelfrice.store import MemoryStore

    MemoryStore(str(db)).close()
    return aelf_dir


# --- normalize round-trip -------------------------------------------------


def test_normalize_defaults_missing_promotion_fields() -> None:
    old = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 5, "evicted_total": 0,
    }
    n = _normalize_for_session(old, "s1", 200)
    assert n["promotion_fires"] == 0
    assert n["promotion_dedup"] == []


def test_normalize_preserves_promotion_fields_when_present() -> None:
    new = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "promotion_fires": 2,
        "promotion_dedup": ["ph1", "ph2"],
    }
    n = _normalize_for_session(new, "s1", 200)
    assert n["promotion_fires"] == 2
    assert n["promotion_dedup"] == ["ph1", "ph2"]


def test_normalize_rejects_malformed_promotion_fields() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "promotion_fires": -3,          # negative -> 0
        "promotion_dedup": "not-a-list",  # wrong type -> []
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["promotion_fires"] == 0
    assert n["promotion_dedup"] == []


# --- record / read round-trip --------------------------------------------


def test_record_and_read_promotion_state(db_root: Path) -> None:
    assert read_promotion_state("s1") == {
        "promotion_fires": 0, "promotion_dedup": []
    }
    assert record_promotion_fire("s1", "ph1") is True
    assert record_promotion_fire("s1", "ph2") is True
    state = read_promotion_state("s1")
    assert state["promotion_fires"] == 2
    assert set(state["promotion_dedup"]) == {"ph1", "ph2"}


def test_record_promotion_fire_counter_increments_but_dedup_is_idempotent(
    db_root: Path,
) -> None:
    record_promotion_fire("s1", "ph1")
    record_promotion_fire("s1", "ph1")  # same key again
    state = read_promotion_state("s1")
    # Counter bumps every call; the dedup list holds the key once.
    assert state["promotion_fires"] == 2
    assert state["promotion_dedup"] == ["ph1"]


def test_record_promotion_fire_rejects_empty_key(db_root: Path) -> None:
    assert record_promotion_fire("s1", "") is False
    assert read_promotion_state("s1")["promotion_fires"] == 0


def test_promotion_state_is_independent_of_phantom_generation(
    db_root: Path,
) -> None:
    """The two opportunity lanes carry separate budgets: a promotion fire must
    not consume the #980 generation budget."""
    from aelfrice.session_ring import read_phantom_state, record_phantom_fire

    record_promotion_fire("s1", "ph1")
    record_phantom_fire("s1", "gap:foo")
    assert read_promotion_state("s1")["promotion_fires"] == 1
    assert read_phantom_state("s1")["phantom_fires"] == 1
