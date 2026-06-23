"""Unit tests for #980 phantom-generation session-ring state.

Covers the per-session fire counter, dedup-key list, and CONTRADICTS
snapshot added to the session ring, plus their backward-compat defaults
for pre-#980 ring files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.session_ring import (
    _normalize_for_session,
    read_phantom_state,
    record_phantom_fire,
    update_phantom_contradicts,
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


# --- backward compat: pre-#980 ring files ---------------------------


def test_normalize_defaults_missing_phantom_fields() -> None:
    old = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 5, "evicted_total": 0,
    }
    n = _normalize_for_session(old, "s1", 200)
    assert n["phantom_fires"] == 0
    assert n["phantom_dedup"] == []
    assert n["phantom_contradicts"] == []


def test_normalize_preserves_phantom_fields_when_present() -> None:
    new = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "phantom_fires": 2,
        "phantom_dedup": ["gap:foo", "new_entity:bar"],
        "phantom_contradicts": ["a|b"],
    }
    n = _normalize_for_session(new, "s1", 200)
    assert n["phantom_fires"] == 2
    assert n["phantom_dedup"] == ["gap:foo", "new_entity:bar"]
    assert n["phantom_contradicts"] == ["a|b"]


def test_normalize_rejects_bad_phantom_fields() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "phantom_fires": -3,
        "phantom_dedup": "not-a-list",
        "phantom_contradicts": [1, "ok", None],
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["phantom_fires"] == 0
    assert n["phantom_dedup"] == []
    assert n["phantom_contradicts"] == ["ok"]


def test_normalize_fresh_session_resets_phantom_fields() -> None:
    stored = {
        "session_id": "old", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "phantom_fires": 9, "phantom_dedup": ["gap:x"],
        "phantom_contradicts": ["a|b"],
    }
    n = _normalize_for_session(stored, "new", 200)
    assert n["phantom_fires"] == 0
    assert n["phantom_dedup"] == []
    assert n["phantom_contradicts"] == []


# --- read default when no ring exists -------------------------------


def test_read_phantom_state_empty_when_absent(db_root: Path) -> None:
    state = read_phantom_state("s1")
    assert state == {
        "phantom_fires": 0,
        "phantom_dedup": [],
        "phantom_contradicts": [],
        "phantom_init": False,
    }


def test_update_phantom_contradicts_sets_init_flag(db_root: Path) -> None:
    assert read_phantom_state("s1")["phantom_init"] is False
    update_phantom_contradicts("s1", ["a|b"])
    assert read_phantom_state("s1")["phantom_init"] is True


def test_phantom_init_resets_on_new_session() -> None:
    stored = {
        "session_id": "old", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "phantom_init": True, "phantom_contradicts": ["a|b"],
    }
    n = _normalize_for_session(stored, "new", 200)
    assert n["phantom_init"] is False


def test_read_phantom_state_empty_for_none_session(db_root: Path) -> None:
    assert read_phantom_state(None)["phantom_fires"] == 0


# --- record_phantom_fire --------------------------------------------


def test_record_phantom_fire_increments_and_dedups(db_root: Path) -> None:
    assert record_phantom_fire("s1", "gap:foo") is True
    s = read_phantom_state("s1")
    assert s["phantom_fires"] == 1
    assert s["phantom_dedup"] == ["gap:foo"]

    assert record_phantom_fire("s1", "new_entity:bar") is True
    s = read_phantom_state("s1")
    assert s["phantom_fires"] == 2
    assert s["phantom_dedup"] == ["gap:foo", "new_entity:bar"]


def test_record_phantom_fire_counter_bumps_even_on_repeat_key(db_root: Path) -> None:
    record_phantom_fire("s1", "gap:foo")
    record_phantom_fire("s1", "gap:foo")
    s = read_phantom_state("s1")
    # Counter increments each call; dedup list stays unique.
    assert s["phantom_fires"] == 2
    assert s["phantom_dedup"] == ["gap:foo"]


def test_record_phantom_fire_rejects_empty_key(db_root: Path) -> None:
    assert record_phantom_fire("s1", "") is False
    assert read_phantom_state("s1")["phantom_fires"] == 0


def test_record_phantom_fire_rejects_none_session(db_root: Path) -> None:
    assert record_phantom_fire(None, "gap:foo") is False


# --- update_phantom_contradicts -------------------------------------


def test_update_phantom_contradicts_sorts_and_dedups(db_root: Path) -> None:
    assert update_phantom_contradicts("s1", ["b|c", "a|b", "a|b", ""]) is True
    s = read_phantom_state("s1")
    assert s["phantom_contradicts"] == ["a|b", "b|c"]


def test_update_phantom_contradicts_replaces_snapshot(db_root: Path) -> None:
    update_phantom_contradicts("s1", ["a|b"])
    update_phantom_contradicts("s1", ["c|d"])
    assert read_phantom_state("s1")["phantom_contradicts"] == ["c|d"]


def test_update_phantom_contradicts_rejects_non_list(db_root: Path) -> None:
    assert update_phantom_contradicts("s1", "a|b") is False  # type: ignore[arg-type]


def test_phantom_fire_and_contradicts_coexist(db_root: Path) -> None:
    record_phantom_fire("s1", "gap:foo")
    update_phantom_contradicts("s1", ["a|b"])
    s = read_phantom_state("s1")
    assert s["phantom_fires"] == 1
    assert s["phantom_dedup"] == ["gap:foo"]
    assert s["phantom_contradicts"] == ["a|b"]
