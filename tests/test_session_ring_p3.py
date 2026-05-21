"""Unit tests for #876 P3 session-ring state slots — PR 2 of 5."""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.session_ring import (
    _normalize_for_session,
    append_ids,
    push_classification,
    read_ring_file,
    read_ring_state,
    update_bytes_at_last_fire,
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


# --- backward compat: pre-#876 ring files ---------------------------


def test_normalize_defaults_missing_p3_fields() -> None:
    old = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 5, "evicted_total": 0,
    }
    n = _normalize_for_session(old, "s1", 200)
    assert n["bytes_at_last_fire"] == 0
    assert n["classifications"] == []


def test_normalize_preserves_p3_fields_when_present() -> None:
    new = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 5, "evicted_total": 0,
        "bytes_at_last_fire": 12_345,
        "classifications": [True, False, True],
    }
    n = _normalize_for_session(new, "s1", 200)
    assert n["bytes_at_last_fire"] == 12_345
    assert n["classifications"] == [True, False, True]


def test_normalize_rejects_negative_bytes_at_last_fire() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "bytes_at_last_fire": -10, "classifications": [],
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["bytes_at_last_fire"] == 0


def test_normalize_rejects_bool_bytes_at_last_fire() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "bytes_at_last_fire": True, "classifications": [],
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["bytes_at_last_fire"] == 0


def test_normalize_filters_non_bool_classifications() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "bytes_at_last_fire": 0,
        "classifications": [True, "false", None, False, 1],
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["classifications"] == [True, False]


def test_normalize_classifications_not_a_list_resets_to_empty() -> None:
    bad = {
        "session_id": "s1", "ring": [], "ring_max": 200,
        "next_fire_idx": 0, "evicted_total": 0,
        "bytes_at_last_fire": 0, "classifications": "not a list",
    }
    n = _normalize_for_session(bad, "s1", 200)
    assert n["classifications"] == []


# --- update_bytes_at_last_fire ---------------------------------------


def test_update_bytes_at_last_fire_persists(db_root: Path) -> None:
    append_ids("s1", ["b1", "b2"])
    assert update_bytes_at_last_fire("s1", 50_000)
    state = read_ring_state("s1")
    assert state["bytes_at_last_fire"] == 50_000


def test_update_bytes_at_last_fire_overwrites(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    update_bytes_at_last_fire("s1", 10_000)
    update_bytes_at_last_fire("s1", 25_000)
    state = read_ring_state("s1")
    assert state["bytes_at_last_fire"] == 25_000


def test_update_bytes_at_last_fire_creates_ring_for_new_session(db_root: Path) -> None:
    assert update_bytes_at_last_fire("s-fresh", 12_345)
    state = read_ring_state("s-fresh")
    assert state["session_id"] == "s-fresh"
    assert state["bytes_at_last_fire"] == 12_345


def test_update_bytes_at_last_fire_rejects_empty_session_id(db_root: Path) -> None:
    assert not update_bytes_at_last_fire("", 100)
    assert not update_bytes_at_last_fire(None, 100)


def test_update_bytes_at_last_fire_rejects_negative(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    assert not update_bytes_at_last_fire("s1", -1)
    state = read_ring_state("s1")
    assert state["bytes_at_last_fire"] == 0


def test_update_bytes_at_last_fire_preserves_ring(db_root: Path) -> None:
    append_ids("s1", ["b1", "b2", "b3"])
    state_before = read_ring_state("s1")
    update_bytes_at_last_fire("s1", 8000)
    state_after = read_ring_state("s1")
    assert [e["id"] for e in state_after["ring"]] == ["b1", "b2", "b3"]
    assert state_after["next_fire_idx"] == state_before["next_fire_idx"]


# --- push_classification ---------------------------------------------


def test_push_classification_appends(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    push_classification("s1", True, window_cap=10)
    push_classification("s1", False, window_cap=10)
    push_classification("s1", True, window_cap=10)
    state = read_ring_state("s1")
    assert state["classifications"] == [True, False, True]


def test_push_classification_fifo_evicts_at_cap(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    for i in range(12):
        push_classification("s1", i % 2 == 0, window_cap=5)
    state = read_ring_state("s1")
    assert len(state["classifications"]) == 5
    # i = 7..11: F,T,F,T,F (7=odd=F, 8=even=T, 9=F, 10=T, 11=F)
    assert state["classifications"] == [False, True, False, True, False]


def test_push_classification_rejects_empty_session_id(db_root: Path) -> None:
    assert not push_classification("", True, window_cap=10)
    assert not push_classification(None, True, window_cap=10)


def test_push_classification_rejects_non_positive_cap(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    assert not push_classification("s1", True, window_cap=0)
    assert not push_classification("s1", True, window_cap=-5)


def test_push_classification_creates_ring_for_new_session(db_root: Path) -> None:
    assert push_classification("s-fresh", True, window_cap=10)
    state = read_ring_state("s-fresh")
    assert state["classifications"] == [True]


def test_push_classification_preserves_ring(db_root: Path) -> None:
    append_ids("s1", ["b1", "b2"])
    state_before = read_ring_state("s1")
    push_classification("s1", True, window_cap=10)
    state_after = read_ring_state("s1")
    assert [e["id"] for e in state_after["ring"]] == ["b1", "b2"]
    assert state_after["next_fire_idx"] == state_before["next_fire_idx"]


# --- Cross-session safety --------------------------------------------


def test_new_session_resets_p3_state(db_root: Path) -> None:
    push_classification("s1", True, window_cap=10)
    update_bytes_at_last_fire("s1", 99_999)
    append_ids("s2", ["b1"])
    state = read_ring_state("s2")
    assert state["classifications"] == []
    assert state["bytes_at_last_fire"] == 0


# --- append_ids round-trips P3 state ---------------------------------


def test_append_ids_preserves_p3_state(db_root: Path) -> None:
    append_ids("s1", ["b1"])
    update_bytes_at_last_fire("s1", 5000)
    push_classification("s1", True, window_cap=10)
    push_classification("s1", False, window_cap=10)
    append_ids("s1", ["b2", "b3"])
    state = read_ring_state("s1")
    assert state["bytes_at_last_fire"] == 5000
    assert state["classifications"] == [True, False]


# --- read_ring_file (no session match) returns full shape ------------


def test_read_ring_file_includes_p3_fields(db_root: Path) -> None:
    push_classification("s1", True, window_cap=10)
    update_bytes_at_last_fire("s1", 42_000)
    raw = read_ring_file()
    assert raw["session_id"] == "s1"
    assert raw["bytes_at_last_fire"] == 42_000
    assert raw["classifications"] == [True]
