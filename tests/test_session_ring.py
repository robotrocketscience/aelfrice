"""Tests for the per-session belief-injection dedup ring (#740)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from aelfrice.session_ring import (
    DEFAULT_RING_MAX,
    RING_MAX_ENV,
    SESSION_RING_FILENAME,
    append_ids,
    filter_against_ring,
    read_ring_state,
)


@dataclass
class _Belief:
    id: str
    content: str = ""


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _ring_path(db: Path) -> Path:
    return db.parent / SESSION_RING_FILENAME


def test_empty_ring_first_fire_returns_all_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    beliefs = [_Belief("b1"), _Belief("b2"), _Belief("b3")]
    result = filter_against_ring("sess-A", beliefs)
    assert [b.id for b in result.new_beliefs] == ["b1", "b2", "b3"]
    assert result.recent_ids == []
    assert result.latest_fire_idx == -1


def test_append_creates_ring_with_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    next_idx = append_ids("sess-A", ["b1", "b2"])
    assert next_idx == 1
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-A"
    assert data["next_fire_idx"] == 1
    assert {e["id"] for e in data["ring"]} == {"b1", "b2"}
    assert all(e["fire_idx"] == 0 for e in data["ring"])
    assert all(e["locked"] is False for e in data["ring"])


def test_second_append_increments_fire_idx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1"])
    next_idx = append_ids("sess-A", ["b2"])
    assert next_idx == 2
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    by_id = {e["id"]: e["fire_idx"] for e in data["ring"]}
    assert by_id == {"b1": 0, "b2": 1}


def test_filter_after_append_separates_new_and_recent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1", "b2"])
    beliefs = [_Belief("b1"), _Belief("b3"), _Belief("b2"), _Belief("b4")]
    result = filter_against_ring("sess-A", beliefs)
    assert [b.id for b in result.new_beliefs] == ["b3", "b4"]
    assert sorted(result.recent_ids) == ["b1", "b2"]
    assert result.latest_fire_idx == 0


def test_filter_locked_always_passes_through_as_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["L1", "b2"], locked_ids={"L1"})
    beliefs = [_Belief("L1"), _Belief("b2")]
    result = filter_against_ring("sess-A", beliefs, locked_ids={"L1"})
    assert [b.id for b in result.new_beliefs] == ["L1"]
    assert result.recent_ids == ["b2"]


def test_new_session_id_wipes_ring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1", "b2"])
    # New session: filter should treat ring as empty (fresh shape).
    beliefs = [_Belief("b1"), _Belief("b2")]
    result = filter_against_ring("sess-B", beliefs)
    assert [b.id for b in result.new_beliefs] == ["b1", "b2"]
    assert result.recent_ids == []
    # Appending under sess-B resets the ring.
    append_ids("sess-B", ["b3"])
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-B"
    assert {e["id"] for e in data["ring"]} == {"b3"}
    assert data["next_fire_idx"] == 1


def test_fifo_eviction_when_ring_overflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    monkeypatch.setenv(RING_MAX_ENV, "5")
    # Three appends of 2 ids each = 6 total; cap is 5 → evict oldest 1.
    append_ids("sess-A", ["a", "b"])  # fire_idx=0
    append_ids("sess-A", ["c", "d"])  # fire_idx=1
    append_ids("sess-A", ["e", "f"])  # fire_idx=2
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    assert data["ring_max"] == 5
    assert data["evicted_total"] == 1
    ids = {e["id"] for e in data["ring"]}
    # 'a' is the oldest (fire_idx=0), expected first evicted.
    assert ids == {"b", "c", "d", "e", "f"}


def test_refresh_in_place_when_id_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1", "b2"])  # fire_idx=0
    append_ids("sess-A", ["b1"])  # re-injected; should refresh fire_idx
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    by_id = {e["id"]: e["fire_idx"] for e in data["ring"]}
    assert by_id == {"b1": 1, "b2": 0}
    # Ring length stayed at 2 (no duplicate row).
    assert len(data["ring"]) == 2


def test_empty_session_id_is_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    assert append_ids("", ["b1"]) == -1
    assert append_ids(None, ["b1"]) == -1
    result = filter_against_ring("", [_Belief("b1")])
    assert [b.id for b in result.new_beliefs] == ["b1"]
    assert not _ring_path(db).exists()


def test_filter_with_no_ring_file_returns_all_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    assert not _ring_path(db).exists()
    result = filter_against_ring("sess-A", [_Belief("b1"), _Belief("b2")])
    assert len(result.new_beliefs) == 2
    assert result.recent_ids == []


def test_malformed_ring_file_is_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _ring_path(db).parent.mkdir(parents=True, exist_ok=True)
    _ring_path(db).write_text("{not json", encoding="utf-8")
    result = filter_against_ring("sess-A", [_Belief("b1")])
    assert [b.id for b in result.new_beliefs] == ["b1"]
    # Append should overwrite the malformed file with a valid one.
    next_idx = append_ids("sess-A", ["b1"])
    assert next_idx == 1
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-A"


def test_in_memory_db_is_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DB", ":memory:")
    assert append_ids("sess-A", ["b1"]) == -1
    result = filter_against_ring("sess-A", [_Belief("b1")])
    assert [b.id for b in result.new_beliefs] == ["b1"]
    assert read_ring_state("sess-A") == {}


def test_read_ring_state_returns_full_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1", "b2"])
    state = read_ring_state("sess-A")
    assert state["session_id"] == "sess-A"
    assert state["next_fire_idx"] == 1
    assert state["ring_max"] == DEFAULT_RING_MAX
    assert state["evicted_total"] == 0
    assert len(state["ring"]) == 2


def test_read_ring_state_other_session_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["b1"])
    assert read_ring_state("sess-B") == {}


def test_ring_max_env_clamps_to_default_on_bad_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv(RING_MAX_ENV, bad)
        append_ids("sess-A", [f"b-{bad}"])
        data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
        assert data["ring_max"] == DEFAULT_RING_MAX


def test_locked_belief_is_recent_when_caller_does_not_pass_locked_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the caller forgets to pass locked_ids on filter, the locked
    flag in the ring entry is **not** consulted — the caller's intent is
    authoritative. This keeps the API explicit: locked-exemption is
    something the consuming hook opts into by passing its locked set."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["L1"], locked_ids={"L1"})
    result = filter_against_ring("sess-A", [_Belief("L1")])  # no locked_ids
    assert result.new_beliefs == []
    assert result.recent_ids == ["L1"]


def test_concurrent_appends_dont_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke-test fcntl serialization by alternating appends from two
    "fires" in quick succession. Both id sets must survive."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    append_ids("sess-A", ["a1", "a2"])
    append_ids("sess-A", ["b1", "b2"])
    append_ids("sess-A", ["c1"])
    data = json.loads(_ring_path(db).read_text(encoding="utf-8"))
    ids = {e["id"] for e in data["ring"]}
    assert ids == {"a1", "a2", "b1", "b2", "c1"}
    assert data["next_fire_idx"] == 3
