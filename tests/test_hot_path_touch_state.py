"""Tests for #816 hot-path touch state (DESIGN.md v1 test plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.hot_path import (
    DEFAULT_TOUCH_WINDOW_K,
    TOUCH_EVENT_KIND_INJECTION,
    is_hot,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _new_belief(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content-{bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-14T00:00:00Z",
        last_retrieved_at=None,
    )


# --------------------------- pure helpers -----------------------------


def test_default_window_k_is_50() -> None:
    """R2c canonical-cell window — DESIGN.md v1 §"Decay shape"."""
    assert DEFAULT_TOUCH_WINDOW_K == 50


def test_injection_bit_is_bit_0() -> None:
    assert TOUCH_EVENT_KIND_INJECTION == 1


@pytest.mark.parametrize(
    ("last", "current", "k", "expected"),
    [
        (10, 20, 15, True),    # inside window
        (5, 20, 15, True),     # right at the boundary (5 == 20 - 15)
        (4, 20, 15, False),    # one outside
        (0, 0, 50, True),      # session start, last touched at fire 0
        (-1, 100, 50, False),  # never-touched sentinel falls off late window
    ],
)
def test_is_hot_boundary(last: int, current: int, k: int, expected: bool) -> None:
    assert is_hot(last_fire_idx=last, current_fire_idx=current, window_k=k) is expected


def test_is_hot_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        is_hot(last_fire_idx=10, current_fire_idx=20, window_k=0)
    with pytest.raises(ValueError):
        is_hot(last_fire_idx=10, current_fire_idx=20, window_k=-5)


def test_is_hot_rejects_negative_current() -> None:
    with pytest.raises(ValueError):
        is_hot(last_fire_idx=0, current_fire_idx=-1, window_k=10)


# --------------------------- store APIs -------------------------------


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    # FK enforcement off: these unit tests exercise belief_touches in
    # isolation without populating `beliefs`. The FK is still verified
    # below in test_fk_cascade_on_belief_delete with FK enforcement on.
    s._conn.execute("PRAGMA foreign_keys = OFF")
    return s


def test_record_touch_inserts_new_row(store: MemoryStore) -> None:
    store.record_touch(
        belief_id="b1", session_id="s1", fire_idx=10,
        event_kind=TOUCH_EVENT_KIND_INJECTION,
    )
    cur = store._conn.execute(
        "SELECT * FROM belief_touches WHERE belief_id=? AND session_id=?",
        ("b1", "s1"),
    )
    row = cur.fetchone()
    assert row is not None
    assert row["last_fire_idx"] == 10
    assert row["touch_count"] == 1
    assert row["event_kinds_bitmask"] == TOUCH_EVENT_KIND_INJECTION


def test_record_touch_upsert_refreshes_fire_idx_and_bumps_count(
    store: MemoryStore,
) -> None:
    """ON CONFLICT DO UPDATE — re-touch updates last_fire_idx and ++ count."""
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=1)
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=25, event_kind=1)
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=30, event_kind=1)
    cur = store._conn.execute(
        "SELECT * FROM belief_touches WHERE belief_id=? AND session_id=?",
        ("b1", "s1"),
    )
    row = cur.fetchone()
    assert row["last_fire_idx"] == 30
    assert row["touch_count"] == 3


def test_record_touch_or_s_event_kind_bits(store: MemoryStore) -> None:
    """Subsequent record_touch with a different bit OR-s into the mask."""
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=1)
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=11, event_kind=2)
    cur = store._conn.execute(
        "SELECT event_kinds_bitmask FROM belief_touches "
        "WHERE belief_id=? AND session_id=?",
        ("b1", "s1"),
    )
    assert cur.fetchone()["event_kinds_bitmask"] == 0b11


def test_record_touch_rejects_invalid_inputs(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.record_touch(belief_id="", session_id="s1", fire_idx=10, event_kind=1)
    with pytest.raises(ValueError):
        store.record_touch(belief_id="b1", session_id="", fire_idx=10, event_kind=1)
    with pytest.raises(ValueError):
        store.record_touch(belief_id="b1", session_id="s1", fire_idx=-1, event_kind=1)
    with pytest.raises(ValueError):
        store.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=0)


def test_read_touch_set_in_window_respects_boundary(store: MemoryStore) -> None:
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=20, event_kind=1)
    store.record_touch(belief_id="b2", session_id="s1", fire_idx=5, event_kind=1)
    store.record_touch(belief_id="b3", session_id="s1", fire_idx=15, event_kind=1)
    # current_fire_idx=20, window_k=5 → threshold=15 → b1 and b3
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=20, window_k=5,
    ) == {"b1", "b3"}
    # widen → all three
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=20, window_k=20,
    ) == {"b1", "b2", "b3"}


def test_read_touch_set_per_session_isolation(store: MemoryStore) -> None:
    """Federation #661: per-session reset — touch state doesn't cross."""
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=1)
    store.record_touch(belief_id="b1", session_id="s2", fire_idx=99, event_kind=1)
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=10, window_k=5,
    ) == {"b1"}
    assert store.read_touch_set_in_window(
        "s2", current_fire_idx=99, window_k=5,
    ) == {"b1"}
    # No row in s2 spillage into s1
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=99, window_k=1,
    ) == set()


def test_read_touch_set_empty_inputs(store: MemoryStore) -> None:
    assert store.read_touch_set_in_window(
        "", current_fire_idx=10, window_k=5,
    ) == set()
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=10, window_k=0,
    ) == set()
    assert store.read_touch_set_in_window(
        "s1", current_fire_idx=10, window_k=-1,
    ) == set()


def test_count_and_list_sessions(store: MemoryStore) -> None:
    store.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=1)
    store.record_touch(belief_id="b2", session_id="s1", fire_idx=20, event_kind=1)
    store.record_touch(belief_id="b3", session_id="s2", fire_idx=5, event_kind=1)
    assert store.count_touches_for_session("s1") == 2
    assert store.count_touches_for_session("s2") == 1
    assert store.count_touches_for_session("missing") == 0
    # ordered by max_fire DESC, then session_id ASC
    assert store.list_touch_sessions() == [
        ("s1", 2, 20),
        ("s2", 1, 5),
    ]


def test_determinism_property(store: MemoryStore) -> None:
    """Same writes in same order → byte-identical row state."""
    writes = [
        ("b1", "s1", 5),
        ("b2", "s1", 7),
        ("b1", "s1", 12),
        ("b3", "s1", 20),
    ]
    for bid, sid, f in writes:
        store.record_touch(belief_id=bid, session_id=sid, fire_idx=f, event_kind=1)
    snap_a = store._conn.execute(
        "SELECT belief_id, session_id, last_fire_idx, touch_count, "
        "event_kinds_bitmask FROM belief_touches ORDER BY belief_id"
    ).fetchall()
    # Replay against a fresh store; rows must match.
    s2 = MemoryStore(":memory:")
    s2._conn.execute("PRAGMA foreign_keys = OFF")
    for bid, sid, f in writes:
        s2.record_touch(belief_id=bid, session_id=sid, fire_idx=f, event_kind=1)
    snap_b = s2._conn.execute(
        "SELECT belief_id, session_id, last_fire_idx, touch_count, "
        "event_kinds_bitmask FROM belief_touches ORDER BY belief_id"
    ).fetchall()
    assert [tuple(r) for r in snap_a] == [tuple(r) for r in snap_b]


def test_fk_cascade_on_belief_delete(tmp_path: Path) -> None:
    """FK CASCADE: deleting a belief drops its belief_touches rows."""
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_new_belief("b1"))
    s.record_touch(belief_id="b1", session_id="s1", fire_idx=10, event_kind=1)
    assert s.count_touches_for_session("s1") == 1
    s._conn.execute("DELETE FROM beliefs WHERE id=?", ("b1",))
    s._conn.commit()
    assert s.count_touches_for_session("s1") == 0
    s.close()


# --------------------------- hook integration -------------------------


def _read_touch_rows(db: Path, session_id: str) -> dict[str, tuple[int, int]]:
    s = MemoryStore(str(db))
    try:
        return {
            str(r["belief_id"]): (int(r["last_fire_idx"]), int(r["touch_count"]))
            for r in s._conn.execute(
                "SELECT belief_id, last_fire_idx, touch_count "
                "FROM belief_touches WHERE session_id=?",
                (session_id,),
            ).fetchall()
        }
    finally:
        s.close()


def test_record_touches_writes_current_turn_only_no_ring_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward-only contract — pre-substrate JSON-ring entries are NOT
    backfilled into ``belief_touches``. Only the current turn's
    ``belief_ids`` land.

    Regression guard for the dropped one-shot migration: a prior
    implementation re-read the ring on every UPS fire and replayed
    every entry through ``record_touch``, which uses ON CONFLICT DO
    UPDATE (touch_count = touch_count + 1) — so the replay was
    non-idempotent. The replay is gone; this test pins the new
    contract.
    """
    from aelfrice import hook, session_ring

    db = tmp_path / "memory.db"
    monkeypatch.setattr(hook, "db_path", lambda: db)
    monkeypatch.setattr(session_ring, "db_path", lambda: db)

    s = MemoryStore(str(db))
    for bid in ("old1", "old2", "new1", "new2"):
        s.insert_belief(_new_belief(bid))
    s.close()

    # Pre-populate the JSON ring with two beliefs at older fire_idx.
    ring_path = db.parent / session_ring.SESSION_RING_FILENAME
    ring_path.parent.mkdir(parents=True, exist_ok=True)
    ring_path.write_text(json.dumps({
        "session_id": "S",
        "ring": [
            {"id": "old1", "fire_idx": 2, "locked": False},
            {"id": "old2", "fire_idx": 3, "locked": True},
        ],
        "ring_max": 200,
        "next_fire_idx": 4,
        "evicted_total": 0,
    }))

    hook._record_touches(
        session_id="S",
        belief_ids=["new1", "new2"],
        fire_idx=4,
    )

    rows = _read_touch_rows(db, "S")
    # old1/old2 were in the ring but are NOT backfilled — forward-only.
    assert "old1" not in rows, rows
    assert "old2" not in rows, rows
    # Current touches landed at fire_idx=4 with touch_count=1.
    assert rows["new1"] == (4, 1), rows
    assert rows["new2"] == (4, 1), rows


def test_record_touches_count_matches_actual_inject_count_across_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``touch_count`` increments once per UPS fire that actually injects
    the belief — not once per ring entry.

    Regression for the pre-fix bug (reviews on PR #821, 2026-05-14):
    every UPS fire walked the entire JSON ring and called
    ``record_touch`` for every entry, so a belief sitting in the ring
    got ``touch_count`` bumped on every fire regardless of whether it
    was in the current injection set; current-turn beliefs got bumped
    twice (once via ring replay, once via the explicit current loop).

    Mimics the production caller in
    ``hook.py:_record_touches`` call site: ``append_ids`` first, then
    ``_record_touches`` with ``fire_idx=_next_fire - 1``.
    """
    from aelfrice import hook, session_ring
    from aelfrice.session_ring import append_ids

    db = tmp_path / "memory.db"
    monkeypatch.setattr(hook, "db_path", lambda: db)
    monkeypatch.setattr(session_ring, "db_path", lambda: db)

    s = MemoryStore(str(db))
    for bid in ("b1", "b2"):
        s.insert_belief(_new_belief(bid))
    s.close()

    # Three UPS fires, modeling the production sequence:
    #   append_ids(...)  → returns next_fire_idx
    #   _record_touches(fire_idx=next_fire - 1)
    # Injection pattern: [b1], [b2], [b1].
    for ids in (["b1"], ["b2"], ["b1"]):
        next_fire = append_ids("S", ids, locked_ids=set())
        assert next_fire >= 1
        hook._record_touches(
            session_id="S",
            belief_ids=ids,
            fire_idx=next_fire - 1,
        )

    rows = _read_touch_rows(db, "S")
    # b1 was actually injected twice → touch_count=2; last_fire_idx is
    # the slot of the third (final) fire.
    assert rows["b1"][1] == 2, rows
    # b2 was injected once → touch_count=1.
    assert rows["b2"][1] == 1, rows
    # last_fire_idx monotonicity: b1's last touch is at the latest fire,
    # which is strictly greater than b2's.
    assert rows["b1"][0] > rows["b2"][0], rows


def test_record_touches_fail_soft_on_missing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """No DB on disk → no exception; one-line stderr warning."""
    from aelfrice import hook

    # Path that doesn't exist and won't be created.
    bogus = tmp_path / "nonexistent" / "memory.db"
    monkeypatch.setattr(hook, "db_path", lambda: bogus)
    # No raise.
    hook._record_touches(session_id="S", belief_ids=["b1"], fire_idx=1)
