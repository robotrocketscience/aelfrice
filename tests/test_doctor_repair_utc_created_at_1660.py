"""#1660: `aelf doctor --repair-utc-created-at` rewrites offset-form
`created_at` values as UTC and re-chains the spine where order changed.

Before #1611, onboard stored commit dates with the author's local offset.
Next to `Z` rows, text order then differs from real order, and the spine
(ordered by `created_at` text) linked some pairs backwards in time.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.doctor import (
    format_utc_created_at_report,
    repair_utc_created_at,
    utc_z_form,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import (
    TEMPORAL_SPINE_EDGE_WEIGHT,
    backfill_temporal_spine,
    clear_temporal_spine,
)


def _mk(store: MemoryStore, bid: str, created_at: str, session: str) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=f"belief {bid} states one fact",
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at=created_at,
            last_retrieved_at=None,
            session_id=session,
        )
    )


def _seed(store: MemoryStore) -> None:
    # Session S, inserted in this order. A is 18:00Z, so its real order is
    # B, A, C, but its text sorts first: the spine chains A, B, C.
    _mk(store, "A", "2026-01-01T10:00:00-08:00", "S")
    _mk(store, "B", "2026-01-01T12:00:00Z", "S")
    _mk(store, "C", "2026-01-01T20:00:00Z", "S")
    # Already UTC, written two ways: never rewritten.
    _mk(store, "U1", "2026-01-02T00:00:00+00:00", "S")
    _mk(store, "U2", "2026-01-02T01:00:00.250000+00:00", "S")
    backfill_temporal_spine(store)
    # A prose-derived TEMPORAL_NEXT row: not the spine's, must survive.
    store.insert_edge(Edge(
        src="B", dst="C", type=EDGE_TEMPORAL_NEXT, weight=1.0,
        anchor_text="B comes after C",
    ))
    # One at the spine's weight: only its anchor text tells it apart.
    store.insert_edge(Edge(
        src="A", dst="C", type=EDGE_TEMPORAL_NEXT,
        weight=TEMPORAL_SPINE_EDGE_WEIGHT, anchor_text="A comes after C",
    ))
    # Session T: an offset row, but its spine was cleared and stays so.
    _mk(store, "T1", "2026-01-03T09:00:00+02:00", "T")
    _mk(store, "T2", "2026-01-03T08:00:00Z", "T")
    # A value that doesn't parse: left as it is.
    _mk(store, "X", "2026-01-04T00:00:00Z", "V")
    store._conn.execute(  # noqa: SLF001 - plant a pre-#1629 value
        "UPDATE beliefs SET created_at = 'unknown' WHERE id = 'X'"
    )
    store._conn.commit()  # noqa: SLF001


def _created_at(store: MemoryStore) -> dict[str, str]:
    rows = store._conn.execute(  # noqa: SLF001 - read-only probe
        "SELECT id, created_at FROM beliefs"
    ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def _edges(store: MemoryStore) -> list[tuple[object, ...]]:
    rows = store._conn.execute(  # noqa: SLF001 - read-only probe
        "SELECT src, dst, type, weight, anchor_text FROM edges ORDER BY 1, 2, 3"
    ).fetchall()
    return [tuple(r) for r in rows]


def _spine(store: MemoryStore) -> set[tuple[str, str]]:
    return {
        (str(e[0]), str(e[1])) for e in _edges(store)
        if e[2] == EDGE_TEMPORAL_NEXT and e[4] is None
        and e[3] == TEMPORAL_SPINE_EDGE_WEIGHT
    }


def _backwards(store: MemoryStore) -> list[tuple[str, str]]:
    """Spine edges whose successor (src) is really older than dst."""
    at = _created_at(store)
    out = []
    for src, dst in _spine(store):
        s = datetime.fromisoformat(at[src])
        d = datetime.fromisoformat(at[dst])
        if s < d:
            out.append((src, dst))
    return sorted(out)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "memory.db"))
    _seed(s)
    yield s
    s.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-01T10:00:00-08:00", "2026-01-01T18:00:00Z"),
        ("2026-01-01T01:30:00+05:30", "2025-12-31T20:00:00Z"),
        ("2026-01-01T10:00:00.5-01:00", "2026-01-01T11:00:00.500000Z"),
        ("2026-01-01T10:00:00Z", None),
        ("2026-01-01T10:00:00+00:00", None),
        ("2026-01-01T10:00:00", None),
        ("unknown", None),
    ],
)
def test_utc_z_form(value: str, expected: str | None) -> None:
    assert utc_z_form(value) == expected


def test_the_seeded_spine_has_a_backwards_edge(store: MemoryStore) -> None:
    """The precondition the repair exists for."""
    assert _spine(store) >= {("B", "A"), ("C", "B")}
    assert _backwards(store) == [("B", "A")]


def test_dry_run_reports_and_changes_nothing(store: MemoryStore) -> None:
    at, edges, gen = _created_at(store), _edges(store), store.store_generation()
    report = repair_utc_created_at(store, dry_run=True)
    assert (report.rows_found, report.sessions_affected) == (2, 2)
    assert (report.spine_edges_removed, report.spine_edges_written) == (2, 2)
    assert report.rewritten == 0
    assert _created_at(store) == at
    assert _edges(store) == edges
    assert store.store_generation() == gen


def test_apply_rewrites_offsets_and_rechains(store: MemoryStore) -> None:
    before = _created_at(store)
    report = repair_utc_created_at(store, dry_run=False)
    assert report.rewritten == 2
    after = _created_at(store)
    assert after["A"] == "2026-01-01T18:00:00Z"
    assert after["T1"] == "2026-01-03T07:00:00Z"
    unchanged = {k for k in before if k not in ("A", "T1")}
    assert {k: after[k] for k in unchanged} == {k: before[k] for k in unchanged}
    # Session S now chains B, A, C: nothing points backwards.
    assert ("A", "B") in _spine(store) and ("C", "A") in _spine(store)
    assert _backwards(store) == []
    # The prose row survives; session T's cleared spine stays cleared.
    assert ("B", "C", EDGE_TEMPORAL_NEXT, 1.0, "B comes after C") in _edges(store)
    assert (
        "A", "C", EDGE_TEMPORAL_NEXT, TEMPORAL_SPINE_EDGE_WEIGHT,
        "A comes after C",
    ) in _edges(store)
    assert not {e for e in _spine(store) if e[0].startswith("T")}


def test_a_second_apply_is_a_no_op(store: MemoryStore) -> None:
    repair_utc_created_at(store, dry_run=False)
    at, edges = _created_at(store), _edges(store)
    again = repair_utc_created_at(store, dry_run=False)
    assert (again.rows_found, again.rewritten) == (0, 0)
    assert (_created_at(store), _edges(store)) == (at, edges)


def test_a_store_without_offsets_is_untouched(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "clean.db"))
    try:
        _mk(s, "Z1", "2026-01-01T00:00:00Z", "S")
        _mk(s, "Z2", "2026-01-01T01:00:00+00:00", "S")
        backfill_temporal_spine(s)
        at, edges, gen = _created_at(s), _edges(s), s.store_generation()
        report = repair_utc_created_at(s, dry_run=False)
        assert report.rows_found == 0
        assert (_created_at(s), _edges(s), s.store_generation()) == (at, edges, gen)
    finally:
        s.close()


def test_a_cleared_spine_is_not_rebuilt(store: MemoryStore) -> None:
    clear_temporal_spine(store)
    repair_utc_created_at(store, dry_run=False)
    assert _spine(store) == set()


def test_refuses_inside_an_open_transaction(store: MemoryStore) -> None:
    at = _created_at(store)
    with store.transaction():
        with pytest.raises(RuntimeError, match="own transaction"):
            repair_utc_created_at(store, dry_run=True)
    assert _created_at(store) == at


def test_format_shows_samples(store: MemoryStore) -> None:
    text = format_utc_created_at_report(repair_utc_created_at(store, dry_run=True))
    assert "non-UTC offset: 2 in 2 session(s)" in text
    assert "2 removed, 2 written" in text
    assert "A: 2026-01-01T10:00:00-08:00 -> 2026-01-01T18:00:00Z" in text
    assert "--apply" in text


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_cli_dry_run_then_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "cli.db"
    s = MemoryStore(str(db))
    try:
        _seed(s)
    finally:
        s.close()
    code, out = _run_cli(monkeypatch, db, "doctor", "--repair-utc-created-at")
    assert code == 0 and "dry-run" in out
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--repair-utc-created-at", "--apply"
    )
    assert code == 0 and "rewritten: 2" in out
    s = MemoryStore(str(db))
    try:
        assert _backwards(s) == []
    finally:
        s.close()


def test_cli_refuses_two_passes_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "cli.db"
    MemoryStore(str(db)).close()
    code, _ = _run_cli(
        monkeypatch, db, "doctor", "--repair-utc-created-at",
        "--gc-filesystem-corroboration",
    )
    assert code == 2
    assert "one at a time" in capsys.readouterr().err
