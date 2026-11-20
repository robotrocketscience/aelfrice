"""The correction proxy counts a belief once per turn (#1954).

Since #1954 a turn's `injection_events.turn_id` is the host's prompt id, so a
duplicated hook registration writes two batches under one
`(session_id, turn_id)`. `load_injections` groups by that pair and must not
list a belief twice.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from benchmarks.correction_attribution_proxy import load_injections


def _db(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT)")
        con.execute(
            "CREATE TABLE injection_events "
            "(session_id TEXT, turn_id TEXT, injected_at TEXT, belief_id TEXT)"
        )
        con.executemany(
            "INSERT INTO beliefs VALUES (?, ?)",
            [("B1", "the cellar holds barrels"), ("B2", "the door sticks")],
        )
        con.executemany("INSERT INTO injection_events VALUES (?, ?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()
    return path


def test_two_batches_under_one_turn_list_each_belief_once(tmp_path: Path) -> None:
    db = _db(tmp_path / "m.db", [
        ("s1", "P1", "2026-11-19T10:00:00+00:00", "B1"),
        ("s1", "P1", "2026-11-19T10:00:00+00:00", "B2"),
        ("s1", "P1", "2026-11-19T10:00:00.400000+00:00", "B1"),
        ("s1", "P1", "2026-11-19T10:00:00.400000+00:00", "B2"),
    ])
    by_session, _ = load_injections(db)
    (turn,) = by_session["s1"]
    assert turn[2] == ["B1", "B2"]


def test_distinct_turns_stay_apart(tmp_path: Path) -> None:
    db = _db(tmp_path / "m.db", [
        ("s1", "P1", "2026-11-19T10:00:00+00:00", "B1"),
        ("s1", "P2", "2026-11-19T10:01:00+00:00", "B1"),
    ])
    by_session, _ = load_injections(db)
    assert [t[2] for t in by_session["s1"]] == [["B1"], ["B1"]]
