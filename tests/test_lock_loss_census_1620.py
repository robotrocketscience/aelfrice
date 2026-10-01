"""#1620 AC5 — the lock-loss census re-derives each figure from a store.

Every store here is synthetic: `MemoryStore` creates the schema on a
`tmp_path` file, the handle is closed, and rows go in through raw
`sqlite3` so each test controls `created_at`, `valid_to`, origin, and
lock level exactly. The census itself never opens a store through
`MemoryStore`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from aelfrice.store import MemoryStore
from benchmarks import lock_loss_census_1620 as census


@pytest.fixture(autouse=True)
def pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


def _make_store(path: Path) -> Path:
    MemoryStore(str(path)).close()
    return path


def _add(
    conn: sqlite3.Connection,
    bid: str,
    content: str,
    *,
    created_at: str = "2026-01-01T00:00:00Z",
    origin: str = "agent_inferred",
    lock_level: str = "none",
    valid_to: str | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> None:
    conn.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, type, "
        "lock_level, created_at, origin, valid_to) "
        "VALUES (?, ?, ?, ?, ?, 'factual', ?, ?, ?, ?)",
        (bid, content, f"h-{bid}", alpha, beta, lock_level, created_at,
         origin, valid_to),
    )


def _rows(result: dict[str, object], section: str) -> list[dict[str, object]]:
    sec = cast("dict[str, object]", result[section])
    return cast("list[dict[str, object]]", sec["rows"])


def _ids(result: dict[str, object], section: str) -> list[str]:
    return [str(r["id"]) for r in _rows(result, section)]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Path]:
    path = _make_store(tmp_path / "census.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "a-locked", "a locked rule", lock_level="user")
        _add(conn, "b-locked-retired", "an old locked rule", lock_level="user",
             valid_to="2026-02-01T00:00:00Z")
        _add(conn, "c-none", "plain fact one")
        _add(conn, "d-none", "plain fact two")
        _add(conn, "e-none-retired", "plain retired fact",
             valid_to="2026-02-01T00:00:00Z")
    conn.close()
    yield path


def test_lock_level_states_active_and_all_separately(store: Path) -> None:
    result = census.measure(str(store), None)
    assert result["lock_level"] == {
        "active": {"none": 2, "user": 1},
        "all": {"none": 3, "user": 2},
    }
    assert result["beliefs_active"] == 3
    assert result["beliefs_all"] == 5


def test_aelf_commands_match_only_a_leading_command(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "cmd.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "c1", "/aelf:lock Some statement.", created_at="2026-01-01")
        _add(conn, "c2", "aelf lock another statement", created_at="2026-01-02")
        _add(conn, "c3", "uv run aelf search something", created_at="2026-01-03")
        _add(conn, "c4", "  /aelf:upgrade", created_at="2026-01-04",
             valid_to="2026-02-01")
        _add(conn, "n1", "please run /aelf:lock on this", created_at="2026-01-05")
        _add(conn, "n2", "aelfrice stores beliefs", created_at="2026-01-06")
    conn.close()
    result = census.measure(str(path), None)
    assert _ids(result, "aelf_commands") == ["c1", "c2", "c3", "c4"]
    sec = result["aelf_commands"]
    assert isinstance(sec, dict)
    assert sec["count"] == 4
    assert sec["active"] == 3
    row = _rows(result, "aelf_commands")[3]
    assert row["valid_to"] == "2026-02-01"
    assert set(row) == {"id", "origin", "lock_level", "created_at",
                        "valid_to", "content_prefix"}


def test_self_ingestion_finds_rendered_belief_blocks(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "self.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "s1", '<belief id="0123456789abcdef" lock="none">x</belief>')
        _add(conn, "n1", "a belief about beliefs, with no block")
    conn.close()
    assert _ids(census.measure(str(path), None), "self_ingestion") == ["s1"]


def test_speculative_active_excludes_retired_and_other_origins(
    tmp_path: Path,
) -> None:
    path = _make_store(tmp_path / "spec.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "p1", "an active phantom", origin="speculative")
        _add(conn, "p2", "a retired phantom", origin="speculative",
             valid_to="2026-02-01")
        _add(conn, "p3", "an inferred belief", origin="agent_inferred")
    conn.close()
    assert _ids(census.measure(str(path), None), "speculative_active") == ["p1"]


def test_content_is_truncated_to_80_characters(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "long.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "l1", "/aelf:lock " + "x" * 200)
    conn.close()
    row = _rows(census.measure(str(path), None), "aelf_commands")[0]
    assert row["content_prefix"] == ("/aelf:lock " + "x" * 200)[:80]


@pytest.fixture
def family_store(tmp_path: Path) -> Path:
    path = _make_store(tmp_path / "family.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "f1", "file every widget through WidgetPrompt",
             created_at="2026-01-01", origin="user_transcript",
             alpha=3.0, beta=1.0)
        _add(conn, "f2", "send each gadget to widgetquestion",
             created_at="2026-01-02", lock_level="none", alpha=1.0, beta=3.0)
        _add(conn, "x1", "an unrelated belief", created_at="2026-01-03")
        for session in ("s1", "s2"):
            conn.execute(
                "INSERT INTO belief_corroborations "
                "(belief_id, ingested_at, source_type, session_id) "
                "VALUES ('f2', '2026-01-04', 'transcript', ?)",
                (session,),
            )
        conn.execute(
            "INSERT INTO belief_corroborations "
            "(belief_id, ingested_at, source_type) "
            "VALUES ('x1', '2026-01-04', 'transcript')"
        )
        for bid, source, ts in (
            ("f2", "lock:unlock", "2026-01-05"),
            ("f1", "lock:expire", "2026-01-06"),
            ("f1", "user_feedback", "2026-01-07"),
            ("x1", "lock:unlock", "2026-01-08"),
        ):
            conn.execute(
                "INSERT INTO feedback_history "
                "(belief_id, valence, source, created_at) VALUES (?, 0.0, ?, ?)",
                (bid, source, ts),
            )
    conn.close()
    return path


_PATTERN = "widgetprompt|widgetquestion"


def test_family_matches_case_insensitively(family_store: Path) -> None:
    result = census.run([str(family_store)], _PATTERN)
    per = cast("dict[str, dict[str, object]]", result["stores"])
    assert _ids(per[str(family_store)], "family") == ["f1", "f2"]


def test_family_rows_carry_posterior_and_corroborations(
    family_store: Path,
) -> None:
    import re
    rows = _rows(
        census.measure(str(family_store), re.compile(_PATTERN, re.IGNORECASE)),
        "family",
    )
    by_id = {r["id"]: r for r in rows}
    assert by_id["f1"]["posterior_mean"] == 0.75
    assert by_id["f2"]["posterior_mean"] == 0.25
    assert by_id["f1"]["corroborations"] == 0
    assert by_id["f2"]["corroborations"] == 2
    assert by_id["f1"]["origin"] == "user_transcript"


def test_family_lock_history_keeps_lock_sources_of_members_only(
    family_store: Path,
) -> None:
    import re
    result = census.measure(
        str(family_store), re.compile(_PATTERN, re.IGNORECASE)
    )
    assert result["family_lock_history"] == [
        {"belief_id": "f2", "source": "lock:unlock", "valence": 0.0,
         "created_at": "2026-01-05"},
        {"belief_id": "f1", "source": "lock:expire", "valence": 0.0,
         "created_at": "2026-01-06"},
    ]


def test_no_pattern_omits_family_sections(family_store: Path) -> None:
    result = census.run([str(family_store)], None)
    assert "cross_store" not in result
    per = cast("dict[str, dict[str, object]]", result["stores"])
    assert "family" not in per[str(family_store)]


def test_cross_store_finds_a_lock_held_in_another_store(tmp_path: Path) -> None:
    here = _make_store(tmp_path / "here.db")
    there = _make_store(tmp_path / "there.db")
    conn = sqlite3.connect(here)
    with conn:
        _add(conn, "h1", "file every widget through widgetprompt",
             created_at="2026-01-02")
        _add(conn, "h2", "widgetprompt, once locked and retired",
             lock_level="user", created_at="2026-01-01", valid_to="2026-01-03")
    conn.close()
    conn = sqlite3.connect(there)
    with conn:
        _add(conn, "t1", "/aelf:lock file every widget through widgetprompt",
             lock_level="user", origin="user_stated", created_at="2026-01-03")
    conn.close()

    result = census.run([str(here), str(there)], _PATTERN)
    cross = cast("dict[str, object]", result["cross_store"])
    assert cross["stores_with_active_lock"] == [str(there)]
    assert cross["family_count"] == 3
    rows = cast("list[dict[str, object]]", cross["rows"])
    assert [(r["store"], r["id"]) for r in rows] == [
        (str(here), "h2"), (str(here), "h1"), (str(there), "t1"),
    ]


def test_missing_store_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        census.measure(str(missing), None)
    assert not missing.exists()


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def test_census_does_not_write_the_store(
    family_store: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Hold a writer open with a pending WAL frame, as a live hook would,
    # so the -wal file exists and is part of what must not change.
    holder = sqlite3.connect(family_store)
    holder.execute("PRAGMA wal_autocheckpoint = 0")
    with holder:
        _add(holder, "w1", "a row that lives only in the WAL")
    wal = Path(f"{family_store}-wal")
    try:
        before = (_digest(family_store), _digest(wal))
        assert before[1] is not None
        assert census.main(["--store", str(family_store), "--pattern", _PATTERN]) == 0
        after = (_digest(family_store), _digest(wal))
    finally:
        holder.close()
    assert before == after
    out = json.loads(capsys.readouterr().out)
    assert out["stores"][str(family_store)]["beliefs_all"] == 4


def test_output_is_deterministic(
    family_store: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    census.main(["--store", str(family_store), "--pattern", _PATTERN])
    first = capsys.readouterr().out
    census.main(["--store", str(family_store), "--pattern", _PATTERN])
    assert capsys.readouterr().out == first
