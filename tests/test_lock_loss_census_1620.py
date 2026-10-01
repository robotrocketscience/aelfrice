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


_NOW = "2026-06-01T00:00:00+00:00"


def test_output_is_deterministic(
    family_store: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["--store", str(family_store), "--pattern", _PATTERN, "--now", _NOW]
    census.main(argv)
    first = capsys.readouterr().out
    census.main(argv)
    assert capsys.readouterr().out == first
    assert json.loads(first)["now"] == _NOW


def test_rows_are_ordered_whatever_the_insert_order(tmp_path: Path) -> None:
    """Rows inserted newest first still come out oldest first, in each section."""
    here = _make_store(tmp_path / "order-a.db")
    there = _make_store(tmp_path / "order-b.db")
    for path, ids in ((here, ("x3", "x1")), (there, ("y2",))):
        conn = sqlite3.connect(path)
        with conn:
            for bid in ids:
                _add(conn, bid, f"/aelf:lock widgetprompt {bid}",
                     created_at=f"2026-01-0{bid[1]}T00:00:00Z")
        conn.close()
    out = census.run([str(here), str(there)], "widgetprompt", now=_NOW)
    stores = cast("dict[str, dict[str, object]]", out["stores"])
    assert _ids(stores[str(here)], "aelf_commands") == ["x1", "x3"]
    assert _ids(stores[str(here)], "family") == ["x1", "x3"]
    cross = cast("dict[str, object]", out["cross_store"])
    rows = cast("list[dict[str, object]]", cross["rows"])
    assert [r["id"] for r in rows] == ["x1", "y2", "x3"]


def test_an_expired_unswept_lock_is_not_counted_as_held(tmp_path: Path) -> None:
    """The product sweeps a due lock to `none` on open; the census never opens that way."""
    path = _make_store(tmp_path / "expiry.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "due", "widgetprompt lapsed", lock_level="user")
        _add(conn, "later", "widgetprompt current", lock_level="user")
        _add(conn, "open", "widgetprompt forever", lock_level="user")
        conn.execute("UPDATE beliefs SET lock_expires_at = ? WHERE id = 'due'",
                     ("2026-05-31T23:59:59+00:00",))
        conn.execute("UPDATE beliefs SET lock_expires_at = ? WHERE id = 'later'",
                     ("2026-06-01T00:00:01+00:00",))
    conn.close()
    out = census.run([str(path)], "widgetprompt", now=_NOW)
    result = cast("dict[str, dict[str, object]]", out["stores"])[str(path)]
    assert result["lock_level"] == {
        "active": {"user": 2, census.EXPIRED_UNSWEPT: 1},
        "all": {"user": 2, census.EXPIRED_UNSWEPT: 1},
    }
    levels = {r["id"]: r["lock_level"] for r in _rows(result, "family")}
    assert levels == {"due": census.EXPIRED_UNSWEPT, "later": "user", "open": "user"}


def test_an_expired_lock_does_not_make_a_store_hold_the_lock(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "only-expired.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "due", "widgetprompt lapsed", lock_level="user")
        conn.execute("UPDATE beliefs SET lock_expires_at = ? WHERE id = 'due'",
                     ("2026-01-01T00:00:00+00:00",))
    conn.close()
    out = census.run([str(path)], "widgetprompt", now=_NOW)
    cross = cast("dict[str, object]", out["cross_store"])
    assert cross["stores_with_active_lock"] == []


def test_the_connection_refuses_writes(store: Path) -> None:
    conn = census.open_read_only(str(store))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE census_probe (x INTEGER)")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("PRAGMA user_version = 7")
    finally:
        conn.close()


@pytest.mark.parametrize("text", [
    "aelf is the CLI entry point",
    "aelf, the tool, stores beliefs",
    "uv run aelfrice-bench --x",
    "aelf notacommand here",
])
def test_prose_about_aelf_is_not_a_command(text: str) -> None:
    assert not census.is_aelf_command(text)


@pytest.mark.parametrize("text", [
    "aelf lock a statement",
    "uv run aelf search something",
    "/aelf:lock a statement",
])
def test_a_registered_subcommand_is_a_command(text: str) -> None:
    assert census.is_aelf_command(text)


def _expiry_store(tmp_path: Path, expiries: dict[str, str]) -> Path:
    """One user lock per id, each with the given `lock_expires_at` text."""
    path = _make_store(tmp_path / "expiry-text.db")
    conn = sqlite3.connect(path)
    with conn:
        for bid, expires in expiries.items():
            _add(conn, bid, f"widgetprompt {bid}", lock_level="user")
            conn.execute(
                "UPDATE beliefs SET lock_expires_at = ? WHERE id = ?",
                (expires, bid),
            )
    conn.close()
    return path


def _levels(path: Path, now: str) -> dict[object, object]:
    """Lock level per family row, required to agree between `run` and `measure`.

    `run` hands `measure` a normalised `now`; calling `measure` directly
    hands it the caller's text, so both entry points are checked.
    """
    import re
    out = census.run([str(path)], "widgetprompt", now=now)
    via_run = cast("dict[str, dict[str, object]]", out["stores"])[str(path)]
    via_measure = census.measure(
        str(path), re.compile("widgetprompt", re.IGNORECASE), now=now,
    )
    levels = [
        {r["id"]: r["lock_level"] for r in _rows(result, "family")}
        for result in (via_run, via_measure)
    ]
    assert levels[0] == levels[1]
    return levels[0]


def test_now_in_another_offset_is_compared_as_an_instant(tmp_path: Path) -> None:
    """01:00+02:00 is 23:00 UTC the day before, so a midnight-UTC expiry is not due."""
    path = _expiry_store(tmp_path, {"k1": "2026-06-01T00:00:00+00:00"})
    assert _levels(path, "2026-06-01T01:00:00+02:00") == {"k1": "user"}


def test_fractional_expiry_after_a_z_now_is_not_due(tmp_path: Path) -> None:
    path = _expiry_store(tmp_path, {"k1": "2026-06-01T00:00:00.500000+00:00"})
    assert _levels(path, "2026-06-01T00:00:00Z") == {"k1": "user"}


def test_an_expiry_in_another_offset_is_compared_as_an_instant(
    tmp_path: Path,
) -> None:
    """01:00+02:00 is before midnight UTC, so the lock is due at midnight UTC."""
    path = _expiry_store(tmp_path, {"k1": "2026-06-01T01:00:00+02:00"})
    assert _levels(path, "2026-06-01T00:00:00+00:00") == {
        "k1": census.EXPIRED_UNSWEPT,
    }


def test_an_expiry_equal_to_now_is_due(tmp_path: Path) -> None:
    """The sweep's boundary: a lock due at exactly `now` is expired."""
    path = _expiry_store(tmp_path, {"k1": "2026-06-01T00:00:00+00:00"})
    assert _levels(path, "2026-06-01T00:00:00Z") == {"k1": census.EXPIRED_UNSWEPT}


def test_now_is_reported_in_utc(store: Path) -> None:
    out = census.run([str(store)], None, now="2026-06-01T01:00:00+02:00")
    assert out["now"] == "2026-05-31T23:00:00+00:00"


@pytest.mark.parametrize("now", ["garbage", "2026-06-01T00:00:00", "2026-06-01"])
def test_run_and_measure_reject_a_now_that_is_not_an_instant(
    store: Path, now: str,
) -> None:
    with pytest.raises(ValueError):
        census.run([str(store)], None, now=now)
    with pytest.raises(ValueError):
        census.measure(str(store), None, now=now)


@pytest.mark.parametrize("now", ["garbage", "2026-06-01T00:00:00"])
def test_main_rejects_a_now_that_is_not_an_instant(
    store: Path, now: str, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        census.main(["--store", str(store), "--now", now])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_an_unparseable_expiry_is_counted_apart_not_expired(tmp_path: Path) -> None:
    path = _expiry_store(tmp_path, {
        "bad": "garbage",
        "naive": "2026-01-01T00:00:00",
        "due": "2026-01-01T00:00:00+00:00",
    })
    assert _levels(path, _NOW) == {
        "bad": "user", "naive": "user", "due": census.EXPIRED_UNSWEPT,
    }
    result = census.measure(str(path), None, now=_NOW)
    assert result["lock_expiry_unparseable"] == {"count": 2, "ids": ["bad", "naive"]}


@pytest.mark.parametrize("text", [
    "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00",
])
def test_an_out_of_range_instant_is_unparseable_not_a_crash(text: str) -> None:
    assert census.parse_instant(text) is None


def test_an_out_of_range_expiry_is_bucketed(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "range.db")
    conn = sqlite3.connect(path)
    with conn:
        _add(conn, "far", "widgetprompt far", lock_level="user")
        conn.execute("UPDATE beliefs SET lock_expires_at = ? WHERE id = 'far'",
                     ("9999-12-31T23:59:59-01:00",))
    conn.close()
    result = census.measure(str(path), None, now=_NOW)
    bucket = cast("dict[str, object]", result["lock_expiry_unparseable"])
    assert bucket["ids"] == ["far"]


def test_main_rejects_an_out_of_range_now(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "range-now.db")
    with pytest.raises(SystemExit) as exc:
        census.main(["--store", str(path), "--now", "0001-01-01T00:00:00+01:00"])
    assert exc.value.code == 2
