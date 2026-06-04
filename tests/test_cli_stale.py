"""Tests for `aelf stale` CLI subcommand (#933).

Threshold-based listing over existing created_at + last_retrieved_at
columns. No decay model — the test fixture pins absolute timestamps
and the test asserts which rows the threshold windows match.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _iso(days_ago: int) -> str:
    """Return an ISO-Z timestamp `days_ago` days before now (UTC)."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _seed(
    db: Path,
    *,
    bid: str,
    content: str,
    age_days: int,
    cold_days: int | None,
    locked: bool = False,
) -> None:
    """Insert a belief with explicit created_at + last_retrieved_at."""
    store = MemoryStore(str(db))
    try:
        store.insert_belief(Belief(
            id=bid,
            content=content,
            content_hash=f"hash-{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER if locked else LOCK_NONE,
            locked_at=_iso(age_days) if locked else None,
            created_at=_iso(age_days),
            last_retrieved_at=_iso(cold_days) if cold_days is not None else None,
            origin=ORIGIN_USER_STATED if locked else ORIGIN_AGENT_INFERRED,
        ))
    finally:
        store.close()


def test_empty_store_prints_no_stale(isolated_db: Path) -> None:
    code, out = _run("stale")
    assert code == 0
    assert "no stale beliefs" in out


def test_default_thresholds_surface_old_and_cold(isolated_db: Path) -> None:
    # OLD + COLD → should surface
    _seed(isolated_db, bid="aaaa00000001", content="stale belief A",
          age_days=100, cold_days=50)
    # NEW + WARM → should not surface
    _seed(isolated_db, bid="bbbb00000002", content="fresh recent belief",
          age_days=2, cold_days=1)
    code, out = _run("stale")
    assert code == 0
    assert "aaaa000000" in out
    assert "bbbb000000" not in out


def test_warm_belief_excluded_even_if_old(isolated_db: Path) -> None:
    _seed(isolated_db, bid="cccc00000003", content="old but warm",
          age_days=100, cold_days=1)
    code, out = _run("stale")
    assert "cccc000000" not in out
    assert "no stale beliefs" in out


def test_never_retrieved_old_belief_surfaces(isolated_db: Path) -> None:
    _seed(isolated_db, bid="dddd00000004", content="never retrieved",
          age_days=100, cold_days=None)
    code, out = _run("stale")
    assert "dddd000000" in out
    # Sort: never-retrieved beliefs have an ∞ cold marker in human mode.
    assert "∞" in out


def test_locked_only_filters_speculative(isolated_db: Path) -> None:
    _seed(isolated_db, bid="eeee00000005", content="speculative stale",
          age_days=100, cold_days=50, locked=False)
    _seed(isolated_db, bid="ffff00000006", content="locked stale",
          age_days=100, cold_days=50, locked=True)
    code, out = _run("stale", "--locked-only")
    assert "ffff000000" in out
    assert "eeee000000" not in out


def test_json_output_shape(isolated_db: Path) -> None:
    _seed(isolated_db, bid="aaaa11111111", content="json stale",
          age_days=100, cold_days=50, locked=True)
    code, out = _run("stale", "--json")
    assert code == 0
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "aaaa11111111"
    assert r["locked"] is True
    assert r["age_days"] >= 99
    assert r["cold_days"] >= 49
    assert "json stale" in r["snippet"]


def test_json_null_cold_when_never_retrieved(isolated_db: Path) -> None:
    _seed(isolated_db, bid="aaaa22222222", content="never seen",
          age_days=100, cold_days=None)
    code, out = _run("stale", "--json")
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert rows[0]["cold_days"] is None


def test_limit_caps_rows(isolated_db: Path) -> None:
    for i in range(5):
        _seed(isolated_db, bid=f"bbbb333333{i:02d}", content=f"row {i}",
              age_days=100 + i, cold_days=50 + i)
    code, out = _run("stale", "--limit", "2", "--json")
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 2


def test_older_than_threshold_excludes_too_recent(isolated_db: Path) -> None:
    # Default --older-than=30; with --older-than=200, this belief is
    # no longer "old enough."
    _seed(isolated_db, bid="aaaa44444444", content="100 days old",
          age_days=100, cold_days=50)
    code, out = _run("stale", "--older-than", "200")
    assert "aaaa444444" not in out
    assert "no stale beliefs" in out


def test_cold_for_threshold_excludes_too_recently_retrieved(
    isolated_db: Path,
) -> None:
    # Default --cold-for=14. With --cold-for=100, a 50-days-cold
    # belief is now considered warm.
    _seed(isolated_db, bid="aaaa55555555", content="50 days cold",
          age_days=100, cold_days=50)
    code, out = _run("stale", "--cold-for", "100")
    assert "aaaa555555" not in out


def test_sort_puts_never_retrieved_first(isolated_db: Path) -> None:
    _seed(isolated_db, bid="bbbb66666666", content="cold 50d",
          age_days=100, cold_days=50)
    _seed(isolated_db, bid="bbbb77777777", content="never retrieved",
          age_days=100, cold_days=None)
    code, out = _run("stale", "--json")
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert rows[0]["id"] == "bbbb77777777"
    assert rows[0]["cold_days"] is None
