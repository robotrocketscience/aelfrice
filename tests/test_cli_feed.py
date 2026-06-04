"""Tests for `aelf feed` CLI subcommand (#931)."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice import feed_log
from aelfrice.cli import main


@pytest.fixture
def isolated_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "feed.jsonl"
    monkeypatch.setattr(
        feed_log, "feed_path",
        lambda db_dir=None: p,
    )
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _write_rows(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _iso(seconds_ago: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).isoformat().replace("+00:00", "Z")


def test_feed_empty_when_no_log(isolated_feed: Path) -> None:
    code, out = _run("feed")
    assert code == 0
    assert "no feed entries" in out


def test_feed_renders_human_rows(isolated_feed: Path) -> None:
    _write_rows(isolated_feed, [
        {"ts": _iso(60), "event": "belief.locked", "id": "abc123",
         "snippet": "atomic commits beat batched"},
        {"ts": _iso(30), "event": "wonder.promoted", "id": "def456",
         "from_": "speculative", "to": "locked"},
    ])
    code, out = _run("feed")
    assert code == 0
    assert "belief.locked" in out
    assert "abc123" in out
    assert "wonder.promoted" in out
    assert "from_=speculative" in out


def test_feed_json_passthrough(isolated_feed: Path) -> None:
    _write_rows(isolated_feed, [
        {"ts": _iso(60), "event": "belief.ingested", "id": "a"},
        {"ts": _iso(30), "event": "feedback.applied", "belief_id": "b"},
    ])
    code, out = _run("feed", "--json")
    parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(parsed) == 2
    assert parsed[0]["event"] == "belief.ingested"
    assert parsed[1]["belief_id"] == "b"


def test_feed_since_filters_to_window(isolated_feed: Path) -> None:
    _write_rows(isolated_feed, [
        {"ts": _iso(3600), "event": "belief.locked", "id": "old"},
        {"ts": _iso(60), "event": "belief.locked", "id": "recent"},
    ])
    code, out = _run("feed", "--since", "5m", "--json")
    parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(parsed) == 1
    assert parsed[0]["id"] == "recent"


def test_feed_since_rejects_bad_spec(isolated_feed: Path) -> None:
    _write_rows(isolated_feed, [
        {"ts": _iso(60), "event": "belief.locked", "id": "x"},
    ])
    code, out = _run("feed", "--since", "garbage")
    assert code == 2
    assert "--since" in out


def test_feed_limit_caps_to_last_n(isolated_feed: Path) -> None:
    _write_rows(isolated_feed, [
        {"ts": _iso(300), "event": "belief.locked", "id": "a"},
        {"ts": _iso(200), "event": "belief.locked", "id": "b"},
        {"ts": _iso(100), "event": "belief.locked", "id": "c"},
        {"ts": _iso(50), "event": "belief.locked", "id": "d"},
    ])
    code, out = _run("feed", "--limit", "2", "--json")
    parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [r["id"] for r in parsed] == ["c", "d"]


def test_feed_skips_malformed_lines(isolated_feed: Path) -> None:
    """Partial / corrupt lines from interrupted writes must not crash."""
    isolated_feed.parent.mkdir(parents=True, exist_ok=True)
    isolated_feed.write_text(
        json.dumps({"ts": _iso(60), "event": "belief.locked", "id": "ok"})
        + "\n"
        + "{this is not json\n"
        + json.dumps({"ts": _iso(30), "event": "belief.ingested", "id": "ok2"})
        + "\n"
    )
    code, out = _run("feed", "--json")
    parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [r["id"] for r in parsed] == ["ok", "ok2"]
