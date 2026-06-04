"""Tests for the feed_log writer module (#931)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import feed_log


@pytest.fixture
def isolated_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect feed_path() to a writable temp dir."""
    target = tmp_path / "feed.jsonl"
    monkeypatch.setattr(
        feed_log, "feed_path",
        lambda db_dir=None: target,
    )
    return target


def test_is_enabled_default_true() -> None:
    assert feed_log.is_enabled(env={}) is True


def test_is_enabled_false_when_env_zero() -> None:
    assert feed_log.is_enabled(env={"AELFRICE_FEED_LOG": "0"}) is False


def test_is_enabled_true_when_env_one() -> None:
    assert feed_log.is_enabled(env={"AELFRICE_FEED_LOG": "1"}) is True


def test_append_writes_jsonl_row(
    isolated_feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    feed_log.append("belief.locked", id="abc123", snippet="hello world")
    assert isolated_feed.exists()
    rows = isolated_feed.read_text().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["event"] == "belief.locked"
    assert parsed["id"] == "abc123"
    assert parsed["snippet"] == "hello world"
    assert parsed["ts"].endswith("Z")


def test_append_appends_multiple_rows(
    isolated_feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    feed_log.append("belief.locked", id="a")
    feed_log.append("belief.ingested", id="b")
    feed_log.append("wonder.promoted", id="c", from_="speculative", to="locked")
    rows = isolated_feed.read_text().splitlines()
    assert len(rows) == 3
    events = [json.loads(r)["event"] for r in rows]
    assert events == ["belief.locked", "belief.ingested", "wonder.promoted"]


def test_append_swallows_disabled_env(
    isolated_feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_FEED_LOG", "0")
    feed_log.append("belief.locked", id="x")
    assert not isolated_feed.exists()


def test_append_swallows_unserialisable_field(
    isolated_feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A field that json.dumps can't handle must not raise to the caller."""
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)

    class _NoSerialize:
        pass

    feed_log.append("belief.locked", id="x", weird=_NoSerialize())
    # File may or may not exist (no row was written) — what matters
    # is that the call returned None instead of raising.


def test_append_swallows_missing_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if the parent dir resolves to something un-mkdir-able,
    the writer must return cleanly."""
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    # Point feed_path at a path under a regular file (mkdir will fail).
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("blocker")
    monkeypatch.setattr(
        feed_log, "feed_path",
        lambda db_dir=None: blocker / "feed.jsonl",
    )
    feed_log.append("belief.locked", id="x")
    # No exception is the assertion.


def test_rotation_renames_oversized_log(
    isolated_feed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log exceeding ROTATE_BYTES is renamed before the next append."""
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    monkeypatch.setattr(feed_log, "ROTATE_BYTES", 100)
    isolated_feed.parent.mkdir(parents=True, exist_ok=True)
    isolated_feed.write_bytes(b"x" * 200)
    feed_log.append("belief.locked", id="post-rotation")
    archives = list(isolated_feed.parent.glob("feed.*.jsonl.archive"))
    assert len(archives) == 1
    # The fresh active log carries only the post-rotation row.
    fresh_rows = isolated_feed.read_text().splitlines()
    assert len(fresh_rows) == 1
    assert json.loads(fresh_rows[0])["id"] == "post-rotation"


def test_read_rows_returns_empty_when_no_file(tmp_path: Path) -> None:
    p = tmp_path / "feed.jsonl"
    assert feed_log.read_rows(p) == []


def test_read_rows_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "feed.jsonl"
    p.write_text(
        '{"ts":"2026-01-01T00:00:00Z","event":"belief.locked","id":"a"}\n'
        "not json\n"
        '{"ts":"2026-01-01T00:00:01Z","event":"belief.ingested","id":"b"}\n'
    )
    rows = feed_log.read_rows(p)
    assert len(rows) == 2
    assert [r["id"] for r in rows] == ["a", "b"]


def test_feed_path_is_sibling_of_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """feed_path() with no arg should resolve next to the resolved DB."""
    import aelfrice.db_paths as _db_paths_mod
    monkeypatch.setattr(
        _db_paths_mod, "db_path",
        lambda: tmp_path / "memory.db",
    )
    assert feed_log.feed_path() == tmp_path / "feed.jsonl"
