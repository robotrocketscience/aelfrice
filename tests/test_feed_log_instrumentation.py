"""Integration: verify the 5 write-path CLI commands emit feed-log rows (#931)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import feed_log
from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db_and_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Per-test DB + redirect feed_log.feed_path() to a tmp file."""
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    feed_file = tmp_path / "feed.jsonl"
    monkeypatch.setattr(
        feed_log, "feed_path",
        lambda db_dir=None: feed_file,
    )
    return feed_file


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _read_feed(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text().splitlines()
        if line.strip()
    ]


def _seed_unlocked(db: Path, bid: str = "deadbeef00000001") -> None:
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id=bid, content="seed belief",
            content_hash=f"hash-{bid}",
            alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
            created_at="2026-05-01T00:00:00Z",
            last_retrieved_at=None,
            origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()


def test_lock_emits_belief_locked(
    isolated_db_and_feed: Path, tmp_path: Path,
) -> None:
    code, _out = _run("lock", "atomic commits beat batched")
    assert code == 0
    rows = _read_feed(isolated_db_and_feed)
    locked_rows = [r for r in rows if r["event"] == "belief.locked"]
    assert len(locked_rows) == 1
    assert "id" in locked_rows[0]
    assert "snippet" in locked_rows[0]
    assert "atomic commits" in locked_rows[0]["snippet"]


def test_lock_respects_feed_log_disabled(
    isolated_db_and_feed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_FEED_LOG", "0")
    code, _out = _run("lock", "this should not write a feed row")
    assert code == 0
    assert not isolated_db_and_feed.exists()


def test_promote_emits_wonder_promoted(
    isolated_db_and_feed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    db = Path(os.environ["AELFRICE_DB"])
    _seed_unlocked(db, bid="deadbeef00000001")
    code, _out = _run("promote", "deadbeef00000001")
    assert code == 0
    rows = _read_feed(isolated_db_and_feed)
    promoted = [r for r in rows if r["event"] == "wonder.promoted"]
    assert len(promoted) == 1
    assert promoted[0]["id"] == "deadbeef00000001"
    assert "from_origin" in promoted[0]
    assert "to_origin" in promoted[0]


def test_confirm_emits_feedback_applied(
    isolated_db_and_feed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    db = Path(os.environ["AELFRICE_DB"])
    _seed_unlocked(db, bid="deadbeef00000002")
    code, _out = _run("confirm", "deadbeef00000002", "--source", "operator")
    assert code == 0
    rows = _read_feed(isolated_db_and_feed)
    fb = [r for r in rows if r["event"] == "feedback.applied"]
    assert len(fb) == 1
    assert fb[0]["kind"] == "confirm"
    assert fb[0]["new_alpha"] > fb[0]["prior_alpha"]


def test_feedback_emits_feedback_applied(
    isolated_db_and_feed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    db = Path(os.environ["AELFRICE_DB"])
    _seed_unlocked(db, bid="deadbeef00000003")
    code, _out = _run(
        "feedback", "deadbeef00000003",
        "used", "--source", "operator",
    )
    assert code == 0
    rows = _read_feed(isolated_db_and_feed)
    fb = [r for r in rows if r["event"] == "feedback.applied"]
    assert len(fb) == 1
    assert fb[0]["kind"] == "feedback"
    assert fb[0]["signal"] == "used"


def test_lock_stdout_unchanged_by_feed_log(
    isolated_db_and_feed: Path,
) -> None:
    """The existing CLI summary must be unchanged — adding feed-log
    instrumentation does NOT alter stdout."""
    code, out = _run("lock", "preserve stdout contract")
    assert code == 0
    # The lock CLI's signature line still leads with "locked:".
    assert "locked:" in out
