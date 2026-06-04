"""SessionStart belief-write recap injection (#934).

Tests for build_session_start_recap_line (pure helper) and the full
session_start() integration path.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    ENV_SESSIONSTART_RECAP,
    ENV_SESSIONSTART_RECAP_THRESHOLD,
    _DEFAULT_RECAP_THRESHOLD,
    build_session_start_recap_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rows(events: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Build a minimal list of feed-log rows from (ts, event) pairs."""
    return [{"ts": ts, "event": ev} for ts, ev in events]


_BELIEF_EVENT = "belief.locked"
_NON_BELIEF_EVENT = "session.start"  # not in the belief-write set


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestBuildSessionStartRecapLine:
    def test_threshold_honored_at_boundary(self) -> None:
        """N == threshold → inject."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * _DEFAULT_RECAP_THRESHOLD
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=_DEFAULT_RECAP_THRESHOLD,
        )
        assert line is not None
        assert str(_DEFAULT_RECAP_THRESHOLD) in line

    def test_below_threshold_returns_none(self) -> None:
        """N == threshold - 1 → no inject."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * (_DEFAULT_RECAP_THRESHOLD - 1)
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=_DEFAULT_RECAP_THRESHOLD,
        )
        assert line is None

    def test_zero_events_returns_none(self) -> None:
        """Empty feed → no output, no '0 beliefs' string."""
        line = build_session_start_recap_line(
            feed_rows=[],
            last_ts=None,
            threshold=_DEFAULT_RECAP_THRESHOLD,
        )
        assert line is None

    def test_threshold_env_override_low(self) -> None:
        """Custom threshold=5, N=4 → no inject."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * 4
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=5,
        )
        assert line is None

    def test_threshold_env_override_high(self) -> None:
        """Custom threshold=5, N=5 → inject."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * 5
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=5,
        )
        assert line is not None
        assert "5 beliefs" in line

    def test_since_last_ts_filtering(self) -> None:
        """Feed has 5 rows; last_ts between row 2 and 3 → count == 3."""
        rows = _make_rows([
            ("2026-06-04T09:00:00Z", _BELIEF_EVENT),
            ("2026-06-04T09:30:00Z", _BELIEF_EVENT),
            ("2026-06-04T10:00:00Z", _BELIEF_EVENT),
            ("2026-06-04T10:30:00Z", _BELIEF_EVENT),
            ("2026-06-04T11:00:00Z", _BELIEF_EVENT),
        ])
        # last_ts is just after row 2 (ts 09:30) and before row 3 (ts 10:00)
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts="2026-06-04T09:45:00Z",
            threshold=3,
        )
        assert line is not None
        assert "3 beliefs" in line

    def test_first_run_no_last_ts_counts_all(self) -> None:
        """No last_ts (first run) → all rows count."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * 4
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=3,
        )
        assert line is not None
        assert "4 beliefs" in line

    def test_non_belief_write_events_excluded(self) -> None:
        """Non-belief-write event types are not counted."""
        rows = _make_rows([
            ("2026-06-04T10:00:00Z", _NON_BELIEF_EVENT),
            ("2026-06-04T10:01:00Z", _NON_BELIEF_EVENT),
            ("2026-06-04T10:02:00Z", _NON_BELIEF_EVENT),
            ("2026-06-04T10:03:00Z", _BELIEF_EVENT),
            ("2026-06-04T10:04:00Z", _BELIEF_EVENT),
        ])
        # Only 2 belief-write events → below threshold=3
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=3,
        )
        assert line is None

    def test_all_belief_write_event_types_counted(self) -> None:
        """All four belief-write event types are recognised."""
        rows = _make_rows([
            ("2026-06-04T10:00:00Z", "belief.locked"),
            ("2026-06-04T10:01:00Z", "belief.ingested"),
            ("2026-06-04T10:02:00Z", "wonder.promoted"),
            ("2026-06-04T10:03:00Z", "feedback.applied"),
        ])
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=3,
        )
        assert line is not None
        assert "4 beliefs" in line

    def test_template_format(self) -> None:
        """Exact output shape matches the spec template."""
        rows = _make_rows(
            [("2026-06-04T10:00:00Z", _BELIEF_EVENT)] * 5
        )
        line = build_session_start_recap_line(
            feed_rows=rows,
            last_ts=None,
            threshold=3,
        )
        assert line == (
            "aelfrice: 5 beliefs written since last session"
            " — `aelf:feed -n 5` to review."
        )


# ---------------------------------------------------------------------------
# Env-var opt-out test (using _recap_enabled helper)
# ---------------------------------------------------------------------------


class TestRecapEnabled:
    def test_enabled_by_default(self) -> None:
        from aelfrice.hook import _recap_enabled
        assert _recap_enabled({}) is True

    def test_disabled_via_env(self) -> None:
        from aelfrice.hook import _recap_enabled
        assert _recap_enabled({ENV_SESSIONSTART_RECAP: "0"}) is False

    def test_non_zero_value_still_enabled(self) -> None:
        from aelfrice.hook import _recap_enabled
        assert _recap_enabled({ENV_SESSIONSTART_RECAP: "1"}) is True


class TestRecapThreshold:
    def test_default(self) -> None:
        from aelfrice.hook import _recap_threshold
        assert _recap_threshold({}) == _DEFAULT_RECAP_THRESHOLD

    def test_override(self) -> None:
        from aelfrice.hook import _recap_threshold
        assert _recap_threshold({ENV_SESSIONSTART_RECAP_THRESHOLD: "7"}) == 7

    def test_invalid_falls_back_to_default(self) -> None:
        from aelfrice.hook import _recap_threshold
        assert _recap_threshold({ENV_SESSIONSTART_RECAP_THRESHOLD: "notanint"}) == (
            _DEFAULT_RECAP_THRESHOLD
        )


# ---------------------------------------------------------------------------
# Integration test: full session_start() path
# ---------------------------------------------------------------------------


def _write_feed_jsonl(db_dir: Path, rows: list[dict[str, object]]) -> None:
    feed = db_dir / "feed.jsonl"
    feed.parent.mkdir(parents=True, exist_ok=True)
    with feed.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_session_start_integration_recap_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_start() emits the recap line when threshold is met."""
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    # Point AELFRICE_DB to a non-existent DB so store opens as empty
    monkeypatch.setenv("AELFRICE_DB", str(db_dir / "memory.db"))
    # Write a feed.jsonl with enough belief-write events
    rows: list[dict[str, object]] = [
        {"ts": f"2026-06-04T10:0{i}:00Z", "event": "belief.locked"}
        for i in range(5)
    ]
    _write_feed_jsonl(db_dir, rows)
    # Use threshold=3 (default)
    monkeypatch.setenv(ENV_SESSIONSTART_RECAP, "1")
    monkeypatch.setenv(ENV_SESSIONSTART_RECAP_THRESHOLD, "3")

    sout = io.StringIO()
    from aelfrice.hook import session_start
    rc = session_start(
        stdin=io.StringIO('{"session_id": "test-session"}'),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    output = sout.getvalue()
    assert "5 beliefs written since last session" in output
    assert "`aelf:feed -n 5` to review." in output


def test_session_start_integration_recap_suppressed_by_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_start() emits NO recap line when AELFRICE_SESSIONSTART_RECAP=0."""
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(db_dir / "memory.db"))
    rows: list[dict[str, object]] = [
        {"ts": f"2026-06-04T10:0{i}:00Z", "event": "belief.locked"}
        for i in range(5)
    ]
    _write_feed_jsonl(db_dir, rows)
    monkeypatch.setenv(ENV_SESSIONSTART_RECAP, "0")

    sout = io.StringIO()
    from aelfrice.hook import session_start
    rc = session_start(
        stdin=io.StringIO('{"session_id": "test-session"}'),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    output = sout.getvalue()
    assert "beliefs written since last session" not in output


def test_session_start_integration_recap_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_start() emits no recap when event count < threshold."""
    db_dir = tmp_path / "aelfrice"
    db_dir.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(db_dir / "memory.db"))
    # Only 2 rows — below default threshold of 3
    rows: list[dict[str, object]] = [
        {"ts": f"2026-06-04T10:0{i}:00Z", "event": "belief.locked"}
        for i in range(2)
    ]
    _write_feed_jsonl(db_dir, rows)
    monkeypatch.setenv(ENV_SESSIONSTART_RECAP, "1")

    sout = io.StringIO()
    from aelfrice.hook import session_start
    rc = session_start(
        stdin=io.StringIO('{"session_id": "test-session"}'),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    output = sout.getvalue()
    assert "beliefs written since last session" not in output
