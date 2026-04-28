"""Tests for `aelfrice.telemetry` — per-session telemetry writer.

Coverage matrix (per issue #140):

1. Compute delta for a known seeded session with mixed beliefs / feedback /
   corrections — exact field values asserted.
2. Empty session_id passed to emit_session_delta: zero counts, no crash,
   writes nothing to the file.
3. Schema validation: emitted row contains every required v=1 top-level key
   and nested key from the reference schema.
4. Window rollup against a synthetic telemetry.jsonl with 7 rows in the past
   5 days → window_7.sessions_in_window == 7.
5. CLI exit 0 when --id is missing / empty (silent no-op, warns to stderr).
6. Multiple sessions in the same DB — no cross-bleed between session_ids.
7. Active session with non-zero deltas: beliefs_created > 0.
8. Idle session (no beliefs tagged): all per-session delta counts are 0,
   row is still emitted.
9. Window rows older than the cutoff are excluded.

All tests:
  - use ``tmp_path`` / ``monkeypatch``; no real ``~/.aelfrice`` writes
  - are deterministic (fixed ``now`` datetime passed explicitly)
  - complete in << 2s (no network, no subprocess)
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.telemetry import (
    DEFAULT_TELEMETRY_PATH,
    compute_session_delta,
    emit_session_delta,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
_SID = "abc123session"
_SID2 = "def456session"


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memory.db"))


def _belief(
    id_: str,
    content: str,
    session_id: str | None = None,
    btype: str = BELIEF_FACTUAL,
) -> Belief:
    import hashlib

    return Belief(
        id=id_,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=btype,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=_FIXED_NOW.isoformat(),
        last_retrieved_at=None,
        session_id=session_id,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_store(store: MemoryStore, session_id: str) -> None:
    """Insert 3 beliefs (2 factual + 1 correction) tagged with session_id."""
    store.insert_belief(_belief("b1", "fact one", session_id, BELIEF_FACTUAL))
    store.insert_belief(_belief("b2", "fact two", session_id, BELIEF_FACTUAL))
    store.insert_belief(_belief("b3", "correction A", session_id, BELIEF_CORRECTION))
    # Add one feedback event on b1
    store.insert_feedback_event("b1", 1.0, "user", _FIXED_NOW.isoformat())


def _make_telemetry_file(tmp_path: Path, n_rows: int, days_ago: float = 2.0) -> Path:
    """Write n_rows telemetry rows dated ``days_ago`` days before _FIXED_NOW."""
    path = tmp_path / "telemetry.jsonl"
    ts = datetime(
        2026, 4, 27 - int(days_ago), 12, 0, 0, tzinfo=timezone.utc
    ).isoformat()
    rows = [
        {
            "v": 1,
            "ts": ts,
            "session": {
                "retrieval_tokens": 0,
                "classification_tokens": 0,
                "beliefs_created": 3,
                "corrections_detected": 1,
                "searches_performed": 0,
                "feedback_given": 1,
                "velocity_items_per_hour": 6.0,
                "velocity_tier": "deep",
                "duration_seconds": 1800.0,
            },
            "feedback": {
                "outcome_counts": {"used": 1, "ignored": 0},
                "detection_layer_counts": {"implicit": 0, "explicit": 1},
                "feedback_rate": 0.333,
            },
            "beliefs": {},
            "graph": {},
            "window_7": {},
            "window_30": {},
        }
        for _ in range(n_rows)
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Test 1: compute delta for seeded session — exact field values
# ---------------------------------------------------------------------------


def test_compute_session_delta_exact_values(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_store(store, _SID)
    tpath = tmp_path / "tel.jsonl"  # empty file, no rows

    row = compute_session_delta(store, _SID, telemetry_path=tpath, now=_FIXED_NOW)
    store.close()

    sess = row["session"]
    assert sess["beliefs_created"] == 3
    assert sess["corrections_detected"] == 1
    assert sess["feedback_given"] == 1
    assert sess["retrieval_tokens"] == 0
    assert sess["classification_tokens"] == 0
    assert sess["searches_performed"] == 0
    assert isinstance(sess["velocity_items_per_hour"], float)
    assert isinstance(sess["velocity_tier"], str)
    assert isinstance(sess["duration_seconds"], float)

    fb = row["feedback"]
    assert fb["outcome_counts"]["used"] == 1
    assert isinstance(fb["feedback_rate"], float)

    assert row["v"] == 1
    assert isinstance(row["ts"], str)


# ---------------------------------------------------------------------------
# Test 2: empty session_id — no crash, nothing written
# ---------------------------------------------------------------------------


def test_emit_empty_session_id_is_noop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = _make_store(tmp_path)
    tpath = tmp_path / "tel.jsonl"
    emit_session_delta("", store=store, path=tpath, now=_FIXED_NOW)
    store.close()

    assert not tpath.exists(), "nothing should be written for empty session_id"
    captured = capsys.readouterr()
    assert "skipping" in captured.err.lower() or "empty" in captured.err.lower()


# ---------------------------------------------------------------------------
# Test 3: schema validation — all v=1 top-level and nested keys present
# ---------------------------------------------------------------------------


def test_emitted_row_schema_complete(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_store(store, _SID)
    tpath = tmp_path / "tel.jsonl"
    emit_session_delta(_SID, store=store, path=tpath, now=_FIXED_NOW)
    store.close()

    assert tpath.exists()
    with tpath.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    row = rows[0]

    # Top-level keys
    for key in ("v", "ts", "session", "feedback", "beliefs", "graph", "window_7", "window_30"):
        assert key in row, f"missing top-level key: {key!r}"

    # session block
    for key in (
        "retrieval_tokens", "classification_tokens", "beliefs_created",
        "corrections_detected", "searches_performed", "feedback_given",
        "velocity_items_per_hour", "velocity_tier", "duration_seconds",
    ):
        assert key in row["session"], f"missing session.{key!r}"

    # feedback block
    for key in ("outcome_counts", "detection_layer_counts", "feedback_rate"):
        assert key in row["feedback"], f"missing feedback.{key!r}"

    # beliefs block
    for key in (
        "total_active", "total_superseded", "total_locked",
        "confidence_distribution", "type_distribution", "source_distribution",
        "churn_rate", "orphan_count",
    ):
        assert key in row["beliefs"], f"missing beliefs.{key!r}"

    # graph block
    for key in ("total_edges", "edge_type_distribution", "avg_edges_per_belief"):
        assert key in row["graph"], f"missing graph.{key!r}"

    # window blocks
    for window in ("window_7", "window_30"):
        for key in ("sessions_in_window", "totals", "averages", "feedback_rate", "correction_rate"):
            assert key in row[window], f"missing {window}.{key!r}"


# ---------------------------------------------------------------------------
# Test 4: window rollup — 7 rows in past 5 days → window_7 count == 7
# ---------------------------------------------------------------------------


def test_window_rollup_7_rows_in_window(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    # 7 rows dated 2 days before _FIXED_NOW (2026-04-25), inside the 7-day window
    tpath = _make_telemetry_file(tmp_path, n_rows=7, days_ago=2.0)

    row = compute_session_delta(store, "nosession", telemetry_path=tpath, now=_FIXED_NOW)
    store.close()

    assert row["window_7"]["sessions_in_window"] == 7
    assert row["window_30"]["sessions_in_window"] == 7


# ---------------------------------------------------------------------------
# Test 5: CLI exit 0 when --id is missing or empty
# ---------------------------------------------------------------------------


def test_cli_session_delta_missing_id_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    tpath = tmp_path / "tel.jsonl"

    # --id not supplied (argparse defaults to None → empty string in handler)
    out = io.StringIO()
    code = cli_main(["session-delta", "--telemetry-path", str(tpath)], out=out)
    assert code == 0
    assert not tpath.exists()

    captured = capsys.readouterr()
    assert "skipping" in captured.err.lower() or "empty" in captured.err.lower()


def test_cli_session_delta_with_id_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    tpath = tmp_path / "tel.jsonl"

    out = io.StringIO()
    code = cli_main(
        ["session-delta", "--id", "somesession123", "--telemetry-path", str(tpath)],
        out=out,
    )
    assert code == 0
    # File should exist (even for an unknown session — idle row emitted)
    assert tpath.exists()
    with tpath.open() as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["v"] == 1
    assert row["session"]["beliefs_created"] == 0  # unknown session → idle


# ---------------------------------------------------------------------------
# Test 6: multiple sessions — no cross-bleed
# ---------------------------------------------------------------------------


def test_no_cross_bleed_between_sessions(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_store(store, _SID)
    # Insert 1 belief under _SID2
    store.insert_belief(_belief("b99", "only in sid2", _SID2, BELIEF_FACTUAL))

    tpath = tmp_path / "tel.jsonl"
    row1 = compute_session_delta(store, _SID, telemetry_path=tpath, now=_FIXED_NOW)
    row2 = compute_session_delta(store, _SID2, telemetry_path=tpath, now=_FIXED_NOW)
    store.close()

    assert row1["session"]["beliefs_created"] == 3
    assert row2["session"]["beliefs_created"] == 1
    assert row1["session"]["corrections_detected"] == 1
    assert row2["session"]["corrections_detected"] == 0


# ---------------------------------------------------------------------------
# Test 7: active session → non-zero beliefs_created
# ---------------------------------------------------------------------------


def test_active_session_nonzero_beliefs_created(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_store(store, _SID)
    tpath = tmp_path / "tel.jsonl"

    row = compute_session_delta(store, _SID, telemetry_path=tpath, now=_FIXED_NOW)
    store.close()

    assert row["session"]["beliefs_created"] > 0


# ---------------------------------------------------------------------------
# Test 8: idle session — zero counts, row still emitted
# ---------------------------------------------------------------------------


def test_idle_session_emits_zero_row(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    tpath = tmp_path / "tel.jsonl"

    emit_session_delta("idle-session-xyz", store=store, path=tpath, now=_FIXED_NOW)
    store.close()

    assert tpath.exists()
    with tpath.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["session"]["beliefs_created"] == 0
    assert row["session"]["corrections_detected"] == 0
    assert row["session"]["feedback_given"] == 0


# ---------------------------------------------------------------------------
# Test 9: window rows older than cutoff are excluded
# ---------------------------------------------------------------------------


def test_window_excludes_rows_older_than_cutoff(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    tpath = tmp_path / "tel.jsonl"

    # Write 3 rows 10 days before _FIXED_NOW (should be OUTSIDE window_7)
    ts_old = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    # Write 2 rows 3 days before _FIXED_NOW (INSIDE window_7)
    ts_recent = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc).isoformat()

    def _row(ts: str) -> dict[str, object]:
        return {
            "v": 1,
            "ts": ts,
            "session": {
                "beliefs_created": 5,
                "corrections_detected": 0,
                "feedback_given": 0,
                "velocity_items_per_hour": 10.0,
            },
            "feedback": {},
            "beliefs": {},
            "graph": {},
            "window_7": {},
            "window_30": {},
        }

    with tpath.open("w", encoding="utf-8") as fh:
        for _ in range(3):
            fh.write(json.dumps(_row(ts_old)) + "\n")
        for _ in range(2):
            fh.write(json.dumps(_row(ts_recent)) + "\n")

    row = compute_session_delta(store, "any", telemetry_path=tpath, now=_FIXED_NOW)
    store.close()

    # Only the 2 recent rows should be in window_7
    assert row["window_7"]["sessions_in_window"] == 2
    # All 5 rows fall within 30 days of _FIXED_NOW
    assert row["window_30"]["sessions_in_window"] == 5
