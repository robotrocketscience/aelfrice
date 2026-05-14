"""Tests for the #288 phase-1a rebuild diagnostic log.

Covers schema validity (round-trip a record), 5 MB size cap, opt-out
via env var, opt-out via .aelfrice.toml, fail-soft on I/O error, and
the no-op when the candidate set is empty.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    DEFAULT_REBUILDER_TOKEN_BUDGET,
    REBUILD_LOG_ENV,
    REBUILD_LOG_MAX_BYTES,
    RecentTurn,
    _append_rebuild_log_record,
    _build_rebuild_log_record,
    _rebuild_log_disabled_via_env,
    load_rebuilder_config,
    rebuild_v14,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---- helpers -----------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db_path: Path, beliefs: list[Belief]) -> MemoryStore:
    store = MemoryStore(str(db_path))
    for b in beliefs:
        store.insert_belief(b)
    return store


# ---- schema validity ---------------------------------------------------


def test_rebuild_log_record_round_trips_required_fields() -> None:
    """The on-disk JSON record carries every field the spec's Layer 1
    schema names, with the right shape."""
    turns = [
        RecentTurn(role="user", text="hello world"),
        RecentTurn(role="assistant", text="hi", session_id="s1"),
    ]
    candidates: list[dict[str, object]] = [
        {
            "belief_id": "abc",
            "rank": 1,
            "scores": {
                "bm25": None, "posterior_mean": None,
                "reranker": None, "final": None,
            },
            "lock_level": "user",
            "decision": "packed",
            "reason": None,
        },
    ]
    pack_summary: dict[str, int] = {
        "n_candidates": 1,
        "n_packed": 1,
        "n_dropped_by_floor": 0,
        "n_dropped_by_dedup": 0,
        "n_dropped_by_budget": 0,
        "total_chars_packed": 11,
    }
    record = _build_rebuild_log_record(
        recent_turns=turns,
        session_id="s1",
        candidates=candidates,
        pack_summary=pack_summary,
    )
    # Round-trip via JSON to confirm it serialises with no exotic types.
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    parsed = json.loads(line)

    assert parsed["session_id"] == "s1"
    assert isinstance(parsed["ts"], str) and parsed["ts"].endswith("Z")
    assert parsed["input"]["n_recent_turns"] == 2
    expected_hash = hashlib.sha256(b"hello worldhi").hexdigest()
    assert parsed["input"]["recent_turns_hash"] == expected_hash
    # extracted_query is whatever the rebuilder feeds retrieve(); it
    # must be a string (possibly empty for stopword-only input).
    assert isinstance(parsed["input"]["extracted_query"], str)
    assert isinstance(parsed["input"]["extracted_entities"], list)
    assert parsed["input"]["extracted_intent"] is None
    assert parsed["candidates"] == candidates
    assert parsed["pack_summary"] == pack_summary


def test_rebuild_v14_writes_one_record_per_invocation(
    tmp_path: Path,
) -> None:
    # L0 lock is the deterministic candidate-set source — it bypasses
    # FTS5 indexing details that aren't the subject under test here.
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1", "user prefers uv over pip",
            lock_level=LOCK_USER, locked_at="2026-04-28T00:00:00Z",
        )],
    )
    log_path = tmp_path / "rebuild_logs" / "sess1.jsonl"
    try:
        rebuild_v14(
            [RecentTurn(role="user", text="anything")],
            store,
            rebuild_log_path=log_path,
            session_id_for_log="sess1",
        )
    finally:
        store.close()

    assert log_path.exists()
    lines = [
        ln for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["session_id"] == "sess1"
    assert record["pack_summary"]["n_candidates"] >= 1
    assert record["pack_summary"]["n_packed"] >= 1
    # Every candidate carries the locked-down score block, even
    # though the values are null in phase-1a.
    for cand in record["candidates"]:
        assert set(cand["scores"].keys()) == {
            "bm25", "posterior_mean", "reranker", "final",
        }
        assert cand["decision"] in ("packed", "dropped")


def test_rebuild_v14_no_log_when_candidate_set_empty(
    tmp_path: Path,
) -> None:
    """Empty store + empty turns -> no candidates -> no log row."""
    store = _seed(tmp_path / "m.db", [])
    log_path = tmp_path / "rebuild_logs" / "empty.jsonl"
    try:
        rebuild_v14(
            [], store,
            rebuild_log_path=log_path,
            session_id_for_log="empty",
        )
    finally:
        store.close()
    assert not log_path.exists()


# ---- size cap ----------------------------------------------------------


def test_rebuild_log_appends_truncated_marker_at_cap(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "sess.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-fill the file to one byte under the cap so the very next
    # record write triggers the truncation path.
    # Pad ends with a newline so the marker appears on its own line,
    # mirroring real JSONL contents (every record ends in \n).
    pad = b"x" * (REBUILD_LOG_MAX_BYTES - 2) + b"\n"
    log_path.write_bytes(pad)
    record: dict[str, object] = {"hello": "world"}
    _append_rebuild_log_record(log_path, record, stderr=io.StringIO())
    # The original padding plus a truncated-marker JSON line must be
    # the only content; the actual record must NOT have been written.
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Trailing line is the truncated marker.
    last_line = text.splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed.get("truncated") is True
    assert "world" not in text  # the dropped record never landed


def test_rebuild_log_drops_writes_after_cap_reached(
    tmp_path: Path,
) -> None:
    """Once a session-file is at/over the cap, no further records get
    appended — not even another truncated marker."""
    log_path = tmp_path / "sess.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"y" * REBUILD_LOG_MAX_BYTES)
    size_before = log_path.stat().st_size
    _append_rebuild_log_record(
        log_path, {"x": 1}, stderr=io.StringIO(),
    )
    assert log_path.stat().st_size == size_before


# ---- opt-out -----------------------------------------------------------


def test_rebuild_log_env_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REBUILD_LOG_ENV, "0")
    assert _rebuild_log_disabled_via_env() is True

    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1", "x", lock_level=LOCK_USER,
            locked_at="2026-04-28T00:00:00Z",
        )],
    )
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"
    try:
        rebuild_v14(
            [RecentTurn(role="user", text="something")],
            store,
            rebuild_log_path=log_path,
            session_id_for_log="sess",
        )
    finally:
        store.close()
    assert not log_path.exists()


def test_rebuild_log_env_other_values_do_not_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the literal string '0' opts out; anything else (including
    'false'/'no') leaves the log enabled. Keeps the contract simple."""
    for val in ("1", "false", "no", "yes", ""):
        monkeypatch.setenv(REBUILD_LOG_ENV, val)
        assert _rebuild_log_disabled_via_env() is False, val


def test_rebuild_log_toml_opt_out(tmp_path: Path) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text(
        "[rebuild_log]\nenabled = false\n",
        encoding="utf-8",
    )
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is False


def test_rebuild_log_toml_default_on(tmp_path: Path) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[rebuilder]\n", encoding="utf-8")  # no log section
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is True


def test_rebuild_log_toml_invalid_type_falls_back_to_default(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text(
        '[rebuild_log]\nenabled = "yes"\n',  # wrong type
        encoding="utf-8",
    )
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is True


def test_rebuild_v14_respects_rebuild_log_enabled_kwarg(
    tmp_path: Path,
) -> None:
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1", "x", lock_level=LOCK_USER,
            locked_at="2026-04-28T00:00:00Z",
        )],
    )
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"
    try:
        rebuild_v14(
            [RecentTurn(role="user", text="something")],
            store,
            rebuild_log_path=log_path,
            session_id_for_log="sess",
            rebuild_log_enabled=False,
        )
    finally:
        store.close()
    assert not log_path.exists()


# ---- fail-soft ---------------------------------------------------------


def test_append_rebuild_log_fail_soft_on_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError raised by the file write is caught, traced to
    stderr, and never propagates."""
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"

    real_open = open

    def boom(*args: object, **kwargs: object) -> object:
        # Fail only when opening for append; let read-only opens
        # (e.g. the size-cap stat path) continue to work via
        # delegating to the real builtin.
        if "a" in str(args[1] if len(args) > 1 else kwargs.get("mode", "")):
            raise OSError("simulated disk-full")
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.open", boom)
    err = io.StringIO()
    # No exception escapes.
    _append_rebuild_log_record(log_path, {"x": 1}, stderr=err)
    assert "rebuild_log write failed" in err.getvalue()


def test_rebuild_v14_fail_soft_when_log_dir_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed log write does not break the rebuild itself."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "L1", "kitchen bananas", lock_level=LOCK_USER,
            locked_at="2026-04-28T00:00:00Z",
        )],
    )
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"

    def boom_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError("simulated permission denied")

    monkeypatch.setattr(Path, "mkdir", boom_mkdir)
    try:
        block = rebuild_v14(
            [RecentTurn(role="user", text="kitchen contents")],
            store,
            rebuild_log_path=log_path,
            session_id_for_log="sess",
        )
    finally:
        store.close()
    # Block was returned despite the log failure.
    assert "<aelfrice-rebuild>" in block
    assert not log_path.exists()
