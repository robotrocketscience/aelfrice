"""Tests for the #288 phase-1a rebuild diagnostic log.

Phase 1a instrumentation: per-rebuild JSONL written by rebuild_v14().

This file covers the write helpers and config plumbing added in the
first atomic commit.  Integration tests (rebuild_v14() wiring) are
added in the second atomic commit that wires the helper into the
packing loop.

Covers in this commit:
  - Schema validity (every Layer 1 field, sha256 hash, ISO8601-Z ts)
  - Determinism: clock injection via now_fn kwarg
  - 5 MB size cap: truncated sentinel + no writes after cap
  - Env opt-out: AELFRICE_REBUILD_LOG=0 (env-var helper only)
  - TOML opt-out: [rebuild_log] enabled = false (config loading)
  - Fail-soft: simulated OSError on append -> error to stderr, no raise
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    REBUILD_LOG_ENV,
    REBUILD_LOG_MAX_BYTES,
    RecentTurn,
    _append_rebuild_log_record,
    _build_rebuild_log_record,
    _rebuild_log_disabled_via_env,
    _rebuild_log_truncated,
    load_rebuilder_config,
)


# ---- schema validity ---------------------------------------------------


def test_rebuild_log_record_shape_matches_spec() -> None:
    """_build_rebuild_log_record() emits every Layer 1 field with the
    right shape; JSON-round-trips cleanly (no unserializable types)."""
    turns = [
        RecentTurn(role="user", text="hello world"),
        RecentTurn(role="assistant", text="hi there", session_id="s1"),
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
    # JSON round-trip: no exotic types.
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    parsed = json.loads(line)

    # Top-level keys
    assert set(parsed.keys()) == {"ts", "session_id", "input", "candidates", "pack_summary"}

    # ts must be ISO8601 with Z
    ts = parsed["ts"]
    assert isinstance(ts, str)
    assert ts.endswith("Z"), f"ts must end with Z, got: {ts!r}"
    # Must be parseable
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

    # session_id
    assert parsed["session_id"] == "s1"

    # input
    inp = parsed["input"]
    assert set(inp.keys()) == {
        "recent_turns_hash", "n_recent_turns",
        "extracted_query", "extracted_entities", "extracted_intent",
    }
    assert inp["n_recent_turns"] == 2
    assert inp["extracted_intent"] is None

    # recent_turns_hash must be sha256 hex (64 hex chars), not prose
    h = inp["recent_turns_hash"]
    assert isinstance(h, str)
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", h), f"not hex: {h!r}"
    # Verify the hash matches the spec: sha256 of turns joined by \n
    expected_hash = hashlib.sha256(
        "\n".join(t.text for t in turns).encode("utf-8")
    ).hexdigest()
    assert h == expected_hash

    # candidates
    cands = parsed["candidates"]
    assert len(cands) == 1
    cand = cands[0]
    assert set(cand.keys()) == {
        "belief_id", "rank", "scores", "lock_level", "decision", "reason",
    }
    assert set(cand["scores"].keys()) == {"bm25", "posterior_mean", "reranker", "final"}
    assert cand["decision"] in ("packed", "dropped")

    # pack_summary
    ps = parsed["pack_summary"]
    assert set(ps.keys()) == {
        "n_candidates", "n_packed",
        "n_dropped_by_floor", "n_dropped_by_dedup",
        "n_dropped_by_budget", "total_chars_packed",
    }


def test_rebuild_log_recent_turns_hash_is_sha256_not_prose() -> None:
    """recent_turns_hash must be the raw hex digest, never a prose snippet."""
    turns = [RecentTurn(role="user", text="some user text here")]
    record = _build_rebuild_log_record(
        recent_turns=turns,
        session_id=None,
        candidates=[],
        pack_summary={
            "n_candidates": 0, "n_packed": 0,
            "n_dropped_by_floor": 0, "n_dropped_by_dedup": 0,
            "n_dropped_by_budget": 0, "total_chars_packed": 0,
        },
    )
    h = record["input"]["recent_turns_hash"]  # type: ignore[index]
    assert isinstance(h, str)
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    # Must NOT contain any prose from the turn
    assert "some" not in h
    assert "user" not in h


def test_rebuild_log_ts_is_iso8601_with_z() -> None:
    """ts field is ISO-8601 UTC with trailing Z."""
    record = _build_rebuild_log_record(
        recent_turns=[],
        session_id=None,
        candidates=[],
        pack_summary={
            "n_candidates": 0, "n_packed": 0,
            "n_dropped_by_floor": 0, "n_dropped_by_dedup": 0,
            "n_dropped_by_budget": 0, "total_chars_packed": 0,
        },
    )
    ts = record["ts"]
    assert isinstance(ts, str)
    assert ts.endswith("Z")
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


# ---- determinism / clock injection ------------------------------------


def test_rebuild_log_clock_injection() -> None:
    """now_fn kwarg controls the ts value; enables deterministic tests."""
    fixed = datetime(2026, 4, 29, 3, 14, 15, tzinfo=timezone.utc)

    def fixed_now() -> datetime:
        return fixed

    record = _build_rebuild_log_record(
        recent_turns=[RecentTurn(role="user", text="x")],
        session_id="s1",
        candidates=[],
        pack_summary={
            "n_candidates": 0, "n_packed": 0,
            "n_dropped_by_floor": 0, "n_dropped_by_dedup": 0,
            "n_dropped_by_budget": 0, "total_chars_packed": 0,
        },
        now_fn=fixed_now,
    )
    assert record["ts"] == "2026-04-29T03:14:15Z"


# ---- size cap ---------------------------------------------------------


def test_rebuild_log_appends_truncated_marker_at_cap(
    tmp_path: Path,
) -> None:
    """When adding a record would exceed 5 MB, writes a truncated sentinel."""
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"
    log_path.parent.mkdir(parents=True)
    # Pre-fill file to just under the cap with valid JSON + padding.
    # json.dumps wrapping adds ~10 bytes of overhead; use -20 to ensure
    # the fill is at REBUILD_LOG_MAX_BYTES - 20, so any small write crosses.
    pad = "x" * (REBUILD_LOG_MAX_BYTES - 20)
    fill_line = json.dumps({"pad": pad}) + "\n"
    # If the fill itself already exceeds the cap, trim to exactly cap - 1.
    fill_bytes = fill_line.encode("utf-8")
    if len(fill_bytes) >= REBUILD_LOG_MAX_BYTES:
        fill_bytes = fill_bytes[: REBUILD_LOG_MAX_BYTES - 1]
    log_path.write_bytes(fill_bytes)
    # Clear module-level truncation state for this path.
    _rebuild_log_truncated.pop(log_path, None)

    _append_rebuild_log_record(log_path, {"key": "value_that_crosses_cap"})

    content = log_path.read_text(encoding="utf-8")
    last_line = [ln for ln in content.splitlines() if ln.strip()][-1]
    sentinel = json.loads(last_line)
    assert sentinel.get("truncated") is True
    assert sentinel.get("reason") == "size_cap_5mb"
    assert "ts" in sentinel


def test_rebuild_log_drops_writes_after_cap_reached(
    tmp_path: Path,
) -> None:
    """After the cap is crossed, further appends produce no additional rows."""
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"
    log_path.parent.mkdir(parents=True)
    pad = "x" * (REBUILD_LOG_MAX_BYTES - 10)
    fill_line = json.dumps({"pad": pad}) + "\n"
    log_path.write_text(fill_line, encoding="utf-8")
    _rebuild_log_truncated.pop(log_path, None)

    _append_rebuild_log_record(log_path, {"a": "first"})  # writes sentinel
    size_after_sentinel = log_path.stat().st_size
    _append_rebuild_log_record(log_path, {"b": "second"})  # must be no-op
    assert log_path.stat().st_size == size_after_sentinel


# ---- env opt-out -------------------------------------------------------


def test_rebuild_log_env_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AELFRICE_REBUILD_LOG=0 causes _rebuild_log_disabled_via_env to return True."""
    monkeypatch.setenv(REBUILD_LOG_ENV, "0")
    assert _rebuild_log_disabled_via_env() is True


def test_rebuild_log_env_other_values_do_not_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only '0' disables; '1', 'false', '' do not."""
    for val in ("1", "false", "off", ""):
        monkeypatch.setenv(REBUILD_LOG_ENV, val)
        assert _rebuild_log_disabled_via_env() is False
    monkeypatch.delenv(REBUILD_LOG_ENV, raising=False)
    assert _rebuild_log_disabled_via_env() is False


# ---- TOML opt-out ------------------------------------------------------


def test_rebuild_log_toml_opt_out(tmp_path: Path) -> None:
    """[rebuild_log] enabled = false disables via config."""
    (tmp_path / ".aelfrice.toml").write_text(
        "[rebuild_log]\nenabled = false\n", encoding="utf-8",
    )
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is False


def test_rebuild_log_toml_default_on(tmp_path: Path) -> None:
    """No [rebuild_log] section -> enabled defaults to True."""
    (tmp_path / ".aelfrice.toml").write_text(
        "[rebuilder]\n", encoding="utf-8",
    )
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is True


def test_rebuild_log_toml_invalid_type_falls_back_to_default(
    tmp_path: Path,
) -> None:
    """[rebuild_log] enabled = 'yes' (wrong type) -> falls back to True."""
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuild_log]\nenabled = "yes"\n', encoding="utf-8",
    )
    config = load_rebuilder_config(start=tmp_path)
    assert config.rebuild_log_enabled is True


# ---- fail-soft ---------------------------------------------------------


def test_append_rebuild_log_fail_soft_on_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OSError on the append is caught, traced to stderr, never raises."""
    log_path = tmp_path / "rebuild_logs" / "sess.jsonl"
    _rebuild_log_truncated.pop(log_path, None)

    real_open = open

    def boom(*args: object, **kwargs: object) -> object:
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "")
        if "a" in str(mode):
            raise OSError("simulated disk-full")
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.open", boom)
    err = io.StringIO()
    _append_rebuild_log_record(log_path, {"x": 1}, stderr=err)
    assert "rebuild_log write failed" in err.getvalue()
