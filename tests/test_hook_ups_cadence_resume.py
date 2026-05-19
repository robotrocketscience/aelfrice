"""Integration tests for UPS-side cadence-resume injection (#871).

Tests against the actual :func:`hook.user_prompt_submit` entry point.
Covers cache hit/miss, TTL gating, malformed-cache tolerance, and
single-shot-per-session semantics.
"""
from __future__ import annotations

import io
import json
import os
import textwrap
import time
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import CONFIG_FILENAME


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        "AELFRICE_CADENCE_CTX_THRESHOLD",
        "AELFRICE_CADENCE_CTX_BYTE_WINDOW",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore
    MemoryStore(str(db)).close()


def _write_cache(db: Path, body: str, policy: str = "p2_ctx_threshold",
                 session_id: str = "prev-sess-12345") -> Path:
    """Write a cadence resume cache next to the brain-graph DB."""
    cache_dir = db.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cadence_resume_cache.json"
    record = {
        "ts": "2026-05-19T22:00:00Z",
        "session_id": session_id,
        "policy": policy,
        "body": body,
    }
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    return cache_path


def _set_cache_mtime_seconds_ago(cache_path: Path, age_seconds: float) -> None:
    """Backdate the cache mtime by ``age_seconds`` for TTL tests."""
    now = time.time()
    os.utime(cache_path, (now - age_seconds, now - age_seconds))


def _ups_payload(session_id: str, prompt: str, cwd: Path) -> str:
    return json.dumps({
        "session_id": session_id,
        "prompt": prompt,
        "cwd": str(cwd),
    })


def _run_ups(payload_str: str) -> tuple[str, str]:
    """Run hook.user_prompt_submit and return (stdout, stderr)."""
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hook.user_prompt_submit(
        stdin=io.StringIO(payload_str),
        stdout=sout,
        stderr=serr,
    )
    assert rc == 0
    return sout.getvalue(), serr.getvalue()


# --- Tests ---------------------------------------------------------------


def test_no_cache_no_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First UPS of a session, no resume cache → no injection in stdout."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)

    out, err = _run_ups(_ups_payload("new-sess-1", "hi", tmp_path))
    assert "<cadence-resume" not in out
    assert "cadence-resume injection" not in err


def test_fresh_cache_injects_on_first_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh cache → injected as <cadence-resume> block + stderr line."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_cache(db, body="<rebuilder synthesis from prior session>")

    out, err = _run_ups(_ups_payload("new-sess-2", "hello", tmp_path))
    assert "<cadence-resume from='prev-ses'" in out
    assert "policy='p2_ctx_threshold'" in out
    assert "rebuilder synthesis from prior session" in out
    assert "</cadence-resume>" in out
    assert "cadence-resume injection (from prev-ses" in err


def test_cache_ttl_expired_no_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache mtime > 1 hour old → no injection."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    cache = _write_cache(db, body="<stale synthesis>")
    # Backdate 2 hours.
    _set_cache_mtime_seconds_ago(cache, age_seconds=7200)

    out, err = _run_ups(_ups_payload("new-sess-3", "hi", tmp_path))
    assert "<cadence-resume" not in out
    assert "cadence-resume injection" not in err


def test_cache_ttl_just_under_limit_injects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache mtime ~30 min old → still within 1-hour TTL → injects."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    cache = _write_cache(db, body="<recent synthesis>")
    _set_cache_mtime_seconds_ago(cache, age_seconds=1800)

    out, _err = _run_ups(_ups_payload("new-sess-4", "hi", tmp_path))
    assert "recent synthesis" in out


def test_malformed_cache_json_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache file is unparseable JSON → no injection, no crash, stderr trace."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    cache = db.parent / "cadence_resume_cache.json"
    cache.write_text("{{{ not valid json")

    out, err = _run_ups(_ups_payload("new-sess-5", "hi", tmp_path))
    assert "<cadence-resume" not in out
    assert "cadence resume read failed" in err


def test_cache_with_empty_body_no_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    cache = db.parent / "cadence_resume_cache.json"
    cache.write_text(json.dumps({
        "ts": "2026-05-19T22:00:00Z",
        "session_id": "x",
        "policy": "p2",
        "body": "",
    }))

    out, _ = _run_ups(_ups_payload("new-sess-6", "hi", tmp_path))
    assert "<cadence-resume" not in out


def test_cache_non_dict_no_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache file is a JSON array (not a dict) → no injection."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    cache = db.parent / "cadence_resume_cache.json"
    cache.write_text(json.dumps(["not", "a", "dict"]))

    out, _ = _run_ups(_ups_payload("new-sess-7", "hi", tmp_path))
    assert "<cadence-resume" not in out


def test_inject_only_on_first_prompt_of_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same session_id over two UPS calls — injects on first, not second."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_cache(db, body="<single-shot synthesis>")

    out1, err1 = _run_ups(_ups_payload("same-sess-8", "hi", tmp_path))
    assert "single-shot synthesis" in out1
    assert "cadence-resume injection" in err1

    out2, err2 = _run_ups(_ups_payload("same-sess-8", "follow-up", tmp_path))
    assert "<cadence-resume" not in out2
    assert "cadence-resume injection" not in err2


def test_different_sessions_each_get_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache not deleted on read → second new session also resumes."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_cache(db, body="<shared synthesis>")

    out1, _ = _run_ups(_ups_payload("sess-A", "hi A", tmp_path))
    out2, _ = _run_ups(_ups_payload("sess-B", "hi B", tmp_path))
    assert "shared synthesis" in out1
    assert "shared synthesis" in out2


def test_in_memory_db_no_inject_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:memory: DB → cache_path is None → clean no-op."""
    monkeypatch.setenv("AELFRICE_DB", ":memory:")

    out, err = _run_ups(_ups_payload("mem-sess", "hi", tmp_path))
    assert "<cadence-resume" not in out
    # Should NOT have logged a read failure — None path is a clean skip.
    assert "cadence resume read failed" not in err


def test_p1_policy_cache_also_injects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both P1 and P2 caches are eligible for resume injection."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_cache(db, body="<P1 synthesis>", policy="p1_every_k_turns")

    out, _ = _run_ups(_ups_payload("sess-p1cache", "hi", tmp_path))
    assert "<cadence-resume" in out
    assert "policy='p1_every_k_turns'" in out
    assert "P1 synthesis" in out


def test_session_id_short_prefix_in_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper truncates prior session_id to first 8 chars."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_cache(db, body="<x>", session_id="abcdefgh-1111-2222-3333")

    out, _ = _run_ups(_ups_payload("sess-trunc", "hi", tmp_path))
    assert "from='abcdefgh'" in out
    assert "from='abcdefgh-'" not in out  # not 9 chars
