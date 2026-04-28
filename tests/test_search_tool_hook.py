"""search-tool hook: dispatch, query extraction, retrieval injection, failure modes.

Mirrors test_commit_ingest_hook.py structure. Acceptance criteria from
docs/search_tool_hook.md numbered in test docstrings.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook_search_tool as hk
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def per_repo_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the hook's `db_path()` resolution to a tmp file."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _payload(
    *,
    pattern: str,
    tool_name: str = "Grep",
    cwd: str | None = None,
) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"pattern": pattern},
        "cwd": cwd or "/tmp",
        "session_id": "smoke",
    }


def _run(payload: dict[str, object]) -> tuple[int, str, str]:
    sin = io.StringIO(json.dumps(payload))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stdout=sout, stderr=serr)
    return rc, sout.getvalue(), serr.getvalue()


_BELIEF_COUNTER = [0]


def _seed_belief(db_path: Path, content: str, *, locked: bool = False) -> str:
    """Insert a belief; return its id."""
    _BELIEF_COUNTER[0] += 1
    bid = f"B{_BELIEF_COUNTER[0]:08x}"
    b = Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-27T00:00:00Z" if locked else None,
        demotion_pressure=0,
        created_at="2026-04-27T00:00:00Z",
        last_retrieved_at=None,
    )
    store = MemoryStore(str(db_path))
    try:
        store.insert_belief(b)
    finally:
        store.close()
    return bid


# --- Acceptance criterion 1: matcher discrimination ----------------------


def test_grep_tool_call_fires_hook(per_repo_db: Path) -> None:
    """AC1: Grep tool calls fire the hook."""
    _seed_belief(per_repo_db, "directive-gate hook is misbehaving")
    rc, out, _ = _run(_payload(pattern="directive-gate", tool_name="Grep"))
    assert rc == 0
    assert "<aelfrice-search" in out
    assert "directive-gate" in out


def test_glob_tool_call_fires_hook(per_repo_db: Path) -> None:
    """AC1: Glob tool calls fire the hook."""
    _seed_belief(per_repo_db, "settings.json was modified by aelf setup")
    rc, out, _ = _run(_payload(pattern="settings.json", tool_name="Glob"))
    assert rc == 0
    assert "<aelfrice-search" in out


def test_other_tool_calls_do_not_fire(per_repo_db: Path) -> None:
    """AC1: Read / Bash / Edit tool calls are ignored."""
    _seed_belief(per_repo_db, "x")
    for tool in ("Read", "Bash", "Edit", "Write", "WebFetch"):
        _, out, _ = _run(_payload(pattern="x", tool_name=tool))
        assert out == "", f"{tool} should not produce output"


# --- Acceptance criterion 2: results emitted on match --------------------


def test_pattern_with_tokens_produces_results(per_repo_db: Path) -> None:
    """AC2: Extractable pattern + matching belief → additionalContext block."""
    bid = _seed_belief(per_repo_db, "alpha decision: use FTS5 for retrieval")
    _, out, _ = _run(_payload(pattern="alpha"))
    payload = json.loads(out)
    body = payload["hookSpecificOutput"]["additionalContext"]
    assert 'query="alpha"' in body
    assert bid[:16] in body or "alpha decision" in body


def test_locked_belief_appears_with_l0_marker(per_repo_db: Path) -> None:
    """AC2: Locked beliefs are tagged [L0]; unlocked are [L1]."""
    _seed_belief(per_repo_db, "locked rule about beta", locked=True)
    _seed_belief(per_repo_db, "regular fact about beta")
    _, out, _ = _run(_payload(pattern="beta"))
    body = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "[L0]" in body
    assert "[L1]" in body


# --- Acceptance criterion 3: empty token sets are no-ops -----------------


def test_pure_glob_pattern_is_noop(per_repo_db: Path) -> None:
    """AC3: Glob patterns with no extractable word tokens skip the hook."""
    _seed_belief(per_repo_db, "x")
    _, out, _ = _run(_payload(pattern="**/*.rs", tool_name="Glob"))
    assert out == ""


def test_short_token_only_is_noop(per_repo_db: Path) -> None:
    """AC3: Patterns with only short tokens (<3 chars) skip."""
    _seed_belief(per_repo_db, "x")
    _, out, _ = _run(_payload(pattern=r"\\b\\d", tool_name="Grep"))
    assert out == ""


def test_empty_pattern_is_noop(per_repo_db: Path) -> None:
    """AC3: Empty pattern skips."""
    _, out, _ = _run(_payload(pattern=""))
    assert out == ""


# --- Acceptance criterion 4: idempotency --------------------------------


def test_repeated_calls_produce_same_output(per_repo_db: Path) -> None:
    """AC4: Same payload twice → identical additionalContext."""
    _seed_belief(per_repo_db, "gamma is the third letter")
    _, out1, _ = _run(_payload(pattern="gamma"))
    _, out2, _ = _run(_payload(pattern="gamma"))
    assert out1 == out2


# --- Acceptance criterion 5: empty store sentinel ------------------------


def test_empty_store_emits_no_match_sentinel(per_repo_db: Path) -> None:
    """AC5: Empty store → 'no matching beliefs' sentinel, not empty stdout."""
    rc, out, _ = _run(_payload(pattern="nonexistent"))
    assert rc == 0
    assert out  # not empty
    body = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "no matching beliefs" in body


def test_missing_db_file_emits_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC5: DB file does not exist → sentinel block, not silence."""
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "absent.db"))
    rc, out, _ = _run(_payload(pattern="anything"))
    assert rc == 0
    body = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "no matching beliefs" in body


# --- Acceptance criterion 7: failure modes are silent --------------------


def test_malformed_json_returns_0_silently() -> None:
    """AC7: Malformed JSON payload → exit 0, no output."""
    sin = io.StringIO("not json {")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_empty_stdin_returns_0_silently() -> None:
    """AC7: Empty stdin → exit 0, no output."""
    sin = io.StringIO("")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_non_dict_payload_returns_0_silently() -> None:
    """AC7: JSON array (not object) → exit 0, no output."""
    sin = io.StringIO("[1, 2, 3]")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


# --- Token extraction unit tests ----------------------------------------


def test_extract_query_picks_alphanumeric_tokens() -> None:
    q = hk._extract_query({"tool_input": {"pattern": "src/**/test_*.py"}})
    assert q is not None
    # 'src' (3 chars), 'test' (4 chars). 'py' filtered (2 chars).
    assert "src" in q
    assert "test" in q
    assert "py" not in q


def test_extract_query_caps_at_token_limit() -> None:
    pattern = "alpha beta gamma delta epsilon zeta eta theta"
    q = hk._extract_query({"tool_input": {"pattern": pattern}})
    assert q is not None
    # First 5 tokens only.
    assert q.count(" OR ") == hk.QUERY_TOKEN_LIMIT - 1


def test_extract_query_returns_none_for_empty_tool_input() -> None:
    assert hk._extract_query({"tool_input": {}}) is None
    assert hk._extract_query({"tool_input": "not a dict"}) is None
    assert hk._extract_query({}) is None


def test_extract_query_returns_none_for_pure_metachars() -> None:
    """Patterns with no token >= MIN_TOKEN_LEN are no-ops."""
    assert hk._extract_query({"tool_input": {"pattern": ".*"}}) is None
    assert hk._extract_query({"tool_input": {"pattern": "**/*.go"}}) is None
    assert hk._extract_query({"tool_input": {"pattern": r"\\b\\d+"}}) is None


# --- Per-line truncation --------------------------------------------------


def test_long_belief_content_is_truncated(per_repo_db: Path) -> None:
    """Each line is capped at PER_LINE_CHAR_CAP to bound injection size."""
    long_content = "alpha " + "x" * 500
    _seed_belief(per_repo_db, long_content)
    _, out, _ = _run(_payload(pattern="alpha"))
    body = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    for line in body.splitlines():
        if line.startswith("[L0]") or line.startswith("[L1]"):
            assert len(line) <= hk.PER_LINE_CHAR_CAP
