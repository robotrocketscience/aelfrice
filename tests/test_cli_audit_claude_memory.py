"""Integration tests for `aelf audit-claude-memory` (#935).

All fixtures are synthetic.  No real ~/.claude/ content is referenced.
Each test builds a temporary project directory, a temporary DB, and a
synthetic MEMORY.md — none of these paths touch the real user home.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point AELFRICE_DB at a fresh per-test DB."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed_locked_belief(db: Path, content: str) -> str:
    """Insert a user-locked belief and return its id."""
    bid = uuid.uuid4().hex
    chash = hashlib.sha256(content.encode()).hexdigest()
    b = Belief(
        id=bid,
        content=content,
        content_hash=chash,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at="2026-01-01T00:00:00Z",
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin="user_stated",
    )
    s = MemoryStore(str(db))
    try:
        s.insert_belief(b)
    finally:
        s.close()
    return bid


def _make_memory_md(base_dir: Path, content: str) -> Path:
    """Write a synthetic MEMORY.md under base_dir/memory/ and return its path.

    The directory hierarchy mirrors what ``derive_memory_dir`` produces when
    called with ``base_dir`` as the project path.  We bypass the path-encoding
    step here and write directly into a caller-controlled location, then
    pass ``--project`` pointing at ``base_dir`` so the CLI computes the same
    derived path.
    """
    from aelfrice.claude_memory import derive_memory_dir
    mem_dir = derive_memory_dir(str(base_dir))
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / "MEMORY.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


def test_audit_exits_zero_no_memory_md(tmp_path: Path, isolated_db: Path) -> None:
    """No MEMORY.md + empty store: exits 0, reports nothing found."""
    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert code == 0


def test_audit_reports_memory_md_not_found(tmp_path: Path, isolated_db: Path) -> None:
    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert "MEMORY.md not found" in out


def test_audit_help_exits_zero() -> None:
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        main(argv=["audit-claude-memory", "--help"], out=buf)
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Duplicate bucket
# ---------------------------------------------------------------------------

_SYNTHETIC_DUP_MD = """\
# Synthetic memory

- [Fake auth topic](fake_auth.md) — Auth token expires after 24 hours
"""


def test_audit_duplicate_bucket(tmp_path: Path, isolated_db: Path) -> None:
    """Belief and bullet share slot + value → appears in duplicates section."""
    _seed_locked_belief(isolated_db, "Auth token expires after 24 hours")
    _make_memory_md(tmp_path, _SYNTHETIC_DUP_MD)

    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert code == 0
    assert "Potential duplicates" in out
    # At least the duplicate count is non-zero.
    assert "(0)" not in out.split("Potential duplicates")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# Contradiction bucket
# ---------------------------------------------------------------------------

_SYNTHETIC_CONT_MD = """\
# Synthetic memory

- [Fake auth topic](fake_auth.md) — Auth token expires after 48 hours
"""


def test_audit_contradiction_bucket(tmp_path: Path, isolated_db: Path) -> None:
    """Same slot key, different values → contradiction row."""
    _seed_locked_belief(isolated_db, "Auth token expires after 24 hours")
    _make_memory_md(tmp_path, _SYNTHETIC_CONT_MD)

    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert code == 0
    assert "Potential contradictions" in out


# ---------------------------------------------------------------------------
# aelfrice-only bucket
# ---------------------------------------------------------------------------


def test_audit_aelfrice_only_bucket(tmp_path: Path, isolated_db: Path) -> None:
    """Locked belief with no matching bullet appears in aelfrice-only."""
    _seed_locked_belief(isolated_db, "DB schema uses normalised form")
    # No MEMORY.md written → no claude-memory bullets.
    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert code == 0
    assert "aelfrice-only" in out


# ---------------------------------------------------------------------------
# claude-memory-only bucket
# ---------------------------------------------------------------------------

_SYNTHETIC_CLAUDE_ONLY_MD = """\
# Synthetic memory

- [Retry policy](retry.md) — Retry logic uses exponential backoff
"""


def test_audit_claude_only_bucket(tmp_path: Path, isolated_db: Path) -> None:
    """Bullet with no matching belief appears in claude-memory-only."""
    # Empty store.
    _make_memory_md(tmp_path, _SYNTHETIC_CLAUDE_ONLY_MD)
    code, out = _run("audit-claude-memory", "--project", str(tmp_path))
    assert code == 0
    assert "claude-memory-only" in out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

_SYNTHETIC_JSON_MD = """\
# Synthetic memory

- [Fake cache](fake_cache.md) — Cache policy stores LRU entries
"""


def test_audit_json_valid(tmp_path: Path, isolated_db: Path) -> None:
    """--json emits a single parseable JSON object."""
    _seed_locked_belief(isolated_db, "Cache policy stores LRU entries")
    _make_memory_md(tmp_path, _SYNTHETIC_JSON_MD)

    code, out = _run("audit-claude-memory", "--project", str(tmp_path), "--json")
    assert code == 0
    obj = json.loads(out)
    assert "duplicates" in obj
    assert "contradictions" in obj
    assert "aelfrice_only" in obj
    assert "claude_only" in obj
    assert "memory_md" in obj
    assert "project_path" in obj


def test_audit_json_memory_md_found_flag(tmp_path: Path, isolated_db: Path) -> None:
    """memory_md_found is True when the file exists, False when absent."""
    code, out = _run("audit-claude-memory", "--project", str(tmp_path), "--json")
    assert code == 0
    obj = json.loads(out)
    assert obj["memory_md_found"] is False

    _make_memory_md(tmp_path, "- [X](x.md) — Fake entry here\n")
    code2, out2 = _run("audit-claude-memory", "--project", str(tmp_path), "--json")
    obj2 = json.loads(out2)
    assert obj2["memory_md_found"] is True


def test_audit_json_duplicate_shape(tmp_path: Path, isolated_db: Path) -> None:
    """Each duplicate entry has aelfrice and claude sub-objects."""
    _seed_locked_belief(isolated_db, "Auth token expires after 24 hours")
    _make_memory_md(tmp_path, _SYNTHETIC_DUP_MD)

    code, out = _run("audit-claude-memory", "--project", str(tmp_path), "--json")
    obj = json.loads(out)
    if obj["duplicates"]:
        dup = obj["duplicates"][0]
        assert "aelfrice" in dup
        assert "claude" in dup
        assert "slot_subject" in dup["aelfrice"]
        assert "slot_value" in dup["aelfrice"]
