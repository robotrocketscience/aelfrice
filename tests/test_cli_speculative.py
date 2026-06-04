"""cli: aelf speculative subcommand tests (#937).

Each test sets AELFRICE_DB to a tmp_path-scoped DB and calls main()
in-process with an io.StringIO so output is capturable without subprocess
overhead. Fixture pattern mirrors test_cli.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets a fresh DB at <tmp>/aelf.db."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed_belief(
    db: Path,
    content: str,
    *,
    origin: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
) -> str:
    """Insert a belief directly via the store and return its id."""
    bid = uuid.uuid4().hex
    chash = hashlib.sha256(content.encode()).hexdigest()
    b = Belief(
        id=bid,
        content=content,
        content_hash=chash,
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )
    s = MemoryStore(str(db))
    try:
        s.insert_belief(b)
    finally:
        s.close()
    return bid


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------


def test_speculative_empty_store_exits_zero(isolated_db: Path) -> None:
    code, out = _run("speculative")
    assert code == 0
    assert "no speculative beliefs" in out


def test_speculative_empty_json_exits_zero(isolated_db: Path) -> None:
    """--json with no rows: exits 0 with the empty message (no JSONL)."""
    code, out = _run("speculative", "--json")
    assert code == 0
    assert "no speculative beliefs" in out


# ---------------------------------------------------------------------------
# User-locked rows excluded
# ---------------------------------------------------------------------------


def test_speculative_excludes_user_locked_rows(isolated_db: Path) -> None:
    """Rows with lock_level='user' must not appear in speculative output."""
    _seed_belief(isolated_db, "user-locked belief", origin="user_stated",
                 lock_level=LOCK_USER)
    code, out = _run("speculative")
    assert code == 0
    assert "no speculative beliefs" in out


def test_speculative_includes_non_locked_rows(isolated_db: Path) -> None:
    """Non-locked rows appear in output."""
    bid = _seed_belief(isolated_db, "agent inferred fact", origin="agent_inferred")
    code, out = _run("speculative")
    assert code == 0
    assert bid in out
    assert "agent inferred fact" in out


def test_speculative_mixed_origins_excludes_locked(isolated_db: Path) -> None:
    """Mixed DB: only non-locked rows surface."""
    locked_bid = _seed_belief(isolated_db, "locked content", origin="user_stated",
                               lock_level=LOCK_USER)
    free_bid = _seed_belief(isolated_db, "free content", origin="agent_inferred")
    code, out = _run("speculative")
    assert code == 0
    assert free_bid in out
    assert locked_bid not in out


# ---------------------------------------------------------------------------
# --origin filter
# ---------------------------------------------------------------------------


def test_speculative_origin_filter(isolated_db: Path) -> None:
    """--origin restricts output to that tag only."""
    bid_a = _seed_belief(isolated_db, "agent fact", origin="agent_inferred")
    bid_s = _seed_belief(isolated_db, "speculative fact", origin="speculative")
    code, out = _run("speculative", "--origin", "agent_inferred")
    assert code == 0
    assert bid_a in out
    assert bid_s not in out


def test_speculative_origin_filter_no_match(isolated_db: Path) -> None:
    """--origin with no matching rows: graceful empty output, exit 0."""
    _seed_belief(isolated_db, "some fact", origin="agent_inferred")
    code, out = _run("speculative", "--origin", "speculative")
    assert code == 0
    assert "no speculative beliefs" in out


# ---------------------------------------------------------------------------
# --limit
# ---------------------------------------------------------------------------


def test_speculative_limit_truncates(isolated_db: Path) -> None:
    """--limit N returns at most N rows."""
    for i in range(5):
        _seed_belief(isolated_db, f"fact {i}", origin="agent_inferred",
                     alpha=float(i + 1))
    code, out = _run("speculative", "--limit", "2")
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------


def test_speculative_json_emits_valid_jsonl(isolated_db: Path) -> None:
    """--json emits one valid JSON object per line."""
    _seed_belief(isolated_db, "some fact", origin="agent_inferred",
                 alpha=3.0, beta=1.0)
    code, out = _run("speculative", "--json")
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "id" in obj
    assert "origin" in obj
    assert "alpha" in obj
    assert "beta" in obj
    assert "created_at" in obj
    assert "snippet" in obj
    assert isinstance(obj["alpha"], float)
    assert isinstance(obj["beta"], float)
    assert obj["origin"] == "agent_inferred"


def test_speculative_json_alpha_is_numeric(isolated_db: Path) -> None:
    """JSONL alpha/beta fields are numeric, not strings."""
    _seed_belief(isolated_db, "numeric check", origin="unknown",
                 alpha=2.5, beta=1.5)
    _, out = _run("speculative", "--json")
    obj = json.loads(out.strip().splitlines()[0])
    assert obj["alpha"] == pytest.approx(2.5)
    assert obj["beta"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Sort order: alpha descending
# ---------------------------------------------------------------------------


def test_speculative_sorted_alpha_descending(isolated_db: Path) -> None:
    """Rows are returned with highest alpha first."""
    bid_low = _seed_belief(isolated_db, "low alpha", origin="agent_inferred",
                           alpha=1.0, beta=1.0)
    bid_high = _seed_belief(isolated_db, "high alpha", origin="agent_inferred",
                            alpha=5.0, beta=1.0)
    code, out = _run("speculative")
    assert code == 0
    pos_high = out.index(bid_high)
    pos_low = out.index(bid_low)
    assert pos_high < pos_low, "high-alpha row should appear before low-alpha row"


def test_speculative_json_sorted_alpha_descending(isolated_db: Path) -> None:
    """JSONL output preserves alpha-descending order."""
    _seed_belief(isolated_db, "first", origin="agent_inferred", alpha=1.0)
    _seed_belief(isolated_db, "second", origin="agent_inferred", alpha=4.0)
    _seed_belief(isolated_db, "third", origin="agent_inferred", alpha=2.0)
    _, out = _run("speculative", "--json")
    rows = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    alphas = [r["alpha"] for r in rows]
    assert alphas == sorted(alphas, reverse=True)
