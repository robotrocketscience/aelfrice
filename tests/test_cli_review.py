"""Tests for the `aelf review` CLI subcommand (#936).

Mirrors test_cli_core.py shape: isolated DB via monkeypatch, main()
called in-process with io.StringIO for output capture.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _mk_belief(
    bid: str,
    content: str = "belief content",
    created_at: str = "2026-01-01T00:00:00Z",
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
    )


def _seed_store(db_path: Path, *beliefs: Belief) -> None:
    s = MemoryStore(str(db_path))
    for b in beliefs:
        s.insert_belief(b)
    s.close()


# ── --generate ────────────────────────────────────────────────────────────────

def test_review_generate_empty_store_exits_zero(
    tmp_path: Path, isolated_db: Path,
) -> None:
    out_path = tmp_path / "review.md"
    code, output = _run("review", "--generate", "--out", str(out_path))
    assert code == 0
    assert out_path.exists()
    assert "0 candidate(s)" in output


def test_review_generate_writes_file_with_candidates(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="the sky is blue"))
    out_path = tmp_path / "review.md"
    code, output = _run("review", "--generate", "--out", str(out_path))
    assert code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "## aelfrice review" in text
    assert "b1" in text
    assert "[ ] keep" in text
    assert "1 candidate(s)" in output


def test_review_generate_default_path_created(
    tmp_path: Path, isolated_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--generate with no --out writes to .aelfrice/review.md under cwd."""
    monkeypatch.chdir(tmp_path)
    _seed_store(isolated_db, _mk_belief("b1", content="some content"))
    code, output = _run("review", "--generate")
    assert code == 0
    assert (tmp_path / ".aelfrice" / "review.md").exists()


def test_review_generate_excludes_locked_beliefs(
    tmp_path: Path, isolated_db: Path,
) -> None:
    b_open = _mk_belief("open", content="open belief")
    b_locked = _mk_belief("locked", content="locked belief", lock_level=LOCK_USER)
    _seed_store(isolated_db, b_open, b_locked)
    out_path = tmp_path / "review.md"
    code, _ = _run("review", "--generate", "--out", str(out_path))
    assert code == 0
    text = out_path.read_text(encoding="utf-8")
    assert "open" in text
    assert "locked" not in text


# ── --apply ───────────────────────────────────────────────────────────────────

def test_review_apply_missing_file_exits_nonzero(
    tmp_path: Path, isolated_db: Path,
) -> None:
    nonexistent = tmp_path / "no_such_file.md"
    code, _ = _run("review", "--apply", "--out", str(nonexistent))
    assert code != 0


def test_review_apply_keep_updates_last_confirmed_at(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="keep me"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [x] keep   [ ] remove   [ ] lock   | b1 (1d old, 1d cold) — keep me\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file))
    assert code == 0
    assert "1 kept" in output
    s = MemoryStore(str(isolated_db))
    b = s.get_belief("b1")
    s.close()
    assert b is not None
    assert b.last_confirmed_at is not None


def test_review_apply_remove_soft_deletes(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="remove me"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [ ] keep   [x] remove   [ ] lock   | b1 (1d old, 1d cold) — remove me\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file))
    assert code == 0
    assert "1 removed" in output
    s = MemoryStore(str(isolated_db))
    b = s.get_belief("b1")
    s.close()
    assert b is not None
    assert b.valid_to is not None


def test_review_apply_lock_promotes(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="lock me"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [ ] keep   [ ] remove   [x] lock   | b1 (1d old, 1d cold) — lock me\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file))
    assert code == 0
    assert "1 locked" in output
    s = MemoryStore(str(isolated_db))
    b = s.get_belief("b1")
    s.close()
    assert b is not None
    assert b.lock_level == LOCK_USER


def test_review_apply_skip_is_noop(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="skip me"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [ ] keep   [ ] remove   [ ] lock   | b1 (1d old, 1d cold) — skip me\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file))
    assert code == 0
    assert "1 skipped" in output


def test_review_apply_ambiguous_row_exits_nonzero(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="ambiguous"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [x] keep   [x] remove   [ ] lock   | b1 (1d old, 1d cold) — ambiguous\n",
        encoding="utf-8",
    )
    code, _ = _run("review", "--apply", "--out", str(review_file))
    assert code != 0


def test_review_apply_json_output(
    tmp_path: Path, isolated_db: Path,
) -> None:
    _seed_store(isolated_db, _mk_belief("b1", content="keep me"))
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [x] keep   [ ] remove   [ ] lock   | b1 (1d old, 1d cold) — keep me\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file), "--json")
    assert code == 0
    data = json.loads(output)
    assert "kept" in data
    assert "b1" in data["kept"]


def test_review_apply_partial_edit_skips_unchecked(
    tmp_path: Path, isolated_db: Path,
) -> None:
    """Unchecked rows in the apply file are skipped; they need no rewrite."""
    b1 = _mk_belief("b1", content="keep me")
    b2 = _mk_belief("b2", content="leave me alone")
    _seed_store(isolated_db, b1, b2)
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "- [x] keep   [ ] remove   [ ] lock   | b1 (1d old, 1d cold) — keep me\n"
        "- [ ] keep   [ ] remove   [ ] lock   | b2 (1d old, 1d cold) — leave me alone\n",
        encoding="utf-8",
    )
    code, output = _run("review", "--apply", "--out", str(review_file))
    assert code == 0
    assert "1 kept" in output
    assert "1 skipped" in output
    # b2 is untouched
    s = MemoryStore(str(isolated_db))
    b2_got = s.get_belief("b2")
    s.close()
    assert b2_got is not None
    assert b2_got.last_confirmed_at is None
    assert b2_got.valid_to is None
