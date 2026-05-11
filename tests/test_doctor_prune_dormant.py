"""Tests for `aelf doctor --prune-dormant` (#594).

Covers the pure scanner (`_check_dormant_dbs`) and the CLI flow
(`_cmd_doctor_prune_dormant`).
"""
from __future__ import annotations

import argparse
import io
import os
import sqlite3
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(path: Path, *, with_origin: bool, beliefs: int) -> None:
    """Create a memory.db with N rows; schema chosen by `with_origin`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    if with_origin:
        con.execute(
            "CREATE TABLE beliefs "
            "(id INTEGER PRIMARY KEY, content TEXT, type TEXT, origin TEXT)"
        )
        for i in range(beliefs):
            con.execute(
                "INSERT INTO beliefs (content, type, origin) VALUES (?, ?, ?)",
                (f"b{i}", "fact", "agent_remembered"),
            )
    else:
        con.execute(
            "CREATE TABLE beliefs (id INTEGER PRIMARY KEY, content TEXT)"
        )
        for i in range(beliefs):
            con.execute("INSERT INTO beliefs (content) VALUES (?)", (f"b{i}",))
    con.commit()
    con.close()


def _set_idle(path: Path, days: int) -> None:
    """Backdate mtime so the file appears idle for `days` days."""
    target = time.time() - days * 86400
    os.utime(path, (target, target))


# ---------------------------------------------------------------------------
# Scanner — _check_dormant_dbs
# ---------------------------------------------------------------------------

def test_scanner_returns_empty_for_missing_root(tmp_path: Path) -> None:
    """Non-existent projects dir → no crash, empty list."""
    from aelfrice.doctor import _check_dormant_dbs

    results = _check_dormant_dbs(
        projects_dir=tmp_path / "no-such-dir", idle_days=30,
    )
    assert results == []


def test_scanner_skips_recent_dbs(tmp_path: Path) -> None:
    """A DB modified today must not appear in results."""
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    fresh = projects_dir / "fresh1234abcd" / "memory.db"
    _make_db(fresh, with_origin=True, beliefs=3)
    # mtime is 'now' from creation; no need to set explicitly.

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    assert results == []


def test_scanner_flags_dormant_db(tmp_path: Path) -> None:
    """A DB idle past the threshold is flagged with row_count + size."""
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    dormant = projects_dir / "dormant5678efab" / "memory.db"
    _make_db(dormant, with_origin=True, beliefs=7)
    _set_idle(dormant, days=45)

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    assert len(results) == 1
    entry = results[0]
    assert entry.path == dormant
    assert entry.row_count == 7
    assert entry.idle_days >= 45
    assert entry.size_bytes > 0


def test_scanner_flags_legacy_and_modern_alike(tmp_path: Path) -> None:
    """Dormancy is schema-agnostic: both legacy + modern DBs surface."""
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    legacy = projects_dir / "aaaaaaaaaaaa" / "memory.db"
    modern = projects_dir / "bbbbbbbbbbbb" / "memory.db"
    _make_db(legacy, with_origin=False, beliefs=2)
    _make_db(modern, with_origin=True, beliefs=4)
    _set_idle(legacy, days=60)
    _set_idle(modern, days=60)

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    paths = sorted(r.path for r in results)
    assert paths == sorted([legacy, modern])


def test_scanner_threshold_boundary(tmp_path: Path) -> None:
    """idle_days=N excludes age=N-1 days and includes age=N+1 days."""
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    young = projects_dir / "ccccccccccc1" / "memory.db"
    old = projects_dir / "ccccccccccc2" / "memory.db"
    _make_db(young, with_origin=True, beliefs=1)
    _make_db(old, with_origin=True, beliefs=1)
    _set_idle(young, days=29)
    _set_idle(old, days=31)

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    paths = [r.path for r in results]
    assert young not in paths
    assert old in paths


def test_scanner_includes_empty_dbs(tmp_path: Path) -> None:
    """An idle DB with zero beliefs is still pruneable; row_count=0."""
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    empty = projects_dir / "empty1234567" / "memory.db"
    _make_db(empty, with_origin=True, beliefs=0)
    _set_idle(empty, days=45)

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    assert len(results) == 1
    assert results[0].row_count == 0


def test_scanner_handles_missing_beliefs_table(tmp_path: Path) -> None:
    """A DB file with no `beliefs` table is still flagged when dormant.

    Schema unreadable → row_count=0; the file itself is pruneable.
    """
    from aelfrice.doctor import _check_dormant_dbs

    projects_dir = tmp_path / "projects"
    no_table = projects_dir / "notable12345" / "memory.db"
    no_table.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(no_table))
    con.execute("CREATE TABLE other (x INTEGER)")
    con.commit()
    con.close()
    _set_idle(no_table, days=45)

    results = _check_dormant_dbs(projects_dir=projects_dir, idle_days=30)
    assert len(results) == 1
    assert results[0].row_count == 0


# ---------------------------------------------------------------------------
# CLI — _cmd_doctor_prune_dormant
# ---------------------------------------------------------------------------

def _run_cmd(
    *,
    apply: bool,
    idle_days: int | None,
    projects_dir: Path | None,
    monkeypatch: pytest.MonkeyPatch,
    answers: list[str] | None = None,
) -> tuple[int, str]:
    """Invoke `_cmd_doctor_prune_dormant` and capture stdout + exit code."""
    from aelfrice.cli import _cmd_doctor_prune_dormant

    args = argparse.Namespace(
        prune_dormant=True,
        apply=apply,
        idle_days=idle_days,
        projects_dir=str(projects_dir) if projects_dir else None,
    )
    out = io.StringIO()
    if answers is not None:
        replies = iter(answers)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    rc = _cmd_doctor_prune_dormant(args, out)
    return rc, out.getvalue()


def test_cmd_dry_run_lists_dormant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --apply, the command lists dormant DBs and does not delete."""
    projects_dir = tmp_path / "projects"
    db = projects_dir / "abc123def456" / "memory.db"
    _make_db(db, with_origin=True, beliefs=5)
    _set_idle(db, days=50)

    rc, output = _run_cmd(
        apply=False, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch,
    )
    assert rc == 0
    assert "found 1 dormant per-project DB" in output
    assert str(db) in output
    assert "dry-run only" in output
    assert "--apply" in output
    assert db.exists(), "dry-run must not delete the DB"


def test_cmd_apply_y_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --apply and 'y' answer, the DB is deleted."""
    projects_dir = tmp_path / "projects"
    db = projects_dir / "deletemmenow1" / "memory.db"
    _make_db(db, with_origin=True, beliefs=3)
    _set_idle(db, days=45)

    rc, output = _run_cmd(
        apply=True, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch, answers=["y"],
    )
    assert rc == 0
    assert "deleted" in output
    assert "1 deleted, 0 kept" in output
    assert not db.exists(), "y answer must delete the DB"


def test_cmd_apply_n_preserves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --apply and 'n' answer, the DB is preserved."""
    projects_dir = tmp_path / "projects"
    db = projects_dir / "keepthisdb12" / "memory.db"
    _make_db(db, with_origin=True, beliefs=3)
    _set_idle(db, days=45)

    rc, output = _run_cmd(
        apply=True, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch, answers=["n"],
    )
    assert rc == 0
    assert "kept" in output
    assert "0 deleted, 1 kept" in output
    assert db.exists(), "n answer must preserve the DB"


def test_cmd_apply_empty_answer_preserves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --apply and an empty (just-Enter) answer, the DB is preserved.

    The default at the [y/N] prompt is N — anything other than y/yes
    must keep the file. Locks the 'never silent delete' constraint.
    """
    projects_dir = tmp_path / "projects"
    db = projects_dir / "defaultnodb1" / "memory.db"
    _make_db(db, with_origin=True, beliefs=3)
    _set_idle(db, days=45)

    rc, output = _run_cmd(
        apply=True, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch, answers=[""],
    )
    assert rc == 0
    assert "0 deleted, 1 kept" in output
    assert db.exists()


def test_cmd_apply_mixed_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-DB y/N: some delete, some preserve in one run."""
    projects_dir = tmp_path / "projects"
    db_yes = projects_dir / "deletea11111" / "memory.db"
    db_no = projects_dir / "keepb2222222" / "memory.db"
    _make_db(db_yes, with_origin=True, beliefs=3)
    _make_db(db_no, with_origin=True, beliefs=3)
    _set_idle(db_yes, days=45)
    _set_idle(db_no, days=45)

    # _check_dormant_dbs sorts by path, so db_no (keepb…) runs after db_yes (deletea…).
    # alphabetical: 'deletea11111' < 'keepb2222222'
    rc, output = _run_cmd(
        apply=True, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch, answers=["y", "n"],
    )
    assert rc == 0
    assert "1 deleted, 1 kept" in output
    assert not db_yes.exists()
    assert db_no.exists()


def test_cmd_no_dormant_dbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When nothing qualifies, exit 0 and report empty cleanly."""
    projects_dir = tmp_path / "projects"
    fresh = projects_dir / "freshdb12345" / "memory.db"
    _make_db(fresh, with_origin=True, beliefs=2)
    # No backdating — fresh stays under the threshold.

    rc, output = _run_cmd(
        apply=True, idle_days=30, projects_dir=projects_dir,
        monkeypatch=monkeypatch, answers=[],
    )
    assert rc == 0
    assert "no per-project DBs idle for >= 30 days" in output
    assert fresh.exists()


def test_cmd_negative_idle_days_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation: idle_days < 0 returns exit 2 without scanning."""
    rc, _output = _run_cmd(
        apply=False, idle_days=-1, projects_dir=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert rc == 2


def test_cmd_eof_on_input_preserves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user pipes /dev/null (EOFError on input), DBs are preserved."""
    from aelfrice.cli import _cmd_doctor_prune_dormant

    projects_dir = tmp_path / "projects"
    db = projects_dir / "eoftestdb123" / "memory.db"
    _make_db(db, with_origin=True, beliefs=3)
    _set_idle(db, days=45)

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    args = argparse.Namespace(
        prune_dormant=True, apply=True, idle_days=30,
        projects_dir=str(projects_dir),
    )
    out = io.StringIO()
    rc = _cmd_doctor_prune_dormant(args, out)
    assert rc == 0
    assert db.exists()
    assert "0 deleted, 1 kept" in out.getvalue()
