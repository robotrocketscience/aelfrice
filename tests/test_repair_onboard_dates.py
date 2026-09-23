"""Tests for scripts/repair_onboard_dates.py (#1609, #1611, #1612).

The onboard handshake stamps every belief accepted in one
`accept_classifications` call with that call's single `now` timestamp
(#1609) -- `derivation_worker.py` sets `created_at=inp.ts` for every
derived belief, so a whole onboard session collapses onto one
`created_at` and the `TEMPORAL_NEXT` spine built over it encodes scan
order, not chronology. These tests reproduce the collapse with the real
handshake (`start_onboard_session` / `accept_classifications`) against a
scratch git repository whose commits carry explicit, deliberately mixed
UTC offsets, then check that the repair script re-dates every belief
correctly and rebuilds a spine with no edge out of real-time order.

Two commits are chosen so that raw ISO-8601 *text* order disagrees with
real time order: `2026-01-05T23:00:00-07:00` (06:00 UTC on the 6th) and
`2026-01-06T01:00:00+00:00` (01:00 UTC on the 6th) sort with the first
string ahead of the second, but the first instant is chronologically
*later* -- exactly the #1611 shape. `backfill_temporal_spine` orders a
session by the raw `created_at` string, so a repair that failed to
normalize every date to a fixed-width UTC form before writing it back
would chain these two beliefs backwards, and `--check` would catch it.
That is also this suite's mutation-check target: breaking
`to_utc_canonical`'s UTC conversion reddens
`test_repair_produces_a_spine_with_no_real_time_violation` (and every
other test that checks an exact `created_at` value).

A fourth commit is a merge that adds a file present in neither parent
(`merge_only.md`, staged into the merge before it is committed). `git log
--name-only` without `-m` never lists a merge's files, so
`_build_file_recency_map` misses that path entirely, and only the #1612
per-path `git log -1 -- <path>` fallback dates it.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.models import BELIEF_FACTUAL
from aelfrice.store import MemoryStore

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "repair_onboard_dates.py"

_spec = importlib.util.spec_from_file_location("_repair_onboard_dates", _SCRIPT)
assert _spec and _spec.loader
repair_onboard_dates = importlib.util.module_from_spec(_spec)
# dataclasses' field-type resolution looks the defining module up in
# sys.modules by name; register it there before exec so `@dataclass` on
# SessionReport doesn't crash on a module that exists but isn't registered.
sys.modules[_spec.name] = repair_onboard_dates
_spec.loader.exec_module(repair_onboard_dates)

main = repair_onboard_dates.main
spine_order_violations = repair_onboard_dates.spine_order_violations
detect_onboard_sessions = repair_onboard_dates.detect_onboard_sessions

# The single wall-clock timestamp the (buggy) handshake stamps on every
# belief in the session -- the #1609 collapse this fixture reproduces.
COLLAPSED_TS = "2026-01-15T00:00:00+00:00"

# Deliberately mixed offsets. T_C1 and T_C2 invert under raw text
# comparison relative to real time (see module docstring).
T_C1 = "2026-01-05T23:00:00-07:00"  # -> 2026-01-06T06:00:00 UTC
T_C2 = "2026-01-06T01:00:00+00:00"  # -> 2026-01-06T01:00:00 UTC (earlier!)
T_C3 = "2026-01-07T08:00:00-05:00"  # -> 2026-01-07T13:00:00 UTC
T_C4 = "2026-01-10T12:00:00+00:00"  # merge commit -> 2026-01-10T12:00:00 UTC


def _utc(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def _commit_env(date: str) -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }


def _build_repo(root: Path) -> dict[str, str]:
    """A scratch git repo with mixed-offset commits, one of them a merge
    that introduces a file present in neither parent. Returns the short
    sha of each commit, keyed by its logical name."""
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test Author")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "checkout", "-q", "-b", "main")

    (root / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n"
    )
    (root / "module.py").write_text(
        '"""Top-level module docstring describing the module purpose."""\n\n'
        "def compute_result():\n"
        '    """A top-level function that returns a constant result value."""\n'
        "    return 1\n"
    )
    _git(root, "add", "README.md", "module.py")
    _git(root, "commit", "-q", "-m", "add core files", env=_commit_env(T_C1))
    c1 = _git(root, "rev-parse", "HEAD")

    _git(root, "checkout", "-q", "-b", "feature")
    (root / "side.md").write_text(
        "The feature branch documents a side capability in its own file.\n"
    )
    _git(root, "add", "side.md")
    _git(root, "commit", "-q", "-m", "add feature side doc", env=_commit_env(T_C2))
    c2 = _git(root, "rev-parse", "HEAD")

    _git(root, "checkout", "-q", "main")
    (root / "notes.txt").write_text("ok\n")  # below the paragraph min: no doc candidate
    _git(root, "add", "notes.txt")
    _git(root, "commit", "-q", "-m", "add notes", env=_commit_env(T_C3))
    c3 = _git(root, "rev-parse", "HEAD")

    _git(root, "merge", "-q", "--no-ff", "--no-commit", "feature")
    (root / "merge_only.md").write_text(
        "This file was staged only while finishing the merge commit itself.\n"
    )
    _git(root, "add", "merge_only.md")
    _git(
        root, "commit", "-q", "-m", "merge feature branch", env=_commit_env(T_C4)
    )
    c4 = _git(root, "rev-parse", "HEAD")

    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _onboard(db_path: Path, repo: Path) -> str:
    """Runs the real handshake against `repo` and returns the resulting
    (collapsed-timestamp) onboard session id."""
    store = MemoryStore(str(db_path))
    try:
        started = start_onboard_session(store, repo, now=COLLAPSED_TS)
        assert started.sentences, "fixture repo produced no onboard candidates"
        cls = [
            HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
            for s in started.sentences
        ]
        outcome = accept_classifications(
            store, started.session_id, cls, now=COLLAPSED_TS
        )
        assert outcome.inserted > 0
        return started.session_id
    finally:
        store.close()


def _belief_dates_by_source(db_path: Path, session_id: str) -> dict[str, str]:
    """{ingest_log.source_path: belief.created_at} for one session, read
    directly -- independent of anything `repair_onboard_dates` computes."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT source_path, derived_belief_ids FROM ingest_log "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        out: dict[str, str] = {}
        import json as _json

        for source_path, raw_ids in rows:
            if not raw_ids:
                continue
            for bid in _json.loads(raw_ids):
                row = con.execute(
                    "SELECT created_at FROM beliefs WHERE id = ?", (bid,)
                ).fetchone()
                if row is not None:
                    out[source_path] = row[0]
        return out
    finally:
        con.close()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _build_repo(root)
    return root


@pytest.fixture
def db_path(tmp_path: Path, repo: Path) -> tuple[Path, str]:
    db = tmp_path / "memory.db"
    session_id = _onboard(db, repo)
    return db, session_id


# --- detection ---------------------------------------------------------------


def test_detects_the_collapsed_onboard_session_without_a_session_flag(
    db_path: tuple[Path, str],
) -> None:
    db, session_id = db_path
    assert detect_onboard_sessions(db) == [session_id]


# --- dry run does not touch the real db --------------------------------------


def test_dry_run_leaves_the_real_db_unchanged(db_path: tuple[Path, str], repo: Path) -> None:
    db, session_id = db_path
    before = _belief_dates_by_source(db, session_id)
    rc = main(["--db", str(db), "--repo", str(repo), "--session", session_id])
    assert rc == 0
    after = _belief_dates_by_source(db, session_id)
    assert after == before
    assert set(after.values()) == {COLLAPSED_TS}


# --- correctness of the repair -----------------------------------------------


def test_apply_dates_doc_and_ast_beliefs_correctly_in_utc(
    db_path: tuple[Path, str], repo: Path
) -> None:
    db, session_id = db_path
    rc = main(["--db", str(db), "--repo", str(repo), "--apply"])
    assert rc == 0

    dates = _belief_dates_by_source(db, session_id)
    assert dates["doc:README.md:p0"] == _utc(T_C1)
    assert dates["ast:module.py:module"] == _utc(T_C1)
    assert dates["ast:module.py:func:compute_result"] == _utc(T_C1)
    assert dates["doc:side.md:p0"] == _utc(T_C2)


def test_apply_dates_a_merge_only_file_via_the_fallback(
    db_path: tuple[Path, str], repo: Path
) -> None:
    db, session_id = db_path
    rc = main(["--db", str(db), "--repo", str(repo), "--apply"])
    assert rc == 0
    dates = _belief_dates_by_source(db, session_id)
    assert dates["doc:merge_only.md:p0"] == _utc(T_C4)


def test_apply_dates_git_commit_sourced_beliefs(
    db_path: tuple[Path, str], repo: Path
) -> None:
    db, session_id = db_path
    rc = main(["--db", str(db), "--repo", str(repo), "--apply"])
    assert rc == 0
    dates = _belief_dates_by_source(db, session_id)
    git_sourced = {k: v for k, v in dates.items() if k.startswith("git:commit:")}
    assert git_sourced, "expected at least one git:commit: sourced belief"
    # Every git:commit: date must be a canonical-UTC string (ends '+00:00',
    # microsecond precision) -- distinguishing a real conversion from a
    # pass-through of git's local-offset `%aI` output.
    for date in git_sourced.values():
        assert date.endswith("+00:00")
        assert len(date) == len("2026-01-05T16:15:00.000000+00:00")


# --- the spine: no edge out of real-time order (mutation-check target) ------


def test_repair_produces_a_spine_with_no_real_time_violation(
    db_path: tuple[Path, str], repo: Path
) -> None:
    db, _session_id = db_path
    rc = main(["--db", str(db), "--repo", str(repo), "--apply", "--check"])
    assert rc == 0
    assert spine_order_violations(db) == []


def test_check_alone_reports_pass_on_a_repaired_store(
    db_path: tuple[Path, str], repo: Path
) -> None:
    db, _session_id = db_path
    assert main(["--db", str(db), "--repo", str(repo), "--apply"]) == 0
    assert main(["--db", str(db), "--check"]) == 0


# --- idempotency ---------------------------------------------------------------


def test_a_second_apply_changes_nothing(db_path: tuple[Path, str], repo: Path) -> None:
    db, session_id = db_path
    assert main(["--db", str(db), "--repo", str(repo), "--apply"]) == 0
    after_first = _belief_dates_by_source(db, session_id)
    edges_after_first = _edge_set(db)

    assert main(["--db", str(db), "--repo", str(repo), "--apply"]) == 0
    after_second = _belief_dates_by_source(db, session_id)
    edges_after_second = _edge_set(db)

    assert after_second == after_first
    assert edges_after_second == edges_after_first


def _edge_set(db_path: Path) -> set[tuple[str, str, str]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return set(con.execute("SELECT src, dst, type FROM edges").fetchall())
    finally:
        con.close()


# --- backup -----------------------------------------------------------------


def test_apply_with_backup_copies_the_pre_write_db(
    db_path: tuple[Path, str], repo: Path, tmp_path: Path
) -> None:
    db, session_id = db_path
    backup = tmp_path / "backup.db"
    assert not backup.exists()
    rc = main(
        ["--db", str(db), "--repo", str(repo), "--apply", "--backup", str(backup)]
    )
    assert rc == 0
    assert backup.exists()
    # The backup was taken before the write: it still carries the
    # collapsed onboard timestamp.
    backup_dates = _belief_dates_by_source(backup, session_id)
    assert set(backup_dates.values()) == {COLLAPSED_TS}
