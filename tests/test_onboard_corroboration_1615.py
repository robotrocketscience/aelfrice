"""#1615: re-reading documents never counts as re-assertion.

Before, every `aelf onboard` run was a new scan session, and nothing
recorded the source path, so re-reading unchanged text added a
corroboration row each run. Measured on this repository, three scans gave
all 12,633 doc beliefs a count of 2. A paragraph at a second path was
also reported as new on every run, because the pre-scan looked it up by
its (source, text) id while the store dedups on content.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    check_onboard_candidates,
    start_onboard_session,
)
from aelfrice.derivation_worker import run_worker
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_TRANSCRIPT,
)
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore

_P = (
    "The widget service retries failed uploads three times with exponential "
    "backoff before giving up."
)


def _corroboration_rows(store: MemoryStore) -> int:
    return store._conn.execute(  # noqa: SLF001 - read-only probe
        "SELECT COUNT(*) FROM belief_corroborations"
    ).fetchone()[0]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs/a.md").write_text(f"# A\n\n{_P}\n", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t",
           "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True, timeout=30)
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True, timeout=30)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=repo, check=True,
                   timeout=30, env=env)
    return repo


def _onboard(store: MemoryStore, repo: Path, day: int) -> tuple[int, int]:
    """One full handshake, accepting everything. Returns (n_new, emitted)."""
    now = f"2026-10-{day:02d}T00:00:00Z"
    n_new = check_onboard_candidates(store, repo).n_new
    result = start_onboard_session(store, repo, now=now)
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    accept_classifications(store, result.session_id, cls, now=now)
    return n_new, len(result.sentences)


@pytest.mark.timeout(60)
def test_the_issue_reproduction_stops_after_the_first_re_onboard(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        assert _onboard(store, repo, 1) == (2, 2)
        # The same paragraph appears at a second path.
        (repo / "docs/b.md").write_text(f"# B\n\n{_P}\n", encoding="utf-8")
        for day in (2, 3, 4):
            assert _onboard(store, repo, day) == (0, 0), f"day {day}"
        assert _corroboration_rows(store) == 0
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_repeated_scans_add_no_corroboration(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        for day in (1, 2, 3):
            scan_repo(store, repo, now=f"2026-10-{day:02d}T00:00:00Z")
        assert _corroboration_rows(store) == 0
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_one_paragraph_at_two_paths_in_one_scan_adds_no_corroboration(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / "docs/b.md").write_text(f"# B\n\n{_P}\n", encoding="utf-8")
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        scan_repo(store, repo, now="2026-10-01T00:00:00Z")
        assert _corroboration_rows(store) == 0
        count = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT COUNT(*) FROM beliefs WHERE content = ?", (_P,)
        ).fetchone()[0]
        assert count == 1
    finally:
        store.close()


# --- the derivation worker ----------------------------------------------


def test_the_worker_does_not_report_a_filesystem_hit_as_corroborated(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        for _ in range(2):
            store.record_ingest(
                source_kind=INGEST_SOURCE_FILESYSTEM, source_path="doc:a.md",
                raw_text=_P,
                raw_meta={"call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST},
            )
        result = run_worker(store)
        assert (result.beliefs_inserted, result.beliefs_corroborated) == (1, 0)
        assert _corroboration_rows(store) == 0
    finally:
        store.close()


def test_a_transcript_row_without_a_call_site_still_corroborates(
    tmp_path: Path,
) -> None:
    """Without the explicit mapping it fell back to filesystem_ingest,
    which since #1615 records nothing."""
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        for session in ("s1", "s2"):
            store.record_ingest(
                source_kind=INGEST_SOURCE_TRANSCRIPT, raw_text=_P,
                session_id=session,
            )
        run_worker(store)
        rows = [
            r[0]
            for r in store._conn.execute(  # noqa: SLF001 - read-only probe
                "SELECT source_type FROM belief_corroborations"
            ).fetchall()
        ]
        assert rows == [CORROBORATION_SOURCE_TRANSCRIPT_INGEST]
    finally:
        store.close()
