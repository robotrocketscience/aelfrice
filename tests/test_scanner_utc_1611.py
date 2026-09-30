"""#1611: git author dates are stored in the UTC form every other writer uses.

git's `%aI` keeps the author's local offset. `created_at` is ordered as
text, so a `-08:00` row sorts before a `Z` row that is earlier in real
time, and the temporal spine then links them backwards. The scanner now
rewrites each git date as `YYYY-MM-DDTHH:MM:SSZ` at the point it reads it,
which covers `scan_repo` and the onboard handshake alike.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.models import BELIEF_FACTUAL
from aelfrice.scanner import (
    _build_file_recency_map,  # pyright: ignore[reportPrivateUsage]
    _git_date_to_utc,  # pyright: ignore[reportPrivateUsage]
    extract_git_log,
    scan_repo,
)
from aelfrice.store import MemoryStore

# 10:00 at -08:00 is 18:00 UTC: later in real time than 17:00 UTC, but
# earlier as text. This is the issue's reproduction.
_WEST = "2026-01-01T10:00:00-08:00"
_UTC = "2026-01-01T17:00:00+00:00"


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = dict(os.environ)
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        [
            "git",
            "-c", "user.email=t@t",
            "-c", "user.name=t",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        env=env,
        timeout=30,
    )


def _two_offset_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "ALPHA.md").write_text(
        "alpha paragraph committed at ten in the morning pacific time",
        encoding="utf-8",
    )
    _git(repo, "add", "ALPHA.md")
    _git(repo, "commit", "-q", "-m", "add alpha notes", date=_WEST)
    (repo / "BETA.md").write_text(
        "beta paragraph committed at five in the afternoon utc",
        encoding="utf-8",
    )
    _git(repo, "add", "BETA.md")
    _git(repo, "commit", "-q", "-m", "add beta notes", date=_UTC)
    return repo


# --- the conversion ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_WEST, "2026-01-01T18:00:00Z"),
        (_UTC, "2026-01-01T17:00:00Z"),
        ("2026-01-01T17:00:00Z", "2026-01-01T17:00:00Z"),
        # Crossing midnight changes the date, not only the hour.
        ("2026-01-01T23:30:00-02:00", "2026-01-02T01:30:00Z"),
        ("2026-01-02T00:30:00+05:30", "2026-01-01T19:00:00Z"),
    ],
)
def test_git_date_is_rewritten_in_utc_z_form(raw: str, expected: str) -> None:
    assert _git_date_to_utc(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "2026-01-01-notes.md",  # a date-shaped line that is not a date (#1629)
        "2026-01-01T10:00:00",  # naive: no offset to convert from
        "not a date",
    ],
)
def test_a_value_that_is_not_an_aware_date_is_returned_unchanged(raw: str) -> None:
    assert _git_date_to_utc(raw) == raw


# --- the two places git dates are read --------------------------------


@pytest.mark.timeout(30)
def test_recency_map_stores_utc(tmp_path: Path) -> None:
    rec = _build_file_recency_map(_two_offset_repo(tmp_path))
    assert rec == {
        "ALPHA.md": "2026-01-01T18:00:00Z",
        "BETA.md": "2026-01-01T17:00:00Z",
    }


@pytest.mark.timeout(30)
def test_git_log_candidates_carry_utc(tmp_path: Path) -> None:
    dates = {
        c.text: c.commit_date for c in extract_git_log(_two_offset_repo(tmp_path))
    }
    assert dates == {
        "add alpha notes": "2026-01-01T18:00:00Z",
        "add beta notes": "2026-01-01T17:00:00Z",
    }


# --- the issue's symptom: text order equals real order ----------------


@pytest.mark.timeout(60)
def test_scan_repo_text_order_is_real_order(tmp_path: Path) -> None:
    repo = _two_offset_repo(tmp_path)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        scan_repo(store, repo, now="2099-01-01T00:00:00Z")
        rows = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT content, created_at FROM beliefs "
            "WHERE content LIKE '%paragraph committed%' ORDER BY created_at"
        ).fetchall()
    finally:
        store.close()
    assert [r[1] for r in rows] == [
        "2026-01-01T17:00:00Z",
        "2026-01-01T18:00:00Z",
    ]
    # BETA was committed first in real time, so it must sort first.
    assert "beta" in rows[0][0]
    assert "alpha" in rows[1][0]


@pytest.mark.timeout(60)
def test_onboard_handshake_stores_utc(tmp_path: Path) -> None:
    repo = _two_offset_repo(tmp_path)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
        target = [s for s in result.sentences if "alpha paragraph" in s.text]
        assert target, "expected the ALPHA.md paragraph among the candidates"
        cls = [
            HostClassification(
                index=target[0].index, belief_type=BELIEF_FACTUAL, persist=True,
            )
        ]
        accept_classifications(
            store, result.session_id, cls, now="2099-01-01T00:00:00Z",
        )
        row = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT created_at FROM beliefs WHERE content LIKE '%alpha paragraph%'"
        ).fetchone()
    finally:
        store.close()
    assert row is not None
    assert row[0] == "2026-01-01T18:00:00Z"
