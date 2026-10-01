"""#1619: each file gets its own commit date, not another file's.

Every earlier recency test used one file and a one-entry map, so a lookup
that took whatever entry was first was indistinguishable from a lookup by
path. These fixtures use two files with distinct dates, and put the other
file's entry first in the map, so a wrong key gives a wrong date.
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
from aelfrice.scanner import extract_ast, extract_filesystem
from aelfrice.store import MemoryStore

_EARLY = "2001-01-01T00:00:00Z"
_LATE = "2022-05-09T00:00:00Z"


def _path_of(source: str) -> str:
    """`doc:<path>:p0` or `ast:<path>:<kind>...` -> `<path>`."""
    return source.split(":", 2)[1]


def _module(name: str) -> str:
    return f'"""Module {name} documents what this module is responsible for."""\n'


def _paragraph(name: str) -> str:
    return f"Paragraph from {name}, stating a fact about this particular file."


def test_extract_ast_dates_each_file_by_its_own_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_module("a"), encoding="utf-8")
    (tmp_path / "b.py").write_text(_module("b"), encoding="utf-8")
    # The other file's entry comes first, so "first entry" is wrong for a.py.
    recency = {"b.py": _LATE, "a.py": _EARLY}
    got = {_path_of(c.source): c.commit_date for c in extract_ast(tmp_path, recency=recency)}
    assert got == {"a.py": _EARLY, "b.py": _LATE}


def test_extract_filesystem_dates_each_file_by_its_own_path(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(_paragraph("a"), encoding="utf-8")
    (tmp_path / "b.md").write_text(_paragraph("b"), encoding="utf-8")
    recency = {"b.md": _LATE, "a.md": _EARLY}
    got = {
        _path_of(c.source): c.commit_date
        for c in extract_filesystem(tmp_path, recency=recency)
    }
    assert got == {"a.md": _EARLY, "b.md": _LATE}


def _commit(repo: Path, name: str, text: str, date: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t",
           "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "add", name], cwd=repo, check=True, timeout=30)
    subprocess.run([*git, "commit", "-q", "-m", f"add {name}"], cwd=repo,
                   check=True, timeout=30, env=env)


@pytest.mark.timeout(60)
def test_the_handshake_keeps_each_files_date_through_accept(tmp_path: Path) -> None:
    """AC3: the date survives candidates_json into created_at, per file."""
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=30)
    _commit(repo, "a.py", _module("a"), _EARLY)
    _commit(repo, "a.md", _paragraph("a"), _EARLY)
    _commit(repo, "b.py", _module("b"), _LATE)
    _commit(repo, "b.md", _paragraph("b"), _LATE)
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
        files = [s for s in result.sentences if not s.source.startswith("git:")]
        cls = [
            HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
            for s in files
        ]
        accept_classifications(store, result.session_id, cls, now="2099-01-01T00:00:00Z")
        rows = store._conn.execute(  # noqa: SLF001 - read-only probe
            "SELECT content, created_at FROM beliefs"
        ).fetchall()
    finally:
        store.close()
    by_file = {}
    for content, created_at in rows:
        for name in ("a", "b"):
            if f"odule {name} " in content or f"from {name}," in content:
                by_file.setdefault(name, set()).add(created_at)
    assert by_file == {"a": {_EARLY}, "b": {_LATE}}
