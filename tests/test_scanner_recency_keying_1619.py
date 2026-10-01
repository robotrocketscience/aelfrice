"""#1619, #1679: each file gets its own commit date, not another file's.

Every earlier recency test used one file and a one-entry map, so a lookup
that took whatever entry was first was indistinguishable from a lookup by
path. These fixtures use two files with distinct dates, and put the other
file's entry first in the map, so a wrong key gives a wrong date.

The two files share a basename in different directories (#1679). At the
repo root `path.name == rel`, so a lookup keyed on the basename would pass
a root-only fixture while every nested file in a real repo lost its date.
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


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_ast_dates_each_file_by_its_own_path(tmp_path: Path) -> None:
    _write(tmp_path, "p/m.py", _module("p"))
    _write(tmp_path, "q/m.py", _module("q"))
    # The other file's entry comes first, so "first entry" is wrong for p/m.py.
    recency = {"q/m.py": _LATE, "p/m.py": _EARLY}
    got = {_path_of(c.source): c.commit_date for c in extract_ast(tmp_path, recency=recency)}
    assert got == {"p/m.py": _EARLY, "q/m.py": _LATE}


def test_extract_filesystem_dates_each_file_by_its_own_path(tmp_path: Path) -> None:
    _write(tmp_path, "p/m.md", _paragraph("p"))
    _write(tmp_path, "q/m.md", _paragraph("q"))
    recency = {"q/m.md": _LATE, "p/m.md": _EARLY}
    got = {
        _path_of(c.source): c.commit_date
        for c in extract_filesystem(tmp_path, recency=recency)
    }
    assert got == {"p/m.md": _EARLY, "q/m.md": _LATE}


def _commit(repo: Path, rel: str, text: str, date: str) -> None:
    _write(repo, rel, text)
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "add", rel], cwd=repo, check=True, timeout=30)
    subprocess.run([*git, "commit", "-q", "-m", f"add {rel}"], cwd=repo,
                   check=True, timeout=30, env=env)


@pytest.mark.timeout(60)
def test_the_handshake_keeps_each_files_date_through_accept(tmp_path: Path) -> None:
    """AC3: the date survives candidates_json into created_at, per file.

    All four files carry distinct dates, so a date taken from the other
    directory or from the other kind of file is visible.
    """
    dates = {
        ("ast", "p"): _EARLY,
        ("doc", "p"): "2010-03-04T00:00:00Z",
        ("ast", "q"): "2016-07-08T00:00:00Z",
        ("doc", "q"): _LATE,
    }
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=30)
    _commit(repo, "p/m.py", _module("p"), dates[("ast", "p")])
    _commit(repo, "p/m.md", _paragraph("p"), dates[("doc", "p")])
    _commit(repo, "q/m.py", _module("q"), dates[("ast", "q")])
    _commit(repo, "q/m.md", _paragraph("q"), dates[("doc", "q")])
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
        files = [s for s in result.sentences if not s.source.startswith("git:")]
        assert {s.source.split(":", 1)[0] for s in files} == {"ast", "doc"}
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
    got: dict[tuple[str, str], set[str]] = {}
    for content, created_at in rows:
        for name in ("p", "q"):
            if f"Module {name} " in content:
                got.setdefault(("ast", name), set()).add(created_at)
            if f"Paragraph from {name}," in content:
                got.setdefault(("doc", name), set()).add(created_at)
    assert got == {key: {date} for key, date in dates.items()}
