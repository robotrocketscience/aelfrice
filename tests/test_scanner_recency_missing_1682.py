"""#1682: a file with no recency entry gets no date, not another file's.

The #1619 and #1679 fixtures give every file an entry, and the other
recency tests use an empty map. A lookup that falls back to some other
entry when a file has none passed both. These fixtures use a non-empty
map that omits one file, which is what an untracked file in a tracked
repository produces.
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

_COMMITTED = "2001-01-01T00:00:00Z"
_SIBLING = "2010-03-04T00:00:00Z"
_START = "2098-01-01T00:00:00Z"
_ACCEPT = "2099-01-01T00:00:00Z"


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


def test_extract_ast_leaves_a_file_missing_from_the_map_undated(
    tmp_path: Path,
) -> None:
    # p/n.py is a dated sibling in the same directory, the common real
    # case: a new untracked file next to tracked ones.
    _write(tmp_path, "p/m.py", _module("p"))
    _write(tmp_path, "p/n.py", _module("n"))
    _write(tmp_path, "q/m.py", _module("q"))
    recency = {"p/n.py": _SIBLING, "q/m.py": _COMMITTED}
    got = {_path_of(c.source): c.commit_date for c in extract_ast(tmp_path, recency=recency)}
    assert got == {"p/m.py": None, "p/n.py": _SIBLING, "q/m.py": _COMMITTED}


def test_extract_filesystem_leaves_a_file_missing_from_the_map_undated(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "p/m.md", _paragraph("p"))
    _write(tmp_path, "p/n.md", _paragraph("n"))
    _write(tmp_path, "q/m.md", _paragraph("q"))
    recency = {"p/n.md": _SIBLING, "q/m.md": _COMMITTED}
    got = {
        _path_of(c.source): c.commit_date
        for c in extract_filesystem(tmp_path, recency=recency)
    }
    assert got == {"p/m.md": None, "p/n.md": _SIBLING, "q/m.md": _COMMITTED}


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    base = ["git", "-c", "user.email=t@t", "-c", "user.name=t",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*base, *args], cwd=repo, check=True, timeout=30, env=env)


@pytest.mark.timeout(60)
def test_the_handshake_dates_an_untracked_file_at_accept_time(
    tmp_path: Path,
) -> None:
    """AC3: an untracked file's beliefs take the accept `now` as created_at.

    The untracked files have no recency entry, so their candidates carry
    no commit date, and accept_classifications falls back to the `now`
    it is given. The start and accept times differ so the test shows
    which one is used.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "q/m.py", _module("q"))
    _write(repo, "q/m.md", _paragraph("q"))
    env = {**os.environ, "GIT_AUTHOR_DATE": _COMMITTED, "GIT_COMMITTER_DATE": _COMMITTED}
    _git(repo, "add", "q/m.py", "q/m.md")
    _git(repo, "commit", "-q", "-m", "add q", env=env)
    _write(repo, "p/m.py", _module("p"))
    _write(repo, "p/m.md", _paragraph("p"))
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        result = start_onboard_session(store, repo, now=_START)
        files = [s for s in result.sentences if not s.source.startswith("git:")]
        assert {s.source.split(":", 1)[0] for s in files} == {"ast", "doc"}
        cls = [
            HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
            for s in files
        ]
        accept_classifications(store, result.session_id, cls, now=_ACCEPT)
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
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
    assert got == {
        ("ast", "p"): {_ACCEPT},
        ("doc", "p"): {_ACCEPT},
        ("ast", "q"): {_COMMITTED},
        ("doc", "q"): {_COMMITTED},
    }
