"""#1798: an onboard session records its folder as a resolved absolute path.

`aelf onboard --emit-candidates .` used to store `.` in
`onboard_sessions.repo_path`, which names no folder once the shell that
typed it is gone, so `aelf store split` couldn't place the session's
beliefs. The row now holds the resolved folder, and the scanner still
reads the tree the typed path names.
"""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.store import MemoryStore


def _git(repo: Path, *args: str) -> None:
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
        capture_output=True,
        timeout=30,
    )


def _repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "add the readme")
    return repo


def _emit(path: str) -> dict[str, object]:
    buf = io.StringIO()
    code = main(argv=["onboard", path, "--emit-candidates"], out=buf)
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert isinstance(payload, dict)
    return payload


def _stored_root(db: Path, session_id: str) -> str:
    store = MemoryStore(str(db))
    try:
        session = store.get_onboard_session(session_id)
    finally:
        store.close()
    assert session is not None
    return session.repo_path


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("cwd_name", "typed"),
    [
        # `/aelf:onboard .` from the repository folder.
        ("repo", "."),
        # A path through a symlink stores the folder it points to.
        (".", "link"),
    ],
)
def test_emit_candidates_records_the_resolved_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cwd_name: str,
    typed: str,
) -> None:
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    repo = _repo(tmp_path)
    (tmp_path / "link").symlink_to(repo, target_is_directory=True)
    monkeypatch.chdir(tmp_path / cwd_name)

    payload = _emit(typed)

    stored = _stored_root(db, str(payload["session_id"]))
    assert Path(stored).is_absolute()
    assert stored == str(repo.resolve())


@pytest.mark.timeout(60)
def test_relative_run_scans_the_same_tree_as_an_absolute_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "aelf.db"))
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)

    relative = _emit(".")["sentences"]
    absolute = _emit(str(repo.resolve()))["sentences"]

    assert relative
    assert relative == absolute
