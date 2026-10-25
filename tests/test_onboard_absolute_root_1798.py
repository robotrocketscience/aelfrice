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

from aelfrice import store_split
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


@pytest.mark.timeout(60)
def test_relative_run_from_a_removed_folder_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A relative path can't be resolved once the current folder is
    # gone, as in a removed worktree. The scan finds nothing there, and
    # the run still exits 0 and records the path as typed.
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()

    payload = _emit(".")

    assert payload["sentences"] == []
    assert _stored_root(db, str(payload["session_id"])) == "."


_REPO = Path(__file__).resolve().parents[1]


def test_the_docs_name_a_removed_folder_run_as_a_relative_root() -> None:
    # The removed-folder run above still records `.`, so no text may say
    # that only a session recorded before #1798 holds a relative root.
    config = (_REPO / "docs" / "user" / "CONFIG.md").read_text(encoding="utf-8")
    assert "A session recorded before that change can hold a relative" not in config
    assert "and in a run from a folder that was removed" in config
    fragment = (
        _REPO / "CHANGELOG" / "unreleased" / "1796-store-split.md"
    ).read_text(encoding="utf-8")
    assert "or when the run started from a removed folder" in fragment
    doc = store_split._onboard_roots.__doc__  # pyright: ignore[reportPrivateUsage]
    assert doc is not None
    assert "or from a removed current folder" in doc
