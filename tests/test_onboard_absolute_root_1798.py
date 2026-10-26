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
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.classification import start_onboard_session
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


def _session_count(db: Path) -> int:
    store = MemoryStore(str(db))
    try:
        return store.count_onboard_sessions()
    finally:
        store.close()


# `aelf onboard` with and without `--emit-candidates`; the regex
# classifier keeps the second one off the network.
_MODES = pytest.mark.parametrize(
    "flags",
    [("--emit-candidates",), ("--llm-classify=false",)],
    ids=["emit-candidates", "scan"],
)


@pytest.mark.timeout(60)
@_MODES
def test_a_relative_run_from_a_removed_folder_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
) -> None:
    # A relative path can't be resolved once the current folder is
    # gone, as in a removed worktree, so the run stops before it
    # records an onboard session.
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    MemoryStore(str(db)).close()
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()

    buf = io.StringIO()
    code = main(argv=["onboard", ".", *flags], out=buf)

    assert code == 2
    assert buf.getvalue() == ""
    err = capsys.readouterr().err
    assert err == "aelf onboard: the current folder no longer exists\n"
    assert _session_count(db) == 0


@pytest.mark.timeout(60)
def test_start_onboard_session_raises_for_a_removed_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A caller other than the CLI gets the error too, and no session row
    # holds the path as typed.
    db = tmp_path / "aelf.db"
    store = MemoryStore(str(db))
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    try:
        with pytest.raises(FileNotFoundError):
            start_onboard_session(store, Path("."))
        assert store.count_onboard_sessions() == 0
    finally:
        store.close()


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform != "darwin" or os.geteuid() == 0,
    reason="macOS can't give the path of a folder you can't search",
)
@_MODES
def test_a_relative_run_from_a_folder_it_cannot_search_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
) -> None:
    db = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    MemoryStore(str(db)).close()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    monkeypatch.chdir(sealed)
    sealed.chmod(0)
    try:
        buf = io.StringIO()
        code = main(argv=["onboard", ".", *flags], out=buf)
    finally:
        sealed.chmod(0o755)

    assert code == 2
    assert buf.getvalue() == ""
    err = capsys.readouterr().err
    assert err == "aelf onboard: cannot list or search the current folder\n"
    assert _session_count(db) == 0
