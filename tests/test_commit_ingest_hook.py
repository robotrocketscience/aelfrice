"""commit-ingest hook: the PostToolUse path opens no store, and setup wiring."""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice import hook_commit_ingest as hk


# --- Fixtures ------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
            timeout=30,
)
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {r.stderr}")
    return r.stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README").write_text("seed\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def per_repo_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the hook's `db_path()` resolution to a tmp file."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _make_commit(repo: Path, message: str) -> tuple[str, str]:
    """Make a commit with `message`, return (branch, short_hash)."""
    msg_file = repo / ".commit-msg"
    msg_file.write_text(message)
    (repo / "x").write_text(f"file for {message[:40]}", encoding="utf-8")
    _git(repo, "add", "x")
    out = _git(repo, "commit", "-q", "-F", str(msg_file))
    msg_file.unlink()
    # The -q flag suppresses bracket prefix; re-run a non-quiet show via log.
    short = _git(repo, "rev-parse", "--short", "HEAD").strip()
    branch = _git(repo, "symbolic-ref", "--short", "HEAD").strip()
    return branch, short


def _payload(
    *,
    command: str,
    stdout: str,
    cwd: str | None = None,
    is_error: bool = False,
    interrupted: bool = False,
    tool_name: str = "Bash",
) -> dict[str, object]:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": command},
        "tool_response": {
            "stdout": stdout,
            "stderr": "",
            "isError": is_error,
            "interrupted": interrupted,
        },
        "cwd": cwd,
    }


def _drive(payload: dict[str, object]) -> int:
    sin = io.StringIO(json.dumps(payload))
    serr = io.StringIO()
    return hk.main(stdin=sin, stderr=serr, argv=[])


# --- Behaviour tests -----------------------------------------------------
#
# The PostToolUse path stores nothing (test_commit_hook_writes_nothing pins
# every table and file). These tests keep the payload shapes the older
# version ingested or skipped, and check none of them opens a store.


@pytest.mark.parametrize("payload", [
    _payload(command="git commit -m 'x'", stdout="[main abc1234] x", tool_name="Read"),
    _payload(command="git status", stdout="On branch main"),
    _payload(command="git commit -m 'x'", stdout="error", is_error=True),
    _payload(command="git commit -m 'x'", stdout="weird output no brackets"),
], ids=["non_bash_tool", "non_commit_bash", "failed_commit", "unparseable_stdout"])
def test_a_payload_the_old_hook_skipped_still_opens_no_store(
    git_repo: Path, per_repo_db: Path, payload: dict[str, object],
) -> None:
    payload = {**payload, "cwd": str(git_repo)}
    assert _drive(payload) == 0
    assert not per_repo_db.exists()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("message", [
    "the new index is supported by faster queries",
    "the spec is derived from the prior memo",
    "a single short subject",
])
def test_a_real_commit_payload_opens_no_store(
    git_repo: Path, per_repo_db: Path, message: str,
) -> None:
    """The older version stored the triple, a session, and log rows here."""
    branch, short = _make_commit(git_repo, message)
    payload = _payload(
        command=f"git commit -m '{message}'",
        stdout=f"[{branch} {short}] {message}",
        cwd=str(git_repo),
    )
    assert _drive(payload) == 0
    assert _drive(payload) == 0
    assert not per_repo_db.exists()


@pytest.fixture
def no_store_env(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """No `AELFRICE_DB`, so a store would resolve to the repository's own.

    Returns the `AELFRICE_USER_DB` path, which doesn't exist yet.
    """
    from aelfrice import db_paths

    home = tmp_path / "home"
    home.mkdir()
    user_db = home / ".aelfrice" / "user" / "memory.db"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user_db))
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.chdir(git_repo)
    db_paths.clear_git_common_dir_cache()
    return user_db


def _real_commit_payload(repo: Path) -> dict[str, object]:
    message = "the new index is supported by faster queries"
    branch, short = _make_commit(repo, message)
    return _payload(
        command=f"git commit -m '{message}'",
        stdout=f"[{branch} {short}] {message}",
        cwd=str(repo),
    )


@pytest.mark.timeout(30)
def test_the_payload_path_creates_no_repo_store(
    git_repo: Path, tmp_path: Path, no_store_env: Path,
) -> None:
    assert _drive(_real_commit_payload(git_repo)) == 0
    assert not (git_repo / ".git" / "aelfrice" / "memory.db").exists()
    assert sorted(tmp_path.rglob("*.db")) == []


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entry", ["git_hook", "tool_hook"])
def test_neither_entry_point_creates_the_user_store(
    git_repo: Path, tmp_path: Path, no_store_env: Path, entry: str,
) -> None:
    if entry == "git_hook":
        _make_commit(git_repo, "the spec is derived from the prior memo")
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=git_repo, capture_output=True,
            text=True, check=True, timeout=30,
        ).stdout.strip()
        rc = hk.main(stdin=io.StringIO(""), stderr=io.StringIO(),
                     argv=[hk.GIT_HOOK_FLAG, rev])
    else:
        rc = _drive(_real_commit_payload(git_repo))
    assert rc == 0
    assert not no_store_env.exists()
    assert sorted(tmp_path.rglob("*.db")) == []


def test_malformed_json_returns_zero(per_repo_db: Path) -> None:
    sin = io.StringIO("{not json")
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stderr=serr)
    assert rc == 0
    assert not per_repo_db.exists()


# --- Setup wiring tests --------------------------------------------------


def test_install_uninstall_idempotent(tmp_path: Path) -> None:
    from aelfrice.setup import (
        install_commit_ingest_hook,
        uninstall_commit_ingest_hook,
    )

    settings = tmp_path / "settings.json"
    r1 = install_commit_ingest_hook(settings, command="aelf-commit-ingest")
    assert r1.installed and not r1.already_present
    r2 = install_commit_ingest_hook(settings, command="aelf-commit-ingest")
    assert not r2.installed and r2.already_present

    data = json.loads(settings.read_text(encoding="utf-8"))
    entries = data["hooks"]["PostToolUse"]
    assert len(entries) == 1
    assert entries[0]["matcher"] == "Bash"

    u1 = uninstall_commit_ingest_hook(
        settings, command_basename="aelf-commit-ingest",
    )
    assert u1.removed == 1
    u2 = uninstall_commit_ingest_hook(
        settings, command_basename="aelf-commit-ingest",
    )
    assert u2.removed == 0


def test_install_does_not_disturb_other_post_tool_use_entries(tmp_path: Path) -> None:
    from aelfrice.setup import (
        install_commit_ingest_hook,
        uninstall_commit_ingest_hook,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {"matcher": "Bash", "hooks": [
                    {"type": "command", "command": "/path/to/some-other-hook"}
                ]},
            ],
        },
    }), encoding="utf-8")
    install_commit_ingest_hook(settings, command="aelf-commit-ingest")
    data = json.loads(settings.read_text(encoding="utf-8"))
    entries = data["hooks"]["PostToolUse"]
    assert len(entries) == 2

    uninstall_commit_ingest_hook(
        settings, command_basename="aelf-commit-ingest",
    )
    data = json.loads(settings.read_text(encoding="utf-8"))
    entries = data["hooks"]["PostToolUse"]
    assert len(entries) == 1
    cmd = entries[0]["hooks"][0]["command"]
    assert cmd == "/path/to/some-other-hook"
