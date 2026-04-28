"""commit-ingest hook: dispatch, idempotency, session derivation, failure modes."""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice import hook_commit_ingest as hk
from aelfrice.models import EDGE_DERIVED_FROM, EDGE_SUPPORTS
from aelfrice.store import MemoryStore


# --- Fixtures ------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
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
    return hk.main(stdin=sin, stderr=serr)


# --- Behaviour tests -----------------------------------------------------


def test_no_op_on_non_bash_tool(git_repo: Path, per_repo_db: Path) -> None:
    rc = _drive(_payload(
        command="git commit -m 'x'", stdout="[main abc1234] x",
        tool_name="Read", cwd=str(git_repo),
    ))
    assert rc == 0
    assert not per_repo_db.exists()


def test_no_op_on_non_commit_bash(git_repo: Path, per_repo_db: Path) -> None:
    rc = _drive(_payload(
        command="git status", stdout="On branch main", cwd=str(git_repo),
    ))
    assert rc == 0
    assert not per_repo_db.exists()


def test_no_op_on_failed_commit(git_repo: Path, per_repo_db: Path) -> None:
    rc = _drive(_payload(
        command="git commit -m 'x'", stdout="error", is_error=True,
        cwd=str(git_repo),
    ))
    assert rc == 0
    assert not per_repo_db.exists()


def test_extracts_triple_from_commit_message(
    git_repo: Path, per_repo_db: Path,
) -> None:
    branch, short = _make_commit(
        git_repo, "the new index supports faster queries"
    )
    rc = _drive(_payload(
        command="git commit -m 'the new index supports faster queries'",
        stdout=f"[{branch} {short}] the new index supports faster queries",
        cwd=str(git_repo),
    ))
    assert rc == 0
    store = MemoryStore(str(per_repo_db))
    try:
        edges = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM edges WHERE type = ?", (EDGE_SUPPORTS,)
        ).fetchall()
        assert len(edges) == 1
        assert edges[0]["anchor_text"]
        beliefs = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT id, content, session_id FROM beliefs"
        ).fetchall()
        assert len(beliefs) == 2
        sids = {b["session_id"] for b in beliefs}
        assert len(sids) == 1
        assert next(iter(sids)) is not None
    finally:
        store.close()


def test_session_id_is_derived_and_stable(
    git_repo: Path, per_repo_db: Path,
) -> None:
    branch, short = _make_commit(
        git_repo, "the new index supports faster queries"
    )
    expected_session = hk._derive_session_id(branch, short)  # pyright: ignore[reportPrivateUsage]
    _drive(_payload(
        command="git commit -m 'msg'",
        stdout=f"[{branch} {short}] msg",
        cwd=str(git_repo),
    ))
    store = MemoryStore(str(per_repo_db))
    try:
        sess = store.get_session(expected_session)
        assert sess is not None
        assert sess.model == "commit-ingest"
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        for row in rows:
            assert row["session_id"] == expected_session
    finally:
        store.close()


def test_idempotent_on_repeated_fire(
    git_repo: Path, per_repo_db: Path,
) -> None:
    branch, short = _make_commit(
        git_repo, "the spec is derived from the prior memo"
    )
    payload = _payload(
        command="git commit -m 'msg'",
        stdout=f"[{branch} {short}] msg",
        cwd=str(git_repo),
    )
    _drive(payload)
    _drive(payload)
    store = MemoryStore(str(per_repo_db))
    try:
        n_edges = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) AS c FROM edges WHERE type = ?",
            (EDGE_DERIVED_FROM,),
        ).fetchone()["c"]
        assert n_edges == 1
    finally:
        store.close()


def test_no_triples_does_not_create_session(
    git_repo: Path, per_repo_db: Path,
) -> None:
    """Commit messages that produce zero triples should not create a
    session row — keeps the sessions table from accumulating empties."""
    branch, short = _make_commit(git_repo, "a single short subject")
    rc = _drive(_payload(
        command="git commit -m 'a single short subject'",
        stdout=f"[{branch} {short}] a single short subject",
        cwd=str(git_repo),
    ))
    assert rc == 0
    if not per_repo_db.exists():
        return  # expected: hook bailed before opening the store
    store = MemoryStore(str(per_repo_db))
    try:
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) AS c FROM sessions"
        ).fetchone()
        assert rows["c"] == 0
    finally:
        store.close()


def test_branch_and_hash_are_parsed_from_stdout() -> None:
    parsed = hk._branch_and_hash_from_stdout(  # pyright: ignore[reportPrivateUsage]
        "[main 7a4b430] feat: ...\n 1 file changed"
    )
    assert parsed == ("main", "7a4b430")


def test_branch_with_slash_parsed() -> None:
    parsed = hk._branch_and_hash_from_stdout(  # pyright: ignore[reportPrivateUsage]
        "[feat/commit-ingest 1234567] message"
    )
    assert parsed == ("feat/commit-ingest", "1234567")


def test_root_commit_prefix_parsed() -> None:
    parsed = hk._branch_and_hash_from_stdout(  # pyright: ignore[reportPrivateUsage]
        "[main (root-commit) abcdef0] initial"
    )
    assert parsed == ("main", "abcdef0")


def test_unparseable_stdout_no_op(git_repo: Path, per_repo_db: Path) -> None:
    rc = _drive(_payload(
        command="git commit -m 'x'", stdout="weird output no brackets",
        cwd=str(git_repo),
    ))
    assert rc == 0
    assert not per_repo_db.exists()


def test_malformed_json_returns_zero(per_repo_db: Path) -> None:
    sin = io.StringIO("{not json")
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stderr=serr)
    assert rc == 0
    assert not per_repo_db.exists()


def test_message_truncation_does_not_explode() -> None:
    """A pathologically long message must not blow the message cap."""
    long = "the index supports queries. " * 1000
    truncated = hk._truncate_for_extraction(long)  # pyright: ignore[reportPrivateUsage]
    assert len(truncated.encode("utf-8")) <= hk.MESSAGE_BYTE_CAP


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
