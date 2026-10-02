"""commit-ingest hook: dispatch, idempotency, session derivation, failure modes."""
from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path

import pytest

from aelfrice import hook_commit_ingest as hk
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_DERIVED_FROM,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


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


@pytest.mark.timeout(30)
def test_extracts_triple_from_commit_message(
    git_repo: Path, per_repo_db: Path,
) -> None:
    branch, short = _make_commit(
        git_repo, "the new index is supported by faster queries"
    )
    rc = _drive(_payload(
        command="git commit -m 'the new index is supported by faster queries'",
        stdout=f"[{branch} {short}] the new index is supported by faster queries",
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


def _sessions(db: Path) -> list[str]:
    if not db.exists():
        return []
    store = MemoryStore(str(db))
    try:
        return [str(r[0]) for r in store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT id FROM sessions WHERE model = 'commit-ingest' ORDER BY id"
        ).fetchall()]
    finally:
        store.close()


def _corroboration_rows(db: Path) -> int:
    if not db.exists():
        return 0
    store = MemoryStore(str(db))
    try:
        return int(store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) FROM belief_corroborations"
        ).fetchone()[0])
    finally:
        store.close()


def _contents(db: Path) -> set[str]:
    store = MemoryStore(str(db))
    try:
        return {str(r[0]) for r in store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content FROM beliefs"
        ).fetchall()}
    finally:
        store.close()


def _commit_payload(repo: Path, command: str = "git commit -q -F m") -> dict[str, object]:
    return _payload(command=command, stdout="", cwd=str(repo))


_MSG = "the spec is derived from the prior memo"


@pytest.mark.timeout(30)
def test_session_id_is_derived_and_stable(
    git_repo: Path, per_repo_db: Path,
) -> None:
    _make_commit(git_repo, "the new index is supported by faster queries")
    parent = _git(git_repo, "log", "-1", "--format=%P").split()[0]
    author_date = _git(git_repo, "log", "-1", "--format=%aI").strip()
    expected_session = hk._derive_session_id(parent, author_date)  # pyright: ignore[reportPrivateUsage]
    _drive(_commit_payload(git_repo))
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


@pytest.mark.timeout(30)
def test_idempotent_on_repeated_fire(
    git_repo: Path, per_repo_db: Path,
) -> None:
    _make_commit(git_repo, _MSG)
    payload = _commit_payload(git_repo)
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
    assert _corroboration_rows(per_repo_db) == 0


@pytest.mark.timeout(30)
def test_no_triples_does_not_create_session(
    git_repo: Path, per_repo_db: Path,
) -> None:
    """Commit messages that produce zero triples should not create a
    session row — keeps the sessions table from accumulating empties."""
    _make_commit(git_repo, "a single short subject")
    assert _drive(_commit_payload(git_repo)) == 0
    assert _sessions(per_repo_db) == []


# --- #1698: which commits a call made ---------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m x",
        "git add a && git commit -q -m x",
        "git -c user.name=n commit -m x",
        "env X=1 git commit -m x",
        "git --no-pager commit -m x",
    ],
)
def test_commit_commands_pass_the_prefilter(command: str) -> None:
    assert hk._MENTIONS_GIT_COMMIT.search(command)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("command", ["git status", "ls commits", "make test"])
def test_other_commands_fail_the_prefilter(command: str) -> None:
    assert not hk._MENTIONS_GIT_COMMIT.search(command)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.timeout(30)
def test_a_chained_quiet_commit_is_ingested(
    git_repo: Path, per_repo_db: Path,
) -> None:
    """`-q` prints no `[branch hash]` line; the reflog still has it."""
    _make_commit(git_repo, _MSG)
    _drive(_commit_payload(git_repo, "git add x && git commit -q -F m"))
    assert len(_sessions(per_repo_db)) == 1


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "command",
    [
        "git commit -q -m nothing-staged || true",
        'gh pr create --body "run git commit first"',
    ],
)
def test_an_old_commit_is_not_ingested(
    git_repo: Path, per_repo_db: Path, monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A call that made no commit finds no recent reflog entry, so it
    never ingests whatever `HEAD` happens to be."""
    _make_commit(git_repo, _MSG)
    later = time.time() + hk.REFLOG_WINDOW_S + 60
    monkeypatch.setattr(hk, "_now", lambda: later)
    _drive(_commit_payload(git_repo, command))
    assert _sessions(per_repo_db) == []


@pytest.mark.timeout(30)
def test_every_commit_in_one_call_is_ingested(
    git_repo: Path, per_repo_db: Path,
) -> None:
    _make_commit(git_repo, _MSG)
    _make_commit(git_repo, "the new index is supported by faster queries")
    _drive(_commit_payload(git_repo, "git commit -q -m a && git commit -q -m b"))
    assert len(_sessions(per_repo_db)) == 2


@pytest.mark.timeout(30)
@pytest.mark.parametrize("edit", [False, True])
def test_an_amend_adds_no_corroboration(
    git_repo: Path, per_repo_db: Path, edit: bool,
) -> None:
    """An amend keeps the parent and author date, so it reuses the
    session, and a session never corroborates its own beliefs."""
    _make_commit(git_repo, _MSG)
    _drive(_commit_payload(git_repo))
    (git_repo / "x").write_text("amended content", encoding="utf-8")
    _git(git_repo, "add", "x")
    extra = "the cache is supported by the new index"
    if edit:
        _git(git_repo, "commit", "-q", "--amend", "-m", f"{_MSG}\n\n{extra}")
    else:
        _git(git_repo, "commit", "-q", "--amend", "--no-edit")
    _drive(_commit_payload(git_repo, "git commit -q --amend"))
    assert len(_sessions(per_repo_db)) == 1
    assert _corroboration_rows(per_repo_db) == 0
    if edit:
        assert "the cache" in " ".join(_contents(per_repo_db))


@pytest.mark.timeout(30)
def test_a_retry_after_a_crash_adds_no_corroboration(
    git_repo: Path, per_repo_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_commit(git_repo, _MSG)
    real = MemoryStore.complete_session
    calls = {"n": 0}

    def flaky(self: MemoryStore, session_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash before completing")
        real(self, session_id)

    monkeypatch.setattr(MemoryStore, "complete_session", flaky)
    _drive(_commit_payload(git_repo))
    _drive(_commit_payload(git_repo))
    assert _corroboration_rows(per_repo_db) == 0


@pytest.mark.timeout(30)
def test_two_commits_with_one_message_and_date_stay_distinct(
    git_repo: Path, per_repo_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Their parents differ, so each is its own session and the second
    corroborates the first."""
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-01-01T00:00:00Z")
    _make_commit(git_repo, _MSG)
    _drive(_commit_payload(git_repo))
    _make_commit(git_repo, _MSG + " ")
    _drive(_commit_payload(git_repo))
    assert len(_sessions(per_repo_db)) == 2
    assert _corroboration_rows(per_repo_db) > 0


def test_unreadable_reflog_is_a_no_op(tmp_path: Path, per_repo_db: Path) -> None:
    """A cwd that isn't a repository: the hook stays silent."""
    assert _drive(_payload(
        command="git commit -m x", stdout="", cwd=str(tmp_path),
    )) == 0
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


@pytest.mark.timeout(30)
def test_a_fresh_checkout_of_an_old_commit_is_not_ingested(
    git_repo: Path, per_repo_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only `commit…` reflog entries count. A checkout writes a fresh
    entry pointing at an old commit, which this call didn't make."""
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2001-01-01T00:00:00Z")
    _make_commit(git_repo, _MSG)
    monkeypatch.delenv("GIT_COMMITTER_DATE")
    _git(git_repo, "checkout", "-q", "-b", "other")
    _drive(_commit_payload(git_repo, "git checkout -b other && git commit -q"))
    assert _sessions(per_repo_db) == []


def _belief(session_id: str) -> Belief:
    return Belief(
        id="b1", content="the spec is derived", content_hash="h1",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None, session_id=session_id,
    )


@pytest.mark.parametrize(
    ("source", "session", "rows"),
    [
        (CORROBORATION_SOURCE_COMMIT_INGEST, "S1", 0),  # its own belief
        (CORROBORATION_SOURCE_COMMIT_INGEST, "S2", 1),  # another commit
        (CORROBORATION_SOURCE_TRANSCRIPT_INGEST, "S1", 1),  # unchanged
    ],
)
def test_a_commit_session_never_corroborates_its_own_belief(
    tmp_path: Path, source: str, session: str, rows: int,
) -> None:
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        store.insert_belief(_belief("S1"))
        store.insert_or_corroborate(
            _belief(session), source_type=source, session_id=session,
        )
        n = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) FROM belief_corroborations"
        ).fetchone()[0]
        assert n == rows
    finally:
        store.close()
