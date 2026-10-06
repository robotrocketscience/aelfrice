"""#1698: commit ingest runs from git's own `post-commit` hook.

The PostToolUse hook guessed which commit a Bash call made from its
command text; in this project's sessions it saw about 2 of 399 commits
(#1683). Git runs `post-commit` inside the repository after every commit
it makes. The installed block resolves `HEAD` and checks for a rebase
synchronously, then hands the hash to a background process.
"""
from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice import hook_commit_ingest as hk
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    LOCK_NONE,
    Belief,
)
from aelfrice.setup import (
    GIT_HOOK_BEGIN,
    install_commit_ingest_git_hook,
    install_commit_ingest_hook,
    uninstall_commit_ingest_git_hook,
)
from aelfrice.store import MemoryStore

_MSG = "the spec is derived from the prior memo"
_MSG2 = "the new index is supported by faster queries"
_SCRIPT = Path(sys.executable).parent / "aelf-commit-ingest"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", check=False, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {r.stderr}")
    return r.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "T")
    _git(r, "config", "commit.gpgsign", "false")
    return r


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


@pytest.fixture
def hooked(repo: Path, db: Path) -> Path:
    """A repository with the real hook installed (background ingest)."""
    if not _SCRIPT.exists():
        pytest.skip("aelf-commit-ingest is not installed next to the interpreter")
    assert install_commit_ingest_git_hook(repo, command=str(_SCRIPT)).status == "installed"
    return repo


def _commit(repo: Path, message: str, name: str = "x") -> None:
    (repo / name).write_text(f"{message}\n{time.time_ns()}", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)


def _q(db: Path, sql: str) -> list[tuple[object, ...]]:
    if not db.exists():
        return []
    store = MemoryStore(str(db))
    try:
        return [tuple(r) for r in store._conn.execute(sql).fetchall()]  # pyright: ignore[reportPrivateUsage]
    finally:
        store.close()


def _sessions(db: Path) -> list[str]:
    return sorted(str(r[0]) for r in _q(db, "SELECT id FROM sessions WHERE model = 'commit-ingest'"))


def _corroborations(db: Path) -> int:
    rows = _q(db, "SELECT COUNT(*) FROM belief_corroborations")
    return int(rows[0][0]) if rows else 0  # type: ignore[arg-type]


def _wait_sessions(db: Path, n: int, timeout: float = 30.0) -> list[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = _sessions(db)
        if len(got) >= n:
            return got
        time.sleep(0.2)
    return _sessions(db)


def _settle() -> None:
    """Long enough for a background ingest to have written, if it ran."""
    time.sleep(3)


# --- the ingest itself ---------------------------------------------------------


def test_the_session_key_uses_both_parent_and_author_date() -> None:
    k = hk._commit_session_id  # pyright: ignore[reportPrivateUsage]
    base = k("p1", "2026-01-01T00:00:00Z")
    assert base == k("p1", "2026-01-01T00:00:00Z")
    assert base != k("p2", "2026-01-01T00:00:00Z")
    assert base != k("p1", "2026-01-01T00:00:01Z")


@pytest.mark.timeout(60)
def test_the_named_commit_is_ingested_not_head(repo: Path, db: Path) -> None:
    """The hash comes from the hook; a later HEAD doesn't matter."""
    _commit(repo, "base", "b")
    _commit(repo, _MSG)
    rev = _git(repo, "rev-parse", "HEAD").strip()
    parent = _git(repo, "log", "-1", "--format=%P").split()[0]
    author_date = _git(repo, "log", "-1", "--format=%aI").strip()
    _commit(repo, _MSG2, "y")  # HEAD moves on before the ingest runs
    hk.git_hook_ingest(rev, str(repo))
    assert _sessions(db) == [hk._commit_session_id(parent, author_date)]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.timeout(60)
def test_a_root_commit_is_ingested(repo: Path, db: Path) -> None:
    _commit(repo, _MSG)
    hk.git_hook_ingest("HEAD", str(repo))
    assert len(_sessions(db)) == 1


@pytest.mark.timeout(60)
def test_a_merge_commit_is_keyed_on_its_first_parent(repo: Path, db: Path) -> None:
    _commit(repo, "base", "b")
    _git(repo, "switch", "-q", "-c", "side")
    _commit(repo, "a side note", "s")
    _git(repo, "switch", "-q", "main")
    _commit(repo, "a main note", "m")
    main_tip = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "merge", "-q", "--no-ff", "side", "-m", _MSG)
    author_date = _git(repo, "log", "-1", "--format=%aI").strip()
    hk.git_hook_ingest("HEAD", str(repo))
    assert _sessions(db) == [hk._commit_session_id(main_tip, author_date)]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("edit", [False, True])
def test_an_amend_adds_no_corroboration(repo: Path, db: Path, edit: bool) -> None:
    _commit(repo, "base", "b")
    _commit(repo, _MSG)
    hk.git_hook_ingest("HEAD", str(repo))
    (repo / "x").write_text("amended", encoding="utf-8")
    _git(repo, "add", "x")
    if edit:
        _git(repo, "commit", "-q", "--amend", "-m", f"{_MSG}\n\n{_MSG2}")
    else:
        _git(repo, "commit", "-q", "--amend", "--no-edit")
    hk.git_hook_ingest("HEAD", str(repo))
    assert len(_sessions(db)) == 1
    assert _corroborations(db) == 0
    contents = " ".join(str(r[0]) for r in _q(db, "SELECT content FROM beliefs"))
    assert ("the new index" in contents) == edit


@pytest.mark.timeout(60)
def test_a_new_commit_with_the_same_message_corroborates(repo: Path, db: Path) -> None:
    _commit(repo, _MSG, "a")
    hk.git_hook_ingest("HEAD", str(repo))
    _commit(repo, _MSG, "b")
    hk.git_hook_ingest("HEAD", str(repo))
    assert len(_sessions(db)) == 2
    assert _corroborations(db) > 0


@pytest.mark.timeout(60)
def test_a_revert_is_not_ingested(repo: Path, db: Path) -> None:
    """Its message quotes the reverted subject; ingesting it would
    corroborate the claim being undone."""
    _commit(repo, _MSG)
    hk.git_hook_ingest("HEAD", str(repo))
    _git(repo, "revert", "--no-edit", "HEAD")
    hk.git_hook_ingest("HEAD", str(repo))
    assert len(_sessions(db)) == 1
    assert _corroborations(db) == 0


def _belief(session_id: str, content_hash: str = "h1") -> Belief:
    return Belief(
        id="b1", content="the spec is derived", content_hash=content_hash,
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None, session_id=session_id,
    )


@pytest.mark.parametrize("content_hash", ["h1", "h-other"])  # hash hit, id collision
def test_a_commit_session_never_corroborates_its_own_belief(
    tmp_path: Path, content_hash: str,
) -> None:
    store = MemoryStore(str(tmp_path / "s.db"))
    try:
        store.insert_belief(_belief("S1"))
        store.insert_or_corroborate(
            _belief("S1", content_hash),
            source_type=CORROBORATION_SOURCE_COMMIT_INGEST, session_id="S1",
        )
        n = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) FROM belief_corroborations"
        ).fetchone()[0]
        assert n == 0
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_main_takes_the_hash_after_the_flag(
    repo: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit(repo, _MSG)
    rev = _git(repo, "rev-parse", "HEAD").strip()
    _commit(repo, "a single short subject", "y")
    monkeypatch.chdir(repo)
    assert hk.main(argv=[hk.GIT_HOOK_FLAG, rev]) == 0
    assert len(_sessions(db)) == 1


def test_a_failure_is_logged_next_to_the_store_and_never_raised(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(rev: str, cwd: str | None = None) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(hk, "git_hook_ingest", boom)
    assert hk.main(argv=[hk.GIT_HOOK_FLAG, "abc"]) == 0
    log = db.parent / hk.GIT_HOOK_LOG_NAME
    assert "simulated failure" in log.read_text(encoding="utf-8")


def test_without_the_flag_main_reads_a_payload(db: Path) -> None:
    """The PostToolUse path is untouched for settings not yet upgraded."""
    assert hk.main(stdin=io.StringIO("{}"), stderr=io.StringIO(), argv=[]) == 0
    assert not db.exists()


# --- end to end through the installed hook -------------------------------------


@pytest.mark.timeout(120)
def test_every_commit_of_a_quick_burst_is_recorded(hooked: Path, db: Path) -> None:
    """Five commits in one shell loop: each hook passes its own hash."""
    script = "; ".join(
        f"echo {i} > f{i} && git add f{i} && git commit -q -m 'module{i} is derived from the legacy{i} tokenizer'"
        for i in range(5)
    )
    subprocess.run(["sh", "-c", script], cwd=hooked, check=True, timeout=60)
    assert len(_wait_sessions(db, 5)) == 5


@pytest.mark.timeout(120)
def test_a_real_rebase_records_no_replays(hooked: Path, db: Path) -> None:
    _commit(hooked, "base", "b")
    _git(hooked, "switch", "-q", "-c", "side")
    _commit(hooked, _MSG, "s")
    _git(hooked, "switch", "-q", "main")
    _commit(hooked, _MSG2, "m")
    _wait_sessions(db, 2)
    _settle()
    before = (_sessions(db), _corroborations(db))
    _git(hooked, "switch", "-q", "side")
    _git(hooked, "rebase", "-q", "main")
    _settle()
    assert (_sessions(db), _corroborations(db)) == before


@pytest.mark.timeout(120)
@pytest.mark.parametrize("state_dir", ["rebase-merge", "rebase-apply"])
def test_a_commit_during_a_paused_rebase_is_skipped(
    hooked: Path, db: Path, state_dir: str,
) -> None:
    """Documented: the rebase check can't tell a hand-made commit apart.
    `rebase-apply` is also what a paused `git am` leaves."""
    _commit(hooked, "base", "b")
    (hooked / ".git" / state_dir).mkdir()
    _commit(hooked, _MSG)
    _settle()
    assert _sessions(db) == []
    (hooked / ".git" / state_dir).rmdir()
    _commit(hooked, _MSG2, "y")
    assert len(_wait_sessions(db, 1)) == 1


@pytest.mark.timeout(120)
def test_a_rebase_in_a_worktree_skips_only_that_worktree(
    hooked: Path, db: Path, tmp_path: Path,
) -> None:
    """A rebase keeps its state in the worktree's own git directory."""
    _commit(hooked, "base", "b")
    wt = tmp_path / "wt"
    _git(hooked, "worktree", "add", "-q", str(wt))
    wt_git_dir = Path(_git(wt, "rev-parse", "--absolute-git-dir").strip())
    (wt_git_dir / "rebase-merge").mkdir()
    _commit(wt, _MSG)  # in the rebasing worktree: skipped
    _settle()
    assert _sessions(db) == []
    _commit(hooked, _MSG2, "y")  # main checkout: recorded
    assert len(_wait_sessions(db, 1)) == 1


@pytest.mark.timeout(120)
def test_a_worktree_commit_lands_in_the_shared_store(
    hooked: Path, db: Path, tmp_path: Path,
) -> None:
    _commit(hooked, "base", "b")
    wt = tmp_path / "wt"
    _git(hooked, "worktree", "add", "-q", str(wt))
    _commit(wt, _MSG)
    assert len(_wait_sessions(db, 1)) >= 1


# --- the installer -------------------------------------------------------------


def _hook(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-commit"


def _write_hook(repo: Path, data: bytes, mode: int = 0o755) -> Path:
    p = _hook(repo)
    p.write_bytes(data)
    p.chmod(mode)
    return p


@pytest.mark.timeout(60)
def test_install_creates_an_executable_hook(repo: Path) -> None:
    r = install_commit_ingest_git_hook(repo, command="/bin/aelf-commit-ingest")
    assert r.status == "installed"
    text = _hook(repo).read_text()
    assert text.startswith("#!/bin/sh\n" + GIT_HOOK_BEGIN)
    assert "'/bin/aelf-commit-ingest' --git-hook \"$_aelf_rev\"" in text
    assert "\r" not in _hook(repo).read_bytes().decode()  # LF on every platform
    # Detached: the commit never waits on the ingest.
    assert '"$_aelf_rev" </dev/null >/dev/null 2>&1 & )' in text
    assert _hook(repo).stat().st_mode & stat.S_IXUSR


@pytest.mark.timeout(60)
def test_install_is_idempotent_and_replaces_a_stale_command(repo: Path) -> None:
    install_commit_ingest_git_hook(repo, command="/old/aelf-commit-ingest")
    assert install_commit_ingest_git_hook(
        repo, command="/old/aelf-commit-ingest",
    ).status == "already_present"
    install_commit_ingest_git_hook(repo, command="/new/aelf-commit-ingest")
    text = _hook(repo).read_text()
    assert text.count(GIT_HOOK_BEGIN) == 1
    assert "/new/" in text and "/old/" not in text


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "original",
    [
        b"#!/bin/bash\necho mine\nexit 0\n",
        b"#!/bin/sh\n",  # the user's own shebang-only file is kept
        b"#!/bin/sh\necho no trailing newline",
    ],
)
def test_install_then_uninstall_restores_a_hook_exactly(
    repo: Path, original: bytes,
) -> None:
    p = _write_hook(repo, original, 0o750)
    assert install_commit_ingest_git_hook(repo, command="/bin/x").status == "installed"
    text = p.read_text()
    first = original.decode().partition("\n")[0]
    assert text.startswith(first + "\n" + GIT_HOOK_BEGIN)  # before any `exit`
    assert stat.S_IMODE(p.stat().st_mode) == 0o750
    assert uninstall_commit_ingest_git_hook(repo).status == "removed"
    assert p.read_bytes() == original
    assert stat.S_IMODE(p.stat().st_mode) == 0o750


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("data", "mode"),
    [
        (b"#!/usr/bin/env python3\nprint('mine')\n", 0o755),
        (b"#!/bin/sh -e\necho mine\n", 0o755),
        (b"#!/bin/sh", 0o755),  # no newline after the shebang
        (b"#!/bin/sh\r\necho mine\r\n", 0o755),  # CRLF
        (b"#!/bin/sh\necho mine\r\n", 0o755),  # CRLF after an LF shebang
        (b"", 0o755),  # empty
        (b"#!/bin/sh\necho \xff\n", 0o755),  # not UTF-8
        (b"#!/bin/sh\necho mine\n", 0o644),  # not executable: git skips it
    ],
)
def test_a_hook_aelfrice_cant_restore_exactly_is_left_alone(
    repo: Path, data: bytes, mode: int,
) -> None:
    p = _write_hook(repo, data, mode)
    assert install_commit_ingest_git_hook(repo, command="/bin/x").status == "foreign_hook"
    assert p.read_bytes() == data
    assert stat.S_IMODE(p.stat().st_mode) == mode


@pytest.mark.timeout(60)
def test_a_symlinked_hook_is_left_alone(repo: Path, tmp_path: Path) -> None:
    target = tmp_path / "real-hook"
    target.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    target.chmod(0o755)
    _hook(repo).symlink_to(target)
    assert install_commit_ingest_git_hook(repo, command="/bin/x").status == "foreign_hook"
    assert target.read_text() == "#!/bin/sh\necho mine\n"


@pytest.mark.timeout(60)
def test_install_honours_a_local_hooks_path_and_worktrees(repo: Path, tmp_path: Path) -> None:
    """Like this project: core.hooksPath in the repository's own config,
    pointing inside .git."""
    local = repo / ".git" / "my-hooks"
    _git(repo, "config", "core.hooksPath", str(local))
    _commit(repo, "base", "b")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    r = install_commit_ingest_git_hook(wt, command="/bin/x")
    assert (r.status, r.path) == ("installed", local / "post-commit")


@pytest.mark.timeout(60)
def test_a_hooks_path_from_global_config_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It would install for every repository using that directory."""
    shared = tmp_path / "global-hooks"
    cfg = tmp_path / "gitconfig"
    cfg.write_text(f"[core]\n\thooksPath = {shared}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    r = install_commit_ingest_git_hook(repo, command="/bin/x")
    assert r.status == "shared_hooks_dir"
    assert not (shared / "post-commit").exists()


@pytest.mark.timeout(60)
def test_a_global_hooks_path_naming_this_repos_hooks_dir_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every repository would run this repository's hooks."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text(
        f"[core]\n\thooksPath = {(repo / '.git' / 'hooks').resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    assert install_commit_ingest_git_hook(repo, command="/bin/x").status == "shared_hooks_dir"


@pytest.mark.timeout(60)
def test_a_hooks_path_from_the_command_line_environment_is_refused(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GIT_CONFIG_COUNT` config has no file origin at all."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "env-hooks"))
    r = install_commit_ingest_git_hook(repo, command="/bin/x")
    assert r.status == "shared_hooks_dir"


@pytest.mark.timeout(60)
def test_a_local_hooks_path_outside_the_git_dir_is_refused(
    repo: Path, tmp_path: Path,
) -> None:
    """One directory named in several repositories' config would run the
    hook for all of them."""
    shared = tmp_path / "my-githooks"
    _git(repo, "config", "core.hooksPath", str(shared))
    assert install_commit_ingest_git_hook(repo, command="/bin/x").status == "shared_hooks_dir"
    assert not (shared / "post-commit").exists()


@pytest.mark.timeout(60)
def test_this_projects_config_shape_works_from_a_subdirectory(repo: Path) -> None:
    """core.hooksPath set in the repository's own config to an absolute
    path inside .git, and setup run from a subdirectory."""
    hooks = (repo / ".git" / "hooks").resolve()
    _git(repo, "config", "core.hooksPath", str(hooks))
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    r = install_commit_ingest_git_hook(sub, command="/bin/x")
    assert (r.status, r.path) == ("installed", hooks / "post-commit")


@pytest.mark.timeout(60)
def test_a_bare_repository_is_refused(tmp_path: Path) -> None:
    """Nothing commits in a bare repository, so the hook would never run."""
    bare = tmp_path / "bare.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    r = install_commit_ingest_git_hook(bare, command="/bin/x")
    assert (r.status, r.path) == ("not_a_repo", None)
    assert not (bare / "hooks" / "post-commit").exists()


@pytest.mark.timeout(60)
def test_a_hooks_path_inside_the_work_tree_is_refused(repo: Path) -> None:
    """A tracked hooks directory (husky-style) would get committed."""
    _git(repo, "config", "core.hooksPath", ".husky")
    r = install_commit_ingest_git_hook(repo, command="/bin/x")
    assert r.status == "tracked_hooks_dir"
    assert not (repo / ".husky" / "post-commit").exists()


def test_not_a_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert install_commit_ingest_git_hook(plain, command="/x").status == "not_a_repo"
    assert uninstall_commit_ingest_git_hook(plain).status == "not_a_repo"


@pytest.mark.timeout(60)
def test_uninstall_keeps_lines_you_added_to_a_hook_aelfrice_created(repo: Path) -> None:
    install_commit_ingest_git_hook(repo, command="/bin/x")
    with _hook(repo).open("a", encoding="utf-8") as fh:
        fh.write("echo mine\n")
    assert uninstall_commit_ingest_git_hook(repo).status == "removed"
    assert _hook(repo).read_text() == "#!/bin/sh\necho mine\n"


@pytest.mark.timeout(60)
def test_uninstall_deletes_a_hook_aelfrice_created(repo: Path) -> None:
    install_commit_ingest_git_hook(repo, command="/bin/x")
    assert uninstall_commit_ingest_git_hook(repo).status == "removed"
    assert not _hook(repo).exists()
    assert uninstall_commit_ingest_git_hook(repo).status == "absent"


# --- `aelf setup` / `unsetup` wiring -------------------------------------------


@pytest.mark.timeout(60)
def test_setup_installs_the_git_hook_and_drops_old_entries_in_both_scopes(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELF_NO_GIT_HOOK_INSTALL", raising=False)
    monkeypatch.chdir(repo)
    user = tmp_path / "user-settings.json"
    monkeypatch.setattr("aelfrice.setup.USER_SETTINGS_PATH", user)
    project = repo / ".claude" / "settings.json"
    install_commit_ingest_hook(user, command="/old/aelf-commit-ingest")
    install_commit_ingest_hook(project, command="/old/aelf-commit-ingest")
    buf = io.StringIO()
    cli_module._setup_commit_ingest_git_hook("project", project, buf)  # pyright: ignore[reportPrivateUsage]
    for sp in (user, project):
        data = json.loads(sp.read_text(encoding="utf-8"))
        assert not data["hooks"].get("PostToolUse"), sp
    assert GIT_HOOK_BEGIN in _hook(repo).read_text()
    assert "installed the commit-ingest git hook" in buf.getvalue()


@pytest.mark.timeout(60)
def test_unsetup_removes_the_block(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELF_NO_GIT_HOOK_INSTALL", raising=False)
    monkeypatch.chdir(repo)
    install_commit_ingest_git_hook(repo, command="/bin/x")
    buf = io.StringIO()
    assert cli_module._unsetup_commit_ingest_git_hook(buf)  # pyright: ignore[reportPrivateUsage]
    assert not _hook(repo).exists()


@pytest.mark.timeout(60)
def test_the_suite_guard_keeps_setup_and_unsetup_out_of_the_checkout(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELF_NO_GIT_HOOK_INSTALL", "1")
    monkeypatch.chdir(repo)
    cli_module._setup_commit_ingest_git_hook(  # pyright: ignore[reportPrivateUsage]
        "project", tmp_path / "settings.json", io.StringIO(),
    )
    assert not _hook(repo).exists()
    _write_hook(repo, b"#!/bin/sh\n" + GIT_HOOK_BEGIN.encode() + b"\n# <<< aelfrice commit-ingest <<<\n")
    assert not cli_module._unsetup_commit_ingest_git_hook(io.StringIO())  # pyright: ignore[reportPrivateUsage]
    assert _hook(repo).exists()


def test_the_suite_runs_with_the_guard_set() -> None:
    """conftest pins it for the whole session."""
    assert os.environ.get("AELF_NO_GIT_HOOK_INSTALL") == "1"
