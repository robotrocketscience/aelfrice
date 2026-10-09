"""User-scope lock store and write path (#1681, PR1 of 3).

A lock typed with `--user` lands in one store that every repository
shares, `~/.aelfrice/user/memory.db` or `$AELFRICE_USER_DB`, instead of
the repository store. These tests pin where that store resolves, that
only a write creates it, that `--user` changes the store and nothing
else, and that `locked` and `unlock` address both scopes.

Every path here is a temp path. The conftest pins `AELFRICE_USER_DB`
and `AELFRICE_DB` at a sandbox for the whole run; the tests below repoint
both per test.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice import cli, db_paths
from aelfrice.hook import (
    COMMAND_ARGUMENT_CAP,
    CommandReason,
    execute_aelf_command,
    user_prompt_submit,
)
from aelfrice.hook_audit import command_outcomes_path_for_db
from aelfrice.lock_gaps import detect_lock_gaps, read_command_outcomes
from aelfrice.store import MemoryStore

STATEMENT = "Route all questions and decisions through the question tool."
REPO_STATEMENT = "This repository builds with make, not cmake."


@pytest.fixture()
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(repo_db, user_db), both temp paths, neither created yet."""
    repo = tmp_path / "repo" / "memory.db"
    user = tmp_path / "home" / ".aelfrice" / "user" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    return repo, user


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    old_err = sys.stderr
    sys.stderr = err
    try:
        rc = cli.main(argv, out=out)
    finally:
        sys.stderr = old_err
    return rc, out.getvalue(), err.getvalue()


def _locked_contents(db: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            r[0] for r in conn.execute(
                "SELECT content FROM beliefs WHERE lock_level = 'user' "
                "AND valid_to IS NULL ORDER BY content"
            )
        ]
    finally:
        conn.close()


def _all_contents(db: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute("SELECT content FROM beliefs")]
    finally:
        conn.close()


def _row(db: Path, content: str) -> sqlite3.Row:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM beliefs WHERE content = ?", (content,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row


# --- path resolution -----------------------------------------------------


def test_aelfrice_user_db_overrides_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "elsewhere" / "u.db"
    monkeypatch.setenv("AELFRICE_USER_DB", str(target))
    assert db_paths.user_db_path() == target


def test_default_path_is_user_subdir_of_default_db_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_USER_DB", raising=False)
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", tmp_path / ".aelfrice")
    assert db_paths.user_db_path() == (
        tmp_path / ".aelfrice" / "user" / "memory.db"
    )


@pytest.mark.timeout(60)
def test_default_path_follows_home_in_a_fresh_process(tmp_path: Path) -> None:
    """`DEFAULT_DB_DIR` is bound from `HOME` at import, so the real check
    is a fresh interpreter whose `HOME` is the fake one."""
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("AELFRICE_USER_DB",)
    }
    env["HOME"] = str(tmp_path)
    env["AELFRICE_DB"] = str(tmp_path / "repo-pin.db")
    program = "from aelfrice.db_paths import user_db_path; print(user_db_path())"
    proc = subprocess.run(
        [sys.executable, "-c", program],
        env=env, capture_output=True, text=True, check=True, timeout=50,
    )
    assert Path(proc.stdout.strip()) == (
        tmp_path / ".aelfrice" / "user" / "memory.db"
    )


def test_aelfrice_db_does_not_redirect_the_user_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_USER_DB", raising=False)
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", tmp_path / ".aelfrice")
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned" / "memory.db"))
    assert db_paths.user_db_path() == (
        tmp_path / ".aelfrice" / "user" / "memory.db"
    )
    assert db_paths.db_path() == tmp_path / "pinned" / "memory.db"


def test_resolving_the_path_creates_nothing(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    db_paths.user_db_path()
    assert not user.parent.exists()


def test_read_open_of_a_missing_store_creates_nothing(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert db_paths.open_user_store_if_present() is None
    assert not user.exists()
    assert not user.parent.exists()


def test_write_open_creates_the_store_with_no_repo_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A path whose parent is named `aelfrice` would derive a repo
    # identity under `db_path()`'s layout rule; the user store must not.
    user = tmp_path / "x" / "aelfrice" / "memory.db"
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    store = db_paths.open_user_store()
    try:
        assert store._project_context_default == ""
    finally:
        store.close()
    assert user.is_file()


# --- aelf lock --user ----------------------------------------------------


@pytest.mark.timeout(60)
def test_lock_user_writes_the_user_store_not_the_repo_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    rc, out, _ = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert _locked_contents(user) == [STATEMENT]
    assert "scope: user" in out
    assert "not yet injected" in out
    if repo.exists():
        assert STATEMENT not in _all_contents(repo)


@pytest.mark.timeout(60)
def test_lock_user_goes_through_the_ingest_log(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT raw_text, source_kind FROM ingest_log"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(STATEMENT, "cli_remember")]


@pytest.mark.timeout(60)
def test_plain_lock_is_unchanged_and_leaves_no_user_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    rc, out, _ = _run(["lock", REPO_STATEMENT])
    assert rc == 0
    assert _locked_contents(repo) == [REPO_STATEMENT]
    assert "scope:" not in out
    assert not user.exists()
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_lock_user_applies_tier_and_window(stores: tuple[Path, Path]) -> None:
    _, user = stores
    rc, out, _ = _run(["lock", "--user", "--reference", "--for", "1w",
                       STATEMENT])
    assert rc == 0
    row = _row(user, STATEMENT)
    assert row["lock_tier"] == "reference"
    assert row["lock_expires_at"] is not None
    assert "tier: reference" in out


@pytest.mark.timeout(60)
def test_near_duplicate_check_reads_the_user_store(
    stores: tuple[Path, Path],
) -> None:
    near = "Route all the questions and decisions through the question tool."
    # A repo lock is not in scope for a user lock's duplicate check.
    assert _run(["lock", STATEMENT])[0] == 0
    rc, out, _ = _run(["lock", "--user", near])
    assert rc == 0
    assert "near-duplicate" not in out
    # A user lock is.
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    rc, out, _ = _run(["lock", "--user", near])
    assert rc == 0
    assert "near-duplicate of 1 existing lock(s)" in out
    assert "`aelf unlock --user`" in out


@pytest.mark.parametrize(
    ("extra", "flag"),
    [(["--doc", "file:///x.md"], "--doc"), (["--category", "c"], "--category")],
)
@pytest.mark.timeout(60)
def test_flags_that_cannot_apply_to_user_scope_are_rejected(
    stores: tuple[Path, Path], extra: list[str], flag: str,
) -> None:
    repo, user = stores
    rc, _, err = _run(["lock", "--user", *extra, STATEMENT])
    assert rc == 1
    assert f"{flag} cannot be used with --user" in err
    assert not user.exists()
    assert not repo.exists()


@pytest.mark.timeout(60)
def test_user_store_naming_the_repo_store_is_refused(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    rc, _, err = _run(["lock", "--user", STATEMENT])
    assert rc == 1
    assert "AELFRICE_USER_DB names the repository store" in err
    assert not repo.exists()


# --- aelf locked ---------------------------------------------------------


@pytest.mark.timeout(60)
def test_locked_lists_both_scopes_with_the_user_ones_tagged(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    rc, out, _ = _run(["locked"])
    assert rc == 0
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith(f": {REPO_STATEMENT}")
    assert "[user]" not in lines[0]
    assert lines[1].endswith(f": {STATEMENT}")
    assert " [user] " in lines[1]


@pytest.mark.timeout(60)
def test_locked_json_carries_scope_in_a_fixed_order(
    stores: tuple[Path, Path],
) -> None:
    # Lock the user one first: the listing order is by scope, not time.
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    rc, out, _ = _run(["locked", "--json"])
    assert rc == 0
    rows = json.loads(out)
    assert [(r["scope"], r["content"]) for r in rows] == [
        ("repo", REPO_STATEMENT),
        ("user", STATEMENT),
    ]
    rc2, out2, _ = _run(["locked", "--json"])
    assert out2 == out


@pytest.mark.timeout(60)
def test_locked_does_not_create_the_user_store(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    rc, out, _ = _run(["locked"])
    assert rc == 0
    assert REPO_STATEMENT in out
    assert not user.exists()
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_locked_with_no_locks_anywhere(stores: tuple[Path, Path]) -> None:
    rc, out, _ = _run(["locked"])
    assert (rc, out.strip()) == (0, "no locked beliefs")
    rc, out, _ = _run(["locked", "--json"])
    assert (rc, json.loads(out)) == (0, [])


# --- aelf unlock --user --------------------------------------------------


def _user_lock_id(statement: str) -> str:
    rc, out, _ = _run(["locked", "--json"])
    assert rc == 0
    [bid] = [
        r["id"] for r in json.loads(out)
        if r["scope"] == "user" and r["content"] == statement
    ]
    return str(bid)


@pytest.mark.timeout(60)
def test_unlock_user_drops_the_user_lock(stores: tuple[Path, Path]) -> None:
    repo, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _user_lock_id(STATEMENT)
    rc, out, _ = _run(["unlock", "--user", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    assert _locked_contents(user) == []
    assert _locked_contents(repo) == [REPO_STATEMENT]


@pytest.mark.timeout(60)
def test_plain_unlock_leaves_a_user_lock_and_points_at_user(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _user_lock_id(STATEMENT)
    rc, _, err = _run(["unlock", bid])
    assert rc == 1
    assert f"aelf unlock --user {bid}" in err
    assert _locked_contents(user) == [STATEMENT]


@pytest.mark.timeout(60)
def test_unlock_user_with_no_user_store_fails_and_creates_nothing(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    rc, _, err = _run(["unlock", "--user", "deadbeefdeadbeef"])
    assert rc == 1
    assert "no user lock store" in err
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_unlock_user_naming_the_repo_store_is_refused(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    rc, _, err = _run(["unlock", "--user", "x"])
    assert rc == 1
    assert "AELFRICE_USER_DB names the repository store" in err
    assert _locked_contents(repo) == [REPO_STATEMENT]


@pytest.mark.timeout(60)
def test_plain_unlock_of_a_missing_id_does_not_hint_at_an_unlocked_user_row(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _user_lock_id(STATEMENT)
    assert _run(["unlock", "--user", bid])[0] == 0
    rc, _, err = _run(["unlock", bid])
    assert rc == 1
    assert "--user" not in err


@pytest.mark.timeout(60)
def test_already_unlocked_in_the_repo_points_at_the_user_lock(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    # One statement locked in both scopes has one id in each.
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _user_lock_id(STATEMENT)
    assert _run(["unlock", bid])[0] == 0
    rc, out, err = _run(["unlock", bid])
    assert rc == 0
    assert out.strip() == f"already unlocked: {bid}"
    assert f"aelf unlock --user {bid}" in err
    assert _locked_contents(user) == [STATEMENT]


# --- the same-file guard and a broken user store -------------------------


@pytest.mark.timeout(60)
def test_locked_lists_a_shared_file_once_as_repo_locks(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    rc, out, _ = _run(["locked", "--json"])
    assert rc == 0
    assert [(r["scope"], r["content"]) for r in json.loads(out)] == [
        ("repo", REPO_STATEMENT),
    ]


@pytest.mark.timeout(60)
def test_same_file_guard_resolves_the_path(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    link = tmp_path / "link-to-repo"
    link.symlink_to(repo.parent, target_is_directory=True)
    monkeypatch.setenv("AELFRICE_USER_DB", str(link / repo.name))
    rc, _, err = _run(["lock", "--user", STATEMENT])
    assert rc == 1
    assert "AELFRICE_USER_DB names the repository store" in err
    assert not repo.exists()


@pytest.mark.timeout(60)
def test_locked_with_a_corrupt_user_store_lists_repo_locks(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    user.parent.mkdir(parents=True)
    user.write_bytes(b"this is not a sqlite database" * 64)
    rc, out, err = _run(["locked", "--json"])
    assert rc == 0
    assert [(r["scope"], r["content"]) for r in json.loads(out)] == [
        ("repo", REPO_STATEMENT),
    ]
    assert "cannot read the user lock store" in err
    assert str(user) in err


@pytest.mark.timeout(60)
def test_plain_unlock_survives_a_corrupt_user_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    [bid] = [
        r["id"] for r in json.loads(_run(["locked", "--json"])[1])
    ]
    user.parent.mkdir(parents=True)
    user.write_bytes(b"this is not a sqlite database" * 64)
    assert _run(["unlock", bid])[0] == 0
    rc, out, _ = _run(["unlock", bid])
    assert (rc, out.strip()) == (0, f"already unlocked: {bid}")
    assert _locked_contents(repo) == []


@pytest.mark.timeout(60)
def test_read_open_falls_back_to_read_only_when_writes_are_refused(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    real = db_paths.MemoryStore

    def refuse_writable(path: str, **kw: object) -> MemoryStore:
        if not kw.get("read_only"):
            exc = sqlite3.OperationalError("attempt to write a readonly database")
            exc.sqlite_errorname = "SQLITE_READONLY"  # type: ignore[attr-defined]
            raise exc
        return real(path, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(db_paths, "MemoryStore", refuse_writable)
    store = db_paths.open_user_store_if_present()
    assert store is not None
    try:
        assert [b.content for b in store.list_locked_beliefs()] == [STATEMENT]
    finally:
        store.close()


def _make_user_dir_read_only(user: Path) -> None:
    """Leave the user store with no sidecars in a directory it can't write.

    The writable open is refused and the read-only fallback can't
    create its `-shm`/`-wal` pair, so it raises
    `ReadOnlyStoreUnavailable` rather than a `sqlite3.Error`.
    """
    for suffix in ("-shm", "-wal"):
        Path(f"{user}{suffix}").unlink(missing_ok=True)
    user.parent.chmod(0o555)


@pytest.fixture()
def restore_user_dir(stores: tuple[Path, Path]):  # type: ignore[no-untyped-def]
    yield
    _, user = stores
    if user.parent.exists():
        user.parent.chmod(0o755)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX directory permissions; root ignores them",
)
@pytest.mark.timeout(60)
def test_locked_with_an_unopenable_user_store_lists_repo_locks(
    stores: tuple[Path, Path], restore_user_dir: None,
) -> None:
    _, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    _make_user_dir_read_only(user)
    rc, out, err = _run(["locked", "--json"])
    assert rc == 0
    assert [(r["scope"], r["content"]) for r in json.loads(out)] == [
        ("repo", REPO_STATEMENT),
    ]
    assert err.count("cannot read the user lock store") == 1


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX directory permissions; root ignores them",
)
@pytest.mark.timeout(60)
def test_plain_unlock_survives_an_unopenable_user_store(
    stores: tuple[Path, Path], restore_user_dir: None,
) -> None:
    _, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    [bid] = [r["id"] for r in json.loads(_run(["locked", "--json"])[1])]
    assert _run(["unlock", bid])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    _make_user_dir_read_only(user)
    rc, out, _ = _run(["unlock", bid])
    assert (rc, out.strip()) == (0, f"already unlocked: {bid}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hardlinks")
@pytest.mark.timeout(60)
def test_same_file_guard_catches_a_hardlink(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    alias = tmp_path / "alias.db"
    os.link(repo, alias)
    monkeypatch.setenv("AELFRICE_USER_DB", str(alias))
    rc, _, err = _run(["lock", "--user", STATEMENT])
    assert rc == 1
    assert "AELFRICE_USER_DB names the repository store" in err
    assert _locked_contents(repo) == [REPO_STATEMENT]


@pytest.mark.timeout(60)
def test_same_file_guard_catches_a_case_variant_on_a_case_insensitive_fs(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    variant = repo.with_name(repo.name.upper())
    if not variant.exists():
        pytest.skip("case-sensitive filesystem")
    monkeypatch.setenv("AELFRICE_USER_DB", str(variant))
    rc, _, err = _run(["lock", "--user", STATEMENT])
    assert rc == 1
    assert "AELFRICE_USER_DB names the repository store" in err
    assert _locked_contents(repo) == [REPO_STATEMENT]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
@pytest.mark.timeout(60)
def test_locked_with_a_symlink_loop_user_path_lists_repo_locks(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    (tmp_path / "loop1").symlink_to(tmp_path / "loop2")
    (tmp_path / "loop2").symlink_to(tmp_path / "loop1")
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "loop1"))
    rc, out, _ = _run(["locked", "--json"])
    assert rc == 0
    assert [(r["scope"], r["content"]) for r in json.loads(out)] == [
        ("repo", REPO_STATEMENT),
    ]


# --- the feed event and a write under the default path -------------------


@pytest.mark.timeout(60)
def test_lock_user_feed_event_is_tagged_with_its_scope(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    rows = [
        json.loads(line)
        for line in (repo.parent / "feed.jsonl").read_text().splitlines()
    ]
    scopes = {
        r["snippet"]: r.get("scope") for r in rows
        if r["event"] == "belief.locked"
    }
    assert scopes == {REPO_STATEMENT: None, STATEMENT: "user"}


def _locked_feed_rows(repo: Path) -> list[dict[str, object]]:
    return [
        r for r in (
            json.loads(line)
            for line in (repo.parent / "feed.jsonl").read_text().splitlines()
        )
        if r["event"] == "belief.locked"
    ]


@pytest.mark.timeout(60)
def test_lock_user_upgrade_feed_event_is_tagged_with_its_scope(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    assert _run(["unlock", "--user", _user_lock_id(STATEMENT)])[0] == 0
    rc, out, _ = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert out.startswith("upgraded existing belief to lock:")
    last = _locked_feed_rows(repo)[-1]
    assert (last.get("kind"), last.get("scope")) == ("upgrade", "user")


@pytest.mark.timeout(60)
def test_lock_user_corroborated_feed_event_is_tagged_with_its_scope(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    from aelfrice.derivation import DerivationInput, derive

    repo, _ = stores
    monkeypatch.delenv("AELFRICE_FEED_LOG", raising=False)
    derived = derive(DerivationInput(
        raw_text=STATEMENT, source_kind="cli_remember",
        ts="2026-01-01T00:00:00Z", session_id=None,
    ))
    assert derived.belief is not None
    # Same text from another source: the content hash matches, the id
    # does not, so the worker corroborates this row.
    other = dataclasses.replace(derived.belief, id="0123456789abcdef")
    store = db_paths.open_user_store()
    try:
        store.insert_belief(other)
    finally:
        store.close()
    rc, out, _ = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert "(corroborated existing)" in out
    last = _locked_feed_rows(repo)[-1]
    assert (last.get("kind"), last.get("scope")) == ("corroborated", "user")


@pytest.mark.timeout(60)
def test_lock_user_writes_under_the_default_dir_when_unset(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = tmp_path / "fakehome" / ".aelfrice"
    monkeypatch.delenv("AELFRICE_USER_DB", raising=False)
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", fake)
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    assert _locked_contents(fake / "user" / "memory.db") == [STATEMENT]


# --- /aelf:lock --user through the prompt hook executor ------------------


@pytest.mark.timeout(120)
def test_typed_lock_user_writes_the_user_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    err = io.StringIO()
    outcome = execute_aelf_command(
        f"/aelf:lock --user {STATEMENT}", stderr=err,
    )
    assert outcome is not None
    assert outcome.took_effect
    assert outcome.user_scope
    assert outcome.argument == STATEMENT
    assert _locked_contents(user) == [STATEMENT]
    if repo.exists():
        assert STATEMENT not in _all_contents(repo)
    assert f"'--user {STATEMENT}'" in outcome.line


@pytest.mark.timeout(120)
def test_typed_lock_user_takes_the_first_line_only(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    outcome = execute_aelf_command(
        f"/aelf:lock --user {STATEMENT}\nand then more prose", stderr=io.StringIO(),
    )
    assert outcome is not None and outcome.took_effect
    assert _locked_contents(user) == [STATEMENT]


@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("/aelf:lock --user", CommandReason.EMPTY_ARGUMENT),
        ("/aelf:lock --user   ", CommandReason.EMPTY_ARGUMENT),
        ("/aelf:lock --user --frozen x", CommandReason.LEADING_DASH),
        ("/aelf:lock --username x", CommandReason.LEADING_DASH),
        ("/aelf:confirm --user x", CommandReason.LEADING_DASH),
        (
            "/aelf:lock --user " + "a" * (COMMAND_ARGUMENT_CAP + 1),
            CommandReason.OVER_CAP,
        ),
    ],
)
@pytest.mark.timeout(60)
def test_typed_lock_user_input_errors_write_nothing(
    stores: tuple[Path, Path], prompt: str, reason: CommandReason,
) -> None:
    repo, user = stores
    outcome = execute_aelf_command(prompt, stderr=io.StringIO())
    assert outcome is not None
    assert outcome.reason is reason
    assert not outcome.took_effect
    assert not user.exists()
    assert not repo.exists()


@pytest.mark.timeout(60)
def test_typed_lock_at_the_cap_is_accepted_after_the_flag(
    stores: tuple[Path, Path],
) -> None:
    """The cap measures the statement, not the flag in front of it."""
    _, user = stores
    statement = "a" * COMMAND_ARGUMENT_CAP
    outcome = execute_aelf_command(
        f"/aelf:lock --user {statement}", stderr=io.StringIO(),
    )
    assert outcome is not None and outcome.took_effect
    assert _locked_contents(user) == [statement]


@pytest.mark.timeout(120)
def test_hook_records_user_scope_and_lock_gaps_skips_it(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    payload = json.dumps(
        {"prompt": f"/aelf:lock --user {STATEMENT}", "session_id": "s1681",
         "cwd": str(tmp_path)}
    )
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=io.StringIO(), stderr=io.StringIO(),
    ) == 0
    assert _locked_contents(user) == [STATEMENT]
    [row] = read_command_outcomes(command_outcomes_path_for_db(repo))
    assert row["scope"] == "user"
    assert row["reason"] == "ok"
    assert row["arg_sha256"] == hashlib.sha256(
        STATEMENT.encode("utf-8")
    ).hexdigest()

    # A failed user lock is not judged against the repository store.
    path = command_outcomes_path_for_db(repo)
    failed = {**row, "reason": "exception"}
    path.write_text(json.dumps(failed) + "\n", encoding="utf-8")
    assert detect_lock_gaps(str(repo), audit_enabled=True).gaps == ()
    unscoped = {k: v for k, v in failed.items() if k != "scope"}
    path.write_text(json.dumps(unscoped) + "\n", encoding="utf-8")
    assert len(detect_lock_gaps(str(repo), audit_enabled=True).gaps) == 1


@pytest.mark.timeout(60)
def test_user_store_shares_the_repo_store_schema(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    assert _run(["lock", REPO_STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0

    def tables(p: Path) -> set[str]:
        s = MemoryStore(str(p), read_only=True)
        try:
            return {
                str(r[0]) for r in s._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            s.close()

    assert tables(user) == tables(repo)
