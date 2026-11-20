"""One statement locked in both scopes is reported, not hidden (#1947).

A lock's id is derived from its text, so the same statement locked with
and without `--user` is one id in two stores. The L0 merge injects the
repository copy and drops the user copy. These tests pin that `lock`,
`locked`, and `unlock` each say when the other scope holds the same
lock, with the command that removes the other copy, and that the check
never creates a store or fails a lock when the other store is broken.

Every path here is a temp path; each test repoints `AELFRICE_DB` and
`AELFRICE_USER_DB`.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from aelfrice import cli

STATEMENT = "Route all questions and decisions through the question tool."
NEAR_STATEMENT = "Route all questions and decisions through the question tool always."
OTHER_STATEMENT = "This repository builds with make, not cmake."


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


def _lock_id(text: str) -> str:
    return hashlib.sha256(f"lock\x00{text}".encode()).hexdigest()[:16]


def _ids(scope: str) -> dict[str, str]:
    rc, out, _ = _run(["locked", "--json"])
    assert rc == 0
    return {r["content"]: r["id"] for r in json.loads(out) if r["scope"] == scope}


# --- aelf lock -----------------------------------------------------------


@pytest.mark.timeout(60)
def test_lock_user_after_a_repo_lock_names_the_repo_unlock(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    rc, out, _ = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    bid = _ids("repo")[STATEMENT]
    assert _ids("user")[STATEMENT] == bid
    assert f"`aelf unlock {bid}`" in out
    assert "--user" not in out.split("warning:", 1)[1]


@pytest.mark.timeout(60)
def test_repo_lock_after_a_user_lock_names_the_user_unlock(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    rc, out, _ = _run(["lock", STATEMENT])
    assert rc == 0
    bid = _ids("user")[STATEMENT]
    assert f"`aelf unlock --user {bid}`" in out


@pytest.mark.timeout(60)
def test_lock_warns_on_a_near_duplicate_in_the_other_scope(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", "--user", NEAR_STATEMENT])[0] == 0
    uid = _ids("user")[NEAR_STATEMENT]
    rc, out, _ = _run(["lock", STATEMENT])
    assert rc == 0
    assert f"`aelf unlock --user {uid}`" in out

    assert _run(["lock", OTHER_STATEMENT])[0] == 0
    rid = _ids("repo")[OTHER_STATEMENT]
    rc, out, _ = _run(["lock", "--user", OTHER_STATEMENT.replace("make,", "make only,")])
    assert rc == 0
    assert f"`aelf unlock {rid}`" in out


@pytest.mark.timeout(60)
def test_lock_with_no_match_in_the_other_scope_says_nothing(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", "--user", OTHER_STATEMENT])[0] == 0
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert "unlock" not in out
    assert err == ""


@pytest.mark.timeout(60)
def test_plain_lock_leaves_no_user_store(stores: tuple[Path, Path]) -> None:
    _, user = stores
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert "unlock" not in out
    assert err == ""
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_lock_user_leaves_no_repo_store(stores: tuple[Path, Path]) -> None:
    repo, _ = stores
    rc, out, err = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert "`aelf unlock" not in out
    assert err == ""
    assert not repo.exists()


@pytest.mark.timeout(60)
def test_plain_lock_survives_a_corrupt_user_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    user.parent.mkdir(parents=True)
    user.write_bytes(b"this is not a sqlite database" * 64)
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert out.startswith("locked: ")
    assert STATEMENT in _ids("repo")
    assert "could not check the other scope" in err
    assert "the user lock store could not be read" in err


# --- aelf locked ---------------------------------------------------------


@pytest.mark.timeout(60)
def test_locked_marks_an_id_locked_in_both_scopes(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    assert _run(["lock", "--user", OTHER_STATEMENT])[0] == 0
    rc, out, err = _run(["locked"])
    assert rc == 0
    bid = _ids("repo")[STATEMENT]
    lines = out.splitlines()
    both = [ln for ln in lines if ln.startswith(f"{bid} ")]
    assert len(both) == 2
    assert all("[also locked in the other scope]" in ln for ln in both)
    [other] = [ln for ln in lines if ln.endswith(f": {OTHER_STATEMENT}")]
    assert "[also locked in the other scope]" not in other
    assert f"`aelf unlock {bid}`" in err
    assert f"`aelf unlock --user {bid}`" in err

    rows = json.loads(_run(["locked", "--json"])[1])
    assert sorted((r["scope"], r["content"], r["in_both_scopes"]) for r in rows) == [
        ("repo", STATEMENT, True),
        ("user", STATEMENT, True),
        ("user", OTHER_STATEMENT, False),
    ]


# --- aelf unlock ---------------------------------------------------------


@pytest.mark.timeout(60)
def test_repo_unlock_says_the_user_copy_is_still_locked(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _ids("repo")[STATEMENT]
    rc, out, err = _run(["unlock", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    assert f"`aelf unlock --user {bid}`" in err


@pytest.mark.timeout(60)
def test_user_unlock_says_the_repo_copy_is_still_locked(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _ids("user")[STATEMENT]
    rc, out, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    assert f"`aelf unlock {bid}`" in err


@pytest.mark.timeout(60)
def test_user_unlock_of_a_repo_only_lock_points_at_the_repo(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", "--user", OTHER_STATEMENT])[0] == 0
    assert _run(["lock", STATEMENT])[0] == 0
    bid = _ids("repo")[STATEMENT]
    rc, _, err = _run(["unlock", "--user", bid])
    assert rc == 1
    assert f"`aelf unlock {bid}`" in err


@pytest.mark.timeout(60)
def test_user_unlock_with_one_copy_says_nothing_more(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    rc, _, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert err == ""
    assert not repo.exists()


@pytest.mark.timeout(60)
def test_user_unlock_ignores_an_expired_repo_lock(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    assert _run(["lock", "--for", "1w", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    # Past its window on disk, as a store no writable open has swept yet.
    conn = sqlite3.connect(repo)
    try:
        conn.execute(
            "UPDATE beliefs SET lock_expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", bid),
        )
        conn.commit()
    finally:
        conn.close()
    rc, out, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    assert err == ""
