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
import re
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
def test_repo_unlock_again_points_at_the_user_copy(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    assert _run(["unlock", bid])[0] == 0
    rc, out, err = _run(["unlock", bid])
    assert rc == 0
    assert out.strip() == f"already unlocked: {bid}"
    assert f"`aelf unlock --user {bid}`" in err


@pytest.mark.timeout(60)
def test_user_unlock_again_points_at_the_repo_copy(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    assert _run(["unlock", "--user", bid])[0] == 0
    rc, out, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert out.strip() == f"already unlocked: {bid}"
    assert f"`aelf unlock {bid}`" in err


# --- the other store is read, never swept, migrated, or created -----------


def _expire(db: Path, bid: str) -> None:
    """Put a lock past its window on disk, as no writable open has swept yet."""
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE beliefs SET lock_expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", bid),
        )
        conn.commit()
    finally:
        conn.close()


def _file_sig(db: Path) -> tuple[int, int]:
    st = db.stat()
    return (st.st_mtime_ns, st.st_size)


def _lock_level(db: Path, bid: str) -> str:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT lock_level FROM beliefs WHERE id = ?", (bid,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return str(row[0])


@pytest.mark.timeout(60)
def test_repo_unlock_leaves_an_expired_user_lock_unswept(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", "--user", "--for", "1w", STATEMENT])[0] == 0
    assert _run(["lock", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    _expire(user, bid)
    before = _file_sig(user)
    rc, out, err = _run(["unlock", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    # Expired, so it is not reported as still locked.
    assert err == ""
    assert _lock_level(user, bid) == "user"
    assert _file_sig(user) == before


@pytest.mark.timeout(60)
def test_user_unlock_leaves_an_expired_repo_lock_unswept(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    assert _run(["lock", "--for", "1w", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    _expire(repo, bid)
    before = _file_sig(repo)
    rc, out, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert out.strip() == f"unlocked: {bid}"
    assert err == ""
    assert _lock_level(repo, bid) == "user"
    assert _file_sig(repo) == before


@pytest.mark.timeout(60)
def test_plain_lock_leaves_an_expired_user_lock_unswept(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", "--user", "--for", "1w", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    _expire(user, bid)
    before = _file_sig(user)
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert "`aelf unlock" not in out
    assert err == ""
    assert _lock_level(user, bid) == "user"
    assert _file_sig(user) == before


@pytest.mark.timeout(60)
def test_lock_user_leaves_an_expired_repo_lock_unswept(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    assert _run(["lock", "--for", "1w", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    _expire(repo, bid)
    before = _file_sig(repo)
    rc, out, err = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert "`aelf unlock" not in out
    assert err == ""
    assert _lock_level(repo, bid) == "user"
    assert _file_sig(repo) == before


@pytest.mark.timeout(60)
def test_lock_with_one_file_for_both_scopes_says_nothing(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    assert _run(["lock", OTHER_STATEMENT])[0] == 0
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert "`aelf unlock" not in out
    assert err == ""


# --- an other-store path with URI metacharacters -------------------------


@pytest.mark.parametrize("char", ["#", "?", "%41"])
@pytest.mark.timeout(60)
def test_a_repo_store_path_with_uri_characters_is_read_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, char: str,
) -> None:
    base = tmp_path / "base"
    repo = base / f"R{char}x" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "u" / "memory.db"))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    assert _run(["lock", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    before = sorted(p.name for p in base.iterdir())
    rc, out, err = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    assert f"`aelf unlock {bid}`" in out
    assert err == ""
    rc, _, err = _run(["unlock", "--user", bid])
    assert rc == 0
    assert f"`aelf unlock {bid}`" in err
    assert sorted(p.name for p in base.iterdir()) == before


@pytest.mark.parametrize("char", ["#", "?", "%41"])
@pytest.mark.timeout(60)
def test_a_user_store_path_with_uri_characters_is_read_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, char: str,
) -> None:
    base = tmp_path / "base"
    user = base / f"U{char}x" / "memory.db"
    (tmp_path / "r").mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "r" / "memory.db"))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    before = sorted(p.name for p in base.iterdir())
    rc, out, err = _run(["lock", STATEMENT])
    assert rc == 0
    assert f"`aelf unlock --user {bid}`" in out
    assert err == ""
    rc, _, err = _run(["unlock", bid])
    assert rc == 0
    assert f"`aelf unlock --user {bid}`" in err
    assert sorted(p.name for p in base.iterdir()) == before


# --- one statement under two ids ------------------------------------------


@pytest.mark.timeout(120)
def test_a_corroborated_repo_lock_matches_the_user_copy_by_text(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    text = (
        "The deploy script must always run the database migrations "
        "before restarting the web workers."
    )
    (proj / "NOTES.md").write_text(f"# Notes\n\n{text}\n")
    assert _run(["onboard", str(proj)])[0] == 0
    rc, out, _ = _run(["lock", text])
    assert rc == 0
    assert "(corroborated existing)" in out
    assert _run(["lock", "--user", text])[0] == 0
    rid, uid = _ids("repo")[text], _ids("user")[text]
    assert rid != uid

    rc, out, err = _run(["locked"])
    assert rc == 0
    tagged = [ln for ln in out.splitlines() if "[also locked in the other scope]" in ln]
    assert sorted(ln.split(" ", 1)[0] for ln in tagged) == sorted([rid, uid])
    assert f"`aelf unlock {rid}`" in err
    assert f"`aelf unlock --user {uid}`" in err
    rows = json.loads(_run(["locked", "--json"])[1])
    assert [r["in_both_scopes"] for r in rows] == [True, True]

    rc, _, err = _run(["unlock", rid])
    assert rc == 0
    assert f"`aelf unlock --user {uid}`" in err
    assert _run(["lock", text])[0] == 0
    rc, _, err = _run(["unlock", "--user", uid])
    assert rc == 0
    assert f"`aelf unlock {rid}`" in err


# --- the warning's length and wording -------------------------------------


@pytest.mark.timeout(120)
def test_the_lock_warning_names_three_ids_and_counts_the_rest(
    stores: tuple[Path, Path],
) -> None:
    for i in range(5):
        assert _run(["lock", "--user", f"{STATEMENT[:-1]} v{i}."])[0] == 0
    rc, out, _ = _run(["lock", STATEMENT])
    assert rc == 0
    [warning] = [ln for ln in out.splitlines() if "warning:" in ln]
    assert re.search(
        r"\(([0-9a-f]{16}), ([0-9a-f]{16}), and ([0-9a-f]{16}) \(\+2 more\)\)",
        warning,
    )
    assert "those copies" in warning
    assert warning.count("`aelf unlock --user ") == 3
    assert ", and `aelf unlock --user " in warning


@pytest.mark.timeout(60)
def test_one_match_is_named_in_the_singular(stores: tuple[Path, Path]) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    rc, out, _ = _run(["lock", "--user", STATEMENT])
    assert rc == 0
    [warning] = [ln for ln in out.splitlines() if "warning:" in ln]
    assert "that copy" in warning
    assert "(+" not in warning


@pytest.mark.timeout(60)
def test_user_unlock_with_no_user_store_points_at_the_repo_copy(
    stores: tuple[Path, Path],
) -> None:
    _, user = stores
    assert _run(["lock", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    rc, _, err = _run(["unlock", "--user", bid])
    assert rc == 1
    assert "no user lock store" in err
    assert f"`aelf unlock {bid}`" in err
    assert not user.parent.exists()


@pytest.mark.parametrize("char", ["#", "?", "%41"])
@pytest.mark.timeout(60)
def test_doctor_reads_a_store_path_with_uri_characters_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, char: str,
) -> None:
    from aelfrice.doctor import diagnose_user_lock_duplicates

    base = tmp_path / "base"
    repo = base / f"R{char}x" / "memory.db"
    user = base / f"U{char}x" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", STATEMENT])[0] == 0
    bid = _lock_id(STATEMENT)
    before = sorted(p.name for p in base.iterdir())
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.known
    assert [(rid, uids) for rid, _, uids in report.pairs] == [(bid, (bid,))]
    assert sorted(p.name for p in base.iterdir()) == before


# --- the three commands agree with the L0 merge ---------------------------


SPACED = "Run uv sync first."
SPACED_APART = "Run uv\n\n\n\n\n\n            sync first."


@pytest.mark.timeout(60)
def test_text_the_merge_keeps_apart_is_not_called_one_statement(
    stores: tuple[Path, Path],
) -> None:
    from aelfrice.store import MemoryStore
    from aelfrice.user_locks import merge_lock_sets, read_user_locks

    repo, _ = stores
    assert _run(["lock", SPACED])[0] == 0
    rc, out, err = _run(["lock", "--user", SPACED_APART])
    assert rc == 0
    assert "`aelf unlock" not in out
    s = MemoryStore(str(repo))
    try:
        merged = merge_lock_sets(s.list_locked_beliefs(), read_user_locks(str(repo)))
    finally:
        s.close()
    # The merge injects both, so no command may say one hides the other.
    assert sorted(b.content for b in merged) == sorted([SPACED, SPACED_APART])
    rc, out, err = _run(["locked"])
    assert rc == 0
    assert "[also locked in the other scope]" not in out
    assert err == ""
    rid = _ids("repo")[SPACED]
    rc, _, err = _run(["unlock", rid])
    assert rc == 0
    assert err == ""


@pytest.mark.timeout(60)
def test_unlock_names_a_near_duplicate_the_merge_drops(
    stores: tuple[Path, Path],
) -> None:
    assert _run(["lock", STATEMENT])[0] == 0
    assert _run(["lock", "--user", NEAR_STATEMENT])[0] == 0
    rid = _ids("repo")[STATEMENT]
    uid = _ids("user")[NEAR_STATEMENT]
    rc, _, err = _run(["unlock", rid])
    assert rc == 0
    assert err == (
        f"  the user store still locks this statement ({uid}), and this "
        f"repository now injects that copy. To drop it too, run "
        f"`aelf unlock --user {uid}`.\n"
    )
    assert _run(["lock", STATEMENT])[0] == 0
    rc, _, err = _run(["unlock", "--user", uid])
    assert rc == 0
    assert err == (
        f"  the repository store still locks this statement ({rid}), and "
        f"this repository injects that copy. To drop it too, run "
        f"`aelf unlock {rid}`.\n"
    )
