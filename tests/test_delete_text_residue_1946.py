"""`aelf delete` keeps the deleted text out of the terminal and the log (#1946).

The confirmation prompt used to print the full belief text to stderr,
so deleting a lock that held a secret wrote the secret into terminals,
transcripts and logs. Without a terminal, `--force` alone stopped at
the prompt and never said to pass `--yes`. A hard delete also left the
text in `ingest_log.raw_text`. `delete` and `retire` now also take
`--user`, as `lock` and `unlock` do.

Every store here is a temp path: the conftest pins `HOME`,
`AELFRICE_DB` and `AELFRICE_USER_DB` at a sandbox, and the fixtures
below repoint both stores per test.
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
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.replay import replay_full_equality
from aelfrice.store import (
    INGEST_META_PURGED_AT,
    INGEST_META_PURGED_SHA256,
    MemoryStore,
)

SECRET = "The staging token is zq7-synthetic-marker-1946."
OTHER = "The build uses make for every target in this tree."


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


def _lock(statement: str, *extra: str) -> str:
    rc, out, err = _run(["lock", statement, *extra])
    assert rc == 0, err
    return out.split("locked:", 1)[1].split()[0]


def _log_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            "SELECT id, raw_text, raw_meta, derived_belief_ids FROM ingest_log"
        ))
    finally:
        conn.close()


def _rows_naming(db: Path, belief_id: str) -> list[sqlite3.Row]:
    return [
        r for r in _log_rows(db)
        if r["derived_belief_ids"] is not None
        and belief_id in json.loads(r["derived_belief_ids"])
    ]


def _belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash="h_" + bid, alpha=1.0,
        beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-11-20T00:00:00Z",
        last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_shared_row(db: Path, ids: list[str], text: str) -> str:
    """One log row that derived every belief in `ids`."""
    s = MemoryStore(str(db))
    try:
        for i, bid in enumerate(ids):
            s.insert_belief(_belief(bid, f"{text} part {i}"))
        return s.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM, raw_text=text,
            source_path="notes.md", raw_meta={"call_site": "test"},
            derived_belief_ids=ids,
        )
    finally:
        s.close()


# --- the prompt ------------------------------------------------------------


def test_the_prompt_names_the_id_and_never_prints_the_text(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid = _lock(SECRET)
    monkeypatch.setattr("builtins.input", lambda _prompt: "wrong")
    rc, _out, err = _run(["delete", bid, "--force"])
    assert rc == 1
    assert bid in err
    assert "zq7" not in err
    assert SECRET[:12] not in err


def test_the_prompt_hint_shows_a_hash_prefix_of_the_text(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid = _lock(SECRET)
    monkeypatch.setattr("builtins.input", lambda _prompt: "wrong")
    _rc, _out, err = _run(["delete", bid, "--force"])
    digest = hashlib.sha256(SECRET.encode("utf-8")).hexdigest()
    assert digest[:12] in err
    assert digest not in err


def test_eof_at_the_prompt_says_to_pass_yes(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid = _lock(SECRET)

    def _eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    rc, _out, err = _run(["delete", bid, "--force"])
    assert rc == 1
    assert "--yes" in err
    assert "zq7" not in err
    repo, _user = stores
    assert _rows_naming(repo, bid)[0]["raw_text"] == SECRET


# --- the ingest log --------------------------------------------------------


def test_a_hard_delete_clears_the_text_from_its_log_rows(
    stores: tuple[Path, Path],
) -> None:
    repo, _user = stores
    bid = _lock(SECRET)
    assert _rows_naming(repo, bid)
    rc, _out, err = _run(["delete", bid, "--force", "--yes"])
    assert rc == 0, err
    rows = _rows_naming(repo, bid)
    assert rows
    for r in rows:
        assert r["raw_text"] == ""
        assert "zq7" not in (r["raw_meta"] or "")
    assert all("zq7" not in r["raw_text"] for r in _log_rows(repo))


def test_the_cleared_row_keeps_only_a_hash_and_a_time(
    stores: tuple[Path, Path],
) -> None:
    """`lock_gaps` matches a deleted lock to its statement through the hash."""
    repo, _user = stores
    bid = _lock(SECRET)
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    meta = json.loads(_rows_naming(repo, bid)[0]["raw_meta"])
    assert set(meta) == {INGEST_META_PURGED_AT, INGEST_META_PURGED_SHA256}
    assert meta[INGEST_META_PURGED_SHA256] == hashlib.sha256(
        SECRET.encode("utf-8")
    ).hexdigest()


def test_a_row_that_still_derives_a_live_belief_keeps_its_text(
    stores: tuple[Path, Path],
) -> None:
    repo, _user = stores
    log_id = _seed_shared_row(repo, ["aaaa000000001946", "bbbb000000001946"], OTHER)
    assert _run(["delete", "aaaa000000001946", "--yes"])[0] == 0
    row = next(r for r in _log_rows(repo) if r["id"] == log_id)
    assert row["raw_text"] == OTHER
    assert json.loads(row["raw_meta"]) == {"call_site": "test"}
    assert _run(["delete", "bbbb000000001946", "--yes"])[0] == 0
    row = next(r for r in _log_rows(repo) if r["id"] == log_id)
    assert row["raw_text"] == ""


def test_a_row_that_derives_a_retired_belief_keeps_its_text(
    stores: tuple[Path, Path],
) -> None:
    """A retired belief still holds its text, and `restore` can bring it back."""
    repo, _user = stores
    log_id = _seed_shared_row(repo, ["cccc000000001946", "dddd000000001946"], OTHER)
    assert _run(["retire", "dddd000000001946"])[0] == 0
    assert _run(["delete", "cccc000000001946", "--yes"])[0] == 0
    row = next(r for r in _log_rows(repo) if r["id"] == log_id)
    assert row["raw_text"] == OTHER


def test_rows_of_other_beliefs_are_untouched(stores: tuple[Path, Path]) -> None:
    repo, _user = stores
    keep = _lock(OTHER)
    bid = _lock(SECRET)
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    assert [r["raw_text"] for r in _rows_naming(repo, keep)] == [OTHER]


def test_a_second_delete_of_the_same_text_keeps_the_first_hash(
    stores: tuple[Path, Path],
) -> None:
    """Locking the text again writes a new row; clearing it again is safe."""
    repo, _user = stores
    bid = _lock(SECRET)
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    assert _lock(SECRET) == bid
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    digest = hashlib.sha256(SECRET.encode("utf-8")).hexdigest()
    rows = _rows_naming(repo, bid)
    assert len(rows) == 2
    for r in rows:
        assert r["raw_text"] == ""
        assert json.loads(r["raw_meta"])[INGEST_META_PURGED_SHA256] == digest


# --- replay ----------------------------------------------------------------


def test_replay_skips_cleared_rows_instead_of_reporting_drift(
    stores: tuple[Path, Path],
) -> None:
    repo, _user = stores
    _lock(OTHER)
    bid = _lock(SECRET)
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    s = MemoryStore(str(repo))
    try:
        report = replay_full_equality(s)
    finally:
        s.close()
    assert report.excluded_purged == 1
    assert report.derived_orphan == 0
    assert not report.has_drift
    assert "zq7" not in json.dumps(report.drift_examples)


def test_doctor_replay_prints_the_cleared_row_count(
    stores: tuple[Path, Path],
) -> None:
    bid = _lock(SECRET)
    assert _run(["delete", bid, "--force", "--yes"])[0] == 0
    rc, out, _err = _run(["doctor", "--replay"])
    assert rc == 0, out
    assert "excluded (deleted):        1" in out
    assert "zq7" not in out


# --- --user ----------------------------------------------------------------


def test_delete_user_removes_the_user_lock_and_clears_its_text(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _lock(OTHER)
    bid = _lock(SECRET, "--user")
    rc, out, err = _run(["delete", "--user", bid, "--force", "--yes"])
    assert rc == 0, err
    assert f"deleted: {bid}" in out
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE id = ?", (bid,),
        ).fetchone() == (0,)
    finally:
        conn.close()
    assert all("zq7" not in r["raw_text"] for r in _log_rows(user))
    assert len(_log_rows(repo)) == 1


def test_plain_delete_does_not_reach_the_user_store(
    stores: tuple[Path, Path],
) -> None:
    _repo, user = stores
    _lock(OTHER)
    bid = _lock(SECRET, "--user")
    rc, _out, err = _run(["delete", bid, "--force", "--yes"])
    assert rc == 1
    assert "not found" in err
    assert _rows_naming(user, bid)[0]["raw_text"] == SECRET


def test_retire_user_retires_the_user_lock(stores: tuple[Path, Path]) -> None:
    _repo, user = stores
    bid = _lock(SECRET, "--user")
    rc, out, err = _run(["retire", "--user", bid, "--force"])
    assert rc == 0, err
    assert f"retired: {bid}" in out
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    try:
        (valid_to,) = conn.execute(
            "SELECT valid_to FROM beliefs WHERE id = ?", (bid,),
        ).fetchone()
    finally:
        conn.close()
    assert valid_to is not None


@pytest.mark.parametrize("verb", ["delete", "retire"])
def test_user_scope_with_no_user_store_fails_and_creates_nothing(
    stores: tuple[Path, Path], verb: str,
) -> None:
    _repo, user = stores
    rc, _out, err = _run([verb, "--user", "aaaa000000001946", "--force", "--yes"]
                         if verb == "delete"
                         else [verb, "--user", "aaaa000000001946", "--force"])
    assert rc == 1
    assert "no user lock store" in err
    assert not user.exists()


@pytest.mark.parametrize("verb", ["delete", "retire"])
def test_user_scope_naming_the_repo_store_is_refused(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    repo, _user = stores
    bid = _lock(SECRET)
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    argv = [verb, "--user", bid, "--force"]
    if verb == "delete":
        argv.append("--yes")
    rc, _out, err = _run(argv)
    assert rc == 1
    assert "--user refused" in err
    assert _rows_naming(repo, bid)[0]["raw_text"] == SECRET


def test_a_row_naming_a_longer_id_that_contains_the_deleted_one_is_kept(
    stores: tuple[Path, Path],
) -> None:
    """The scan narrows by substring; only a row that names the id is cleared."""
    repo, _user = stores
    s = MemoryStore(str(repo))
    try:
        s.insert_belief(_belief("eeee1946", OTHER))
        log_id = s.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM, raw_text=SECRET,
            source_path="notes.md", derived_belief_ids=["eeee1946ffff0000"],
        )
    finally:
        s.close()
    assert _run(["delete", "eeee1946", "--yes"])[0] == 0
    row = next(r for r in _log_rows(repo) if r["id"] == log_id)
    assert row["raw_text"] == SECRET


def test_lock_gaps_spells_the_hash_key_as_the_store_does() -> None:
    from aelfrice import lock_gaps

    assert lock_gaps._INGEST_META_PURGED_SHA256 == INGEST_META_PURGED_SHA256  # pyright: ignore[reportPrivateUsage]
