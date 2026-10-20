"""Moving locks between scopes, and finding the overlap (#1681, PR3 of 3).

Operator ruling Q4: `aelf lock --promote-user <id>` copies a repository
lock into the user store, and `aelf doctor` lists the repository locks
that near-duplicate a user lock, so the user can unlock them by hand.
Nothing merges automatically. These tests also pin that `lock_gaps`
judges a failed `/aelf:lock --user` against the user store, and that
`aelf core` and `aelf search --show-conflicts` include the user locks.

Every path is a temp path. The conftest pins `AELFRICE_DB` and
`AELFRICE_USER_DB` at a sandbox; each test repoints both.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import json
import os
import shlex
import sqlite3
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest

from aelfrice import cli
from aelfrice import user_locks as ul
from aelfrice.doctor import (
    DoctorReport,
    UserLockDuplicateReport,
    diagnose_lock_gaps,
    diagnose_user_lock_duplicates,
    format_report,
)
from aelfrice.hook import build_lock_gap_notice, user_prompt_submit
from aelfrice.hook_audit import command_outcomes_path_for_db
from aelfrice.lock_gaps import LockGap, detect_lock_gaps
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

REPO_RULE = "This repository builds with make, not cmake."
USER_RULE = "Route all questions and decisions through the question tool."
# Near-duplicate of REPO_RULE: same tokens, one character apart.
NEAR_DUP = "This repository builds with make, not cmake!"
# Shares most words with REPO_RULE but stays under the thresholds.
SIMILAR = "This repository builds the docs with sphinx, not mkdocs."
STATEMENT = "Keep every widget in the blue drawer."
_REAL_CLI_MAIN = cli.main


def _lock(
    bid: str,
    content: str,
    *,
    locked_at: str = "2026-10-01T00:00:00+00:00",
    expires: str | None = None,
    tier: str = "frozen",
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        alpha=9.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at=locked_at,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
        lock_expires_at=expires,
        lock_tier=tier,
    )


def _plain(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _seed(path: Path, beliefs: list[Belief]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(str(path))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _locks(path: Path) -> list[tuple[str, str, str, str | None]]:
    """`(id, content, lock_tier, lock_expires_at)` of the locks at `path`."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, content, lock_tier, lock_expires_at FROM beliefs "
            "WHERE lock_level = 'user' AND valid_to IS NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [(str(a), str(b), str(c), d) for a, b, c, d in rows]


@pytest.fixture()
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(repo_db, user_db), both temp paths, neither created yet."""
    repo = tmp_path / "repo" / "memory.db"
    user = tmp_path / "home" / ".aelfrice" / "user" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.setenv("AELF_SESSION_ID", "s1681")
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    return repo, user


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]


def _cli(
    argv: list[str], capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    out = io.StringIO()
    rc = cli.main(argv, out=out)
    return rc, out.getvalue(), capsys.readouterr().err


def _feed_rows(repo: Path) -> list[dict[str, object]]:
    path = repo.parent / "feed.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(ln) for ln in path.read_text("utf-8").splitlines()
        if ln.strip()
    ]


# --- aelf lock --promote-user ---------------------------------------------------


@pytest.mark.timeout(60)
def test_promote_copies_the_lock_and_keeps_the_repo_lock(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    before = _locks(repo)
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0, out
    assert [c for _, c, _, _ in _locks(user)] == [REPO_RULE]
    assert _locks(repo) == before
    assert "aelf unlock r1" in out


@pytest.mark.timeout(60)
def test_promote_keeps_the_tier_and_the_expiry(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    expires = "2099-01-02T03:04:05+00:00"
    _seed(repo, [_lock("r1", REPO_RULE, expires=expires, tier="reference")])
    rc, _, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    [(_, content, tier, exp)] = _locks(user)
    assert (content, tier, exp) == (REPO_RULE, "reference", expires)


@pytest.mark.timeout(60)
def test_promote_records_a_user_scoped_feed_event(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    [(uid, _, _, _)] = _locks(user)
    [row] = [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"]
    assert row["scope"] == "user"
    assert row["promoted_from"] == "r1"
    assert row["id"] == uid


@pytest.mark.timeout(60)
def test_promote_an_unknown_id_exits_1_and_creates_nothing(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    rc, _, err = _cli(["lock", "--promote-user", "nope"], capsys)
    assert rc == 1
    assert "no belief with id 'nope'" in err
    assert not user.exists()


@pytest.mark.timeout(60)
def test_promote_takes_an_exact_id_only(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1abcdef", REPO_RULE)])
    rc, _, err = _cli(["lock", "--promote-user", "r1abc"], capsys)
    assert rc == 1
    assert "no belief with id 'r1abc'" in err
    assert not user.exists()


@pytest.mark.timeout(60)
def test_promote_a_belief_that_is_not_a_lock_exits_1(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_plain("p1", REPO_RULE)])
    rc, _, err = _cli(["lock", "--promote-user", "p1"], capsys)
    assert rc == 1
    assert "p1 is not a repository lock" in err
    assert not user.exists()


@pytest.mark.timeout(60)
def test_promote_refuses_a_lock_past_its_window_on_an_unswept_handle(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only handle runs no sweep, so the row still reads
    `lock_level = 'user'` after its window closed."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE, expires="2026-01-01T00:00:00+00:00")])
    # Re-mark it locked after the seeding open swept it.
    conn = sqlite3.connect(str(repo))
    conn.execute("UPDATE beliefs SET lock_level = 'user' WHERE id = 'r1'")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        cli, "_open_store",
        lambda: MemoryStore(str(repo), read_only=True),
    )
    rc, _, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1
    assert "r1 is not a repository lock" in err
    assert not user.exists()


@pytest.mark.timeout(60)
def test_promote_twice_adds_no_second_copy(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    before = _locks(user)
    n_feed = len(_feed_rows(repo))
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    assert "not promoted" in out
    assert _locks(user) == before
    assert len(_feed_rows(repo)) == n_feed


@pytest.mark.timeout(60)
def test_promote_with_the_same_id_in_the_user_store_adds_nothing(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("r1", "Different wording held under the same id.")])
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    assert "has a lock with the same id" in out
    assert [i for i, _, _, _ in _locks(user)] == ["r1"]


@pytest.mark.timeout(60)
def test_promote_skips_a_near_duplicate_user_lock(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    assert "near-duplicate lock: u1" in out
    assert [i for i, _, _, _ in _locks(user)] == ["u1"]


@pytest.mark.timeout(60)
def test_promote_adds_beside_a_user_lock_that_is_only_similar(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", SIMILAR)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    assert sorted(c for _, c, _, _ in _locks(user)) == sorted([SIMILAR, REPO_RULE])


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "flags",
    [
        ["--reference"], ["--frozen"], ["--for", "3d"],
        ["--until", "2099-01-01"], ["--doc", "file:///x"],
        ["--category", "c"],
    ],
    ids=["reference", "frozen", "for", "until", "doc", "category"],
)
def test_promote_refuses_a_flag_the_copy_cannot_honour(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    flags: list[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    rc, _, err = _cli(["lock", "--promote-user", "r1", *flags], capsys)
    assert rc == 1
    assert "cannot be used with --promote-user" in err
    assert not user.exists()


@pytest.mark.timeout(60)
def test_promote_refuses_when_the_user_store_is_the_repo_store(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    monkeypatch.setenv("AELFRICE_USER_DB", str(repo))
    before = _locks(repo)
    rc, _, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1
    assert "--user refused" in err
    assert _locks(repo) == before


@pytest.mark.timeout(60)
def test_promote_refuses_when_the_user_store_cannot_be_read(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    user.parent.mkdir(parents=True)
    user.write_bytes(b"not a sqlite file" * 100)
    rc, _, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1
    assert "cannot read the user lock store" in err
    assert user.read_bytes() == b"not a sqlite file" * 100


@pytest.mark.timeout(60)
def test_a_plain_lock_is_unchanged_by_the_new_flag(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --promote-user the positional is still the statement."""
    repo, user = stores
    rc, _, _ = _cli(["lock", REPO_RULE], capsys)
    assert rc == 0
    assert [c for _, c, _, _ in _locks(repo)] == [REPO_RULE]
    assert not user.exists()


# --- aelf doctor: repository locks that near-duplicate a user lock --------------


@pytest.mark.timeout(60)
def test_doctor_lists_a_repo_lock_that_near_duplicates_a_user_lock(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE), _lock("r2", USER_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.known and report.user_store_present
    assert report.pairs == (("r1", REPO_RULE, ("u1",)),)
    text = format_report(DoctorReport(user_lock_duplicates=report))
    assert "repository locks that near-duplicate a user lock:" in text
    assert "  - r1: " + REPO_RULE in text
    assert "      user lock: u1" in text
    assert "      fix: aelf unlock r1" in text
    assert "r2" not in text


@pytest.mark.timeout(60)
def test_doctor_lists_the_same_statement_locked_in_both_scopes(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("same", REPO_RULE)])
    _seed(user, [_lock("same", REPO_RULE)])
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.pairs == (("same", REPO_RULE, ("same",)),)


@pytest.mark.timeout(60)
def test_doctor_does_not_list_a_lock_that_is_only_similar(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", SIMILAR)])
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.known and report.pairs == ()
    text = format_report(DoctorReport(user_lock_duplicates=report))
    assert f"  none (user lock store: {user})" in text


@pytest.mark.timeout(60)
def test_doctor_lists_what_the_merge_drops(stores: tuple[Path, Path]) -> None:
    """The listing and the L0 merge agree on every pair."""
    repo, user = stores
    repo_locks = [_lock("r1", REPO_RULE), _lock("r2", USER_RULE)]
    # "r2" shares only its id with a repository lock, not its text.
    user_locks = [
        _lock("u1", NEAR_DUP), _lock("u2", SIMILAR),
        _lock("r2", "A different statement held under a repository id."),
    ]
    _seed(repo, repo_locks)
    _seed(user, user_locks)
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    listed = {u for _, _, uids in report.pairs for u in uids}
    kept = {b.id for b in ul.merge_lock_sets(repo_locks, user_locks)}
    dropped = {b.id for b in user_locks} - (kept - {"r1", "r2"})
    assert listed == dropped == {"u1", "r2"}


@pytest.mark.timeout(60)
def test_doctor_lists_nothing_and_creates_nothing_without_a_user_store(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.known and not report.user_store_present
    assert report.pairs == ()
    text = format_report(DoctorReport(user_lock_duplicates=report))
    assert f"  none (no user lock store at {user})" in text
    assert not user.exists()
    assert not user.parent.exists()


@pytest.mark.timeout(60)
def test_doctor_reads_the_user_store_without_writing_it(
    stores: tuple[Path, Path],
) -> None:
    """A writable open would sweep the expired user lock to 'none'."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    conn = sqlite3.connect(str(user))
    conn.execute(
        "UPDATE beliefs SET lock_expires_at = '2026-01-01T00:00:00+00:00' "
        "WHERE id = 'u1'"
    )
    conn.commit()
    conn.close()
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    # Expired: not a lock the merge would see, so not listed either.
    assert report.pairs == ()
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    [(level,)] = conn.execute(
        "SELECT lock_level FROM beliefs WHERE id = 'u1'"
    ).fetchall()
    conn.close()
    assert level == LOCK_USER


@pytest.mark.timeout(60)
def test_doctor_lists_nothing_when_the_user_store_is_the_repo_store(
    stores: tuple[Path, Path],
) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    report = diagnose_user_lock_duplicates(str(repo), str(repo))
    assert report.known and report.pairs == ()


@pytest.mark.timeout(60)
def test_doctor_says_unknown_when_the_user_store_cannot_be_read(
    stores: tuple[Path, Path],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    user.parent.mkdir(parents=True)
    user.write_bytes(b"not a sqlite file" * 100)
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert not report.known
    text = format_report(DoctorReport(user_lock_duplicates=report))
    assert "  unknown: the user lock store could not be read" in text


@pytest.mark.timeout(60)
@pytest.mark.parametrize("scanned", [False, True], ids=["no-settings", "settings"])
def test_both_report_paths_render_the_duplicate_section(
    tmp_path: Path, scanned: bool,
) -> None:
    report = DoctorReport(
        user_lock_duplicates=UserLockDuplicateReport(
            known=True, user_store="/u.db", user_store_present=True,
            pairs=(("r1", REPO_RULE, ("u1",)),),
        ),
    )
    if scanned:
        settings = tmp_path / "settings.json"
        settings.write_text("{}", encoding="utf-8")
        report.scopes_scanned.append(("user", settings))
    assert "fix: aelf unlock r1" in format_report(report)


@pytest.mark.timeout(120)
def test_aelf_doctor_prints_the_duplicate_section_and_exits_0(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command itself, so the wiring through `diagnose` is covered;
    a listed pair is informational and doesn't fail the run."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    rc = cli.main(["doctor", "hooks"], out=buf)
    text = buf.getvalue() + capsys.readouterr().out
    assert "      fix: aelf unlock r1" in text
    assert rc == 0


# --- lock_gaps reads the user store ---------------------------------------------


def _boom(*_a: object, **_k: object) -> int:
    raise RuntimeError("synthetic store fault")


def _fail_typed_lock(
    prompt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "main", _boom)
    payload = json.dumps(
        {"prompt": prompt, "session_id": "s1681", "cwd": str(tmp_path)}
    )
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=io.StringIO(),
        stderr=io.StringIO(),
    ) == 0
    monkeypatch.setattr(cli, "main", _REAL_CLI_MAIN)


def _backdate(repo: Path, ts: str = "2026-01-01T00:00:00Z") -> None:
    path = command_outcomes_path_for_db(repo)
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines()
            if ln.strip()]
    path.write_text(
        "".join(json.dumps({**r, "ts": ts}) + "\n" for r in rows),
        encoding="utf-8",
    )


@pytest.mark.timeout(120)
def test_a_failed_user_lock_is_listed_with_a_user_fix(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    [gap] = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps
    assert gap.user_scope
    assert gap.fix_command == f"aelf lock --user {shlex.quote(STATEMENT)}"
    assert not user.exists()


@pytest.mark.timeout(120)
def test_a_failed_user_lock_closes_when_the_user_store_holds_it(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    assert _cli(["lock", "--user", STATEMENT], capsys)[0] == 0
    assert detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps == ()


@pytest.mark.timeout(120)
def test_a_failed_user_lock_stays_open_when_only_the_repo_holds_it(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    assert _cli(["lock", STATEMENT], capsys)[0] == 0
    [gap] = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps
    assert gap.user_scope


@pytest.mark.timeout(120)
def test_a_user_unlock_after_the_failure_closes_the_gap(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The removal rules apply to the user store's own audit rows."""
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    _backdate(repo)
    rc, out, _ = _cli(["lock", "--user", STATEMENT], capsys)
    assert rc == 0
    bid = out.split("locked: ", 1)[1].split()[0]
    assert _cli(["unlock", "--user", bid], capsys)[0] == 0
    assert detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps == ()


@pytest.mark.timeout(120)
def test_one_statement_failing_in_both_scopes_is_two_gaps(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock {STATEMENT}", tmp_path, monkeypatch)
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    gaps = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps
    assert sorted(g.scope for g in gaps) == ["repo", "user"]
    # Applying it in one scope closes that scope's gap only.
    assert _cli(["lock", STATEMENT], capsys)[0] == 0
    [gap] = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    ).gaps
    assert gap.user_scope


@pytest.mark.timeout(120)
def test_lock_gaps_defaults_to_the_configured_user_store(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    assert len(detect_lock_gaps(str(repo), audit_enabled=True).gaps) == 1
    assert _cli(["lock", "--user", STATEMENT], capsys)[0] == 0
    assert detect_lock_gaps(str(repo), audit_enabled=True).gaps == ()


@pytest.mark.timeout(120)
def test_an_unreadable_user_store_makes_the_answer_unknown(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    user.parent.mkdir(parents=True)
    user.write_bytes(b"not a sqlite file" * 100)
    report = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=str(user),
    )
    assert not report.known
    assert report.unknown_reason is not None
    assert report.unknown_reason.startswith("the user lock store could not be read")


@pytest.mark.timeout(120)
def test_doctor_and_session_start_list_a_failed_user_lock(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    text = format_report(
        DoctorReport(lock_gaps=diagnose_lock_gaps(str(repo), tmp_path)),
    )
    assert "(exception, user scope): " + STATEMENT in text
    assert f"fix: aelf lock --user {shlex.quote(STATEMENT)}" in text
    notice = build_lock_gap_notice(stderr=io.StringIO())
    assert notice is not None and "1 /aelf:lock request failed" in notice


@pytest.mark.timeout(60)
def test_a_truncated_user_gap_names_the_user_command(tmp_path: Path) -> None:
    from aelfrice.lock_gaps import LockGapReport

    gap = LockGap("0" * 64, "Keep", 700, "exception",
                  "2026-01-01T00:00:00Z", None, 1, scope="user")
    text = format_report(
        DoctorReport(lock_gaps=LockGapReport(known=True, gaps=(gap,))),
    )
    assert "run `aelf lock --user` with the full statement" in text


# --- aelf core and aelf search --show-conflicts include the user locks ----------


@pytest.mark.timeout(60)
def test_core_lists_the_user_lock_tagged_after_the_repo_locks(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    rc, out, _ = _cli(["core", "--locked-only"], capsys)
    assert rc == 0
    lines = out.splitlines()
    assert lines[0].startswith("r1 [LOCK")
    assert lines[1].startswith("u1 [user] [LOCK")
    assert lines[1].endswith(USER_RULE)
    rc, out, _ = _cli(["core", "--locked-only", "--json"], capsys)
    rows = {r["id"]: r for r in json.loads(out)}
    assert rows["u1"]["scope"] == "user"
    assert "scope" not in rows["r1"]


@pytest.mark.timeout(60)
def test_core_drops_a_user_lock_a_repo_lock_near_duplicates(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP), _lock("u2", USER_RULE)])
    rc, out, _ = _cli(["core", "--locked-only", "--json"], capsys)
    assert rc == 0
    assert [r["id"] for r in json.loads(out)] == ["r1", "u2"]


@pytest.mark.timeout(60)
def test_core_without_locks_leaves_the_user_locks_out(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    rc, out, _ = _cli(["core", "--no-locked"], capsys)
    assert rc == 0
    assert "u1" not in out


@pytest.mark.timeout(60)
def test_core_never_creates_or_writes_the_user_store(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    assert _cli(["core", "--locked-only"], capsys)[0] == 0
    assert not user.exists()
    _seed(user, [_lock("u2", "Expired rule about tabs.",
                       expires="2026-01-01T00:00:00+00:00")])
    conn = sqlite3.connect(str(user))
    conn.execute("UPDATE beliefs SET lock_level = 'user' WHERE id = 'u2'")
    conn.commit()
    conn.close()
    rc, out, _ = _cli(["core", "--locked-only"], capsys)
    assert rc == 0
    assert "u2" not in out
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    [(level,)] = conn.execute(
        "SELECT lock_level FROM beliefs WHERE id = 'u2'"
    ).fetchall()
    conn.close()
    assert level == LOCK_USER


@pytest.mark.timeout(60)
def test_show_conflicts_checks_hits_against_the_user_locks(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, [_plain("p1", "The default server port is 9090.")])
    _seed(user, [_lock("u1", "The default server port is 8080.")])
    monkeypatch.setenv("AELF_SHOW_CONFLICTS", "1")
    rc, out, _ = _cli(["search", "server port"], capsys)
    assert rc == 0
    assert "[!] p1: The default server port is 9090." in out
    rc, out, _ = _cli(["search", "server port", "--json"], capsys)
    rows = {r["id"]: r for r in json.loads(out)}
    assert rows["p1"]["slot_conflict_with"] == "u1"
    assert rows["p1"]["slot_conflict_scope"] == "user"


@pytest.mark.timeout(60)
def test_show_conflicts_leaves_a_repo_conflict_row_unchanged(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = stores
    _seed(repo, [
        _plain("p1", "The default server port is 9090."),
        _lock("r1", "The default server port is 8080."),
    ])
    monkeypatch.setenv("AELF_SHOW_CONFLICTS", "1")
    rc, out, _ = _cli(["search", "server port", "--json"], capsys)
    assert rc == 0
    rows = {r["id"]: r for r in json.loads(out)}
    assert rows["p1"]["slot_conflict_with"] == "r1"
    assert "slot_conflict_scope" not in rows["p1"]


@pytest.mark.timeout(60)
def test_core_on_an_unreadable_user_store_warns_and_lists_repo_locks(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    user.parent.mkdir(parents=True)
    user.write_bytes(b"not a sqlite file" * 100)
    rc, out, err = _cli(["core", "--locked-only"], capsys)
    assert rc == 0
    assert out.startswith("r1 [LOCK")
    assert "listing repository locks only" in err
    assert os.path.getsize(user) == len(b"not a sqlite file" * 100)


# --- review round: one transaction, the expiry filter, open failures ------------


def _count(path: Path, table: str) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


@pytest.mark.timeout(60)
def test_a_failure_at_the_tier_and_expiry_step_leaves_nothing_and_a_retry_works(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy commits row, tier, and expiry together or not at all."""
    repo, user = stores
    expires = "2099-01-02T03:04:05+00:00"
    _seed(repo, [_lock("r1", REPO_RULE, expires=expires, tier="reference")])
    real_update = MemoryStore.update_belief

    def failing_update(self: MemoryStore, b: Belief) -> None:
        if b.lock_expires_at == expires:
            raise sqlite3.OperationalError("database is locked")
        real_update(self, b)

    monkeypatch.setattr(MemoryStore, "update_belief", failing_update)
    rc, out, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1
    assert out == ""
    assert err.count("\n") == 1
    assert "the copy failed: database is locked" in err
    # No user store existed, so none is left behind, nor a temporary file.
    assert not user.exists()
    assert sorted(f.name for f in user.parent.iterdir()) == []
    assert [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"] == []

    monkeypatch.setattr(MemoryStore, "update_belief", real_update)
    rc, _, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    [(_, content, tier, exp)] = _locks(user)
    assert (content, tier, exp) == (REPO_RULE, "reference", expires)
    [row] = [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"]
    assert row["promoted_from"] == "r1"


@pytest.mark.timeout(60)
def test_an_expired_unswept_near_duplicate_does_not_block_a_promote(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """The user store is read without the sweep, so the window is checked."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    # Inserted after the seeding open's sweep, so still 'user' on disk.
    _seed(user, [_lock("u1", NEAR_DUP, expires="2026-01-01T00:00:00+00:00")])
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0, out
    assert "not promoted" not in out
    assert [c for _, c, _, _ in _locks(user)] == [REPO_RULE]


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "argv",
    [["lock", "--promote-user", "r1"], ["lock", "--user", USER_RULE]],
    ids=["promote", "lock-user"],
)
def test_an_unopenable_user_store_prints_one_line_and_exits_1(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str],
) -> None:
    repo, _ = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("AELFRICE_USER_DB", str(blocker / "user" / "memory.db"))
    rc, out, err = _cli(argv, capsys)
    assert rc == 1
    assert ("the copy failed" if "--promote-user" in argv
            else "cannot open the user lock store") in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    if "--promote-user" in argv:
        assert out == ""


# --- feedback and confirm on an id both stores hold (ruling 2026-10-19) ---------


def _feedback_rows(path: Path, bid: str) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM feedback_history WHERE belief_id = ?", (bid,),
        ).fetchone()[0])
    finally:
        conn.close()


def _alpha(path: Path, bid: str) -> float:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return float(conn.execute(
            "SELECT alpha FROM beliefs WHERE id = ?", (bid,),
        ).fetchone()[0])
    finally:
        conn.close()


@pytest.mark.timeout(60)
def test_feedback_and_confirm_apply_to_a_repo_lock_that_shares_a_user_lock_id(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """The repository copy is the one this repository injects."""
    repo, user = stores
    _seed(repo, [_lock("same", REPO_RULE)])
    _seed(user, [_lock("same", REPO_RULE)])
    before_user = user.read_bytes()
    rc, _, err = _cli(["feedback", "same", "used"], capsys)
    assert rc == 0, err
    assert "applies to it" in err
    assert _feedback_rows(repo, "same") == 1
    alpha = _alpha(repo, "same")
    rc, _, err = _cli(["confirm", "same"], capsys)
    assert rc == 0, err
    assert _alpha(repo, "same") == alpha + 1.0
    assert user.read_bytes() == before_user


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "command", [["feedback", "same", "used"], ["confirm", "same"]],
    ids=["feedback", "confirm"],
)
def test_feedback_and_confirm_refuse_when_the_repo_row_is_not_a_lock(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    repo, user = stores
    _seed(repo, [_plain("same", REPO_RULE)])
    _seed(user, [_lock("same", REPO_RULE)])
    before_user = user.read_bytes()
    rc, _, err = _cli(command, capsys)
    assert rc == 1
    assert "is a user-scope lock" in err
    assert _feedback_rows(repo, "same") == 0
    assert _alpha(repo, "same") == 1.0
    assert user.read_bytes() == before_user


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "command", [["feedback", "u1", "used"], ["confirm", "u1"]],
    ids=["feedback", "confirm"],
)
def test_feedback_and_confirm_refuse_a_user_store_only_id(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    before_user = user.read_bytes()
    rc, _, err = _cli(command, capsys)
    assert rc == 1
    assert "is a user-scope lock" in err
    assert user.read_bytes() == before_user


@pytest.mark.timeout(60)
def test_sentiment_credits_a_repo_lock_that_won_over_a_same_id_user_lock(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repository lock is injected, so its audit row is a repository
    row and the sentiment lane records feedback on it as before."""
    from typing import cast

    from aelfrice.hook import (
        AUDIT_HOOK_USER_PROMPT_SUBMIT,
        _audit_path_for_db,  # pyright: ignore[reportPrivateUsage]
        read_hook_audit,
    )
    from aelfrice.sentiment_feedback import ENV_SENTIMENT

    monkeypatch.setenv(ENV_SENTIMENT, "1")
    repo, user = stores
    _seed(repo, [_lock("same", USER_RULE)])
    _seed(user, [_lock("same", USER_RULE)])
    for prompt in ("which question tool do decisions use", "perfect, thanks"):
        payload = json.dumps({
            "prompt": prompt, "session_id": "senti1681b", "cwd": str(tmp_path),
        })
        assert user_prompt_submit(
            stdin=io.StringIO(payload), stdout=io.StringIO(),
            stderr=io.StringIO(),
        ) == 0
    rows = [
        r for r in read_hook_audit(_audit_path_for_db(repo))
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    first = cast("list[dict[str, object]]", rows[0]["beliefs"])
    [same] = [b for b in first if b["id"] == "same"]
    assert "scope" not in same
    assert _feedback_rows(repo, "same") >= 1


@pytest.mark.timeout(60)
def test_core_shows_no_gate_label_on_a_user_lock() -> None:
    """Labels are keyed by repository content hashes; a user lock with the
    same text must not show a repository belief's label."""
    import argparse

    user_lock = _lock("u1", USER_RULE)
    setattr(user_lock, ul.USER_SCOPE_ATTR, ul.USER_SCOPE_VALUE)
    repo_lock = _lock("r1", REPO_RULE)
    labels = {user_lock.content_hash: "A", repo_lock.content_hash: "B"}
    args = argparse.Namespace(
        json=True, limit=None, min_corroboration=2,
        min_posterior=2.0 / 3.0, min_alpha_beta=4,
    )
    out = io.StringIO()
    cli._emit_core(  # pyright: ignore[reportPrivateUsage]
        [repo_lock, user_lock], [], args, out, {}, labels,
    )
    rows = {r["id"]: r for r in json.loads(out.getvalue())}
    assert rows["r1"]["core_gate_label"] == "B"
    assert "core_gate_label" not in rows["u1"]


# --- second review: existing store, unsearchable dirs, untested lines ----------


@pytest.mark.timeout(60)
def test_a_failed_promote_into_an_existing_user_store_writes_nothing(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    expires = "2099-01-02T03:04:05+00:00"
    _seed(repo, [_lock("r1", REPO_RULE, expires=expires)])
    _seed(user, [_lock("u1", USER_RULE)])
    before = (_count(user, "beliefs"), _count(user, "ingest_log"))
    real_update = MemoryStore.update_belief

    def failing_update(self: MemoryStore, b: Belief) -> None:
        if b.lock_expires_at == expires:
            raise sqlite3.OperationalError("database is locked")
        real_update(self, b)

    monkeypatch.setattr(MemoryStore, "update_belief", failing_update)
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1 and out == ""
    assert (_count(user, "beliefs"), _count(user, "ingest_log")) == before


@contextlib.contextmanager
def _chmod0_parent(
    path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Take search permission off `path`'s directory.

    `Path.exists()` and `Path.is_file()` are also made to return False
    for anything under that directory, which is what Python 3.14 does
    (3.12 and 3.13 raise), so a revert to either fails on every version.
    """
    locked = os.path.abspath(path.parent) + os.sep
    real_exists, real_is_file = Path.exists, Path.is_file

    def under(p: Path) -> bool:
        return os.path.abspath(p).startswith(locked)

    def exists(self: Path, *a: object, **k: object) -> bool:
        return False if under(self) else real_exists(self, *a, **k)  # type: ignore[arg-type]

    def is_file(self: Path, *a: object, **k: object) -> bool:
        return False if under(self) else real_is_file(self, *a, **k)  # type: ignore[arg-type]

    path.parent.mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", exists)
        m.setattr(Path, "is_file", is_file)
        os.chmod(path.parent, 0)
        try:
            yield
        finally:
            os.chmod(path.parent, 0o755)


_SKIP_CHMOD = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_user_store_dir_makes_lock_gaps_unknown(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The store may be there, so the answer is unknown, not a false gap."""
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    assert _cli(["lock", "--user", STATEMENT], capsys)[0] == 0
    with _chmod0_parent(user, monkeypatch):
        report = detect_lock_gaps(
            str(repo), audit_enabled=True, user_store_path=str(user),
        )
        assert not report.known
        assert report.unknown_reason is not None
        assert report.unknown_reason.startswith("the user lock store could not be read")


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_repo_store_dir_makes_lock_gaps_unknown(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _fail_typed_lock(f"/aelf:lock {STATEMENT}", tmp_path, monkeypatch)
    _seed(repo, [])
    outcomes = command_outcomes_path_for_db(repo)
    moved = tmp_path / "outcomes-copy"
    # The outcome log sits beside the store; read it before the dir closes.
    moved.write_bytes(outcomes.read_bytes())
    from aelfrice import lock_gaps as lg

    def outcomes_copy(_db: Path) -> Path:
        return moved

    monkeypatch.setattr(lg, "command_outcomes_path_for_db", outcomes_copy)
    with _chmod0_parent(repo, monkeypatch):
        report = detect_lock_gaps(
            str(repo), audit_enabled=True, user_store_path=str(user),
        )
        assert not report.known
        assert report.unknown_reason is not None
        assert report.unknown_reason.startswith("the store could not be read")


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_user_store_dir_makes_the_doctor_listing_unknown(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    with _chmod0_parent(user, monkeypatch):
        report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert not report.known
    assert report.unknown_reason is not None
    assert report.unknown_reason.startswith("the user lock store could not be read")


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_user_store_dir_refuses_a_promote(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The duplicate check can't run, so nothing is added."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    with _chmod0_parent(user, monkeypatch):
        rc, _, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1
    assert "cannot read the user lock store" in err
    assert [i for i, _, _, _ in _locks(user)] == ["u1"]


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_user_store_dir_warns_in_core(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", USER_RULE)])
    with _chmod0_parent(user, monkeypatch):
        rc, out, err = _cli(["core", "--locked-only"], capsys)
    assert rc == 0
    assert "u1" not in out
    assert "cannot read the user lock store" in err


@pytest.mark.timeout(60)
def test_lock_gaps_with_an_in_memory_user_store_resolves_nothing(
    tmp_path: Path, stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = stores
    _fail_typed_lock(f"/aelf:lock --user {STATEMENT}", tmp_path, monkeypatch)
    assert _cli(["lock", STATEMENT], capsys)[0] == 0
    [gap] = detect_lock_gaps(
        str(repo), audit_enabled=True, user_store_path=":memory:",
    ).gaps
    assert gap.user_scope


@pytest.mark.timeout(60)
def test_doctor_skips_a_retired_repo_lock(stores: tuple[Path, Path]) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    conn = sqlite3.connect(str(repo))
    conn.execute(
        "UPDATE beliefs SET valid_to = '2026-10-02T00:00:00Z' WHERE id = 'r1'"
    )
    conn.commit()
    conn.close()
    report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert report.known and report.pairs == ()


def _lock_id(text: str) -> str:
    from aelfrice.derivation import (
        DerivationInput,
        derive,
    )
    from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER

    d = derive(DerivationInput(
        raw_text=text, source_kind=INGEST_SOURCE_CLI_REMEMBER,
        ts="2026-10-01T00:00:00Z", session_id=None,
    ))
    assert d.belief is not None
    return d.belief.id


def _plain_with(bid: str, content: str, origin: str) -> Belief:
    b = _plain(bid, content)
    b.origin = origin
    return b


@pytest.mark.timeout(60)
def test_promote_onto_an_unlocked_row_at_the_lock_id_is_an_upgrade(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A user lock unlocked earlier keeps its row; the promote re-locks it."""
    from aelfrice.models import ORIGIN_USER_VALIDATED

    repo, user = stores
    uid = _lock_id(REPO_RULE)
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_plain_with(uid, REPO_RULE, ORIGIN_USER_VALIDATED)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    [row] = [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"]
    assert row["kind"] == "upgrade" and row["id"] == uid
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    [(origin, level)] = conn.execute(
        "SELECT origin, lock_level FROM beliefs WHERE id = ?", (uid,)
    ).fetchall()
    conn.close()
    # A user-validated row keeps its stronger origin.
    assert (origin, level) == (ORIGIN_USER_VALIDATED, LOCK_USER)


@pytest.mark.timeout(60)
def test_promote_onto_a_same_text_row_is_corroborated_and_takes_user_origin(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    """A row with the statement's text under another id is reused."""
    from aelfrice.models import ORIGIN_AGENT_INFERRED, ORIGIN_USER_STATED

    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_plain_with("other", REPO_RULE, ORIGIN_AGENT_INFERRED)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    [row] = [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"]
    assert row["kind"] == "corroborated" and row["id"] == "other"
    conn = sqlite3.connect(f"file:{user}?mode=ro", uri=True)
    [(origin, level)] = conn.execute(
        "SELECT origin, lock_level FROM beliefs WHERE id = 'other'"
    ).fetchall()
    conn.close()
    assert (origin, level) == (ORIGIN_USER_STATED, LOCK_USER)


@pytest.mark.timeout(60)
def test_a_promote_whose_lock_write_did_not_take_writes_nothing(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final read is the guard: a write that left the row unlocked
    must not be reported, and the transaction rolls back."""
    repo, user = stores
    uid = _lock_id(REPO_RULE)
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_plain(uid, REPO_RULE)])
    before = (_count(user, "ingest_log"), _locks(user))
    def no_update(self: MemoryStore, b: Belief) -> None:
        return None

    monkeypatch.setattr(MemoryStore, "update_belief", no_update)
    rc, out, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1 and out == ""
    assert f"{uid} is not locked" in err
    assert (_count(user, "ingest_log"), _locks(user)) == before
    assert [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"] == []


@pytest.mark.timeout(60)
def test_a_near_duplicate_added_after_the_first_check_is_caught_in_the_write(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent promote that lands between the read and the write."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u0", USER_RULE)])
    def none_yet() -> list[tuple[str, str]]:
        return []

    monkeypatch.setattr(cli, "_read_user_lock_pairs", none_yet)
    _seed(user, [_lock("u1", NEAR_DUP)])
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0
    assert "near-duplicate lock: u1" in out
    assert sorted(i for i, _, _, _ in _locks(user)) == ["u0", "u1"]
    assert [r for r in _feed_rows(repo) if r.get("event") == "belief.locked"] == []


@pytest.mark.timeout(60)
def test_show_conflicts_marks_a_peer_hit_against_a_user_lock(
    tmp_path: Path, stores: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    peer = tmp_path / "peer.db"
    peer_belief = _plain("pp1", "The default server port is 9090.")
    peer_belief.scope = "global"
    _seed(peer, [peer_belief])
    deps = tmp_path / "knowledge_deps.json"
    deps.write_text(json.dumps(
        {"version": 1, "deps": [{"name": "peerA", "path": str(peer)}]},
    ), encoding="utf-8")
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps))
    _seed(repo, [_plain("p0", "an unrelated repository note")])
    _seed(user, [_lock("u1", "The default server port is 8080.")])
    monkeypatch.setenv("AELF_SHOW_CONFLICTS", "1")
    rc, out, _ = _cli(["search", "server port", "--json"], capsys)
    assert rc == 0
    rows = {r["id"]: r for r in json.loads(out)}
    assert rows["pp1"]["scope"] == "peerA"
    assert rows["pp1"]["slot_conflict_with"] == "u1"
    assert rows["pp1"]["slot_conflict_scope"] == "user"


# --- third review: link fallbacks, cleanup after success, repo-dir doctor -------


@_SKIP_CHMOD
@pytest.mark.timeout(60)
def test_an_unsearchable_repo_store_dir_makes_the_doctor_listing_unknown(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    _seed(user, [_lock("u1", NEAR_DUP)])
    with _chmod0_parent(repo, monkeypatch):
        report = diagnose_user_lock_duplicates(str(repo), str(user))
    assert not report.known
    assert report.unknown_reason is not None
    assert report.unknown_reason.startswith("the repository store could not be read")


@pytest.mark.timeout(60)
def test_a_parent_that_is_a_file_reads_as_missing(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    assert ul.store_file_present(blocker / "memory.db") is False
    report = diagnose_user_lock_duplicates(
        str(tmp_path / "repo.db"), str(blocker / "memory.db"),
    )
    assert report.known and not report.user_store_present


def _user_dir_listing(user: Path) -> list[str]:
    return sorted(f.name for f in user.parent.iterdir())


@pytest.mark.timeout(60)
def test_a_successful_promote_leaves_only_the_store_file(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    assert _cli(["lock", "--promote-user", "r1"], capsys)[0] == 0
    assert _user_dir_listing(user) == ["memory.db"]
    assert [c for _, c, _, _ in _locks(user)] == [REPO_RULE]


@pytest.mark.timeout(60)
def test_a_store_that_appears_before_the_link_gets_the_copy_once(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another process creates the store first; the copy goes into it."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])

    def racing_link(src: object, dst: object, *a: object, **k: object) -> None:
        _seed(user, [_lock("u9", USER_RULE)])
        raise FileExistsError(errno.EEXIST, "File exists", str(dst))

    monkeypatch.setattr(os, "link", racing_link)
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0, out
    contents = sorted(c for _, c, _, _ in _locks(user))
    assert contents == sorted([REPO_RULE, USER_RULE])
    assert not [f for f in _user_dir_listing(user) if ".promote-" in f]


@pytest.mark.timeout(60)
def test_a_store_that_appears_and_is_the_repo_store_is_refused(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same-file guard runs again on a store that appeared late."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    calls = {"n": 0}
    real_guard = cli._user_store_is_repo_store  # pyright: ignore[reportPrivateUsage]

    def guard() -> bool:
        calls["n"] += 1
        return real_guard() if calls["n"] == 1 else True

    def racing_link(src: object, dst: object, *a: object, **k: object) -> None:
        _seed(user, [_lock("u9", USER_RULE)])
        raise FileExistsError(errno.EEXIST, "File exists", str(dst))

    monkeypatch.setattr(cli, "_user_store_is_repo_store", guard)
    monkeypatch.setattr(os, "link", racing_link)
    rc, out, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1 and out == ""
    assert "names the repository store" in err
    assert [c for _, c, _, _ in _locks(user)] == [USER_RULE]


@pytest.mark.timeout(60)
def test_a_filesystem_without_hard_links_gets_the_copy_once(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])

    def no_links(src: object, dst: object, *a: object, **k: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted", str(dst))

    monkeypatch.setattr(os, "link", no_links)
    rc, out, _ = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 0, out
    assert [c for _, c, _, _ in _locks(user)] == [REPO_RULE]
    assert _count(user, "ingest_log") == 1
    assert not [f for f in _user_dir_listing(user) if ".promote-" in f]


@pytest.mark.timeout(60)
def test_a_failed_in_place_write_after_a_failed_link_leaves_no_store(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback removes the store it created when its write fails."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE)])
    real_update = MemoryStore.update_belief

    def no_links(src: object, dst: object, *a: object, **k: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted", str(dst))

    def failing_in_place(self: MemoryStore, b: Belief) -> None:
        if os.path.abspath(self.db_path) == os.path.abspath(user):
            raise sqlite3.OperationalError("database is locked")
        real_update(self, b)

    monkeypatch.setattr(os, "link", no_links)
    monkeypatch.setattr(MemoryStore, "update_belief", failing_in_place)
    rc, out, err = _cli(["lock", "--promote-user", "r1"], capsys)
    assert rc == 1 and out == ""
    assert "nothing was written to the user store" in err
    assert not user.exists()
    assert _user_dir_listing(user) == []
