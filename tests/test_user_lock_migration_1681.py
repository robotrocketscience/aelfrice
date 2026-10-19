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

import hashlib
import io
import json
import os
import shlex
import sqlite3
from collections.abc import Iterator
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
def test_promote_refuses_a_lock_past_its_window_on_a_read_only_store(
    stores: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the read-only fallback no sweep has run, so the row still reads
    `lock_level = 'user'` after its window closed."""
    repo, user = stores
    _seed(repo, [_lock("r1", REPO_RULE, expires="2026-01-01T00:00:00+00:00")])
    # Re-mark it locked after the seeding open swept it.
    conn = sqlite3.connect(str(repo))
    conn.execute("UPDATE beliefs SET lock_level = 'user' WHERE id = 'r1'")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        cli, "open_store_for_read",
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
