"""A typed lock that failed stays visible until it is applied (#1622).

The hook runs a typed `/aelf:lock` itself and reports a failure in that
turn (#1626). These arms pin what happens after the turn: the outcome is
recorded where it is known, `aelf doctor` lists the request while it is
still unapplied, SessionStart says so in one line, and the store, not a
later request, decides when the gap is closed.

Every arm drives the real hook entry point and reads a real tmp store
back. Failure is injected by replacing `cli.main`, which is the one
boundary the executor crosses to run the command.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice import cli
from aelfrice.doctor import (
    DoctorReport,
    _format_lock_gaps_section,
    diagnose_lock_gaps,
    format_report,
)
from aelfrice.hook import (
    CommandReason,
    session_start,
    user_prompt_submit,
)
from aelfrice.hook_audit import command_outcomes_path_for_db
from aelfrice.lock_gaps import (
    _INGEST_SOURCE_LOCK,
    _LOCK_ENDED_SOURCES,
    _LOCK_LEVEL_USER,
    GAP_REASONS,
    LockGap,
    LockGapReport,
    detect_lock_gaps,
    read_command_outcomes,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.store import MemoryStore

STATEMENT = "Keep every widget in the blue drawer."
_REAL_CLI_MAIN = cli.main


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fire(prompt: str, tmp_path: Path) -> tuple[str, str]:
    payload = json.dumps(
        {"prompt": prompt, "session_id": "s1622", "cwd": str(tmp_path)}
    )
    out, err = io.StringIO(), io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=out, stderr=err
    ) == 0
    return out.getvalue(), err.getvalue()


def _boom(*_a: object, **_k: object) -> int:
    raise RuntimeError("synthetic store fault")


def _gaps(db: Path) -> LockGapReport:
    return detect_lock_gaps(str(db), audit_enabled=True)


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    return path


def _fail_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
               statement: str = STATEMENT) -> None:
    monkeypatch.setattr(cli, "main", _boom)
    _fire(f"/aelf:lock {statement}", tmp_path)
    monkeypatch.setattr(cli, "main", _REAL_CLI_MAIN)


FAILED_AT = "2026-01-01T00:00:00Z"


def _backdate_outcomes(db: Path, ts: str = FAILED_AT) -> None:
    """Restamp every recorded outcome at `ts`.

    A removal closes a gap only in a strictly later second, so a test
    that fails a lock and then removes it within one second would see
    the gap stay open. Moving the failure into the past makes "later"
    hold by construction instead of by the clock.
    """
    path = command_outcomes_path_for_db(db)
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines()
            if ln.strip()]
    path.write_text(
        "".join(json.dumps({**r, "ts": ts}) + "\n" for r in rows),
        encoding="utf-8",
    )


def _write_failure(db: Path, ts: str, statement: str = STATEMENT) -> None:
    command_outcomes_path_for_db(db).write_text(json.dumps({
        "ts": ts, "hook": "aelf_command",
        "command": "lock", "reason": "exception",
        "arg_sha256": _sha(statement), "arg_len": len(statement),
        "statement": statement,
    }) + "\n", encoding="utf-8")


def _seed_unlocked(db: Path, bid: str = "aabbccddeeff1622",
                   statement: str = STATEMENT) -> str:
    """An ordinary, unlocked belief that carries the statement's hash."""
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id=bid, content=statement, content_hash=_sha(statement),
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, created_at="2025-01-01T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()
    return bid


def _feedback(db: Path, bid: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            (str(src), str(at)) for src, at in conn.execute(
                "SELECT source, created_at FROM feedback_history "
                "WHERE belief_id = ? ORDER BY rowid", (bid,),
            )
        ]
    finally:
        conn.close()


# --- the stored hash identifies the belief ------------------------------


@pytest.mark.timeout(120)
def test_arg_sha256_equals_the_stored_content_hash(
    tmp_path: Path, db: Path,
) -> None:
    """The detector's whole join rests on this equality, so it is checked
    against what the store actually wrote, not against the formula."""
    _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    [row] = read_command_outcomes(command_outcomes_path_for_db(db))
    assert row["reason"] == "ok"
    assert row["arg_len"] == len(STATEMENT)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        stored = conn.execute(
            "SELECT content_hash, lock_level FROM beliefs WHERE content = ?",
            (STATEMENT,),
        ).fetchall()
    finally:
        conn.close()
    assert stored == [(row["arg_sha256"], _LOCK_LEVEL_USER)]
    assert row["arg_sha256"] == _sha(STATEMENT)


# --- AC4: the gap is reported, and only when it is one -------------------


@pytest.mark.timeout(120)
def test_a_lost_lock_is_reported(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    report = _gaps(db)
    assert report.known
    [gap] = report.gaps
    assert gap.statement == STATEMENT
    assert gap.reason == CommandReason.EXCEPTION.value
    assert gap.arg_sha256 == _sha(STATEMENT)
    assert gap.fix_command == f"aelf lock '{STATEMENT}'"


@pytest.mark.timeout(120)
def test_a_nonzero_exit_is_reported(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _exit2(*_a: object, **_k: object) -> int:
        return 2

    monkeypatch.setattr(cli, "main", _exit2)
    _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    [gap] = _gaps(db).gaps
    assert gap.reason == CommandReason.NONZERO_EXIT.value


@pytest.mark.timeout(120)
def test_an_applied_lock_is_not_a_gap(tmp_path: Path, db: Path) -> None:
    _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    report = _gaps(db)
    assert report.known and report.records_seen == 1
    assert report.gaps == ()


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("/aelf:lock --help", "leading_dash"),
        ("/aelf:lock", "empty_argument"),
        ("/aelf:lock " + "x" * 4001, "over_cap"),
    ],
    ids=["leading-dash", "empty", "over-cap"],
)
def test_an_input_error_is_recorded_but_is_not_a_gap(
    tmp_path: Path, db: Path, prompt: str, reason: str,
) -> None:
    """Refused in the turn already; counting it would nag every session."""
    _fire(prompt, tmp_path)
    rows = read_command_outcomes(command_outcomes_path_for_db(db))
    assert [r["reason"] for r in rows] == [reason]
    assert _gaps(db).gaps == ()


# --- the store closes the gap -------------------------------------------


@pytest.mark.timeout(120)
def test_a_later_cli_lock_clears_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI writes no outcome row, so only the store can say so."""
    _fail_lock(tmp_path, monkeypatch)
    assert len(_gaps(db).gaps) == 1
    assert cli.main(["lock", STATEMENT], out=io.StringIO()) == 0
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_an_unlock_after_the_failure_clears_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    buf = io.StringIO()
    assert cli.main(["lock", STATEMENT], out=buf) == 0
    bid = buf.getvalue().split("locked:", 1)[1].split()[0]
    assert cli.main(["unlock", bid], out=io.StringIO()) == 0
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_an_unlock_before_the_failure_does_not_clear_it(
    tmp_path: Path, db: Path,
) -> None:
    """"Later" is the rule: an old unlock says nothing about a new failure."""
    buf = io.StringIO()
    assert cli.main(["lock", STATEMENT], out=buf) == 0
    bid = buf.getvalue().split("locked:", 1)[1].split()[0]
    assert cli.main(["unlock", bid], out=io.StringIO()) == 0
    _write_failure(db, "2099-01-01T00:00:00Z")
    assert len(_gaps(db).gaps) == 1


def _cli_lock(statement: str = STATEMENT) -> str:
    buf = io.StringIO()
    assert cli.main(["lock", statement], out=buf) == 0
    return buf.getvalue().split("locked:", 1)[1].split()[0]


@pytest.mark.timeout(120)
def test_a_forced_retire_after_the_failure_clears_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user removed the lock on purpose. Reporting it as missing would
    tell them to put back what they just took out."""
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    bid = _cli_lock()
    assert cli.main(["retire", bid, "--force"], out=io.StringIO()) == 0
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_a_forced_delete_after_the_failure_clears_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aelf delete` removes the beliefs row, so `content_hash` cannot
    reach it; the detector finds it through `ingest_log`."""
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    bid = _cli_lock()
    assert cli.main(
        ["delete", bid, "--force", "--yes"], out=io.StringIO(),
    ) == 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE id = ?", (bid,),
        ).fetchone() == (0,)
    finally:
        conn.close()
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_a_delete_before_the_failure_does_not_clear_it(
    tmp_path: Path, db: Path,
) -> None:
    """The ordering rule holds for a deleted belief too."""
    bid = _cli_lock()
    assert cli.main(
        ["delete", bid, "--force", "--yes"], out=io.StringIO(),
    ) == 0
    _write_failure(db, "2099-01-01T00:00:00Z")
    assert len(_gaps(db).gaps) == 1


# --- "later" is a strictly later second, whatever the stamp format -------


@pytest.mark.timeout(120)
def test_an_unlock_in_the_failures_second_does_not_clear_it(
    tmp_path: Path, db: Path,
) -> None:
    """The outcome `ts` is truncated to the second, so an unlock stamped
    in the same second may have come first. A tie leaves the gap open."""
    bid = _cli_lock()
    assert cli.main(["unlock", bid], out=io.StringIO()) == 0
    [(_, unlocked_at)] = [
        r for r in _feedback(db, bid) if r[0] == "lock:unlock"
    ]
    _write_failure(db, unlocked_at)
    assert len(_gaps(db).gaps) == 1


def _expire_lock_at(db: Path, now: str) -> None:
    s = MemoryStore(str(db))
    try:
        assert s.sweep_expired_locks(now=now) == 1
    finally:
        s.close()


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("expired_at", "open_gaps"),
    [
        ("2099-01-01T00:00:00.900000+00:00", 1),
        ("2099-01-01T00:00:01.000001+00:00", 0),
    ],
    ids=["same-second", "next-second"],
)
def test_an_expiry_stamp_is_compared_as_an_instant(
    db: Path, expired_at: str, open_gaps: int,
) -> None:
    """`lock:expire` stamps `isoformat()` with microseconds and `+00:00`;
    the failure stamps `%Y-%m-%dT%H:%M:%SZ`. Both are read as instants at
    whole-second resolution, so 0.9 s into the failure's second is a tie
    and the next second is later."""
    assert cli.main(["lock", STATEMENT, "--for", "1d"],
                    out=io.StringIO()) == 0
    _expire_lock_at(db, expired_at)
    _write_failure(db, "2099-01-01T00:00:00Z")
    assert len(_gaps(db).gaps) == open_gaps


# --- a retire counts only while the statement stays retired ---------------


@pytest.mark.timeout(120)
def test_a_retire_after_the_failure_clears_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid = _seed_unlocked(db)
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    assert len(_gaps(db).gaps) == 1
    assert cli.main(["retire", bid], out=io.StringIO()) == 0
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_a_restore_after_the_retire_reopens_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The statement is back and still unlocked, so the request the user
    typed is still unapplied. The old retire row must not hide that."""
    bid = _seed_unlocked(db)
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    assert cli.main(["retire", bid], out=io.StringIO()) == 0
    assert cli.main(["restore", bid], out=io.StringIO()) == 0
    assert [src for src, _ in _feedback(db, bid)
            if src.startswith("user_")] == ["user_retired", "user_restored"]
    assert len(_gaps(db).gaps) == 1


# --- a delete stays visible after the orphan feedback is collected ------


@pytest.mark.timeout(120)
def test_a_delete_still_clears_the_gap_after_orphan_feedback_gc(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--gc-orphan-feedback --apply` removes the delete's own audit row.
    The later `aelf lock` in `ingest_log` still shows the user applied
    the lock and then removed it."""
    _fail_lock(tmp_path, monkeypatch)
    _backdate_outcomes(db)
    bid = _cli_lock()
    assert cli.main(
        ["delete", bid, "--force", "--yes"], out=io.StringIO(),
    ) == 0
    assert cli.main(
        ["doctor", "--gc-orphan-feedback", "--apply"], out=io.StringIO(),
    ) == 0
    assert _feedback(db, bid) == []
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(120)
def test_the_failed_attempts_own_ingest_does_not_clear_the_gap(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock wrote its `ingest_log` row and died before the belief
    existed. That row maps the statement to a missing belief too, but it
    is not later than the failure, so it says nothing about a delete."""
    from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER

    def _ingest_then_die(*_a: object, **_k: object) -> int:
        s = MemoryStore(str(db))
        try:
            s.record_ingest(
                source_kind=INGEST_SOURCE_CLI_REMEMBER, raw_text=STATEMENT,
                derived_belief_ids=["0123456789abcdef"], ts=FAILED_AT,
            )
        finally:
            s.close()
        raise RuntimeError("died after the ingest row")

    monkeypatch.setattr(cli, "main", _ingest_then_die)
    _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    monkeypatch.setattr(cli, "main", _REAL_CLI_MAIN)
    _backdate_outcomes(db, FAILED_AT)
    [gap] = _gaps(db).gaps
    assert gap.statement == STATEMENT


# --- the record is written where the outcome is known --------------------


@pytest.mark.timeout(120)
def test_the_record_survives_a_retrieval_that_dies(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store fault that fails the lock is likely to fail retrieval too.

    `SystemExit` stands in for the hook being killed at its timeout: it is
    not caught by the turn's `except Exception`, so nothing after the
    raise runs.
    """
    from aelfrice import hook

    def _killed(*_a: object, **_k: object) -> None:
        raise SystemExit("killed mid-retrieval")

    monkeypatch.setattr(hook, "_retrieve", _killed)
    monkeypatch.setattr(hook, "_write_hook_audit_record", _killed)
    monkeypatch.setattr(cli, "main", _boom)
    with pytest.raises(SystemExit):
        _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    monkeypatch.setattr(cli, "main", _REAL_CLI_MAIN)
    [gap] = _gaps(db).gaps
    assert gap.statement == STATEMENT


@pytest.mark.timeout(120)
def test_no_record_is_written_while_the_hook_audit_is_disabled(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row holds statement text, so the audit switch must stop it."""
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "0")
    _fail_lock(tmp_path, monkeypatch)
    _fire(f"/aelf:lock {STATEMENT}", tmp_path)
    assert not command_outcomes_path_for_db(db).exists()


@pytest.mark.timeout(120)
def test_a_lone_surrogate_is_refused_as_an_input_error(
    tmp_path: Path, db: Path,
) -> None:
    """No store can hold it, so a recorded gap could never close."""
    payload = (
        '{"prompt": "/aelf:lock Keep \\ud800 this", "session_id": "s",'
        f' "cwd": {json.dumps(str(tmp_path))}}}'
    )
    err = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=io.StringIO(), stderr=err,
    ) == 0
    assert "not valid Unicode text" in err.getvalue()
    [row] = read_command_outcomes(command_outcomes_path_for_db(db))
    assert row["reason"] == CommandReason.INVALID_TEXT.value
    report = _gaps(db)
    assert report.known and report.gaps == ()
    assert "/aelf:lock request" not in _session_start()


# --- doctor -------------------------------------------------------------


def _section(report: LockGapReport | None) -> str:
    lines: list[str] = []
    _format_lock_gaps_section(DoctorReport(lock_gaps=report), lines)
    return "\n".join(lines)


@pytest.mark.timeout(120)
def test_doctor_lists_the_statement_and_the_fix(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    text = _section(diagnose_lock_gaps(str(db), tmp_path))
    assert STATEMENT in text
    assert f"fix: aelf lock '{STATEMENT}'" in text


@pytest.mark.timeout(60)
def test_doctor_says_unknown_when_the_audit_is_disabled(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "0")
    text = _section(diagnose_lock_gaps(str(db), tmp_path))
    assert "unknown: the hook audit is disabled" in text
    assert "none" not in text


@pytest.mark.timeout(60)
def test_doctor_says_unknown_when_there_is_no_audit(
    tmp_path: Path, db: Path,
) -> None:
    text = _section(diagnose_lock_gaps(str(db), tmp_path))
    assert "unknown: no hook audit found" in text


@pytest.mark.timeout(60)
def test_doctor_renders_the_section_with_no_store() -> None:
    assert "unknown: no store was checked" in _section(None)


@pytest.mark.timeout(120)
def test_aelf_doctor_prints_the_section(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command itself, so the wiring in `diagnose` is covered."""
    _fail_lock(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    cli.main(["doctor"], out=buf)
    text = buf.getvalue() + capsys.readouterr().out
    assert "typed /aelf:lock requests that did not take effect:" in text
    assert f"fix: aelf lock '{STATEMENT}'" in text


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "scanned", [False, True], ids=["no-settings", "settings"],
)
def test_both_report_paths_render_the_section(
    tmp_path: Path, scanned: bool,
) -> None:
    gap = LockGap(_sha(STATEMENT), STATEMENT, len(STATEMENT), "exception",
                  "2026-01-01T00:00:00Z", None, 1)
    report = DoctorReport(lock_gaps=LockGapReport(known=True, gaps=(gap,)))
    if scanned:
        settings = tmp_path / "settings.json"
        settings.write_text("{}", encoding="utf-8")
        report.scopes_scanned.append(("user", settings))
    assert f"fix: aelf lock '{STATEMENT}'" in format_report(report)


@pytest.mark.timeout(60)
def test_doctor_says_unknown_when_the_store_cannot_be_read(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    db.write_bytes(b"not a sqlite file" * 100)
    report = _gaps(db)
    assert not report.known
    assert report.unknown_reason is not None
    assert report.unknown_reason.startswith("the store could not be read")
    text = _section(report)
    assert "unknown: the store could not be read" in text
    assert "none" not in text


@pytest.mark.timeout(120)
def test_a_truncated_row_prints_no_runnable_fix(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command built from the prefix would lock text the user never
    typed, at user tier, and leave the gap open."""
    long_statement = (
        "Keep every widget in the blue drawer and " * 14
    ).strip()
    assert 500 < len(long_statement) <= 4000
    _fail_lock(tmp_path, monkeypatch, long_statement)
    [gap] = _gaps(db).gaps
    assert gap.truncated and gap.fix_command is None
    text = _section(_gaps(db))
    assert "fix: aelf lock" not in text
    assert f"of {len(long_statement)} characters" in text
    _cli_lock(long_statement)
    assert _gaps(db).gaps == ()


@pytest.mark.timeout(60)
def test_a_recorded_lone_surrogate_cannot_break_doctor() -> None:
    """Defence in depth for a row the input check did not stop: printed
    raw, it raises on a strict UTF-8 stream and ends the whole run."""
    statement = "Keep \ud800 this"
    gap = LockGap("0" * 64, statement, len(statement), "exception",
                  "2026-01-01T00:00:00Z", None, 1)
    # The whole statement is recorded, so only the invalid text, not
    # truncation, can be what withholds the fix command.
    assert not gap.truncated and not gap.valid_text
    report = DoctorReport(lock_gaps=LockGapReport(known=True, gaps=(gap,)))
    text = format_report(report)
    text.encode("utf-8")
    assert "Keep \\ud800 this" in text
    assert gap.fix_command is None
    assert "fix: aelf lock" not in text


# --- SessionStart -------------------------------------------------------


def _session_start() -> str:
    payload = json.dumps({"session_id": "s2", "transcript_path": "/dev/null",
                          "hook_event_name": "SessionStart"})
    out = io.StringIO()
    assert session_start(stdin=io.StringIO(payload), stdout=out,
                         stderr=io.StringIO()) == 0
    return out.getvalue()


@pytest.mark.timeout(120)
def test_session_start_prints_one_line_while_a_gap_remains(
    tmp_path: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_lock(tmp_path, monkeypatch)
    lines = [ln for ln in _session_start().splitlines()
             if "/aelf:lock request" in ln]
    assert lines == [
        "aelfrice: 1 /aelf:lock request failed and is still not locked"
        " — `/aelf:doctor` lists the statements and the fix."
    ]
    assert cli.main(["lock", STATEMENT], out=io.StringIO()) == 0
    assert "/aelf:lock request" not in _session_start()


# --- the literals the detector cannot import ------------------------------


def test_the_detector_literals_match_their_sources() -> None:
    from aelfrice.models import (
        FEEDBACK_SOURCE_LOCK_EXPIRE,
        INGEST_SOURCE_CLI_REMEMBER,
        LOCK_USER,
    )
    from aelfrice.promotion import SOURCE_LOCK_UNLOCK

    assert GAP_REASONS == {
        CommandReason.EXCEPTION.value, CommandReason.NONZERO_EXIT.value,
    }
    assert _LOCK_ENDED_SOURCES[:2] == (
        SOURCE_LOCK_UNLOCK, FEEDBACK_SOURCE_LOCK_EXPIRE,
    )
    assert _LOCK_LEVEL_USER == LOCK_USER
    assert _INGEST_SOURCE_LOCK == INGEST_SOURCE_CLI_REMEMBER
