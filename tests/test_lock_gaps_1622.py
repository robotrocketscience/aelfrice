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
    LockGapReport,
    detect_lock_gaps,
    read_command_outcomes,
)

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
    path = command_outcomes_path_for_db(db)
    path.write_text(json.dumps({
        "ts": "2099-01-01T00:00:00Z", "hook": "aelf_command",
        "command": "lock", "reason": "exception",
        "arg_sha256": _sha(STATEMENT), "arg_len": len(STATEMENT),
        "statement": STATEMENT,
    }) + "\n", encoding="utf-8")
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
    command_outcomes_path_for_db(db).write_text(json.dumps({
        "ts": "2099-01-01T00:00:00Z", "hook": "aelf_command",
        "command": "lock", "reason": "exception",
        "arg_sha256": _sha(STATEMENT), "arg_len": len(STATEMENT),
        "statement": STATEMENT,
    }) + "\n", encoding="utf-8")
    assert len(_gaps(db).gaps) == 1


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
