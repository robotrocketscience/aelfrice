"""#1780: `aelf doctor --retire-scheduled-prompts` retires the beliefs that
only scheduled prompts produced.

The fixture store is built by the real ingest path: a synthetic
`turns.jsonl` goes through `ingest_jsonl`, so every `ingest_log` row
carries the `turn_sha` ingest stamps. A synthetic host session file holds
the matching host records, flagged or typed. Nothing here reads a real
transcript or store.
"""
from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.ingest import ingest_jsonl, turn_sha, user_turn_key
from aelfrice.models import LOCK_USER
from aelfrice.scheduled_prompts import (
    format_scheduled_prompt_report,
    retire_scheduled_prompt_beliefs,
    scan_host_sessions,
)
from aelfrice.store import MemoryStore

SESSION_A = "sess-a-1780"
SESSION_B = "sess-b-1780"

# One sentence per prompt, so each prompt yields one belief.
SCHEDULED_ONLY = "The release checklist requires a signed tag before publishing."
TYPED_ONLY = "The staging database lives on the second replica host."
MIXED_SENTENCE = "The nightly job rotates the audit log at midnight."
MIXED_TYPED = MIXED_SENTENCE + " The rotation keeps seven days of history."
SAME_SESSION_BOTH = "The cache warmer runs before the first request arrives."
OTHER_SESSION_TYPED = "The backup script compresses archives with zstd."
LOCKED_SCHEDULED = "The deploy key is stored in the project vault."


def _turn(session: str, minute: int, text: str) -> dict[str, object]:
    return {
        "role": "user",
        "text": text,
        "session_id": session,
        "ts": f"2026-10-11T15:{minute:02d}:00.000000+00:00",
        "turn_id": f"t{minute}",
        "schema_version": 1,
    }


def _host(text: str, *, scheduled: bool, session: str = SESSION_A) -> dict[str, object]:
    rec: dict[str, object] = {
        "type": "user",
        "sessionId": session,
        "timestamp": "2026-10-11T15:00:00.000Z",
        "message": {"role": "user", "content": text},
    }
    if scheduled:
        rec.update(
            isMeta=True, promptSource="system", turnOrigin="scheduled",
            scheduledTaskId="task-1", queuePriority="later",
        )
    else:
        rec.update(promptSource="typed", turnOrigin="human")
    return rec


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
    )
    return path


def _seed(store: MemoryStore, tmp_path: Path) -> Path:
    """Ingest the turns and return the session A host file."""
    turns = _write_jsonl(tmp_path / "turns.jsonl", [
        _turn(SESSION_A, 1, SCHEDULED_ONLY),
        _turn(SESSION_A, 2, TYPED_ONLY),
        _turn(SESSION_A, 3, MIXED_SENTENCE),
        _turn(SESSION_A, 4, MIXED_TYPED),
        _turn(SESSION_A, 5, SAME_SESSION_BOTH),
        _turn(SESSION_A, 6, SAME_SESSION_BOTH),
        _turn(SESSION_A, 7, OTHER_SESSION_TYPED),
        _turn(SESSION_B, 8, OTHER_SESSION_TYPED),
        _turn(SESSION_A, 9, LOCKED_SCHEDULED),
    ])
    ingest_jsonl(store, turns)
    locked = _belief_id(store, LOCKED_SCHEDULED)
    b = store.get_belief(locked)
    assert b is not None
    b.lock_level = LOCK_USER
    b.locked_at = "2026-10-12T00:00:00Z"
    store.update_belief(b)
    host_dir = tmp_path / "host" / "project"
    host_dir.mkdir(parents=True)
    return _write_jsonl(host_dir / f"{SESSION_A}.jsonl", [
        _host(SCHEDULED_ONLY, scheduled=True),
        _host(TYPED_ONLY, scheduled=False),
        _host(MIXED_SENTENCE, scheduled=True),
        _host(MIXED_TYPED, scheduled=False),
        _host(SAME_SESSION_BOTH, scheduled=True),
        _host(SAME_SESSION_BOTH, scheduled=False),
        _host(OTHER_SESSION_TYPED, scheduled=True),
        _host(LOCKED_SCHEDULED, scheduled=True),
        {"type": "assistant", "sessionId": SESSION_A,
         "message": {"role": "assistant", "content": "Ok."}},
    ])


def _belief_id(store: MemoryStore, sentence: str) -> str:
    hits = [b.id for b in store.list_active_beliefs() if b.content == sentence]
    assert len(hits) == 1, (sentence, hits)
    return hits[0]


def _active_ids(store: MemoryStore) -> set[str]:
    return {b.id for b in store.list_active_beliefs()}


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        yield s
    finally:
        s.close()


def test_host_key_matches_the_ingest_log_turn_sha(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _seed(store, tmp_path)
    key = user_turn_key(_host(SCHEDULED_ONLY, scheduled=True))
    assert key == (SESSION_A, turn_sha(SCHEDULED_ONLY))
    bid = _belief_id(store, SCHEDULED_ONLY)
    rows = [(s, h) for b, s, h in store.list_ingest_turn_derivations() if b == bid]
    assert rows == [key]


def test_host_key_cuts_harness_text_as_ingest_does(
    store: MemoryStore, tmp_path: Path,
) -> None:
    text = (
        "<system-reminder>\nsynthetic reminder body\n</system-reminder>\n"
        + SCHEDULED_ONLY
    )
    turns = _write_jsonl(tmp_path / "turns.jsonl", [_turn(SESSION_A, 1, text)])
    ingest_jsonl(store, turns)
    bid = _belief_id(store, SCHEDULED_ONLY)
    rows = [(s, h) for b, s, h in store.list_ingest_turn_derivations() if b == bid]
    assert rows == [user_turn_key(_host(text, scheduled=True))]


def test_scan_drops_a_key_a_typed_record_shares(tmp_path: Path) -> None:
    host = _write_jsonl(tmp_path / "h.jsonl", [
        _host(SAME_SESSION_BOTH, scheduled=True),
        _host(SAME_SESSION_BOTH, scheduled=False),
        _host(SCHEDULED_ONLY, scheduled=True),
    ])
    scan = scan_host_sessions([host])
    assert scan.flagged_records == 2
    assert scan.flagged == {(SESSION_A, turn_sha(SCHEDULED_ONLY))}
    assert scan.ambiguous == {(SESSION_A, turn_sha(SAME_SESSION_BOTH))}


def test_dry_run_lists_scheduled_only_beliefs_and_changes_nothing(
    store: MemoryStore, tmp_path: Path,
) -> None:
    host = _seed(store, tmp_path)
    before = _active_ids(store)
    report = retire_scheduled_prompt_beliefs(store, [host], dry_run=True)
    assert [bid for bid, _ in report.candidates] == [
        _belief_id(store, SCHEDULED_ONLY),
    ]
    assert report.locked == [_belief_id(store, LOCKED_SCHEDULED)]
    assert report.retired == 0
    assert _active_ids(store) == before


def test_apply_retires_exactly_the_dry_run_list(
    store: MemoryStore, tmp_path: Path,
) -> None:
    host = _seed(store, tmp_path)
    before = _active_ids(store)
    dry = retire_scheduled_prompt_beliefs(store, [host], dry_run=True)
    applied = retire_scheduled_prompt_beliefs(store, [host], dry_run=False)
    assert applied.candidates == dry.candidates
    assert applied.retired == 1
    retired = {bid for bid, _ in dry.candidates}
    assert _active_ids(store) == before - retired
    for bid in retired:
        b = store.get_belief(bid, include_retired=True)
        assert b is not None and b.valid_to is not None


@pytest.mark.parametrize(
    "kept",
    [TYPED_ONLY, MIXED_SENTENCE, SAME_SESSION_BOTH, OTHER_SESSION_TYPED,
     LOCKED_SCHEDULED],
    ids=["typed", "scheduled-and-typed", "same-session-ambiguous",
         "other-session-typed", "locked"],
)
def test_apply_keeps(store: MemoryStore, tmp_path: Path, kept: str) -> None:
    host = _seed(store, tmp_path)
    bid = _belief_id(store, kept)
    retire_scheduled_prompt_beliefs(store, [host], dry_run=False)
    assert bid in _active_ids(store)


def test_second_apply_is_a_no_op(store: MemoryStore, tmp_path: Path) -> None:
    host = _seed(store, tmp_path)
    retire_scheduled_prompt_beliefs(store, [host], dry_run=False)
    again = retire_scheduled_prompt_beliefs(store, [host], dry_run=False)
    assert again.candidates == []
    assert again.retired == 0


def test_directory_is_searched_for_jsonl(
    store: MemoryStore, tmp_path: Path,
) -> None:
    host = _seed(store, tmp_path)
    report = retire_scheduled_prompt_beliefs(
        store, [tmp_path / "host"], dry_run=True,
    )
    assert report.scan.files_read == 1
    assert [bid for bid, _ in report.candidates] == [
        _belief_id(store, SCHEDULED_ONLY),
    ]
    assert host.parent.parent == tmp_path / "host"


def test_no_flagged_records_selects_nothing(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _seed(store, tmp_path)
    host = _write_jsonl(tmp_path / "typed.jsonl", [
        _host(SCHEDULED_ONLY, scheduled=False),
    ])
    report = retire_scheduled_prompt_beliefs(store, [host], dry_run=False)
    assert report.candidates == []
    assert _belief_id(store, SCHEDULED_ONLY) in _active_ids(store)


def test_format_names_the_beliefs(store: MemoryStore, tmp_path: Path) -> None:
    host = _seed(store, tmp_path)
    text = format_scheduled_prompt_report(
        retire_scheduled_prompt_beliefs(store, [host], dry_run=True),
    )
    assert f"would retire {_belief_id(store, SCHEDULED_ONLY)}" in text
    assert "--apply" in text


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str,
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_cli_dry_run_then_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    try:
        host = _seed(s, tmp_path)
        target = _belief_id(s, SCHEDULED_ONLY)
    finally:
        s.close()
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--retire-scheduled-prompts", str(host),
    )
    assert code == 0
    assert f"would retire {target}" in out
    s = MemoryStore(str(db))
    try:
        assert target in _active_ids(s)
    finally:
        s.close()
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--retire-scheduled-prompts", str(host),
        "--apply",
    )
    assert code == 0
    assert "retired 1 belief(s)." in out
    s = MemoryStore(str(db))
    try:
        assert target not in _active_ids(s)
    finally:
        s.close()


def test_cli_missing_path_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, _ = _run_cli(
        monkeypatch, tmp_path / "memory.db",
        "doctor", "--retire-scheduled-prompts", str(tmp_path / "absent.jsonl"),
    )
    assert code == 2


def test_cli_refuses_a_second_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _write_jsonl(tmp_path / "h.jsonl", [])
    code, _ = _run_cli(
        monkeypatch, tmp_path / "memory.db",
        "doctor", "--retire-scheduled-prompts", str(host),
        "--gc-orphan-feedback",
    )
    assert code == 2
