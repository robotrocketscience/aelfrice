"""#1877: `aelf doctor --retire-relay-rows` retires beliefs stored from relay
text as the user's typed words, and `--restore-relay-rows` undoes it.

The fixture stores are built by the real ingest path, `ingest_jsonl` over a
synthetic `turns.jsonl`, with the two cuts that now keep relay text out
turned off, so the rows look as ingest wrote them before #1875: the relay cut
(`noise_filter.cut_relay`) and the #1649 harness block cut, which takes
`<cross-session-message>` blocks. Nothing here reads a real transcript or
store.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
import aelfrice.ingest as ingest_module
from aelfrice.ingest import ingest_jsonl
from aelfrice.models import (
    ANCHOR_TEXT_MAX_LEN,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_TRANSCRIPT,
)
from aelfrice.noise_filter import (
    RELAY_ROW_NOTICE,
    RELAY_ROW_OPENER,
    relay_row_shape,
)
from aelfrice.relay_rows import (
    SHAPE_BLOCK,
    SOURCE_RELAY_RETIRE,
    format_relay_rows_report,
    restore_relay_rows,
    retire_relay_rows,
)
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore

SESSION = "sess-1877"
OTHER = "sess-1877-b"

TEAM_TAG = '<teammate-message teammate_id="team-lead">'
AGENT_TAG = '<agent-message from="worker-7">'
CROSS_TAG = '<cross-session-message from="uds:/tmp/s.sock" from-name="peer">'
BODY_1 = "The retry wrapper stays as written."
BODY_2 = "The runner config moves to the shared folder."
MINE = "Check the wrapper timing first."
TYPED = "The staging database lives on the second replica host."
NOTICE = '[Cross-session idle notice] "peer", which you asked to be notified about, is idle now.'


def _block(tag: str, *lines: str) -> str:
    name = tag[1:].split()[0].rstrip(">")
    return "\n".join([tag, *lines, f"</{name}>"])


@pytest.fixture
def pre_1875(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest as it ran before #1875 and #1649 cut relay blocks."""
    never = re.compile(r"(?!)")
    monkeypatch.setattr(ingest_module, "cut_relay", lambda text: text)
    monkeypatch.setattr(ingest_module, "_HARNESS_BLOCK_RE", never)
    monkeypatch.setattr(ingest_module, "_UNCLOSED_BLOCK_RE", never)


def _ingest(
    tmp_path: Path, turns: list[tuple[str, str, str]], name: str = "memory.db",
) -> Path:
    """Ingest `(session, ts, text)` user turns into a new store; return its path."""
    log = tmp_path / f"{name}.turns.jsonl"
    log.write_text("".join(
        json.dumps({"schema_version": 1, "role": "user", "session_id": s,
                    "ts": ts, "text": text}) + "\n"
        for s, ts, text in turns
    ), encoding="utf-8")
    db = tmp_path / name
    store = MemoryStore(str(db))
    try:
        ingest_jsonl(store, log)
    finally:
        store.close()
    return db


def _ts(minute: int) -> str:
    return f"2026-09-14T10:{minute:02d}:00Z"


@pytest.fixture
def db(tmp_path: Path, pre_1875: None) -> Path:
    return _ingest(tmp_path, [
        # A relay block opens the record; the user's words follow its closer.
        (SESSION, _ts(1), _block(TEAM_TAG, BODY_1, BODY_2) + "\n" + MINE),
        (SESSION, _ts(2), TYPED),
        # A relay block after the user's first words stays theirs.
        (SESSION, _ts(3), "I asked the peer about the cache.\n"
         + _block(AGENT_TAG, "The cache warmer runs before the first request.")),
        # A tag the user names in a sentence.
        (SESSION, _ts(4), "<teammate-message> blocks are cut by ingest before storage."),
        (SESSION, _ts(5), NOTICE),
        (SESSION, _ts(6), "Ping me when the nightly job finishes."),
    ])


@pytest.fixture
def store(db: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(db))
    try:
        yield s
    finally:
        s.close()


def _belief_id(store: MemoryStore, content: str) -> str:
    row = store._conn.execute(  # noqa: SLF001
        "SELECT id FROM beliefs WHERE content = ?", (content,),
    ).fetchone()
    assert row is not None, content
    return str(row[0])


def _active_ids(store: MemoryStore) -> set[str]:
    return {str(r[0]) for r in store._conn.execute(  # noqa: SLF001
        "SELECT id FROM beliefs WHERE valid_to IS NULL")}


def _origin(store: MemoryStore, content: str) -> str:
    belief = store.get_belief(_belief_id(store, content), include_retired=True)
    assert belief is not None
    return belief.origin


# --- relay_row_shape ------------------------------------------------------


@pytest.mark.parametrize("row", [
    TEAM_TAG, AGENT_TAG, CROSS_TAG, "<teammate-message>", "  <AGENT-MESSAGE>  ",
    "﻿<cross-session-message>",
])
def test_a_relay_tag_alone_is_an_opener_row(row: str) -> None:
    assert relay_row_shape(row) == RELAY_ROW_OPENER


@pytest.mark.parametrize("row", [
    NOTICE,
    '[Cross-session idle notice] "peer", which you asked to be notified about, has exited.',
    "[Cross-session delivery notice] Your message to another session was held by that session.",
])
def test_a_host_notice_is_a_notice_row(row: str) -> None:
    assert relay_row_shape(row) == RELAY_ROW_NOTICE


@pytest.mark.parametrize("row", [
    "<teammate-message> blocks are cut by ingest before storage.",
    f"{TEAM_TAG} {BODY_1}",
    "<agent-message>\nThe body sits on the next line.",
    "</teammate-message>",
    "<agent-message/>",
    "<agent-messages>",
    '<agent-message from="worker-7"',
    f"The host wraps a brief in {TEAM_TAG}",
    "[Cross-session idle notice] the peer went quiet.",
    "[Cross-session delivery notice] something else happened.",
    f"See: {NOTICE}",
    BODY_1,
])
def test_a_lookalike_row_has_no_shape(row: str) -> None:
    assert relay_row_shape(row) is None


# --- the defect the pass repairs -------------------------------------------


def test_the_planted_store_holds_relay_text_as_typed(store: MemoryStore) -> None:
    # The defect: with the old ingest, a relay brief became the user's words.
    for content in (BODY_1, BODY_2, MINE, TYPED):
        assert _origin(store, content) == ORIGIN_USER_TRANSCRIPT


def test_current_ingest_stores_no_relay_text(tmp_path: Path) -> None:
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), _block(TEAM_TAG, BODY_1, BODY_2) + "\n" + MINE),
        (SESSION, _ts(2), TYPED),
    ])
    s = MemoryStore(str(db))
    try:
        contents = {str(r[0]) for r in s._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs")}
    finally:
        s.close()
    assert contents == {MINE, TYPED}


# --- selection ---------------------------------------------------------------


def test_dry_run_selects_the_relay_rows_and_nothing_else(store: MemoryStore) -> None:
    report = retire_relay_rows(store, dry_run=True)
    assert dict(report.rows_by_shape) == {
        RELAY_ROW_OPENER: 1, SHAPE_BLOCK: 2, RELAY_ROW_NOTICE: 1,
    }
    retire = {_belief_id(store, c) for c in (BODY_1, BODY_2, TEAM_TAG, NOTICE)}
    assert set(report.candidates) == retire & {
        b for b in _active_ids(store)
        if store.get_belief(b) is not None
        and store.get_belief(b).origin == ORIGIN_USER_TRANSCRIPT  # type: ignore[union-attr]
    }
    assert {_belief_id(store, BODY_1), _belief_id(store, BODY_2)} <= set(report.candidates)
    assert _belief_id(store, NOTICE) in report.candidates
    for kept in (
        MINE, TYPED, "I asked the peer about the cache.",
        "The cache warmer runs before the first request.",
        "<teammate-message> blocks are cut by ingest before storage.",
        "Ping me when the nightly job finishes.",
    ):
        assert _belief_id(store, kept) not in report.candidates, kept
    assert report.anchors_read == 2
    assert report.anchors_matched == 2


def test_dry_run_writes_nothing(db: Path) -> None:
    before = db.read_bytes()
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        assert report.candidates
        assert report.changed == 0
    finally:
        s.close()
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE valid_to IS NOT NULL").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM feedback_history WHERE source = ?",
            (SOURCE_RELAY_RETIRE,)).fetchone()[0] == 0
    finally:
        conn.close()
    assert len(db.read_bytes()) == len(before)


def test_a_relay_row_after_a_typed_row_in_one_group_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    # A bulk ingest can give several records one timestamp. The second
    # record's opener can't be told from a relay inside the first record.
    shared = _ts(7)
    db = _ingest(tmp_path, [
        (SESSION, shared, "The backup script compresses archives with zstd."),
        (SESSION, shared, _block(CROSS_TAG, "The deploy key sits in the project vault.")),
        (SESSION, shared, "The release checklist requires a signed tag."),
        (SESSION, _ts(8), TYPED),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
    finally:
        s.close()
    assert sum(report.rows_by_shape.values()) == 0
    assert report.candidates == []


def test_typed_rows_in_a_relay_rows_group_are_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    # The relay record starts the group and typed records share its
    # timestamp. Only the rows its anchor shows are taken.
    shared = _ts(10)
    relay = "The audit log rotates at midnight."
    db = _ingest(tmp_path, [
        (SESSION, shared, _block(AGENT_TAG, relay)),
        (SESSION, shared, "The rotation keeps seven days of history."),
        (SESSION, shared, "The archive moves to cold storage weekly."),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        assert dict(report.rows_by_shape) == {RELAY_ROW_OPENER: 1, SHAPE_BLOCK: 1}
        assert _belief_id(s, relay) in report.candidates
        for typed in ("The rotation keeps seven days of history.",
                      "The archive moves to cold storage weekly."):
            assert _belief_id(s, typed) not in report.candidates
    finally:
        s.close()


def test_a_notice_inside_a_typed_record_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"Ping me later.\n{NOTICE}"),
        (SESSION, _ts(2), TYPED),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
    finally:
        s.close()
    assert sum(report.rows_by_shape.values()) == 0


def test_an_anchor_matches_rows_only_in_a_session_that_derived_both_ends(
    tmp_path: Path, pre_1875: None,
) -> None:
    # Session B typed the same sentences after an empty block, so B derived
    # the edge's target but not its source. Session A's anchor must not
    # reach B's rows.
    empty = TEAM_TAG + "\n</teammate-message>\n"
    db = _ingest(tmp_path, [
        (OTHER, _ts(1), empty + "\n".join([BODY_1, BODY_2, MINE])),
        (SESSION, _ts(2), _block(TEAM_TAG, BODY_1, BODY_2) + "\n" + MINE),
        (SESSION, _ts(3), TYPED),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        assert dict(report.rows_by_shape) == {RELAY_ROW_OPENER: 2, SHAPE_BLOCK: 2}
        # B's rows derived the same beliefs, so those stay.
        assert _belief_id(s, BODY_1) not in report.candidates
    finally:
        s.close()


def test_a_capped_anchor_takes_the_rows_before_the_cap(
    tmp_path: Path, pre_1875: None,
) -> None:
    body = [f"Step {n} of the migration copies table {n} to the new host."
            for n in range(1, 30)]
    record = _block(TEAM_TAG, *body) + "\n" + MINE
    assert len(record) > ANCHOR_TEXT_MAX_LEN
    visible = record[:ANCHOR_TEXT_MAX_LEN]
    # The cap ends inside a sentence, so the anchor's last one is cut short.
    assert any(b not in visible and b[:20] in visible for b in body)
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), record),
        (SESSION, _ts(2), TYPED),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        shown = [b for b in body if b in visible]
        assert report.rows_by_shape[SHAPE_BLOCK] == len(shown)
        assert report.rows_by_shape[SHAPE_BLOCK] >= 10
        assert _belief_id(s, MINE) not in report.candidates
        assert _belief_id(s, body[-1]) not in report.candidates
    finally:
        s.close()


def test_a_run_that_differs_from_its_anchor_at_the_end_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    # The last record opens like the first but ends with other words, and
    # no later turn anchors it.
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), _block(TEAM_TAG, BODY_1, BODY_2)),
        (SESSION, _ts(2), TYPED),
        (SESSION, _ts(3), TEAM_TAG + "\n" + BODY_1 + "\n" + MINE),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
    finally:
        s.close()
    assert dict(report.rows_by_shape) == {RELAY_ROW_OPENER: 2, SHAPE_BLOCK: 2}


def test_rows_from_other_source_kinds_are_not_read(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        s.record_ingest(
            source_kind="filesystem", source_path="docs/relay.md",
            raw_text=TEAM_TAG, session_id=SESSION, ts=_ts(1),
        )
        report = retire_relay_rows(s, dry_run=True)
    finally:
        s.close()
    assert report.rows_read == 0
    assert sum(report.rows_by_shape.values()) == 0


def test_a_relay_record_with_no_next_turn_gives_only_its_opener(
    tmp_path: Path, pre_1875: None,
) -> None:
    # No later turn, no anchor, so the body rows can't be bounded.
    db = _ingest(tmp_path, [(SESSION, _ts(1), _block(TEAM_TAG, BODY_1, BODY_2))])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
    finally:
        s.close()
    assert dict(report.rows_by_shape) == {RELAY_ROW_OPENER: 1}


def test_a_belief_a_typed_row_also_derived_is_kept(
    tmp_path: Path, pre_1875: None,
) -> None:
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), _block(TEAM_TAG, BODY_1, BODY_2)),
        (SESSION, _ts(2), BODY_1),
    ])
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        assert report.rows_by_shape[SHAPE_BLOCK] == 2
        assert _belief_id(s, BODY_1) not in report.candidates
        assert _belief_id(s, BODY_2) in report.candidates
    finally:
        s.close()


def test_a_locked_relay_belief_is_kept(store: MemoryStore) -> None:
    bid = _belief_id(store, BODY_1)
    belief = store.get_belief(bid)
    assert belief is not None
    belief.lock_level = LOCK_USER
    belief.locked_at = "2026-10-12T00:00:00Z"
    store.update_belief(belief)
    report = retire_relay_rows(store, dry_run=False)
    assert bid in report.locked
    assert bid not in report.candidates
    assert bid in _active_ids(store)


def test_an_agent_inferred_relay_belief_is_kept_as_it_is(
    tmp_path: Path, pre_1875: None,
) -> None:
    # Rows under another source label give agent_inferred.
    log = tmp_path / "turns.jsonl"
    log.write_text("".join(json.dumps({
        "schema_version": 1, "role": "user", "session_id": SESSION,
        "ts": _ts(m), "text": text,
    }) + "\n" for m, text in (
        (1, _block(TEAM_TAG, BODY_1)), (2, TYPED),
    )), encoding="utf-8")
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(s, log, source_label="session-end")
        assert _origin(s, BODY_1) == ORIGIN_AGENT_INFERRED
        report = retire_relay_rows(s, dry_run=False)
        assert _belief_id(s, BODY_1) in report.agent_inferred
        assert report.candidates == []
    finally:
        s.close()


# --- apply, undo, replay, doctor ----------------------------------------------


def _replay_clean(store: MemoryStore) -> None:
    report = replay_full_equality(store)
    assert not report.has_drift, report.drift_examples
    assert report.total_log_rows > 0


def test_apply_retires_and_restore_undoes_with_replay_clean(store: MemoryStore) -> None:
    _replay_clean(store)
    active = _active_ids(store)
    report = retire_relay_rows(store, dry_run=False)
    assert report.changed == len(report.candidates) > 0
    assert _active_ids(store) == active - set(report.candidates)
    for bid in report.candidates:
        sources = [e.source for e in store.list_feedback_events(bid)]
        assert sources == [SOURCE_RELAY_RETIRE]
        belief = store.get_belief(bid, include_retired=True)
        assert belief is not None
        assert belief.origin == ORIGIN_USER_TRANSCRIPT
    _replay_clean(store)
    assert retire_relay_rows(store, dry_run=False).candidates == []
    preview = restore_relay_rows(store, dry_run=True)
    assert preview.candidates == report.candidates
    assert _active_ids(store) == active - set(report.candidates)
    undo = restore_relay_rows(store, dry_run=False)
    assert undo.changed == len(report.candidates)
    assert _active_ids(store) == active
    assert restore_relay_rows(store, dry_run=True).candidates == []
    _replay_clean(store)
    # A restored belief carries `user_restored`, so a later pass keeps it.
    again = retire_relay_rows(store, dry_run=False)
    assert again.candidates == []
    assert set(again.other_evidence) == set(report.candidates)
    assert restore_relay_rows(store, dry_run=False).changed == 0


def test_demoting_the_origin_would_break_replay(store: MemoryStore) -> None:
    # Why the pass retires: replay derives the origin from the log row, so
    # an origin rewritten in place reads as drift.
    belief = store.get_belief(_belief_id(store, BODY_1))
    assert belief is not None
    belief.origin = ORIGIN_AGENT_INFERRED
    store.update_belief(belief)
    assert replay_full_equality(store).has_drift


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, db: Path, *argv: str,
) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _doctor_clean(monkeypatch: pytest.MonkeyPatch, db: Path, tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {}}), encoding="utf-8")
    code, out = _run_cli(
        monkeypatch, db, "doctor", "--user-settings", str(settings),
        "--project-root", str(tmp_path / "noproj"),
    )
    assert code == 0, out
    code, out = _run_cli(monkeypatch, db, "doctor", "--replay")
    assert code == 0, out


def test_cli_dry_run_apply_and_restore(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _doctor_clean(monkeypatch, db, tmp_path)
    s = MemoryStore(str(db))
    try:
        target = _belief_id(s, BODY_1)
        texts = [str(r[0]) for r in s._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs")]
    finally:
        s.close()
    code, out = _run_cli(monkeypatch, db, "doctor", "--retire-relay-rows")
    assert code == 0
    assert f"store: {db}" in out
    assert "relay rows: 4 (1 opener, 1 notice, 2 block)" in out
    assert f"  {target}\n" in out
    assert "dry run: re-run with --apply" in out
    # The report names ids and counts, never a belief's text.
    for text in texts:
        assert text not in out
    code, out = _run_cli(monkeypatch, db, "doctor", "--retire-relay-rows", "--apply")
    assert code == 0
    assert "--restore-relay-rows --apply` undoes this pass" in out
    _doctor_clean(monkeypatch, db, tmp_path)
    code, out = _run_cli(monkeypatch, db, "doctor", "--restore-relay-rows")
    assert code == 0
    assert f"  {target}\n" in out
    assert "would restore" in out
    code, out = _run_cli(monkeypatch, db, "doctor", "--restore-relay-rows", "--apply")
    assert code == 0
    assert "restored " in out
    _doctor_clean(monkeypatch, db, tmp_path)
    s = MemoryStore(str(db))
    try:
        assert target in _active_ids(s)
    finally:
        s.close()


def test_cli_refuses_two_passes_at_once(
    db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    code, _ = _run_cli(
        monkeypatch, db, "doctor", "--retire-relay-rows", "--restore-relay-rows",
    )
    assert code == 2
    assert "separate passes" in capsys.readouterr().err


def test_format_names_no_belief_text(store: MemoryStore) -> None:
    report = retire_relay_rows(store, dry_run=True)
    text = format_relay_rows_report(report, "x.db")
    for content in (BODY_1, BODY_2, NOTICE, TEAM_TAG):
        assert content not in text
