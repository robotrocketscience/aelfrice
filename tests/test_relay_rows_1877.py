"""#1877: `aelf doctor --retire-relay-rows` retires beliefs stored from relay
text as the user's typed words, and `--restore-relay-rows` undoes it.

The fixture stores are built by the real ingest path, `ingest_jsonl` over a
synthetic `turns.jsonl`. `pre_1875` turns ingest back to how it ran before
#1649, which split and anchored the raw record, and dates the rows it writes
before the pass's cutoff, so the rows look as the stored relay rows do.
`between_1649_and_1875` keeps today's speech filter and clock and turns off
only the two relay cuts, the way ingest ran from #1649 until #1875. Nothing
here reads a real transcript or store.
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
import aelfrice.store as store_module
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
    relay_cut_is_host_shaped,
    relay_row_shape,
)
from aelfrice.relay_rows import (
    RAW_RECORD_CUTOFF_MS,
    SHAPE_BLOCK,
    SOURCE_RELAY_RETIRE,
    format_relay_rows_report,
    restore_relay_rows,
    retire_relay_rows,
)
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore
from aelfrice.ulid import make_generator

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


# 2026-09-14T10:00:00Z, two weeks before the pass's cutoff.
_PRE_1649_CLOCK = 1_789_380_000.0


@pytest.fixture
def pre_1875(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest as it ran before #1649: the raw record split and anchored."""
    assert _PRE_1649_CLOCK * 1000 < RAW_RECORD_CUTOFF_MS
    monkeypatch.setattr(
        ingest_module, "_user_speech", lambda text: text if text.strip() else None,
    )
    monkeypatch.setattr(store_module, "ulid", make_generator(
        rand_source=lambda n: bytes(n), time_source=lambda: _PRE_1649_CLOCK,
    ))


@pytest.fixture
def between_1649_and_1875(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest from #1649 until #1875: speech filtered, relay cuts off, today's clock."""
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
    TEAM_TAG, AGENT_TAG, CROSS_TAG, '  <AGENT-MESSAGE FROM="w">  ',
    "\ufeff" + CROSS_TAG,
    '<teammate-message teammate_id="lead" summary="ruling">',
])
def test_a_relay_tag_alone_is_an_opener_row(row: str) -> None:
    assert relay_row_shape(row) == RELAY_ROW_OPENER


@pytest.mark.parametrize("row", [
    NOTICE,
    '[Cross-session idle notice] "peer", which you asked to be notified about, has exited.',
    "[Cross-session delivery notice] Your message to another session was held by that session.",
    "[Cross-session delivery notice] Your message to another session was refused",
])
def test_a_host_notice_is_a_notice_row(row: str) -> None:
    assert relay_row_shape(row) == RELAY_ROW_NOTICE


@pytest.mark.parametrize("row", [
    "<teammate-message> blocks are cut by ingest before storage.",
    f"{TEAM_TAG} {BODY_1}",
    AGENT_TAG + "\nThe body sits on the next line.",
    # A tag a person types bare, or without every attribute the host writes.
    "<teammate-message>", "<agent-message>", "  <AGENT-MESSAGE>  ",
    "<cross-session-message>", '<cross-session-message from="uds:/tmp/s.sock">',
    '<teammate-message from="lead">', '<agent-message to="worker-7">',
    # Words after the notice template's last fixed word may be the user's.
    NOTICE[:-1] + " and the cache must be warmed first.",
    "[Cross-session delivery notice] Your message to another session was refused twice today.",
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


# --- review round: typed tags, notices, dropped first lines, spans ----------


Q = "I want to understand why ingest cuts this tag at all."
R = "My own note says the cache must be warmed before traffic."
NEXT = "Ping me when the nightly job finishes tonight."


def _dry(db: Path) -> tuple[dict[str, int], set[str]]:
    s = MemoryStore(str(db))
    try:
        report = retire_relay_rows(s, dry_run=True)
        contents: set[str] = set()
        for bid in report.candidates:
            belief = s.get_belief(bid)
            assert belief is not None
            contents.add(belief.content)
        return dict(report.rows_by_shape), contents
    finally:
        s.close()


@pytest.mark.parametrize("tag", ["<agent-message>", "<teammate-message>"])
def test_a_bare_typed_tag_with_typed_lines_after_it_is_not_taken(
    tmp_path: Path, pre_1875: None, tag: str,
) -> None:
    # No closer, so the next turn's anchor runs to the end of the record.
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"{tag}\n{Q}\n{R}"),
        (SESSION, _ts(2), NEXT),
    ])
    assert _dry(db) == ({}, set())


def test_a_bare_typed_tag_before_many_typed_lines_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    typed = [f"Typed sentence number {i} explains the plan in detail." for i in range(40)]
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), "<agent-message>\n" + " ".join(typed)),
        (SESSION, _ts(2), NEXT),
    ])
    assert _dry(db) == ({}, set())


@pytest.mark.parametrize("first", [
    f"Context:\n{CROSS_TAG}\n{Q}\n{R}",
    f"Context:\n{NOTICE}\n{R}",
])
def test_a_record_a_dropped_line_opens_is_not_taken(
    tmp_path: Path, pre_1875: None, first: str,
) -> None:
    # `cut_relay` keeps a record whose relay text follows a line, even one
    # the sentence splitter drops, and the next turn's anchor shows that line.
    db = _ingest(tmp_path, [(SESSION, _ts(1), first), (SESSION, _ts(2), NEXT)])
    assert _dry(db) == ({}, set())


def test_typed_words_after_a_notice_on_its_line_are_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    line = NOTICE[:-1] + " and my own note says the cache must be warmed."
    db = _ingest(tmp_path, [(SESSION, _ts(1), line), (SESSION, _ts(2), NEXT)])
    assert _dry(db) == ({}, set())


def test_an_inline_span_across_the_closer_takes_no_typed_words(
    tmp_path: Path, pre_1875: None,
) -> None:
    # The span joins the relay line to the user's line, so the cut part's
    # sentences aren't the record's first sentences.
    typed = "The user typed this sentence after the block today."
    record = (
        f"{CROSS_TAG}\nRelay mentions `code that spans\n</cross-session-message>\n"
        f"{typed}\nAnd a second typed sentence follows it` here."
    )
    db = _ingest(tmp_path, [(SESSION, _ts(1), record), (SESSION, _ts(2), NEXT)])
    shapes, contents = _dry(db)
    assert shapes == {RELAY_ROW_OPENER: 1}
    assert not any("typed" in c for c in contents), contents


def test_rows_stored_without_inline_spans_are_matched(
    tmp_path: Path, pre_1875: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before #1874 a stored sentence lost its inline code; the anchor kept it.
    monkeypatch.setattr(ingest_module, "is_typed_by_user", lambda *_: False)
    body = "Run `make check` before the release tag goes out."
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), _block(TEAM_TAG, body)),
        (SESSION, _ts(2), NEXT),
    ])
    shapes, _ = _dry(db)
    assert shapes == {RELAY_ROW_OPENER: 1, SHAPE_BLOCK: 1}


def test_a_capped_anchor_skips_a_session_that_derived_only_its_source(
    tmp_path: Path, pre_1875: None,
) -> None:
    # Session B types the next turn's sentence, then opens a last record with
    # the same visible relay lines and ends differently. B never derived the
    # edge's target, and no later turn anchors its record.
    body = [f"Step {n} of the migration copies table {n} to the new host."
            for n in range(1, 30)]
    record = _block(TEAM_TAG, *body) + "\n" + MINE
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), record),
        (SESSION, _ts(2), TYPED),
        (OTHER, _ts(3), TYPED),
        (OTHER, _ts(4), "\n".join([TEAM_TAG, *body[:20]]) + "\nThe end differs here today."),
    ])
    shapes, _ = _dry(db)
    visible = record[:ANCHOR_TEXT_MAX_LEN]
    assert shapes[SHAPE_BLOCK] == sum(1 for b in body if b in visible)


def test_a_second_label_ingest_of_one_record_starts_its_own_turn(
    tmp_path: Path, pre_1875: None,
) -> None:
    # The same record under `transcript` and then `session-end` shares its
    # session and timestamp; the label keeps the two runs apart.
    relay = _block(TEAM_TAG, BODY_1)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    rec = {"schema_version": 1, "role": "user", "session_id": SESSION}
    first.write_text(json.dumps({**rec, "ts": _ts(1), "text": relay}) + "\n",
                     encoding="utf-8")
    second.write_text("".join(json.dumps({**rec, "ts": ts, "text": text}) + "\n"
                              for ts, text in ((_ts(1), relay), (_ts(2), NEXT))),
                      encoding="utf-8")
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(s, first)
        ingest_jsonl(s, second, source_label="session-end")
        report = retire_relay_rows(s, dry_run=True)
        assert _belief_id(s, BODY_1) in report.candidates
    finally:
        s.close()


def test_restore_reads_the_latest_retire_of_a_belief(store: MemoryStore) -> None:
    # Restored without feedback, the belief is a candidate again, and its
    # second retire is the one restore has to match.
    target = _belief_id(store, BODY_1)
    retire_relay_rows(store, dry_run=False)
    assert store.restore_belief(target)
    assert target in retire_relay_rows(store, dry_run=False).candidates
    assert len([e for e in store.list_feedback_events(target)
                if e.source == SOURCE_RELAY_RETIRE]) == 2
    assert target in restore_relay_rows(store, dry_run=True).candidates


def test_restore_leaves_a_belief_retired_again_another_way(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = MemoryStore(str(db))
    try:
        target = _belief_id(s, BODY_1)
        retire_relay_rows(s, dry_run=False)
        restore_relay_rows(s, dry_run=False)
    finally:
        s.close()
    code, _ = _run_cli(monkeypatch, db, "retire", target)
    assert code == 0
    s = MemoryStore(str(db))
    try:
        assert target not in restore_relay_rows(s, dry_run=True).candidates
        assert restore_relay_rows(s, dry_run=False).changed == 0
        assert target not in _active_ids(s)
    finally:
        s.close()


# --- second review round: spoofed attributes, vetoes, last-turn notices ------


T1 = "My staging database password rotates every Tuesday morning."
T2 = "I keep the backup keys in the hardware safe downstairs."


@pytest.mark.parametrize("tag", [
    '<agent-message title="notes from=me">',
    "<agent-message title='notes from=me'>",
    '<agent-message title="notes  from=me  ">',
    '<agent-message title="from=">',
    '<agent-message from>',
    '<cross-session-message from="uds:/tmp/s.sock" title="from-name=x">',
    '<cross-session-message from-name="from=uds:/tmp/s.sock">',
    '<teammate-message summary="teammate_id=lead">',
])
def test_an_attribute_name_inside_a_value_is_not_an_attribute(tag: str) -> None:
    assert relay_row_shape(tag) is None


@pytest.mark.parametrize("tag", [
    '<agent-message title="notes from=me">',
    "<agent-message title='notes from=me'>",
    '<cross-session-message from-name="from=uds:/tmp/s.sock">',
])
def test_a_spoofed_tag_with_typed_lines_after_it_is_not_taken(
    tmp_path: Path, pre_1875: None, tag: str,
) -> None:
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"{tag}\n{T1}\n{T2}"),
        (SESSION, _ts(2), NEXT),
    ])
    assert _dry(db) == ({}, set())


@pytest.mark.parametrize("rec", [
    f"<agent-message>\n{T1}\n{T2}",
    f"<agent-message>\n{T1}\n</agent-message>\n{T2}",
    f"<Teammate-Message>\n{T1}\n{T2}",
    f"<cross-session-message>\n{T1}\n{T2}",
    f"{CROSS_TAG}\nhi there friend.\n</cross-session-message>\n<agent-message>\n{T1}",
    "<agent-message>\n" + " ".join(f"Typed line {i} of my own plan here." for i in range(60)),
])
@pytest.mark.parametrize("next_turn", [True, False])
def test_a_bare_tag_never_takes_typed_words(
    tmp_path: Path, pre_1875: None, rec: str, next_turn: bool,
) -> None:
    turns = [(SESSION, _ts(1), rec)]
    if next_turn:
        turns.append((SESSION, _ts(2), NEXT))
    _, contents = _dry(_ingest(tmp_path, turns))
    assert not any("Typed" in c or c in (T1, T2) for c in contents), contents


@pytest.mark.parametrize("next_turn", [True, False])
def test_a_notice_row_needs_the_record_to_show_its_line(
    tmp_path: Path, pre_1875: None, next_turn: bool,
) -> None:
    # Typed words follow the template on the same line, in a sentence of
    # their own. With no next turn, nothing shows the line.
    line = ("[Cross-session delivery notice] Your message to another session was "
            "refused. I think the peer session crashed overnight again.")
    turns = [(SESSION, _ts(1), line)]
    if next_turn:
        turns.append((SESSION, _ts(2), NEXT))
    assert _dry(_ingest(tmp_path, turns)) == ({}, set())


def test_a_notice_record_with_no_next_turn_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    assert _dry(_ingest(tmp_path, [(SESSION, _ts(1), NOTICE)])) == ({}, set())


def test_a_veto_holds_against_another_anchor(
    tmp_path: Path, pre_1875: None,
) -> None:
    # The first record opens with a line the splitter drops, so `cut_relay`
    # keeps it whole. The second has the same rows and opens with the tag.
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"Context:\n{CROSS_TAG}\n{T1}\n{T2}"),
        (SESSION, _ts(2), NEXT),
        (SESSION, _ts(3), f"{CROSS_TAG}\n{T1}\n{T2}"),
        (SESSION, _ts(4), "A last sentence the user typed today."),
    ])
    # The first anchor matches both records' rows, since they're the same,
    # and vetoes both, so the second anchor takes none of them.
    assert _dry(db) == ({}, set())


def test_a_capped_record_a_dropped_line_opens_is_not_taken(
    tmp_path: Path, pre_1875: None,
) -> None:
    typed = " ".join(f"Typed sentence {i} lays out my own migration plan." for i in range(40))
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"Context:\n{CROSS_TAG}\n{typed}"),
        (SESSION, _ts(2), NEXT),
    ])
    assert _dry(db) == ({}, set())


def test_a_host_relay_quoting_a_malformed_tag_is_skipped_whole(
    tmp_path: Path, pre_1875: None,
) -> None:
    # A known miss: the self-closed tag in the body fails the host check, so
    # the anchor vetoes the record's rows, the opener's included.
    rec = (f"{CROSS_TAG}\nThe docs show <agent-message/> as an example here.\n"
           f"{BODY_1}\n</cross-session-message>\n{T1}")
    db = _ingest(tmp_path, [(SESSION, _ts(1), rec), (SESSION, _ts(2), NEXT)])
    assert _dry(db) == ({}, set())


def test_a_run_stays_inside_its_group(tmp_path: Path, pre_1875: None) -> None:
    # Rows split across two timestamps, as a different writer could store
    # them. The anchor's sentences continue past the first group.
    from aelfrice.derivation_worker import run_worker
    from aelfrice.models import EDGE_DERIVED_FROM, Edge

    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        meta = {"role": "user", "call_site": "transcript_ingest"}
        for ts, text in ((_ts(1), CROSS_TAG), (_ts(1), BODY_1),
                         (_ts(2), BODY_2), (_ts(2), T1)):
            s.record_ingest(
                source_kind="transcript", source_path="transcript", raw_text=text,
                session_id=SESSION, ts=ts, raw_meta=meta,
            )
        run_worker(s)
        s.insert_edge(Edge(
            src=_belief_id(s, T1), dst=_belief_id(s, BODY_1),
            type=EDGE_DERIVED_FROM, weight=1.0,
            anchor_text=f"{CROSS_TAG}\n{BODY_1}\n{BODY_2}",
        ))
        report = retire_relay_rows(s, dry_run=True)
        assert _belief_id(s, BODY_2) not in report.candidates
    finally:
        s.close()


def test_a_belief_a_peer_owns_is_neither_retired_nor_restored(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _belief_id(store, BODY_1)
    peer = {target: "peer-scope"}
    monkeypatch.setattr(store, "find_foreign_owner", lambda bid: peer.get(bid))
    report = retire_relay_rows(store, dry_run=False)
    assert report.foreign == [target]
    assert "owned by a federation peer, kept: 1" in format_relay_rows_report(report, "x.db")
    assert target not in report.candidates
    assert target in _active_ids(store)
    peer.clear()
    retire_relay_rows(store, dry_run=False)
    assert target not in _active_ids(store)
    peer[target] = "peer-scope"
    assert target not in restore_relay_rows(store, dry_run=True).candidates


def test_undo_leaves_only_the_audit_trail_behind(db: Path) -> None:
    # What undo can't take back: each belief's two feedback rows, its version
    # counter, its rewritten search row, and the store generation.
    def dump() -> dict[str, list[str]]:
        conn = sqlite3.connect(str(db))
        try:
            names = [str(r[0]) for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'")]
            out: dict[str, list[str]] = {}
            for name in names:
                try:
                    rows = conn.execute(f'SELECT rowid, * FROM "{name}"').fetchall()
                except sqlite3.DatabaseError:
                    rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()
                out[name] = sorted(map(repr, rows))
            return out
        finally:
            conn.close()

    before = dump()
    s = MemoryStore(str(db))
    try:
        retired = retire_relay_rows(s, dry_run=False).candidates
        assert restore_relay_rows(s, dry_run=False).changed == len(retired)
        sources = sorted(
            (e.source, e.valence) for b in retired for e in s.list_feedback_events(b)
        )
    finally:
        s.close()
    after = dump()
    changed = {t for t in set(before) | set(after) if before.get(t) != after.get(t)}
    assert {t for t in changed if not t.startswith("beliefs_fts")} == {
        "belief_versions", "feedback_history", "schema_meta",
    }
    assert sources == sorted(
        [(SOURCE_RELAY_RETIRE, 0.0)] * len(retired)
        + [("user_restored", 1.0)] * len(retired)
    )


# --- third review round: line ends, raw records, host tag form ----------------


REFUSED = "[Cross-session delivery notice] Your message to another session was refused."
CRASHED = "I think the peer session crashed overnight again."
EOLS = ["\n", "\r\n"]
LEADS = [
    "<system-reminder>host note here</system-reminder>\n",
    "[Request interrupted by user]\n",
]


@pytest.mark.parametrize("eol", EOLS)
def test_typed_words_after_a_notice_on_its_line_stay_with_any_line_end(
    tmp_path: Path, pre_1875: None, eol: str,
) -> None:
    rec = f"{REFUSED} {CRASHED}{eol}Ping me later about it today."
    db = _ingest(tmp_path, [(SESSION, _ts(1), rec), (SESSION, _ts(2), NEXT)])
    _, contents = _dry(db)
    assert not any(CRASHED in c for c in contents), contents


@pytest.mark.parametrize("eol", EOLS)
def test_a_notice_record_gives_its_notice_row_with_any_line_end(
    tmp_path: Path, pre_1875: None, eol: str,
) -> None:
    db = _ingest(tmp_path, [(SESSION, _ts(1), f"{REFUSED}{eol}{T1}"), (SESSION, _ts(2), NEXT)])
    shapes, contents = _dry(db)
    assert shapes == {RELAY_ROW_NOTICE: 1}
    assert T1 not in contents


@pytest.mark.parametrize("eol", EOLS)
def test_a_typed_template_notice_takes_only_its_own_row(
    tmp_path: Path, pre_1875: None, eol: str,
) -> None:
    # A known gap: a person who types a notice's fixed words exactly loses
    # that row, and only that row.
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"{REFUSED}{eol}{T1}{eol}{T2}"), (SESSION, _ts(2), NEXT),
    ])
    shapes, contents = _dry(db)
    assert shapes == {RELAY_ROW_NOTICE: 1}
    assert T1 not in contents and T2 not in contents


@pytest.mark.parametrize("eol", EOLS)
def test_a_notice_first_veto_holds_against_another_anchor(
    tmp_path: Path, pre_1875: None, eol: str,
) -> None:
    db = _ingest(tmp_path, [
        (SESSION, _ts(1), f"Context:{eol}{REFUSED}{eol}{T1}"),
        (SESSION, _ts(2), NEXT),
        (SESSION, _ts(3), f"{REFUSED}{eol}{T1}"),
        (SESSION, _ts(4), "A last sentence the user typed today."),
    ])
    assert _dry(db) == ({}, set())


@pytest.mark.parametrize("eol", EOLS)
def test_a_notice_on_the_last_turn_is_not_taken_with_any_line_end(
    tmp_path: Path, pre_1875: None, eol: str,
) -> None:
    db = _ingest(tmp_path, [(SESSION, _ts(1), f"{REFUSED}{eol}{T1}")])
    assert _dry(db) == ({}, set())


@pytest.mark.parametrize("tag", [AGENT_TAG, TEAM_TAG, CROSS_TAG])
@pytest.mark.parametrize("lead", LEADS)
def test_a_relay_behind_a_lead_line_is_not_taken(
    tmp_path: Path, pre_1875: None, lead: str, tag: str,
) -> None:
    # `cut_relay` keeps a relay behind a reminder or a marker line.
    rec = f"{lead}{tag}\n{T1}\n{T2}"
    db = _ingest(tmp_path, [(SESSION, _ts(1), rec), (SESSION, _ts(2), NEXT)])
    _, contents = _dry(db)
    assert T1 not in contents and T2 not in contents


@pytest.mark.parametrize("tag", [AGENT_TAG, TEAM_TAG])
@pytest.mark.parametrize("lead", ["", *LEADS])
def test_rows_written_after_the_cutoff_are_not_taken(
    tmp_path: Path, between_1649_and_1875: None, lead: str, tag: str,
) -> None:
    # From #1649 on the anchor is the filtered speech, so a relay behind a
    # reminder looks like one that opens the record. The row times say so.
    rec = f"{lead}{tag}\n{T1}\n{T2}"
    db = _ingest(tmp_path, [(SESSION, _ts(1), rec), (SESSION, _ts(2), NEXT)])
    assert _dry(db) == ({}, set())


def test_a_later_ingest_of_the_record_stops_the_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pre-#1649 rows match the anchor, but an ingest after the cutoff
    # stored the same first sentence, so the anchor may be its speech.
    log = tmp_path / "turns.jsonl"
    rec = {"schema_version": 1, "role": "user", "session_id": SESSION}
    log.write_text("".join(json.dumps({**rec, "ts": ts, "text": text}) + "\n" for ts, text in (
        (_ts(1), _block(TEAM_TAG, BODY_1, BODY_2)), (_ts(2), NEXT),
    )), encoding="utf-8")
    s = MemoryStore(str(tmp_path / "memory.db"))
    try:
        with monkeypatch.context() as m:
            m.setattr(ingest_module, "_user_speech", lambda t: t if t.strip() else None)
            m.setattr(store_module, "ulid", make_generator(
                rand_source=lambda n: bytes(n), time_source=lambda: _PRE_1649_CLOCK,
            ))
            ingest_jsonl(s, log)
        before = retire_relay_rows(s, dry_run=True)
        assert before.rows_by_shape[SHAPE_BLOCK] == 2
        with monkeypatch.context() as m:
            m.setattr(ingest_module, "cut_relay", lambda text: text)
            ingest_jsonl(s, log, source_label="session-end")
        after = retire_relay_rows(s, dry_run=True)
        assert after.rows_by_shape[SHAPE_BLOCK] == 0
        assert after.rows_by_shape[RELAY_ROW_OPENER] == 1
    finally:
        s.close()


@pytest.mark.parametrize("tag", [
    "<agent-message from=worker-7>",
    '<agent-message title=a from="x">',
    '<agent-message from="">',
    "<agent-message from='worker-7'>",
    '<agent-message  from="worker-7">',
    '<agent-message from="worker-7" >',
    '<agent-message\tfrom="worker-7">',
    '<agent-message from="worker-7">',
    '<agent-message from="worker-7">',
    '<agent-message　from="worker-7">',
    '<agent-message\u0085from="worker-7">',
    '<agent-message\x0cfrom="worker-7">',
    '<cross-session-message from="a" from-name="b">',
])
def test_a_tag_outside_the_host_form_is_not_an_opener(tag: str) -> None:
    assert relay_row_shape(tag) is None
    assert not relay_cut_is_host_shaped(f"{tag}\n{T1}\n")


def test_the_host_form_takes_attributes_in_any_order() -> None:
    tag = '<cross-session-message from-name="peer" via="uds" from="uds:/tmp/s.sock">'
    assert relay_row_shape(tag) == RELAY_ROW_OPENER


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
