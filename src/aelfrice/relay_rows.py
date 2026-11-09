"""Retire beliefs stored from relay text as the user's typed words (#1877).

Before #1875 ingest stored relay text that opened a user record, such as
another session's message, as the user's own words, with
`origin=user_transcript`. #1875 stops new rows. This pass finds the stored
ones with the rule `noise_filter.cut_relay` applies to a record, read
through what the store kept.

`cut_relay` reads the raw record, and the store doesn't keep one. Before #1649
ingest split the raw record into rows and anchored the next turn's
DERIVED_FROM edge on it, line breaks kept, up to 1,000 characters. From #1649
on it does both with the speech that `ingest._user_speech` leaves, after
reminders, host marker lines, harness blocks and pasted output are gone, so a
relay behind a reminder, which `cut_relay` keeps, looks like one that opens
the record. A row's id is a ULID that carries the time ingest wrote it. The
pass reads only rows written before `RAW_RECORD_CUTOFF_MS`, the time of
#1649's first commit. A later row, or one whose id carries no time, adds
nothing to what the pass takes: it can only stop an anchor whose first
sentence it holds in its session, or keep a belief it also derived.

An `ingest_log` row holds one sentence, so it can't show line breaks, a
closer, or where its record began. A stretch is a run of rows, in id order,
with one `(session_id, ts, source_path)`: one turn, or several that a bulk
ingest gave one timestamp. A row starts its turn when it opens a stretch
whose key no earlier stretch had, or had from the same first row, as when a
transcript is ingested again. The part of a turn after another writer's row
opens nothing. A stretch holding a row from after the cutoff, or one with no
time, isn't read. Three shapes are read:

- `opener`: a row that starts its turn and holds a relay opening tag and
  nothing else, in the form the host writes it, with its attributes
  (`noise_filter.relay_row_shape`).
- `notice` and `block`: the rows of a record that a DERIVED_FROM anchor shows
  opening with relay text. `cut_relay` reads the anchor as it reads a record,
  and every relay tag and notice line in the part it takes must be the host's
  (`noise_filter.relay_cut_is_host_shaped`). The anchor's sentences are
  matched in order to a run of rows that opens at a row that starts its turn,
  takes that row, and fills its stretch, or stays in it when the anchor is
  capped. Ingest anchors the edge `src -> dst` on the turn whose last belief
  is `dst`, when the next turn of that transcript has `src` as its last. So a
  run counts only when its stretch's last row that derived a belief derives
  `dst`, and the next stretch with the same session and source label ends on
  a row that derived `src` (`_Turns.pinned`). A stretch with the record's
  key and rows is the record ingested again, and is taken with it. Rows keep
  no closer, so when runs in two records, by session and timestamp, fit one
  edge, or the record's stretches differ in label or rows, the anchor takes
  neither.
  The run's rows that fall in the part `cut_relay` takes are relay rows: its
  first row is a `notice` when it holds a host notice, and the rest are
  `block`. An anchor at the cap may end inside a sentence, so its last
  sentence may go unmatched.

An anchor that `cut_relay` doesn't cut, or that fails the host check, vetoes
every row of each run it matches when the run's first row has the `opener` or
`notice` shape: `cut_relay` keeps that record whole, as when a line the
sentence splitter dropped comes first. No other anchor can take a vetoed row,
so any record whose rows match a vetoed run's rows loses them too.

A relay record whose first row shares its stretch with the row before it, as
in a bulk ingest that gives several records one timestamp, has no row that
starts its turn, so none of its rows is taken. No row is taken because its
stretch holds a relay row.

What a row can't show leaves gaps. A record that opens with a tag in the
host's form reads as relay text, even when a person typed or pasted it, and
an unclosed one takes every sentence the anchor shows. When no later turn in
the session produced a belief, no anchor shows a record, so a line the
sentence splitter dropped before the tag can't stop its `opener` row. A host
relay whose cut part holds any other relay tag the host wouldn't write, such
as `<agent-message/>` quoted in its body, is skipped whole, its opener row
included. An install from before #1649 that wrote rows after the cutoff has
those rows skipped, and a later install whose clock ran behind the cutoff has
its rows read as raw records. When the turn that wrote an edge has no rows
from before the cutoff, another turn with the same rows that fits the edge is
read through that edge's anchor. The edge fit misses relay rows in a turn
followed by one that produced no belief, which ingest chains past, in a turn
from before #1364, which chained on the last belief a turn inserted, in a
turn another writer's row split, in a bulk ingest's shared timestamp, and in
a record ingested under a second label. Two files of one session ingested
under one label read as one transcript.

A belief is retired only when every `ingest_log` row that derived or
corroborated it is a relay row, and `scheduled_prompts.split_by_evidence`
keeps it on any other evidence: a lock, a promotion, a confirm, a restore,
a `keep` in `aelf review`, or another corroboration source. A belief whose
relay rows already gave it `agent_inferred` is kept as it is.

A belief a federation peer owns is never retired or restored here, as
`aelf retire` and `aelf restore` refuse one.

`aelf doctor --retire-relay-rows` runs the pass, and
`aelf doctor --restore-relay-rows` undoes it for each belief whose retire
time is still the pass's. Each is a dry run unless `--apply` is given.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from aelfrice.models import ANCHOR_TEXT_MAX_LEN, ORIGIN_AGENT_INFERRED
from aelfrice.noise_filter import (
    RELAY_ROW_NOTICE,
    RELAY_ROW_OPENER,
    cut_relay,
    relay_cut_is_host_shaped,
    relay_row_shape,
)
from aelfrice.scheduled_prompts import split_by_evidence
from aelfrice.sentence_partition import partition_sentences
from aelfrice.ulid import ulid_ms

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

SHAPE_BLOCK: Final[str] = "block"
# 2026-09-30T21:52:33Z, the author time of #1649's first commit (41f5d85df).
# Rows written before it come from ingest that split and anchored the raw record.
RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000
# The audit row each retired belief gets. Valence 0 moves no posterior.
SOURCE_RELAY_RETIRE: Final[str] = "relay_rows:retire"
# What `aelf restore` writes, so a restored belief stays on a later run.
SOURCE_USER_RESTORED: Final[str] = "user_restored"

_Key = tuple[str | None, str, str | None]


@dataclass
class RelayRowsReport:
    """Result of `retire_relay_rows` or `restore_relay_rows`."""

    dry_run: bool
    rows_read: int = 0
    rows_by_shape: Counter[str] = field(default_factory=Counter[str])
    anchors_read: int = 0
    anchors_matched: int = 0
    anchors_ambiguous: int = 0
    candidates: list[str] = field(default_factory=list[str])
    candidates_by_shape: Counter[str] = field(default_factory=Counter[str])
    locked: list[str] = field(default_factory=list[str])
    other_evidence: list[str] = field(default_factory=list[str])
    agent_inferred: list[str] = field(default_factory=list[str])
    foreign: list[str] = field(default_factory=list[str])
    changed: int = 0


@dataclass
class _Rows:
    """The transcript rows written before the cutoff, in id order."""

    ids: list[str]
    keys: list[_Key]
    texts: list[str]
    # Where a stretch begins: the row before has another key.
    starts: list[bool]
    # Where a turn begins: a stretch whose key no earlier stretch had.
    opens: list[bool]
    # `(session_id, sentence)` of each row written at or after the cutoff, or
    # with an id that carries no time.
    late: set[tuple[str | None, str]]


def _read_rows(store: "MemoryStore") -> _Rows:
    late: set[tuple[str | None, str]] = set()
    # Each stretch of the listing, with its rows and whether all were raw.
    stretches: list[tuple[_Key, list[tuple[str, str]], bool]] = []
    for log_id, sid, ts, path, text in store.list_transcript_sentences():
        ms = ulid_ms(log_id)
        raw = ms is not None and ms < RAW_RECORD_CUTOFF_MS
        if not raw:
            late.add((sid, text))
        key = (sid, ts, path)
        if not stretches or stretches[-1][0] != key:
            stretches.append((key, [], True))
        k, held, all_raw = stretches[-1]
        held.append((log_id, text))
        stretches[-1] = (k, held, all_raw and raw)
    ids: list[str] = []
    keys: list[_Key] = []
    texts: list[str] = []
    starts: list[bool] = []
    opens: list[bool] = []
    first_text: dict[_Key, str] = {}
    for key, held, all_raw in stretches:
        # A stretch holding a row from after the cutoff, or one with no time,
        # isn't read at all.
        if not all_raw:
            continue
        # Ingesting a transcript again repeats a turn's key from its first
        # row. A key that comes back at another row is a turn another
        # writer's row split, and that part opens nothing.
        opens_turn = first_text.setdefault(key, held[0][1]) == held[0][1]
        for j, (log_id, text) in enumerate(held):
            starts.append(j == 0)
            opens.append(opens_turn and j == 0)
            ids.append(log_id)
            keys.append(key)
            texts.append(text)
    return _Rows(ids=ids, keys=keys, texts=texts, starts=starts, opens=opens, late=late)


class _Turns:
    """Each stretch's last derived beliefs, and the stretch that follows it.

    A stretch is a run of rows with one `(session_id, ts, source_path)`. It
    holds one turn, or several when a bulk ingest gave them one timestamp.
    """

    def __init__(self, rows: _Rows, derives: list[set[str]]) -> None:
        self._of: list[int] = []
        self._first: list[int] = []
        self._end: list[int] = []
        self._head: list[set[str]] = []
        self._keys: list[_Key] = []
        for i, key in enumerate(rows.keys):
            if rows.starts[i]:
                self._keys.append(key)
                self._first.append(i)
                self._end.append(i)
                self._head.append(set())
            self._of.append(len(self._keys) - 1)
            self._end[-1] = i + 1
            if derives[i]:
                self._head[-1] = derives[i]
        # Ingest chains turns within one transcript file, so the next stretch
        # is the next one with the same session and source label.
        self._next: list[int | None] = [None] * len(self._keys)
        last: dict[tuple[str | None, str | None], int] = {}
        for t in range(len(self._keys) - 1, -1, -1):
            sid, _, path = self._keys[t]
            self._next[t] = last.get((sid, path))
            last[(sid, path)] = t

    def rows_of(self, i: int) -> tuple[int, int]:
        """`(first, end)` of the stretch row `i` is in."""
        t = self._of[i]
        return self._first[t], self._end[t]

    def pinned(self, i: int, n: int, capped: bool, src: str, dst: str) -> bool:
        """True when the run `[i, i + n)` could be the turn that wrote `src -> dst`.

        Ingest anchors `src -> dst` on a turn whose last belief is `dst`, when
        the next turn in its transcript has `src` as its last. The run opens
        its stretch, and has to fill it unless the anchor is capped. The
        stretch's last row that derived a belief has to derive `dst`, and the
        next stretch with the same session and source label has to end on a
        row that derived `src`.
        """
        t = self._of[i]
        if self._keys[t][0] is None or (not capped and i + n != self._end[t]):
            return False
        nxt = self._next[t]
        return dst in self._head[t] and nxt is not None and src in self._head[nxt]


def _runs(
    rows: _Rows, starts_by_text: dict[str, list[int]], whole: list[str], capped: bool,
) -> list[tuple[int, int]]:
    """`(start, length)` of each run of rows that `whole` matches in order.

    A run opens at a row that starts its turn, stays in that row's group, and
    must match every sentence, or all but the last when the anchor is capped.
    """
    need = len(whole) - 1 if capped else len(whole)
    out: list[tuple[int, int]] = []
    for i in starts_by_text.get(whole[0], []):
        n = 0
        while (
            n < len(whole) and i + n < len(rows.texts)
            and rows.keys[i + n] == rows.keys[i]
            and rows.texts[i + n] == whole[n]
        ):
            n += 1
        if n >= need:
            out.append((i, n))
    return out


def _block_rows(
    rows: _Rows,
    anchors: list[tuple[str, str, str]],
    turns: _Turns,
    report: RelayRowsReport,
) -> tuple[dict[int, str], set[int]]:
    """Relay rows that an anchor shows, by position, and the vetoed positions.

    A veto covers every row of a run whose first row has a relay shape and
    whose anchor `cut_relay` doesn't cut, or cuts with a tag or notice line
    that isn't the host's: that record doesn't open with relay text, even when
    its first stored row looks like it does. A vetoed row is never returned.

    An anchor's rows are taken only from a run in a turn that could have
    written its edge (`_Turns.pinned`), and only when every such run is in one
    record, by session and timestamp, under one label and with one set of
    rows: rows keep no closer, so a second record with the same rows may have
    closed its relay sooner. Copies of the record ingested again are taken.
    """
    starts_by_text: dict[str, list[int]] = {}
    for i, text in enumerate(rows.texts):
        if rows.opens[i]:
            starts_by_text.setdefault(text, []).append(i)
    found: dict[int, str] = {}
    vetoed: set[int] = set()
    for src, dst, anchor in anchors:
        rest = cut_relay(anchor)
        cut = anchor[: len(anchor) - len(rest)]
        relay = rest != anchor and relay_cut_is_host_shaped(cut)
        if rest != anchor:
            report.anchors_read += 1
        # At the cap the anchor may end inside a sentence.
        capped = len(anchor) >= ANCHOR_TEXT_MAX_LEN
        runs: list[tuple[int, int, bool, bool]] = []
        seen: list[list[str]] = []
        # Rows stored before #1874 dropped inline code and URLs.
        for keep_spans in (True, False):
            whole = partition_sentences(anchor, keep_spans=keep_spans)[0]
            if not whole or whole in seen:
                continue
            seen.append(whole)
            if not relay:
                if relay_row_shape(whole[0]) is not None:
                    for i, n in _runs(rows, starts_by_text, whole, capped):
                        vetoed.update(range(i, i + n))
                continue
            head = partition_sentences(cut, keep_spans=keep_spans)[0]
            if not head or whole[: len(head)] != head:
                continue
            notice = relay_row_shape(whole[0]) == RELAY_ROW_NOTICE
            for i, n in _runs(rows, starts_by_text, whole, capped):
                pinned = turns.pinned(i, n, capped, src, dst)
                runs.append((i, min(n, len(head)), notice, pinned))
        takes = [r for r in runs if r[3]]
        if not takes:
            continue
        # A record is its session and timestamp. A stretch with the same key
        # and rows is the record ingested again, and is taken with it. Rows
        # keep no closer, so when runs in two records fit the edge, or the
        # record's stretches differ in label or rows, the anchor takes neither.
        records = {rows.keys[i][:2] for i, _, _, _ in takes}
        takes = [r for r in runs if rows.keys[r[0]][:2] in records]
        labels = {rows.keys[i] for i, _, _, _ in takes}
        bodies = {tuple(rows.texts[slice(*turns.rows_of(i))]) for i, _, _, _ in takes}
        if len(labels) > 1 or len(bodies) > 1:
            report.anchors_ambiguous += 1
            continue
        # A later ingest that saw this record stored its speech, and may have
        # written the anchor from it.
        if any((rows.keys[i][0], rows.texts[i]) in rows.late for i, _, _, _ in takes):
            continue
        report.anchors_matched += 1
        for i, n, notice, _ in takes:
            for j in range(n):
                first = j == 0 and notice
                found.setdefault(i + j, RELAY_ROW_NOTICE if first else SHAPE_BLOCK)
    return {i: shape for i, shape in found.items() if i not in vetoed}, vetoed


def _select(store: "MemoryStore", report: RelayRowsReport) -> None:
    rows = _read_rows(store)
    report.rows_read = len(rows.ids)
    position = {log_id: i for i, log_id in enumerate(rows.ids)}
    derivations = store.list_ingest_derivation_rows()
    derives: list[set[str]] = [set() for _ in rows.ids]
    for bid, log_id in derivations:
        i = position.get(log_id)
        if i is not None:
            derives[i].add(bid)
    found, vetoed = _block_rows(
        rows, store.list_derived_from_anchors(), _Turns(rows, derives), report,
    )
    shape_at: dict[int, str] = {}
    # A notice row is taken only through an anchor: a sentence after the notice
    # on its line is a row of its own, and only the record shows the line.
    for i, text in enumerate(rows.texts):
        if rows.opens[i] and i not in vetoed:
            if relay_row_shape(text) == RELAY_ROW_OPENER:
                shape_at[i] = RELAY_ROW_OPENER
    for i, shape in found.items():
        shape_at.setdefault(i, shape)
    report.rows_by_shape = Counter(shape_at.values())
    every_row_relay: dict[str, bool] = {}
    first_shape: dict[str, str] = {}
    for bid, log_id in derivations:
        i = position.get(log_id)
        shape = shape_at.get(i) if i is not None else None
        every_row_relay[bid] = every_row_relay.get(bid, True) and shape is not None
        if shape is not None:
            first_shape.setdefault(bid, shape)
    relay_only = [b for b, ok in every_row_relay.items() if ok]
    report.foreign = sorted(b for b in relay_only if store.find_foreign_owner(b))
    foreign = set(report.foreign)
    candidates, report.locked, other = split_by_evidence(
        store, (b for b in relay_only if b not in foreign),
    )
    for bid in other:
        belief = store.get_belief(bid)
        if belief is not None and belief.origin == ORIGIN_AGENT_INFERRED:
            report.agent_inferred.append(bid)
        else:
            report.other_evidence.append(bid)
    report.candidates = [bid for bid, _ in candidates]
    report.candidates_by_shape = Counter(first_shape[b] for b in report.candidates)


def retire_relay_rows(
    store: "MemoryStore", *, dry_run: bool = True,
) -> RelayRowsReport:
    """List, and with `dry_run=False` retire, beliefs only relay rows made.

    Retiring sets `valid_to` (`MemoryStore.soft_delete_belief`) and writes a
    `relay_rows:retire` audit row of valence 0, so `restore_relay_rows` or
    `aelf restore <id>` undoes it. The dry run writes nothing.
    """
    report = RelayRowsReport(dry_run=dry_run)
    if dry_run:
        _select(store, report)
        return report
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with store.transaction(immediate=True):
        _select(store, report)
        for bid in report.candidates:
            store.insert_feedback_event(bid, 0.0, SOURCE_RELAY_RETIRE, ts)
            store.soft_delete_belief(bid, ts)
    report.changed = len(report.candidates)
    return report


def _retired_by_pass(store: "MemoryStore") -> list[str]:
    """Beliefs this pass retired that nothing has retired again since.

    The pass stamps `valid_to` with its audit row's time, so a belief counts
    when its `valid_to` equals its latest `relay_rows:retire` row. One that was
    restored and then retired another way, such as with `aelf retire`, holds a
    later `valid_to` and is left alone.
    """
    out: list[str] = []
    for bid, ts in store.latest_feedback_times(SOURCE_RELAY_RETIRE).items():
        belief = store.get_belief(bid, include_retired=True)
        if (
            belief is not None and belief.valid_to == ts
            and not store.find_foreign_owner(bid)
        ):
            out.append(bid)
    return sorted(out)


def restore_relay_rows(
    store: "MemoryStore", *, dry_run: bool = True,
) -> RelayRowsReport:
    """List, and with `dry_run=False` restore, the beliefs this pass retired.

    Each one `_retired_by_pass` finds goes back to active and gets a
    `user_restored` feedback row of valence 1, the row `aelf restore` writes,
    so a later `retire_relay_rows` keeps it. A belief a federation peer owns is
    skipped. The dry run writes nothing.
    """
    report = RelayRowsReport(dry_run=dry_run)
    if dry_run:
        report.candidates = _retired_by_pass(store)
        return report
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with store.transaction(immediate=True):
        report.candidates = _retired_by_pass(store)
        for bid in report.candidates:
            if store.restore_belief(bid):
                store.insert_feedback_event(bid, 1.0, SOURCE_USER_RESTORED, ts)
                report.changed += 1
    return report


def _ids(lines: list[str], ids: Iterable[str]) -> None:
    lines.extend(f"  {bid}" for bid in ids)


def format_relay_rows_report(report: RelayRowsReport, store_path: str) -> str:
    """Human-readable rendering of `retire_relay_rows`. Prints no belief text."""
    shapes = report.rows_by_shape
    lines = [
        f"store: {store_path}",
        f"transcript log rows written before the cutoff: {report.rows_read}",
        f"relay rows: {sum(shapes.values())} ({shapes['opener']} opener, "
        f"{shapes['notice']} notice, {shapes[SHAPE_BLOCK]} block)",
        f"anchors that open with relay text: {report.anchors_read} "
        f"({report.anchors_matched} matched to rows, "
        f"{report.anchors_ambiguous} matched to more than one turn and skipped)",
    ]
    by = report.candidates_by_shape
    verb = "would retire" if report.dry_run else "retired"
    lines.append(
        f"beliefs derived only from relay rows, {verb}: {len(report.candidates)} "
        f"({by['opener']} opener, {by['notice']} notice, {by[SHAPE_BLOCK]} block)"
    )
    _ids(lines, report.candidates)
    if report.locked:
        lines.append(f"locked, kept: {len(report.locked)}")
        _ids(lines, report.locked)
    if report.other_evidence:
        lines.append(
            "promoted, confirmed, restored, kept in review or corroborated "
            f"by another source, kept: {len(report.other_evidence)}"
        )
        _ids(lines, report.other_evidence)
    if report.agent_inferred:
        lines.append(f"already agent_inferred, kept: {len(report.agent_inferred)}")
    if report.foreign:
        lines.append(f"owned by a federation peer, kept: {len(report.foreign)}")
    if not report.dry_run:
        lines.append(
            f"retired {report.changed} belief(s). "
            "`aelf doctor --restore-relay-rows --apply` undoes this pass."
        )
    elif report.candidates:
        lines.append(
            "dry run: re-run with --apply to retire them "
            "(`aelf doctor --restore-relay-rows --apply` undoes the pass, "
            "`aelf restore <id>` one belief)."
        )
    else:
        lines.append("nothing to do.")
    return "\n".join(lines)


def format_relay_restore_report(report: RelayRowsReport, store_path: str) -> str:
    """Human-readable rendering of `restore_relay_rows`."""
    verb = "would restore" if report.dry_run else "restored"
    lines = [
        f"store: {store_path}",
        f"beliefs --retire-relay-rows retired that nothing has retired again, {verb}: "
        f"{len(report.candidates)}",
    ]
    _ids(lines, report.candidates)
    if report.dry_run:
        lines.append(
            "dry run: re-run with --apply to restore them."
            if report.candidates else "nothing to do."
        )
    else:
        lines.append(f"restored {report.changed} belief(s).")
    return "\n".join(lines)
