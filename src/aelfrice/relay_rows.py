"""Retire beliefs stored from relay text as the user's typed words (#1877).

Before #1875 ingest stored relay text that opened a user record, such as
another session's message, as the user's own words, with
`origin=user_transcript`. #1875 stops new rows. This pass finds the stored
ones with the rule `noise_filter.cut_relay` applies to a record, read
through what the store kept.

An `ingest_log` row holds one sentence, so it can't show line breaks, a
closer, or where its record began. A row starts its turn when the row before
it, in id order, has another `(session_id, ts, source_path)`: ingest writes a
turn's rows together, and they share all three. Three shapes are read:

- `opener`: a row that starts its turn and holds a relay opening tag and
  nothing else, with the attributes the host writes
  (`noise_filter.relay_row_shape`).
- `notice` and `block`: the rows of a record that a later turn's
  DERIVED_FROM anchor shows opening with relay text. Between two user turns,
  `ingest_jsonl` anchors the edge on the earlier turn's text, line breaks kept,
  up to 1,000 characters. `cut_relay` reads that text as it reads a record,
  and every relay tag and notice line in the part it takes must be the host's
  (`noise_filter.relay_cut_is_host_shaped`). The anchor's sentences are
  matched in order to a run of rows that opens at a row that starts its turn
  and stays in that row's group, in a session that derived both ends of the
  edge. The run's rows that fall in the part `cut_relay` takes are relay
  rows: its first row is a `notice` when it holds a host notice, and the rest
  are `block`. An anchor at the cap may end inside a sentence, so its last
  sentence may go unmatched.

An anchor that `cut_relay` doesn't cut, or that fails the host check, vetoes
every row of each run it matches when the run's first row has the `opener` or
`notice` shape: `cut_relay` keeps that record whole, as when a line the
sentence splitter dropped comes first. No other anchor can take a vetoed row,
so a record whose rows match a vetoed run's rows loses them too.

A relay record whose first row shares its group with the row before it, as in
a bulk ingest that gives several records one timestamp, has no row that starts
its turn, so none of its rows is taken. No row is taken because its group
holds a relay row.

What a row can't show leaves gaps. A record that opens with a tag carrying the
host's attributes reads as relay text, even when a person typed or pasted it,
and an unclosed one takes every sentence the anchor shows. With no later turn,
no anchor shows a record, so a dropped first line can't stop its `opener` row.
A host relay whose cut part holds any other relay tag the host wouldn't write,
such as `<agent-message/>` quoted in its body, is skipped whole, its opener
row included. Rows are
matched by their sentences, so a record whose rows repeat a relay record's
rows word for word, opener included, in a session that derived both ends of
the edge, reads as that relay record.

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

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

SHAPE_BLOCK: Final[str] = "block"
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
    candidates: list[str] = field(default_factory=list[str])
    candidates_by_shape: Counter[str] = field(default_factory=Counter[str])
    locked: list[str] = field(default_factory=list[str])
    other_evidence: list[str] = field(default_factory=list[str])
    agent_inferred: list[str] = field(default_factory=list[str])
    foreign: list[str] = field(default_factory=list[str])
    changed: int = 0


@dataclass
class _Rows:
    ids: list[str]
    keys: list[_Key]
    texts: list[str]
    starts: list[bool]


def _read_rows(store: "MemoryStore") -> _Rows:
    ids: list[str] = []
    keys: list[_Key] = []
    texts: list[str] = []
    starts: list[bool] = []
    for log_id, sid, ts, path, text in store.list_transcript_sentences():
        key = (sid, ts, path)
        starts.append(not keys or keys[-1] != key)
        ids.append(log_id)
        keys.append(key)
        texts.append(text)
    return _Rows(ids=ids, keys=keys, texts=texts, starts=starts)


def _runs(
    rows: _Rows, starts_by_text: dict[str, list[int]], sessions: set[str],
    whole: list[str], capped: bool,
) -> list[tuple[int, int]]:
    """`(start, length)` of each run of rows that `whole` matches in order.

    A run opens at a row that starts its turn, stays in that row's group, and
    must match every sentence, or all but the last when the anchor is capped.
    """
    need = len(whole) - 1 if capped else len(whole)
    out: list[tuple[int, int]] = []
    for i in starts_by_text.get(whole[0], []):
        sid = rows.keys[i][0]
        if sid is None or sid not in sessions:
            continue
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
    sessions_of: dict[str, set[str]],
    report: RelayRowsReport,
) -> tuple[dict[int, str], set[int]]:
    """Relay rows that an anchor shows, by position, and the vetoed positions.

    A veto covers every row of a run whose first row has a relay shape and
    whose anchor `cut_relay` doesn't cut, or cuts with a tag or notice line
    that isn't the host's: that record doesn't open with relay text, even when
    its first stored row looks like it does. A vetoed row is never returned.
    """
    starts_by_text: dict[str, list[int]] = {}
    for i, text in enumerate(rows.texts):
        if rows.starts[i]:
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
        # A session holds the turn only if it derived both ends of the edge.
        sessions = sessions_of.get(dst, set()) & sessions_of.get(src, set())
        matched = False
        seen: list[list[str]] = []
        # Rows stored before #1874 dropped inline code and URLs.
        for keep_spans in (True, False):
            whole = partition_sentences(anchor, keep_spans=keep_spans)[0]
            if not whole or whole in seen:
                continue
            seen.append(whole)
            if not relay:
                if relay_row_shape(whole[0]) is not None:
                    for i, n in _runs(rows, starts_by_text, sessions, whole, capped):
                        vetoed.update(range(i, i + n))
                continue
            head = partition_sentences(cut, keep_spans=keep_spans)[0]
            if not head or whole[: len(head)] != head:
                continue
            for i, n in _runs(rows, starts_by_text, sessions, whole, capped):
                matched = True
                for j in range(min(n, len(head))):
                    first = j == 0 and relay_row_shape(whole[0]) == RELAY_ROW_NOTICE
                    found.setdefault(i + j, RELAY_ROW_NOTICE if first else SHAPE_BLOCK)
        if matched:
            report.anchors_matched += 1
    return {i: shape for i, shape in found.items() if i not in vetoed}, vetoed


def _select(store: "MemoryStore", report: RelayRowsReport) -> None:
    rows = _read_rows(store)
    report.rows_read = len(rows.ids)
    position = {log_id: i for i, log_id in enumerate(rows.ids)}
    derivations = store.list_ingest_derivation_rows()
    sessions_of: dict[str, set[str]] = {}
    for bid, log_id in derivations:
        i = position.get(log_id)
        sid = rows.keys[i][0] if i is not None else None
        if sid is not None:
            sessions_of.setdefault(bid, set()).add(sid)
    found, vetoed = _block_rows(
        rows, store.list_derived_from_anchors(), sessions_of, report,
    )
    shape_at: dict[int, str] = {}
    # A notice row is taken only through an anchor: a sentence after the notice
    # on its line is a row of its own, and only the record shows the line.
    for i, text in enumerate(rows.texts):
        if rows.starts[i] and i not in vetoed:
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
        f"transcript log rows read: {report.rows_read}",
        f"relay rows: {sum(shapes.values())} ({shapes['opener']} opener, "
        f"{shapes['notice']} notice, {shapes[SHAPE_BLOCK]} block)",
        f"anchors that open with relay text: {report.anchors_read} "
        f"({report.anchors_matched} matched to rows)",
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
