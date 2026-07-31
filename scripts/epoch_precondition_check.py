"""Epoch precondition check for the session injection ledger (#1252 / #1177).

Proposal 11 in #1177 (session injection ledger + turn-differential lock
rendering) rests on one assumption: that the harness reliably fires
`SessionStart` with `source == "compact"` after a context reset. The
ledger's epoch increments on that event and on nothing else, so if the
event is unreliable the ledger renders `full` once and `manifest`
forever after — the always-injected guarantee degrades to
never-reinjected, silently.

#1177 specified the check as "count SessionStart rows with
`source=='compact'` in `hook_audit.jsonl` against observed context
resets". #1252 blocked that spec because "observed context resets" has
no witness. This script is the re-spec: it takes both sides from the
**host transcript**, which records the reset and the hook firing as two
independent record types, and neither of which any aelfrice code emits.

## Why the transcript and not `hook_audit.jsonl`

`hook_audit.jsonl` cannot answer this, for three independent reasons —
each verified against `src/aelfrice/hook.py` rather than assumed:

1. **`source` is never recorded.** `_write_hook_audit_record` has no
   `source` parameter, and the `session_start` call site does not pass
   one. Its `session_start` rows are indistinguishable across
   `startup` / `resume` / `clear` / `compact`. The specified numerator
   is not computable.
2. **The row is conditional on a non-empty baseline block.** The audit
   write sits inside `if body:`. A `SessionStart` that fires against an
   empty locked set writes no row at all, so the row count is not a
   count of firings even before the `source` problem.
3. **The rebuild block is excluded from the row.** `rendered_block` is
   the baseline `body`, computed and written *before* the compact-only
   rebuild block is appended to stdout. So compact-ness cannot be
   recovered from the stored block either.

The host transcript has neither problem. It is the same artifact the
hook payload already names (`transcript_path`), so this is not a
bespoke instrumentation channel.

## The two sides

- **Denominator — a context reset happened.** A transcript record with
  `subtype == "compact_boundary"`, carrying `compactMetadata.trigger`
  (`manual` or `auto`) plus pre/post token counts. Written by the host
  when it compacts. This is the independent witness #1252 says does not
  exist; it exists, it is just not in the audit log.
- **Numerator — the hook fired for that reset.** A transcript record
  whose text carries the `SessionStart:compact` hook-result marker.
  Written by the host when it runs the hook, not by the hook.

Both are host-emitted, so neither can be self-confirming: aelfrice
cannot cause a `compact_boundary` to appear, and cannot suppress one.

## Pairing rule

Aggregate counts are not enough — 17 == 17 can be two unrelated
populations. Each `compact_boundary` is paired with the **first**
`SessionStart:compact` marker that follows it in the same session file,
in file order, and each marker may be consumed by at most one boundary.
A boundary with no successor marker is **unfired**.

The final boundary in a session is reported separately as
`trailing_unfired` and excluded from the rate. The session may have
ended between the reset and the next hook fire, which is truncation,
not unreliability. Counting it as a failure would bias the rate down
by roughly one per session.

## Decision rule — FIXED BEFORE THE RUN

Let `fire_rate = paired / (paired + unfired)`, excluding trailing
boundaries, over a population of at least 20 scoreable boundaries.

- `fire_rate >= 0.98` -> **CLEARS.** The epoch event is reliable and
  proposal 11's precondition holds.
- `fire_rate < 0.90` -> **KILLS.** The design has no dependable epoch.
  Proposal 11 closes; the ruling funded the measurement, not the
  feature.
- `0.90 <= fire_rate < 0.98` -> **GREY.** Does not clear on its own.
  The grey band resolves one way only: proposal 11 may proceed **iff**
  it carries a second, independent epoch increment (the turn-count TTL
  backstop #1177 already gestures at), specified and measured before
  any ledger code. Absent that, grey is a kill.
- Fewer than 20 scoreable boundaries -> **NO VERDICT.** Report the
  count and re-run on a larger window. An underpowered pass is not a
  pass.

`manual` and `auto` triggers are reported separately as well as
pooled. If they disagree by more than 10 percentage points the result
is `NO VERDICT` — a design that holds for an explicit compaction but
not for an automatic one has not cleared, since auto is the case the
user cannot see coming.

This guard is a rung inside `verdict()`, ahead of the `CLEARS` rung,
rather than an advisory line printed beneath the headline. A headline
reading `CLEARS` under a footnote saying the pooled rate is not the
verdict would be re-runnable, pasteable and greppable evidence for the
opposite of the rule.

## The never-compacted session

This check deliberately says nothing about sessions that never compact,
because there is no boundary in them to score. That is a real cost of
the design and it is **not** in the fire rate: a session that never
compacts never increments the epoch, so a ledger keyed on it renders
`full` once and `manifest` for the rest of the session no matter how
reliable the event is. The count of such sessions is reported as
`sessions_without_boundary` so the size of that population is on the
record next to the rate, but it is a design question for #1177, not a
precondition failure.

## Population

Every `*.jsonl` under the transcript root, filtered to files that
contain at least one `compact_boundary`. Files without one contribute
only to `sessions_without_boundary`. No sampling, no truncation — if a
window is used, it is `--since`, and the applied window is printed.

Output is counts and rates only. It never prints transcript text.

Usage:

    uv run python scripts/epoch_precondition_check.py --transcript-root PATH
    uv run python scripts/epoch_precondition_check.py --transcript-root PATH --since 2026-07-01

This is a contributor diagnostic, not a CI gate. It reads local host
data and runs locally only.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"
SESSION_START_COMPACT_MARKER = "SessionStart:compact"

CLEARS_AT = 0.98
KILLS_BELOW = 0.90
MIN_SCOREABLE = 20
TRIGGER_DIVERGENCE_LIMIT = 0.10


@dataclass
class Boundary:
    """One host-recorded context reset."""

    trigger: str
    timestamp: str
    fired: bool = False
    trailing: bool = False


@dataclass
class Tally:
    boundaries: list[Boundary] = field(default_factory=list)
    sessions_with_boundary: int = 0
    sessions_without_boundary: int = 0
    markers_seen: int = 0
    markers_unpaired: int = 0


def _record_timestamp(record: dict[str, object]) -> str:
    value = record.get("timestamp")
    return value if isinstance(value, str) else ""


def _is_boundary(record: dict[str, object]) -> bool:
    return record.get("subtype") == COMPACT_BOUNDARY_SUBTYPE


def _boundary_trigger(record: dict[str, object]) -> str:
    meta = record.get("compactMetadata")
    if isinstance(meta, dict):
        trigger = meta.get("trigger")
        if isinstance(trigger, str) and trigger:
            return trigger
    return "unknown"


def _has_session_start_compact(raw_line: str) -> bool:
    """True when the raw record carries the hook-result marker.

    Matched on the raw line rather than a parsed field: the host nests
    hook output at different depths across record types, and the marker
    itself is unambiguous — no other event renders that literal.
    """
    return SESSION_START_COMPACT_MARKER in raw_line


def scan_file(path: Path, since: str | None, tally: Tally) -> None:
    """Fold one transcript file into `tally`.

    Single forward pass. A boundary is left `fired=False` until a
    marker appears after it; each marker satisfies at most one
    outstanding boundary, oldest first, so a burst of markers cannot
    paper over a run of unfired boundaries.
    """
    pending: list[Boundary] = []
    found_any = False
    try:
        handle = path.open(errors="replace")
    except OSError:
        return
    with handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            marker = _has_session_start_compact(raw_line)
            if not marker and COMPACT_BOUNDARY_SUBTYPE not in raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except (ValueError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            timestamp = _record_timestamp(record)
            if since and timestamp and timestamp < since:
                continue
            if _is_boundary(record):
                found_any = True
                boundary = Boundary(
                    trigger=_boundary_trigger(record), timestamp=timestamp
                )
                pending.append(boundary)
                tally.boundaries.append(boundary)
            elif marker:
                tally.markers_seen += 1
                if pending:
                    pending.pop(0).fired = True
                else:
                    tally.markers_unpaired += 1
    # Whatever is still pending never saw a marker. Only the last one is
    # attributable to the session ending; an earlier unfired boundary had
    # a whole subsequent session in which to fire and did not.
    if pending:
        pending[-1].trailing = True
    if found_any:
        tally.sessions_with_boundary += 1
    else:
        tally.sessions_without_boundary += 1


def _rate(fired: int, unfired: int) -> float | None:
    total = fired + unfired
    return (fired / total) if total else None


def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def trigger_divergence(trigger_rates: dict[str, float] | None) -> float | None:
    """Spread between the best- and worst-firing trigger, or None.

    None when fewer than two triggers were observed — one trigger
    cannot diverge from anything, and treating that as 0.0 would let a
    corpus with a single trigger clear on a rule that never ran.
    """
    if not trigger_rates or len(trigger_rates) < 2:
        return None
    return max(trigger_rates.values()) - min(trigger_rates.values())


def verdict(
    rate: float | None,
    scoreable: int,
    trigger_rates: dict[str, float] | None = None,
) -> str:
    """Apply the pre-registered rule and return the whole verdict.

    The divergence guard is a rung *inside* this function, not an
    advisory line printed under it. A headline that says CLEARS with a
    footnote saying the pooled rate is not the verdict is the one thing
    a pre-registered rule cannot afford: the point of fixing the rule
    before the run is that the outcome cannot be re-scored afterwards,
    and the first line is what gets re-run, pasted and grepped.
    """
    if scoreable < MIN_SCOREABLE or rate is None:
        return "NO VERDICT (underpowered)"
    spread = trigger_divergence(trigger_rates)
    if spread is not None and spread > TRIGGER_DIVERGENCE_LIMIT:
        return (
            f"NO VERDICT (triggers diverge by {spread * 100:.1f}pp > "
            f"{TRIGGER_DIVERGENCE_LIMIT * 100:.0f}pp; the pooled rate is "
            "not the verdict)"
        )
    if rate >= CLEARS_AT:
        return "CLEARS"
    if rate < KILLS_BELOW:
        return "KILLS"
    return "GREY (kill unless a second epoch trigger is specified first)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--transcript-root",
        required=True,
        type=Path,
        help="Directory searched recursively for host transcript *.jsonl.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO-8601 lower bound on record timestamp. Applied window "
        "is echoed in the report.",
    )
    args = parser.parse_args(argv)

    root: Path = args.transcript_root
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    tally = Tally()
    files = sorted(root.rglob("*.jsonl"))
    for path in files:
        scan_file(path, args.since, tally)

    scoreable = [b for b in tally.boundaries if not b.trailing]
    fired = sum(1 for b in scoreable if b.fired)
    unfired = len(scoreable) - fired
    trailing = [b for b in tally.boundaries if b.trailing]
    rate = _rate(fired, unfired)

    by_trigger: dict[str, list[Boundary]] = {}
    for boundary in scoreable:
        by_trigger.setdefault(boundary.trigger, []).append(boundary)

    print("=== epoch precondition check (#1252, proposal 11 of #1177) ===")
    print(f"transcript root      : {root}")
    print(f"window (--since)     : {args.since or 'all'}")
    print(f"files scanned        : {len(files)}")
    print(f"sessions w/ boundary : {tally.sessions_with_boundary}")
    print(f"sessions w/o boundary: {tally.sessions_without_boundary}")
    print(f"boundaries total     : {len(tally.boundaries)}")
    print(f"  scoreable          : {len(scoreable)}")
    print(f"  trailing (excluded): {len(trailing)}")
    print(f"markers seen         : {tally.markers_seen}")
    print(f"markers unpaired     : {tally.markers_unpaired}")
    print("")
    print(f"fired                : {fired}")
    print(f"unfired              : {unfired}")
    print(f"fire_rate            : {_fmt_rate(rate)}")
    print("")
    print("--- by trigger ---")
    trigger_rates: dict[str, float] = {}
    for name in sorted(by_trigger):
        group = by_trigger[name]
        group_fired = sum(1 for b in group if b.fired)
        group_rate = _rate(group_fired, len(group) - group_fired)
        if group_rate is not None:
            trigger_rates[name] = group_rate
        print(
            f"{name:8s} n={len(group):4d}  fired={group_fired:4d}  "
            f"rate={_fmt_rate(group_rate)}"
        )

    spread = trigger_divergence(trigger_rates)
    print("")
    if spread is not None:
        print(f"trigger divergence   : {spread * 100:.1f}pp "
              f"(limit {TRIGGER_DIVERGENCE_LIMIT * 100:.0f}pp)")
    print(f"VERDICT: {verdict(rate, len(scoreable), trigger_rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
