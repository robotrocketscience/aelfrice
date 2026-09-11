"""#1407 — the BM25 sidecar rebuild rate, counted rather than inferred.

#1380's cost case is `cold_cost x cold_rate`. After the 2026-08-06
re-derivation `cold_cost` is known (2.89 s cold first-fire at 44,668 beliefs)
and `cold_rate` was not: the only estimate was a latency proxy (a fire is
"cold" if `latency_ms >= 1000`) that yielded 8.5% but cannot attribute any
individual slow fire to a rebuild rather than to SQLite lock contention, a cold
page cache, or an unrelated stall. The operator ruling of 2026-08-06 ~18:00Z is
that #1380 must not be decided on the proxy.

This reads the `sidecar_outcome` field that #1407 added and reports the direct
counts.

## Three states, not a boolean

`fresh` / `incremental` / `full_rebuild`. Collapsing the middle one is what made
#1199's 86.2% ("sidecar not fresh") and the 8.5% proxy ("fire was slow") look
contradictory when they measure different events: since #1199 shipped the
incremental path, a stale sidecar no longer implies a full rebuild.

**`full_rebuild` is the rate #1380 is priced on.** `incremental` is cheap.

## Rows with no outcome are three different populations, not one

A row with no `sidecar_outcome` key is **excluded and counted**, never treated
as `fresh` — folding a fire that did no index work into the denominator as a
cache hit would drive the measured rebuild rate toward zero. But "excluded" is
three distinct things and reporting them as one number names a cause that has
stopped existing:

- **gate-skipped** — `prompt_shape_gate_skip` is set *and* the row still has no
  outcome. The shape gate refused the prompt, so the main retrieval never ran.
  It does not follow that no index work happened: the cadence dispatch runs
  above the gate and reaches `BM25IndexCache.get()`, so a gate-skipped fire
  that paid a rebuild there carries the key and is scored like any other. What
  lands here is the fire that was refused *and* built nothing — a measured
  zero, not a gap. This is the largest bucket and it is permanent; a warning
  phrased as "wait for more data" will never clear against it.
- **pre-field** — logged before #1407 shipped. Genuinely "not enough data yet",
  and genuinely does shrink over time.
- **no index work** — retrieval ran but built no index (L1 lane off). Neither a
  measurement nor a wait; it is a real, ongoing population.

The pre-field boundary is derived from the data (the earliest `ts` carrying an
outcome), not hardcoded, so it stays correct if the field's ship date moves.

## Which denominator

`full_rebuild / scored` is **not** interchangeable with the 8.5% latency proxy
it replaces: that proxy was computed over *all* UPS fires, and `scored` can only
ever contain fires that did index work. Reading one against the other turns a
confirmation of the proxy into an apparent doubling, which is why all three
denominators are printed.

**Every one of them excludes the unmeasured rows** — pre-#1407 and, before any
keyed row exists, unclassified. Those rows cannot enter the numerator, so
leaving them under the line is arithmetically identical to scoring an
unmeasured fire as not-a-rebuild: the bias the bullets above forbid, applied
silently. Gate-skipped and no-index-work rows are *kept*: the fire happened and
built nothing, which is a measured zero, and a per-fire `cold_rate` needs it.

The 2.30x separation quoted for that pair (8.69% all-fires against 20.00%
retrieval-fires) is **a worked example on a constructed log, not a live
measurement, and cannot yet be one**: the field is written only by this
branch's code while installed hooks run the released package, so every
`user_prompt_submit` row on the real log predates it and the script correctly
reports NO MEASUREMENT YET before any rate is printed. There is no live ratio
to cite. CHANGELOG/v4.md carries the same relabel.

## Position within the session is a separate axis (#1513)

The pooled rate above is an average over a population that is not
homogeneous. Bucketed by whether a scored fire is the first one carrying its
`session_id`, the two halves do not resemble each other: the session-FIRST
bucket carries a materially higher `full_rebuild` rate than the LATER one.
The cost is a session-first tail, and a single pooled number hides it —
which is why the `BY POSITION WITHIN THE SESSION` section prints two rates
and never one. Read the LATER bucket alongside it: a "fix" that merely
defers the rebuild shows up as that number rising.

**This file publishes no magnitude for that split, deliberately.** Three
re-derivations over three weeks moved the session-FIRST rate by more than a
factor of three while the sign never flipped, because the population is a
live, growing, single-slot-rotating log set: the rows a run scores today are
not the rows it scored last month, and the ones rotation dropped are gone.
A frozen percentage here would be a number nobody can reproduce. Run the
script and read the two lines it prints.

Rows with no `session_id` have no position. They are reported as their own
count and kept out of BOTH buckets: sweeping them into LATER is the same
bias as scoring an unmeasured fire as not-a-rebuild, applied to this axis
instead of to the denominators above.

Usage:
    uv run python benchmarks/sidecar_rebuild_rate.py [--since TS] [--until TS] \
        [AUDIT_LOG ...]

With no arguments this globs `hook_audit.jsonl*`, which includes **rotated**
logs (`hook_audit.jsonl.1`, ...). The totals it prints are therefore across all
rotations, not the single live file — pass an explicit path for a single-file
count.

**The default population is one repository's log, and that is rarely the
population a claim is about.** `_default_logs()` globs the `aelfrice/`
directory under the common directory of the repository you run it from, so a
rate read with no arguments is a rate for that one repository. To read it
over every store on the machine, enumerate them and pass them in::

    find ~ -name 'hook_audit.jsonl*' -not -path '*/node_modules/*' \
        | sort > /tmp/logs.txt
    xargs uv run python benchmarks/sidecar_rebuild_rate.py < /tmp/logs.txt

A figure quoted from one of those two commands has to name which one,
because they do not share a denominator.

## Two windows out of one log set (#1513)

`--since TS` and `--until TS` filter `user_prompt_submit` rows by their `ts`,
half-open on the upper bound, so `--until T` and `--since T` partition an
input with no row in both and none lost. This exists so a before-and-after
comparison is about one population: taking a baseline in one month and a
post-change rate in another compares two log sets rather than two periods,
and their difference is not the change you shipped. Cut both windows from
the same files in one pair of invocations instead.

These flags select ROWS. The `window first row` and `WINDOW TRUNCATED` lines
are the #1528 report on the FILES and are unaffected by `--since` /
`--until`: they still describe what the input can cover, which is what tells
you whether the window you asked for was present in it at all.

## The window, printed alongside the rate (#1528)

Rotation is single-slot: the second rollover overwrites the first `.1` and
that history is gone. A rate over a truncated window is a rate over "whatever
the recent period looked like", and until #1528 nothing in the output said so.
`audit_window` reads the rotation markers and this script now prints the ts
range it actually covered, whether a `.1` was present, and a WINDOW TRUNCATED
line when generations have been discarded.

The verdict is three-valued, and the third value is the point. A log that has
never rotated is complete. A log whose markers account for every generation is
complete or truncated by an exact count. A log rotated before #1528 shipped —
a `.1` with no generation stamp — is neither: one rollover discards nothing, a
second discards the first archive, and the files cannot say which. That prints
WINDOW UNKNOWN. Once such a log rolls over again the marker records "at least
one discarded" rather than pretending to a count it cannot have.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from aelfrice.hook_audit import AUDIT_ROTATION_HOOK, audit_window

OUTCOMES = ("fresh", "incremental", "full_rebuild")


def _default_logs() -> list[Path]:
    from aelfrice.db_paths import _git_common_dir

    git_dir = _git_common_dir()
    if git_dir is None:
        raise SystemExit("not in a git work-tree; pass audit log paths")
    d = git_dir / "aelfrice"
    return sorted(d.glob("hook_audit.jsonl*"))


def in_window(
    ts: object, since: str | None, until: str | None
) -> bool:
    """True when `ts` falls in the half-open window `[since, until)`.

    `ts` is the ISO-8601 Z string the audit rows carry, which sorts
    lexicographically in time order, so the comparison is a string compare
    and needs no parsing.

    A row with no `ts` is **out** of any window that was asked for, and in
    when neither bound was given. Keeping an undated row inside a window
    would put a fire of unknown date into a rate the caller asked to be
    about a specific period — the same unearned attribution the module
    docstring refuses for unmeasured rows, applied to the time axis.
    """
    if since is None and until is None:
        return True
    if not isinstance(ts, str):
        return False
    if since is not None and ts < since:
        return False
    return not (until is not None and ts >= until)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    # #1513 round 2. Without these, a baseline and a post-fix rate can only
    # be taken as two runs weeks apart, over a log set that grew and rotated
    # in between — so the two numbers are not about the same population and
    # the difference between them is not the fix. With them, both windows
    # are cut from ONE log set in ONE pair of invocations, and only the
    # window differs.
    ap.add_argument(
        "--since",
        metavar="TS",
        help="keep rows whose ts is >= TS (ISO-8601, e.g. 2026-09-11T00:00:00Z)",
    )
    ap.add_argument(
        "--until",
        metavar="TS",
        help="keep rows whose ts is < TS. Half-open, so --until T and "
        "--since T partition one log set with no row in both and none lost.",
    )
    args = ap.parse_args()

    logs = args.logs or _default_logs()
    logs = [p for p in logs if p.is_file()]
    if not logs:
        print("no audit logs found", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    non_ups = 0
    # Rows with no outcome, split by why. `gate_skipped` was refused by the
    # shape gate and built nothing above it either; `no_index_work` ran
    # retrieval but built nothing; `pre_field` predates the field. The first
    # two are measured zeros; only `pre_field` shrinks over time.
    gate_skipped = 0
    # Fires dropped by --since/--until. Printed rather than silently
    # discarded: a window that ate most of the input is the difference
    # between a rate and a rate over three fires.
    out_of_window = 0
    unkeyed: list[str | None] = []
    scored_ts: list[str] = []
    # #1513: (ts, read-order, session_id, outcome) per scored fire, so the
    # session-first vs later split below can be recovered. Read order is
    # carried as the tie-break because `ts` is second-resolution and two
    # fires in one second are common.
    scored_rows: list[tuple[str | None, int, str | None, str]] = []
    seq = 0

    for path in logs:
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("hook") == AUDIT_ROTATION_HOOK:
                # #1528 bookkeeping about the FILE, not a fire in it.
                # Counting it under "non-UPS rows" would quietly inflate a
                # number this script prints as a row census.
                continue
            if rec.get("hook") != "user_prompt_submit":
                non_ups += 1
                continue
            if not in_window(rec.get("ts"), args.since, args.until):
                out_of_window += 1
                continue
            outcome = rec.get("sidecar_outcome")
            if outcome is None:
                if rec.get("prompt_shape_gate_skip"):
                    # Refused by the shape gate AND carrying no outcome: the
                    # main retrieval never ran and the cadence dispatch above
                    # the gate built nothing either. A measured zero, and a
                    # permanent population — not "not yet". (A gate-skipped
                    # fire that *did* pay a rebuild has the key and was
                    # scored above; it never reaches this branch.)
                    gate_skipped += 1
                else:
                    ts = rec.get("ts")
                    unkeyed.append(str(ts) if ts is not None else None)
                continue
            if outcome not in OUTCOMES:
                unknown[str(outcome)] += 1
                continue
            counts[str(outcome)] += 1
            ts = rec.get("ts")
            if ts is not None:
                scored_ts.append(str(ts))
            sid = rec.get("session_id")
            scored_rows.append(
                (
                    str(ts) if ts is not None else None,
                    seq,
                    str(sid) if sid else None,
                    str(outcome),
                )
            )
            seq += 1

    scored = sum(counts.values())
    # Derive the field's arrival from the data rather than hardcoding a date:
    # an unkeyed row older than the earliest keyed row predates the field.
    first_scored_ts = min(scored_ts) if scored_ts else None
    if first_scored_ts is None:
        # No keyed row exists, so there is no boundary to split on. Do not
        # guess: calling all of these "pre-field" would assert something the
        # data cannot support, in the one script whose whole job is to stop
        # exactly that kind of unearned attribution.
        # `no_index_work` is deliberately left unbound here: with no keyed
        # row there is no boundary to split on, and the only line that
        # prints it is in the branch this arm does not take. Binding it to
        # None would be a value nothing reads.
        pre_field = None
        unclassified = len(unkeyed)
    else:
        pre_field = sum(1 for t in unkeyed if t is not None and t < first_scored_ts)
        no_index_work = len(unkeyed) - pre_field
        unclassified = 0
    missing = gate_skipped + len(unkeyed)
    all_fires = scored + missing + sum(unknown.values())
    # A row that can never enter the numerator must not sit in a denominator.
    # Pre-#1407 and unclassified rows are *unmeasured*: keeping them below the
    # line is arithmetically identical to scoring an unmeasured fire as
    # not-a-rebuild, which is the exact bias this script exists to refuse.
    # (Executed before the fix: 50 pre-field rows against 60 scored ones with
    # 10 rebuilds printed 10/110 = 9.09% for a true measured 16.67%, and the
    # CAUTION below never fired because it keys on `pre_field > scored`.)
    #
    # Gate-skipped and no-index-work rows are NOT subtracted. Those are
    # measured zeros -- the fire happened and built nothing -- and an
    # all-fires cold_rate has to contain them or it stops being per-fire.
    unmeasured = (pre_field or 0) + unclassified
    all_fires_measured = all_fires - unmeasured
    retrieval_fires_measured = all_fires_measured - gate_skipped

    window = audit_window(logs)

    print("#1407 — BM25 sidecar outcome per user_prompt_submit fire")
    for p in logs:
        print(f"  log                            {p}")
    if args.since is not None or args.until is not None:
        print(
            f"  ts window (half-open)          "
            f"[{args.since or '-inf'}, {args.until or '+inf'})"
        )
        print(f"  UPS fires outside the window   {out_of_window}")
    # #1528: the window this rate is over, printed before the rate itself.
    # `first_ts`/`last_ts` are over every row in the files, not only the
    # UPS rows scored below, so this is the coverage of the INPUT.
    print(f"  window first row               {window.first_ts or '(none)'}")
    print(f"  window last row                {window.last_ts or '(none)'}")
    print(f"  rotated .1 present             {window.rotated_present}")
    generation_label = (
        f"{window.generation} (at least)"
        if window.discarded_unknown
        else str(window.generation)
    )
    print(f"  rotation generation            {generation_label}")
    # Three-valued by construction (`AuditWindow.truncated` / `.complete`).
    # Collapsing it to truncated-or-complete is how a false completeness
    # claim ships: the pre-#1528 population reads as neither.
    if window.truncated:
        atleast = "at least " if window.discarded_unknown else ""
        print(
            f"  WINDOW TRUNCATED               {atleast}"
            f"{window.discarded_generations} "
            "rotation generation(s) discarded"
        )
        print(
            "    Single-slot rotation overwrote older archives. Every rate "
            "below is"
        )
        print(
            "    over the surviving tail only, and is biased toward whatever "
            "the recent"
        )
        print("    period looked like. Do not read it as a long-horizon rate.")
        if window.discarded_unknown:
            print(
                "    The destroyed history carried no #1528 marker, so the "
                "count above"
            )
            print(
                "    is a floor. How much more was discarded is not "
                "recoverable."
            )
    elif not window.complete:
        # Unknown: a `.1` whose generation nothing in this input can
        # state. Either the live file predates the marker, or no live file
        # was passed at all and the archives cannot see the loss their own
        # rotation caused. Saying "complete" here would be the exact
        # unearned claim the marker exists to prevent.
        print(
            "  WINDOW UNKNOWN                 a `.1` with no generation stamp"
        )
        print(
            "    Rotated before the #1528 marker, or handed in without its "
            "live file."
        )
        print(
            "    One rollover discards nothing, a second discards the first "
            "archive,"
        )
        print("    and this input cannot say which. Treat as possibly short.")
    elif window.generation == 1 and not window.rotated_present:
        print("  window complete                no rotation has occurred")
    else:
        # Generation >= 2 with nothing discarded: the first rollover fills
        # an empty slot and destroys nothing. Gated on the GENERATION, not
        # on `.1` being in `paths` -- a marked live file passed on its own
        # has provably rotated even though no `.1` was handed in, and
        # printing "no rotation has occurred" under "generation 2" was a
        # self-contradiction.
        print("  window complete                nothing discarded by rotation")
    print(f"  user_prompt_submit fires       {all_fires}")
    print(f"  non-UPS rows (ignored)         {non_ups}")
    print(f"  fires with an outcome (scored) {scored}")
    print(
        f"  no key: gate-skipped           {gate_skipped}   "
        "<- refused AND built nothing: a measured zero"
    )
    # Gate on whether the split was COMPUTABLE, not on the count. With no
    # keyed row `pre_field` is None, and a log whose unclassified count is
    # also 0 -- every fire gate-skipped -- fell through to the else branch
    # and printed the literal "None" as a row count.
    if pre_field is None:
        print(
            f"  no key: unclassified           {unclassified}   <- no keyed row yet, "
            "so pre-field and no-index-work cannot be told apart"
        )
    else:
        print(f"  no key: pre-#1407              {pre_field}   <- shrinks as the log grows")
        print(f"  no key: retrieval, no index    {no_index_work}   <- ongoing, not a wait")
    if unknown:
        print(f"  unrecognised outcome values    {dict(unknown)}  <- vocabulary drift")
    print()

    if scored == 0:
        print("  NO MEASUREMENT YET. No user_prompt_submit row carries the")
        print("  sidecar_outcome field. Let the log accumulate before pricing")
        print("  #1380 — do not read this as a low rebuild rate.")
        return 0

    for name in OUTCOMES:
        n = counts[name]
        print(f"  {name:<14} {n:>6}  {n / scored:>7.2%}")
    print()

    rebuilds = counts["full_rebuild"]
    print("  FULL-REBUILD RATE, on each denominator.")
    print(f"  {unmeasured} unmeasured rows (pre-#1407 + unclassified) are excluded")
    print("  from every denominator below: they can never carry an outcome, so")
    print("  leaving them under the line scores an unmeasured fire as")
    print("  not-a-rebuild. Gate-skipped and no-index-work rows are kept —")
    print("  those are measured zeros, and a per-fire rate needs them.")
    print(f"    of scored fires             {rebuilds}/{scored} = {rebuilds / scored:.2%}")
    if retrieval_fires_measured:
        print(
            f"    of measured retrieval fires {rebuilds}/{retrieval_fires_measured} = "
            f"{rebuilds / retrieval_fires_measured:.2%}"
        )
    if all_fires_measured:
        print(
            f"    of ALL measured UPS fires   {rebuilds}/{all_fires_measured} = "
            f"{rebuilds / all_fires_measured:.2%}"
        )
    print()
    print("  #1380 is priced per-fire, so 'of ALL measured UPS fires' is the")
    print("  cold_rate term in cold_cost x cold_rate. That is also the only one")
    print("  comparable to the 8.5% latency proxy, which was computed over all")
    print("  fires — the scored-fires figure excludes the gate-skipped majority")
    print("  and reads higher for that reason alone.")

    _print_session_position_split(scored_rows)

    if pre_field is not None and pre_field > scored:
        print()
        print(f"  NOTE: {pre_field} pre-#1407 rows against {scored} scored. They are")
        print("  already out of every denominator above, so the rate is not")
        print("  biased by them — but the measured sample is the smaller of the")
        print("  two, so treat it as provisional until the scored count grows.")
    return 0


def bucket_by_session_position(
    scored_rows: list[tuple[str | None, int, str | None, str]],
) -> tuple[Counter[str], Counter[str], int]:
    """Split scored fires into session-FIRST, LATER, and unattributed (#1513).

    A fire is session-FIRST when it is the earliest *scored* fire carrying
    its `session_id`. That is the definition the #1513 measurement used, and
    it is the one the fix targets: the sidecar warm is spawned at
    `SessionStart`, so the fire it can spare is the first one.

    Rows carrying no `session_id` have no position and are returned as a
    third count rather than folded into either bucket. Folding them into
    LATER would drag the session-first rate toward zero exactly the way the
    module docstring forbids for unmeasured rows — the same bias, applied to
    a different axis.

    Ordering is `(ts, read-order)` with missing timestamps sorted last, so a
    rotated-log set is walked in the order the fires happened rather than in
    the order the files were globbed.
    """
    ordered = sorted(scored_rows, key=lambda r: (r[0] is None, r[0] or "", r[1]))
    first: Counter[str] = Counter()
    later: Counter[str] = Counter()
    unattributed = 0
    seen: set[str] = set()
    for _ts, _seq, sid, outcome in ordered:
        if sid is None:
            unattributed += 1
            continue
        if sid in seen:
            later[outcome] += 1
        else:
            seen.add(sid)
            first[outcome] += 1
    return first, later, unattributed


def _print_session_position_split(
    scored_rows: list[tuple[str | None, int, str | None, str]],
) -> None:
    """Report the #1513 split. Two rates, never one."""
    first, later, unattributed = bucket_by_session_position(scored_rows)
    n_first = sum(first.values())
    n_later = sum(later.values())
    print()
    print("  BY POSITION WITHIN THE SESSION (#1513).")
    print("  The cost is a session-first tail, not an average: a rate pooled")
    print("  over both buckets hides it. These two lines must never collapse")
    print("  into one number.")
    for label, bucket, total in (
        ("session-FIRST", first, n_first),
        ("LATER        ", later, n_later),
    ):
        detail = "  ".join(f"{name} {bucket[name]}" for name in OUTCOMES)
        if total:
            rate = f"{bucket['full_rebuild']}/{total} = {bucket['full_rebuild'] / total:.1%}"
        else:
            rate = "no scored fires in this bucket"
        print(f"    {label} fires   {detail}   ->  {rate}")
    if unattributed:
        print(
            f"    no session_id            {unattributed}   <- no position; "
            "excluded from BOTH buckets above, never folded into LATER"
        )
    # Exactly one session-FIRST fire per session that has any scored fire,
    # so this is the session count, printed so the two are read together.
    print(f"    sessions with a scored fire   {n_first}")


if __name__ == "__main__":
    raise SystemExit(main())
