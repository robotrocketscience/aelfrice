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

Usage:
    uv run python benchmarks/sidecar_rebuild_rate.py [AUDIT_LOG ...]

With no arguments this globs `hook_audit.jsonl*`, which includes **rotated**
logs (`hook_audit.jsonl.1`, ...). The totals it prints are therefore across all
rotations, not the single live file — pass an explicit path for a single-file
count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

OUTCOMES = ("fresh", "incremental", "full_rebuild")


def _default_logs() -> list[Path]:
    from aelfrice.db_paths import _git_common_dir

    git_dir = _git_common_dir()
    if git_dir is None:
        raise SystemExit("not in a git work-tree; pass audit log paths")
    d = git_dir / "aelfrice"
    return sorted(d.glob("hook_audit.jsonl*"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
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
    unkeyed: list[str | None] = []
    scored_ts: list[str] = []

    for path in logs:
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("hook") != "user_prompt_submit":
                non_ups += 1
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

    print("#1407 — BM25 sidecar outcome per user_prompt_submit fire")
    for p in logs:
        print(f"  log                            {p}")
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

    if pre_field is not None and pre_field > scored:
        print()
        print(f"  NOTE: {pre_field} pre-#1407 rows against {scored} scored. They are")
        print("  already out of every denominator above, so the rate is not")
        print("  biased by them — but the measured sample is the smaller of the")
        print("  two, so treat it as provisional until the scored count grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
