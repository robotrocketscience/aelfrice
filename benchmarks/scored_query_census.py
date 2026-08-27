#!/usr/bin/env python3
"""Is the `input.scored_query` population big enough to measure on yet (#1516)?

#1516 parks a measurement behind a threshold that was **committed before any
measurement was taken**, which is the whole point of it:

    Run the measurement when `input.scored_query` holds >= 500 distinct
    queries spanning >= 30 days. Below either bound, report the counts.

This script is the census that decides that, and nothing else. It does not run
the measurement, and it deliberately exposes **no flag for the two bounds** --
they are module constants so that re-running the census cannot quietly move the
bar it is checking. A future session that wants a different bar has to change
this file, in a diff, with a reason.

What it counts, and why each number is separate:

* rows scanned -- the whole corpus, so a share has a denominator;
* rows carrying `input.scored_query` -- forward-only (#1405); rows written
  before that change have no such key, and absent means "unknown", never
  "no transform was applied";
* DISTINCT scored queries -- the bound is on distinct queries, and repeats of
  one query are one query. Reporting only the row count inflates the
  population by however often an operator re-asks the same thing;
* active days and the calendar span -- also separate. Four rows a day for four
  days is not a 30-day span, and a 30-day span with two active days in it is
  not 30 days of accrual either. Both are printed; the bound is on the span.

Read-only. Never opens a `MemoryStore` -- see `db_paths` for why a diagnostic
must not, since opening one runs pending migrations. Rows are read with the
loader in `rebuild_log_query_population`, so the two censuses cannot drift on
what counts as a parseable row.

Where the logs live: `<git-common-dir>/aelfrice/rebuild_logs/<session>.jsonl`,
derived in `context_rebuilder._rebuild_log_dir_for_db` from the repo-local DB
path resolved by `db_paths.db_path`. It is repo-local, so a machine has one
such directory per repo, plus `~/.aelfrice/rebuild_logs` for work done outside
any git work-tree. `--discover` finds them all; a machine-wide census is what
#1516's prior baseline used and is what makes the numbers comparable.

Usage::

    # this repo only
    python benchmarks/scored_query_census.py

    # every rebuild_logs directory under a root (the #1516 baseline's scope)
    python benchmarks/scored_query_census.py --discover ~

    # what would be read, without reading it
    python benchmarks/scored_query_census.py --discover ~ --dry-run

    # machine-readable, for stamping alongside a result
    python benchmarks/scored_query_census.py --discover ~ --json

Exit codes::

    0   census ran; BOTH bounds met -- the measurement is runnable
    2   failure: no log directory resolved, or no parseable rows in it
    3   census ran; a bound is unmet -- do not run the measurement

3 is a real outcome, not an error, but it is non-zero on purpose: a caller that
gates on "did this succeed" must not read "not enough data yet" as a green
light to measure anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Final

from benchmarks.rebuild_log_query_population import load_rows

# #1516's committed bounds. Constants, not flags -- see the module docstring.
THRESHOLD_DISTINCT_QUERIES: Final[int] = 500
THRESHOLD_SPAN_DAYS: Final[int] = 30

#: `_build_rebuild_log_record` stamps `datetime.now(timezone.utc)` as
#: ``%Y-%m-%dT%H:%M:%SZ``, so the first ten characters are the UTC calendar
#: day. Sliced rather than parsed: a malformed row must fall out of the day
#: histogram, not abort the census.
_TS_DATE_WIDTH: Final[int] = 10

#: Directory name under `<git-common-dir>/aelfrice/`, mirroring
#: `context_rebuilder.REBUILD_LOG_DIRNAME`. Duplicated rather than imported to
#: keep this script's import graph stdlib-side.
REBUILD_LOG_DIRNAME: Final[str] = "rebuild_logs"

#: Subtrees a discovery walk must not descend into. Package and cache trees
#: hold no rebuild logs and dominate the walk time on a developer machine.
_PRUNE_DIRNAMES: Final[frozenset[str]] = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "Library", "site-packages", ".cache",
})

_DEFAULT_MAX_DEPTH: Final[int] = 4


def _git_common_dir(here: Path) -> Path | None:
    """`here`'s git common directory, or None if `here` is not a work-tree.

    In an ordinary checkout `.git` is a directory and is itself the answer.
    In a LINKED WORK-TREE `.git` is a plain file reading `gitdir: <path>`,
    and that path is the per-worktree gitdir, not the common one; the
    common dir is named by the `commondir` file beside it. Resolved the
    same way `db_paths` does, because the audit and rebuild logs live
    under the COMMON dir and are therefore shared by every linked
    work-tree of a repo.

    Read off the filesystem rather than by shelling out to `git
    rev-parse`, so a census over a large tree does not fork once per
    visited directory.
    """
    dot_git = here / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    try:
        text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:"):].strip())
    if not gitdir.is_absolute():
        gitdir = (here / gitdir).resolve()
    commondir = gitdir / "commondir"
    if not commondir.is_file():
        return gitdir
    try:
        rel = commondir.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return gitdir
    if not rel:
        return gitdir
    common = Path(rel)
    return common if common.is_absolute() else (gitdir / common).resolve()


def discover_log_dirs(root: Path, max_depth: int = _DEFAULT_MAX_DEPTH) -> list[Path]:
    """Every rebuild-log directory at or under `root`, sorted.

    Three shapes are checked at each visited directory: `<git-common-dir>/
    aelfrice/rebuild_logs` (the repo-local store, which is where essentially
    all of the corpus is), `<d>/.aelfrice/rebuild_logs` (the home fallback
    used when cwd is outside any work-tree), and `<d>` itself when it is
    already named `rebuild_logs`, so that pointing `--discover` straight at
    a log directory does what "at or under `root`" says. `.git` itself is
    pruned from the descent so the walk does not wander into object storage.

    The git-common-dir resolution is not decoration: in a linked work-tree
    `.git` is a FILE, so testing `<d>/.git/aelfrice/...` for a directory
    silently finds nothing there. Results are de-duplicated by resolved
    path, so the many linked work-trees of one repo contribute their
    shared common dir once rather than once each.
    """
    root = root.expanduser()
    found: set[Path] = set()
    root_depth = len(root.resolve().parts)
    for dirpath, dirnames, _files in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if here.name == REBUILD_LOG_DIRNAME:
            found.add(here.resolve())
        candidates = [here / ".aelfrice" / REBUILD_LOG_DIRNAME]
        common = _git_common_dir(here)
        if common is not None:
            candidates.append(common / "aelfrice" / REBUILD_LOG_DIRNAME)
        for candidate in candidates:
            if candidate.is_dir():
                found.add(candidate.resolve())
        if len(here.resolve().parts) - root_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in _PRUNE_DIRNAMES)
    return sorted(found)


def jsonl_files(log_dirs: list[Path]) -> list[Path]:
    """Every `*.jsonl` under the given directories, sorted and deduplicated."""
    files: set[Path] = set()
    for d in log_dirs:
        files.update(p.resolve() for p in d.glob("*.jsonl"))
    return sorted(files)


def corpus_identity(log_dirs: list[Path], files: list[Path]) -> dict[str, Any]:
    """Everything needed to say which corpus produced a count.

    A population figure without its corpus identity is what `benchmarks/
    README.md` warns about: the same script over a machine that has since
    accrued more sessions gives a different number, and neither is wrong.
    The digest is over `(path, byte size)` for each file, which changes
    whenever the append-only logs grow.
    """
    h = hashlib.sha256()
    total_bytes = 0
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:  # pragma: no cover - raced deletion
            size = -1
        total_bytes += max(size, 0)
        h.update(f"{path}\0{size}\0".encode())
    return {
        "log_dirs": [str(d) for d in log_dirs],
        "n_log_dirs": len(log_dirs),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "digest": h.hexdigest()[:16],
    }


def _scored_query(row: dict[str, Any]) -> str | None:
    """The row's scored query, or None when it carries none.

    Both the absent key (rows predating #1405) and an explicit null (the
    default when a caller passes nothing) mean the same thing here: this row
    contributes no query to the population. So does a row whose `input` is not
    a mapping at all: like `_day`, a malformed row falls out of the count
    rather than aborting the census with a traceback.

    The **stripped** form is returned, not the raw one, because it is also the
    form blankness is judged on. Keying identity on the raw value while
    judging blankness on the stripped one would count `"q"`, `" q"` and `"q "`
    as three distinct queries, inflating the exact number #1516's 500-query
    bound is checked against.
    """
    inner = row.get("input")
    if not isinstance(inner, dict):
        return None
    value = inner.get("scored_query")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _day(row: dict[str, Any]) -> str | None:
    """UTC calendar day of the row's `ts`, or None when unusable."""
    ts = row.get("ts")
    if not isinstance(ts, str) or len(ts) < _TS_DATE_WIDTH:
        return None
    day = ts[:_TS_DATE_WIDTH]
    try:
        date.fromisoformat(day)
    except ValueError:
        return None
    return day


def census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the scored-query population in `rows`.

    Total-vs-distinct and rows-per-day-vs-distinct-per-day are kept apart
    throughout: they are different numbers and #1516's bound is on the
    distinct one.
    """
    scored_rows = 0
    undated = 0
    distinct: set[str] = set()
    rows_per_day: Counter[str] = Counter()
    queries_per_day: dict[str, set[str]] = {}

    for row in rows:
        query = _scored_query(row)
        if query is None:
            continue
        scored_rows += 1
        distinct.add(query)
        day = _day(row)
        if day is None:
            undated += 1
            continue
        rows_per_day[day] += 1
        queries_per_day.setdefault(day, set()).add(query)

    days = sorted(rows_per_day)
    first, last = (days[0], days[-1]) if days else (None, None)
    if first is not None and last is not None:
        span_days = (date.fromisoformat(last) - date.fromisoformat(first)).days + 1
    else:
        span_days = 0

    return {
        "rows_scanned": len(rows),
        "rows_with_scored_query": scored_rows,
        "distinct_scored_queries": len(distinct),
        "rows_without_usable_ts": undated,
        "active_days": len(days),
        "first_day": first,
        "last_day": last,
        "span_days": span_days,
        "per_day": [
            {
                "day": d,
                "rows": rows_per_day[d],
                "distinct": len(queries_per_day[d]),
            }
            for d in days
        ],
    }


def verdict(counts: dict[str, Any]) -> dict[str, Any]:
    """Which bounds are met, and the linear distance to the query bound.

    The accrual rate is reported with an explicit warning attached because on
    the #1516 baseline it was severely lumpy -- 27 of 41 queries landed on one
    day -- and a linear extrapolation from a lumpy rate is a weak estimate,
    not a forecast. Reported so the number can be quoted with its caveat
    rather than re-derived without one.
    """
    distinct = counts["distinct_scored_queries"]
    span = counts["span_days"]
    per_day = counts["per_day"]
    rate = (distinct / span) if span else 0.0
    shortfall = max(THRESHOLD_DISTINCT_QUERIES - distinct, 0)
    busiest = max((d["distinct"] for d in per_day), default=0)
    return {
        "queries_bound_met": distinct >= THRESHOLD_DISTINCT_QUERIES,
        "span_bound_met": span >= THRESHOLD_SPAN_DAYS,
        "runnable": (
            distinct >= THRESHOLD_DISTINCT_QUERIES and span >= THRESHOLD_SPAN_DAYS
        ),
        "queries_short_by": shortfall,
        "distinct_per_span_day": rate,
        "days_to_bound_linear": (shortfall / rate) if rate else None,
        "busiest_day_distinct": busiest,
        "busiest_day_share": (busiest / distinct) if distinct else 0.0,
    }


def _render(counts: dict[str, Any], v: dict[str, Any], ident: dict[str, Any]) -> str:
    lines = [
        "input.scored_query population census (#1516)",
        "=" * 62,
        f"log dirs   : {ident['n_log_dirs']}",
        f"files      : {ident['n_files']}  ({ident['total_bytes']} bytes)",
        f"digest     : {ident['digest']}",
        "",
        "Counts",
        "-" * 62,
        f"  rows scanned                    {counts['rows_scanned']:>7}",
        f"  rows with input.scored_query    {counts['rows_with_scored_query']:>7}",
        f"  DISTINCT scored queries         {counts['distinct_scored_queries']:>7}",
        f"  days with >=1 scored query      {counts['active_days']:>7}",
        f"  calendar span (inclusive)       {counts['span_days']:>7}",
        f"  window                          "
        f"{counts['first_day'] or '-'} -> {counts['last_day'] or '-'}",
    ]
    if counts["rows_without_usable_ts"]:
        lines.append(
            f"  scored rows with no usable ts   "
            f"{counts['rows_without_usable_ts']:>7}   <- excluded from the days"
        )
    lines += ["", "Per day", "-" * 62]
    if counts["per_day"]:
        lines.append("  day           rows   distinct")
        for entry in counts["per_day"]:
            lines.append(
                f"  {entry['day']}  {entry['rows']:>6}   {entry['distinct']:>8}"
            )
    else:
        lines.append("  (no scored queries recorded)")

    lines += [
        "",
        f"Committed bounds (#1516): >= {THRESHOLD_DISTINCT_QUERIES} distinct "
        f"queries, >= {THRESHOLD_SPAN_DAYS}-day span",
        "-" * 62,
        f"  distinct queries  {counts['distinct_scored_queries']:>6} / "
        f"{THRESHOLD_DISTINCT_QUERIES:<6} "
        f"{'MET' if v['queries_bound_met'] else 'UNMET'}",
        f"  span (days)       {counts['span_days']:>6} / "
        f"{THRESHOLD_SPAN_DAYS:<6} "
        f"{'MET' if v['span_bound_met'] else 'UNMET'}",
        "",
        f"  accrual           {v['distinct_per_span_day']:.2f} distinct "
        "queries per span-day",
        f"  short by          {v['queries_short_by']} queries"
        + (
            f"  (~{v['days_to_bound_linear']:.0f} days at that rate)"
            if v["days_to_bound_linear"] is not None
            else ""
        ),
        f"  lumpiness         busiest single day holds "
        f"{v['busiest_day_distinct']} of "
        f"{counts['distinct_scored_queries']} distinct queries "
        f"({v['busiest_day_share']:.0%}) -- a linear extrapolation from a rate "
        "this lumpy is a weak estimate, not a forecast",
        "",
        (
            "VERDICT: both bounds met -- the measurement is runnable"
            if v["runnable"]
            else "VERDICT: a bound is unmet -- do NOT run the measurement"
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Census of the input.scored_query population (#1516).",
    )
    parser.add_argument(
        "--logs", type=Path, action="append", default=None, metavar="DIR",
        help="a rebuild_log directory to read; repeatable. Default: this "
             "repo's .git/aelfrice/rebuild_logs.",
    )
    parser.add_argument(
        "--discover", type=Path, action="append", default=None, metavar="ROOT",
        help="find every rebuild_logs directory under ROOT; repeatable. "
             "Use `--discover ~` for the machine-wide census #1516 baselines.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=_DEFAULT_MAX_DEPTH,
        help=f"discovery descent limit below each ROOT "
             f"(default: {_DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the directories and files that would be read, then exit "
             "without reading any of them",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the counts as JSON on stdout instead of the text report",
    )
    args = parser.parse_args(argv)

    log_dirs: list[Path] = []
    for root in args.discover or []:
        log_dirs.extend(discover_log_dirs(root, max_depth=args.max_depth))
    for explicit in args.logs or []:
        log_dirs.append(explicit.expanduser())
    if not args.discover and not args.logs:
        log_dirs.append(Path(".git") / "aelfrice" / REBUILD_LOG_DIRNAME)

    # Resolved before de-duplication: `jsonl_files` dedupes files by their
    # resolved path, so a directory reachable by two spellings (a relative
    # `--logs`, a symlink, `~`) would otherwise survive this set, be read
    # twice by `load_rows`, and double every row count under a corpus
    # identity that still reported one file.
    log_dirs = sorted(
        {d for d in (x.expanduser().resolve() for x in log_dirs) if d.is_dir()}
    )
    if not log_dirs:
        print("no rebuild_log directory resolved", file=sys.stderr)
        return 2

    files = jsonl_files(log_dirs)
    if args.dry_run:
        for d in log_dirs:
            print(f"dir  {d}")
        for f in files:
            print(f"file {f}")
        print(f"{len(log_dirs)} directories, {len(files)} files (nothing read)")
        return 0

    rows: list[dict[str, Any]] = []
    for d in log_dirs:
        # A directory that resolved a moment ago can be unreadable now — a
        # permissions change, an unmounted share, a worktree pruned by a
        # concurrent session. That is a census failure (2), the same class
        # as resolving no directory at all, not a traceback: the caller
        # reads the exit code to decide whether a population figure exists.
        try:
            rows.extend(load_rows(d))
        except OSError as exc:
            print(f"cannot read {d}: {exc}", file=sys.stderr)
            return 2
    if not rows:
        print(
            f"no parseable rebuild_log rows across {len(log_dirs)} directories",
            file=sys.stderr,
        )
        return 2

    counts = census(rows)
    v = verdict(counts)
    ident = corpus_identity(log_dirs, files)

    if args.json:
        print(json.dumps(
            {"corpus": ident, "counts": counts, "verdict": v,
             "thresholds": {
                 "distinct_queries": THRESHOLD_DISTINCT_QUERIES,
                 "span_days": THRESHOLD_SPAN_DAYS,
             }},
            indent=2, sort_keys=True,
        ))
    else:
        print(_render(counts, v, ident))

    return 0 if v["runnable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
