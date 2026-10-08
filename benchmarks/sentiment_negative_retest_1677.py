"""Negative sentiment-from-prose precision on fresh prompts (#1677).

#1647 turned praise on by default and left complaints off behind
`[feedback] sentiment_negative`, because the tightened negative patterns
measured 73.5% and 61.8% precise on held-out prompts against a pre-registered
bar of 70% on each of two graders. The ruling was to re-test on prompts
collected after 2026-09-30, since the earlier prompts were used for tuning.
This is that instrument.

**Population.** With the negative lane off, every negative match lands in the
hook audit as a `sentiment_feedback` row with `abstained = "negative_disabled"`.
Since #1677 the row also lists `target_ids`, the prior turn's beliefs the fire
would have demoted. A row counts when it is timestamped on or after
`--since`, and its `prompt_prefix` is shorter than the audit's 200-character
cap. A prefix at the cap may be a truncated longer prompt, so it's excluded,
which keeps the population to prompts of at most 200 characters.

**Grading is not automated.** `sample` writes a seeded sheet of fires. Two
graders label each fire independently, `true` when the prompt is a verdict on
the previous answer and `false` when it isn't, or `null` when unclear. `score`
reads both label files and reports each grader's precision with its Wilson
95% lower bound, Cohen's kappa on the fires both graded, and the verdict
against the bar. A fire a grader marked unclear is left out of that grader's
precision and out of kappa.

The sheet holds prompt text. Keep it, and the label files, out of the public
repository.

Usage::

    uv run python -m benchmarks.sentiment_negative_retest_1677 count --audit <hook_audit.jsonl>...
    uv run python -m benchmarks.sentiment_negative_retest_1677 sample --audit <...> --seed 1677 --out sheet.jsonl
    uv run python -m benchmarks.sentiment_negative_retest_1677 score --sheet sheet.jsonl --labels a.jsonl b.jsonl

`sample --dry-run` prints the counts and the sample size without writing the
sheet. Every subcommand exits 1 on bad input, and `score` exits 1 when the
bar fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Final

from benchmarks.context_rebuilder.kappa import cohens_kappa

DEFAULT_SINCE: Final[str] = "2026-10-01"
"""First day of the fresh window: prompts after 2026-09-30 (#1677 AC1)."""
PROMPT_PREFIX_CAP: Final[int] = 200
"""`hook.AUDIT_PROMPT_PREFIX_CAP`: a prefix this long may be truncated."""
BAR: Final[float] = 0.70
"""Pre-registered precision bar, on each grader (#1647)."""
DEFAULT_MAX: Final[int] = 200
"""Largest sample #1647 graded."""
Z_95: Final[float] = 1.959963984540054


def wilson_lower(successes: int, n: int, z: float = Z_95) -> float:
    """Wilson score interval lower bound; 0.0 when n is 0."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def fire_id(row: dict[str, Any]) -> str:
    """Stable id for one audit row, from its time, session, and prompt."""
    key = f"{row.get('ts')}\0{row.get('session_id')}\0{row.get('prompt_prefix')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Every JSON object line of the given audit files; others skipped."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def fresh_negative_fires(
    rows: list[dict[str, Any]], since: str = DEFAULT_SINCE,
) -> list[dict[str, Any]]:
    """The population: disabled negative fires on short prompts since `since`.

    Deduplicated by `fire_id`, since a rotated audit file and its successor
    can both be passed. Sorted by id so sampling doesn't depend on file order.
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("hook") != "sentiment_feedback":
            continue
        if row.get("abstained") != "negative_disabled":
            continue
        if str(row.get("ts") or "") < since:
            continue
        prefix = row.get("prompt_prefix")
        if not isinstance(prefix, str) or not prefix or len(prefix) >= PROMPT_PREFIX_CAP:
            continue
        seen.setdefault(fire_id(row), row)
    return [seen[k] for k in sorted(seen)]


def draw_sample(
    fires: list[dict[str, Any]], seed: int, max_n: int = DEFAULT_MAX,
) -> list[dict[str, Any]]:
    """A seeded sample of at most `max_n` fires, as grading-sheet rows."""
    chosen = fires if len(fires) <= max_n else random.Random(seed).sample(fires, max_n)
    return [
        {
            "id": fire_id(row),
            "ts": row.get("ts"),
            "prompt": row.get("prompt_prefix"),
            "pattern": row.get("pattern"),
            "matched_text": row.get("matched_text"),
            "target_ids": row.get("target_ids"),
        }
        for row in sorted(chosen, key=fire_id)
    ]


def read_labels(path: Path) -> dict[str, bool | None]:
    """{fire id: True | False | None} from a JSONL file of {"id", "correct"}."""
    labels: dict[str, bool | None] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            correct = row.get("correct")
            if not isinstance(row.get("id"), str) or correct not in (True, False, None):
                raise ValueError(f"{path}:{n}: need an id and correct=true|false|null")
            labels[row["id"]] = correct
    return labels


def score(
    sheet_ids: list[str], a: dict[str, bool | None], b: dict[str, bool | None],
) -> dict[str, Any]:
    """Per-grader precision, Wilson bounds, kappa, and the verdict."""
    missing = [i for i in sheet_ids if i not in a or i not in b]
    if missing:
        raise ValueError(f"{len(missing)} sheet fires lack a label from both graders")
    graders: list[dict[str, Any]] = []
    for labels in (a, b):
        graded = [labels[i] for i in sheet_ids if labels[i] is not None]
        correct = sum(1 for x in graded if x)
        n = len(graded)
        graders.append({
            "graded": n,
            "correct": correct,
            "precision": correct / n if n else 0.0,
            "wilson_lower": wilson_lower(correct, n),
        })
    both = [i for i in sheet_ids if a[i] is not None and b[i] is not None]
    kappa = cohens_kappa([bool(a[i]) for i in both], [bool(b[i]) for i in both])
    passed = all(g["graded"] > 0 and g["precision"] >= BAR for g in graders)
    return {
        "fires": len(sheet_ids),
        "graders": graders,
        "kappa": kappa,
        "kappa_n": len(both),
        "agreed": sum(1 for i in both if a[i] == b[i]),
        "bar": BAR,
        "passed": passed,
    }


def _date_range(fires: list[dict[str, Any]]) -> str:
    stamps = sorted(str(r.get("ts")) for r in fires)
    return f"{stamps[0]}..{stamps[-1]}" if stamps else "(none)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("count", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--audit", type=Path, nargs="+", required=True)
        p.add_argument("--since", default=DEFAULT_SINCE)
    sample = sub.choices["sample"]
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--max", type=int, default=DEFAULT_MAX)
    sample.add_argument("--out", type=Path)
    sample.add_argument("--dry-run", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--sheet", type=Path, required=True)
    sc.add_argument("--labels", type=Path, nargs=2, required=True)
    args = parser.parse_args(argv)

    try:
        if args.cmd in ("count", "sample"):
            fires = fresh_negative_fires(read_rows(args.audit), args.since)
            with_targets = sum(1 for r in fires if r.get("target_ids"))
            print(f"fires={len(fires)} with_targets={with_targets} "
                  f"since={args.since} range={_date_range(fires)}")
            if args.cmd == "count":
                return 0
            rows = draw_sample(fires, args.seed, args.max)
            print(f"sample={len(rows)} seed={args.seed}")
            if args.dry_run:
                return 0
            if args.out is None:
                print("error: sample needs --out unless --dry-run", file=sys.stderr)
                return 1
            args.out.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            print(f"wrote {args.out}")
            return 0
        sheet_ids = [str(r["id"]) for r in read_rows([args.sheet])]
        report = score(sheet_ids, read_labels(args.labels[0]), read_labels(args.labels[1]))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
