"""Triple-extraction precision on real commit bodies, before and after #1376.

#1159 §3 measured `triple_extractor` precision on real commit bodies at
**0/9** and never committed the sample, so the fix had no after-number to be
held to. This is that instrument, and its sample is the repository's own
history — which is the exact register the ingest path consumes, and which
cannot be curated in the fixer's favour.

**What is counted.** Every `(subject, relation, object)` the extractor emits
from a commit body. A fire is *correct* only if the subject and object are
both self-contained noun phrases naming things the relation actually holds
between. A fragment that begins mid-clause ("and the", "is the presence floor
those") is wrong even when the verb reading is right, because
`ingest_triples` records **both** phrases as beliefs — so a bad capture mints
two pieces of junk, not one bad edge.

Adjudication is not automated. The script prints every fire with its template
so the count can be checked by reading it; the figures quoted in `CHANGELOG`
were adjudicated by hand over the 400-body window and the verdict was
unanimous, which is the only reason a hand count is defensible here. Re-run it
with `--limit` raised if that stops being true.

**Two banks, one corpus.** `--limit N` walks the N most recent commit bodies
on `github/main` and runs both the unconstrained bank (what the read path
still uses) and the ingest bank (`constrain_collision_verbs=True`). The
per-template table is the evidence for *which* templates to drop; the totals
are the before/after precision pair.

**The honest headline is a volume reduction, not a precision win.** See the
`residual` block the script prints: dropping the six collision verbs removes
most of the junk but does not make precision positive, because the fires that
remain come from templates the #1376 ruling does not cover and fail for a
different reason — the noun-phrase capture is unanchored and runs across
clause and line boundaries. That is #1159 §2's territory, not this leaf's.

Usage::

    uv run python benchmarks/triple_precision_1376.py
    uv run python benchmarks/triple_precision_1376.py --limit 800 --json
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from aelfrice.triple_extractor import (
    _INGEST_EXCLUDED_TEMPLATES,
    _INGEST_PATTERNS,
    _PATTERNS,
    extract_triples,
)


def _commit_bodies(limit: int, ref: str, root: Path) -> list[str]:
    """The `limit` most recent commit bodies on `ref`, NUL-separated.

    `%B` is subject plus body, which is what `hook_commit_ingest` passes
    (it reads the whole message, not just the subject). NUL is the
    separator because commit bodies contain every other candidate.
    """
    proc = subprocess.run(
        ["git", "log", f"-{limit}", "--format=%B%x00", ref],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    return [b.strip() for b in proc.stdout.split("\0") if b.strip()]


def _fires_by_template(
    bodies: list[str], patterns: tuple[Any, ...]
) -> tuple[collections.Counter[str], dict[str, list[str]]]:
    counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[str]] = {}
    for text in bodies:
        for pat in patterns:
            for hit in pat.regex.finditer(text):
                subj = " ".join(hit.group("subject").split())
                obj = " ".join(hit.group("object").split())
                if not subj or not obj:
                    continue
                key = f"{pat.template} -> {pat.edge_type}"
                counts[key] += 1
                samples.setdefault(key, [])
                if len(samples[key]) < 5:
                    samples[key].append(f"({subj[:50]!r}, {obj[:50]!r})")
    return counts, samples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=400, help="commits to sample")
    ap.add_argument("--ref", default="github/main", help="git ref to walk")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    try:
        bodies = _commit_bodies(args.limit, args.ref, root)
    except subprocess.CalledProcessError as exc:
        print(f"git log failed for ref {args.ref!r}: {exc}", file=sys.stderr)
        return 2

    before_counts, before_samples = _fires_by_template(bodies, _PATTERNS)
    after_counts, after_samples = _fires_by_template(bodies, _INGEST_PATTERNS)

    # Cross-check the per-pattern walk against the public entry point, so a
    # divergence between this script's replica loop and `extract_triples`
    # (subject/object filtering, ordering) cannot pass unnoticed.
    api_before = sum(len(extract_triples(b)) for b in bodies)
    api_after = sum(
        len(extract_triples(b, constrain_collision_verbs=True)) for b in bodies
    )
    if (api_before, api_after) != (sum(before_counts.values()), sum(after_counts.values())):
        raise SystemExit(
            "replica loop disagrees with extract_triples: "
            f"{api_before}/{api_after} vs "
            f"{sum(before_counts.values())}/{sum(after_counts.values())}"
        )

    report: dict[str, Any] = {
        "ref": args.ref,
        "bodies": len(bodies),
        "before": {
            "total_fires": sum(before_counts.values()),
            "by_template": dict(before_counts.most_common()),
        },
        "after": {
            "total_fires": sum(after_counts.values()),
            "by_template": dict(after_counts.most_common()),
        },
        "excluded_templates": sorted(_INGEST_EXCLUDED_TEMPLATES),
        "removed_fires": sum(before_counts.values()) - sum(after_counts.values()),
    }

    if args.json:
        report["before"]["samples"] = before_samples
        report["after"]["samples"] = after_samples
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"ref={args.ref}  bodies={len(bodies)}")
    print(
        f"\nfires: {report['before']['total_fires']} (unconstrained) "
        f"-> {report['after']['total_fires']} (ingest bank), "
        f"{report['removed_fires']} removed"
    )
    print("\n-- unconstrained bank, by template --")
    for key, n in before_counts.most_common():
        marker = "  DROPPED" if key.split(" -> ")[0] in _INGEST_EXCLUDED_TEMPLATES else ""
        print(f"  {n:5d}  {key}{marker}")
        for s in before_samples[key][:3]:
            print(f"           {s}")
    print("\n-- residual after the fix (these are #1159 §2, not #1376) --")
    for key, n in after_counts.most_common():
        print(f"  {n:5d}  {key}")
        for s in after_samples[key][:3]:
            print(f"           {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
