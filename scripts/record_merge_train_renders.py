#!/usr/bin/env python3
"""Re-record the GitHub renders `tests/test_merge_train_close_keywords_1549.py` replays.

Usage:
    python3 scripts/record_merge_train_renders.py --dry-run
    python3 scripts/record_merge_train_renders.py --repo OWNER/NAME

`scripts/merge_train_linked_issues.py` decides which issues a merged pull
request closes by rendering its body through GitHub's own Markdown renderer and
reading which references GitHub anchored. The tests must not reach the network,
so they replay responses recorded in
`tests/data/merge_train_github_renders.json`; this is what records them.

Each record holds the body that was sent and the HTML that came back. This
re-sends every body and rewrites its HTML. `--dry-run` sends them, reports which
records GitHub now answers differently, changes nothing, and exits 1 if any
drifted -- run it when a test that pins GitHub's behaviour starts to look wrong.

Exits non-zero on any render failure, and on drift under `--dry-run`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_train_linked_issues import (  # noqa: E402
    RendererUnavailable,
    render_markdown,
)

RECORDS = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "merge_train_github_renders.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo",
        default="",
        help=(
            "render context; defaults to the `_context` the file was recorded "
            "with, then to $GITHUB_REPOSITORY"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report drift and write nothing; exit 1 if any record drifted",
    )
    args = ap.parse_args(argv)

    data = json.loads(RECORDS.read_text(encoding="utf-8"))
    repo = args.repo or data.get("_context") or os.environ.get(
        "GITHUB_REPOSITORY", ""
    )
    if not repo:
        print("error: no render context; pass --repo OWNER/NAME", file=sys.stderr)
        return 2

    drifted = []
    for name, record in sorted(data["records"].items()):
        try:
            html = render_markdown(record["body"], repo)
        except RendererUnavailable as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 2
        if html != record["html"]:
            drifted.append(name)
        record["html"] = html

    if args.dry_run:
        for name in drifted:
            print(f"drifted: {name}")
        print(f"{len(drifted)} of {len(data['records'])} records drifted.")
        return 1 if drifted else 0

    data["_context"] = repo
    RECORDS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"rewrote {RECORDS} ({len(drifted)} record(s) changed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
