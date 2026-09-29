"""Count aelfrice hook outputs the host saved to a file instead of inlining (#1639).

The host inlines at most 10,000 characters of a hook's output. Past that
it saves the output and injects a `<persisted-output>` stub carrying a
2,000-character preview and the saved file's path
(https://code.claude.com/docs/en/hooks.md). This reads the host's
session transcripts and reports, for aelfrice hook outputs:

- `inline` and `saved`: how many were inlined and how many were saved,
  and `saved_by_event`: the saved ones split by hook event.
- `saved_size_kb`: the minimum, median, and maximum size the host
  reported for saved outputs.
- `saved_with_locks`: of the saved outputs whose full text (read from the
  saved file, while it still exists) carries user locks, how many lost at
  least one of those locks from the preview (`lost_from_preview`) and how
  many kept them all (`all_in_preview`). `file_gone` counts saved outputs
  whose file no longer exists.
- `max_inline_chars`: the largest output the host inlined, which brackets
  the limit from below.

It is a contributor diagnostic, not a CI gate: it reads local transcripts.
The output is aggregate counts and is safe to paste into an issue.

Usage:

    uv run python scripts/hook_output_census.py [--root DIR] [--since ISO8601]

`--root` defaults to the host's transcript directory under your home
directory. `--since` counts only outputs timestamped at or after it.
Exits 2 when the root does not exist.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_LOCK_RE = re.compile(r'<belief id="([0-9A-Za-z]+)" lock="user"')
_SAVED_RE = re.compile(r"Full output saved to: (\S+)")
_TOO_LARGE_RE = re.compile(r"Output too large \(([\d.]+)KB\)")
# Bounds on the scan, so a runaway directory cannot hang the census.
_MAX_FILES = 20_000
_MAX_LINES_PER_FILE = 1_000_000


def _default_root() -> Path:
    return Path.home() / ".claude" / "projects"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def census(root: Path, since: datetime | None = None) -> dict[str, object]:
    """Return the #1639 counts for every transcript under `root`."""
    inline = saved = lost = kept = gone = 0
    max_inline = 0
    sizes: list[float] = []
    by_event: dict[str, int] = {}
    for path in sorted(root.glob("*/*.jsonl"))[:_MAX_FILES]:
        with path.open(errors="replace") as fh:
            for n, line in enumerate(fh):
                if n >= _MAX_LINES_PER_FILE:
                    break
                if '"hook_success"' not in line or "aelfrice" not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                att = event.get("attachment")
                if not isinstance(att, dict) or att.get("type") != "hook_success":
                    continue
                content = att.get("content") or ""
                if not isinstance(content, str) or "aelfrice" not in content:
                    continue
                ts = event.get("timestamp")
                if since is not None and (not ts or _parse_ts(ts) < since):
                    continue
                size = _TOO_LARGE_RE.search(content)
                if size is None:
                    inline += 1
                    max_inline = max(max_inline, len(content))
                    continue
                saved += 1
                sizes.append(float(size.group(1)))
                hook_event = str(att.get("hookEvent", "?"))
                by_event[hook_event] = by_event.get(hook_event, 0) + 1
                target = _SAVED_RE.search(content)
                if target is None or not Path(target.group(1)).is_file():
                    gone += 1
                    continue
                full = Path(target.group(1)).read_text(errors="replace")
                full_locks = set(_LOCK_RE.findall(full))
                if not full_locks:
                    continue
                if full_locks - set(_LOCK_RE.findall(content)):
                    lost += 1
                else:
                    kept += 1
    sizes.sort()
    return {
        "inline": inline,
        "saved": saved,
        "saved_by_event": dict(sorted(by_event.items())),
        "saved_size_kb": (
            {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]}
            if sizes else None
        ),
        "saved_with_locks": {
            "lost_from_preview": lost,
            "all_in_preview": kept,
            "file_gone": gone,
        },
        "max_inline_chars": max_inline,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--since", default=None)
    args = parser.parse_args(argv)
    root: Path = args.root or _default_root()
    if not root.is_dir():
        print(f"hook_output_census: {root} not found", file=sys.stderr)
        return 2
    since = _parse_ts(args.since) if args.since else None
    print(json.dumps(census(root, since), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
