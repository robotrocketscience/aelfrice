"""Operator-side gate aggregator surface for `aelf gate list` (#475).

Reads open issues with `gate:*` labels via `gh`, counts asks-block lines
in the most recent `[gate:ratify]` comment per ratify issue, and emits a
single screenful so the operator can speed-read pending decisions.

The four sections, in print order:
- gate:ratify   — open design decisions; operator picks an option
- gate:prereq   — waiting on sister issues to land
- gate:bench    — waiting on bench evidence (label is `bench-gated`)
- gate:license  — waiting on legal/license review

Sort: oldest-open-ask-first for ratify (proxy: oldest `[gate:ratify]`
comment); oldest-issue-first (lowest issue number) for the rest.

Repo is auto-detected from cwd's git remote — the same model as
`aelf-claim.sh` and the rest of the repo's gh-driven tooling.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Each tuple: (label as filtered on gh, header label as printed).
# `bench-gated` is the existing label; the printed header `gate:bench`
# matches the four-shape vocabulary in #475's body.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("gate:ratify", "gate:ratify"),
    ("gate:prereq", "gate:prereq"),
    ("bench-gated", "gate:bench"),
    ("gate:license", "gate:license"),
)

_ASKS_RE = re.compile(r"^\*\*Ask (\d+):", re.MULTILINE)

GhRunner = Callable[[Sequence[str]], str]


@dataclass
class GateItem:
    number: int
    title: str
    asks_count: int = 0
    most_recent_gate_comment_at: datetime | None = None


@dataclass
class GateReport:
    sections: dict[str, list[GateItem]] = field(default_factory=dict)


class GhError(RuntimeError):
    pass


def _default_runner(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise GhError("gh CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise GhError(f"gh timed out: {' '.join(args)}") from e
    if proc.returncode != 0:
        raise GhError(
            f"gh exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def parse_asks_count(body: str) -> int:
    """Count `^**Ask N:` headers in a `[gate:ratify]` comment body."""
    return len(_ASKS_RE.findall(body))


def _list_issues(label: str, runner: GhRunner) -> list[dict]:
    out = runner(
        [
            "issue", "list",
            "--label", label,
            "--state", "open",
            "--json", "number,title,createdAt",
            "--limit", "200",
        ]
    )
    out = out.strip()
    if not out:
        return []
    data = json.loads(out)
    if not isinstance(data, list):
        raise GhError(f"unexpected gh issue list payload: {data!r}")
    return data


def _most_recent_gate_comment(num: int, kind: str, runner: GhRunner) -> dict | None:
    out = runner(["issue", "view", str(num), "--json", "comments"])
    data = json.loads(out)
    comments = data.get("comments") or []
    marker = f"[gate:{kind}]"
    matches = [c for c in comments if (c.get("body") or "").lstrip().startswith(marker)]
    if not matches:
        return None
    return max(matches, key=lambda c: c["createdAt"])


def _gather_ratify(runner: GhRunner) -> list[GateItem]:
    items: list[GateItem] = []
    for issue in _list_issues("gate:ratify", runner):
        comment = _most_recent_gate_comment(issue["number"], "ratify", runner)
        body = (comment.get("body") if comment else "") or ""
        items.append(
            GateItem(
                number=issue["number"],
                title=issue["title"],
                asks_count=parse_asks_count(body),
                most_recent_gate_comment_at=(
                    _parse_iso(comment["createdAt"]) if comment else None
                ),
            )
        )
    # Oldest gate-comment first; comments without one sort last (treat as
    # "no asks block yet" — the operator should see them after live ones).
    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    items.sort(key=lambda x: x.most_recent_gate_comment_at or sentinel)
    return items


def _gather_simple(label: str, runner: GhRunner) -> list[GateItem]:
    items = [
        GateItem(number=i["number"], title=i["title"])
        for i in _list_issues(label, runner)
    ]
    items.sort(key=lambda x: x.number)
    return items


def collect(runner: GhRunner | None = None) -> GateReport:
    r = runner or _default_runner
    sections: dict[str, list[GateItem]] = {}
    for label, header in SECTIONS:
        if header == "gate:ratify":
            sections[header] = _gather_ratify(r)
        else:
            sections[header] = _gather_simple(label, r)
    return GateReport(sections=sections)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _humanise_age(now: datetime, ts: datetime | None) -> str:
    if ts is None:
        return "no asks-block"
    delta = now - ts
    secs = int(delta.total_seconds())
    if secs < 0:
        return "just now"
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


def format_text(report: GateReport, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    lines: list[str] = []
    for _, header in SECTIONS:
        items = report.sections.get(header, [])
        if header == "gate:ratify":
            total_asks = sum(i.asks_count for i in items)
            lines.append(
                f"{header} ({len(items)} open, {total_asks} total asks):"
            )
            if not items:
                lines.append("  (none)")
            for it in items:
                lines.append(
                    f"  #{it.number}  "
                    f"{it.asks_count} ask{'s' if it.asks_count != 1 else ''}  "
                    f"oldest {_humanise_age(now, it.most_recent_gate_comment_at)}        "
                    f"{it.title}"
                )
        else:
            lines.append(f"{header} ({len(items)} open):")
            if not items:
                lines.append("  (none)")
            for it in items:
                lines.append(f"  #{it.number}  {it.title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_json(report: GateReport) -> str:
    payload: dict = {"sections": {}}
    for _, header in SECTIONS:
        items = report.sections.get(header, [])
        payload["sections"][header] = [
            {
                "number": i.number,
                "title": i.title,
                "asks_count": i.asks_count,
                "most_recent_gate_comment_at": (
                    i.most_recent_gate_comment_at.isoformat()
                    if i.most_recent_gate_comment_at
                    else None
                ),
            }
            for i in items
        ]
    return json.dumps(payload, indent=2) + "\n"
