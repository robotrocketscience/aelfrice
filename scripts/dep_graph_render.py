#!/usr/bin/env python3
"""Render a Mermaid dependency graph from open-issue Blocked-by links (#581).

Pure-stdlib script (json, re, subprocess, argparse) so it can run on a
GitHub Actions ubuntu-latest runner with only `gh` CLI installed.

Pipeline:
  1. Query GitHub GraphQL for all open issues matching a label filter
     (default: ``v2.1``) plus their ``trackedIssues`` and
     ``trackedInIssues`` connections.
  2. Walk the response and emit a Mermaid ``graph TD`` block:
       - one node per issue (``#NUM[short-title (claim)]``)
       - one ``-->`` edge per Blocks/Blocked-by link
       - colour: red = has open blockers, green = leaf (work-ready),
         yellow = in-progress (assignee or author-* label).
  3. Update (or create) a sticky comment on a target tracker issue,
     identified by the magic header ``<!-- dep-graph-auto -->``.

Idempotency: re-running on unchanged data overwrites the comment in-place
(same body → noop edit; gh skips when the body matches).

Designed for offline test fixtures: the GraphQL fetch is isolated in
``fetch_graph_data()`` so callers can pass a pre-built dict to
``render_mermaid()`` directly.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any

STICKY_HEADER = "<!-- dep-graph-auto -->"
DEFAULT_LABEL = "v2.1"
DEFAULT_TRACKER_ISSUE = 474

# GraphQL query — paginated by 50 issues per page.
_QUERY = """
query($owner: String!, $repo: String!, $label: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issues(
      first: 50
      after: $cursor
      states: OPEN
      labels: [$label]
      orderBy: {field: CREATED_AT, direction: ASC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        labels(first: 30) { nodes { name } }
        assignees(first: 5) { nodes { login } }
        trackedIssues(first: 30) { nodes { number state } }
        trackedInIssues(first: 30) { nodes { number state } }
      }
    }
  }
}
"""


def fetch_graph_data(owner: str, repo: str, label: str) -> list[dict[str, Any]]:
    """Page the GraphQL query and return the flat issue list."""
    issues: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        args = [
            "gh", "api", "graphql",
            "-f", f"query={_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"repo={repo}",
            "-F", f"label={label}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        result = subprocess.run(
            args, check=True, capture_output=True, text=True,
        )
        page = json.loads(result.stdout)["data"]["repository"]["issues"]
        issues.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return issues


def _classify(node: dict[str, Any], open_numbers: set[int]) -> str:
    """Return the Mermaid class name for an issue node.

    - ``inProgress``: any assignee, or any label starting with ``author-``.
    - ``hasBlockers``: any ``trackedInIssues`` whose number is still open.
    - ``leaf``: otherwise.
    """
    labels = {n["name"] for n in node["labels"]["nodes"]}
    if node["assignees"]["nodes"] or any(name.startswith("author-") for name in labels):
        return "inProgress"
    blockers = [
        b for b in node["trackedInIssues"]["nodes"]
        if b["number"] in open_numbers
    ]
    if blockers:
        return "hasBlockers"
    return "leaf"


def _short_title(title: str, limit: int = 50) -> str:
    """Truncate to a Mermaid-safe single-line label, square-bracket-escaped."""
    cleaned = re.sub(r"[\[\]\"`]", "", title)
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _claim_label(node: dict[str, Any]) -> str:
    """Return the lowercased claim suffix (e.g. ``setr``) or empty string."""
    for n in node["labels"]["nodes"]:
        name = n["name"]
        if name.startswith("author-"):
            return name[len("author-") :].lower()
    return ""


def render_mermaid(issues: list[dict[str, Any]]) -> str:
    """Build the Mermaid graph block.

    Determinism: nodes sorted ascending by number; edges sorted by
    (src, dst). Two runs with the same input data produce byte-identical
    output, which is required for the sticky-comment idempotency check.
    """
    issues_sorted = sorted(issues, key=lambda n: n["number"])
    open_numbers = {n["number"] for n in issues_sorted}

    lines: list[str] = ["```mermaid", "graph TD"]

    # Nodes.
    for node in issues_sorted:
        num = node["number"]
        title = _short_title(node["title"])
        claim = _claim_label(node)
        suffix = f" ({claim})" if claim else ""
        lines.append(f'  N{num}["#{num} {title}{suffix}"]')

    # Edges. trackedIssues = "this issue blocks X"; emit X --> this.
    edges: set[tuple[int, int]] = set()
    for node in issues_sorted:
        num = node["number"]
        for blocked in node["trackedIssues"]["nodes"]:
            if blocked["number"] in open_numbers:
                edges.add((num, blocked["number"]))
        for blocker in node["trackedInIssues"]["nodes"]:
            if blocker["number"] in open_numbers:
                edges.add((blocker["number"], num))
    for src, dst in sorted(edges):
        lines.append(f"  N{src} --> N{dst}")

    # Class assignments.
    classes: dict[str, list[int]] = {"hasBlockers": [], "leaf": [], "inProgress": []}
    for node in issues_sorted:
        cls = _classify(node, open_numbers)
        classes[cls].append(node["number"])

    lines.append("  classDef hasBlockers fill:#fdd,stroke:#c33")
    lines.append("  classDef leaf fill:#dfd,stroke:#393")
    lines.append("  classDef inProgress fill:#ffe9a8,stroke:#b80")
    for cls_name, nums in classes.items():
        if nums:
            lines.append(f"  class {','.join(f'N{n}' for n in sorted(nums))} {cls_name}")

    lines.append("```")

    n_nodes = len(issues_sorted)
    n_edges = len(edges)
    return "\n".join(lines), n_nodes, n_edges


def render_comment_body(issues: list[dict[str, Any]], timestamp: str) -> str:
    """Wrap the Mermaid block in the sticky comment template."""
    block, n_nodes, n_edges = render_mermaid(issues)
    return (
        f"{STICKY_HEADER}\n"
        f"# Dependency graph (auto-generated)\n\n"
        f"Nodes: **{n_nodes}** · Edges: **{n_edges}** · "
        f"Last rendered: `{timestamp}`\n\n"
        f"Legend: 🟥 has open blockers · 🟩 leaf (work-ready) · 🟨 in-progress\n\n"
        f"{block}\n"
    )


def find_sticky_comment_id(owner: str, repo: str, issue_num: int) -> str | None:
    """Locate an existing sticky comment by header marker; return its node ID or None."""
    args = [
        "gh", "api",
        f"repos/{owner}/{repo}/issues/{issue_num}/comments",
        "--paginate",
        "--jq", f'[.[] | select(.body | startswith("{STICKY_HEADER}"))][0].id // empty',
    ]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    cid = result.stdout.strip()
    return cid or None


def post_or_update_comment(
    owner: str,
    repo: str,
    issue_num: int,
    body: str,
) -> str:
    """Create or update the sticky comment. Returns the action: ``created`` or ``updated``."""
    sticky_id = find_sticky_comment_id(owner, repo, issue_num)
    if sticky_id:
        subprocess.run(
            [
                "gh", "api", "-X", "PATCH",
                f"repos/{owner}/{repo}/issues/comments/{sticky_id}",
                "-f", f"body={body}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return "updated"
    subprocess.run(
        [
            "gh", "issue", "comment", str(issue_num),
            "--repo", f"{owner}/{repo}",
            "--body", body,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return "created"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", default="robotrocketscience")
    p.add_argument("--repo", default="aelfrice")
    p.add_argument("--label", default=DEFAULT_LABEL,
                   help=f"open-issue label filter (default: {DEFAULT_LABEL})")
    p.add_argument("--tracker-issue", type=int, default=DEFAULT_TRACKER_ISSUE,
                   help=f"issue number for the sticky comment (default: {DEFAULT_TRACKER_ISSUE})")
    p.add_argument("--timestamp", default=None,
                   help="override the rendered timestamp (default: current UTC)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the comment body to stdout instead of posting")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    issues = fetch_graph_data(args.owner, args.repo, args.label)

    if args.timestamp is None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ts = args.timestamp

    body = render_comment_body(issues, ts)

    if args.dry_run:
        print(body)
        return 0

    action = post_or_update_comment(args.owner, args.repo, args.tracker_issue, body)
    print(f"sticky comment {action} on #{args.tracker_issue} ({len(issues)} issues)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
