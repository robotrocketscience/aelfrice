#!/usr/bin/env python3
"""
Dry-run replay of the auto-unblock-on-blocker-close workflow logic.

Given an issue number and a simulated-closed issue number, this script:
  1. Fetches the issue body from GitHub (or uses --body to supply one).
  2. Parses all blocker references via the same regex the workflow uses.
  3. Checks the real GitHub state of each referenced blocker.
  4. Prints the parsed blocker list and the would-be flip decision.
  5. Does NOT mutate any issue, label, or project-board status.

Usage
-----
  uv run python scripts/replay-auto-unblock.py <issue> --simulate-closed <closed>
  uv run python scripts/replay-auto-unblock.py 154 --simulate-closed 437
  uv run python scripts/replay-auto-unblock.py 154 --simulate-closed 437 \\
      --body "Blocked-by: #307\nDepends on: #437"

Options
-------
  --simulate-closed N   Treat issue N as closed for this replay (overrides
                        its live GitHub state with CLOSED).
  --body TEXT           Use TEXT as the issue body instead of fetching it.
                        Supports \\n escape sequences.
  --repo OWNER/REPO     GitHub repository (default: robotrocketscience/aelfrice).
  --status-option-blocked  optionId for Blocked status (default: 3bc23bae).

False-positive regression
-------------------------
  Run with --body to verify that e.g. #1542 does NOT match when #154 closes:
    uv run python scripts/replay-auto-unblock.py 154 --simulate-closed 154 \\
        --body "Blocked-by: #1542"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Optional


REPO_DEFAULT = "robotrocketscience/aelfrice"

# Same pattern set as the workflow (case-insensitive, word-boundary on issue number).
BLOCKER_PATTERN = re.compile(
    r"(?:blocked[-\s]by:?\s*|blocked\s+by:?\s*|depends\s+on:?\s*|gate:?\s*)#(\d+)\b",
    re.IGNORECASE,
)


def gh_json(args: list[str]) -> dict | list:
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def fetch_issue_body(number: int, repo: str) -> str:
    data = gh_json(["issue", "view", str(number), "--repo", repo, "--json", "body"])
    return data.get("body") or ""


def fetch_issue_state(number: int, repo: str) -> tuple[str, Optional[str]]:
    """Return (state, closedAt) for the given issue number."""
    try:
        data = gh_json(
            ["issue", "view", str(number), "--repo", repo, "--json", "state,closedAt"]
        )
        return data.get("state", "UNKNOWN"), data.get("closedAt")
    except RuntimeError:
        return "UNKNOWN", None


def parse_blockers(body: str) -> list[int]:
    """Return sorted list of unique blocker issue numbers from the body."""
    return sorted({int(m) for m in BLOCKER_PATTERN.findall(body)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run replay of the auto-unblock workflow logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("issue", type=int, help="Issue number to check.")
    parser.add_argument(
        "--simulate-closed",
        dest="simulate_closed",
        type=int,
        required=True,
        help="Issue number to treat as CLOSED for this replay.",
    )
    parser.add_argument(
        "--body",
        default=None,
        help=(
            "Override body text (supports \\n escapes). "
            "If omitted, the live GitHub body is fetched."
        ),
    )
    parser.add_argument(
        "--repo",
        default=REPO_DEFAULT,
        help=f"GitHub repository (default: {REPO_DEFAULT}).",
    )
    parser.add_argument(
        "--status-option-blocked",
        default="3bc23bae",
        help="optionId representing Blocked on the project board (default: 3bc23bae).",
    )
    args = parser.parse_args()

    issue_num = args.issue
    closed_num = args.simulate_closed
    repo = args.repo

    print(f"=== auto-unblock replay ===")
    print(f"  Target issue:    #{issue_num}")
    print(f"  Simulated close: #{closed_num}")
    print(f"  Repository:      {repo}")
    print()

    # ── 1. Obtain body ─────────────────────────────────────────────────────
    if args.body is not None:
        body = args.body.replace("\\n", "\n")
        print(f"[body source] provided via --body flag")
    else:
        print(f"[body source] fetching #{issue_num} from GitHub...")
        body = fetch_issue_body(issue_num, repo)

    print()
    print("── Body snippet (first 300 chars) ──────────────────────────────")
    snippet = body[:300].replace("\n", "\\n")
    print(f"  {snippet}")
    print()

    # ── 2. Check if body references the simulated-closed issue as a blocker
    trigger_pattern = re.compile(
        rf"(?:blocked[-\s]by:?\s*|blocked\s+by:?\s*|depends\s+on:?\s*|gate:?\s*)"
        rf"#{re.escape(str(closed_num))}\b",
        re.IGNORECASE,
    )
    triggers = trigger_pattern.findall(body)
    print(f"── Trigger check: does body reference #{closed_num} as a blocker? ──")
    if triggers:
        print(f"  YES — matched pattern(s): {triggers}")
    else:
        print(f"  NO  — no formal blocker reference to #{closed_num} found in body.")
        print()
        print(
            "  The workflow would NOT process this issue when #{} closes.".format(
                closed_num
            )
        )
        print(
            "  To test the logic, supply a body with formal syntax via --body, e.g.:"
        )
        print(
            f"    --body 'Blocked-by: #{closed_num}'"
        )
        sys.exit(0)

    print()

    # ── 3. Parse ALL blocker refs ─────────────────────────────────────────
    all_blockers = parse_blockers(body)
    print(f"── All blocker references in body ──────────────────────────────")
    if all_blockers:
        for b in all_blockers:
            print(f"  #{b}")
    else:
        print("  (none)")
    print()

    # ── 4. Resolve blocker states (override simulated-closed) ─────────────
    print(f"── Blocker states (#{closed_num} simulated as CLOSED) ──────────")
    still_open: list[int] = []
    all_closed_details: list[tuple[int, str]] = []

    for b in all_blockers:
        if b == closed_num:
            state = "CLOSED"
            closed_at = "(simulated)"
        else:
            print(f"  Fetching state of #{b}...", end=" ", flush=True)
            state, closed_at = fetch_issue_state(b, repo)
            print(f"{state}")

        if state != "CLOSED":
            still_open.append(b)
        else:
            all_closed_details.append((b, closed_at or "unknown"))

    print()
    for b, ca in all_closed_details:
        print(f"  #{b}: CLOSED  (closedAt: {ca})")
    for b in still_open:
        print(f"  #{b}: OPEN (or UNKNOWN)")

    print()

    # ── 5. Decision ───────────────────────────────────────────────────────
    print("── Decision ────────────────────────────────────────────────────")
    if still_open:
        print(f"  WOULD NOT FLIP — {len(still_open)} blocker(s) still open: "
              f"{', '.join('#' + str(b) for b in still_open)}")
        print(f"  Workflow action: post a 'still blocked by ...' comment on #{issue_num}.")
    else:
        print(f"  WOULD FLIP #{issue_num}: Blocked → Todo")
        print(f"    (all {len(all_blocked := all_closed_details)} blocker(s) are CLOSED)")
        print(f"  Workflow action: flip project-board status + post confirmation comment.")

    print()

    # ── 6. False-positive guard demo ─────────────────────────────────────
    # Show that #1542 does NOT match when #154 closes.
    wider_num = int(str(closed_num) + "2")  # e.g. 154 → 1542
    fp_body = f"Blocked-by: #{wider_num}"
    fp_pattern = re.compile(
        rf"(?:blocked[-\s]by:?\s*|blocked\s+by:?\s*|depends\s+on:?\s*|gate:?\s*)"
        rf"#{re.escape(str(closed_num))}\b",
        re.IGNORECASE,
    )
    fp_match = bool(fp_pattern.search(fp_body))
    print("── False-positive guard ────────────────────────────────────────")
    print(f"  Test body: '{fp_body}'")
    print(f"  Matches #{closed_num}? {'YES (BUG!)' if fp_match else 'NO  (correct — \\b prevents #' + str(wider_num) + ' matching #' + str(closed_num) + ')'}")
    print()


if __name__ == "__main__":
    main()
