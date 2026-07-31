"""#1274 injection-block ordering — movable-set and lock-displacement bound.

Sizes an ordering A/B *before* it is run, from the hook audit alone. Two
questions, both answerable without a reader model or a judge:

  1. **Movable set.** A block renders byte-identically under every policy
     unless there are at least two verbatim beliefs to permute. Beyond that
     the criterion is **per policy**, because the two relocating policies
     move different tiers:

       * `score_desc` re-sorts the non-locked hits among themselves and
         leaves the locked tier lane-leading, so it is the identity unless
         there are >=2 non-locked hits.
       * `locks_last` moves the whole locked tier to the end, so a single
         lock and a single non-lock already render differently
         (`[L1, n1]` -> `[n1, L1]`). Its criterion is that *both tiers are
         present*, which is a strictly larger set.

     Reporting only the `score_desc` threshold understates the live arm:
     `score_desc` is currently unreachable from the config knob (no call
     site supplies scores), so `locks_last` is the policy an A/B can
     actually run. Prompts outside the movable set cannot contribute signal
     at any sample size, so an A/B that ignores them reports a null it was
     never able to avoid.

  2. **Lock displacement.** Any policy that moves the user-locked tier is
     spending that tier's block position to buy one for the non-locked
     tier. This reports how much is being spent: how often position 1 is a
     lock today, and how many locks precede the first non-locked belief.

Reads the verbatim hook-audit JSONL, which records the rendered block per
turn. Restricted to `user_prompt_submit` rows (the real injection surface)
and, by default, to rows after the 2026-06-30 #1016-B reference-lock regime
break, which moved the non-locked share of `injection_events` from 2.1% to
50.3% — pooling across it compares two different products.

No live-store *content* is emitted: the parse looks only at the `<belief>`
tag attributes (id and lock tier), never at belief text.

Run: `python benchmarks/order_policy_movable_bound.py [AUDIT.jsonl ...]`
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# `id` is a hex hash and `lock` a fixed literal chosen by an equality test,
# so this matches the render in hook.py::_split_belief_lines exactly.
# Reference locks render to the manifest block, not as <belief> lines, and
# are therefore correctly outside the permutable tier.
BELIEF_RE = re.compile(r'<belief id="([0-9a-f]+)" lock="(user|none)"')

REGIME_BREAK = "2026-06-30"
AUDIT_FILENAME = "hook_audit.jsonl"


def default_audit_paths() -> list[Path]:
    """The audit log beside the current repo's store, plus its rotation."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return []
    base = Path(common) / "aelfrice"
    return [base / AUDIT_FILENAME, base / (AUDIT_FILENAME + ".1")]


def load_blocks(paths: list[Path], since: str) -> list[str]:
    """Rendered blocks from user_prompt_submit rows newer than `since`."""
    blocks: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("hook") != "user_prompt_submit":
                continue
            if str(row.get("ts", "")) <= since:
                continue
            block = row.get("rendered_block")
            if isinstance(block, str) and block:
                blocks.append(block)
    return blocks


def report(blocks: list[str]) -> dict[str, float]:
    n = len(blocks)
    if not n:
        print("no matching audit rows — nothing to bound")
        return {}

    movable_any = movable_score_desc = all_locks = first_is_lock = 0
    both_tiers = 0
    nonlock_shares: list[float] = []
    locks_before_first_nonlock: list[int] = []

    for block in blocks:
        tiers = [lock for _, lock in BELIEF_RE.findall(block)]
        if not tiers:
            continue
        verbatim = len(tiers)
        nonlock = sum(1 for t in tiers if t == "none")
        if verbatim >= 2:
            movable_any += 1
        if nonlock >= 2:
            movable_score_desc += 1
        if nonlock == 0:
            all_locks += 1
        if tiers[0] == "user":
            first_is_lock += 1
        if "user" in tiers and "none" in tiers:
            both_tiers += 1
            locks_before_first_nonlock.append(tiers.index("none"))
        nonlock_shares.append(nonlock / verbatim)

    def pct(k: int) -> str:
        return f"{k:5d}/{n}  {k / n:6.1%}"

    print(f"user_prompt_submit blocks analysed: {n}")
    print()
    print("MOVABLE SET")
    print(f"  >=2 verbatim beliefs (any policy):   {pct(movable_any)}")
    print(f"  >=2 non-locked (score_desc):         {pct(movable_score_desc)}")
    print(f"  both tiers present (locks_last):     {pct(both_tiers)}")
    print(f"  100% locks (no non-locked tier):     {pct(all_locks)}")
    print()
    print("LOCK DISPLACEMENT")
    print(f"  position 1 is a user lock:           {pct(first_is_lock)}")
    if locks_before_first_nonlock:
        ordered = sorted(locks_before_first_nonlock)
        median = ordered[len(ordered) // 2]
        print(f"  median locks before 1st non-lock:    {median:5d}")
    if nonlock_shares:
        ordered_shares = sorted(nonlock_shares)
        print(
            "  median non-locked share of block:    "
            f"{ordered_shares[len(ordered_shares) // 2]:6.1%}"
        )
    return {
        "n": n,
        "movable_any": movable_any / n,
        "movable_score_desc": movable_score_desc / n,
        "movable_locks_last": both_tiers / n,
        "first_is_lock": first_is_lock / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "audit",
        nargs="*",
        type=Path,
        help="hook-audit JSONL path(s); defaults to this repo's audit log",
    )
    ap.add_argument(
        "--since",
        default=REGIME_BREAK,
        help=(
            "ignore rows with ts <= this (default: the 2026-06-30 #1016-B "
            "regime break; pass an empty string to pool everything)"
        ),
    )
    args = ap.parse_args()
    paths = args.audit or default_audit_paths()
    if not paths:
        print("no audit path given and none discoverable")
        return 1
    report(load_blocks(paths, args.since))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
