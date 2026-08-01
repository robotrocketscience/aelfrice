"""#1286 two-tier rendering (#1177 proposal 15) — injected-token saving bound.

Answers the parent proposal's own kill question offline, with no reader
model, no judge and no live store: **replay the rendered blocks the hook
already recorded, apply the proposed dispatch rule, and report the delta in
injected tokens.** The parent sets the bar at ~10% of block tokens; under it
the saving is dominated by the locked tier and the `<core>` section, which
the rule cannot touch, and the build is not worth its complexity.

The rule under test, verbatim from the parent spec::

    render_mode(b, rank_in_pack, cost) =
        "verbatim"  if b.lock_level == LOCK_USER
        "verbatim"  if rank_in_pack < K_verbatim   (default 5)
        "verbatim"  if cost <= SHORT_TOKENS        (default 60)
        "headline"  otherwise

Three properties this script is built around, because each is a way for the
measurement to come out wrong in a way nobody would notice:

* **It reuses the shipped functions rather than restating them.** The
  headline is `compression._headline` and the cost is
  `retrieval._estimate_tokens`, imported. A reimplementation would be
  measuring a rule that is not the one that would ship — and `_headline` in
  particular has three branches (single code fence, sentence boundary
  outside a fence, hard cut at whitespace) that a paraphrase gets wrong.
* **It counts the whole injected block, not just the belief lines.** The
  denominator is what the hook actually injected, framing header and
  reference manifest included. Dividing the saving by the belief lines alone
  would inflate the percentage against a denominator nobody pays.
* **It reports the per-block distribution, not only the pooled total.** A
  pooled ratio over blocks of very different sizes is dominated by the few
  largest, so a rule that helps three blocks enormously and 349 not at all
  would read as a win. Median and quartiles are what bind.

Restricted to `user_prompt_submit` rows — the real injection surface — and,
by default, to rows after the 2026-06-30 #1016-B regime break, which moved
the non-locked share of `injection_events` from 2.1% to 50.3%; pooling
across it compares two different products.

**No belief content is emitted.** The parse reads belief text in order to
compute its headline and its token cost, and prints only counts and token
aggregates. Nothing in the output carries store text.

Run: ``python benchmarks/two_tier_render_bound.py [AUDIT.jsonl ...]``
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

from aelfrice.compression import _headline
from aelfrice.retrieval import _estimate_tokens

# Matches the render in `hook._split_belief_lines` exactly: a hex id, a
# fixed `lock` literal chosen by an equality test, an optional
# `speculative` flag, then escaped content up to the closing tag. Reference
# locks render to the manifest instead of a `<belief>` line and so are
# correctly outside the tier this rule dispatches over.
BELIEF_RE = re.compile(
    r'<belief id="([0-9a-f]+)" lock="(user|none)"( speculative="1")?>'
    r"(.*?)</belief>",
    re.S,
)

REGIME_BREAK = "2026-06-30"
AUDIT_FILENAME = "hook_audit.jsonl"

# Parent-spec defaults.
K_VERBATIM = 5
SHORT_TOKENS = 60

# The rule adds one line to the framing header explaining the marker. It is
# a real cost and is charged per block, so a block whose saving is smaller
# than the explanation is correctly reported as a loss.
HEADER_NOTE = (
    'Entries marked truncated="1" show only their first sentence; '
    "run `aelf show <id>` for the full text."
)


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


def unescape(rendered: str) -> str:
    """Invert `hook._escape_for_hook_block`.

    That function is exactly ``content.replace("<", "&lt;").replace(">",
    "&gt;")`` — it does not escape `&`, so this inverse is exact unless the
    stored content itself contained the literal `&lt;`, which would come
    back as `<`. That case cannot be distinguished from the record alone and
    is left as-is; it changes a character count, never a rendering decision.
    """
    return rendered.replace("&lt;", "<").replace("&gt;", ">")


def escape(content: str) -> str:
    """`hook._escape_for_hook_block`, so the counterfactual line is counted
    in the same bytes the real renderer would emit."""
    return content.replace("<", "&lt;").replace(">", "&gt;")


def rewrite_block(
    block: str,
    *,
    k_verbatim: int = K_VERBATIM,
    short_tokens: int = SHORT_TOKENS,
) -> tuple[str, dict[str, int]]:
    """Return the block as the rule would have rendered it, plus counters.

    Belief lines are visited in render order, which is what `rank_in_pack`
    means: the rule is evaluated after the pack has decided membership *and*
    order, so rank is position in the block, not retrieval score.

    Note which reading of `rank_in_pack` this is, because it decides the
    direction of the error: rank counts *every* belief line, so a non-locked
    belief far down a lock-led block is headline-eligible. Ranking among the
    non-locked tier only would exempt its first `k_verbatim` members too and
    headline strictly fewer beliefs. This reading is therefore the one most
    favourable to the proposal, which is what makes the result a bound.
    """
    counts = {
        "beliefs": 0,
        "exempt_lock": 0,
        "exempt_rank": 0,
        "exempt_short": 0,
        "headlined": 0,
        "headline_no_gain": 0,
    }
    out: list[str] = []
    cursor = 0
    rank = 0
    for m in BELIEF_RE.finditer(block):
        out.append(block[cursor : m.start()])
        cursor = m.end()
        bid, lock, spec, rendered = m.groups()
        counts["beliefs"] += 1
        content = unescape(rendered)
        cost = _estimate_tokens(content)

        # Exemptions are checked in the spec's order and counted for the
        # first that fires, so the tallies partition the pack rather than
        # overlapping — which is what makes "the rule is inert because
        # everything is short" a legible finding.
        if lock == "user":
            counts["exempt_lock"] += 1
            keep = True
        elif rank < k_verbatim:
            counts["exempt_rank"] += 1
            keep = True
        elif cost <= short_tokens:
            counts["exempt_short"] += 1
            keep = True
        else:
            keep = False

        if keep:
            out.append(m.group(0))
        else:
            head = _headline(content)
            if len(head) >= len(content):
                # `_headline` never expands its source, but it does return
                # content unchanged for a single code fence — headlining
                # that belief costs the marker and saves nothing.
                counts["headline_no_gain"] += 1
                out.append(m.group(0))
            else:
                counts["headlined"] += 1
                spec_attr = spec or ""
                out.append(
                    f'<belief id="{bid}" lock="{lock}"{spec_attr}'
                    f' truncated="1">{escape(head)}</belief>'
                )
        rank += 1
    out.append(block[cursor:])
    return "".join(out), counts


def pooled_saving(
    blocks: list[str], *, k_verbatim: int, short_tokens: int
) -> float:
    """Pooled token saving for one parameter setting.

    Used for the sensitivity sweep, whose job is to separate two very
    different findings: "the defaults are badly chosen" and "the locked tier
    the rule may not touch is most of the block". Only the second is a
    property of the surface rather than of the spec.
    """
    before_total = after_total = 0
    for block in blocks:
        rewritten, counts = rewrite_block(
            block, k_verbatim=k_verbatim, short_tokens=short_tokens
        )
        if not counts["beliefs"]:
            continue
        note = _estimate_tokens(HEADER_NOTE) if counts["headlined"] else 0
        before_total += _estimate_tokens(block)
        after_total += _estimate_tokens(rewritten) + note
    if not before_total:
        return 0.0
    return 100.0 * (before_total - after_total) / before_total


def report(blocks: list[str]) -> dict[str, float]:
    n = len(blocks)
    if not n:
        print("no matching audit rows — nothing to bound")
        return {}

    before_total = after_total = 0
    per_block_pct: list[float] = []
    unchanged = 0
    regressed = 0
    totals = {
        "beliefs": 0,
        "exempt_lock": 0,
        "exempt_rank": 0,
        "exempt_short": 0,
        "headlined": 0,
        "headline_no_gain": 0,
    }
    parsed = 0

    for block in blocks:
        rewritten, counts = rewrite_block(block)
        if not counts["beliefs"]:
            continue
        parsed += 1
        for k, v in counts.items():
            totals[k] += v
        before = _estimate_tokens(block)
        # The header note is charged only to blocks the rule actually
        # changes; an unchanged block would not carry the marker.
        note = _estimate_tokens(HEADER_NOTE) if counts["headlined"] else 0
        after = _estimate_tokens(rewritten) + note
        before_total += before
        after_total += after
        if counts["headlined"] == 0:
            unchanged += 1
        if after > before:
            regressed += 1
        per_block_pct.append(100.0 * (before - after) / before if before else 0.0)

    if not parsed:
        print("no block carried a parseable <belief> line — nothing to bound")
        return {}

    pooled = 100.0 * (before_total - after_total) / before_total
    per_block_pct.sort()
    median = statistics.median(per_block_pct)
    q1 = per_block_pct[len(per_block_pct) // 4]
    q3 = per_block_pct[(3 * len(per_block_pct)) // 4]

    print(f"blocks considered:            {n}")
    print(f"blocks with >=1 belief line:  {parsed}")
    print(f"belief lines total:           {totals['beliefs']}")
    print()
    print("dispatch outcome (partitioned, first exemption wins):")
    for key, label in (
        ("exempt_lock", "verbatim — user lock"),
        ("exempt_rank", f"verbatim — rank < {K_VERBATIM}"),
        ("exempt_short", f"verbatim — cost <= {SHORT_TOKENS}"),
        ("headline_no_gain", "headline declined — no shorter"),
        ("headlined", "HEADLINED"),
    ):
        v = totals[key]
        share = 100.0 * v / totals["beliefs"] if totals["beliefs"] else 0.0
        print(f"  {label:<34} {v:>6}  ({share:5.1f}%)")
    print()
    print(f"injected tokens today:        {before_total}")
    print(f"injected tokens under rule:   {after_total}")
    print(f"pooled saving:                {pooled:.2f}%")
    print(f"per-block saving  median:     {median:.2f}%")
    print(f"                  q1 / q3:    {q1:.2f}% / {q3:.2f}%")
    print(f"                  max:        {per_block_pct[-1]:.2f}%")
    print(f"blocks the rule does not change: {unchanged} "
          f"({100.0 * unchanged / parsed:.1f}%)")
    print(f"blocks made larger by the rule:  {regressed}")
    print()
    # Where the block's tokens actually are. The parent predicted the
    # saving would be "dominated by the lock and <core> redundancy"; this
    # is the direct test of that claim rather than an inference from the
    # sweep, and it is the number that generalises past this rule.
    lock_tok = nonlock_tok = 0
    for block in blocks:
        for _bid, lock, _spec, rendered in BELIEF_RE.findall(block):
            tok = _estimate_tokens(unescape(rendered))
            if lock == "user":
                lock_tok += tok
            else:
                nonlock_tok += tok
    other = before_total - lock_tok - nonlock_tok
    print("block composition (tokens):")
    for v, label in (
        (lock_tok, "user-locked belief lines (exempt)"),
        (nonlock_tok, "non-locked belief lines (eligible)"),
        (other, "framing, <core>, manifest, markup"),
    ):
        print(f"  {label:<38} {v:>9}  ({100.0 * v / before_total:5.1f}%)")
    eligible_pct = 100.0 * nonlock_tok / before_total
    # The bound that does not depend on the rule at all: even a renderer
    # that compressed every eligible belief to zero tokens could not save
    # more than the eligible share. Stated separately because it survives
    # any change to the dispatch, the headline function, or the constants.
    print(
        f"  -> absolute ceiling for ANY lock-exempt render rule: "
        f"{eligible_pct:.1f}%"
    )
    print()

    # Is the answer the constants or the surface? The last row headlines
    # every non-locked belief regardless of rank or length — the most
    # aggressive form of this rule that still honours the one exemption the
    # proposal calls non-negotiable. If even that misses the bar, no choice
    # of `K_verbatim` / `SHORT_TOKENS` reaches it.
    print("sensitivity — pooled saving by parameter setting:")
    for kv, st, label in (
        (K_VERBATIM, SHORT_TOKENS, "spec defaults (5 / 60)"),
        (5, 30, "shorter short-cut  (5 / 30)"),
        (2, 30, "shallower verbatim (2 / 30)"),
        (0, 0, "CEILING: headline every non-lock"),
    ):
        print(
            f"  {label:<34} "
            f"{pooled_saving(blocks, k_verbatim=kv, short_tokens=st):6.2f}%"
        )
    print()
    bar = 10.0
    verdict = "CLEARS" if pooled >= bar else "DOES NOT CLEAR"
    print(f"parent bar: pooled saving >= {bar:.0f}%  ->  {verdict}")
    return {
        "blocks": float(parsed),
        "pooled_saving_pct": pooled,
        "median_saving_pct": median,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audit", nargs="*", type=Path, help="hook_audit.jsonl paths")
    ap.add_argument(
        "--since",
        default=REGIME_BREAK,
        help=(
            "only rows with ts > this (default: the 2026-06-30 #1016-B "
            "regime break; pass an empty string to pool across it)"
        ),
    )
    args = ap.parse_args()
    paths = args.audit or default_audit_paths()
    report(load_blocks(paths, args.since))


if __name__ == "__main__":
    main()
