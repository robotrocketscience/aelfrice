#!/usr/bin/env python3
"""#1551 — at how many user locks does the injection ceiling start trimming?

`HOOK_BLOCK_TOKEN_CEILING`'s docstring used to say it was "set well above
the sum of the per-lane budgets so it does not fire on a healthy store".
That was not measured and is not true: the first prompt of a session puts
the `<locked>` sub-block and the per-turn hits in one envelope, and 66
ordinary locks of 150 characters are enough, or 58 of 200. This script is
how those numbers are produced, so they can be re-derived rather than taken
on trust when either constant moves. This docstring published 68 while the
script's own default run printed 66, which is what the constant's docstring
and the CHANGELOG entry both said; `--emit-figures` lets CI re-run the sweep
so the pair cannot drift again.
<!-- derived: scripts/measure_block_ceiling.py#first_trim_locks_150 = 66 -->
<!-- derived: scripts/measure_block_ceiling.py#first_trim_locks_200 = 58 -->

It drives the real `user_prompt_submit` hook against a temporary store of
N identical user locks, walking N upwards until the ceiling first reports
a trim or an overrun, and prints the crossing point per lock length.

`--lanes` answers the second question: **which lane does the ceiling shed
first?** It fires one mixed store — user locks, `<core>` beliefs whose
content does not mention the prompt, and retrieval hits that do — with the
ceiling off and with it on, and prints how many of each survived. That is
the pair `_ceiling_drop_order` quotes, and the reason the drop order is by
lane rather than by position in the body.

`--gate-skip` answers the third: **how big was the branch that had no
bound?** It fires the `elif gate_skip:` path — the one a first prompt under
12 characters reaches — on a 300-lock store with the ceiling disabled, and
prints the untrimmed size. That is the 17,201 the CHANGELOG entry, both
`hook.py` sites and the wiring test publish. `--lock-chars` and the
`_lock` fixture both name the *padding*: `chars=150` is a 159-character
lock, and reading it as the content length is worth 675 tokens here
(16,526 for a lock of exactly 150 characters).
<!-- derived: scripts/measure_block_ceiling.py#gate_skip_tokens_300_locks_150 = 17201 -->

Usage:
    uv run python scripts/measure_block_ceiling.py
    uv run python scripts/measure_block_ceiling.py --lock-chars 150 200 700
    uv run python scripts/measure_block_ceiling.py --max-locks 400 --json
    uv run python scripts/measure_block_ceiling.py --lanes
    uv run python scripts/measure_block_ceiling.py --gate-skip
    uv run python scripts/measure_block_ceiling.py --dry-run
    uv run python scripts/measure_block_ceiling.py --emit-figures

`--emit-figures` is the protocol `scripts/check_derived_figures.py` speaks: a
JSON object of key -> value on stdout and nothing else, so CI re-runs this
sweep and hard-fails when a published crossing no longer matches it.

Exits non-zero if no crossing is found below `--max-locks`, which means
either the ceiling moved or the fixture stopped growing the block. Under
`--lanes` it exits non-zero if the ceiling dropped nothing, which would
make the comparison vacuous.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aelfrice.hook import (  # noqa: E402
    HOOK_BLOCK_TOKEN_CEILING,
    _audit_tokens_from_block,
    user_prompt_submit,
)
from aelfrice.models import (  # noqa: E402
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore  # noqa: E402

# Long enough to clear the #674 prompt-shape gate, so the fire takes the
# retrieval branch rather than the gate-skip one.
PROMPT = "tell me everything about the locked material please"

# Under `hook._MIN_PROMPT_LEN` (12), so `_should_skip_bm25` refuses BM25 and
# the fire takes the `elif gate_skip:` emit path instead of the retrieval one.
GATED_PROMPT = "ok"

# The term the `--lanes` prompt is built around. Only the hit lane carries
# it, so a surviving-element count separates prompt-matched content from
# prompt-independent content by construction.
LANE_WORD = "banana"
LANE_PROMPT = f"tell me everything about the {LANE_WORD} please"

_ELEMENT_ID_RE = re.compile(r'<belief id="([^"]+)"')


def _belief(
    bid: str, content: str, *, locked: bool = False, alpha: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-26T00:00:00Z" if locked else None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _lock(index: int, chars: int) -> Belief:
    return _belief(
        f"L{index:031d}", "lockword " + "q" * chars, locked=True,
    )


def fire(n_locks: int, chars: int) -> tuple[int, str]:
    """Run one hook fire against a store of `n_locks` locks of `chars`."""
    work = Path(tempfile.mkdtemp(prefix="aelf-ceiling-"))
    db = work / "memory.db"
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            store.insert_belief(_lock(i, chars))
    finally:
        store.close()
    os.environ["AELFRICE_DB"] = str(db)
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": f"measure-{chars}-{n_locks}",
            "transcript_path": "/dev/null",
            "cwd": str(work),
            "hook_event_name": "UserPromptSubmit",
            "prompt": PROMPT,
        }
    )
    rc = user_prompt_submit(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    if rc != 0:
        raise SystemExit(f"hook returned {rc}")
    return _audit_tokens_from_block(sout.getvalue()), serr.getvalue()


def gate_skip_tokens(n_locks: int, chars: int) -> int:
    """Size of the untrimmed gate-skip block, in estimated tokens.

    The figure `hook._write_memory_block`, the `elif gate_skip:` branch, the
    CHANGELOG entry and `test_hook_injection_ceiling_wiring.py` all publish
    for "the branch that was unbounded". The ceiling is disabled for the
    fire, because the shipped code now bounds this branch and the published
    number is what it emits without that bound — the locks are exempt from
    the drop, so on a lock-only store the two differ only in the stderr note.

    `chars` is the padding, not the content length: `_lock` prepends
    `"lockword "`, so `chars=150` is a 159-character lock. The distinction is
    worth 675 tokens at 300 locks (17,201 against 16,526), which is the size
    of the confusion this key exists to prevent.
    """
    work = Path(tempfile.mkdtemp(prefix="aelf-gateskip-"))
    db = work / "memory.db"
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            store.insert_belief(_lock(i, chars))
    finally:
        store.close()
    os.environ["AELFRICE_DB"] = str(db)
    previous = os.environ.get("AELFRICE_HOOK_BLOCK_CEILING")
    os.environ["AELFRICE_HOOK_BLOCK_CEILING"] = "0"
    try:
        sout, serr = io.StringIO(), io.StringIO()
        payload = json.dumps(
            {
                "session_id": f"gateskip-{chars}-{n_locks}",
                "transcript_path": "/dev/null",
                "cwd": str(work),
                "hook_event_name": "UserPromptSubmit",
                "prompt": GATED_PROMPT,
            }
        )
        rc = user_prompt_submit(
            stdin=io.StringIO(payload), stdout=sout, stderr=serr
        )
        if rc != 0:
            raise SystemExit(f"hook returned {rc}")
        return _audit_tokens_from_block(sout.getvalue())
    finally:
        # Restore, so a later crossing sweep in the same process is not
        # measured against a disabled ceiling.
        if previous is None:
            os.environ.pop("AELFRICE_HOOK_BLOCK_CEILING", None)
        else:
            os.environ["AELFRICE_HOOK_BLOCK_CEILING"] = previous


def fire_lanes(ceiling: int | None) -> dict[str, object]:
    """One mixed-store fire; return how much of each lane survived.

    The store holds `n_locks` user locks, `n_core` corroborated beliefs
    that do NOT carry `LANE_WORD`, and `n_hits` unlocked beliefs that do.
    So the `<core>` section is prompt-independent and the per-turn hits
    are exactly the prompt-matched lane, and counting survivors of each in
    the emitted block says which one the ceiling shed.

    `ceiling=None` uses the shipped default; an integer is exported as
    `AELFRICE_HOOK_BLOCK_CEILING`, where `0` disables the trim entirely
    and gives the untrimmed control arm.
    """
    n_locks, n_core, n_hits = 50, 20, 20
    work = Path(tempfile.mkdtemp(prefix="aelf-lanes-"))
    db = work / "memory.db"
    core_ids: list[str] = []
    hit_ids: list[str] = []
    store = MemoryStore(str(db))
    try:
        for i in range(n_locks):
            store.insert_belief(_lock(i, 150))
        for i in range(n_core):
            bid = f"C{i:031d}"
            store.insert_belief(
                _belief(
                    bid, "coreword unrelated material " + "w" * 200, alpha=4.0,
                )
            )
            core_ids.append(bid)
        for i in range(n_hits):
            bid = f"H{i:031d}"
            store.insert_belief(_belief(bid, f"{LANE_WORD} fact " + "z" * 400))
            hit_ids.append(bid)
    finally:
        store.close()
    os.environ["AELFRICE_DB"] = str(db)
    if ceiling is None:
        os.environ.pop("AELFRICE_HOOK_BLOCK_CEILING", None)
    else:
        os.environ["AELFRICE_HOOK_BLOCK_CEILING"] = str(ceiling)
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps(
        {
            "session_id": f"lanes-{ceiling}",
            "transcript_path": "/dev/null",
            "cwd": str(work),
            "hook_event_name": "UserPromptSubmit",
            "prompt": LANE_PROMPT,
        }
    )
    rc = user_prompt_submit(stdin=io.StringIO(payload), stdout=sout, stderr=serr)
    if rc != 0:
        raise SystemExit(f"hook returned {rc}")
    out, err = sout.getvalue(), serr.getvalue()
    rendered = set(_ELEMENT_ID_RE.findall(out))
    dropped = re.search(r"dropped (\d+) belief element", err)
    return {
        "ceiling": ceiling if ceiling is not None else HOOK_BLOCK_TOKEN_CEILING,
        "tokens": _audit_tokens_from_block(out),
        "hits": sum(1 for b in hit_ids if b in rendered),
        "n_hits": n_hits,
        "core": sum(1 for b in core_ids if b in rendered),
        "n_core": n_core,
        "dropped": int(dropped.group(1)) if dropped else 0,
    }


def crossing(chars: int, max_locks: int, step: int) -> dict[str, object]:
    """First lock count at which the ceiling reports a trim or an overrun."""
    for n in range(step, max_locks + 1, step):
        tokens, err = fire(n, chars)
        if "ceiling" in err:
            return {"lock_chars": chars, "locks": n, "tokens": tokens}
    return {"lock_chars": chars, "locks": None, "tokens": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lock-chars", type=int, nargs="+", default=[150, 200],
        help="lock content lengths to sweep (default: 150 200)",
    )
    ap.add_argument("--max-locks", type=int, default=300)
    ap.add_argument(
        "--step", type=int, default=1,
        help="lock-count increment; 1 gives the exact crossing",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--lanes", action="store_true",
        help="report which lane the ceiling sheds, trimmed vs untrimmed",
    )
    ap.add_argument(
        "--gate-skip", action="store_true",
        help="print the untrimmed gate-skip block size at 300 locks",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print what would be swept and exit 0 without firing the hook",
    )
    ap.add_argument(
        "--emit-figures", action="store_true",
        help="print the published crossings as JSON for "
             "scripts/check_derived_figures.py",
    )
    args = ap.parse_args(argv)

    if args.emit_figures:
        # The two lengths the docstring, `HOOK_BLOCK_TOKEN_CEILING` and the
        # CHANGELOG entry all publish. Fixed here rather than read off
        # `--lock-chars`, because the key names are what the markers cite:
        # a sweep the caller re-pointed would emit keys nothing published.
        figures: dict[str, object] = {
            f"first_trim_locks_{chars}": crossing(
                chars, args.max_locks, args.step,
            )["locks"]
            for chars in (150, 200)
        }
        figures["gate_skip_tokens_300_locks_150"] = gate_skip_tokens(300, 150)
        print(json.dumps(figures))
        return 0

    if args.dry_run:
        if args.gate_skip:
            print(
                "would fire one 300-lock store of 159-character locks at the "
                "gate-skip branch with the ceiling disabled"
            )
            return 0
        if args.lanes:
            print(
                "would fire one 50-lock / 20-core / 20-hit store twice, with "
                f"the trim off and at {HOOK_BLOCK_TOKEN_CEILING} tokens"
            )
            return 0
        print(
            f"would sweep lock counts {args.step}..{args.max_locks} "
            f"step {args.step} for lock lengths {args.lock_chars}, against "
            f"a ceiling of {HOOK_BLOCK_TOKEN_CEILING} tokens"
        )
        return 0

    if args.gate_skip:
        tokens = gate_skip_tokens(300, 150)
        if args.json:
            print(json.dumps({"gate_skip_tokens_300_locks_150": tokens}))
        else:
            print(
                f"gate-skip branch, untrimmed: {tokens} estimated tokens from "
                f"300 locks of 159 characters (\"lockword \" + 150 padding), "
                f"against a {HOOK_BLOCK_TOKEN_CEILING}-token ceiling"
            )
        return 0

    if args.lanes:
        rows = [fire_lanes(0), fire_lanes(None)]
        if args.json:
            print(json.dumps({"lanes": rows}, indent=2))
        else:
            for row in rows:
                label = "off" if row["ceiling"] == 0 else str(row["ceiling"])
                print(
                    f"  ceiling {label:>5}: {row['hits']}/{row['n_hits']} hits, "
                    f"{row['core']}/{row['n_core']} core, "
                    f"{row['tokens']} tokens, dropped {row['dropped']}"
                )
        # A run in which nothing was dropped compares two identical arms.
        return 0 if rows[1]["dropped"] else 1

    results = [
        crossing(chars, args.max_locks, args.step) for chars in args.lock_chars
    ]
    if args.json:
        print(json.dumps(
            {"ceiling_tokens": HOOK_BLOCK_TOKEN_CEILING, "crossings": results},
            indent=2,
        ))
    else:
        print(f"ceiling: {HOOK_BLOCK_TOKEN_CEILING} estimated tokens")
        for row in results:
            if row["locks"] is None:
                print(
                    f"  {row['lock_chars']:>4}-char locks: no crossing below "
                    f"{args.max_locks} locks"
                )
            else:
                print(
                    f"  {row['lock_chars']:>4}-char locks: trims from "
                    f"{row['locks']} locks ({row['tokens']} tokens emitted)"
                )
    return 0 if all(r["locks"] is not None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
