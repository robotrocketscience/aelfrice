#!/usr/bin/env python3
"""#1551 — at how many user locks does the injection ceiling start trimming?

`HOOK_BLOCK_TOKEN_CEILING`'s docstring used to say it was "set well above
the sum of the per-lane budgets so it does not fire on a healthy store".
That was not measured and is not true: the first prompt of a session puts
the `<locked>` sub-block and the per-turn hits in one envelope, and 68
ordinary locks are enough. This script is how that number is produced, so
it can be re-derived rather than taken on trust when either constant moves.

It drives the real `user_prompt_submit` hook against a temporary store of
N identical user locks, walking N upwards until the ceiling first reports
a trim or an overrun, and prints the crossing point per lock length.

Usage:
    uv run python scripts/measure_block_ceiling.py
    uv run python scripts/measure_block_ceiling.py --lock-chars 150 200 700
    uv run python scripts/measure_block_ceiling.py --max-locks 400 --json
    uv run python scripts/measure_block_ceiling.py --dry-run

Exits non-zero if no crossing is found below `--max-locks`, which means
either the ceiling moved or the fixture stopped growing the block.
"""
from __future__ import annotations

import argparse
import io
import json
import os
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
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief  # noqa: E402
from aelfrice.store import MemoryStore  # noqa: E402

# Long enough to clear the #674 prompt-shape gate, so the fire takes the
# retrieval branch rather than the gate-skip one.
PROMPT = "tell me everything about the locked material please"


def _lock(index: int, chars: int) -> Belief:
    return Belief(
        id=f"L{index:031d}",
        content="lockword " + "q" * chars,
        content_hash=f"h{index}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at="2026-04-26T00:00:00Z",
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
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
        "--dry-run", action="store_true",
        help="print what would be swept and exit 0 without firing the hook",
    )
    args = ap.parse_args(argv)

    if args.dry_run:
        print(
            f"would sweep lock counts {args.step}..{args.max_locks} "
            f"step {args.step} for lock lengths {args.lock_chars}, against "
            f"a ceiling of {HOOK_BLOCK_TOKEN_CEILING} tokens"
        )
        return 0

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
