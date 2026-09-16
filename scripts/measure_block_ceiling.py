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

`--reference-tier` answers the fourth: **does demoting a long lock to the
bounded reference tier shrink the block?** It fires the four writes against
one 30,026-character lock, once per tier. Estimated tokens emitted, frozen
against reference:

* first prompt, gate-skip branch, 7700 frozen and 7700 reference;
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_gate_skip_first_frozen = 7700 -->
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_gate_skip_first_reference = 7700 -->
* first prompt, retrieval branch, 7796 frozen and 7796 reference, the
  reference arm carrying the full text *and* a `ref` pointer to it;
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_retrieval_first_frozen = 7796 -->
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_retrieval_first_reference = 7796 -->
* turn two, retrieval branch, 7683 frozen against 256 reference;
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_turn_two_frozen = 7683 -->
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_turn_two_reference = 256 -->
* `session_start`, 7660 frozen against 233 reference.
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_session_start_frozen = 7660 -->
  <!-- derived: scripts/measure_block_ceiling.py#ref_lock_30026_session_start_reference = 233 -->

So the answer is no on a session's first prompt and yes after it, which is
#1558 — the `<locked>` loop of `_build_session_start_subblock` has no
`is_reference_lock` branch — and that is why `_write_memory_block`'s
overrun note prescribes no remedy. An earlier revision of this table
published 7,784 / 244 / 221 on the three writes that carry a manifest line,
from a fixture nothing recorded. Those three are a function of the lock's
*content*, not only its length, because `lock_manifest_line` embeds
`_lock_topic` of it; the fixture is a module constant here for that reason.

`--exploration` answers the fifth: **does the #1279 slot change the block
the ceiling emits?** It fires one 60-lock / 20-core / 12-hit store twice,
with the slot on and off, and compares the bytes. At the shipped ceiling
they are identical at 5923 estimated tokens — the drawn belief is appended
to the tail of the pack and the per-turn lane is shed tail-first, so the
draw is the first element deleted, and 0 of the drawn ids are in the
emitted block — while `exploration_events`, written before the ceiling
runs, still records one draw and one displacement.
<!-- derived: scripts/measure_block_ceiling.py#exploration_slot_block_tokens = 5923 -->
<!-- derived: scripts/measure_block_ceiling.py#exploration_slot_drawn_emitted = 0 -->

Usage:
    uv run python scripts/measure_block_ceiling.py
    uv run python scripts/measure_block_ceiling.py --lock-chars 150 200 700
    uv run python scripts/measure_block_ceiling.py --max-locks 400 --json
    uv run python scripts/measure_block_ceiling.py --lanes
    uv run python scripts/measure_block_ceiling.py --gate-skip
    uv run python scripts/measure_block_ceiling.py --reference-tier
    uv run python scripts/measure_block_ceiling.py --exploration
    uv run python scripts/measure_block_ceiling.py --dry-run
    uv run python scripts/measure_block_ceiling.py --emit-figures

`--emit-figures` is the protocol `scripts/check_derived_figures.py` speaks: a
JSON object of key -> value on stdout and nothing else, so CI re-runs this
sweep and hard-fails when a published crossing no longer matches it.

Exits non-zero if no crossing is found below `--max-locks`, which means
either the ceiling moved or the fixture stopped growing the block. Under
`--lanes` it exits non-zero if the ceiling dropped nothing, which would
make the comparison vacuous; under `--reference-tier` if the two tiers
agree on every write, which would mean the tier is inert everywhere rather
than only on the first prompt; and under `--exploration` if the slot never
fired or the ceiling dropped nothing.
"""
from __future__ import annotations

import argparse
import hashlib
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
    session_start,
    user_prompt_submit,
)
from aelfrice.models import (  # noqa: E402
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
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


REF_LOCK_CHARS = 30_026
REF_LOCK_CONTENT = "lockword " + "q" * (REF_LOCK_CHARS - len("lockword "))
REF_LOCK_ID = "L" + "0" * 31
# A second prompt of the same session. Any prompt over `_MIN_PROMPT_LEN`
# does; it is spelled out so the turn-two arm is reproducible.
REF_TURN_TWO_PROMPT = "turn two: tell me more about the locked material"
REF_WRITES = (
    "gate_skip_first", "retrieval_first", "turn_two", "session_start",
)


def _reference_lock_store(tier: str) -> tuple[Path, Path]:
    """A store holding exactly one 30,026-character lock at `tier`."""
    work = Path(tempfile.mkdtemp(prefix="aelf-reflock-"))
    db = work / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            Belief(
                id=REF_LOCK_ID,
                content=REF_LOCK_CONTENT,
                content_hash="h_reflock",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_USER,
                lock_tier=tier,
                locked_at="2026-04-26T00:00:00Z",
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    finally:
        store.close()
    return work, db


def _fire_for_reference(
    work: Path, db: Path, *, prompt: str | None, session_id: str,
) -> str:
    """One hook fire against `db` with the ceiling disabled.

    `prompt=None` fires `session_start` instead of `user_prompt_submit`.
    The ceiling is off for the same reason `gate_skip_tokens` turns it
    off: what is being measured is what the render path produces, and a
    lock-only store is exempt from the drop anyway, so the two arms
    would differ only in the stderr note.
    """
    os.environ["AELFRICE_DB"] = str(db)
    previous = os.environ.get("AELFRICE_HOOK_BLOCK_CEILING")
    os.environ["AELFRICE_HOOK_BLOCK_CEILING"] = "0"
    try:
        sout, serr = io.StringIO(), io.StringIO()
        payload: dict[str, object] = {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(work),
        }
        if prompt is None:
            payload["hook_event_name"] = "SessionStart"
            rc = session_start(
                stdin=io.StringIO(json.dumps(payload)),
                stdout=sout,
                stderr=serr,
            )
        else:
            payload["hook_event_name"] = "UserPromptSubmit"
            payload["prompt"] = prompt
            rc = user_prompt_submit(
                stdin=io.StringIO(json.dumps(payload)),
                stdout=sout,
                stderr=serr,
            )
        if rc != 0:
            raise SystemExit(f"hook returned {rc}")
        return sout.getvalue()
    finally:
        if previous is None:
            os.environ.pop("AELFRICE_HOOK_BLOCK_CEILING", None)
        else:
            os.environ["AELFRICE_HOOK_BLOCK_CEILING"] = previous


def reference_tier_table() -> dict[str, dict[str, int]]:
    """Both tiers, with the vacuity guard the figures depend on.

    Equal columns would mean the reference tier is inert on every write,
    not that #1558 scopes it to the first prompt — and every figure below
    would still be publishable. The guard is here rather than in the CLI
    arm so `--emit-figures` inherits it.
    """
    rows = {
        tier: reference_tier(tier)
        for tier in (LOCK_TIER_FROZEN, LOCK_TIER_REFERENCE)
    }
    if all(
        rows[LOCK_TIER_FROZEN][w] == rows[LOCK_TIER_REFERENCE][w]
        for w in REF_WRITES
    ):
        raise SystemExit(
            "the two lock tiers agree on every write: the table would be "
            "vacuous"
        )
    return rows


def reference_tier(tier: str) -> dict[str, int]:
    """Estimated tokens of each write, for one lock demoted to `tier`.

    The #1558 render gap, measured: `_build_session_start_subblock`'s
    `<locked>` loop renders every lock verbatim with no
    `is_reference_lock` branch, unlike `_split_belief_lines` and
    `_core_belief_line`, which both divert a reference lock to
    `retrieval.lock_manifest_line`. So the bounded tier is honoured
    everywhere except the envelope that embeds the session-start
    sub-block — which is a session's first prompt, and a first prompt is
    when a lock-only store overruns. Running this for both tiers is what
    makes the pair a measurement rather than an assertion: the two are
    equal on the first prompt and differ by an order of magnitude after
    it.

    **The fixture is the figure.** These numbers are a function of the
    lock's content, not just its length: on the reference arm the block
    carries `lock_manifest_line`, whose topic is `_lock_topic` of that
    content, capped at 80 characters. An earlier revision of this table
    published 7,784 / 244 / 221 on the three manifest-bearing writes —
    48 characters, 12 estimated tokens, below what the fixture above
    produces — from a store nothing recorded. That is why the fixture is
    a module constant and the table has a producer.
    """
    out: dict[str, int] = {}
    work, db = _reference_lock_store(tier)
    out["gate_skip_first"] = _audit_tokens_from_block(
        _fire_for_reference(
            work, db, prompt=GATED_PROMPT, session_id=f"gs-{tier}",
        )
    )
    work, db = _reference_lock_store(tier)
    out["retrieval_first"] = _audit_tokens_from_block(
        _fire_for_reference(work, db, prompt=PROMPT, session_id=f"r1-{tier}")
    )
    # Same store and same session id, so the second fire is turn two and
    # the `<session-start>` sub-block is gone from the envelope.
    out["turn_two"] = _audit_tokens_from_block(
        _fire_for_reference(
            work, db, prompt=REF_TURN_TWO_PROMPT, session_id=f"r1-{tier}",
        )
    )
    work, db = _reference_lock_store(tier)
    out["session_start"] = _audit_tokens_from_block(
        _fire_for_reference(work, db, prompt=None, session_id=f"ss-{tier}")
    )
    return out


def exploration_slot() -> dict[str, object]:
    """Does the #1279 slot change the block the ceiling emits?

    The drawn belief is appended to the tail of the pack and the ceiling
    sheds the per-turn lane tail-first, so on an over-ceiling block the
    draw is the first element deleted: the `exploration_events` row —
    written upstream, before the ceiling runs — can name a belief the
    model never saw. This fires one store twice, with the slot on and
    off, and compares the emitted bytes.

    Returns the emitted token count, whether the two blocks are
    byte-identical, and how many of the ledger's drawn and displaced ids
    reached the block.

    **The vacuity guard is in here, not in the caller.** A run in which
    the slot never fired, or in which the ceiling dropped nothing,
    satisfies `drawn_emitted == 0` and reports the same token count as
    the shipped fixture — so a `--emit-figures` run that had lost the
    slot would publish figures identical to a run that had it. Guarding
    only the `--exploration` CLI arm left exactly that hole: running
    both arms with the slot off passed the derived-figures gate.
    """
    n_locks, n_core, n_hits = 60, 20, 12
    slot_env = (
        "AELFRICE_EXPLORATION",
        "AELFRICE_EXPLORATION_CADENCE",
        "AELFRICE_EXPLORATION_SLOTS",
    )

    def build() -> tuple[Path, Path]:
        work = Path(tempfile.mkdtemp(prefix="aelf-explore-"))
        db = work / "memory.db"
        store = MemoryStore(str(db))
        try:
            for i in range(n_locks):
                store.insert_belief(_lock(i, 150))
            for i in range(n_core):
                store.insert_belief(
                    _belief(
                        f"C{i:031d}",
                        "coreword unrelated material " + "w" * 200,
                        alpha=4.0,
                    )
                )
            for i in range(n_hits):
                store.insert_belief(
                    _belief(f"H{i:031d}", f"{LANE_WORD} fact " + "z" * 400)
                )
        finally:
            store.close()
        return work, db

    def run(work: Path, db: Path, *, slot_on: bool) -> tuple[str, str]:
        os.environ["AELFRICE_DB"] = str(db)
        os.environ.pop("AELFRICE_HOOK_BLOCK_CEILING", None)
        for name in slot_env:
            os.environ.pop(name, None)
        if slot_on:
            for name in slot_env:
                os.environ[name] = "1"
        sout, serr = io.StringIO(), io.StringIO()
        payload = json.dumps(
            {
                "session_id": "explore",
                "transcript_path": "/dev/null",
                "cwd": str(work),
                "hook_event_name": "UserPromptSubmit",
                "prompt": LANE_PROMPT,
            }
        )
        rc = user_prompt_submit(
            stdin=io.StringIO(payload), stdout=sout, stderr=serr
        )
        for name in slot_env:
            os.environ.pop(name, None)
        if rc != 0:
            raise SystemExit(f"hook returned {rc}")
        return sout.getvalue(), serr.getvalue()

    work_off, db_off = build()
    off, _ = run(work_off, db_off, slot_on=False)
    work_on, db_on = build()
    on, err_on = run(work_on, db_on, slot_on=True)

    rendered = set(_ELEMENT_ID_RE.findall(on))
    store = MemoryStore(str(db_on))
    try:
        rows = [
            (json.loads(r["drawn_ids"]), json.loads(r["displaced_ids"]))
            for r in store._conn.execute(
                "SELECT drawn_ids, displaced_ids FROM exploration_events"
            )
        ]
    finally:
        store.close()
    drawn = [bid for row in rows for bid in row[0]]
    displaced = [bid for row in rows for bid in row[1]]
    dropped_match = re.search(r"dropped (\d+) belief element", err_on)
    n_dropped = int(dropped_match.group(1)) if dropped_match else 0
    if not rows:
        raise SystemExit(
            "exploration slot did not fire: the comparison would be vacuous"
        )
    if not n_dropped:
        raise SystemExit(
            "the ceiling dropped nothing: the comparison would be vacuous"
        )
    return {
        "tokens": _audit_tokens_from_block(on),
        "identical": on == off,
        "sha256": hashlib.sha256(on.encode()).hexdigest(),
        "ledger_rows": len(rows),
        "drawn": len(drawn),
        "drawn_emitted": sum(1 for b in drawn if b in rendered),
        "displaced": len(displaced),
        "displaced_emitted": sum(1 for b in displaced if b in rendered),
        "dropped": n_dropped,
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
        "--reference-tier", action="store_true",
        help="print each write's size for one 30,026-character lock, "
             "frozen tier against reference tier",
    )
    ap.add_argument(
        "--exploration", action="store_true",
        help="compare the emitted block with the #1279 slot on and off",
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
        for tier, writes in reference_tier_table().items():
            for write, tokens in writes.items():
                figures[f"ref_lock_30026_{write}_{tier}"] = tokens
        slot = exploration_slot()
        figures["exploration_slot_block_tokens"] = slot["tokens"]
        figures["exploration_slot_drawn_emitted"] = slot["drawn_emitted"]
        print(json.dumps(figures))
        return 0

    if args.dry_run:
        if args.reference_tier:
            print(
                "would fire four writes against a one-lock store of "
                f"{REF_LOCK_CHARS} characters, once per lock tier, with the "
                "ceiling disabled"
            )
            return 0
        if args.exploration:
            print(
                "would fire one 60-lock / 20-core / 12-hit store twice at "
                f"{HOOK_BLOCK_TOKEN_CEILING} tokens, with the #1279 "
                "exploration slot on and off"
            )
            return 0
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

    if args.reference_tier:
        rows = reference_tier_table()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(
                f"one {REF_LOCK_CHARS}-character user lock, estimated tokens "
                "emitted per write"
            )
            print(f"  {'write':<18}{'frozen':>10}{'reference':>12}")
            for write in REF_WRITES:
                print(
                    f"  {write:<18}{rows[LOCK_TIER_FROZEN][write]:>10}"
                    f"{rows[LOCK_TIER_REFERENCE][write]:>12}"
                )
        # Vacuity is refused inside `reference_tier_table`.
        return 0

    if args.exploration:
        slot = exploration_slot()
        if args.json:
            print(json.dumps(slot, indent=2))
        else:
            print(
                f"  block: {slot['tokens']} tokens, identical with the slot "
                f"on and off: {slot['identical']}"
            )
            print(f"  sha256: {slot['sha256']}")
            print(
                f"  ledger: {slot['ledger_rows']} row(s), "
                f"{slot['drawn']} drawn ({slot['drawn_emitted']} emitted), "
                f"{slot['displaced']} displaced "
                f"({slot['displaced_emitted']} emitted)"
            )
            print(f"  ceiling dropped: {slot['dropped']} element(s)")
        # Vacuity is refused inside `exploration_slot`, so reaching here
        # means the slot fired and the ceiling acted.
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
