"""#1564: where a `<cadence-resume>` recap sits in the ceiling's shed order.

The #871 recap is prepended to the session-start sub-block, so its
`<belief>` elements sit outside both `<core>` and `<recent-work>`.
`_ceiling_drop_order` used to bucket "everything outside those two
sections" as the per-turn hits -- a negative test that silently accepted a
third kind of element -- and because the recap is *prepended* it landed at
that bucket's head, where `reversed(lane)` shed it last. #1560 round two
measured the consequence and this module pinned it: on one first prompt the
prompt's own matched beliefs went from 6 to 0 to make room for a recap of an
earlier session.

The operator ruled on 2026-09-17 that **the recap sheds first, and whole,
and never suppresses a hit**. This module now pins that ruling, and the
tests are the shed-order claims turned over:

1. the ceiling sheds the recap whole -- every `<belief>` element of it and
   the `<cadence-resume>` wrapper with them, so nothing is left holding a
   fragment of a recap (AC4);
2. `_ceiling_drop_order` puts every recap element ahead of every `<core>`
   element and every per-turn hit, so the module reds if the recap is ever
   reclassified into the hits bucket again (AC2);
3. a `lock="user"` element inside a recap keeps the wrapper, because the
   #379 always-injected contract outranks the whole-shed rule. No shipped
   render produces one -- `context_rebuilder` spells a lock
   `locked="true"` -- so this arm is built by hand against
   `enforce_block_ceiling` rather than fired through the hook.

**Nothing about the recap in the fired arms is hand-written.** The body is
produced by the real `_rebuild_and_format`, persisted by the real
`_write_cadence_resume_cache`, and read and wrapped by the real
`_maybe_read_cadence_resume` on a genuine first prompt. A canned recap
would decide the attribute spelling and the wrapper shape rather than
observing either.

The figures these behaviours produce are
`scripts/measure_block_ceiling.py --resume-drop`. This module asserts the
*directions* -- whole, and first -- so a render change moves
the producer's numbers without reddening a fixture, while a change to what
the dropper does reddens here.

**Mutation-proved, one mutation per claim, each reverted by hand and the
module re-run green afterwards.** Dropping the recap lane out of
`_ceiling_drop_order`, so its elements fall into the `else` bucket again,
fails claim 2 and nothing else. Skipping the whole-shed branch in
`enforce_block_ceiling`, so the recap sheds an element at a time, fails
claim 1: the wrapper comes back holding a fragment. Removing the
`lock="user"` fallback from `_recap_shed`, so the span is always cut
whole, fails claim 3.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.context_rebuilder import RecentTurn
from aelfrice.hook import (
    _BELIEF_ELEMENT_RE,
    _LOCKED_ATTR,
    CORE_CLOSE_TAG,
    CORE_OPEN_TAG,
    SESSION_START_SUBBLOCK_OPEN,
    _ceiling_drop_order,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.rebuild_log import load_rebuilder_config
from aelfrice.store import MemoryStore

_CEILING_ENV = "AELFRICE_HOOK_BLOCK_CEILING"
_RESUME_OPEN = "<cadence-resume"
_RESUME_CLOSE = "</cadence-resume>"

_WORD = "banana"
_PROMPT = f"tell me everything about the {_WORD} please"

# The `--lanes` / `--resume-drop` fixture shape: only the `H` rows carry
# the prompt's word, so "a prompt-matched belief reached the model" is
# countable by construction rather than by judgement.
_LOCKS = 40
_CORE = 20
_HITS = 20

_SEEN_ID_RE = re.compile(r'^  seen (.+?): ".*"$', re.MULTILINE)


@pytest.fixture(autouse=True)
def _pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exported value may decide a result here."""
    monkeypatch.delenv(_CEILING_ENV, raising=False)
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
    ):
        monkeypatch.delenv(var, raising=False)


def _mk(bid: str, content: str, *, locked: bool = False,
        alpha: float = 1.0) -> Belief:
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


def _seed(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        for i in range(_LOCKS):
            store.insert_belief(
                _mk(f"L{i:031d}", "lockword " + "q" * 150, locked=True)
            )
        for i in range(_CORE):
            store.insert_belief(
                _mk(
                    f"C{i:031d}",
                    "coreword unrelated material " + "w" * 200,
                    alpha=4.0,
                )
            )
        for i in range(_HITS):
            store.insert_belief(
                _mk(f"H{i:031d}", f"{_WORD} fact " + "z" * 400)
            )
    finally:
        store.close()


def _arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    recap: bool,
    ceiling: int | None,
) -> tuple[str, str]:
    """One first-prompt fire against a store seeded for this arm alone.

    A store per arm, never one store fired twice: a fire stamps
    `last_retrieved_at` on every belief it renders, which reorders the
    next fire's retrieval, so shared-store arms would differ by their
    order as well as by the variable under test.

    `transcript_path` is `/dev/null` deliberately. With a real window the
    rebuild lane re-queries and the per-turn hit lane comes back empty,
    and a fixture with no hits cannot show a hit being displaced. The
    recap below is built from its own explicit window instead.
    """
    work = tmp_path / name
    work.mkdir()
    db = work / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    if ceiling is None:
        monkeypatch.delenv(_CEILING_ENV, raising=False)
    else:
        monkeypatch.setenv(_CEILING_ENV, str(ceiling))
    if recap:
        _write_recap(work, db)
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": f"resume-{name}",
        "transcript_path": "/dev/null",
        "cwd": str(work),
        "hook_event_name": "UserPromptSubmit",
        "prompt": _PROMPT,
    })
    rc = user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr
    )
    assert rc == 0
    # The hook fails soft, so an exception inside it becomes a stderr
    # trace and rc 0 — indistinguishable from a small payload.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def _write_recap(work: Path, db: Path) -> None:
    """Persist a real rebuilder body where the resume reader will find it.

    `_rebuild_and_format` is the function a P1 cadence fire calls, and
    `_write_cadence_resume_cache` is the function it hands the result to,
    so the bytes the hook later wraps are the bytes a cadence fire would
    have left. Only the window is supplied here rather than read off a
    transcript.
    """
    cfg = load_rebuilder_config(work)
    body = hook._rebuild_and_format(
        [
            RecentTurn(role="user", text=f"what about the {_WORD} store"),
            RecentTurn(
                role="assistant",
                text=f"the {_WORD} store holds coreword unrelated material",
            ),
        ],
        cfg.token_budget,
        rebuild_log_enabled=False,
    )
    assert '<belief id="' in body, body[:400]
    hook._write_cadence_resume_cache(
        body, "resume-prev", "p1_every_k_turns", io.StringIO()
    )
    assert (db.parent / "cadence_resume_cache.json").exists()


def _envelope(out: str) -> str:
    """The `<aelfrice-memory>` envelope, tags included."""
    start = out.index(hook.OPEN_TAG)
    end = out.index(hook.CLOSE_TAG) + len(hook.CLOSE_TAG)
    return out[start:end]


def _recap_span(body: str) -> tuple[int, int]:
    return body.index(_RESUME_OPEN), body.index(_RESUME_CLOSE)


def _recap_elements(body: str) -> set[str]:
    lo, hi = _recap_span(body)
    return {m.group("id") for m in _BELIEF_ELEMENT_RE.finditer(body[lo:hi])}


def _element_ids(out: str) -> set[str]:
    """Ids rendered as a full `<belief>` element anywhere in `out`."""
    return {m.group("id") for m in _BELIEF_ELEMENT_RE.finditer(out)}


def _reached(out: str) -> set[str]:
    """Prompt-matched belief ids the model can read in this payload.

    An element and a `seen` pointer both count: the pointer's contract is
    that the text is elsewhere in this window. Counting elements alone
    would read #1547's dedupe as a loss.
    """
    ids = {m.group("id") for m in _BELIEF_ELEMENT_RE.finditer(out)}
    ids |= set(_SEEN_ID_RE.findall(out))
    return {i for i in ids if i.startswith("H")}


def test_the_ceiling_sheds_the_whole_recap_wrapper_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 1 (AC4): no fragment of a recap survives the trim.

    Two arms so the shed is a difference rather than an assertion about one
    number: the same recap with the trim disabled and at the shipped
    ceiling. Before #1564 this arm left a wrapper holding roughly half of
    its original beliefs with nothing saying the rest were gone.
    """
    untrimmed, _ = _arm(
        tmp_path, monkeypatch, name="untrimmed", recap=True, ceiling=0,
    )
    trimmed, err = _arm(
        tmp_path, monkeypatch, name="trimmed", recap=True, ceiling=None,
    )
    assert _RESUME_OPEN in untrimmed, "the untrimmed arm carries no recap"

    # The trim is live. A fire the ceiling never acted on says nothing
    # about what the ceiling sheds.
    assert "dropped" in err, err

    before = _recap_elements(untrimmed)
    assert before, "the untrimmed recap carries no element to shed"

    assert _RESUME_OPEN not in trimmed, (
        "the <cadence-resume> wrapper survived its own beliefs: a wrapper "
        "holding a fragment of a recap is what #1564 AC4 rules out"
    )
    assert _RESUME_CLOSE not in trimmed
    # The cut is bounded: the recap's neighbour in the same string is the
    # session-start sub-block, and a span running past the close tag would
    # take that with it. The ids cannot be compared instead — `<core>`
    # renders the same beliefs the recap does, so a surviving id says
    # nothing about which of the two rendered it.
    assert SESSION_START_SUBBLOCK_OPEN in trimmed


def test_the_recap_sheds_ahead_of_core_and_of_the_prompts_own_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 2 (AC2): the recap is a lane, not a passenger in the hits.

    The order is read off the real `_ceiling_drop_order` rather than
    inferred from what survived, so the assertion is about the shed
    sequence and not about one fixture's arithmetic. All three lanes are
    asserted non-empty first: an order with no hit in it would pass this
    by having nothing to order the recap against.
    """
    untrimmed, _ = _arm(
        tmp_path, monkeypatch, name="untrimmed", recap=True, ceiling=0,
    )
    body = _envelope(untrimmed)
    rlo, rhi = _recap_span(body)
    clo = body.index(CORE_OPEN_TAG)
    chi = body.index(CORE_CLOSE_TAG)

    def lane(pos: int) -> str:
        if rlo <= pos < rhi:
            return "recap"
        if clo <= pos < chi:
            return "core"
        return "hit"

    elements = list(_BELIEF_ELEMENT_RE.finditer(body))
    droppable = [
        m for m in elements if _LOCKED_ATTR not in m.group("attrs")
    ]
    order = _ceiling_drop_order(body, droppable)
    lanes = [lane(m.start()) for m in order]
    for name in ("recap", "core", "hit"):
        assert name in lanes, f"no {name} element reached the drop order"

    last_recap = max(i for i, x in enumerate(lanes) if x == "recap")
    first_core = min(i for i, x in enumerate(lanes) if x == "core")
    last_core = max(i for i, x in enumerate(lanes) if x == "core")
    first_hit = min(i for i, x in enumerate(lanes) if x == "hit")
    assert last_recap < first_core, (
        "a <core> element is shed before a recap element; the recap is no "
        "longer the first lane"
    )
    assert last_core < first_hit, (
        "a per-turn hit is shed before a <core> element"
    )


def test_a_user_locked_element_inside_a_recap_keeps_the_wrapper() -> None:
    """Claim 3: #379 outranks the whole-shed rule.

    Built by hand rather than fired, and deliberately so: no shipped render
    puts `lock="user"` inside a recap, because `context_rebuilder` spells a
    lock `locked="true"`. The branch exists for the day that changes, and
    an untested branch is the one that breaks then.

    The recap's droppable elements still go, and they still go before
    `<core>`; what survives is the lock and the wrapper around it, which is
    a remainder a reader can explain.
    """
    pad = "y" * 900
    recap = (
        "<cadence-resume from='prev' policy='p1' ts='t'>\n"
        f'<belief id="R1" lock="none">{pad}</belief>\n'
        f'<belief id="R2" lock="user">{pad}</belief>\n'
        f'<belief id="R3" lock="none">{pad}</belief>\n'
        "</cadence-resume>"
    )
    core = (
        f"{CORE_OPEN_TAG}\n"
        f'<belief id="C1" lock="none">{pad}</belief>\n'
        f"{CORE_CLOSE_TAG}"
    )
    body = f"{hook.OPEN_TAG}\n{recap}\n{core}\n{hook.CLOSE_TAG}\n"
    outcome = hook.enforce_block_ceiling(body, ceiling=600)

    assert set(outcome.dropped_ids) >= {"R1", "R3"}, outcome.dropped_ids
    assert "R2" not in outcome.dropped_ids, (
        "a lock=\"user\" element was shed with the recap around it; the "
        "#379 always-injected contract does not survive the whole-shed rule"
    )
    assert _RESUME_OPEN in outcome.body and _RESUME_CLOSE in outcome.body, (
        "the wrapper went even though a belief it holds had to stay"
    )
    assert _element_ids(outcome.body) & {"R1", "R2", "R3"} == {"R2"}
