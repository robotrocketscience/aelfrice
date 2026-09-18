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
tests are the old claims turned over:

1. the ceiling sheds the recap whole -- every `<belief>` element of it and
   the `<cadence-resume>` wrapper with them, so nothing is left holding a
   fragment of a recap (AC4);
2. `_ceiling_drop_order` puts every recap element ahead of every `<core>`
   element and every per-turn hit, so the module reds if the recap is ever
   reclassified into the hits bucket again (AC2);
3. a belief the recap rendered does not come back below it as a `seen`
   pointer: the recap is outside #1547's envelope dedupe, so it cannot
   collapse a prompt hit into a pointer the trim then deletes;
4. the same store and the same prompt reach the same prompt-matched
   beliefs with a resume cache and without one, which is the A/B of AC3
   and AC7 as restated -- a prompt hit must not be lost because a recap
   displaced or collapsed it;
5. the #1382 ledger keeps a hit whose text the session-start sub-block
   rendered, even though the pack emitted that hit as a `seen` pointer.
   #1564 read the call site as a fourth defect and asked for the
   renderer's augmented set instead; the alternative was implemented, the
   suite disproved it, and this states the property the other way;
6. a `lock="user"` element inside a recap keeps the wrapper, because the
   #379 always-injected contract outranks the whole-shed rule. No shipped
   render produces one -- `context_rebuilder` spells a lock
   `locked="true"` -- so this arm is built by hand against
   `enforce_block_ceiling` rather than fired through the hook;
7. `dropped_ids` names only what the block lost. Claim 3 takes the recap
   out of the envelope dedupe, so every belief the recap carries is
   rendered a second time below it, and the whole-shed cut does not take
   that copy. Reporting the span's ids wholesale told four accounting
   surfaces that beliefs the model was shown were never shown -- 40 user
   locks among them, on the `--resume-drop` fixture. Two arms: the
   invariant on one hand-built body that carries all three kinds of
   second render, and the consequence on a fired hook, where the audit
   row `aelf tail` prints must name every prompt hit the envelope shows.

**Nothing about the recap in the fired arms is hand-written.** The body is
produced by the real `_rebuild_and_format`, persisted by the real
`_write_cadence_resume_cache`, and read and wrapped by the real
`_maybe_read_cadence_resume` on a genuine first prompt. A canned recap
would decide the attribute spelling and the wrapper shape rather than
observing either.

The figures these behaviours produce are
`scripts/measure_block_ceiling.py --resume-drop`. This module asserts the
*directions* -- whole, first, and no hit lost -- so a render change moves
the producer's numbers without reddening a fixture, while a change to what
the dropper does reddens here.

**Mutation-proved, one mutation per claim, each reverted by hand and the
module re-run green afterwards.** Dropping the recap lane out of
`_ceiling_drop_order`, so its elements fall into the `else` bucket again,
fails claim 2 and nothing else. Skipping the whole-shed branch in
`enforce_block_ceiling`, so the recap sheds an element at a time, fails
claim 1: the wrapper comes back holding a fragment. Putting
`_ids_rendered_verbatim_in` back in place of `_session_start_dedupe_ids`,
so the recap's ids dedupe again, fails claims 3 and 4. Handing
`_verbatim_ids` the renderer's augmented set, the change #1564 asked for,
fails claim 5. Removing the `lock="user"` fallback from `_recap_shed`, so
the span is always cut whole, fails claim 6. Dropping the
`ids_rendered_outside_recap` test in `enforce_block_ceiling`, so the whole
shed reports every id in the span, fails claim 7 on both arms.
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
    AUDIT_HOOK_USER_PROMPT_SUBMIT,
    CORE_CLOSE_TAG,
    CORE_OPEN_TAG,
    SESSION_START_SUBBLOCK_OPEN,
    _audit_path_for_db,
    _ceiling_drop_order,
    read_hook_audit,
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


def test_a_recap_id_does_not_suppress_the_prompts_own_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 3, at the renderer: the recap is outside the envelope dedupe.

    `_format_hits_with_session_start` reads `_ids_rendered_verbatim_in` off
    the string it is handed, and the recap is concatenated onto the front
    of that string before it arrives. Every id the recap rendered used to
    downgrade the prompt's own hit to a `seen` pointer -- and the pointer
    is then deleted with the element it names, because that is what
    `enforce_block_ceiling` does to a pointer whose referent it sheds.

    The fixture's `<core>` and the recap render the same beliefs, so the
    premise is asserted rather than assumed: the recap must carry at least
    one id the prompt also retrieved, or there is nothing here to suppress.
    """
    untrimmed, _ = _arm(
        tmp_path, monkeypatch, name="untrimmed", recap=True, ceiling=0,
    )
    body = _envelope(untrimmed)
    recap_ids = _recap_elements(body)
    hit_ids = {i for i in recap_ids if i.startswith("H")}
    assert hit_ids, (
        "the recap rendered no belief the prompt also retrieves, so this "
        "fixture cannot show one being suppressed"
    )
    lo, hi = _recap_span(body)
    below = body[hi:]
    pointed = set(_SEEN_ID_RE.findall(below))
    assert not (hit_ids & pointed), (
        "a belief the recap rendered came back as a `seen` pointer below "
        f"it: {sorted(hit_ids & pointed)}"
    )
    assert hit_ids <= _element_ids(below), (
        "a prompt hit the recap also carries is missing its own element "
        "below the recap"
    )


def test_the_recap_costs_the_envelope_no_prompt_matched_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 4 (AC3, and AC7 as restated): the A/B, with and without a cache.

    Same store shape, same prompt, same ceiling; the only difference is
    whether a resume cache is there to be read. The control arm is what
    keeps this from passing vacuously -- a fixture that reaches no
    prompt-matched belief either way would assert nothing.

    Both an element and a `seen` pointer count as reaching the model: the
    pointer's contract is that the text is elsewhere in this window, so
    counting elements alone would read #1547's dedupe as a loss.
    """
    control, _ = _arm(
        tmp_path, monkeypatch, name="control", recap=False, ceiling=None,
    )
    with_recap, err = _arm(
        tmp_path, monkeypatch, name="recap", recap=True, ceiling=None,
    )
    assert _RESUME_OPEN not in control
    assert "dropped" in err, err

    reached_without = _reached(control)
    reached_with = _reached(with_recap)
    assert reached_without, (
        "the control arm reached no prompt-matched belief, so equality "
        "here would be equality between two empty sets"
    )
    assert reached_with >= reached_without, (
        "the recap cost the envelope a prompt-matched belief: "
        f"{sorted(reached_without)} without it, {sorted(reached_with)} "
        "with it"
    )


def test_the_ledger_keeps_a_belief_the_sub_block_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 5: `_verbatim_ids` takes the caller's set, and must keep doing so.

    An adversarial read of #1564 called the call site a fourth defect:
    `_verbatim_ids` is handed `read_rendered(session_id)` while the
    renderer dedupes against that set unioned with the session-start
    sub-block's own ids, and #1382 AC4 asks for one derivation, not two.
    The alternative was implemented and it is wrong. The two sets differ on
    exactly the hits the sub-block rendered verbatim *above* the pack, and
    those are the hits whose text is in the window -- the sub-block put it
    there. Recording them is the true claim; the union drops it, and turn
    two's `seen` pointers go to zero.

    This is the property stated positively, on one fire: a hit the envelope
    emitted as a `seen` pointer, whose text the sub-block rendered in the
    same envelope, is in the ledger. `test_hook_injection_ceiling_wiring
    .py::test_ups_seen_pointer_on_turn_two_names_a_belief_turn_one_rendered`
    is the cross-turn consequence.

    The ceiling is off here on purpose. A trim-dropped belief must *not* be
    recorded, #1551 closed that through `emitted_hits`, and leaving the
    trim on would mix the two rules.
    """
    from aelfrice.injection_ledger import (
        TURN_DIFFERENTIAL_ENV_VAR,
        read_rendered,
    )

    monkeypatch.setenv(TURN_DIFFERENTIAL_ENV_VAR, "1")
    out, _ = _arm(
        tmp_path, monkeypatch, name="ledger", recap=True, ceiling=0,
    )
    pointed = set(_SEEN_ID_RE.findall(out))
    assert pointed, (
        "no belief was collapsed to a `seen` pointer on this fire, so the "
        "renderer's set and the caller's set cannot be told apart here"
    )
    # The premise the claim rests on: a pointer names text this envelope
    # carries. If that stops holding, recording the id stops being true and
    # this test is about the wrong thing.
    assert pointed <= _element_ids(out), sorted(pointed - _element_ids(out))
    recorded = read_rendered("resume-ledger")
    assert pointed <= recorded, (
        "the ledger dropped a belief whose text is in this window: "
        f"{sorted(pointed - recorded)}. The next turn will re-render it in "
        "full instead of pointing at it, which is #1382 not working."
    )


def test_a_user_locked_element_inside_a_recap_keeps_the_wrapper() -> None:
    """Claim 6: #379 outranks the whole-shed rule.

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


def test_a_recap_id_the_body_still_renders_is_not_reported_dropped() -> None:
    """Claim 7: `dropped_ids` names what the block lost, not what one cut took.

    The whole-recap shed removes a span, and since claim 3 took the recap
    out of #1547's envelope dedupe every belief in that span is *also*
    rendered somewhere else in the same body -- in `<locked>`, in `<core>`
    or as the prompt's own hit. Reporting the span's ids wholesale
    therefore names beliefs the model was shown, and
    `BlockCeilingOutcome.dropped_ids` is read by four call sites as the
    opposite claim: `user_prompt_submit` filters `emitted_hits` by it, and
    that list is what reaches `record_retrieval`, the audit row's
    `beliefs[]`, the #740 dedup ring and the #1382 ledger.

    Built by hand so the three kinds of second render sit in one body and
    the invariant is stated once. `R9` is the control: a belief the recap
    alone carries is still reported, so this cannot pass by reporting
    nothing.
    """
    pad = "y" * 900
    recap = (
        "<cadence-resume from='prev' policy='p1' ts='t'>\n"
        f'<belief id="L1" locked="true">{pad}</belief>\n'
        f'<belief id="C1" locked="true">{pad}</belief>\n'
        f'<belief id="H1" locked="true">{pad}</belief>\n'
        f'<belief id="R9" locked="true">{pad}</belief>\n'
        "</cadence-resume>"
    )
    locked = f'<belief id="L1" {_LOCKED_ATTR}>{pad}</belief>'
    core = (
        f"{CORE_OPEN_TAG}\n"
        f'<belief id="C1" lock="none">{pad}</belief>\n'
        f"{CORE_CLOSE_TAG}"
    )
    hit = f'<belief id="H1" lock="none">{pad}</belief>'
    body = (
        f"{hook.OPEN_TAG}\n{SESSION_START_SUBBLOCK_OPEN}\n"
        f"{recap}\n{locked}\n{core}\n</session-start>\n"
        f"{hit}\n{hook.CLOSE_TAG}\n"
    )
    outcome = hook.enforce_block_ceiling(body, ceiling=900)

    assert _RESUME_OPEN not in outcome.body, "the recap did not shed"
    assert "R9" in outcome.dropped_ids, (
        "a belief only the recap rendered was not reported dropped, so "
        f"this fixture reports nothing: {outcome.dropped_ids}"
    )
    still_rendered = _element_ids(outcome.body) & set(outcome.dropped_ids)
    assert not still_rendered, (
        "`dropped_ids` names a belief the emitted body still renders in "
        f"full: {sorted(still_rendered)}. Every consumer of this list "
        "reads it as `the model never saw this`."
    )
    assert "L1" not in outcome.dropped_ids, (
        "a user lock reached `dropped_ids`. The #379 pool is uncapped and "
        "two call sites in `user_prompt_submit` state outright that a "
        "locked belief is never in this list."
    )


def test_the_audit_row_records_every_prompt_hit_the_envelope_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 7, fired: `aelf tail` prints what the model was actually given.

    `_write_hook_audit_record` is handed `emitted_hits`, which is `hits`
    minus `dropped_ids`, and `n_beliefs` is its length. A prompt hit the
    envelope renders in full but the audit row omits is the #1551 defect
    inverted: instead of claiming exposure for deleted text, the hook
    withholds it for text it shipped, and the same list drives the
    exposure rows, the #740 dedup ring and the #1382 ledger.

    The trim must be live and it must shed the recap, or the arm cannot
    show the whole-recap shed reporting anything.
    """
    out, err = _arm(
        tmp_path, monkeypatch, name="audit", recap=True, ceiling=None,
    )
    assert "dropped" in err, err
    assert _RESUME_OPEN not in out, "the recap survived; nothing was shed"

    db = tmp_path / "audit" / "memory.db"
    rows = [
        r for r in read_hook_audit(_audit_path_for_db(db))
        if r.get("hook") == AUDIT_HOOK_USER_PROMPT_SUBMIT
    ]
    assert len(rows) == 1, [r.get("hook") for r in rows]
    beliefs = rows[0].get("beliefs")
    assert isinstance(beliefs, list)
    recorded = {
        b["id"] for b in beliefs if isinstance(b, dict) and b.get("id")
    }
    assert rows[0].get("n_beliefs") == len(beliefs)

    shown = {i for i in _element_ids(_envelope(out)) if i.startswith("H")}
    assert shown, (
        "the envelope rendered no prompt hit as an element, so there is "
        "nothing here to have been miscounted"
    )
    assert shown <= recorded, (
        "the hook rendered a prompt hit and recorded it as dropped: "
        f"{sorted(shown - recorded)}. `aelf tail` will show "
        f"{rows[0].get('n_beliefs')} injected beliefs beside a block "
        "carrying more than that."
    )
