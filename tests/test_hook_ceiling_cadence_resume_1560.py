"""#1560 round two: what the block ceiling does to a `<cadence-resume>` recap.

`HOOK_BLOCK_TOKEN_CEILING`'s docstring used to say the #871 recap was
charged to the `<aelfrice-memory>` envelope but that, "like a user lock,
it is not a `<belief>` element and the dropper cannot shed it", and
`_ceiling_drop_order` used to say that "elements outside both named
sections are the per-turn hits by construction". Both were false, and
they were false for as long as they were because nothing asserted either
of them. This module is the assertion.

Four claims, one per test:

1. the dropper sheds the recap's `<belief>` elements, and spares only the
   `<cadence-resume>` wrapper;
2. a `lock="user"` belief rendered *inside* the recap is shed too — the
   context rebuilder writes `locked="true"`, which `_LOCKED_ATTR` does not
   match, so the #379 exemption does not reach into the recap;
3. the recap's elements are bucketed with the per-turn hits and sit at
   that bucket's head, so the prompt's own hits are shed first;
4. a recap therefore costs the envelope prompt-matched beliefs that the
   same store and the same prompt reach without one.

**Nothing about the recap is hand-written.** The body is produced by the
real `_rebuild_and_format`, persisted by the real
`_write_cadence_resume_cache`, and read and wrapped by the real
`_maybe_read_cadence_resume` on a genuine first prompt. A canned recap
would decide the attribute spelling claim 2 turns on, and the wrapper
shape claim 1 turns on, rather than observing either.

The figures these behaviours produce are
`scripts/measure_block_ceiling.py --resume-drop`. This module asserts the
*directions* — shed, spared, ordered, lost — so a render change moves the
producer's numbers without reddening a fixture, while a change to what
the dropper does reddens here.

**Mutation-proved, two mutations covering all four tests.** Adding a
recap-span exemption to `enforce_block_ceiling`'s `droppable` filter —
the behaviour the old docstring asserted — fails tests 1 and 2 and
nothing else. Replacing `reversed(lane)` with `lane` in
`_ceiling_drop_order`, which puts the recap at the front of its bucket
instead of the back, fails tests 3 and 4; under it the prompt keeps every
one of its matched beliefs, which is the displacement claim stated the
other way round.
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


def _reached(out: str) -> set[str]:
    """Prompt-matched belief ids the model can read in this payload.

    An element and a `seen` pointer both count: the pointer's contract is
    that the text is elsewhere in this window. Counting elements alone
    would read #1547's dedupe as a loss.
    """
    ids = {m.group("id") for m in _BELIEF_ELEMENT_RE.finditer(out)}
    ids |= set(_SEEN_ID_RE.findall(out))
    return {i for i in ids if i.startswith("H")}


def test_the_dropper_sheds_recap_elements_and_spares_only_the_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 1, against the sentence that said the dropper could not.

    Two arms so the shed is a difference rather than an assertion about
    one number: the same recap with the trim disabled and at the shipped
    ceiling.
    """
    untrimmed, _ = _arm(
        tmp_path, monkeypatch, name="untrimmed", recap=True, ceiling=0,
    )
    trimmed, err = _arm(
        tmp_path, monkeypatch, name="trimmed", recap=True, ceiling=None,
    )
    assert _RESUME_OPEN in untrimmed and _RESUME_OPEN in trimmed

    # The trim is live. A fire the ceiling never acted on says nothing
    # about what the ceiling sheds.
    assert "dropped" in err, err

    before = _recap_elements(untrimmed)
    after = _recap_elements(trimmed)
    assert before, "the untrimmed recap carries no element to shed"
    assert after < before, (
        "the ceiling shed no <belief> element from the <cadence-resume> "
        f"recap: {len(before)} before, {len(after)} after"
    )

    # The wrapper is what survives, and it survives whole — a recap
    # trimmed down to an opening tag would be a different defect.
    assert _RESUME_OPEN in trimmed
    assert _RESUME_CLOSE in trimmed


def test_a_locked_belief_inside_the_recap_is_shed_like_any_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 2: the #379 exemption does not reach into the recap.

    `enforce_block_ceiling` spares `lock="user"`. The context rebuilder
    renders the same belief as `locked="true"`, so the attribute the
    dropper reads is simply absent from every element of the recap — a
    user lock inside it included. The premise is asserted first: if the
    recap ever stops carrying a locked row, the claim below would pass by
    having nothing to test.
    """
    untrimmed, _ = _arm(
        tmp_path, monkeypatch, name="untrimmed", recap=True, ceiling=0,
    )
    trimmed, _ = _arm(
        tmp_path, monkeypatch, name="trimmed", recap=True, ceiling=None,
    )
    lo, hi = _recap_span(untrimmed)
    recap_body = untrimmed[lo:hi]
    locked_inside = {
        m.group("id")
        for m in _BELIEF_ELEMENT_RE.finditer(recap_body)
        if 'locked="true"' in m.group("attrs")
    }
    assert locked_inside, (
        "the recap rendered no locked belief, so this fixture cannot show "
        "one being shed"
    )
    # The premise the claim rests on: the dropper's own marker is absent.
    assert _LOCKED_ATTR not in recap_body, (
        "the recap now carries the attribute the #379 exemption reads, so "
        "its locks are no longer droppable and this test is about the "
        "wrong mechanism"
    )
    survivors = _recap_elements(trimmed)
    assert locked_inside - survivors, (
        "every locked belief inside the recap survived the ceiling: the "
        "exemption is reaching content this test says it cannot"
    )


def test_the_recap_is_bucketed_behind_the_prompts_own_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 3, against `_ceiling_drop_order`'s "by construction".

    The order is read off the real function rather than inferred from
    what survived, so the assertion is about the shed sequence and not
    about one fixture's arithmetic. `<core>` first, then the per-turn
    hits, then the recap — which is the prompt-independent lane going
    last, the inverse of what the ordering exists to do.
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
    assert "recap" in lanes, "no recap element reached the drop order"
    assert "hit" in lanes, (
        "no per-turn hit element is droppable on this fixture, so the "
        "recap has nothing to be ordered against"
    )
    assert "core" in lanes, "no <core> element reached the drop order"

    first_recap = lanes.index("recap")
    assert max(i for i, x in enumerate(lanes) if x == "hit") < first_recap, (
        "a per-turn hit is shed after a recap element; the recap is no "
        "longer at the head of the hit bucket"
    )
    assert max(i for i, x in enumerate(lanes) if x == "core") < lanes.index(
        "hit"
    ), "a <core> element is shed after a per-turn hit"


def test_the_recap_costs_the_envelope_prompt_matched_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 4: the trade the scope paragraph said nobody had measured.

    Same store shape, same prompt, same ceiling; the only difference is
    whether a resume cache is there to be read. The control arm is what
    keeps this from passing vacuously — a fixture that reaches no
    prompt-matched belief either way would assert nothing.
    """
    control, _ = _arm(
        tmp_path, monkeypatch, name="control", recap=False, ceiling=None,
    )
    with_recap, _ = _arm(
        tmp_path, monkeypatch, name="recap", recap=True, ceiling=None,
    )
    assert _RESUME_OPEN not in control
    assert _RESUME_OPEN in with_recap

    reached_without = _reached(control)
    reached_with = _reached(with_recap)
    assert reached_without, (
        "the control arm reached no prompt-matched belief, so a fall to "
        "fewer would be a fall from nothing"
    )
    assert len(reached_with) < len(reached_without), (
        "the recap cost the envelope no prompt-matched belief: "
        f"{len(reached_without)} without it, {len(reached_with)} with it"
    )
