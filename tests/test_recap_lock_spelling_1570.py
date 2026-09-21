"""#1570: the hook reads the lock attribute the recap's renderer writes.

`hook` renders a user lock `lock="user"`. `context_rebuilder._format_block`
renders one `locked="true"`. The two never met on the ordinary hook path,
so nothing had failed -- but the #871 `<cadence-resume>` recap is rebuilder
output spliced into a hook-rendered envelope, and on that path the dropper
could not see that an element was a lock. #1564's lock exemption therefore
never fired on shipped output, and the #379 always-injected contract did
not hold inside a recap.

**The hook was widened; the rebuilder was not changed.** `_element_is_locked`
accepts both spellings. The reasons are in its docstring; the enumeration
AC4 asks for, of what reads the rebuilder's attribute outside the hook, is
here because this is where a future change to that attribute gets caught:

1. the model. `<aelfrice-rebuild>` is written straight to stdout by the
   PreCompact lane (`hook.session_start` on `source == "compact"`), so the
   attribute is part of a published schema and not an internal token;
2. `docs/design/context_rebuilder.md` and the `context_rebuilder` module
   docstring, both of which show `locked="true"` / `locked="false"` as the
   block's output schema;
3. `tests/test_context_rebuilder.py`, `tests/test_context_rebuilder_hook.py`
   and `tests/test_lock_manifest_injection.py`, which assert the rendered
   attribute directly -- the last of them in the reference-tier form
   `locked="true" tier="reference"`;
4. `tests/test_render_cost_1526.py::test_rebuild_block_budget_used_reports_the_rendered_lines`,
   which reads the emitted line widths back; the attribute's *width* is in
   the `budget_used` numerator the block publishes about itself.

No machine consumer parses the attribute outside `hook`: the only other
reader of a rendered element, `scripts/measure_block_ceiling.py`, matches
`<belief id="..."` and nothing else. `test_the_rebuilder_still_renders_the_schema_it_publishes`
pins item 2 and the emitted bytes with it, so this fix is provably
byte-neutral on every lane but the dropper's decision.

**What the exemption actually keys on.** Not "is a lock" but "is a lock
this cut would be the last render of". Exempting every recap lock was
implemented first and reverted: `<locked>` renders the same locks uncapped
in the same envelope, so on the `--resume-drop` fixture's 40 locks it
pinned about 2,000 tokens of duplicate text and took the prompt's own
matched beliefs from 6 of 6 to 0 -- #1564's AC3 loss, restaged. The
surviving invariant is single: **no user-locked id ever reaches
`dropped_ids`**, which is what `_recap_shed` and its caller now implement
together.

That leaves exactly one class where the recap's element is kept: a #1558
reference lock, whose `<locked>` render is a `ref` manifest line rather
than an element. Its bounded-topic element in the recap is the only
element there is.

**Mutation-proved, one mutation per claim, each reverted by hand and the
module re-run green afterwards.** The mutations and their effects are
recorded in each test's docstring.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.context_rebuilder import RecentTurn, _format_block
from aelfrice.hook import (
    _BELIEF_ELEMENT_RE,
    _LOCKED_ATTR,
    _element_is_locked,
    enforce_block_ceiling,
    user_prompt_submit,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)
from aelfrice.rebuild_log import load_rebuilder_config
from aelfrice.retrieval import _lock_topic
from aelfrice.store import MemoryStore

_CEILING_ENV = "AELFRICE_HOOK_BLOCK_CEILING"
_RESUME_OPEN = "<cadence-resume"
_RESUME_CLOSE = "</cadence-resume>"

_WORD = "banana"
_PROMPT = f"tell me everything about the {_WORD} please"

# The reference lock's content: long enough that its verbatim render and
# its bounded topic are not the same string, so "the topic reached the
# model" is distinguishable from "the whole lock did".
_REF_BODY = (
    "The corpus mount is read through AELFRICE_CORPUS_ROOT. "
    + "Never hard-code a checkout path into a fixture. " * 40
)
_REF_TOPIC = _lock_topic(_REF_BODY)

# The `--resume-drop` fixture's lock count and length, so the
# duplicate-bytes arm below is measured against the store #1564 was ruled
# on rather than against a store too small to show the loss.
_FROZEN_LOCKS = 40


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


def _mk(
    bid: str,
    content: str,
    *,
    locked: bool = False,
    tier: str = LOCK_TIER_FROZEN,
    alpha: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-09-18T00:00:00Z" if locked else None,
        created_at="2026-09-18T00:00:00Z",
        last_retrieved_at=None,
        lock_tier=tier,
    )


def _attrs_of(block: str, bid: str) -> str:
    """The attribute tail a renderer wrote for one element, off its output.

    Read back through the dropper's own pattern rather than by splitting
    the line, so a change to `_BELIEF_ELEMENT_RE` that stopped capturing
    the tail reds here instead of silently making every caller's answer
    False.
    """
    for m in _BELIEF_ELEMENT_RE.finditer(block):
        if m.group("id") == bid:
            return m.group("attrs")
    raise AssertionError(f"no <belief id={bid!r}> element in:\n{block[:600]}")


def _element_ids(body: str) -> set[str]:
    return {m.group("id") for m in _BELIEF_ELEMENT_RE.finditer(body)}


# ---------------------------------------------------------------------------
# AC1 -- one predicate, two spellings, both read off a live renderer
# ---------------------------------------------------------------------------


def test_the_hook_reads_the_lock_spelling_the_rebuilder_writes() -> None:
    """AC1: `_element_is_locked` answers for real `_format_block` output.

    The attribute is taken off the renderer rather than written as a
    literal here, so the claim is "the two agree" and not "these two
    strings agree".

    Two controls make the direction unambiguous. `_LOCKED_ATTR` must be
    *absent* from the rebuilder's locked element -- that is the whole
    defect, and if it were present this test would pass on the old code.
    And `locked="false"`, which the rebuilder writes and the hook never
    does, must not read as a lock: the widened predicate uses substring
    tests, and `locked="true"` is one character from a spelling that means
    the opposite.

    Mutation: drop `_REBUILDER_LOCKED_ATTR` from `_element_is_locked`, so
    it reads `lock="user"` alone. FAIL -- `assert _element_is_locked(lock)`
    at the rebuilder arm. Restored: PASS.
    """
    block = _format_block(
        [],
        [
            _mk("RBLOCK", "a locked fact about the store", locked=True),
            _mk("RPLAIN", "an unlocked fact about the store"),
        ],
        set(),
        token_budget=4000,
    )
    lock_attrs = _attrs_of(block, "RBLOCK")
    plain_attrs = _attrs_of(block, "RPLAIN")

    assert _LOCKED_ATTR not in lock_attrs, (
        "the rebuilder now writes the hook's own spelling, so this module "
        "is testing a mismatch that no longer exists; re-read #1570 AC4 "
        "before deleting it"
    )
    assert _element_is_locked(lock_attrs), (
        f"the dropper cannot see a lock in {lock_attrs!r}, which is what "
        "`context_rebuilder._format_block` writes for an L0 belief"
    )
    assert not _element_is_locked(plain_attrs), (
        f"{plain_attrs!r} read as a lock; `locked=\"false\"` means the "
        "opposite and the ceiling would stop being able to shed anything "
        "the rebuilder rendered"
    )


def test_the_hook_still_reads_its_own_lock_spelling() -> None:
    """AC1, the other half: widening the predicate kept the old answer.

    `hook._format_hits` is the renderer for the per-turn hit lane, and its
    elements are the ones `enforce_block_ceiling` was written against. A
    predicate that traded one spelling for the other would pass the test
    above and break every lane this module is not about.

    Mutation: drop `_LOCKED_ATTR` from `_element_is_locked`. FAIL here,
    and in `test_hook_injection_ceiling.py` besides. Restored: PASS.
    """
    out = hook._format_hits(
        [
            _mk("HLOCK", "a locked fact", locked=True),
            _mk("HPLAIN", "an unlocked fact"),
        ]
    )
    assert _element_is_locked(_attrs_of(out, "HLOCK"))
    assert not _element_is_locked(_attrs_of(out, "HPLAIN"))


def test_the_rebuilder_still_renders_the_schema_it_publishes() -> None:
    """AC4: the fix moved no byte the rebuilder emits.

    The enumeration in this module's docstring is the reason the hook was
    the side that changed. This is that enumeration turned into a guard:
    the attribute those consumers read is asserted here in the exact two
    forms `docs/design/context_rebuilder.md` publishes, so changing the
    rebuilder's spelling later reds a test that names them rather than
    only the three modules that happen to assert it in passing.

    Mutation: change `_format_block` to emit `lock="user"`. FAIL on the
    `locked="true"` arm. Restored: PASS.
    """
    block = _format_block(
        [],
        [
            _mk("SCHLOCK", "a locked fact", locked=True),
            _mk("SCHPLAIN", "an unlocked fact"),
            _mk("SCHREF", _REF_BODY, locked=True, tier=LOCK_TIER_REFERENCE),
        ],
        set(),
        token_budget=4000,
    )
    assert 'id="SCHLOCK" locked="true"' in block
    assert 'id="SCHPLAIN" locked="false"' in block
    assert 'id="SCHREF" locked="true" tier="reference"' in block


# ---------------------------------------------------------------------------
# AC2 -- the exemption fires on real rebuilder output
# ---------------------------------------------------------------------------


def _recap_from_rebuilder(hits: list[Belief]) -> str:
    """Real `_format_block` output inside the real recap wrapper shape.

    The body is what a cadence fire would have cached; the wrapper is the
    shape `_maybe_read_cadence_resume` builds around it, attributes
    included, because `_ceiling_drop_order` finds the span by
    `RESUME_OPEN_TAG` and a wrapper written without its attributes would
    test a tag this code never sees.
    """
    body = _format_block([], hits, set(), token_budget=8000)
    return (
        f"{hook.RESUME_OPEN_TAG} from='prevsess' policy='p1_every_k_turns' "
        f"ts='2026-09-18T00:00:00Z'>\n{body}\n{hook.RESUME_CLOSE_TAG}"
    )


def test_a_rebuilder_rendered_lock_keeps_the_recap_it_is_the_last_render_of(
) -> None:
    """AC2: the #1564 exemption fires on output the rebuilder produced.

    The recap is real `_format_block` output in the real wrapper, and the
    envelope around it renders the lock nowhere else, so shedding the span
    would put a user lock in `dropped_ids`. Before #1570 that is exactly
    what happened: the dropper read `lock="user"` only, the lock was in
    the droppable set, and the whole span went.

    The padding is `<core>` so the ceiling has something to want, and the
    recap's own unlocked beliefs are the elements that must still go: the
    claim is "the lock survived", not "the trim did nothing".

    Mutation: drop `_REBUILDER_LOCKED_ATTR` from `_element_is_locked`.
    FAIL -- `RLOCK` is gone from the body and present in `dropped_ids`.
    Restored: PASS.
    """
    pad = "y" * 900
    recap = _recap_from_rebuilder(
        [
            _mk("RLOCK", "the locked fact only the recap renders", locked=True),
            _mk("RDROP1", pad),
            _mk("RDROP2", pad),
        ]
    )
    core = (
        f"{hook.CORE_OPEN_TAG}\n"
        f'<belief id="C1" lock="none">{pad}</belief>\n'
        f'<belief id="C2" lock="none">{pad}</belief>\n'
        f"{hook.CORE_CLOSE_TAG}"
    )
    body = f"{hook.OPEN_TAG}\n{recap}\n{core}\n{hook.CLOSE_TAG}\n"
    outcome = enforce_block_ceiling(body, ceiling=600)

    assert "RLOCK" not in outcome.dropped_ids, (
        "a user lock reached `dropped_ids`. Four call sites read that list "
        "as `the model never saw this`, and #379 says a lock always reaches "
        f"it: {outcome.dropped_ids}"
    )
    assert "RLOCK" in _element_ids(outcome.body), (
        "the lock's only element render was shed with the recap around it"
    )
    assert "the locked fact only the recap renders" in outcome.body
    assert _RESUME_OPEN in outcome.body and _RESUME_CLOSE in outcome.body, (
        "the wrapper went even though a belief it holds had to stay"
    )
    assert set(outcome.dropped_ids) >= {"RDROP1", "RDROP2"}, (
        "the recap's droppable elements survived, so this fixture proves "
        f"nothing about the shed: {outcome.dropped_ids}"
    )


def test_a_rebuilder_rendered_lock_rendered_outside_sheds_with_the_recap(
) -> None:
    """AC2's counterweight: recognising a lock is not exempting every lock.

    `<locked>` renders every non-reference lock uncapped in the same
    envelope, so the recap's copy is a duplicate and shedding it loses
    nothing. Exempting it instead pins bytes the block already carries and
    starves the prompt's own hits -- measured at 6 matched beliefs to 0 on
    the `--resume-drop` fixture, which is the #1564 AC3 regression this
    arm exists to catch.

    Mutation: drop the `rendered_outside` test from `_recap_shed`, so
    every recognised lock is exempt. FAIL -- the wrapper survives and
    `DUPLOCK` is still in the recap. Restored: PASS.
    """
    pad = "y" * 900
    recap = _recap_from_rebuilder(
        [
            _mk("DUPLOCK", "a lock the session-start block also renders",
                locked=True),
            _mk("RDROP1", pad),
        ]
    )
    locked_section = (
        "<locked>\n"
        '<belief id="DUPLOCK" lock="user">'
        "a lock the session-start block also renders</belief>\n"
        "</locked>"
    )
    core = (
        f"{hook.CORE_OPEN_TAG}\n"
        f'<belief id="C1" lock="none">{pad}</belief>\n'
        f"{hook.CORE_CLOSE_TAG}"
    )
    body = (
        f"{hook.OPEN_TAG}\n{hook.SESSION_START_SUBBLOCK_OPEN}\n"
        f"{recap}\n{locked_section}\n{core}\n</session-start>\n"
        f"{hook.CLOSE_TAG}\n"
    )
    outcome = enforce_block_ceiling(body, ceiling=400)

    assert _RESUME_OPEN not in outcome.body, (
        "the recap kept a lock the same envelope renders in <locked>; that "
        "pins duplicate bytes and is what took the --resume-drop fixture "
        "from 6 prompt-matched beliefs to 0"
    )
    assert "DUPLOCK" not in outcome.dropped_ids, (
        "the lock was reported dropped even though <locked> still renders "
        f"it: {outcome.dropped_ids}"
    )
    assert "a lock the session-start block also renders" in outcome.body, (
        "the lock's surviving render went too, so the content did not "
        "reach the model at all"
    )


# ---------------------------------------------------------------------------
# AC3 -- the #1558 reference lock, fired end to end
# ---------------------------------------------------------------------------


def _seed(db: Path) -> None:
    """One reference lock, plus enough bulk to put the block over the ceiling.

    The reference lock is the subject; the frozen locks beside it are the
    control that separates "a reference lock's element is kept" from "any
    lock's element is kept".
    """
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            _mk("RF" + "0" * 29, _REF_BODY, locked=True,
                tier=LOCK_TIER_REFERENCE)
        )
        # Forty frozen locks, the `--resume-drop` fixture's count and
        # length. One would prove the reference lock is not simply "any
        # lock kept"; forty is what makes
        # `test_the_recap_does_not_cost_the_prompt_a_matched_belief`
        # discriminating, because the duplicate bytes an over-broad
        # exemption would pin have to be worth more than the prompt's own
        # hits before their loss is visible.
        for i in range(_FROZEN_LOCKS):
            store.insert_belief(
                _mk(f"F{i:031d}", "frozenword " + "q" * 150, locked=True)
            )
        for i in range(20):
            store.insert_belief(
                _mk(
                    f"C{i:031d}",
                    "coreword unrelated material " + "w" * 200,
                    alpha=4.0,
                )
            )
        for i in range(20):
            store.insert_belief(
                _mk(f"H{i:031d}", f"{_WORD} fact " + "z" * 400)
            )
    finally:
        store.close()


def _write_recap(work: Path, db: Path) -> None:
    """Persist a real rebuilder body where the resume reader will find it.

    `_rebuild_and_format` is the function a P1 cadence fire calls and
    `_write_cadence_resume_cache` is the one it hands the result to, so
    the bytes the hook later wraps are the bytes a cadence fire would have
    left. Only the turn window is supplied here rather than read off a
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
    assert 'tier="reference"' in body, body[:800]
    hook._write_cadence_resume_cache(
        body, "resume-prev", "p1_every_k_turns", io.StringIO()
    )
    assert (db.parent / "cadence_resume_cache.json").exists()


def _fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str,
) -> tuple[str, str]:
    """One first-prompt fire against a store seeded for this arm alone.

    A store per arm, never one store fired twice: a fire stamps
    `last_retrieved_at` on every belief it renders, which reorders the
    next fire's retrieval.

    `transcript_path` is `/dev/null` for the reason
    `test_hook_recap_shed_order_1564.py` gives: with a real window the
    rebuild lane re-queries and the per-turn hit lane comes back empty.
    """
    work = tmp_path / name
    work.mkdir()
    db = work / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _write_recap(work, db)
    sout, serr = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "session_id": f"ref-{name}",
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
    # trace and rc 0 -- indistinguishable from a small payload.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    return sout.getvalue(), serr.getvalue()


def test_a_reference_locks_only_element_render_survives_a_shed_recap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: what reaches the model when the recap carries the only element.

    A #1558 reference lock renders in `<locked>` as a `ref` manifest line,
    not as an element, so the recap's bounded-topic element is the only
    element in the envelope carrying that content. Before #1570 the shed
    took it and the model got the pointer alone -- the exception the issue
    was filed for.

    Everything about the recap here is real: the body is
    `_rebuild_and_format`, the cache is `_write_cadence_resume_cache`, and
    the wrapper is `_maybe_read_cadence_resume` on a genuine first prompt.

    Four assertions, in the order a reader needs them: the trim was live;
    the frozen lock's recap element still sheds, so this is not "the recap
    survived intact"; the reference lock's element is still there; and the
    `ref` manifest line is there beside it, because the element surviving
    is not licence to stop naming it.

    Mutation: drop `_REBUILDER_LOCKED_ATTR` from `_element_is_locked`.
    FAIL -- no `tier="reference"` element in the payload and no
    `<cadence-resume>` wrapper. Restored: PASS.
    """
    out, err = _fire(tmp_path, monkeypatch, name="ref")

    # The trim is live. A fire the ceiling never acted on says nothing
    # about what the ceiling sheds.
    assert "dropped" in err, err

    ref_id = "RF" + "0" * 29
    ids = _element_ids(out)
    frozen_kept = {i for i in _recap_ids(out) if i.startswith("F")}

    assert not frozen_kept, (
        "a frozen lock's recap element survived. <locked> renders it "
        "uncapped in this same envelope, so keeping the recap's copy pins "
        "duplicate bytes -- and this arm can no longer tell a kept "
        f"reference lock from a kept recap: {sorted(frozen_kept)}"
    )
    assert ref_id in ids, (
        "the reference lock's bounded-topic element was shed with the "
        "recap. It is the only element render of that content in the "
        "envelope, so the model got a pointer to text that is not there"
    )
    assert _REF_TOPIC in out, (
        "the element is present but its topic is not, so what reaches the "
        "model is an empty render"
    )
    assert f"ref {ref_id}:" in out, (
        "the <aelfrice-locks-manifest> entry went. The element surviving "
        "does not replace the line that names the lock to the model"
    )
    assert _REF_BODY not in out, (
        "the reference lock was injected verbatim, which is the #1016-B "
        "bound gone rather than this fix working"
    )


def _recap_ids(out: str) -> set[str]:
    """Element ids inside the `<cadence-resume>` span, or none if it shed."""
    if _RESUME_OPEN not in out:
        return set()
    lo = out.index(_RESUME_OPEN)
    hi = out.index(_RESUME_CLOSE)
    return _element_ids(out[lo:hi])


def test_the_reference_lock_keeps_the_wrapper_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3, the remainder: a reader can explain what is left of the recap.

    `_recap_shed` keeps the wrapper when it keeps an element, and #1564
    AC4 rules out leaving a wrapper around an arbitrary fragment. The
    remainder here is exactly the elements the cut would have been the
    last render of -- on this store, the one reference lock.

    Mutation: make `_recap_shed` return the droppable spans unconditionally
    rather than cutting the whole span when nothing is kept, so a recap
    with no lock leaves a bare wrapper. That is `test_hook_recap_shed_order_1564.py`'s
    claim 1, and it reds there; this arm's own mutation is dropping the
    `rendered_outside` test, which FAILs the frozen-lock assertion below.
    Restored: PASS.
    """
    out, _ = _fire(tmp_path, monkeypatch, name="remainder")

    assert _RESUME_OPEN in out, "the wrapper went with the elements it kept"
    kept = _recap_ids(out)
    assert kept == {"RF" + "0" * 29}, (
        "the recap's remainder is not the reference lock alone: "
        f"{sorted(kept)}"
    )


def test_the_recap_does_not_cost_the_prompt_a_matched_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1564's AC3, re-asserted on a store the #1570 exemption fires on.

    The exemption keeps bytes inside the envelope's first lane, which is
    the lane #1564 ruled must never cost the prompt a hit. The reference
    lock's topic is bounded, so it should not -- but "should not" is the
    claim #1564 was filed about, and this store is the one where the new
    branch is live.

    Both an element and a `seen` pointer count as reaching the model: the
    pointer's contract is that the text is elsewhere in this window.

    Mutation: exempt every recognised lock in `_recap_shed` by dropping
    the `rendered_outside` test. FAIL -- the frozen lock's recap copy is
    pinned and the matched beliefs fall to zero. Restored: PASS.
    """
    out, err = _fire(tmp_path, monkeypatch, name="hits")
    assert "dropped" in err, err
    seen = set(re.findall(r'^  seen (.+?): ".*"$', out, re.MULTILINE))
    reached = {i for i in _element_ids(out) | seen if i.startswith("H")}
    assert reached, (
        "no prompt-matched belief reached the model on a fire whose recap "
        "keeps a reference lock; the #1564 ruling is that the recap never "
        "suppresses a hit"
    )
