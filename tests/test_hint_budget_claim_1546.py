"""The `MEMORY_BLOCK_HINT` docstring names the budget its cost is read against (#1546).

The paragraph used to say that #1526 "re-denominated `DEFAULT_HOOK_TOKEN_BUDGET`
and reserved the framing header out of it". Neither half happened. The
constant's own docstring says #1526 did not move it, the framing header's
docstring says #1526 measured the header and left it uncharged, and the budget
the packer fills on a UserPromptSubmit fire is the bare constant: in
`user_prompt_submit`, `budget` is the caller's `token_budget` or
`DEFAULT_HOOK_TOKEN_BUDGET`, and nothing between there and the packer in
`retrieval.retrieve_with_tiers` takes any of it off. The paragraph now says
that. Seven tests hold it there.

The first reads the docstring as text and fails on the two phrases that
carried the false claim, or on the loss of the sentences that state the
true one, so a rewording of the false claim has to delete those too. `tests/test_docs_truth_1511.py` records why a text
match alone is weak: it breaks on rephrasing and proves nothing about the
tree. The first test is a text match and nothing more, because the claim is
about a change that did not happen, and the tree cannot state a negative;
the phrases are the only trace a regression would leave. The third and
fourth tests also match phrases in the paragraph, but only after each has
measured the tree fact the phrase states. Every test that reads the
paragraph carries `source_scan` (#1746).

The second checks the tree for what the paragraph now says: the packer fills
`DEFAULT_HOOK_TOKEN_BUDGET` itself, with nothing taken off for the framing
header or for the hint. It seeds more beliefs than the budget holds, each
priced at exactly `_SLOT_COST` by `_ups_belief_line_cost`, the cost the
packer charges on this lane. It wraps `hook._retrieve` and asserts the hits
that come back price to the whole budget. Reading the `token_budget` kwarg
at `search_for_prompt` instead would stop one hop short of `retrieve()`,
which resolves the budget again. A reservation of any size changes the sum
and fails here, whether `user_prompt_submit`, `_retrieve`,
`search_for_prompt`, `retrieve()`, the budget resolution or the packer
takes it.

The third covers the hint's second emitting path. On a session's first
prompt that the #674 shape gate refuses, the hook emits the session-start
block and the hint without calling `_retrieve`. What that block holds
depends on the store, the cwd and the cadence-resume cache, so the test
fires the path once per shape: a lock with a plain cwd, a lock with a git
cwd, which adds `<recent-work>`, a reference lock, which renders in an
`<aelfrice-locks-manifest>` block after `</session-start>` instead of in
`<locked>`, and an empty store with a cadence-resume cache, which leaves
no `<locked>` or `<core>` at all. Each fire asserts the hint is emitted
with no `_retrieve` call and records the sub-blocks ahead of it. The
paragraph then has to say `_retrieve` never runs there and name every one
of those sub-blocks.

The fourth covers the retrieving path. It wraps `_retrieve`, prices the
hits it returns with `_ups_belief_line_cost`, the cost the packer charges
on this lane, and asserts the hint is appended although the hits priced
under `DEFAULT_HOOK_TOKEN_BUDGET`. The paragraph then must not say
that budget was spent before the hint, and must cite
`docs/design/budget_discriminability_k0_result.md`, where `binds_on` is
`pool` on 7 of 7 queries on every lane.

The fifth covers the retrieving fire that keeps no hit. The hint sits in
the `if hits:` branch of `user_prompt_submit`, so a first prompt whose
`_retrieve` returns nothing emits no hint and drops the session-start
build, even with a fresh cadence-resume cache. The test fires that path,
asserts the output holds neither, and checks with a gate-skip control
that the cache and the hint do render elsewhere. The paragraph then has
to scope the retrieving sentence to a fire that keeps at least one hit
and say what the zero-hit fire emits.

The sixth covers the gate-skip fire whose session-start build is
empty. The hint sits under `if session_start_block` in the gate-skip
branch, so with an empty store and no cadence-resume cache that fire
emits nothing. The test fires that path, asserts the output is empty,
and checks with a control fire after the cache is written that the
hint does render. The paragraph then has to say the empty build
emits no hint.

The seventh covers the retrieving fire whose hits the filters after
`_retrieve` drop. `_retrieve` returns one hit tagged with another
project context and `_filter_by_project_context` removes it, so the
fire emits no hint although retrieval found something. A control fire
with the context unset has to emit the hint. The paragraph then has
to say a fire whose filters drop every hit emits no hint.
"""
from __future__ import annotations

import ast
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aelfrice import hook
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

_FALSE_PHRASES = ("re-denominated", "reserved the framing header")
"""The two phrases the false sentence was built from.

Either one coming back is the regression the first test exists to catch.
The third and fourth tests match other phrases, each after measuring the
fact the phrase states.
"""

_TRUE_CLAIMS = (
    "#1526 did not move `DEFAULT_HOOK_TOKEN_BUDGET` and left the framing "
    "header uncharged",
    "The packer never charges the hint",
)
"""The sentences that state what #1526 did and did not do to the budget.

The first test asserts each is present, so the false claim cannot come
back reworded while these still stand beside it unchanged.
"""


def _hint_docstring() -> str:
    """The string literal that follows the `MEMORY_BLOCK_HINT` assignment.

    An attribute docstring is a bare expression statement, so the runtime
    keeps no `__doc__` for it and the only way to read it is from the
    source. The module is parsed rather than scanned by regex so the text
    is the literal's value, with the quotes and escapes resolved.
    """
    tree = ast.parse(Path(hook.__file__).read_text(encoding="utf-8"))
    body = tree.body
    found = False
    doc: str | None = None
    for i, node in enumerate(body[:-1]):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
        if target != "MEMORY_BLOCK_HINT":
            continue
        found = True
        nxt = body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant):
            if isinstance(nxt.value.value, str):
                doc = nxt.value.value
        break
    if not found:
        pytest.fail("MEMORY_BLOCK_HINT is not assigned at module level in hook.py")
    if doc is None:
        pytest.fail(
            "MEMORY_BLOCK_HINT has no attribute docstring after it; the "
            "paragraph this file pins is gone"
        )
    return doc


@pytest.mark.source_scan
def test_the_hint_docstring_claims_no_header_reservation() -> None:
    """The two phrases of the false sentence stay out of the paragraph.

    The vacuity guard first: the paragraph has to still be about the
    budget, or an empty replacement would pass the absence asserts.
    The positive claims last: the paragraph has to say #1526 left the
    budget and the header alone and that the packer never charges the
    hint, so a false claim reworded around the two phrases still fails
    unless it also deletes those sentences.
    Whitespace runs are collapsed first, so a phrase wrapped across a line
    break still matches.
    """
    doc = " ".join(_hint_docstring().split())
    assert "DEFAULT_HOOK_TOKEN_BUDGET" in doc, (
        "the MEMORY_BLOCK_HINT docstring no longer names "
        "DEFAULT_HOOK_TOKEN_BUDGET, so the absence asserts below check nothing"
    )
    for phrase in _FALSE_PHRASES:
        assert phrase not in doc, (
            f"the MEMORY_BLOCK_HINT docstring says {phrase!r} again. #1526 "
            "did not move DEFAULT_HOOK_TOKEN_BUDGET and left the framing "
            "header uncharged; see the constant's and the header's own "
            "docstrings (#1546)."
        )
    for claim in _TRUE_CLAIMS:
        assert claim in doc, (
            f"the MEMORY_BLOCK_HINT docstring no longer says {claim!r}. "
            "#1526 did not move DEFAULT_HOOK_TOKEN_BUDGET, left the framing "
            "header uncharged, and the packer never charges the hint (#1546)."
        )


def _payload(
    cwd: Path,
    session_id: str = "s1",
    # Long enough to pass the #674 prompt-shape gate, so the hook reaches
    # retrieval instead of the gate-skip branch.
    prompt: str = "tell me about bananas",
) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(cwd),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


_SLOT_COST = 100
"""What `_ups_belief_line_cost` charges each belief the packer test seeds.

`DEFAULT_HOOK_TOKEN_BUDGET` is a whole number of these, so the budget holds
an exact count of beliefs and a reservation of even one token leaves room
for one fewer.
"""


def _belief_costing(i: int, cost: int) -> Belief:
    """An unlocked banana belief that `_ups_belief_line_cost` prices at `cost`.

    The filler words carry the index, so no two contents share more than
    the leading phrase and no dedup or clustering lane can fold them.
    """
    filler = " ".join(f"q{i:02d}r{j:03d}" for j in range(400))
    found: Belief | None = None
    for pad in range(len(filler) + 1):
        b = Belief(
            id=f"F{i:02d}",
            content=f"banana note {i:02d} {filler[:pad]}",
            content_hash=f"h_F{i:02d}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-04-26T00:00:00Z",
            last_retrieved_at=None,
        )
        if hook._ups_belief_line_cost(b) == cost:
            found = b
            break
    if found is None:
        pytest.fail(f"no padding of belief {i} costs exactly {cost} tokens")
    return found


def test_the_packer_fills_the_bare_hook_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hits a fire packs price to `DEFAULT_HOOK_TOKEN_BUDGET`, unreduced.

    This is the fact the paragraph now states and the one a reservation
    would change: the framing header's 126 estimated tokens and the hint's
    25 are both outside what the packer spends. The store holds five more
    beliefs than the budget fits, each priced at `_SLOT_COST`, so the
    packer stops on the budget, not on the pool. The arity and
    "fewer than seeded" asserts are the distinguishing arm: a fire that
    skipped `_retrieve`, or a pool that ran out before the budget did,
    fails there rather than on the sum.
    """
    budget = hook.DEFAULT_HOOK_TOKEN_BUDGET
    assert budget % _SLOT_COST == 0, (
        f"DEFAULT_HOOK_TOKEN_BUDGET={budget} is not a whole number of "
        f"{_SLOT_COST}-token beliefs; pick a _SLOT_COST that divides it"
    )
    seeded = budget // _SLOT_COST + 5
    packed: list[list[Belief]] = []
    real = hook._retrieve

    def capturing(*args: Any, **kwargs: Any) -> Any:
        hits = real(*args, **kwargs)
        packed.append(list(hits))
        return hits

    monkeypatch.setattr(hook, "_retrieve", capturing)
    monkeypatch.delenv("AELFRICE_RETRIEVAL_TOKEN_BUDGET", raising=False)
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        for i in range(seeded):
            store.insert_belief(_belief_costing(i, _SLOT_COST))
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    serr = io.StringIO()

    rc = hook.user_prompt_submit(
        stdin=io.StringIO(_payload(tmp_path)), stdout=sout, stderr=serr,
    )

    assert rc == 0
    # The hook fails soft, so a wrapper that raised would still return 0.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    assert len(packed) == 1, (
        f"_retrieve ran {len(packed)} times, not once, so this test "
        "measured nothing about the packer"
    )
    hits = packed[0]
    assert 0 < len(hits) < seeded, (
        f"the packer returned {len(hits)} of {seeded} seeded beliefs, so "
        "the pool, not the budget, ended retrieval and the sum below "
        "measures nothing"
    )
    assert {hook._ups_belief_line_cost(b) for b in hits} == {_SLOT_COST}
    spent = sum(hook._ups_belief_line_cost(b) for b in hits)
    assert spent == budget, (
        f"the packer filled {spent} tokens of "
        f"DEFAULT_HOOK_TOKEN_BUDGET={budget}. The MEMORY_BLOCK_HINT "
        "docstring says nothing is reserved out of that budget and the "
        "packer never charges the hint; change the paragraph in the same "
        "commit (#1546)."
    )

_GATE_SKIP_TAGS = (
    "cadence-resume", "locked", "core", "aelfrice-locks-manifest",
    "recent-work",
)
"""Every sub-block the gate-skip branch can put ahead of the hint."""


def _seed_lock(db: Path, tier: str = LOCK_TIER_FROZEN) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            Belief(
                id="L1",
                content="always use uv for python packaging",
                content_hash="h_L1",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_USER,
                locked_at="2026-04-26T00:00:00Z",
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
                lock_tier=tier,
            )
        )
    finally:
        store.close()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True,
        timeout=30,
    )


def _git_checkout(path: Path) -> None:
    """A repository on branch `work` with one commit, so `<recent-work>` renders."""
    path.mkdir()
    _git(path, "init", "-q", "-b", "work")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "a.txt").write_text("a", encoding="utf-8")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "feat: seed")


def _write_cadence_resume_cache() -> None:
    """A fresh cache where the hook reads it, beside the scratch store."""
    cache = hook._cadence_resume_cache_path()
    assert cache is not None
    cache.write_text(
        json.dumps(
            {
                "ts": "2026-10-27T00:00:00Z",
                "session_id": "previous",
                "policy": "p1",
                "body": "pick up the packaging work",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.timeout(60)
@pytest.mark.source_scan
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("lock", {"locked", "core"}),
        ("lock-in-git", {"locked", "core", "recent-work"}),
        ("reference-lock", {"locked", "core", "aelfrice-locks-manifest"}),
        ("cadence-only", {"cadence-resume"}),
    ],
)
def test_the_gate_skip_fire_emits_the_hint_without_retrieving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    case: str, expected: set[str],
) -> None:
    """The paragraph names the gate-skip path and what the hint follows there.

    "ok" is a session's first prompt and too short for the #674 shape gate,
    so the hook takes the gate-skip branch and emits the session-start
    block with the hint and no retrieval. Each case asserts the exact set
    of sub-blocks ahead of the hint, so a case that stopped producing the
    shape it is named for fails before the paragraph is read. The control
    fire, a long first prompt in a second session, has to call `_retrieve`
    once, so a wrapper the hook routes around fails there rather than
    passing on zero calls. Only then is the paragraph read; it is marked
    `source_scan` (#1746) for that read.
    """
    calls: list[int] = []
    real = hook._retrieve

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hook, "_retrieve", counting)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    cwd = tmp_path / "cwd"
    if case == "lock-in-git":
        _git_checkout(cwd)
    else:
        cwd.mkdir()
    db = tmp_path / "memory.db"
    if case == "cadence-only":
        MemoryStore(str(db)).close()
    elif case == "reference-lock":
        _seed_lock(db, LOCK_TIER_REFERENCE)
    else:
        _seed_lock(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    if case == "cadence-only":
        _write_cadence_resume_cache()

    def fire(session_id: str, prompt: str) -> str:
        sout = io.StringIO()
        serr = io.StringIO()
        rc = hook.user_prompt_submit(
            stdin=io.StringIO(_payload(cwd, session_id, prompt)),
            stdout=sout,
            stderr=serr,
        )
        assert rc == 0
        assert "Traceback" not in serr.getvalue(), serr.getvalue()
        return sout.getvalue()

    skipped = fire("gate", "ok")
    assert len(calls) == 0, (
        f"the gate-skip fire called _retrieve {len(calls)} times; if the "
        "#674 gate no longer skips retrieval, the MEMORY_BLOCK_HINT "
        "paragraph's gate-skip sentence is wrong (#1546)"
    )
    assert hook.MEMORY_BLOCK_HINT in skipped, (
        "the gate-skip first-prompt fire emitted no hint, so this test "
        f"measured nothing: {skipped!r}"
    )
    ahead = skipped[: skipped.index(hook.MEMORY_BLOCK_HINT)]
    seen = {tag for tag in _GATE_SKIP_TAGS if f"<{tag}" in ahead}
    assert seen == expected, (
        f"the {case!r} gate-skip fire put {sorted(seen)} ahead of the hint, "
        f"not {sorted(expected)}: {skipped!r}"
    )
    if case == "reference-lock":
        # The lock left `<locked>` for a `ref` line, and that line sits in
        # the manifest after the sub-block closes, not inside it.
        assert '<belief id="L1"' not in ahead, skipped
        assert "ref L1:" in ahead, skipped
        assert ahead.index("</session-start>") < ahead.index(
            "<aelfrice-locks-manifest"
        ), skipped
    fire("retrieve", "how should I manage python packaging with uv here")
    assert len(calls) == 1, (
        f"the retrieving control fire called _retrieve {len(calls)} times, "
        "not once, so the wrapper is not on the path the hook takes"
    )
    doc = " ".join(_hint_docstring().split())
    assert "`_retrieve` never runs" in doc, (
        "the MEMORY_BLOCK_HINT docstring describes only the retrieving path. "
        "On a gate-skip first prompt the hint is emitted with no _retrieve "
        "call, so no budget was spent before it; say so (#1546)."
    )
    for tag in sorted(seen):
        assert f"`<{tag}>`" in doc, (
            f"on a gate-skip first prompt the hint can follow <{tag}>, and "
            "the MEMORY_BLOCK_HINT docstring does not name it (#1546)"
        )
    if "aelfrice-locks-manifest" in seen:
        assert "`<aelfrice-locks-manifest>` block after `</session-start>`" in doc, (
            "on a gate-skip first prompt a reference lock's manifest block "
            "comes after </session-start>, and the MEMORY_BLOCK_HINT "
            "docstring does not say where (#1546)"
        )
    if "locked" not in seen:
        assert "no `<locked>` or `<core>`" in doc, (
            "a gate-skip first prompt can emit the hint with no <locked> or "
            "<core> ahead of it, and the MEMORY_BLOCK_HINT docstring does "
            "not say so (#1546)"
        )


@pytest.mark.timeout(60)
@pytest.mark.source_scan
def test_a_gate_skip_fire_with_an_empty_build_emits_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate-skip first prompt whose session-start build is empty emits nothing.

    The store is empty and no cadence-resume cache exists, so the build
    returns nothing and the gate-skip branch has no block to write the hint
    into. The control fire, a gate-skip first prompt in a second session
    after the cache is written, has to emit the hint, so a branch that no
    longer reaches the write fails there rather than passing the absence
    assert. Neither fire may call `_retrieve`. Only then is the paragraph
    read; it is marked `source_scan` (#1746) for that read.
    """
    calls: list[int] = []
    real = hook._retrieve

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hook, "_retrieve", counting)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    monkeypatch.setenv("AELFRICE_DB", str(db))

    def fire(session_id: str) -> str:
        sout = io.StringIO()
        serr = io.StringIO()
        rc = hook.user_prompt_submit(
            stdin=io.StringIO(_payload(tmp_path, session_id, "ok")),
            stdout=sout,
            stderr=serr,
        )
        assert rc == 0
        assert "Traceback" not in serr.getvalue(), serr.getvalue()
        return sout.getvalue()

    empty = fire("empty")
    assert empty == "", (
        "a gate-skip first prompt with an empty session-start build emitted "
        f"output; the MEMORY_BLOCK_HINT docstring says it emits none (#1546): "
        f"{empty!r}"
    )
    _write_cadence_resume_cache()
    control = fire("control")
    assert hook.MEMORY_BLOCK_HINT in control, (
        "the gate-skip control with a cadence-resume cache emitted no hint, "
        f"so the absence assert above checks nothing: {control!r}"
    )
    assert calls == [], (
        f"a gate-skip fire called _retrieve {len(calls)} times, so these "
        "fires did not take the gate-skip branch"
    )
    doc = " ".join(_hint_docstring().split())
    assert (
        "If the session-start build returned nothing, that fire emits no "
        "block and no hint." in doc
    ), (
        "the MEMORY_BLOCK_HINT docstring no longer says a gate-skip first "
        "prompt with an empty session-start build emits no hint (#1546)"
    )


_K0_RESULT = "docs/design/budget_discriminability_k0_result.md"
"""The measurement the paragraph cites for where retrieval stops."""


@pytest.mark.source_scan
def test_a_retrieving_fire_appends_the_hint_under_an_unspent_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hint follows `_retrieve` whatever share of the budget the hits used.

    One unlocked belief matches the prompt, so the pool prices far below
    `DEFAULT_HOOK_TOKEN_BUDGET` and retrieval ends on the pool. The wrapper
    prices the returned hits with `_ups_belief_line_cost`, the cost the
    packer charges on this lane. The arity and nonzero asserts are the
    distinguishing arm: a fire that skipped `_retrieve` or returned no hits
    would make "under budget" true of nothing. Only then is the paragraph
    read; it is marked `source_scan` (#1746) for that read and for the
    citation's existence check.
    """
    spent: list[int] = []
    real = hook._retrieve

    def pricing(*args: Any, **kwargs: Any) -> Any:
        hits = real(*args, **kwargs)
        spent.append(sum(hook._ups_belief_line_cost(b) for b in hits))
        return hits

    monkeypatch.setattr(hook, "_retrieve", pricing)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            Belief(
                id="F1",
                content="banana is a yellow fruit",
                content_hash="h_F1",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    serr = io.StringIO()

    rc = hook.user_prompt_submit(
        stdin=io.StringIO(_payload(tmp_path)), stdout=sout, stderr=serr,
    )

    assert rc == 0
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    assert len(spent) == 1, (
        f"_retrieve ran {len(spent)} times, not once, so this test measured "
        "nothing about the retrieving path"
    )
    assert 0 < spent[0] < hook.DEFAULT_HOOK_TOKEN_BUDGET, (
        f"the hits priced {spent[0]} tokens against "
        f"DEFAULT_HOOK_TOKEN_BUDGET={hook.DEFAULT_HOOK_TOKEN_BUDGET}; the "
        "fixture no longer retrieves a pool that ends under the budget"
    )
    assert hook.MEMORY_BLOCK_HINT in sout.getvalue(), (
        "a retrieving fire with hits emitted no hint: "
        f"{sout.getvalue()!r}"
    )
    doc = " ".join(_hint_docstring().split())
    assert "spent that budget" not in doc, (
        f"retrieval priced {spent[0]} of {hook.DEFAULT_HOOK_TOKEN_BUDGET} "
        "tokens and the hint was still appended, yet the MEMORY_BLOCK_HINT "
        "docstring says the budget was spent before the hint (#1546)"
    )
    assert f"`{_K0_RESULT}`" in doc, (
        "the MEMORY_BLOCK_HINT docstring no longer cites the K0 result for "
        "where retrieval stops (#1546)"
    )
    root = Path(__file__).resolve().parents[1]
    assert (root / _K0_RESULT).is_file(), (
        f"the MEMORY_BLOCK_HINT docstring cites {_K0_RESULT}, which is gone"
    )


@pytest.mark.timeout(60)
@pytest.mark.source_scan
def test_a_retrieving_fire_with_no_hit_emits_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fire that retrieves and keeps no hit emits nothing, hint included.

    The store holds one unlocked belief the prompt does not match, and a
    fresh cadence-resume cache sits beside it, so the first prompt's
    session-start build returns a `<cadence-resume>` block. The wrapper
    counts the hits `_retrieve` returns. The arity and zero-hit asserts are
    the distinguishing arm: a fire that skipped `_retrieve`, or one that
    found a hit, says nothing about this path. The control fire, a
    gate-skip first prompt in a second session, has to emit that block
    and the hint, so an absent cache or a broken hint fails there rather
    than passing the absence asserts. Only then is the paragraph read; it
    is marked `source_scan` (#1746) for that read.
    """
    returned: list[int] = []
    real = hook._retrieve

    def counting(*args: Any, **kwargs: Any) -> Any:
        hits = real(*args, **kwargs)
        returned.append(len(hits))
        return hits

    monkeypatch.setattr(hook, "_retrieve", counting)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            Belief(
                id="Z1",
                content="zebra stripes are black",
                content_hash="h_Z1",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
            )
        )
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _write_cadence_resume_cache()

    def fire(session_id: str, prompt: str) -> str:
        sout = io.StringIO()
        serr = io.StringIO()
        rc = hook.user_prompt_submit(
            stdin=io.StringIO(_payload(tmp_path, session_id, prompt)),
            stdout=sout,
            stderr=serr,
        )
        assert rc == 0
        assert "Traceback" not in serr.getvalue(), serr.getvalue()
        return sout.getvalue()

    empty = fire("s1", "tell me about bananas")
    assert returned == [0], (
        f"_retrieve returned {returned} hit counts, not one fire with zero "
        "hits, so this test measured nothing about the zero-hit path"
    )
    assert hook.MEMORY_BLOCK_HINT not in empty, (
        "a retrieving fire with no hit emitted the hint; the "
        "MEMORY_BLOCK_HINT docstring says it emits none (#1546): "
        f"{empty!r}"
    )
    assert "<cadence-resume" not in empty, (
        "a zero-hit first prompt emitted the cadence-resume block; the "
        f"MEMORY_BLOCK_HINT docstring says it emits none (#1546): {empty!r}"
    )
    control = fire("gate", "ok")
    assert returned == [0], (
        "the gate-skip control fire called _retrieve, so it is not the "
        "path the control needs"
    )
    assert "<cadence-resume" in control, (
        "the gate-skip control emitted no cadence-resume block, so the "
        f"absence assert above checks nothing: {control!r}"
    )
    assert hook.MEMORY_BLOCK_HINT in control, (
        "the gate-skip control emitted no hint, so the absence assert "
        f"above checks nothing: {control!r}"
    )
    doc = " ".join(_hint_docstring().split())
    assert "On a fire that retrieves, the hint" not in doc, (
        "the MEMORY_BLOCK_HINT docstring says every retrieving fire gets "
        "the hint; a fire that keeps no hit emits none (#1546)"
    )
    for phrase in (
        "retrieves and keeps at least one hit, the hint is appended",
        "retrieves and keeps no hit",
        "so it emits no hint",
        "a fresh `<cadence-resume>` block included",
    ):
        assert phrase in doc, (
            f"the MEMORY_BLOCK_HINT docstring no longer says {phrase!r}; "
            "a retrieving fire with no hit emits no hint and, on a first "
            "prompt, no session-start block (#1546)"
        )


@pytest.mark.timeout(60)
@pytest.mark.source_scan
def test_a_retrieving_fire_whose_filters_drop_every_hit_emits_no_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fire whose hits the post-retrieval filters drop emits no hint.

    The store holds one banana belief tagged with another project context,
    and `AELFRICE_PROJECT_CONTEXT` names a different one, so `_retrieve`
    returns the hit and `_filter_by_project_context` drops it before the
    `if hits:` branch. The wrapper counts the hits `_retrieve` returns. The
    one-hit assert is the distinguishing arm: unlike the zero-hit test, the
    hit reaches the hook and only a filter removes it. The control fire, a
    second session with the context unset, has to emit the hint, so a
    broken fixture fails there rather than passing the absence assert. Only
    then is the paragraph read; it is marked `source_scan` (#1746) for that
    read.
    """
    returned: list[int] = []
    real = hook._retrieve

    def counting(*args: Any, **kwargs: Any) -> Any:
        hits = real(*args, **kwargs)
        returned.append(len(hits))
        return hits

    monkeypatch.setattr(hook, "_retrieve", counting)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            Belief(
                id="F1",
                content="banana is a yellow fruit",
                content_hash="h_F1",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at="2026-04-26T00:00:00Z",
                last_retrieved_at=None,
                project_context="repo-b",
            )
        )
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))

    def fire(session_id: str) -> str:
        sout = io.StringIO()
        serr = io.StringIO()
        rc = hook.user_prompt_submit(
            stdin=io.StringIO(
                _payload(tmp_path, session_id, "tell me about bananas")
            ),
            stdout=sout,
            stderr=serr,
        )
        assert rc == 0
        assert "Traceback" not in serr.getvalue(), serr.getvalue()
        return sout.getvalue()

    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "repo-a")
    filtered = fire("filtered")
    assert returned == [1], (
        f"_retrieve returned {returned} hit counts, not one fire with one "
        "hit, so the filter had nothing to drop"
    )
    assert hook.MEMORY_BLOCK_HINT not in filtered, (
        "a retrieving fire whose filters dropped every hit emitted the hint; "
        f"the MEMORY_BLOCK_HINT docstring says it emits none (#1546): "
        f"{filtered!r}"
    )
    monkeypatch.delenv("AELFRICE_PROJECT_CONTEXT")
    control = fire("control")
    assert returned == [1, 1], returned
    assert hook.MEMORY_BLOCK_HINT in control, (
        "the unfiltered control fire emitted no hint, so the absence assert "
        f"above checks nothing: {control!r}"
    )
    doc = " ".join(_hint_docstring().split())
    assert "or the filters after it dropped them all" in doc, (
        "the MEMORY_BLOCK_HINT docstring no longer says a fire whose filters "
        "drop every hit emits no hint (#1546)"
    )
