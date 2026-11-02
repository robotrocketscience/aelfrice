"""The `MEMORY_BLOCK_HINT` docstring names the budget its cost is read against (#1546).

The paragraph used to say that #1526 "re-denominated `DEFAULT_HOOK_TOKEN_BUDGET`
and reserved the framing header out of it". Neither half happened. The
constant's own docstring says #1526 did not move it, the framing header's
docstring says #1526 measured the header and left it uncharged, and the budget
the UserPromptSubmit hook hands `retrieve()` is the bare constant: in
`user_prompt_submit`, `budget` is the caller's `token_budget` or
`DEFAULT_HOOK_TOKEN_BUDGET`, and `_retrieve` passes it on unchanged. The
paragraph now says that. Four tests hold it there.

The first reads the docstring as text and fails on the two phrases that
carried the false claim. `tests/test_docs_truth_1511.py` records why a text
match is the exception and not the rule: a match breaks on rephrasing and
proves nothing about the tree. It is used here because the claim is about a
change that did not happen, and the tree cannot state a negative; the phrases
are the only trace a regression would leave. The read carries `source_scan`
(#1746).

The second checks the tree for what the paragraph now says: the budget the
hook passes is `DEFAULT_HOOK_TOKEN_BUDGET` itself, with nothing taken off for
the framing header or for the hint. It replaces `search_for_prompt`, the
callable `hook._retrieve` resolves, with a recorder and reads the
`token_budget` kwarg off the wire, the way
`tests/test_privacy_budget_claims_1546.py` reads the SessionStart lane's. A
reservation of either kind, added at the budget resolution or inside
`_retrieve`, changes that value and fails here.

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

Either one coming back is the regression this file exists to catch, and
nothing else in the paragraph is text-matched.
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
    for i, node in enumerate(body[:-1]):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
        if target != "MEMORY_BLOCK_HINT":
            continue
        nxt = body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant):
            if isinstance(nxt.value.value, str):
                return nxt.value.value
        pytest.fail(
            "MEMORY_BLOCK_HINT has no attribute docstring after it; the "
            "paragraph this file pins is gone"
        )
    pytest.fail("MEMORY_BLOCK_HINT is not assigned at module level in hook.py")


@pytest.mark.source_scan
def test_the_hint_docstring_claims_no_header_reservation() -> None:
    """The two phrases of the false sentence stay out of the paragraph.

    The vacuity guard first: the paragraph has to still be about the
    budget, or an empty replacement would pass the absence asserts.
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


def test_the_hook_hands_retrieve_the_bare_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`token_budget` on the wire is `DEFAULT_HOOK_TOKEN_BUDGET`, unreduced.

    This is the fact the paragraph now states and the one a reservation
    would change: the framing header's 126 estimated tokens and the hint's
    25 are both outside what `retrieve()` is given to spend. The recorder
    returns no hits, so the fire emits nothing and only the kwarg matters.
    The arity assert is the distinguishing arm: a resolution order that
    routed around the recorder fails there rather than passing on an empty
    list.
    """
    calls: list[dict[str, Any]] = []

    def recorder(store: Any, prompt: str, **kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(hook, "search_for_prompt", recorder)
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
    # The hook fails soft, so a recorder that raised would still return 0.
    assert "Traceback" not in serr.getvalue(), serr.getvalue()
    assert len(calls) == 1, (
        f"the recorder was called {len(calls)} times, so this test measured "
        "nothing: hook._lazy did not resolve search_for_prompt to it"
    )
    assert calls[0].get("token_budget") == hook.DEFAULT_HOOK_TOKEN_BUDGET, (
        f"the UserPromptSubmit hook passed token_budget="
        f"{calls[0].get('token_budget')!r} to retrieval, not "
        f"DEFAULT_HOOK_TOKEN_BUDGET={hook.DEFAULT_HOOK_TOKEN_BUDGET}. The "
        "MEMORY_BLOCK_HINT docstring says nothing is reserved out of that "
        "budget; change the paragraph in the same commit (#1546)."
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
