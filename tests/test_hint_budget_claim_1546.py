"""The `MEMORY_BLOCK_HINT` docstring names the budget its cost is read against (#1546).

The paragraph used to say that #1526 "re-denominated `DEFAULT_HOOK_TOKEN_BUDGET`
and reserved the framing header out of it". Neither half happened. The
constant's own docstring says #1526 did not move it, the framing header's
docstring says #1526 measured the header and left it uncharged, and the budget
the UserPromptSubmit hook hands `retrieve()` is the bare constant: in
`user_prompt_submit`, `budget` is the caller's `token_budget` or
`DEFAULT_HOOK_TOKEN_BUDGET`, and `_retrieve` passes it on unchanged. The
paragraph now says that. Three tests hold it there.

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
prompt that the #674 shape gate refuses, the hook emits the locked
sub-block and the hint without calling `_retrieve`, so a paragraph that
says the hint always follows a spent budget is true of one path only. The
test fires that path, asserts the hint is emitted with no `_retrieve`
call, and then requires the paragraph to say `_retrieve` never runs there.
"""
from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any

import pytest

from aelfrice import hook
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
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


@pytest.mark.source_scan
def test_the_gate_skip_fire_emits_the_hint_without_retrieving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paragraph names the emitting path on which `_retrieve` never runs.

    "ok" is a session's first prompt and too short for the #674 shape gate,
    so the hook takes the gate-skip branch and emits the locked sub-block
    with the hint and no retrieval. The control fire, a long first prompt in
    a second session, has to call `_retrieve` once, so a wrapper the hook
    routes around fails there rather than passing on zero calls. Only then
    is the paragraph read; it is marked `source_scan` (#1746) for that read.
    """
    calls: list[int] = []
    real = hook._retrieve

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hook, "_retrieve", counting)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    db = tmp_path / "memory.db"
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
            )
        )
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))

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
