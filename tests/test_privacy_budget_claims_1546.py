"""`docs/user/PRIVACY.md`'s budget mitigations are held to the code (#1546).

That page lists what bounds the context aelfrice hands the cloud model, and a
reader takes those bounds as promises. One of them was not true: the page said
a 1,500-token budget bounded the SessionStart hook's injection. No such budget
existed on that lane even before #1546 -- the block retrieves on an empty
query, so only L0 contributes and L0 is never trimmed (#379) -- and #1546
deleted the constant and the parameter that carried the claim's last trace in
the code. Nothing in `tests/` read the page, so the promise and the code could
part without anything going red.

The other privacy promises have guards (`test_no_network_on_retrieval_path_
1240.py` is the closest relative). These are this page's budget promises, and
they are asserted in the two forms the page makes them in:

  * **A figure** -- 1,500 and 2,400 -- checked against the shipped constant
    rather than against a literal transcribed here, which would be a
    tautology. Editing either constant without editing the page is red.
  * **An absence** -- "carries no token budget" -- which is a signature, not a
    number. Re-adding a budget parameter anywhere on the SessionStart call
    path is red, and the failure names the sentence that would become false.

The assertions are on the exact sentences, so rewording the page fails here
too. That is the intended cost: the sentence and its guard move together, and
a reviewer is told which paragraph to re-check.

Why the `#1469` derived-figure marker is not the mechanism: markers in this
file are not read. `scripts/check_derived_figures.py` blanks inline-code spans
before parsing, pairing backticks across the whole file, and a ``` fence line
carries an odd one. `PRIVACY.md` has six fence lines, and two markers planted
in it parsed as zero while the gate reported green over prose that contradicted
them (removing the fence lines from a copy makes both parse). That is a hole in
the gate rather than in this page, and it is reported rather than worked
around.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from aelfrice import hook
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET

PRIVACY = Path(__file__).resolve().parents[1] / "docs" / "user" / "PRIVACY.md"

# The one paragraph these claims live in. Scoping the search to it keeps a
# match somewhere else in the page from standing in for the promise.
_SECTION_HEADING = "## What aelfrice does not control"


def _mitigations() -> str:
    """The text of the section that lists what bounds the injection."""
    text = PRIVACY.read_text(encoding="utf-8")
    assert _SECTION_HEADING in text, (
        f"{PRIVACY} no longer has a {_SECTION_HEADING!r} section; the budget "
        "promises this module guards were in it"
    )
    body = text.split(_SECTION_HEADING, 1)[1]
    return body.split("\n## ", 1)[0]


def test_the_published_hook_budget_is_the_shipped_hook_budget() -> None:
    """1,500 in the page is `hook.DEFAULT_HOOK_TOKEN_BUDGET`, not a literal."""
    want = f"{hook.DEFAULT_HOOK_TOKEN_BUDGET:,} tokens for the UserPromptSubmit hook"
    assert want in _mitigations(), (
        f"docs/user/PRIVACY.md must publish {want!r}. The constant is "
        f"{hook.DEFAULT_HOOK_TOKEN_BUDGET}; the page says otherwise, and a "
        "reader takes that number for a bound on what reaches the model."
    )


def test_the_published_library_budget_is_the_shipped_library_budget() -> None:
    """2,400 in the page is `retrieval.DEFAULT_TOKEN_BUDGET`."""
    want = f"{DEFAULT_TOKEN_BUDGET:,} tokens for the library retrieval API"
    assert want in _mitigations(), (
        f"docs/user/PRIVACY.md must publish {want!r}. The constant is "
        f"{DEFAULT_TOKEN_BUDGET}; the page says otherwise."
    )


def test_the_page_says_the_session_start_block_carries_no_budget() -> None:
    """The absence is promised in words, so the words have to be there.

    Without this arm the signature arm below passes over a page that has gone
    back to promising a budget: an absence in the code proves nothing about
    what the page claims.
    """
    section = _mitigations()
    assert "carries no token budget" in section, (
        "docs/user/PRIVACY.md must say the SessionStart baseline block "
        "carries no token budget. The code has none to carry -- "
        f"session_start{inspect.signature(hook.session_start)} -- and the "
        "page is where a user reads it."
    )


def test_no_budget_reaches_the_session_start_call_path() -> None:
    """The code half of the same promise, at both frames of the lane.

    `session_start` is the hook entry point and
    `_retrieve_baseline_with_block` is the frame that used to spend the
    budget; #1546 deleted the parameter from both. A parameter restored at
    either one makes the page's sentence false, so this fails and says so
    rather than leaving the doc to drift again.
    """
    for func in (hook.session_start, hook._retrieve_baseline_with_block):
        offenders = [
            name
            for name in inspect.signature(func).parameters
            if "budget" in name
        ]
        assert not offenders, (
            f"{func.__name__}{inspect.signature(func)} takes {offenders}, but "
            "docs/user/PRIVACY.md tells users the SessionStart block carries "
            "no token budget. Update the page in the same change, or the "
            "promise is false (#1546)."
        )


def test_the_deleted_constant_has_not_come_back() -> None:
    """`DEFAULT_SESSION_START_TOKEN_BUDGET` is what the page used to describe.

    A re-added constant would not fail the signature arm above -- nothing
    forces a caller to pass it -- but it is the shape a reader of the page
    would next mistake for a bound.
    """
    assert not hasattr(hook, "DEFAULT_SESSION_START_TOKEN_BUDGET"), (
        "hook.DEFAULT_SESSION_START_TOKEN_BUDGET is back. #1546 deleted it "
        "because no value of it could change the block it named; if the lane "
        "now has something a budget can trim, docs/user/PRIVACY.md has to say "
        "so."
    )
