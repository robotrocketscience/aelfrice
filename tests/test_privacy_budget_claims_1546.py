"""The SessionStart lane carries no budget, which is what a user is told (#1546).

`docs/user/PRIVACY.md` lists what bounds the context aelfrice hands the cloud
model, and a reader takes those bounds as promises. One of them was not true:
the page said a 1,500-token budget bounded the SessionStart hook's injection.
No such budget existed on that lane even before #1546 -- the block retrieves on
an empty query, so only L0 contributes and L0 is never trimmed (#379) -- and
#1546 deleted the constant and the parameter that were the claim's last trace
in the code. Nothing read the page, so the promise and the code could part
without anything going red.

**What is asserted here is the absence, because that is what this side can
assert.** "Carries no token budget" is a sentence, not a figure, and the fact
that makes it true is a signature. Re-adding a budget parameter to either frame
of the lane, or re-adding the deleted constant, turns these red and the message
names the page that would become false.

**The page itself is deliberately not read here, and that is a gap with a
number.** A test that reads `docs/user/PRIVACY.md` makes documentation
load-bearing for the suite, and `tests/test_ci_path_filter.py` forbids that:
`test_code_filter_covers_every_in_repo_package_the_suite_uses` fails the moment
a test under `tests/` reads `docs/`, because `docs/**` is not in `ci.yml`'s
`code` filter. Putting it there fails
`test_docs_only_changes_still_skip_the_suite`, and `e2e.yml`'s `paths` list
mirrors that filter
(`test_e2e_paths_cover_the_ci_code_filter`), so widening also puts the pytest
matrix and e2e's three-leg install matrix on every user-doc pull request --
reversing #413/#427 and #1420 §1 AC2. That is an operator call, not a side
effect of this branch.

Until it is made, the page's two published figures are guarded by review rather
than by CI. The #1469 derived-figure marker is the mechanism that would guard
them, and it does not work in this file: `scripts/check_derived_figures.py`
pairs backticks across the whole file, `PRIVACY.md` carries six fence lines, and
markers planted below them parse as absent while the gate reports green. That
hole is #1556; when it closes, the two figures can carry markers and the
`derived-figures` job -- which has no path filter and runs on every pull request
-- re-derives them from the shipped constants.
"""
from __future__ import annotations

import inspect

from aelfrice import hook


def test_no_budget_reaches_the_session_start_call_path() -> None:
    """Both frames of the lane, because either one could take the budget back.

    `session_start` is the hook entry point and
    `_retrieve_baseline_with_block` is the frame that used to spend the
    budget; #1546 deleted the parameter from both. A parameter restored at
    either one makes `docs/user/PRIVACY.md` false, so this fails and says so
    rather than leaving the page to drift again.
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
