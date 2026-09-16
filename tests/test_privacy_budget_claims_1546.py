"""The SessionStart lane carries no budget, which is what a user is told (#1546).

`docs/user/PRIVACY.md` lists what bounds the context aelfrice hands the cloud
model, and a reader takes those bounds as promises. One of them was not true:
the page said a 1,500-token budget bounded the SessionStart hook's injection.
No such budget existed on that lane even before #1546 -- the block retrieves on
an empty query, so only L0 contributes and L0 is never trimmed (#379) -- and
#1546 deleted the constant and the parameter that were the claim's last trace
in the code. Nothing read the page, so the promise and the code could part
without anything going red.

**Two of the three tests here assert an absence, and the third asserts the
call, because a signature is not the claim.** `docs/user/CONFIG.md` now tells a
user that `[retrieval] token_budget` reaches this lane, and the mechanism it
names is that the caller passes no budget of its own for the resolver's
explicit tier to take. A signature cannot hold that: restoring
`token_budget=1500` to the `retrieve()` call inside
`_retrieve_baseline_with_block` leaves both signatures clean, re-shadows the
TOML key and makes the page false. So the last test spies the `retrieve`
callable the lane actually resolves and asserts both halves at once -- that
`token_budget` is absent from the kwargs, and that the resolver fed exactly
what the caller passed lands on the value in the file.

**The page itself is deliberately not read here, and that is a gap with a
number.** A test that reads `docs/user/PRIVACY.md` makes documentation
load-bearing for the suite, and `tests/test_ci_path_filter.py` forbids that:
`test_code_filter_covers_every_in_repo_package_the_suite_uses` fails the moment
a test under `tests/` reads `docs/` with the path spelled inline, because
`docs/**` is not in `ci.yml`'s `code` filter.

That guard is spelling-dependent, and the spelling it misses is not the way
out. Its scanner is two literal regexes, one anchored on `parents[1]` and one
on `parent.parent`, each requiring a quoted first component immediately after
the slash. Binding the repo root to a name first and joining the directory
onto that name -- which is how `test_ci_path_filter.py` writes its own `_REPO`
-- therefore reads the page with the module green. What that would buy is
nothing: the same filter decides whether the suite runs at all, so a guard on
the page hidden from it would be skipped on exactly the docs-only pull request
that changes the page. That is the #1160 failure mode -- a required check
reporting success from an `echo`, having run nothing.

So the way out is the filter, and it is an operator call rather than a side
effect of this branch. Putting `docs/**` in it fails
`test_docs_only_changes_still_skip_the_suite`, and `e2e.yml`'s `paths` list
mirrors that filter (`test_e2e_paths_cover_the_ci_code_filter`), so widening
also puts the pytest matrix and e2e's three-leg install matrix on every
user-doc pull request -- reversing #413/#427 and #1420 §1 AC2.

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
from pathlib import Path
from typing import Any

import pytest

from aelfrice import hook
# The module object, not its names: `monkeypatch.setattr` below has to install
# the recorder on the module `hook._lazy` resolves, and importing the same
# module both ways trips CodeQL's py/import-and-import-from.
from aelfrice import retrieval
from aelfrice.store import MemoryStore


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


def test_the_lane_hands_retrieve_no_budget_so_the_toml_key_reaches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The call, not the signature: this is what CONFIG.md's claim rests on.

    `docs/user/CONFIG.md` says `[retrieval] token_budget` now reaches the
    `SessionStart` hook "because the caller no longer shadows it". The
    resolver ranks an explicit keyword argument above TOML, so that sentence
    is true only while the `retrieve()` call inside
    `_retrieve_baseline_with_block` passes no `token_budget` -- a fact about
    one call site, which the two signature guards above cannot see. Restoring
    `token_budget=1500` to that call leaves both signatures clean and makes
    the page false.

    So the callable the lane resolves is replaced with a recorder and the
    kwargs it receives are read off the wire. `hook._lazy("retrieve")` prefers
    a binding in `hook`'s own globals and otherwise imports
    `aelfrice.retrieval`, which is where the recorder is installed; the arity
    assert below is the distinguishing arm, so a resolution order that routed
    around the recorder fails here instead of passing on an empty list.

    The resolution assert closes the composition, and it runs *before* the
    absence assert rather than after it. `[retrieval] token_budget = 77` sits
    on disk with the environment variable cleared, and the resolver is fed
    exactly what the caller passed: 77 on the shipped code, 1500 under the
    mutation. Behind the absence assert it would be dead weight -- reached
    only once the argument had been proved absent, so reducing to
    `resolve_token_budget(None) == 77`, and never reached at all on the
    mutation that restores the argument. In this order the mutation's first
    failure names both numbers.
    """
    calls: list[dict[str, Any]] = []

    def recorder(store: Any, query: str, **kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(retrieval, "retrieve", recorder)
    monkeypatch.setattr(
        hook, "_open_store", lambda: MemoryStore(str(tmp_path / "memory.db"))
    )
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\ntoken_budget = 77\n", encoding="utf-8"
    )
    # `resolve_token_budget` walks up from the process's cwd, and the
    # environment variable outranks every other tier, so both have to be set
    # before the call for the TOML tier to be the one under test.
    monkeypatch.delenv(retrieval.ENV_RETRIEVAL_TOKEN_BUDGET, raising=False)
    monkeypatch.chdir(tmp_path)

    hook._retrieve_baseline_with_block()

    assert len(calls) == 1, (
        "the recorder was called "
        f"{len(calls)} times, so this test measured nothing: "
        "hook._lazy did not resolve aelfrice.retrieval.retrieve"
    )
    # The resolution assert runs first, and the order is the whole point of
    # it: behind an `assert "token_budget" not in calls[0]` this line would
    # only ever be reached with the argument already proved absent, so it
    # would reduce to `resolve_token_budget(None) == 77` -- the resolver
    # returning the literal this test wrote to disk four lines ago, and never
    # reached at all under the mutation it exists to catch.
    resolved = retrieval.resolve_token_budget(calls[0].get("token_budget"))
    assert resolved == 77, (
        f"with [retrieval] token_budget = 77 on disk and the caller passing "
        f"token_budget={calls[0].get('token_budget')!r}, the lane resolves to "
        f"{resolved}. docs/user/CONFIG.md tells users that key reaches this "
        "lane because the caller passes none; delete the argument or fix the "
        "page (#1546)."
    )
    # Second, and narrower: a caller passing `token_budget=77` would satisfy
    # the resolution above and still shadow the key for every other value.
    assert "token_budget" not in calls[0], (
        "the SessionStart lane passes "
        f"token_budget={calls[0]['token_budget']!r} to retrieve(), which "
        "outranks [retrieval] token_budget in .aelfrice.toml even where the "
        "two agree. docs/user/CONFIG.md tells users that key reaches this "
        "lane because the caller passes none (#1546)."
    )
