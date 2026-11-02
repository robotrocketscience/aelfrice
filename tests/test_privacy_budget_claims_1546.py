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

**The page itself is not read here.** When this file was written, reading
`docs/user/PRIVACY.md` from a test was the #1160 failure mode: `ci.yml`'s
`code` filter did not list the page, so the suite was skipped on exactly the
docs-only pull request that changed it, and a guard on the page would have
reported a pass from an `echo`. The filter now lists each file outside the
package directories that a test builds a path to by name (#1786 CI-07),
`PRIVACY.md` among them, and a test that reads a whole directory or glob
outside them runs in the filter's `tree_reads` step. `tests/test_ci_path_filter.py` derives that list from the tests'
own path expressions and from the paths and readers of the repo modules a
test loads, and fails on a read under the repo root that it cannot resolve.
A new page read through a path the scan follows is either in the filter or
fails that module on the PR that adds it; that module's docstring names the
reads it does not follow. A test here can read the page.

No test in this module reads it. The page's two published figures carry #1469
derived-figure markers, and CI checks both. `scripts/check_derived_figures.py`
used to pair backticks across the whole file, so markers below the page's fence
lines parsed as absent while the gate reported green; #1556 fixed the scanner.
The `derived-figures` job, which has no path filter and runs on every pull
request, re-derives both figures from the shipped constants, and
`tests/test_code_span_scanner_1556.py::test_the_privacy_page_publishes_the_two_budgets_it_names`
asserts that both markers parse with no hard failure.
"""
from __future__ import annotations

import ast
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

_REPO = Path(__file__).resolve().parents[1]
# The test that guards the page's two figures, which the docstring names.
_GUARD_MODULE = "tests/test_code_span_scanner_1556.py"
_GUARD_TEST = "test_the_privacy_page_publishes_the_two_budgets_it_names"


def test_the_docstring_names_the_test_that_guards_the_page_figures() -> None:
    """The account above of what guards `PRIVACY.md`'s figures must name a real test.

    It once said review guarded them and that markers on the page parsed as
    absent, after #1556 had made both markers parse and a test held them to
    it. Naming that test, and checking that it exists, fails the account when
    the docstring drops it or the test is renamed or removed.
    """
    doc = " ".join((__doc__ or "").split())
    assert f"{_GUARD_MODULE}::{_GUARD_TEST}" in doc, (
        f"the module docstring no longer names {_GUARD_MODULE}::{_GUARD_TEST} "
        "as the guard on docs/user/PRIVACY.md's two figures"
    )
    tree = ast.parse((_REPO / _GUARD_MODULE).read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert _GUARD_TEST in names, f"{_GUARD_MODULE} has no {_GUARD_TEST}"


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
