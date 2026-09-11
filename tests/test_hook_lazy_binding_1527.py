"""#1527: the deferred retrieval-subtree names must still bind.

`hook.py` stopped importing `retrieve`, `search_for_prompt` and four
`context_rebuilder` entry points at module scope, so a gate-skipped fire no
longer loads the subtree it exists to skip. The names now resolve through
`_lazy()` at the call site.

The failure mode that change introduces is quiet. A key typo'd in
`_LAZY_RETRIEVAL_NAMES`, a call site asking for a name the table does not
carry, or a module path that no longer holds the function all raise inside a
lane wrapped in a fail-soft handler — `pre_compact`, `_rebuild_and_format`,
`_retrieve_and_format_baseline` and `_run_cadence_rebuild` each swallow and
trace. The visible result is a missing audit row or an empty block, not a
crash, which is exactly the shape that survives a green suite.

The existing hook tests do not close this. `_rebuild_and_format` and
`_read_recent_for_pre_compact` are monkeypatched to stubs everywhere they
appear, so the real bodies — and therefore the real `_lazy()` calls — never
execute under test.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import aelfrice.hook as hook


def test_every_declared_lazy_name_resolves_to_the_real_attribute() -> None:
    """Each table entry must name a module that really holds that function.

    Identity, not truthiness: `getattr` on the wrong module would raise, but a
    table pointing at a module that happens to re-export a *different* object
    under the same name would not.
    """
    assert hook._LAZY_RETRIEVAL_NAMES, "the lazy table is empty"
    for name, module_path in hook._LAZY_RETRIEVAL_NAMES.items():
        expected = getattr(importlib.import_module(module_path), name)
        assert hook._lazy(name) is expected, (
            f"_lazy({name!r}) did not return {module_path}.{name}"
        )


def _lazy_call_names() -> list[str]:
    """Every string literal `hook.py` passes to `_lazy(...)`.

    Parsed rather than grepped: a substring scan cannot tell a call from a
    mention in one of the comments above the table.
    """
    tree = ast.parse(
        Path(hook.__file__).read_text(encoding="utf-8")
    )
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_lazy"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.append(node.args[0].value)
    return out


def test_every_lazy_call_site_asks_for_a_declared_name() -> None:
    """A call site naming a key that is not in the table raises `KeyError`.

    Inside `_rebuild_and_format` or `_read_recent_for_pre_compact` that is
    caught by the caller's fail-soft handler and reported as a missing block,
    so nothing else in the suite would go red.
    """
    called = _lazy_call_names()
    assert called, (
        "no `_lazy(\"...\")` call sites found in hook.py — either the "
        "deferral was reverted, or this test stopped being able to see it, "
        "and in both cases the assertion below proves nothing"
    )
    undeclared = sorted(
        set(called) - set(hook._LAZY_RETRIEVAL_NAMES)
    )
    assert not undeclared, (
        f"hook.py calls _lazy() for {undeclared}, which _LAZY_RETRIEVAL_NAMES "
        "does not carry. That is a KeyError inside a fail-soft lane."
    )


def test_every_declared_name_is_actually_used() -> None:
    """The table is not a place to leave entries behind.

    An entry no call site asks for is a name that could have stayed eager or
    been deleted, and it makes the table stop describing what the module does.
    """
    unused = sorted(
        set(hook._LAZY_RETRIEVAL_NAMES) - set(_lazy_call_names())
    )
    assert not unused, (
        f"_LAZY_RETRIEVAL_NAMES declares {unused}, which no call site asks "
        "for. Remove the entry, or restore the call that needed it."
    )


def test_a_monkeypatched_attribute_wins_over_the_real_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property a patching test module depends on.

    `tests/test_hook_user_prompt_submit.py` sets `aelfrice.hook.search_for_prompt`
    and expects the hook to call the stub. `_lazy` reads `globals()` before
    importing, which is where that write lands. Asserted here directly rather
    than left to be inferred from whichever caller happens to still exercise
    the path. No count of such modules is pinned: an earlier draft published
    one that no re-derivation reproduced, and the property is what this test
    is for, not the size of the population relying on it.
    """
    sentinel = object()
    monkeypatch.setattr(hook, "search_for_prompt", sentinel)
    assert hook._lazy("search_for_prompt") is sentinel


def test_the_module_answers_for_an_unresolved_name() -> None:
    """PEP 562 fallback: `monkeypatch.setattr(..., raising=True)` reads first.

    Without `__getattr__`, patching a name the module has not resolved yet
    would fail with AttributeError before the test body ran.
    """
    for name, module_path in hook._LAZY_RETRIEVAL_NAMES.items():
        expected = getattr(importlib.import_module(module_path), name)
        assert getattr(hook, name) is expected


def test_an_undeclared_attribute_still_raises() -> None:
    """`__getattr__` must not answer for everything.

    A fallback that returned something for any name would turn every typo
    against this module into a silent None-like value.
    """
    with pytest.raises(AttributeError):
        _ = hook.no_such_attribute_1527
