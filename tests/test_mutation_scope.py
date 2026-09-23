"""#1605 — the mutation job's scope decision, tested where the workflow cannot be.

`.github/workflows/mutation.yml` used to hand mutmut every changed file
under `src/aelfrice/`. `only_mutate` takes file paths, so a file enters the
mutant set whole, and a comment-only edit to `cli.py` — 11k lines, zero
non-comment lines changed — spent the job's entire 60-minute budget and was
cancelled with no report.

The decision now lives in `scripts/mutation_scope.py`, which is why it can
be tested at all: the workflow is YAML and PyYAML is not importable in CI
(#1436), so logic embedded there is unverifiable by construction.

Every test here drives the real function rather than a restatement of it,
and the two that matter most are the pair: a comment-only change must be
skipped, and a real change must NOT be. A skipper that skips everything
passes the first alone.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "mutation_scope.py"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("mutation_scope", str(_SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mutation_scope"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scope_mod() -> Any:
    return _load()


_BASE = '''\
"""A docstring."""
# A comment.


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''


def test_a_comment_only_change_is_not_mutable(scope_mod: Any) -> None:
    """The reported case: the diff cannot produce a mutant."""
    after = _BASE.replace("# A comment.", "# A different comment entirely.")
    assert after != _BASE, "fixture did not change"
    assert not scope_mod.has_mutable_change(_BASE, after)


def test_a_docstring_change_is_not_mutable(scope_mod: Any) -> None:
    after = _BASE.replace('"""Add two numbers."""', '"""Return the sum."""')
    assert after != _BASE
    assert not scope_mod.has_mutable_change(_BASE, after)


def test_adding_a_docstring_is_not_mutable(scope_mod: Any) -> None:
    """Presence, not just content: a docstring added is still inert.

    The stripper removes the node rather than blanking it, so a function
    that gains its first docstring compares equal to one without.
    """
    before = _BASE.replace('    """Add two numbers."""\n', "")
    assert before != _BASE
    assert not scope_mod.has_mutable_change(before, _BASE)


def test_blank_lines_are_not_mutable(scope_mod: Any) -> None:
    after = _BASE.replace("def add", "\n\ndef add")
    assert after != _BASE
    assert not scope_mod.has_mutable_change(_BASE, after)


def test_a_real_change_is_mutable(scope_mod: Any) -> None:
    """The control. Without it, a skipper that skips everything passes.

    `a + b` to `a - b` is exactly the shape mutmut mutates, so a scope
    that dropped this file would drop the mutant the job exists to run.
    """
    after = _BASE.replace("return a + b", "return a - b")
    assert after != _BASE
    assert scope_mod.has_mutable_change(_BASE, after)


def test_a_change_inside_a_docstringed_function_is_mutable(
    scope_mod: Any,
) -> None:
    """Stripping docstrings must not strip the code beside them."""
    after = _BASE.replace("    return a + b", "    return b + a")
    assert after != _BASE
    assert scope_mod.has_mutable_change(_BASE, after)


def test_an_added_file_is_mutable(scope_mod: Any) -> None:
    """`None` on the base side means the file is new: all of it is new code."""
    assert scope_mod.has_mutable_change(None, _BASE)


def test_an_unparseable_version_fails_open(scope_mod: Any) -> None:
    """A file that will not parse is mutated rather than skipped.

    The asymmetry decides it: mutating a file needlessly costs runner
    time, and skipping one wrongly costs an unmeasured mutant.
    """
    assert scope_mod.has_mutable_change(_BASE, "def broken(:\n")
    assert scope_mod.has_mutable_change("def broken(:\n", _BASE)


def test_line_numbers_do_not_leak_into_the_comparison(scope_mod: Any) -> None:
    """The dump must not carry attributes, or every comment reads mutable.

    Adding a comment above a statement moves its line number. A dump
    built with `include_attributes=True` would therefore differ, and the
    fix would be inert while looking correct — which is the failure this
    pins.
    """
    after = "# A new leading comment.\n" + _BASE
    assert not scope_mod.has_mutable_change(_BASE, after)
    dumped = scope_mod.normalised_ast(_BASE)
    assert dumped is not None
    assert "lineno" not in dumped


def test_normalised_ast_returns_none_on_a_syntax_error(scope_mod: Any) -> None:
    assert scope_mod.normalised_ast("def broken(:\n") is None
    assert scope_mod.normalised_ast(_BASE) is not None


def test_a_docstring_only_module_survives_stripping(scope_mod: Any) -> None:
    """Emptying a body is a syntax error on reparse; `pass` stands in."""
    only_doc = '"""Nothing but a docstring."""\n'
    dumped = scope_mod.normalised_ast(only_doc)
    assert dumped is not None
    assert "Pass" in dumped


def test_the_pathspec_matches_the_workflow(scope_mod: Any) -> None:
    """The script and the job it serves must scan the same tree.

    If they diverge, the job either mutates a path the script never
    considered or skips one it did, and neither shows up as a failure.
    """
    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github" / "workflows" / "mutation.yml"
    ).read_text(encoding="utf-8")
    assert scope_mod.PATHSPEC == "src/aelfrice/*.py"
    assert "mutation_scope.py" in workflow, (
        "the workflow no longer calls the script this module tests"
    )
