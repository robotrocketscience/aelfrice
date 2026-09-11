"""The `context_rebuilder` -> `rebuild_log` compatibility surface (#1527).

#1527 moved the rebuilder config and the phase-1a rebuild log out of
`aelfrice.context_rebuilder` into `aelfrice.rebuild_log`, so a hook fire the
prompt-shape gate refuses stops importing the retrieval subtree it is
skipping. `context_rebuilder` re-exports the moved names so the existing
`from aelfrice.context_rebuilder import ...` callers keep working.

An earlier draft of that sentence said `context_rebuilder` re-exports *every*
name, which is false in two ways: two entries in `rebuild_log.__all__` are
deliberately not re-exported, and twelve names that were reachable on
`context_rebuilder` before the move are gone because they were imports that
had leaked into its namespace. These tests are what makes the corrected
sentence checkable instead of asserted -- they hold the exception list to
exactly those two names, and they hold every real caller to resolving.
"""
from __future__ import annotations

import ast
from pathlib import Path

import aelfrice.context_rebuilder as cr
import aelfrice.rebuild_log as rl

#: The only two entries in `rebuild_log.__all__` that `context_rebuilder`
#: does not re-export. Both are private helpers nothing outside
#: `rebuild_log.py` calls, so re-exporting them would add two dead imports.
NOT_RE_EXPORTED = frozenset({"_extracted_entities_for_log", "_recent_turns_hash"})

#: Directories searched for `from aelfrice.context_rebuilder import ...`.
_SEARCH_ROOTS = ("src", "tests", "benchmarks", "scripts")


def _context_rebuilder_from_imports() -> list[tuple[Path, int, str]]:
    """Every `from aelfrice.context_rebuilder import NAME` in the tree."""
    root = Path(__file__).resolve().parent.parent
    found: list[tuple[Path, int, str]] = []
    for sub in _SEARCH_ROOTS:
        for path in sorted((root / sub).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "aelfrice.context_rebuilder":
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        found.append((path, node.lineno, alias.name))
    return found


def test_the_re_export_block_covers_all_but_the_two_named_exceptions() -> None:
    """The universal claim, enumerated rather than asserted.

    The docstring in `rebuild_log.py`, the `ARCHITECTURE.md` module table and
    the changelog entry all name these two as the exceptions. If the block
    drifts -- a name dropped, or one of these two quietly added back -- this
    fails and those three sentences have to be rewritten with it.
    """
    declared = set(rl.__all__)
    missing = declared - set(dir(cr))
    assert missing == set(NOT_RE_EXPORTED), (
        f"context_rebuilder re-exports all of rebuild_log.__all__ except "
        f"{sorted(NOT_RE_EXPORTED)}; it is now missing {sorted(missing)}. "
        "Update the block, or update the three places that name the "
        "exceptions: rebuild_log.py's module docstring, the rebuild_log.py "
        "row of docs/concepts/ARCHITECTURE.md, and "
        "CHANGELOG/unreleased/1527-gate-skip-leaf.md."
    )


def test_every_context_rebuilder_import_in_the_tree_resolves() -> None:
    """The consequence the corrected sentence claims, checked against callers.

    The re-export block is worth nothing if a real caller still asks for a
    name the move took away. Driven off the source tree rather than a hand
    list, because a hand list is how the false claim got published.
    """
    imports = _context_rebuilder_from_imports()
    assert imports, (
        "no `from aelfrice.context_rebuilder import ...` found under "
        f"{list(_SEARCH_ROOTS)} -- the search is broken, not the surface"
    )
    unresolved = [
        f"{path}:{lineno} imports {name}"
        for path, lineno, name in imports
        if not hasattr(cr, name)
    ]
    assert not unresolved, (
        "these callers import a name aelfrice.context_rebuilder no longer "
        f"has: {unresolved}"
    )
