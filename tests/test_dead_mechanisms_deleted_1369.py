"""#1369 deleted four unwired mechanisms. They must stay deleted.

#1162's disposition for each was "wire it, or delete it and strike its
docs". Each was deleted, on evidence of zero callers under `src/` plus a
named reason wiring it is unfunded. The failure mode this file guards is
resurrection-by-copy: the surfaces read as plausible infrastructure, and
three of the four had docs asserting them as live before #1218 / #1369
corrected the prose.

Each assertion is an *absence*, so each needs a distinguishing check that
would fail if the symbol came back — not merely a suite that no longer
imports it. Reverting any one of the four deletion commits turns exactly
one test here red.

Deliberately does not read `docs/`. CI's `code` path filter excludes
`docs/**`; a suite that read it would tax every docs-only PR with the
full pytest matrix, which `test_ci_path_filter` pins against.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "aelfrice"

# The posterior-decay surface deleted under #1369 §1 (filed as #1162 §4).
# Named rather than globbed on "decay": `retrieval._apply_temporal_decay`
# and `meta_beliefs._decay_toward_static` are separate, live mechanisms
# that act on ranking position and on a meta-belief series respectively,
# never on a belief's stored posterior. Conflating them is what #1218
# existed to undo.
_POSTERIOR_DECAY_NAMES = frozenset(
    {"decay", "type_half_life", "TYPE_HALF_LIFE_SECONDS"}
)


def _module_asts() -> dict[str, ast.Module]:
    """Parsed `src/aelfrice/**.py`, keyed by file name.

    Parsed, not grepped: every name checked here also occurs in prose in
    comments and docstrings across `src/` that explain why the mechanism
    is gone. A text search would report those as live references.
    """
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(_SRC.rglob("*.py"))
    }


def _bound_names(tree: ast.Module) -> set[str]:
    """Module-level names a module binds: defs, classes, assignments."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def test_posterior_decay_surface_is_gone_from_scoring() -> None:
    """`scoring` defines none of `decay` / `type_half_life` / the table.

    The import-level check below passes vacuously if the functions exist
    but nobody imports them — which is precisely the state #1162 refused
    to leave the tree in. This asserts the definitions themselves.
    """
    scoring = _module_asts()["scoring.py"]
    assert _bound_names(scoring) & _POSTERIOR_DECAY_NAMES == set()


def test_no_module_imports_the_posterior_decay_surface() -> None:
    """Nothing under `src/` reaches the names by import or attribute.

    Catches a reimplementation landing in some other module and being
    re-exported, which the definition check above would miss.
    """
    offenders: dict[str, set[str]] = {}
    for name, tree in _module_asts().items():
        hits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "aelfrice.scoring":
                hits |= {a.name for a in node.names} & _POSTERIOR_DECAY_NAMES
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in _POSTERIOR_DECAY_NAMES
                and isinstance(node.value, ast.Name)
                and node.value.id == "scoring"
            ):
                hits.add(node.attr)
        if hits:
            offenders[name] = hits
    assert offenders == {}


def test_stored_posteriors_are_never_moved_by_age() -> None:
    """The behavioural half: `scoring` exposes no age-taking function.

    The two checks above are name-based and a rename would slip past
    both. Posterior decay needs an age or half-life input to do its job,
    so an age-typed parameter anywhere in `scoring`'s public surface is
    the shape of the deleted mechanism regardless of what it is called.
    """
    scoring = _module_asts()["scoring.py"]
    age_taking: dict[str, list[str]] = {}
    for node in scoring.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        aged = [
            p
            for p in params
            if "age" in p or "half_life" in p or p in {"now", "now_ts"}
        ]
        if aged:
            age_taking[node.name] = aged
    assert age_taking == {}
