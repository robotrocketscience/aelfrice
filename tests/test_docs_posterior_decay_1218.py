"""ARCHITECTURE / PHILOSOPHY describe posterior decay honestly (#1218).

Three statements asserted posterior decay as shipped behaviour:
`scoring.decay`, `type_half_life` and `TYPE_HALF_LIFE_SECONDS` have no
caller under `src/`, so nothing ever moves a stored `(alpha, beta)`
toward the Jeffreys prior. The PHILOSOPHY lock story rested on that
function's lock short-circuit — an exemption from a mechanism that does
not run.

The docs now say which decay ships. This pins the premise they were
rewritten against, so the day `scoring.decay` acquires a caller the
prose is revisited rather than silently becoming stale in the other
direction. #1162 holds the wire-or-delete disposition; that is exactly
the change this test is watching for.

Deliberately *not* asserting the docs' wording. A text match on prose
breaks on rephrasing and says nothing about whether the claim is true.
What is checkable is the code fact the prose depends on.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "aelfrice"

# The posterior-decay surface. Named rather than globbed on "decay":
# `_apply_temporal_decay` and the meta-belief decay engine are separate,
# live mechanisms that act on ranking, not on the stored posterior.
_UNWIRED = {"decay", "type_half_life", "TYPE_HALF_LIFE_SECONDS"}


def _referencing_modules() -> dict[str, set[str]]:
    """`{module: names}` for `src/` modules touching the decay surface.

    Parsed, not grepped: every one of these names occurs in prose in
    comments and docstrings across `src/`, so a text search reports
    modules that merely mention posterior decay while explaining that it
    is not wired.

    Only two things count as a reference — importing the name from
    `aelfrice.scoring`, or reaching it as `scoring.<name>`. A bare
    `decay` identifier does not: `retrieval._apply_temporal_decay` has a
    local of that name for the *ranking* decay factor, which is a
    different mechanism and exactly the conflation #1218 exists to undo.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == (
                "aelfrice.scoring"
            ):
                hits |= {a.name for a in node.names} & _UNWIRED
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in _UNWIRED
                and isinstance(node.value, ast.Name)
                and node.value.id == "scoring"
            ):
                hits.add(node.attr)
        if hits and path.name != "scoring.py":
            found[path.name] = hits
    return found


def test_posterior_decay_still_has_no_production_caller() -> None:
    """The premise the rewritten docs rest on.

    If this fails, posterior decay was wired — good news, and the three
    statements in ARCHITECTURE.md and PHILOSOPHY.md that #1218 corrected
    now understate what ships. Update them in the same change.
    """
    assert _referencing_modules() == {}


def test_the_decay_surface_is_still_there_to_be_unwired() -> None:
    """Negative control.

    Deleting `scoring.decay` outright — the other half of #1162's
    disposition — also invalidates the docs, in the opposite direction:
    they describe an unwired function that would no longer exist. Without
    this, the test above passes most emphatically when the surface is
    gone entirely.
    """
    from aelfrice import scoring

    for name in sorted(_UNWIRED):
        assert hasattr(scoring, name), (
            f"scoring.{name} was removed; PHILOSOPHY's 'Locks, not just "
            f"decay' section and the ARCHITECTURE scoring.py row describe "
            f"it as present-but-unwired and need updating (#1162)"
        )
