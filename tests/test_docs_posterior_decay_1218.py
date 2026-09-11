"""ARCHITECTURE and PHILOSOPHY describe posterior decay honestly.

Three statements once asserted posterior decay as shipped behaviour.
Nothing under `src/` ever called the surface that implemented it, so
nothing ever moved a stored `(alpha, beta)` toward the Jeffreys prior,
and the PHILOSOPHY lock story rested on that function's lock
short-circuit — an exemption from a mechanism that did not run.
[#1218](https://github.com/robotrocketscience/aelfrice/issues/1218)
corrected the prose and pinned the premise with a two-directional
guard; the code stayed, described as present-but-unwired.

[#1369](https://github.com/robotrocketscience/aelfrice/issues/1369)
discharged the wire-or-delete disposition
[#1162](https://github.com/robotrocketscience/aelfrice/issues/1162)
held open, by deleting it. So the premise flipped, and one direction of
the old guard is now the whole guard: the docs say the surface is gone,
and this fails if it comes back.

Deliberately *not* asserting the docs' wording. A text match on prose
breaks on rephrasing and says nothing about whether the claim is true.
What is checkable is the code fact the prose depends on.
"""
from __future__ import annotations

import pytest

# The deleted posterior-decay surface. Named rather than globbed on
# "decay": `retrieval._apply_temporal_decay` and
# `meta_beliefs.decay_toward_default` are separate and live, and they act
# on a ranking score and on a meta-belief respectively, not on a stored
# belief posterior. That conflation is what #1218 exists to undo, and the
# corrected prose cites both by name. Neither needs a guard here: the
# suite already imports `_apply_temporal_decay` directly in
# `tests/test_retrieve_v2_temporal_sort.py`, and `store.py` imports
# `decay_toward_default` at module scope, so a rename of either fails
# loudly without this file's help.
_DELETED = ("decay", "type_half_life", "TYPE_HALF_LIFE_SECONDS")


@pytest.mark.parametrize("name", _DELETED)
def test_the_posterior_decay_surface_is_gone(name: str) -> None:
    """The premise the rewritten docs rest on.

    If this fails, the surface came back. `docs/concepts/ARCHITECTURE.md`
    (principle 5 and the `scoring.py` module-table row) and
    `docs/concepts/PHILOSOPHY.md` ("Locks, not just decay") both state
    that it does not exist; update them in the same change, and say
    whether the returning code has a caller this time.
    """
    from aelfrice import scoring

    assert not hasattr(scoring, name), (
        f"scoring.{name} is back; ARCHITECTURE.md and PHILOSOPHY.md say "
        f"the posterior-decay surface was deleted (#1369) and need "
        f"updating"
    )
