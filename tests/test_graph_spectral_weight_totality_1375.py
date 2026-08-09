"""`EDGE_LAPLACIAN_WEIGHTS` is total over `models.EDGE_TYPES` (#1375).

`build_signed_adjacency` reads the weight table through `w_map.get(e.type,
0.0)` and drops any edge whose weight is 0.0. Before #1375 the table
listed only the six edge types that existed when #149 was written, so
`IMPLEMENTS`, `TESTS`, `TEMPORAL_NEXT` and `RESOLVES` fell through the
`.get` default and left the spectral graph silently — `TEMPORAL_NEXT`
being ~88% of the edges on a live store. Nothing anywhere said so.

The fix is not a weight change. Every one of those four stays at 0.0;
what changes is that the zeros are written down and that adding an
eleventh edge type to `models` without a decision now fails the import
instead of being absorbed by the `.get` default.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from aelfrice import models
from aelfrice.graph_spectral import (
    EDGE_LAPLACIAN_WEIGHTS,
    missing_laplacian_weights,
)
from aelfrice.models import (
    EDGE_IMPLEMENTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_RESOLVES,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    EDGE_TYPES,
)

# The four types that #149 predates. Named explicitly so that a future
# decision to give any of them a non-zero weight has to edit a test that
# says what the old value meant, rather than silently pass.
DELIBERATE_ZEROS: tuple[str, ...] = (
    EDGE_IMPLEMENTS,
    EDGE_TESTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_RESOLVES,
)


def test_every_edge_type_has_an_explicit_laplacian_weight() -> None:
    assert missing_laplacian_weights(EDGE_LAPLACIAN_WEIGHTS, EDGE_TYPES) == (
        frozenset()
    )
    for edge_type in EDGE_TYPES:
        assert edge_type in EDGE_LAPLACIAN_WEIGHTS


def test_missing_laplacian_weights_names_what_is_absent() -> None:
    """The guard's helper must be able to fail, not just to return empty.

    Asserting only that the shipped table is total cannot tell a working
    guard from one that returns `frozenset()` unconditionally.
    """
    truncated = {
        k: v
        for k, v in EDGE_LAPLACIAN_WEIGHTS.items()
        if k != EDGE_TEMPORAL_NEXT
    }
    assert missing_laplacian_weights(truncated, EDGE_TYPES) == frozenset(
        {EDGE_TEMPORAL_NEXT}
    )


def test_deliberate_zeros_are_present_and_zero() -> None:
    """The post-#149 types are excluded on purpose, at 0.0.

    Pinning the value as well as the key: an entry that exists but has
    drifted off 0.0 is a ranking change to every heat-kernel score, and
    it would otherwise be caught by nothing.
    """
    for edge_type in DELIBERATE_ZEROS:
        assert EDGE_LAPLACIAN_WEIGHTS[edge_type] == 0.0


def test_marker_edge_is_not_required_to_carry_a_weight() -> None:
    """Containment is one-directional — weights ⊇ EDGE_TYPES, not equal.

    `POTENTIALLY_STALE` is a marker tag rather than a structural
    relation and is deliberately outside `EDGE_TYPES`, so the guard must
    not demand an entry for it.
    """
    assert EDGE_POTENTIALLY_STALE not in EDGE_TYPES
    assert EDGE_POTENTIALLY_STALE not in EDGE_LAPLACIAN_WEIGHTS


def test_import_fails_when_a_new_edge_type_has_no_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding an edge type to `models` without a weight breaks the import.

    This is the assertion the whole section exists for: the previous
    behaviour was for such a type to be absorbed by
    `w_map.get(e.type, 0.0)` and never mentioned again.
    """
    # `from aelfrice.graph_spectral import ...` at the top of this file has
    # already put the module in `sys.modules`, which is what `delitem` below
    # needs. The alias import that used to sit here existed only to feed an
    # `is not None` assertion -- a tautology, and CodeQL flagged the resulting
    # import-both-ways as the smell it is.
    assert "aelfrice.graph_spectral" in sys.modules
    monkeypatch.setattr(
        models, "EDGE_TYPES", models.EDGE_TYPES | {"NEWLY_ADDED_EDGE"}
    )
    monkeypatch.delitem(sys.modules, "aelfrice.graph_spectral")
    with pytest.raises(RuntimeError, match="NEWLY_ADDED_EDGE"):
        importlib.import_module("aelfrice.graph_spectral")
