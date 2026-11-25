"""Contradiction precedence reads the origin and never the belief type.

A belief takes the precedence class of its lock or its origin. Type
`correction` is a form, so a correction whose origin isn't in the explicit
list takes `document_recent`, like any other type.

- The first group pins that rule at the class lookup and at the resolver.
- The second group pins the class of every origin that has its own class,
  for every belief type.
- The third group pins the precedence ordering itself.

The expected classes are string literals, so a change to a `PRECEDENCE_*`
constant can't move both sides of an assertion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from aelfrice import models
from aelfrice.contradiction import (
    CLASS_NAMES,
    SOURCE_PREFIX,
    auto_resolve_all_contradictions,
    precedence_class_name,
    resolve_contradiction,
)
from aelfrice.detector_thresholds import THRESHOLDS, pin_value, size_of
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    BELIEF_TYPES,
    EDGE_CONTRADICTS,
    EDGE_RESOLVES,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    LOCK_USER,
    ORIGINS,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

# Origins that name their own class.
EXPLICIT_CLASSES: Final[dict[str, str]] = {
    "user_stated": "user_stated",
    "user_corrected": "user_corrected",
    "user_validated": "user_validated",
    "user_transcript": "user_transcript",
    "evidence_promoted": "user_transcript",
    "agent_inferred": "agent_inferred",
}

# Origins outside the explicit list. All of them take `document_recent`.
DEFAULT_ORIGINS: Final[tuple[str, ...]] = (
    "agent_remembered",
    "document_recent",
    "speculative",
    "unknown",
)

# The value that `CLASS_NAMES` has in the detector threshold manifest.
CLASS_NAMES_PIN: Final[str] = (
    "sha256:4a55a5bd080b914f9d77b3b7c58c85dca6eaf48a05c0ea9ff06b91a4ed365749"
)

NON_CORRECTION_TYPES: Final[tuple[str, ...]] = tuple(
    sorted(BELIEF_TYPES - {BELIEF_CORRECTION})
)

OLDER: Final[str] = "2026-03-01T00:00:00Z"
NEWER: Final[str] = "2026-03-02T00:00:00Z"
NOW: Final[str] = "2026-03-03T00:00:00Z"


def _mk(
    bid: str,
    *,
    btype: str,
    origin: str,
    created_at: str = OLDER,
    locked: bool = False,
) -> Belief:
    return Belief(
        id=bid,
        content=f"The invented greenhouse note {bid} sets the valve.",
        content_hash=f"hash_{bid}",
        alpha=1.0,
        beta=1.0,
        type=btype,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at=OLDER if locked else None,
        created_at=created_at,
        last_retrieved_at=None,
        origin=origin,
    )


def _assert_resolves(
    path: Path, winner: Belief, loser: Belief, rule: str, *, flip: bool,
) -> None:
    """Resolve the pair in a store on disk and check each result it writes."""
    src, dst = (loser, winner) if flip else (winner, loser)
    store = MemoryStore(str(path))
    try:
        for belief in (src, dst):
            store.insert_belief(belief)
            stored = store.get_belief(belief.id)
            assert stored is not None
            # A store that rewrites the origin would hide the rule.
            assert (stored.type, stored.origin) == (belief.type, belief.origin)
        store.insert_edge(
            Edge(src=src.id, dst=dst.id, type=EDGE_CONTRADICTS, weight=1.0)
        )

        results = auto_resolve_all_contradictions(store, now=NOW)

        assert [(r.winner_id, r.loser_id, r.rule_fired) for r in results] == [
            (winner.id, loser.id, rule)
        ]
        for edge_type in (EDGE_SUPERSEDES, EDGE_RESOLVES):
            assert store.get_edge(winner.id, loser.id, edge_type) is not None
            assert store.get_edge(loser.id, winner.id, edge_type) is None
        events = [
            (e.belief_id, e.source) for e in store.list_feedback_events()
        ]
        assert events == [(loser.id, f"{SOURCE_PREFIX}:{rule}")]
        # A direct call on the same pair agrees, in either argument order.
        for a, b in ((winner, loser), (loser, winner)):
            again = resolve_contradiction(store, a.id, b.id, now=NOW)
            assert (again.winner_id, again.rule_fired) == (winner.id, rule)
    finally:
        store.close()


# --- Group 1: a correction without an explicit origin ---------------------


def test_correction_with_unknown_origin_is_document_recent() -> None:
    belief = _mk("m", btype=BELIEF_CORRECTION, origin="unknown")
    assert precedence_class_name(belief) == "document_recent"


@pytest.mark.parametrize(
    "origin",
    # The last value has no constant: an origin that the code doesn't know
    # is outside the explicit list too.
    [o for o in DEFAULT_ORIGINS if o != "unknown"] + ["invented_origin"],
)
def test_correction_with_other_default_origin_is_document_recent(
    origin: str,
) -> None:
    belief = _mk("m", btype=BELIEF_CORRECTION, origin=origin)
    assert precedence_class_name(belief) == "document_recent"


# In the rows with a class difference, recency and id both favor the belief
# that must lose, so the winner comes from the class and nothing else.
@pytest.mark.parametrize("flip", [False, True], ids=["fwd", "rev"])
@pytest.mark.parametrize(
    ("other_origin", "other_newer", "correction_wins", "rule"),
    [
        pytest.param(
            "user_validated", False, False,
            "user_validated_beats_document_recent",
            id="loses-to-user-validated",
        ),
        pytest.param(
            "user_transcript", False, False,
            "user_transcript_beats_document_recent",
            id="loses-to-user-transcript",
        ),
        pytest.param(
            "document_recent", True, False,
            "document_recent_beats_document_recent_by_recency",
            id="loses-to-newer-document",
        ),
        pytest.param(
            "unknown", False, True,
            "document_recent_beats_document_recent_by_recency",
            id="beats-older-unknown",
        ),
        pytest.param(
            "agent_inferred", True, True,
            "document_recent_beats_agent_inferred",
            id="beats-agent-inferred",
        ),
    ],
)
def test_resolver_ranks_unknown_correction_as_document_recent(
    tmp_path: Path,
    other_origin: str,
    other_newer: bool,
    correction_wins: bool,
    rule: str,
    flip: bool,
) -> None:
    correction = _mk(
        "m", btype=BELIEF_CORRECTION, origin="unknown",
        created_at=OLDER if other_newer else NEWER,
    )
    other = _mk(
        "z" if other_newer else "a", btype=BELIEF_FACTUAL,
        origin=other_origin,
        created_at=NEWER if other_newer else OLDER,
    )
    winner, loser = (
        (correction, other) if correction_wins else (other, correction)
    )
    _assert_resolves(
        tmp_path / "memory.db", winner, loser, rule, flip=flip,
    )


# --- Group 2: every explicit origin keeps its class -----------------------


def test_every_origin_constant_is_listed() -> None:
    constants = {
        value for name, value in vars(models).items()
        if name.startswith("ORIGIN_") and isinstance(value, str)
    }
    # An empty or partial sweep would pass the checks below.
    assert "user_corrected" in constants
    listed = set(EXPLICIT_CLASSES) | set(DEFAULT_ORIGINS)
    assert constants - listed == set()
    assert set(ORIGINS) - listed == set()
    assert set(EXPLICIT_CLASSES).isdisjoint(DEFAULT_ORIGINS)


@pytest.mark.parametrize(
    "btype", [BELIEF_CORRECTION, *NON_CORRECTION_TYPES],
)
@pytest.mark.parametrize(
    ("origin", "expected"), sorted(EXPLICIT_CLASSES.items()),
)
def test_explicit_origin_keeps_its_class(
    origin: str, expected: str, btype: str,
) -> None:
    assert precedence_class_name(
        _mk("m", btype=btype, origin=origin)
    ) == expected


@pytest.mark.parametrize("btype", NON_CORRECTION_TYPES)
@pytest.mark.parametrize("origin", DEFAULT_ORIGINS)
def test_default_origin_is_document_recent_for_other_types(
    origin: str, btype: str,
) -> None:
    assert precedence_class_name(
        _mk("m", btype=btype, origin=origin)
    ) == "document_recent"


@pytest.mark.parametrize(
    "origin", sorted([*EXPLICIT_CLASSES, *DEFAULT_ORIGINS]),
)
def test_locked_correction_is_user_stated(origin: str) -> None:
    belief = _mk("m", btype=BELIEF_CORRECTION, origin=origin, locked=True)
    assert precedence_class_name(belief) == "user_stated"


@pytest.mark.parametrize("flip", [False, True], ids=["fwd", "rev"])
def test_resolver_keeps_user_corrected_origin(
    tmp_path: Path, flip: bool,
) -> None:
    """A correction whose origin says so still beats a newer belief."""
    _assert_resolves(
        tmp_path / "memory.db",
        _mk("m", btype=BELIEF_CORRECTION, origin="user_corrected"),
        _mk(
            "z", btype=BELIEF_FACTUAL, origin="user_validated",
            created_at=NEWER,
        ),
        "user_corrected_beats_user_validated",
        flip=flip,
    )


# --- Group 3: the precedence ordering -------------------------------------


def test_class_names_digest_is_unchanged() -> None:
    entries = [
        t for t in THRESHOLDS
        if (t.module, t.name) == ("aelfrice.contradiction", "CLASS_NAMES")
    ]
    assert [(t.value, t.size) for t in entries] == [(CLASS_NAMES_PIN, 6)]
    assert pin_value(CLASS_NAMES) == CLASS_NAMES_PIN
    assert size_of(CLASS_NAMES) == 6
