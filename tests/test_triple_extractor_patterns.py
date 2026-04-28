"""Per-relation extraction templates fire at least once on each fixture
sentence; emitted relations all live in EDGE_TYPES."""
from __future__ import annotations

import pytest

from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TYPES,
)
from aelfrice.triple_extractor import extract_triples


@pytest.mark.parametrize(
    "text,expected_relation",
    [
        ("the new index supports faster queries", EDGE_SUPPORTS),
        ("the new index is supported by the cache layer", EDGE_SUPPORTS),
        ("the proposal cites the prior memo", EDGE_CITES),
        ("the proposal mentions the prior memo", EDGE_CITES),
        ("the new finding contradicts the earlier paper", EDGE_CONTRADICTS),
        ("the new finding disagrees with the earlier paper", EDGE_CONTRADICTS),
        ("this commit supersedes the legacy parser", EDGE_SUPERSEDES),
        ("this commit replaces the legacy parser", EDGE_SUPERSEDES),
        ("the cache layer relates to retrieval", EDGE_RELATES_TO),
        ("the cache layer is related to retrieval", EDGE_RELATES_TO),
        ("the spec is derived from the prior memo", EDGE_DERIVED_FROM),
        ("the spec is based on the prior memo", EDGE_DERIVED_FROM),
        ("the spec extends the prior memo", EDGE_DERIVED_FROM),
    ],
)
def test_pattern_emits_expected_relation(text: str, expected_relation: str) -> None:
    triples = extract_triples(text)
    assert len(triples) >= 1
    assert any(t.relation == expected_relation for t in triples)


def test_passive_form_swaps_subject_and_object() -> None:
    triples = extract_triples("the new index is supported by the cache layer")
    assert len(triples) >= 1
    t = triples[0]
    assert t.relation == EDGE_SUPPORTS
    assert t.subject == "the cache layer"  # active-voice agent
    assert t.object == "the new index"  # active-voice patient


def test_every_relation_in_edge_types() -> None:
    text = (
        "the index supports queries. "
        "the proposal cites the memo. "
        "the spec contradicts the paper. "
        "this commit replaces the parser. "
        "the cache relates to retrieval. "
        "the spec is derived from the memo."
    )
    triples = extract_triples(text)
    assert triples
    for t in triples:
        assert t.relation in EDGE_TYPES, (
            f"extractor emitted {t.relation!r} not in EDGE_TYPES"
        )


def test_derived_from_template_fires_on_classic_phrasing() -> None:
    """Acceptance criterion 8: validates the DERIVED_FROM re-add has
    a real producer in the extractor."""
    triples = extract_triples("X is derived from Y")
    assert any(t.relation == EDGE_DERIVED_FROM for t in triples)
