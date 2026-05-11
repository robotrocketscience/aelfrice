"""Unit tests for the skill-layer ↔ ``wonder_ingest`` adapter (#552).

Covers the pure-Python contract that the ``/aelf:wonder --axes``
slash-command flow depends on:

* ``SubagentDocument`` shape.
* ``documents_to_phantoms`` produces one ``Phantom`` per document with
  the documented constituent / generator / content / score fields.
* ``load_documents_jsonl`` parses the on-disk JSONL the CLI consumes,
  surfacing clean ``ValueError``s on the malformed-row cases.

The end-to-end test that drives the actual ``--axes`` CLI emission
through the loader and into ``wonder_ingest`` lives in
``test_wonder_skill_integration_e2e.py`` (added with the CLI commit).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.models import Phantom
from aelfrice.wonder.skill_integration import (
    DEFAULT_SUBAGENT_SCORE,
    GENERATOR_PREFIX,
    SubagentDocument,
    documents_to_phantoms,
    load_documents_jsonl,
)


def test_documents_to_phantoms_empty_input() -> None:
    out = documents_to_phantoms([], ("a", "b"))
    assert out == []


def test_documents_to_phantoms_preserves_order_and_count() -> None:
    docs = [
        SubagentDocument(axis_name="domain", content="alpha"),
        SubagentDocument(axis_name="gap_internal", content="beta"),
        SubagentDocument(axis_name="contradiction", content="gamma"),
    ]
    out = documents_to_phantoms(docs, ("seed1", "seed2"))
    assert len(out) == 3
    assert [p.content for p in out] == ["alpha", "beta", "gamma"]


def test_documents_to_phantoms_anchors_all_to_same_constituents() -> None:
    docs = [
        SubagentDocument(axis_name="a", content="x"),
        SubagentDocument(axis_name="b", content="y"),
    ]
    anchors = ("seed1", "seed2", "seed3")
    out = documents_to_phantoms(docs, anchors)
    for p in out:
        assert isinstance(p, Phantom)
        assert p.constituent_belief_ids == anchors


def test_documents_to_phantoms_generator_label_carries_axis() -> None:
    docs = [SubagentDocument(axis_name="contradiction_resolve", content="z")]
    out = documents_to_phantoms(docs, ("s",))
    assert out[0].generator == f"{GENERATOR_PREFIX}:contradiction_resolve"


def test_documents_to_phantoms_score_default_and_override() -> None:
    docs = [SubagentDocument(axis_name="a", content="x")]
    default_out = documents_to_phantoms(docs, ("s",))
    assert default_out[0].score == DEFAULT_SUBAGENT_SCORE

    custom_out = documents_to_phantoms(docs, ("s",), score=0.42)
    assert custom_out[0].score == 0.42


def test_load_documents_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    rows = [
        {"axis_name": "domain", "content": "doc1", "anchor_ids": ["b1", "b2"]},
        {"axis_name": "gap_internal", "content": "doc2", "anchor_ids": ["b1", "b2"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    docs, anchors = load_documents_jsonl(path)
    assert anchors == ("b1", "b2")
    assert [d.axis_name for d in docs] == ["domain", "gap_internal"]
    assert [d.content for d in docs] == ["doc1", "doc2"]


def test_load_documents_jsonl_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    docs, anchors = load_documents_jsonl(path)
    assert docs == []
    assert anchors == ()


def test_load_documents_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.jsonl"
    path.write_text(
        '{"axis_name": "a", "content": "x", "anchor_ids": ["s"]}\n'
        "\n"
        '{"axis_name": "b", "content": "y", "anchor_ids": ["s"]}\n'
        "   \n",
        encoding="utf-8",
    )
    docs, anchors = load_documents_jsonl(path)
    assert len(docs) == 2
    assert anchors == ("s",)


def test_load_documents_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"axis_name": "a"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_documents_jsonl(path)


def test_load_documents_jsonl_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    path.write_text(
        '{"axis_name": "a", "anchor_ids": ["s"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing key 'content'"):
        load_documents_jsonl(path)


def test_load_documents_jsonl_rejects_mixed_anchor_sets(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"axis_name": "a", "content": "x", "anchor_ids": ["s1"]}\n'
        '{"axis_name": "b", "content": "y", "anchor_ids": ["s2"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="anchor_ids mismatch"):
        load_documents_jsonl(path)
