"""Tests for the shared claude-memory ingest core (#1089).

`ingest_memory_text` is the per-file logic lifted out of the #985 mirror
hook so the reconcile sweep and the hook share one frontmatter ->
origin/prior mapping. These tests pin that mapping at the function level.
"""
from __future__ import annotations

from aelfrice.claude_memory_reconcile import ingest_memory_text
from aelfrice.models import ORIGIN_AGENT_INFERRED, ORIGIN_USER_VALIDATED
from aelfrice.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(":memory:")


def _file(mtype: str, body: str = "The build tool is standardised.") -> str:
    return f"---\nname: x\nmetadata:\n  type: {mtype}\n---\n\n{body}\n"


def test_user_type_maps_to_user_validated() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("user"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_USER_VALIDATED
    finally:
        s.close()


def test_feedback_type_maps_to_user_validated() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("feedback"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_USER_VALIDATED
    finally:
        s.close()


def test_project_type_maps_to_agent_inferred() -> None:
    s = _store()
    try:
        bid = ingest_memory_text(s, _file("project"))
        assert bid is not None
        b = s.get_belief(bid)
        assert b is not None
        assert b.origin == ORIGIN_AGENT_INFERRED
    finally:
        s.close()


def test_no_frontmatter_returns_none_and_writes_nothing() -> None:
    s = _store()
    try:
        assert ingest_memory_text(s, "just a body, no fence") is None
        assert s.count_beliefs() == 0
    finally:
        s.close()


def test_empty_body_returns_none() -> None:
    s = _store()
    try:
        assert ingest_memory_text(s, "---\nname: x\n---\n\n   \n") is None
    finally:
        s.close()


def test_reingest_is_idempotent_corroborates_not_duplicates() -> None:
    s = _store()
    try:
        text = _file("user")
        first = ingest_memory_text(s, text)
        second = ingest_memory_text(s, text)
        assert first == second  # content-derived id
        assert s.count_beliefs() == 1
    finally:
        s.close()
