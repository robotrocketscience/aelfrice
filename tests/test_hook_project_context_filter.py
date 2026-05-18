"""Hook-side project-context filter tests for v3.2 #858.

Covers `_filter_by_project_context()` — the post-`_retrieve()`
hook-lane filter that drops hits whose stored `project_context`
disagrees with the active `AELFRICE_PROJECT_CONTEXT` env var.

Matrix:

| scope    | belief.project_context | active | kept? |
|----------|------------------------|--------|-------|
| project  | ''                     | (any)  | yes   |  cross-context legacy row
| project  | retrieval-v3           | ''     | yes   |  no-filter mode
| project  | retrieval-v3           | retrieval-v3 | yes |  match
| project  | retrieval-v3           | other        | no  |  drop
| user     | retrieval-v3           | other        | yes |  promoted bypasses
| global   | retrieval-v3           | other        | yes |  federation bypasses
| shared:peer | retrieval-v3        | other        | yes |  federation bypasses

Also covers the empty-input fast path (no env var read).
"""
from __future__ import annotations

import pytest

from aelfrice.hook import _filter_by_project_context
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    BELIEF_SCOPE_PROJECT,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
)


def _b(
    id_: str,
    *,
    project_context: str = "",
    scope: str = BELIEF_SCOPE_PROJECT,
    lock_level: str = LOCK_NONE,
    origin: str = ORIGIN_AGENT_INFERRED,
) -> Belief:
    return Belief(
        id=id_,
        content=f"belief {id_}",
        content_hash=id_ + "-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-05-18T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
        scope=scope,
        project_context=project_context,
    )


def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty hit list: short-circuit. No env-var read needed."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    assert _filter_by_project_context([]) == []


def test_no_active_context_returns_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env var: filter is a no-op, returns input unchanged."""
    monkeypatch.delenv("AELFRICE_PROJECT_CONTEXT", raising=False)
    hits = [
        _b("a", project_context=""),
        _b("b", project_context="retrieval-v3"),
        _b("c", project_context="other"),
    ]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["a", "b", "c"]


def test_legacy_empty_context_always_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """project_context='' (cross-context legacy default) is never dropped."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [_b("legacy", project_context="")]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["legacy"]


def test_matching_context_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [_b("m", project_context="retrieval-v3")]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["m"]


def test_mismatching_context_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [
        _b("keep", project_context="retrieval-v3"),
        _b("drop", project_context="other-context"),
    ]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["keep"]


def test_user_scope_bypasses_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user-promoted belief (scope='user') is cross-context regardless
    of its stored project_context. Cross-context promotion is what scope
    'user' MEANS in v3.0 #688's federation visibility taxonomy."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [
        _b("u", project_context="other", scope="user",
           lock_level=LOCK_USER, origin=ORIGIN_USER_STATED),
    ]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["u"]


def test_global_scope_bypasses_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Federation 'global' bypasses project_context — a federation-
    shared belief is cross-context across peers AND contexts."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [_b("g", project_context="other", scope=BELIEF_SCOPE_GLOBAL)]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["g"]


def test_shared_peer_scope_bypasses_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shared:<peer>` scope (federation peer-group) also bypasses."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "retrieval-v3")
    hits = [_b("s", project_context="other", scope="shared:work")]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["s"]


def test_whitespace_active_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only env value resolves to '' → filter no-op."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "   ")
    hits = [
        _b("a", project_context="anything"),
        _b("b", project_context="other"),
    ]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["a", "b"]


def test_preserves_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filter is order-preserving — caller's ranking is not perturbed."""
    monkeypatch.setenv("AELFRICE_PROJECT_CONTEXT", "ctx-a")
    hits = [
        _b("first", project_context="ctx-a"),
        _b("dropped", project_context="ctx-b"),
        _b("second", project_context=""),
        _b("third", project_context="ctx-a"),
    ]
    out = _filter_by_project_context(hits)
    assert [h.id for h in out] == ["first", "second", "third"]
