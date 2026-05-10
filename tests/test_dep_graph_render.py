"""Tests for scripts/dep_graph_render.py (#581).

Exercises the offline-renderable path: the GraphQL fetch is mocked at the
module boundary and ``render_mermaid`` / ``render_comment_body`` operate
on a hand-rolled fixture that mirrors the GraphQL response shape.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "dep_graph_render.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("dep_graph_render", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load()
render_mermaid = _mod.render_mermaid  # type: ignore[attr-defined]
render_comment_body = _mod.render_comment_body  # type: ignore[attr-defined]
_short_title = _mod._short_title  # type: ignore[attr-defined]
_classify = _mod._classify  # type: ignore[attr-defined]
STICKY_HEADER = _mod.STICKY_HEADER  # type: ignore[attr-defined]


def _node(
    number: int,
    title: str = "issue",
    *,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    tracked_issues: list[int] | None = None,
    tracked_in_issues: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "labels": {"nodes": [{"name": n} for n in (labels or [])]},
        "assignees": {"nodes": [{"login": login} for login in (assignees or [])]},
        "trackedIssues": {
            "nodes": [{"number": n, "state": "OPEN"} for n in (tracked_issues or [])],
        },
        "trackedInIssues": {
            "nodes": [{"number": n, "state": "OPEN"} for n in (tracked_in_issues or [])],
        },
    }


# --- _short_title -----------------------------------------------------------


class TestShortTitle:
    def test_passthrough_under_limit(self) -> None:
        assert _short_title("hello world") == "hello world"

    def test_truncates_with_ellipsis(self) -> None:
        s = _short_title("a" * 60, limit=20)
        assert s.endswith("…")
        assert len(s) == 20

    def test_strips_brackets_and_quotes(self) -> None:
        # Mermaid breaks on unescaped [, ], ", `
        assert _short_title('foo [bar] "baz" `qux`') == "foo bar baz qux"


# --- _classify --------------------------------------------------------------


class TestClassify:
    def test_leaf(self) -> None:
        n = _node(1)
        assert _classify(n, {1}) == "leaf"

    def test_in_progress_via_assignee(self) -> None:
        n = _node(1, assignees=["someone"])
        assert _classify(n, {1}) == "inProgress"

    def test_in_progress_via_author_label(self) -> None:
        n = _node(1, labels=["author-Maxwell"])
        assert _classify(n, {1}) == "inProgress"

    def test_has_blockers_when_blocker_open(self) -> None:
        n = _node(2, tracked_in_issues=[1])
        assert _classify(n, {1, 2}) == "hasBlockers"

    def test_no_blockers_when_blocker_closed(self) -> None:
        n = _node(2, tracked_in_issues=[1])
        # Blocker #1 is not in open_numbers → treated as closed → leaf.
        assert _classify(n, {2}) == "leaf"


# --- render_mermaid ---------------------------------------------------------


class TestRenderMermaid:
    def test_smoke_single_node(self) -> None:
        out, n_nodes, n_edges = render_mermaid([_node(1, "A")])
        assert "graph TD" in out
        assert 'N1["#1 A"]' in out
        assert n_nodes == 1
        assert n_edges == 0

    def test_edge_from_tracked_issues(self) -> None:
        # Issue 1 blocks issue 2.
        a = _node(1, "A", tracked_issues=[2])
        b = _node(2, "B")
        out, _, n_edges = render_mermaid([a, b])
        assert "N1 --> N2" in out
        assert n_edges == 1

    def test_edge_from_tracked_in_issues_dedup(self) -> None:
        # Same edge expressed from both sides — must dedupe.
        a = _node(1, "A", tracked_issues=[2])
        b = _node(2, "B", tracked_in_issues=[1])
        out, _, n_edges = render_mermaid([a, b])
        assert out.count("N1 --> N2") == 1
        assert n_edges == 1

    def test_drops_edge_to_closed_issue(self) -> None:
        # 999 is not in the open set; the edge to it should be dropped.
        a = _node(1, "A", tracked_issues=[999])
        out, _, n_edges = render_mermaid([a])
        assert "N999" not in out
        assert n_edges == 0

    def test_class_assignment(self) -> None:
        a = _node(1, "leaf")
        b = _node(2, "blocked", tracked_in_issues=[1])
        c = _node(3, "wip", labels=["author-Faraday"])
        out, _, _ = render_mermaid([a, b, c])
        assert "class N1 leaf" in out
        assert "class N2 hasBlockers" in out
        assert "class N3 inProgress" in out

    def test_claim_label_in_node_text(self) -> None:
        n = _node(1, "thing", labels=["author-Faraday"])
        out, _, _ = render_mermaid([n])
        assert "(faraday)" in out

    def test_deterministic_output(self) -> None:
        # Same input set in different order → same output.
        a = _node(2, "B", tracked_in_issues=[1])
        b = _node(1, "A", tracked_issues=[2])
        out_1, _, _ = render_mermaid([a, b])
        out_2, _, _ = render_mermaid([b, a])
        assert out_1 == out_2


# --- render_comment_body ----------------------------------------------------


class TestCommentBody:
    def test_starts_with_sticky_header(self) -> None:
        body = render_comment_body([_node(1, "A")], "2026-05-10T00:00:00Z")
        assert body.startswith(STICKY_HEADER)

    def test_contains_counts_and_timestamp(self) -> None:
        body = render_comment_body(
            [_node(1, "A", tracked_issues=[2]), _node(2, "B")],
            "2026-05-10T12:34:56Z",
        )
        assert "Nodes: **2**" in body
        assert "Edges: **1**" in body
        assert "2026-05-10T12:34:56Z" in body

    def test_contains_legend(self) -> None:
        body = render_comment_body([_node(1)], "t")
        assert "leaf" in body.lower()
        assert "blocker" in body.lower()


# --- end-to-end shape check -------------------------------------------------


def test_full_fixture_shape() -> None:
    """Mirror a realistic GraphQL response and check the rendered block end-to-end."""
    fixture = [
        _node(549, "C3 wonder CLI", labels=["v2.1", "author-Leibniz"]),
        _node(550, "C4 promotion trigger", labels=["v2.1"], tracked_in_issues=[549]),
        _node(580, "pre-push freshness", labels=["v2.1"]),
    ]
    body = render_comment_body(fixture, "2026-05-10T00:00:00Z")
    # Sanity: each issue rendered, the blocked->blocker edge drawn,
    # the in-progress class assigned to the claimed issue.
    assert 'N549["#549 C3 wonder CLI (leibniz)"]' in body
    assert "N549 --> N550" in body
    assert "class N549 inProgress" in body
    assert "class N550 hasBlockers" in body
    assert "class N580 leaf" in body
