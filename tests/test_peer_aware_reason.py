"""Peer-aware graph walk tests (#690).

Two-scope acceptance for the read-only BFS walk extension. Project B's
``aelf reason`` BFS follows edges that originate in project A (a peer)
when the seed lands on a foreign belief.

Setup pattern mirrors ``test_federation_readonly.py``: create a "peerA"
DB on disk seeded with a 2-hop belief chain whose all beliefs have
``scope='global'``, then open a local "project B" store wired to the
peer via ``AELFRICE_KNOWLEDGE_DEPS`` and assert the walk surfaces the
peer hops with ``owning_scope='peerA'``.

Updated for #688 scope field semantics: peer-side beliefs are
constructed with ``scope='global'`` so they're visible across the
federation boundary (``project`` scope stays local-only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.bfs_multihop import ScoredHop, expand_bfs
from aelfrice.models import (
    EDGE_SUPPORTS,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.reason import Verdict, suggested_updates
from aelfrice.store import MemoryStore


def _belief(id_: str, content: str, *, scope: str = "global") -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-hash",
        alpha=4.0,
        beta=1.0,
        type="factual",
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-12T00:00:00+00:00",
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_AGENT_INFERRED,
        scope=scope,
    )


def _seed_peer_with_chain(path: Path) -> None:
    """Create a peer DB with a 2-hop SUPPORTS chain a→b→c, all global."""
    store = MemoryStore(str(path))
    try:
        for b in (
            _belief("peer-a", "anchor belief in peer A"),
            _belief("peer-b", "one-hop neighbour of peer-a"),
            _belief("peer-c", "two-hop neighbour of peer-a"),
        ):
            store.insert_belief(b)
        store.insert_edge(
            Edge(src="peer-a", dst="peer-b", type=EDGE_SUPPORTS, weight=1.0)
        )
        store.insert_edge(
            Edge(src="peer-b", dst="peer-c", type=EDGE_SUPPORTS, weight=1.0)
        )
    finally:
        store.close()


def _wire_peer(tmp_path: Path, peer_path: Path, monkeypatch) -> Path:
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps(
            {
                "version": 1,
                "deps": [{"name": "peerA", "path": str(peer_path)}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))
    return deps_file


def test_edges_from_in_scope_routes_to_peer(tmp_path: Path, monkeypatch):
    """Edges read from the peer DB when ``owning_scope='peerA'``."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer_with_chain(peer_path)
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        peer_edges = local.edges_from_in_scope("peer-a", "peerA")
        local_edges = local.edges_from_in_scope("peer-a", None)
        assert local_edges == []
        assert len(peer_edges) == 1
        assert peer_edges[0].src == "peer-a"
        assert peer_edges[0].dst == "peer-b"
        assert peer_edges[0].type == EDGE_SUPPORTS
    finally:
        local.close()


def test_get_belief_in_scope_routes_to_peer(tmp_path: Path, monkeypatch):
    """Belief materialises from the peer DB when ``owning_scope='peerA'``."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer_with_chain(peer_path)
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        b_peer = local.get_belief_in_scope("peer-b", "peerA")
        b_local = local.get_belief_in_scope("peer-b", None)
        assert b_peer is not None
        assert b_peer.id == "peer-b"
        assert b_peer.content == "one-hop neighbour of peer-a"
        assert b_peer.scope == "global"
        assert b_local is None
    finally:
        local.close()


def test_in_scope_helpers_tolerate_unreachable_peer(
    tmp_path: Path, monkeypatch
):
    """Unreachable peer → empty/None, never raises."""
    peer_path = tmp_path / "missing-peerA.db"  # never created
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        assert local.edges_from_in_scope("peer-a", "peerA") == []
        assert local.get_belief_in_scope("peer-a", "peerA") is None
    finally:
        local.close()


def test_in_scope_helpers_local_passthrough(tmp_path: Path):
    """``owning_scope=None`` is a transparent delegate to local methods."""
    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        local.insert_belief(_belief("local-1", "a local belief"))
        local.insert_belief(_belief("local-2", "another local"))
        local.insert_edge(
            Edge(
                src="local-1", dst="local-2",
                type=EDGE_SUPPORTS, weight=1.0,
            )
        )
        assert local.get_belief_in_scope("local-1", None) is not None
        assert len(local.edges_from_in_scope("local-1", None)) == 1
    finally:
        local.close()


def test_expand_bfs_follows_peer_edges_two_hops(
    tmp_path: Path, monkeypatch
):
    """Acceptance bullet 1: peer 2-hop neighbour surfaces with owning_scope."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer_with_chain(peer_path)
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        seed = local.get_belief_in_scope("peer-a", "peerA")
        assert seed is not None
        hops = expand_bfs(
            [seed],
            local,
            seed_scopes={"peer-a": "peerA"},
        )
        ids = [h.belief.id for h in hops]
        assert "peer-b" in ids and "peer-c" in ids
        for h in hops:
            assert h.owning_scope == "peerA"
            assert h.belief.scope == "global"
        # Determinism: SUPPORTS-only chain ranks peer-b before peer-c
        # (lower depth → higher compound score).
        assert hops[0].belief.id == "peer-b"
        assert hops[1].belief.id == "peer-c"
        assert hops[0].depth == 1
        assert hops[1].depth == 2
    finally:
        local.close()


def test_expand_bfs_local_seed_walks_local_edges_only(tmp_path: Path):
    """Pre-federation behaviour preserved: local seeds → owning_scope=None."""
    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        local.insert_belief(_belief("local-a", "anchor"))
        local.insert_belief(_belief("local-b", "neighbour"))
        local.insert_edge(
            Edge(
                src="local-a", dst="local-b",
                type=EDGE_SUPPORTS, weight=1.0,
            )
        )
        seed = local.get_belief_in_scope("local-a", None)
        assert seed is not None
        hops = expand_bfs([seed], local)  # no seed_scopes
        assert len(hops) == 1
        assert hops[0].belief.id == "local-b"
        assert hops[0].owning_scope is None
    finally:
        local.close()


def test_expand_bfs_unreachable_peer_yields_no_hops(
    tmp_path: Path, monkeypatch
):
    """When the peer DB is missing, the walk degrades to zero hops."""
    peer_path = tmp_path / "missing.db"
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        synthetic_seed = _belief("peer-a", "stand-in for the missing seed")
        hops = expand_bfs(
            [synthetic_seed],
            local,
            seed_scopes={"peer-a": "peerA"},
        )
        assert hops == []
    finally:
        local.close()


def test_suggested_updates_flags_foreign_ids():
    """Acceptance bullet 5: SuggestedUpdate rows for peer hops carry owning_scope."""
    local_b = _belief("local-b", "local belief on chain")
    peer_b = _belief("peer-b", "peer belief on chain")
    hops = [
        ScoredHop(
            belief=local_b,
            score=0.8,
            depth=1,
            path=[EDGE_SUPPORTS],
            belief_id_trail=("local-a", "local-b"),
            owning_scope=None,
        ),
        ScoredHop(
            belief=peer_b,
            score=0.6,
            depth=1,
            path=[EDGE_SUPPORTS],
            belief_id_trail=("peer-a", "peer-b"),
            owning_scope="peerA",
        ),
    ]
    rows = suggested_updates(Verdict.SUFFICIENT, impasses=[], hops=hops)
    by_id = {r.belief_id: r for r in rows}
    assert by_id["local-b"].owning_scope is None
    assert by_id["peer-b"].owning_scope == "peerA"
    # Both rows are direction=+1 (confident, non-impasse hops).
    assert by_id["local-b"].direction == "+1"
    assert by_id["peer-b"].direction == "+1"
