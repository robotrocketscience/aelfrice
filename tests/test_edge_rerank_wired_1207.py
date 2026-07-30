"""`expand_bfs` actually calls the marker-edge rerank (#1207).

`edge_rerank` had no importer. `BFS_EDGE_WEIGHTS` pinned
``POTENTIALLY_STALE`` at 0.0 with a comment saying demotion happens in
the rerank pass — and that pass never ran, so nothing demoted a
potentially-stale belief anywhere in production. #1208 made the module
correct; it did not make it reachable.

Operator decision (2026-07-30): wire it on unconditionally, no lane
flag. The producer is already the opt-in — ``aelf doctor
--detect-stale`` defaults off, so a store holding no marker edges makes
the pass an identity.

These tests assert both halves of that reasoning through the real
retrieval path rather than the helper: a marker edge demotes, and the
absence of one changes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.bfs_multihop import expand_bfs
from aelfrice.edge_rerank import DEFAULT_STALE_PENALTY
from aelfrice.federation import ENV_KNOWLEDGE_DEPS
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_POTENTIALLY_STALE,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

_SEED = "B" + "0" * 15
_PLAIN = "B" + "1" * 15
_STALE = "B" + "2" * 15


def _mk(belief_id: str) -> Belief:
    return Belief(
        id=belief_id, content=f"content of {belief_id}",
        content_hash=f"h_{belief_id}", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-05-05T00:00:00Z", last_retrieved_at=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """A seed supporting two beliefs, reachable at equal path score."""
    s = MemoryStore(str(tmp_path / "memory.db"))
    for bid in (_SEED, _PLAIN, _STALE):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src=_SEED, dst=_PLAIN, type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src=_SEED, dst=_STALE, type=EDGE_SUPPORTS, weight=1.0))
    yield s
    s.close()


def _walk(s: MemoryStore, **kwargs: object) -> dict[str, float]:
    seed = s.get_belief(_SEED)
    assert seed is not None
    return {h.belief.id: h.score for h in expand_bfs([seed], s, **kwargs)}


# --- the wire ------------------------------------------------------------


def test_a_stale_belief_is_demoted_through_expand_bfs(
    store: MemoryStore,
) -> None:
    """The gap the issue names: nothing demoted a stale belief.

    Asserted through `expand_bfs`, not `apply_edge_type_rerank` — the
    module was already correct in isolation and still inert.
    """
    before = _walk(store)
    assert before[_PLAIN] == pytest.approx(before[_STALE]), (
        "fixture must reach both at the same score, or the demotion "
        "below is indistinguishable from the paths differing"
    )

    store.insert_edge(
        Edge(src=_SEED, dst=_STALE, type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    after = _walk(store)

    assert after[_STALE] == pytest.approx(
        before[_STALE] * DEFAULT_STALE_PENALTY
    )
    assert after[_PLAIN] == pytest.approx(before[_PLAIN])


def test_demotion_reorders_the_returned_list(store: MemoryStore) -> None:
    """Scores moving is only half of it — the rank must move too.

    The stale belief is given the *stronger* path (CONTRADICTS, 0.85 vs
    SUPPORTS' 0.60) so it outranks its rival before the pass runs. With
    both on equal paths the id tie-break alone produces the expected
    order, and this test would pass against an unwired `expand_bfs` —
    verified, which is why the fixture is overridden here.
    """
    store.insert_edge(
        Edge(src=_SEED, dst=_STALE, type=EDGE_CONTRADICTS, weight=1.0)
    )
    seed = store.get_belief(_SEED)
    assert seed is not None

    before = [h.belief.id for h in expand_bfs([seed], store)]
    assert before.index(_STALE) < before.index(_PLAIN), (
        "fixture must rank the stale belief first, or the flip below "
        "is not attributable to the demotion"
    )

    store.insert_edge(
        Edge(src=_SEED, dst=_STALE, type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    after = [h.belief.id for h in expand_bfs([seed], store)]
    assert after.index(_PLAIN) < after.index(_STALE)


# --- and the identity the decision rests on ------------------------------


def test_a_store_with_no_marker_edges_is_unaffected(
    store: MemoryStore,
) -> None:
    """Why no lane flag was added.

    `aelf doctor --detect-stale` is opt-in, so most stores hold zero
    marker edges. For them the pass computes an empty firing set and
    re-sorts on the key `expand_bfs` already applied — an identity.
    Wiring it changes nothing for anyone who has not opted in, and this
    test is what makes that claim checkable rather than asserted.
    """
    seed = store.get_belief(_SEED)
    assert seed is not None
    hops = expand_bfs([seed], store)
    assert [(h.belief.id, h.score) for h in hops] == [
        (_PLAIN, 1.0 * 0.60), (_STALE, 1.0 * 0.60),
    ]
    assert [h.belief.id for h in hops] == sorted([_PLAIN, _STALE])


def test_marker_edges_still_are_not_traversed(store: MemoryStore) -> None:
    """`BFS_EDGE_WEIGHTS[POTENTIALLY_STALE] = 0.0` is untouched.

    The pin governs traversal *through* a marker edge; the rerank keys
    off marker edges *incoming to* a surfaced belief. Wiring the rerank
    must not make the marker a path — a belief reachable only that way
    stays unreachable.
    """
    orphan = "B" + "9" * 15
    store.insert_belief(_mk(orphan))
    store.insert_edge(
        Edge(src=_SEED, dst=orphan, type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    assert orphan not in _walk(store)


def test_the_hop_survives_the_pass_intact(store: MemoryStore) -> None:
    """Regression guard on #1208 now that the pass is on the live path.

    Before #1208 the rerank rebuilt each hop field by field, erasing
    `belief_id_trail` and `owning_scope`. That was harmless while
    nothing called it; wiring it makes any recurrence a live defect.
    """
    store.insert_edge(
        Edge(src=_SEED, dst=_STALE, type=EDGE_POTENTIALLY_STALE, weight=1.0)
    )
    seed = store.get_belief(_SEED)
    assert seed is not None
    stale = next(h for h in expand_bfs([seed], store) if h.belief.id == _STALE)
    assert stale.belief_id_trail == (_SEED, _STALE)
    assert stale.depth == 1
    assert stale.path == [EDGE_SUPPORTS]


# --- federation ----------------------------------------------------------


def _peer_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A peer DB holding a belief with an incoming marker edge."""
    peer_path = tmp_path / "peer.db"
    peer = MemoryStore(str(peer_path))
    try:
        peer.insert_belief(_mk(_STALE))
        peer.insert_edge(
            Edge(
                src=_SEED, dst=_STALE,
                type=EDGE_POTENTIALLY_STALE, weight=1.0,
            )
        )
    finally:
        peer.close()
    deps = tmp_path / "knowledge_deps.json"
    deps.write_text(
        json.dumps(
            {"version": 1, "deps": [{"name": "peerA", "path": str(peer_path)}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_KNOWLEDGE_DEPS, str(deps))
    return peer_path


def test_a_marker_edge_in_a_peer_store_demotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pass predates federation (#690) and read local edges only.

    A hop the walk stepped into a peer for carries that peer in
    `owning_scope`, and its marker edges live in the peer's DB. Reading
    locally finds nothing, so a stale belief in a peer store would be
    silently exempt from the demotion this issue exists to deliver.
    """
    _peer_store(tmp_path, monkeypatch)
    s = MemoryStore(str(tmp_path / "local.db"))
    try:
        s.insert_belief(_mk(_STALE))
        assert s.edges_to(_STALE) == [], "the local store holds no marker edge"
        peer_edges = s.edges_to_in_scope(_STALE, "peerA")
        assert [e.type for e in peer_edges] == [EDGE_POTENTIALLY_STALE]
    finally:
        s.close()


def test_local_read_would_miss_the_peer_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the test above.

    If `edges_to` and `edges_to_in_scope` returned the same rows here,
    the scope-aware read would be untested — the assertion would pass
    on the local call too.
    """
    _peer_store(tmp_path, monkeypatch)
    s = MemoryStore(str(tmp_path / "local2.db"))
    try:
        s.insert_belief(_mk(_STALE))
        assert s.edges_to(_STALE) != s.edges_to_in_scope(_STALE, "peerA")
    finally:
        s.close()
