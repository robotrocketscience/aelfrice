"""Equivalence guard: `retrieve_v2` with all post-#1064 lanes OFF must be
byte-identical to the legacy `retrieve()` — the regression net the #1107
production cutover depends on.

The cutover migrates 8 production call sites from `retrieve()` (returns
`list[Belief]`) to `retrieve_v2()` (returns `RetrievalResult`). That is only
safe if `retrieve_v2`, with the lanes the hook does not want (temporal spine,
entity-persist demotion, origin tie-break, HRR-expand) forced off, produces
the same ranked ids as `retrieve()` for the same inputs. These tests pin that
equivalence across every tier and budget regime so a later change to either
implementation cannot silently diverge production retrieval.

`manifest_reference_locks` parity (#1016-B) was `retrieve()`-only until the
#1107 Phase-0 port; the manifest cases below would have failed before it.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_FROZEN,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    Belief,
    Edge,
    EDGE_DERIVED_FROM,
)
from aelfrice.retrieval import retrieve, retrieve_v2
from aelfrice.store import MemoryStore

# All post-#1064 lanes forced off — the configuration the production hook
# path needs after the cutover (byte-identical to today's retrieve()).
LANES_OFF = dict(
    use_temporal_spine=False,
    use_entity_persist_demote=False,
    use_origin_tiebreak=False,
    use_hrr_expand=False,
)


def _mk(
    store: MemoryStore,
    bid: str,
    content: str,
    *,
    lock: str = LOCK_NONE,
    tier: str = LOCK_TIER_FROZEN,
) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=9.0 if lock == LOCK_USER else 1.0,
            beta=0.5 if lock == LOCK_USER else 1.0,
            type=BELIEF_FACTUAL,
            lock_level=lock,
            locked_at="2026-06-01T00:00:00Z" if lock == LOCK_USER else None,
            created_at="2026-06-01T00:00:00Z",
            last_retrieved_at=None,
            lock_tier=tier,
        )
    )


def _v1(store: MemoryStore, query: str, **kw) -> list[str]:
    return [b.id for b in retrieve(store, query, **kw)]


def _v2(store: MemoryStore, query: str, **kw) -> list[str]:
    return [b.id for b in retrieve_v2(store, query, **{**LANES_OFF, **kw}).beliefs]


def test_equivalence_l1_vocab() -> None:
    """L1 BM25 path: shared and distinct vocabulary."""
    s = MemoryStore(":memory:")
    docs = [
        ("b1", "the widget module lives in src/widget.py"),
        ("b2", "widget rebased and pushed on branch feature/x"),
        ("b3", "the parser handles error code E42 in parser.py"),
        ("b4", "parser refactor merged in PR 991"),
        ("b5", "authentication uses a bearer token flow"),
        ("b6", "auth token refresh happens every hour"),
        ("b7", "the cache eviction policy is LRU with 500 entries"),
        ("b8", "retrieval ranks beliefs by BM25 then posterior"),
    ]
    for bid, c in docs:
        _mk(s, bid, c)
    try:
        for q in ("widget module", "parser error code", "auth token",
                  "cache eviction", "retrieval ranking", "the"):
            assert _v1(s, q, token_budget=2400) == _v2(s, q, budget=2400), q
    finally:
        s.close()


def test_equivalence_l0_locks() -> None:
    """L0 locked beliefs always win overflow in both implementations."""
    s = MemoryStore(":memory:")
    _mk(s, "L1", "the deploy runbook is in docs/deploy.md")
    _mk(s, "L2", "never force-push to main", lock=LOCK_USER)
    _mk(s, "L3", "deploy uses blue-green rollout")
    try:
        assert _v1(s, "deploy", token_budget=2400) == _v2(s, "deploy", budget=2400)
    finally:
        s.close()


@pytest.mark.parametrize("budget", [2400, 300, 120, 60])
def test_equivalence_budget_pressure(budget: int) -> None:
    """Tail-trim under a tight budget must match."""
    s = MemoryStore(":memory:")
    for i in range(12):
        _mk(s, f"p{i}", f"performance tuning note {i} about caching and latency")
    try:
        assert _v1(s, "performance caching latency", token_budget=budget) == _v2(
            s, "performance caching latency", budget=budget
        )
    finally:
        s.close()


def test_equivalence_locks_plus_budget() -> None:
    """A locked belief survives the trim in both; the rest match."""
    s = MemoryStore(":memory:")
    _mk(s, "k0", "cache latency locked rule", lock=LOCK_USER)
    for i in range(10):
        _mk(s, f"q{i}", f"cache latency tuning note {i}")
    try:
        assert _v1(s, "cache latency", token_budget=60) == _v2(
            s, "cache latency", budget=60
        )
    finally:
        s.close()


def test_equivalence_bfs() -> None:
    """L3 BFS traversal over derived-from edges must match."""
    s = MemoryStore(":memory:")
    _mk(s, "n1", "the auth service issues tokens")
    _mk(s, "n2", "tokens expire after one hour")
    _mk(s, "n3", "expiry is configurable via env")
    s.insert_edge(Edge(src="n2", dst="n1", type=EDGE_DERIVED_FROM, weight=1.0))
    s.insert_edge(Edge(src="n3", dst="n2", type=EDGE_DERIVED_FROM, weight=1.0))
    try:
        assert _v1(s, "auth service tokens", token_budget=2400, bfs_enabled=True) == _v2(
            s, "auth service tokens", budget=2400, use_bfs=True
        )
    finally:
        s.close()


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("budget", [2000, 1200, 900, 700])
def test_equivalence_manifest_reference_locks(manifest: bool, budget: int) -> None:
    """#1016-B manifest_reference_locks parity: with large reference-tier
    locks freeing relevance budget, retrieve_v2 must match retrieve() in
    BOTH manifest modes. This is the case that failed before the Phase-0
    port (retrieve_v2 had no manifest support)."""
    s = MemoryStore(":memory:")
    huge = " ".join(["cache latency deployment tuning directive"] * 80)
    for i in range(4):
        _mk(s, f"ref{i}", f"{huge} n{i}", lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    med = " ".join(["cache latency deployment note detail context"] * 10)
    for i in range(20):
        _mk(s, f"d{i}", f"{med} {i}")
    q = "cache latency deployment"
    try:
        assert _v1(s, q, token_budget=budget, manifest_reference_locks=manifest) == _v2(
            s, q, budget=budget, manifest_reference_locks=manifest
        )
    finally:
        s.close()


def test_manifest_flag_actually_changes_output() -> None:
    """Guard the guard: prove the manifest flag is load-bearing here, so the
    parity test above is not vacuously comparing two identical no-op runs."""
    s = MemoryStore(":memory:")
    huge = " ".join(["cache latency deployment tuning directive"] * 80)
    for i in range(4):
        _mk(s, f"ref{i}", f"{huge} n{i}", lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    med = " ".join(["cache latency deployment note detail context"] * 10)
    for i in range(20):
        _mk(s, f"d{i}", f"{med} {i}")
    q = "cache latency deployment"
    try:
        off = _v1(s, q, token_budget=1200, manifest_reference_locks=False)
        on = _v1(s, q, token_budget=1200, manifest_reference_locks=True)
        assert len(on) > len(off), (off, on)
    finally:
        s.close()
