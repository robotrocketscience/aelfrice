"""Equivalence guard: `retrieve()` (the production adapter) must be
byte-identical to `retrieve_v2` run with the exact lane config the #1107 shim
pins — the regression net the production cutover depends on.

The cutover migrated the production call sites from the legacy bare
`retrieve()` pack loop to a thin adapter over `retrieve_v2()`. Post-#1107
cutover the shim runs the graduated lanes ON (resolver-driven) — **temporal
spine** (#1064, Phase 2), **entity-persist demotion** (#1096, Phase 3), and
**intentional clustering** (#436, Phase 4) — and the remaining three staged
lanes (origin tie-break, HRR-expand, HRR-structural) OFF. `SHIM_LANES` below
is that exact config, so `retrieve() == retrieve_v2(**SHIM_LANES)` is a true
identity by construction (both route through `retrieve_v2` with the same
lanes); the L0/L1/L2.5/BFS/manifest cases pin the per-tier behaviour under
that shared config and guard against drift between the shim's inline lane
config and `SHIM_LANES`. `test_shim_runs_graduated_lanes_others_off` covers
each graduated lane being live and a held lane staying off, non-vacuously.

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
    EDGE_TEMPORAL_NEXT,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
)
from aelfrice.retrieval import retrieve, retrieve_v2
from aelfrice.store import MemoryStore

# The production lane config the #1107 shim pins, verbatim. The graduated
# lanes are resolver-driven (`None` -> resolver, default ON): temporal spine
# (#1064), entity-persist demotion (#1096), intentional clustering (#436). The
# other three staged lanes stay forced OFF because `retrieve()`'s historical
# pack loop never ran them. Because this mirrors the shim exactly,
# `retrieve() == retrieve_v2(**SHIM_LANES)` holds by construction whether or
# not a given corpus makes a lane fire — the equivalence cases pin the per-tier
# behaviour, not lane no-op-ness. Keep this in lock-step with the shim's inline
# `retrieve_v2(...)` lane kwargs.
SHIM_LANES = dict(
    use_temporal_spine=None,
    use_entity_persist_demote=None,
    use_intentional_clustering=None,
    use_origin_tiebreak=False,
    use_hrr_expand=False,
    use_hrr_structural=False,
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


def _add_entity(
    store: MemoryStore, bid: str, lower: str, kind: str,
) -> None:
    """Insert a belief_entities row (entity-persist demotion reads these)."""
    store._conn.execute(
        "INSERT INTO belief_entities(belief_id, entity_lower, entity_raw, "
        "kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
        (bid, lower, lower, kind),
    )
    store._conn.commit()


def _v1(store: MemoryStore, query: str, **kw) -> list[str]:
    return [b.id for b in retrieve(store, query, **kw)]


def _v2(store: MemoryStore, query: str, **kw) -> list[str]:
    return [b.id for b in retrieve_v2(store, query, **{**SHIM_LANES, **kw}).beliefs]


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


def test_equivalence_empty_query() -> None:
    """Empty query → L0 only. The hook.py SessionStart baseline calls
    retrieve(store, "", ...), so this path must match exactly."""
    s = MemoryStore(":memory:")
    _mk(s, "lk1", "never force-push to main", lock=LOCK_USER)
    _mk(s, "lk2", "deploy only from the release branch", lock=LOCK_USER)
    _mk(s, "u1", "an unlocked note about caching")
    try:
        assert _v1(s, "", token_budget=2400) == _v2(s, "", budget=2400)
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


def test_equivalence_clustered_corpus() -> None:
    """Two dense DERIVED_FROM clusters — the case where intentional
    clustering reorders the L1 pack. Post-#1107 Phase 4 `retrieve()` runs
    clustering (graduated), and `SHIM_LANES` carries clustering ON too, so
    both cluster identically and match. A config whose clustering flag drifts
    from the shim's is exactly what silently diverges here."""
    s = MemoryStore(":memory:")
    for i in range(6):
        _mk(s, f"a{i}", f"cache latency tuning cluster note {i}")
    for i in range(6):
        _mk(s, f"b{i}", f"cache latency deployment cluster note {i}")
    for i in range(5):
        s.insert_edge(Edge(src=f"a{i+1}", dst=f"a{i}", type=EDGE_DERIVED_FROM, weight=1.0))
        s.insert_edge(Edge(src=f"b{i+1}", dst=f"b{i}", type=EDGE_DERIVED_FROM, weight=1.0))
    try:
        for q in ("cache latency", "cache latency deployment tuning", "note"):
            assert _v1(s, q, token_budget=2400) == _v2(s, q, budget=2400), q
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


def test_shim_runs_graduated_lanes_others_off() -> None:
    """Contract guard for the #1107 shim: `retrieve()` (the production
    adapter) runs the graduated lanes ON (resolver default) and the remaining
    held lanes OFF. Every clause is non-vacuous — each isolates one lane while
    the others resolve identically on both sides (common-mode), so the diff is
    attributable to the named lane:

      * spine ON — on a `TEMPORAL_NEXT`-connected corpus, `retrieve()`
        surfaces the chronological neighbour that shares no query vocabulary,
        matching retrieve_v2(spine on) and differing from spine off.
      * entity-persist ON — on a corpus with a durable-grounded and a
        coordination-grounded belief tied on relevance, `retrieve()` demotes
        the coordination one, matching retrieve_v2(demote on) and differing
        from demote off.
      * clustering ON — on two dense DERIVED_FROM clusters where clustering
        reorders the L1 pack, `retrieve()` matches retrieve_v2(clustering on)
        and differs from clustering off.
      * a held lane OFF — origin tie-break: on two same-content beliefs with
        different origins, `retrieve()` matches retrieve_v2(tie-break off) and
        differs from tie-break on, proving the held lanes are not silently on.

    If a graduated lane regresses off, or a held lane is flipped on in the
    shim, one of these fails."""
    # --- temporal-spine lane is live in the shim ---
    s = MemoryStore(":memory:")
    _mk(s, "anchor", "cache latency tuning note")
    _mk(s, "neighbor", "zebra quokka xylophone unrelated marmalade")
    # A single spine edge; the lane traverses depth-1 in both directions.
    s.insert_edge(
        Edge(src="anchor", dst="neighbor",
             type=EDGE_TEMPORAL_NEXT, weight=0.8)
    )
    q = "cache latency"
    try:
        prod = [b.id for b in retrieve(s, q, token_budget=2400)]
        spine_on = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_temporal_spine": True}).beliefs
        ]
        spine_off = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_temporal_spine": False}).beliefs
        ]
        assert "neighbor" in prod, "shim must run the temporal-spine lane"
        assert prod == spine_on
        assert prod != spine_off, "spine lane must be load-bearing here"
    finally:
        s.close()

    # --- entity-persist demotion lane is live in the shim ---
    s = MemoryStore(":memory:")
    _mk(s, "durable", "widget the module lives here")
    _mk(s, "ephemeral", "widget rebased and pushed")
    _add_entity(s, "durable", "src/widget.py", "file_path")   # durable
    _add_entity(s, "ephemeral", "#412", "identifier")         # bare-# transient
    q = "widget"
    try:
        prod = [b.id for b in retrieve(s, q, token_budget=2400)]
        demote_on = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_entity_persist_demote": True}).beliefs
        ]
        demote_off = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_entity_persist_demote": False}).beliefs
        ]
        assert prod == demote_on
        assert prod.index("durable") <= prod.index("ephemeral"), (
            "shim must run entity-persist demotion (coordination belief demoted)"
        )
        assert prod != demote_off, "demotion must be load-bearing here"
    finally:
        s.close()

    # --- intentional clustering lane is live in the shim ---
    s = MemoryStore(":memory:")
    for i in range(6):
        _mk(s, f"a{i}", f"cache latency tuning cluster note {i}")
    for i in range(6):
        _mk(s, f"b{i}", f"cache latency deployment cluster note {i}")
    for i in range(5):
        s.insert_edge(Edge(src=f"a{i+1}", dst=f"a{i}", type=EDGE_DERIVED_FROM, weight=1.0))
        s.insert_edge(Edge(src=f"b{i+1}", dst=f"b{i}", type=EDGE_DERIVED_FROM, weight=1.0))
    q = "cache latency"
    try:
        prod = [b.id for b in retrieve(s, q, token_budget=2400)]
        # Explicit off-arm — SHIM_LANES now carries clustering ON, so `_v2`
        # would cluster too; pin the off arm directly.
        clustering_on = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_intentional_clustering": True}).beliefs
        ]
        clustering_off = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_intentional_clustering": False}).beliefs
        ]
        assert prod == clustering_on, "shim must run the intentional-clustering lane"
        assert prod != clustering_off, "clustering must be load-bearing here"
    finally:
        s.close()

    # --- a held lane (origin tie-break) stays off in the shim ---
    s = MemoryStore(":memory:")
    # Same content ties on BM25; ids sort transcript-first. Origin tie-break,
    # if on, lifts the curated (higher-trust) belief above the transcript one.
    for bid, origin in (
        ("aaa_transcript", ORIGIN_USER_TRANSCRIPT),
        ("zzz_curated", ORIGIN_USER_VALIDATED),
    ):
        s.insert_belief(Belief(
            id=bid, content="widget configuration lives here",
            content_hash=f"h_{bid}", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE, locked_at=None,
            created_at="2026-06-01T00:00:00Z", last_retrieved_at=None,
            origin=origin,
        ))
    q = "widget configuration"
    try:
        prod = [b.id for b in retrieve(s, q, token_budget=2400)]
        tiebreak_on = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_origin_tiebreak": True}).beliefs
        ]
        tiebreak_off = [
            b.id for b in retrieve_v2(
                s, q, budget=2400,
                **{**SHIM_LANES, "use_origin_tiebreak": False}).beliefs
        ]
        assert prod == tiebreak_off, "retrieve() must have origin tie-break OFF"
        assert prod != tiebreak_on, "tie-break must actually reorder here"
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
