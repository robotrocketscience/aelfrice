"""#1135 persistent BM25F sidecar: `BM25IndexCache` load/persist.

Contract under test: a built index is persisted next to the DB and a
fresh cache (fresh process stand-in) loads it instead of rebuilding;
any content mutation invalidates via the durable generation stamp;
mismatched parameters, corrupt blobs, and in-memory stores all fall
back to a build; a loaded index scores byte-identically to a built
one (the retrieval byte-identity AC depends on this).
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.bm25 import BM25Index, BM25IndexCache, sidecar_path_for
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-21T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(store: MemoryStore) -> None:
    store.insert_belief(_mk_belief("b1", "the quick brown fox jumps"))
    store.insert_belief(_mk_belief("b2", "sqlite stores beliefs durably"))
    store.insert_belief(_mk_belief("b3", "the fox likes sqlite"))


def test_sidecar_written_on_build(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed(store)
        cache = BM25IndexCache(store)
        cache.get()
        sidecar = sidecar_path_for(store)
        assert sidecar is not None and sidecar.is_file()
        assert sidecar.stat().st_size > 0
    finally:
        store.close()


def test_fresh_cache_loads_sidecar_without_building(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    try:
        _seed(store)
        BM25IndexCache(store).get()  # builds + persists
    finally:
        store.close()

    # Fresh store + fresh cache = fresh hook process. A build here
    # means the sidecar was not honoured.
    builds: list[int] = []
    real_build = BM25Index.build

    def counting_build(*args: object, **kwargs: object) -> BM25Index:
        builds.append(1)
        return real_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BM25Index, "build", counting_build)
    store2 = MemoryStore(str(db))
    try:
        idx = BM25IndexCache(store2).get()
        assert builds == [], "sidecar present and current, but build ran"
        assert idx.score("fox", top_k=3)  # loaded index actually works
    finally:
        store2.close()


def test_loaded_index_scores_identical_to_built(tmp_path: Path) -> None:
    """Byte-identity AC: the sidecar path must not perturb ranking."""
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    try:
        _seed(store)
        built = BM25IndexCache(store).get()
        loaded = BM25IndexCache(store)._load_sidecar()
        assert loaded is not None
        for query in ("fox", "sqlite beliefs", "quick brown sqlite"):
            assert built.score(query, top_k=10) == loaded.score(
                query, top_k=10,
            )
    finally:
        store.close()


def test_mutation_invalidates_sidecar_via_generation(
    tmp_path: Path,
) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    try:
        _seed(store)
        BM25IndexCache(store).get()
        gen_at_build = store.store_generation()
        store.insert_belief(_mk_belief("b4", "a brand new belief row"))
        assert store.store_generation() > gen_at_build
        # A fresh cache must reject the stale sidecar and rebuild —
        # the new belief has to be retrievable.
        idx = BM25IndexCache(store)._load_sidecar()
        assert idx is None, "stale sidecar accepted after mutation"
        rebuilt = BM25IndexCache(store).get()
        assert any(
            bid == "b4" for bid, _ in rebuilt.score("brand new belief", top_k=5)
        )
    finally:
        store.close()


def test_anchor_weight_mismatch_rejects_sidecar(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    try:
        _seed(store)
        BM25IndexCache(store, anchor_weight=3).get()
        assert BM25IndexCache(store, anchor_weight=5)._load_sidecar() is None
    finally:
        store.close()


def test_corrupt_sidecar_falls_back_to_build(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    try:
        _seed(store)
        cache = BM25IndexCache(store)
        cache.get()
        sidecar = sidecar_path_for(store)
        assert sidecar is not None
        sidecar.write_bytes(b"garbage not an index blob")
        fresh = BM25IndexCache(store)
        assert fresh._load_sidecar() is None
        idx = fresh.get()  # must build, not raise
        assert idx.score("fox", top_k=3)
    finally:
        store.close()


def test_memory_store_has_no_sidecar(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    try:
        _seed(store)
        assert sidecar_path_for(store) is None
        BM25IndexCache(store).get()  # no crash, no file
        assert list(tmp_path.iterdir()) == []
    finally:
        store.close()


def test_generation_bump_rides_mutation_transaction(
    tmp_path: Path,
) -> None:
    """The durable counter moves with every content mutation and holds
    still for non-content writes (feedback/touches don't reindex)."""
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        g0 = store.store_generation()
        store.insert_belief(_mk_belief("b1", "content one"))
        g1 = store.store_generation()
        assert g1 == g0 + 1
        with store.transaction():
            store.insert_belief(_mk_belief("b2", "content two"))
            store.insert_belief(_mk_belief("b3", "content three"))
        g2 = store.store_generation()
        assert g2 == g1 + 2
        store.stamp_retrieved(["b1"])  # ranking input, not index content
        assert store.store_generation() == g2
    finally:
        store.close()
