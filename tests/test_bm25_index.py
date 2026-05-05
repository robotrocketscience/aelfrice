"""Tests for `aelfrice.bm25.BM25Index` covering acceptance criteria
of #148.

Coverage map (AC numbers from the issue body):

- AC1 / AC6 — `test_build_dimensions`, `test_build_rebuild_under_1s_at_n10k`
- AC2     — `test_augmented_doc_length_invariant`
- AC3     — `test_w0_topk_matches_fts5_baseline`
- AC4     — `test_vocab_shift_recovery_with_w3`
- AC5     — `test_score_latency_under_5ms_n50k` (gated on --run-perf)
- AC7     — `test_serialize_roundtrip_deterministic`

Plus L1-lane integration tests for the `use_bm25f_anchors` opt-in plumb
into `retrieve()`, and a rebuild-on-mutation test for the
`BM25IndexCache` invalidation hookup.

All tests are deterministic. Per project policy the perf test is
opt-in via `--run-perf`; in CI it is collected but skipped by
default to keep wall-clock under the 5s pytest timeout.
"""
from __future__ import annotations

import time

import pytest

from aelfrice.bm25 import (
    BM25Index,
    BM25IndexCache,
    DEFAULT_ANCHOR_WEIGHT,
    tokenize,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover
    """Hook so the --run-perf flag is accepted at the per-file level
    without polluting the suite-wide conftest. pytest collects this
    if the suite imports this file with `pytest -p tests.test_bm25_index`,
    but since we register it inline, we just guard the perf test with
    a skip when the flag is missing."""
    pass


def _has_run_perf(request: pytest.FixtureRequest) -> bool:
    try:
        return bool(request.config.getoption("--run-perf", default=False))
    except (AttributeError, ValueError):
        return False


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# --- AC1: build dimensions ------------------------------------------------


def test_build_dimensions() -> None:
    """AC1: row count equals belief count; column count equals
    canonicalised unique token count."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha beta gamma"))
    s.insert_belief(_mk("b2", "beta delta"))
    s.insert_belief(_mk("b3", "epsilon"))

    idx = BM25Index.build(s, anchor_weight=0)

    assert idx.tf.shape[0] == 3
    expected_tokens = {"alpha", "beta", "gamma", "delta", "epsilon"}
    assert idx.tf.shape[1] == len(expected_tokens)
    assert set(idx.vocabulary) == expected_tokens
    assert idx.belief_ids == ["b1", "b2", "b3"]


def test_build_empty_store_returns_empty_index() -> None:
    """An empty store produces a zero-shape index without raising."""
    s = MemoryStore(":memory:")
    idx = BM25Index.build(s)
    assert idx.tf.shape == (0, 0)
    assert idx.belief_ids == []
    assert idx.score("anything") == []


# --- AC2: augmented-doc length invariant ----------------------------------


def test_augmented_doc_length_invariant() -> None:
    """AC2: len(augmented) == len(content_tokens) + W * total_anchor_tokens."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "the quick fox"))      # 3 tokens
    s.insert_belief(_mk("b2", "lazy dog"))
    # Two incoming edges with 2 + 3 = 5 anchor tokens total.
    s.insert_edge(Edge(src="b2", dst="b1", type="cites",
                       weight=1.0, anchor_text="brown fox"))
    s.insert_edge(Edge(src="b2", dst="b1", type="supports",
                       weight=1.0, anchor_text="quick brown fox"))

    w = 3
    idx = BM25Index.build(s, anchor_weight=w)
    # Expected augmented length for b1 = 3 + 3 * 5 = 18.
    b1_idx = idx.belief_ids.index("b1")
    assert int(idx.dl[b1_idx]) == 3 + w * 5


# --- AC3: W=0 equivalence with FTS5 ---------------------------------------


def test_w0_topk_matches_fts5_baseline() -> None:
    """AC3: with W=0 and no anchor text in the corpus, BM25Index
    returns the same top-K belief set as the FTS5 BM25 path."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "kitchen full of bananas"))
    s.insert_belief(_mk("b2", "garage full of tools"))
    s.insert_belief(_mk("b3", "garden grows tomatoes"))

    idx = BM25Index.build(s, anchor_weight=0)
    bm25_hits = {bid for bid, _ in idx.score("bananas", top_k=10)}
    fts_hits = {b.id for b in s.search_beliefs("bananas", limit=10)}
    assert bm25_hits == fts_hits

    bm25_hits_2 = {bid for bid, _ in idx.score("garage tools", top_k=10)}
    fts_hits_2 = {b.id for b in s.search_beliefs("garage tools", limit=10)}
    assert bm25_hits_2 == fts_hits_2


# --- AC4: vocab-shift recovery -------------------------------------------


def test_vocab_shift_recovery_with_w3() -> None:
    """AC4: a vocab-shifted belief cited under topic-correct anchor
    text appears in the BM25F top-K with W=3 but not with W=0."""
    s = MemoryStore(":memory:")
    # Topic = "neural networks". Three citers describe the
    # vocab-shifted belief in topic terms; the belief itself uses
    # esoteric jargon.
    s.insert_belief(_mk("citer1", "neural networks discussion notes"))
    s.insert_belief(_mk("citer2", "more neural networks reading"))
    s.insert_belief(_mk("citer3", "topic on neural networks"))
    s.insert_belief(_mk("shifted", "perceptron synapse foobarbaz xyzzy"))

    for src in ("citer1", "citer2", "citer3"):
        s.insert_edge(Edge(
            src=src, dst="shifted", type="cites", weight=1.0,
            anchor_text="neural networks",
        ))

    idx_w0 = BM25Index.build(s, anchor_weight=0)
    idx_w3 = BM25Index.build(s, anchor_weight=3)

    top_w0 = [bid for bid, _ in idx_w0.score("neural networks", top_k=10)]
    top_w3 = [bid for bid, _ in idx_w3.score("neural networks", top_k=10)]

    assert "shifted" not in top_w0
    assert "shifted" in top_w3


# --- AC5: latency micro-benchmark (opt-in) --------------------------------


def test_score_latency_under_5ms_n50k(request: pytest.FixtureRequest) -> None:
    """AC5: median sparse-matvec score latency <= 5ms at N=50k.

    Skipped by default to keep CI under the per-test 5s wall-clock
    cap. Run locally with `pytest --run-perf`.
    """
    if not _has_run_perf(request):
        pytest.skip("perf test gated on --run-perf")
    s = MemoryStore(":memory:")
    n = 50_000
    for i in range(n):
        s.insert_belief(_mk(f"b{i:06d}", f"token{i % 1000} content blob"))
    idx = BM25Index.build(s, anchor_weight=0)
    # Warm up and measure 25 queries; report the median.
    samples: list[float] = []
    for k in range(25):
        t0 = time.perf_counter()
        idx.score(f"token{k}", top_k=50)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    assert median <= 5.0, f"median latency {median:.2f}ms exceeds 5ms"


# --- AC6: rebuild speed --------------------------------------------------


def test_build_under_1s_at_n10k() -> None:
    """AC6: index rebuild completes in <= 1s at N=10k. Run inline
    (not perf-gated) since 10k inserts + build still finishes well
    under the per-test 5s timeout."""
    s = MemoryStore(":memory:")
    n = 10_000
    for i in range(n):
        s.insert_belief(_mk(f"b{i:05d}", f"token{i % 256} content"))
    t0 = time.perf_counter()
    BM25Index.build(s, anchor_weight=0)
    elapsed = time.perf_counter() - t0
    assert elapsed <= 1.0, f"build took {elapsed:.2f}s at N=10k (> 1s)"


# --- AC7: deterministic serialise/deserialise ----------------------------


def test_serialize_roundtrip_deterministic() -> None:
    """AC7: same inputs produce same bytes; deserialize is the
    exact inverse."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha beta gamma"))
    s.insert_belief(_mk("b2", "alpha gamma delta"))
    s.insert_edge(Edge(src="b2", dst="b1", type="cites",
                       weight=1.0, anchor_text="alpha gamma"))

    idx_a = BM25Index.build(s, anchor_weight=DEFAULT_ANCHOR_WEIGHT)
    idx_b = BM25Index.build(s, anchor_weight=DEFAULT_ANCHOR_WEIGHT)
    blob_a = idx_a.serialize()
    blob_b = idx_b.serialize()
    assert blob_a == blob_b

    restored = BM25Index.deserialize(blob_a)
    assert restored.belief_ids == idx_a.belief_ids
    assert restored.vocabulary == idx_a.vocabulary
    assert restored.anchor_weight == idx_a.anchor_weight
    assert restored.dl.tolist() == idx_a.dl.tolist()
    assert restored.idf.tolist() == idx_a.idf.tolist()
    assert (restored.tf - idx_a.tf).nnz == 0
    # Restored scores match original scores byte-for-byte.
    for q in ("alpha", "beta gamma", "delta"):
        assert restored.score(q, top_k=10) == idx_a.score(q, top_k=10)


# --- Tokeniser sanity ----------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("") == []
    assert tokenize("   ") == []


# --- BM25IndexCache + retrieve() integration -----------------------------


def test_cache_rebuilds_on_store_mutation() -> None:
    """The cache subscribes to invalidation; mutations drop the index."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "alpha"))
    cache = BM25IndexCache(s)
    idx_a = cache.get()
    assert idx_a is cache.get()  # second call returns same cached index
    s.insert_belief(_mk("b2", "beta"))
    idx_b = cache.get()
    assert idx_b is not idx_a
    assert idx_b.belief_ids == ["b1", "b2"]


def test_retrieve_with_use_bm25f_anchors_returns_anchor_recovered_belief() -> None:
    """retrieve(..., use_bm25f_anchors=True) surfaces a vocab-shifted
    belief that the FTS5 path with use_bm25f_anchors=False does not."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("citer1", "neural networks discussion"))
    s.insert_belief(_mk("citer2", "more neural networks reading"))
    s.insert_belief(_mk("citer3", "topic on neural networks"))
    s.insert_belief(_mk("shifted", "perceptron synapse foobarbaz xyzzy"))
    for src in ("citer1", "citer2", "citer3"):
        s.insert_edge(Edge(
            src=src, dst="shifted", type="cites", weight=1.0,
            anchor_text="neural networks",
        ))

    bm25f_hits = {b.id for b in retrieve(
        s, "neural networks", use_bm25f_anchors=True,
    )}
    fts_hits = {b.id for b in retrieve(
        s, "neural networks", use_bm25f_anchors=False,
    )}
    assert "shifted" in bm25f_hits
    assert "shifted" not in fts_hits


def test_retrieve_default_on_byte_identical_to_explicit_on() -> None:
    """Default-on contract (#154 v1.7.0): with use_bm25f_anchors unset
    (None), the output equals the explicit use_bm25f_anchors=True path.

    Replaces the v1.5/v1.6 default-off byte-identity check. Default
    flipped on +0.6650 NDCG@k uplift evidence on the
    `tests/corpus/v2_0/retrieve_uplift/v0_1.jsonl` lab fixture
    post-stemming (see #154 comment 4380967901). Guards against
    accidental flip-back in a future commit; the legacy FTS5 path
    remains reachable via explicit `use_bm25f_anchors=False`.
    """
    s = MemoryStore(":memory:")
    for i in range(8):
        s.insert_belief(_mk(f"b{i}", f"token{i} content blob"))
    default = [b.id for b in retrieve(s, "token3 content")]
    explicit_on = [b.id for b in retrieve(
        s, "token3 content", use_bm25f_anchors=True,
    )]
    assert default == explicit_on
