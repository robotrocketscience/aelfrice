"""Unit tests for HRR vocabulary bridge (#433).

Covers the algebraic invariants — self-recovery cosine, build
determinism from a path-derived seed, token-universe filter,
empty-store no-op — without depending on small-corpus cross-talk
magnitudes (which are inherent to HRR and corpus-size-dependent;
the bench gate checks signal-to-noise on a real corpus).
"""
from __future__ import annotations

import numpy as np
import pytest

from aelfrice.hrr import unbind
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.vocab_bridge import (
    DEFAULT_TOP_K,
    VocabBridge,
    _seed_from_path,
)


def _mk_belief(
    bid: str,
    content: str,
    *,
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
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class="fact",
    )


def _populate(store: MemoryStore) -> None:
    store.insert_belief(_mk_belief("b1", "SQLite is the storage substrate."))
    store.insert_belief(_mk_belief("b2", "Python provides the language."))
    store.insert_belief(_mk_belief("b3", "Numpy provides matrix algebra."))
    store.insert_belief(
        _mk_belief("b4", "User locked SQLite as canonical.", lock_level=LOCK_USER)
    )


# --- Build determinism -------------------------------------------------


def test_empty_store_builds_no_op_bridge() -> None:
    s = MemoryStore(":memory:")
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/empty")
    assert vb.size() == 0
    assert vb.canonicals == []
    assert vb.surfaces == []
    # No-op rewrite: empty bridge returns the query verbatim.
    assert vb.rewrite("anything") == "anything"


def test_build_is_deterministic_given_same_path() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    a = VocabBridge()
    a.build(s, store_path="/tmp/det-a")
    b = VocabBridge()
    b.build(s, store_path="/tmp/det-a")
    assert a.canonicals == b.canonicals
    assert a.surfaces == b.surfaces
    np.testing.assert_array_equal(a.bridge_vec, b.bridge_vec)
    for c in a.canonicals:
        np.testing.assert_array_equal(a.canonical_vecs[c], b.canonical_vecs[c])


def test_build_differs_across_paths() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    a = VocabBridge()
    a.build(s, store_path="/tmp/det-a")
    b = VocabBridge()
    b.build(s, store_path="/tmp/det-b")
    # Canonicals/surfaces are corpus-derived, identical across paths.
    assert a.canonicals == b.canonicals
    # bridge_vec should differ (different seed → different vectors).
    assert not np.array_equal(a.bridge_vec, b.bridge_vec)


def test_seed_from_path_is_64_bit_and_stable() -> None:
    s1 = _seed_from_path("/store/path")
    s2 = _seed_from_path("/store/path")
    assert s1 == s2
    assert 0 <= s1 <= 0xFFFFFFFFFFFFFFFF
    # Different paths fan out.
    assert _seed_from_path("/other") != s1
    # XOR salt fans out further.
    assert _seed_from_path("/store/path", salt=0xDEAD) != s1


def test_explicit_seed_overrides_store_path() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    a = VocabBridge()
    a.build(s, store_path="/tmp/a", seed=42)
    b = VocabBridge()
    b.build(s, store_path="/tmp/b", seed=42)
    np.testing.assert_array_equal(a.bridge_vec, b.bridge_vec)


# --- Algebraic invariants ----------------------------------------------


def test_self_recovery_cosine_above_noise_floor() -> None:
    """unbind(token_vec[t], bridge_vec) should recover canonical_vec[t]
    above the noise floor when t is itself a canonical surface form."""
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/recover")
    floor = vb.noise_floor()
    # Every canonical that is its own surface form (the standard MVP
    # case) should self-recover above the noise floor.
    for c in vb.canonicals:
        if c not in vb.token_vecs:
            continue
        recovered = unbind(vb.token_vecs[c], vb.bridge_vec)
        norm = float(np.linalg.norm(recovered))
        assert norm > 0.0
        cv = vb.canonical_vecs[c]
        cosine = float(
            (recovered / norm) @ (cv / float(np.linalg.norm(cv)))
        )
        assert cosine > floor, (
            f"self-recovery for {c!r} cosine {cosine:.4f} <= floor "
            f"{floor:.4f}"
        )


def test_unseen_token_does_not_appear_in_rewrite() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/unseen")
    out = vb.rewrite("xxnotpresentxx")
    # Unseen tokens short-circuit — query is preserved.
    assert out == "xxnotpresentxx"


def test_short_tokens_below_three_chars_drop() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/short")
    # Query "is the" has tokens "is" and "the", but only "the" is
    # ≥3 chars and gets harvested; "is" is filtered out.
    assert "is" not in vb.canonicals
    # "the" passes the filter.
    assert "the" in vb.canonicals or "the" in vb.surfaces


def test_rewrite_preserves_original_query() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/preserve")
    out = vb.rewrite("sqlite", top_k=3)
    # The original token is always the first whitespace-separated
    # member of the rewritten query.
    assert out.split()[0] == "sqlite"


def test_rewrite_is_deterministic() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/det-rewrite")
    a = vb.rewrite("sqlite python", top_k=3)
    b = vb.rewrite("sqlite python", top_k=3)
    assert a == b


def test_high_min_score_drops_all_appendees() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/strict")
    # Threshold 1.5 is unreachable; nothing passes — query verbatim.
    assert vb.rewrite("sqlite", min_score=1.5) == "sqlite"


def test_top_k_zero_short_circuits() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/topk0")
    assert vb.rewrite("sqlite", top_k=0) == "sqlite"


def test_token_vecs_match_canonical_vecs_for_self_canonicals() -> None:
    """In the MVP, every canonical is its own surface form. Their
    token_vec and canonical_vec are different draws (different
    Generators), so they must NOT be equal."""
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/distinct-streams")
    for c in vb.canonicals:
        tv = vb.token_vecs.get(c)
        cv = vb.canonical_vecs.get(c)
        if tv is None or cv is None:
            continue
        assert not np.array_equal(tv, cv), (
            f"{c!r} token_vec and canonical_vec collided — "
            f"streams should be distinct"
        )


# --- Anchor-text harvest ------------------------------------------------


def test_anchor_text_is_a_surface_source() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1", "Storage layer details."))
    s.insert_belief(_mk_belief("b2", "Performance discussion."))
    # Add an edge with anchor_text mentioning a token absent from
    # belief contents — the bridge should still harvest it.
    s.insert_edge(
        Edge(src="b2", dst="b1", type="CITES", weight=1.0,
             anchor_text="see also: dynamodb migration notes"),
    )
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/anchor")
    # "dynamodb" is harvested from anchor_text only.
    assert "dynamodb" in vb.canonicals or "dynamodb" in vb.surfaces


# --- Inspection / boundary --------------------------------------------


def test_size_reflects_canonical_count() -> None:
    s = MemoryStore(":memory:")
    _populate(s)
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/size")
    assert vb.size() == len(vb.canonicals)
    assert vb.size() > 0


def test_default_top_k_is_three() -> None:
    # Spec: "Default `3`."
    assert DEFAULT_TOP_K == 3


def test_noise_floor_matches_one_over_sqrt_dim() -> None:
    vb = VocabBridge(dim=2048)
    expected = 1.0 / float(np.sqrt(2048))
    assert vb.noise_floor() == pytest.approx(expected)


def test_rebuild_overwrites_previous_state() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1", "First content with database tokens."))
    vb = VocabBridge()
    vb.build(s, store_path="/tmp/rebuild")
    first_canonicals = list(vb.canonicals)
    # Add a new belief and rebuild.
    s.insert_belief(_mk_belief("b2", "Second content with cache tokens."))
    vb.build(s, store_path="/tmp/rebuild")
    assert set(vb.canonicals) >= set(first_canonicals)
    assert "cache" in vb.canonicals
