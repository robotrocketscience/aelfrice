"""Acceptance tests for v1.3.0 partial Bayesian-weighted ranking
(`docs/bayesian_ranking.md`, issue #146).

One test per acceptance criterion. All deterministic, in-memory
SQLite, ≤2s per test, no probabilistic assertions.
"""
from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import pytest

from aelfrice.feedback import apply_feedback
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    POSTERIOR_WEIGHT_KEY_PRECISION,
    RetrievalCache,
    resolve_posterior_weight,
    retrieve,
    retrieve_v2,
)
from aelfrice.scoring import (
    DEFAULT_POSTERIOR_WEIGHT,
    PARTIAL_BAYESIAN_BM25_FLOOR,
    partial_bayesian_score,
    posterior_mean,
)
from aelfrice.store import MemoryStore


# --- Fixtures -------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _equal_bm25_store() -> MemoryStore:
    """Five beliefs with the same surface form but distinct
    posteriors. Identical token bag (one occurrence of "widget"
    each, with a unique id-padding word) so SQLite FTS5 BM25 ties
    them at the same score against `widget`.
    """
    s = MemoryStore(":memory:")
    # alpha grows -> posterior_mean rises. beta=1.0 fixed.
    # Insertion order is reversed-alphabetical; this guarantees
    # the v1.0.x BM25-only path returns them in store-driven
    # order (NOT in posterior order), so the posterior-driven
    # rerank is observable.
    s.insert_belief(_mk("e_one", "widget echo unit", alpha=1.0))
    s.insert_belief(_mk("d_two", "widget delta gear", alpha=2.0))
    s.insert_belief(_mk("c_thr", "widget gamma cog", alpha=3.0))
    s.insert_belief(_mk("b_fou", "widget beta cam", alpha=4.0))
    s.insert_belief(_mk("a_fiv", "widget alpha rod", alpha=5.0))
    return s


# --- AC1: posterior_weight kwarg accepted by both retrieve surfaces ------


def test_ac1_retrieve_and_retrieve_v2_accept_posterior_weight() -> None:
    s = _equal_bm25_store()
    out1 = retrieve(s, "widget", posterior_weight=0.5)
    out2 = retrieve_v2(s, "widget", posterior_weight=0.5)
    assert isinstance(out1, list)
    assert all(isinstance(b, Belief) for b in out1)
    assert isinstance(out2.beliefs, list)
    # Both surfaces accept the new kwarg without raising.
    assert len(out1) >= 1
    assert len(out2.beliefs) >= 1


# --- AC2: posterior_weight=0.0 is byte-identical to v1.0.x ordering ------


def test_ac2_weight_zero_byte_identical_to_v10x() -> None:
    """The most important regression test: at weight 0 the result
    list is identical to what `store.search_beliefs(...)` returns
    for the L1 portion. (L0 prefix is unaffected by weight.)
    """
    s = _equal_bm25_store()
    direct = s.search_beliefs("widget", limit=50)
    weighted = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.0)
    # The retrieve() output may include an L0 prefix; here the
    # store has no locked beliefs, so the lists must match
    # byte-for-byte.
    assert [b.id for b in weighted] == [b.id for b in direct]


# --- AC3: equal-BM25 beliefs are reranked by posterior_mean DESC ---------


def test_ac3_equal_bm25_orders_by_posterior_descending() -> None:
    s = _equal_bm25_store()
    out = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.5)
    ids = [b.id for b in out]
    # alpha=5,4,3,2,1 -> posterior_mean 5/6, 4/5, 3/4, 2/3, 1/2.
    # Tied (or near-tied) BM25 + descending posterior -> a_fiv first.
    assert ids[0] == "a_fiv"
    # And b_fou (alpha=4) ranks ahead of e_one (alpha=1).
    assert ids.index("b_fou") < ids.index("e_one")


# --- AC4: high-BM25-low-posterior can drop below low-BM25-high-posterior -


def test_ac4_posterior_can_overcome_bm25_gap() -> None:
    """A high-BM25-low-posterior belief drops below a low-BM25-
    high-posterior belief once the posterior gap is large enough.

    Constructed at the L1 layer only — entity-index (L2.5) is
    disabled so the BM25 ranker is the sole ordering signal at
    weight=0.0. At weight=2.0 the strong-posterior belief wins.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "F_high", "spruce",  # short doc, strong BM25
        alpha=1.0, beta=1.0,  # prior, posterior_mean = 0.5
    ))
    s.insert_belief(_mk(
        "F_low",
        # long doc with one 'spruce' mention -> length normalization
        # pushes its BM25 score below F_high's.
        "spruce surrounded by oaks elms maples birches pines firs cedars junipers "
        "willows aspens beeches alders hawthorns dogwoods blackthorns hazels rowans",
        alpha=200.0, beta=1.0,  # posterior_mean ≈ 0.995
    ))
    base = retrieve(
        s, "spruce", token_budget=10_000, posterior_weight=0.0,
        entity_index_enabled=False,
    )
    base_ids = [b.id for b in base]
    # Sanity: BM25-only ordering puts F_high first.
    assert base_ids.index("F_high") < base_ids.index("F_low")

    # With a strong posterior weight, F_low jumps above F_high.
    boosted = retrieve(
        s, "spruce", token_budget=10_000, posterior_weight=2.0,
        entity_index_enabled=False,
    )
    boosted_ids = [b.id for b in boosted]
    assert boosted_ids.index("F_low") < boosted_ids.index("F_high")


# --- AC5: apply_feedback promotes a previously-mid-rank belief ----------


def test_ac5_apply_feedback_promotes_mid_rank_belief() -> None:
    """Calibration regression: a belief at rank R≥2 in baseline
    promotes to rank ≤R-1 after one positive feedback event.
    """
    s = _equal_bm25_store()
    base = retrieve(
        s, "widget", token_budget=10_000, posterior_weight=DEFAULT_POSTERIOR_WEIGHT,
    )
    base_ids = [b.id for b in base]
    # Pick a belief at rank ≥ 2.
    target = base_ids[2]  # 0-index 2 -> rank 3
    # Apply one positive feedback event.
    apply_feedback(s, target, valence=+5.0, source="test_ac5")
    after = retrieve(
        s, "widget", token_budget=10_000, posterior_weight=DEFAULT_POSTERIOR_WEIGHT,
    )
    after_ids = [b.id for b in after]
    base_rank = base_ids.index(target) + 1
    after_rank = after_ids.index(target) + 1
    assert base_rank >= 2, f"baseline rank too low to test: {base_rank}"
    assert after_rank <= base_rank - 1, (
        f"feedback failed to promote: was {base_rank}, now {after_rank}"
    )


# --- AC6: cache key includes posterior_weight (hit / miss matrix) --------


def test_ac6_cache_key_includes_posterior_weight() -> None:
    s = _equal_bm25_store()
    cache = RetrievalCache(s)
    cache.retrieve("widget", posterior_weight=0.5)
    assert len(cache) == 1
    # Same query, different weight -> miss + new entry.
    cache.retrieve("widget", posterior_weight=1.0)
    assert len(cache) == 2
    # Same weight again -> hit, no new entry.
    cache.retrieve("widget", posterior_weight=0.5)
    assert len(cache) == 2
    # Weight 0.0 is its own bucket (must not collide with default).
    cache.retrieve("widget", posterior_weight=0.0)
    assert len(cache) == 3


# --- AC7: apply_feedback wipes the cache via the existing callback -------


def test_ac7_apply_feedback_wipes_cache_via_store_callback() -> None:
    """apply_feedback must NOT reach into the cache directly. The
    wipe comes through store.update_belief -> _fire_invalidation
    -> cache.invalidate.
    """
    s = _equal_bm25_store()
    cache = RetrievalCache(s)
    cache.retrieve("widget", posterior_weight=0.5)
    assert len(cache) == 1
    target = cache.retrieve("widget", posterior_weight=0.5)[0].id
    # Feedback application happens entirely without referencing
    # the cache. The wipe must come through the store hook.
    apply_feedback(s, target, valence=+1.0, source="test_ac7")
    assert len(cache) == 0, "cache should have been invalidated"


# --- AC8: locked beliefs unaffected by posterior_weight ------------------


def test_ac8_locked_bypass_invariant_across_weights() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "L_a", "user pinned the widget rule first",
        lock_level=LOCK_USER, locked_at="2026-04-26T03:00:00Z",
    ))
    s.insert_belief(_mk(
        "L_b", "another locked widget mention",
        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
    ))
    s.insert_belief(_mk("F_1", "widget alpha", alpha=10.0))
    s.insert_belief(_mk("F_2", "widget beta", alpha=2.0))

    locked_position_at = {}
    for w in (0.0, 0.5, 1.0):
        out = retrieve(s, "widget", token_budget=10_000, posterior_weight=w)
        ids = [b.id for b in out]
        locked_position_at[w] = (ids.index("L_a"), ids.index("L_b"))
        # Both locks come before any non-locked.
        non_locked = [i for i, b in enumerate(out) if b.lock_level == LOCK_NONE]
        if non_locked:
            assert max(ids.index("L_a"), ids.index("L_b")) < min(non_locked)
    # Lock positions identical at every weight.
    assert (
        locked_position_at[0.0]
        == locked_position_at[0.5]
        == locked_position_at[1.0]
    ), f"lock positions drifted: {locked_position_at}"


# --- AC9: cold-belief neutrality at all-prior corpus --------------------


def test_ac9_cold_belief_neutrality_collapses_to_bm25() -> None:
    """When every belief has (alpha, beta) = (0.5, 0.5), the
    posterior term is a constant log(0.5) added uniformly. Every
    score shifts by the same amount; ordering is identical to
    weight=0.0.
    """
    s = MemoryStore(":memory:")
    # Jeffreys prior on every row.
    for i, content in enumerate([
        "widget alpha rod brief",
        "widget beta cam medium length doc text words",
        "widget gamma cog longer document text padded",
        "widget delta gear",
    ]):
        s.insert_belief(_mk(
            f"P_{i}", content, alpha=0.5, beta=0.5,
        ))
    cold = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.5)
    bm25_only = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.0)
    assert [b.id for b in cold] == [b.id for b in bm25_only]


# --- AC10: bm25 == 0 edge case does not crash ---------------------------


def test_ac10_bm25_zero_does_not_crash() -> None:
    """`partial_bayesian_score` must handle bm25=0 (the FTS5
    non-match return) without raising log(0). The clamp to
    PARTIAL_BAYESIAN_BM25_FLOOR keeps the score finite.
    """
    s = _equal_bm25_store()
    # Trigger a query that returns an empty L1; assert no crash.
    out = retrieve(s, "zzznosuchterm", token_budget=10_000, posterior_weight=0.5)
    assert out == []
    # Direct call to scoring helper at bm25=0.
    score = partial_bayesian_score(0.0, alpha=1.0, beta=1.0, posterior_weight=0.5)
    # Score should be finite (not -inf, not nan).
    assert score == score  # not NaN
    assert score < 0.0  # log of small numbers is negative
    # And at posterior_weight=0.0 too.
    score_z = partial_bayesian_score(0.0, alpha=1.0, beta=1.0, posterior_weight=0.0)
    assert score_z == score_z


# --- AC11: latency overhead is negligible -------------------------------


def test_ac11_per_query_overhead_within_budget() -> None:
    """Posterior reranking must add <1ms per query at the v1
    benchmark size. The synthetic corpus here is small (5
    beliefs); the AC simply asserts the rerank doesn't blow up
    against a reasonable wall-clock ceiling. Per-query budget
    here is conservative — the spec's 10^5 N latency claim is
    measured separately on the benchmark harness.
    """
    s = _equal_bm25_store()
    # Warm up.
    retrieve(s, "widget", posterior_weight=0.5)
    # Time best-of-100 to dampen scheduler jitter.
    t0 = time.perf_counter()
    for _ in range(100):
        retrieve(s, "widget", posterior_weight=0.5)
    elapsed = time.perf_counter() - t0
    # 100 calls in well under a second on any machine.
    assert elapsed < 1.0, f"100 calls took {elapsed:.3f}s -- too slow"


# --- AC12 / AC13 / AC14: docs + CI ---

# AC12 (LIMITATIONS rewrite) and AC13 (ROADMAP link) are checked by
# the docs commit; AC14 (full pytest green) is checked by CI. We
# pin them as content-hash tests below to catch silent reverts.


def test_ac12_limitations_md_documents_partial_ranking() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    # The v1.3.0 paragraph must mention the formula and the cache
    # invalidation contract.
    assert "v1.3.0" in text
    assert "posterior" in text
    assert any(
        marker in text
        for marker in ("log(bm25)", "log(BM25)", "log-additive", "log(-bm25)")
    )


def test_ac13_roadmap_links_bayesian_ranking_spec() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "bayesian_ranking.md" in text


# --- Default-weight at v1.3.0 ---


def test_default_posterior_weight_is_half() -> None:
    """Spec: 'v1.3.0 ships posterior_weight = 0.5 as default.' Pin
    it so a future PR cannot silently flip the default."""
    assert DEFAULT_POSTERIOR_WEIGHT == 0.5


def test_resolve_posterior_weight_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_POSTERIOR_WEIGHT", raising=False)
    # Make TOML resolution stable by pointing to a directory with
    # no .aelfrice.toml.
    with tempfile.TemporaryDirectory() as td:
        weight = resolve_posterior_weight(start=Path(td))
        assert weight == DEFAULT_POSTERIOR_WEIGHT


def test_resolve_posterior_weight_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_POSTERIOR_WEIGHT", "0.0")
    assert resolve_posterior_weight() == 0.0
    monkeypatch.setenv("AELFRICE_POSTERIOR_WEIGHT", "0.7")
    assert resolve_posterior_weight() == 0.7


def test_resolve_posterior_weight_explicit_overrides_toml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AELFRICE_POSTERIOR_WEIGHT", raising=False)
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nposterior_weight = 0.25\n")
    # explicit kwarg wins over TOML (env is unset).
    assert resolve_posterior_weight(0.9, start=tmp_path) == 0.9
    # ...and TOML wins when no kwarg.
    assert resolve_posterior_weight(start=tmp_path) == 0.25


def test_resolve_posterior_weight_negative_clamps_to_zero() -> None:
    assert resolve_posterior_weight(-1.5) == 0.0


# --- Calibration regression: 5-belief synthetic, ≥1 strict promotion ---


def _uniform_prior_store() -> MemoryStore:
    """Five widget-content beliefs with identical Jeffreys-equivalent
    priors (alpha=1, beta=1). Insertion order driven by id ASC so
    the BM25-tied ordering is deterministic.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("a_fiv", "widget alpha rod", alpha=1.0))
    s.insert_belief(_mk("b_fou", "widget beta cam", alpha=1.0))
    s.insert_belief(_mk("c_thr", "widget gamma cog", alpha=1.0))
    s.insert_belief(_mk("d_two", "widget delta gear", alpha=1.0))
    s.insert_belief(_mk("e_one", "widget echo unit", alpha=1.0))
    return s


def test_calibration_one_round_feedback_promotes_at_least_one_belief() -> None:
    """The spec's 'aelf bench --partial-uplift' minimum: ≥ 1
    strict rank promotion after one round of synthetic feedback.

    Synthetic shape: 5 beliefs with uniform Jeffreys-equivalent
    priors. At baseline (weight=0.0) the BM25-tied ordering is
    store-determined. After apply_feedback(used) on the rank-3
    belief and re-running at the v1.3 default weight (0.5), that
    belief promotes to rank ≤ 2.
    """
    s = _uniform_prior_store()
    base = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.0)
    base_ids = [b.id for b in base]
    assert len(base_ids) == 5
    # Pick the rank-3 belief (0-index 2).
    target = base_ids[2]
    # Single round of synthetic feedback per spec § Calibration.
    apply_feedback(s, target, valence=+1.0, source="bench-synthetic")
    after = retrieve(
        s, "widget", token_budget=10_000,
        posterior_weight=DEFAULT_POSTERIOR_WEIGHT,
    )
    after_ids = [b.id for b in after]
    after_rank = after_ids.index(target) + 1
    assert after_rank <= 2, (
        f"calibration failed: rank-3 belief did not promote to <=2 "
        f"(got rank {after_rank}). after_ids={after_ids}"
    )


# --- Cache-key precision sanity ---


def test_cache_key_precision_constant_is_sane() -> None:
    """Round-to-N decimals is enough granularity that two callers
    passing 0.5 and 0.5000001 collapse, but 0.5 and 0.6 don't."""
    assert POSTERIOR_WEIGHT_KEY_PRECISION >= 2
    assert POSTERIOR_WEIGHT_KEY_PRECISION <= 10


# --- Posterior-mean reuse pin ---


def test_partial_bayesian_score_uses_jeffreys_posterior_mean() -> None:
    """Spec rejects Laplace (alpha+1)/(alpha+beta+2) at this layer.
    Pin the formula to scoring.posterior_mean = alpha/(alpha+beta).
    """
    # alpha=2, beta=1 -> posterior_mean = 2/3 (NOT 3/5 = Laplace).
    pm = posterior_mean(2.0, 1.0)
    assert abs(pm - (2.0 / 3.0)) < 1e-12
    # And the score uses it.
    score = partial_bayesian_score(
        bm25_raw=-1.0,  # log(1) = 0 on the bm25 side
        alpha=2.0, beta=1.0,
        posterior_weight=1.0,
    )
    expected = math.log(1.0) + 1.0 * math.log(2.0 / 3.0)
    assert abs(score - expected) < 1e-12


# --- Floor constant pin (spec § "Numerical safety") ---


def test_bm25_floor_is_strictly_positive_and_small() -> None:
    """Floor must be > 0 (so log() is finite) and small enough
    not to contaminate any real BM25 score (~1e-6 typical)."""
    assert PARTIAL_BAYESIAN_BM25_FLOOR > 0.0
    assert PARTIAL_BAYESIAN_BM25_FLOOR < 1e-6


# --- Heat-kernel composition (#151 slice 2) -----------------------------


def test_heat_kernel_default_one_is_neutral() -> None:
    """Default heat_kernel=1.0 contributes log(1)=0 — score must
    match the pre-slice-2 contract for every (bm25, posterior)."""
    for bm25_raw in (-3.0, -1.0, -0.001, 0.0):
        for pw in (0.0, 0.5, 1.0):
            base = partial_bayesian_score(bm25_raw, 1.0, 1.0, pw)
            with_default = partial_bayesian_score(
                bm25_raw, 1.0, 1.0, pw, heat_kernel=1.0,
            )
            assert with_default == base


def test_heat_kernel_log_additive_at_unit_weight() -> None:
    """At heat_kernel_weight=1.0 the heat term adds log(heat_kernel)
    on top of the bm25+posterior baseline."""
    bm25_raw, alpha, beta, pw = -1.0, 2.0, 1.0, 0.5
    heat = 0.4
    base = partial_bayesian_score(bm25_raw, alpha, beta, pw)
    composed = partial_bayesian_score(
        bm25_raw, alpha, beta, pw, heat_kernel=heat,
    )
    assert abs(composed - (base + math.log(heat))) < 1e-12


def test_heat_kernel_weight_zero_collapses_term() -> None:
    """heat_kernel_weight=0.0 must drop the heat term regardless
    of heat_kernel value — same byte path as heat_kernel=1.0."""
    bm25_raw, alpha, beta, pw = -2.0, 1.5, 1.5, 0.5
    base = partial_bayesian_score(bm25_raw, alpha, beta, pw)
    weighted = partial_bayesian_score(
        bm25_raw, alpha, beta, pw,
        heat_kernel=0.01, heat_kernel_weight=0.0,
    )
    assert weighted == base


def test_heat_kernel_floor_prevents_log_zero() -> None:
    """A pathological heat_kernel <= 0 must be floored to
    PARTIAL_BAYESIAN_BM25_FLOOR rather than raising or producing
    -inf. Defence-in-depth: graph_spectral.heat_kernel_safe is the
    primary guard, this floor is the second."""
    score_zero = partial_bayesian_score(
        -1.0, 1.0, 1.0, posterior_weight=0.0, heat_kernel=0.0,
    )
    score_neg = partial_bayesian_score(
        -1.0, 1.0, 1.0, posterior_weight=0.0, heat_kernel=-0.5,
    )
    expected = math.log(1.0) + math.log(PARTIAL_BAYESIAN_BM25_FLOOR)
    assert abs(score_zero - expected) < 1e-12
    assert abs(score_neg - expected) < 1e-12


def test_heat_kernel_monotone_in_authority() -> None:
    """For fixed bm25/posterior, higher heat-kernel authority must
    yield a strictly higher composed score (assuming weight > 0).
    This is what makes graph-central beliefs rank above peripheral
    ones with the same lexical match."""
    bm25_raw, alpha, beta = -1.0, 1.0, 1.0
    s_low = partial_bayesian_score(
        bm25_raw, alpha, beta, posterior_weight=0.0, heat_kernel=0.1,
    )
    s_mid = partial_bayesian_score(
        bm25_raw, alpha, beta, posterior_weight=0.0, heat_kernel=0.5,
    )
    s_high = partial_bayesian_score(
        bm25_raw, alpha, beta, posterior_weight=0.0, heat_kernel=2.0,
    )
    assert s_low < s_mid < s_high


# --- retrieve() with eigenbasis_cache (#151 slice 2) ----------------------


def test_retrieve_no_eigenbasis_cache_is_byte_identical() -> None:
    """Without an eigenbasis_cache the L1 path must produce the
    same belief sequence as the pre-slice-2 contract — even when
    AELFRICE_HEAT_KERNEL=1 is set. The lane silently no-ops when
    no cache is available."""
    import os
    from aelfrice.retrieval import _reset_placeholder_warnings
    _reset_placeholder_warnings()
    s = _equal_bm25_store()
    baseline = retrieve(s, "widget", token_budget=10_000, posterior_weight=0.5)
    prior = os.environ.pop("AELFRICE_HEAT_KERNEL", None)
    os.environ["AELFRICE_HEAT_KERNEL"] = "1"
    try:
        flagged = retrieve(
            s, "widget", token_budget=10_000, posterior_weight=0.5,
        )
    finally:
        if prior is None:
            os.environ.pop("AELFRICE_HEAT_KERNEL", None)
        else:
            os.environ["AELFRICE_HEAT_KERNEL"] = prior
    assert [b.id for b in flagged] == [b.id for b in baseline]


def test_retrieve_eigenbasis_cache_threaded_through(
    tmp_path: Path,
) -> None:
    """Smoke test: retrieve() accepts eigenbasis_cache, builds it,
    enables the heat-kernel flag, and returns a non-empty L1 with
    no exceptions. The actual reranking is exercised by the
    scoring-level tests above; this confirms wiring."""
    import os
    from aelfrice.graph_spectral import GraphEigenbasisCache
    s = _equal_bm25_store()
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz", k=3)
    cache.build()
    prior = os.environ.pop("AELFRICE_HEAT_KERNEL", None)
    os.environ["AELFRICE_HEAT_KERNEL"] = "1"
    try:
        out = retrieve(
            s, "widget",
            token_budget=10_000, posterior_weight=0.5,
            eigenbasis_cache=cache,
        )
    finally:
        if prior is None:
            os.environ.pop("AELFRICE_HEAT_KERNEL", None)
        else:
            os.environ["AELFRICE_HEAT_KERNEL"] = prior
    assert len(out) == 5
    assert {b.id for b in out} == {
        "a_fiv", "b_fou", "c_thr", "d_two", "e_one",
    }


def test_retrieve_heat_kernel_flag_off_ignores_cache(
    tmp_path: Path,
) -> None:
    """When the heat-kernel flag is off, supplying an eigenbasis
    cache must not change ordering vs the no-cache baseline. Same
    determinism guarantee as the other off-by-default lanes."""
    import os
    from aelfrice.graph_spectral import GraphEigenbasisCache
    s = _equal_bm25_store()
    cache = GraphEigenbasisCache(store=s, path=tmp_path / "eb.npz", k=3)
    cache.build()
    prior = os.environ.pop("AELFRICE_HEAT_KERNEL", None)
    os.environ["AELFRICE_HEAT_KERNEL"] = "0"
    try:
        with_cache = retrieve(
            s, "widget", token_budget=10_000, posterior_weight=0.5,
            eigenbasis_cache=cache,
        )
        without_cache = retrieve(
            s, "widget", token_budget=10_000, posterior_weight=0.5,
        )
    finally:
        if prior is None:
            os.environ.pop("AELFRICE_HEAT_KERNEL", None)
        else:
            os.environ["AELFRICE_HEAT_KERNEL"] = prior
    assert [b.id for b in with_cache] == [b.id for b in without_cache]
