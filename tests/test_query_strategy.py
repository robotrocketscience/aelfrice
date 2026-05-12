"""Unit tests for the v1.7 (#291 PR-2) query-strategy dispatcher,
per-store BM25 + IDF-quantile cache, and rebuilder/hook plumbing.

Covers:
  - `transform_query` legacy passthrough (default, byte-identical)
  - `transform_query` stack-r1-r3 path applies R1 + R3 against the
    store's own IDF distribution
  - Unknown strategy raises
  - `store_cache.get_bm25_and_quantiles` is cached per store and
    invalidates on belief mutation via the existing callback chain
  - `RebuilderConfig.query_strategy` parses + validates from
    `[rebuilder] query_strategy` in `.aelfrice.toml`
  - `rebuild_v14(query_strategy=...)` produces the same result as
    `legacy-bm25` when no opt-in is requested (no behavior change at
    default), and routes through the stack on opt-in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    RebuilderConfig,
    RecentTurn,
    load_rebuilder_config,
    rebuild_v14,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.query_understanding import (
    DEFAULT_STRATEGY,
    LEGACY_STRATEGY,
    STACK_R1_R3_STRATEGY,
    VALID_STRATEGIES,
    transform_query,
)
from aelfrice.query_understanding.store_cache import (
    _cache_size_for_test,
    get_bm25_and_quantiles,
    invalidate,
)
from aelfrice.store import MemoryStore


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


def _seed(db_path: Path, beliefs: list[Belief]) -> MemoryStore:
    store = MemoryStore(str(db_path))
    for b in beliefs:
        store.insert_belief(b)
    return store


# --- transform_query --------------------------------------------------------


def test_transform_legacy_returns_query_unchanged(tmp_path: Path) -> None:
    store = _seed(
        tmp_path / "m.db",
        [_mk("a", "rebuilder bayesian inference")],
    )
    try:
        out = transform_query("Why does the Rebuilder use Bayesian inference",
                              store, LEGACY_STRATEGY)
        assert out == "Why does the Rebuilder use Bayesian inference"
    finally:
        store.close()


def test_transform_default_is_stack_r1_r3() -> None:
    assert DEFAULT_STRATEGY == STACK_R1_R3_STRATEGY


def test_transform_stack_lowercases_capitalised(tmp_path: Path) -> None:
    """The stack tokenises and lowercases capitalised tokens via R1.
    IDF-clip may then drop or boost them depending on the per-store
    distribution; what we assert here is the lower-case + non-empty
    behaviour, not numerical lift (graded by the bench harness)."""
    # Seed with content where "Foo" is rare relative to filler vocab,
    # so that "foo" is high-IDF and survives the R3 clip.
    beliefs = [
        _mk(f"f{i:03d}", f"common fillerword{i % 5} commonword another")
        for i in range(30)
    ]
    beliefs.append(_mk("rare", "foo special-rare-marker"))
    store = _seed(tmp_path / "m.db", beliefs)
    try:
        out = transform_query("Foo", store, STACK_R1_R3_STRATEGY)
        # Stack output is space-joined lowercased tokens. "Foo" must
        # not appear with its original capitalisation.
        assert "Foo" not in out
        # And the stack must have produced some non-empty output --
        # "foo" is in vocabulary so R3 either keeps or boosts it.
        assert out.strip() != ""
    finally:
        store.close()


def test_transform_empty_query_returns_empty(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("a", "x")])
    try:
        assert transform_query("", store, LEGACY_STRATEGY) == ""
        assert transform_query("   ", store, STACK_R1_R3_STRATEGY) == ""
    finally:
        store.close()


def test_transform_unknown_strategy_raises(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("a", "x")])
    try:
        with pytest.raises(ValueError, match="unknown query_strategy"):
            transform_query("foo", store, "bogus-strategy")
    finally:
        store.close()


def test_valid_strategies_set_is_exactly_two() -> None:
    assert VALID_STRATEGIES == frozenset({LEGACY_STRATEGY, STACK_R1_R3_STRATEGY})


# --- store_cache ------------------------------------------------------------


def test_cache_returns_same_object_on_second_call(tmp_path: Path) -> None:
    store = _seed(
        tmp_path / "m.db",
        [_mk(f"b{i}", f"alpha beta gamma {i}") for i in range(10)],
    )
    try:
        first = get_bm25_and_quantiles(store)
        second = get_bm25_and_quantiles(store)
        # Identity check: the cache returns the exact same tuple,
        # so the BM25Index is built once.
        assert first is second
    finally:
        store.close()


def test_cache_invalidates_on_belief_mutation(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("a", "alpha")])
    try:
        first = get_bm25_and_quantiles(store)
        # Insert a new belief; the invalidation callback chain should
        # drop the cached entry.
        store.insert_belief(_mk("b", "beta gamma delta"))
        second = get_bm25_and_quantiles(store)
        assert first is not second
        # And the new index sees the new vocabulary.
        new_index, _ = second
        assert "delta" in new_index.vocabulary
    finally:
        store.close()


def test_cache_invalidate_helper_drops_entry(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("a", "x")])
    try:
        get_bm25_and_quantiles(store)
        size_before = _cache_size_for_test()
        invalidate(store)
        assert _cache_size_for_test() == size_before - 1
    finally:
        store.close()


def test_cache_quantile_args_passed_through(tmp_path: Path) -> None:
    store = _seed(
        tmp_path / "m.db",
        [_mk(f"b{i}", "alpha beta gamma delta epsilon") for i in range(5)],
    )
    try:
        invalidate(store)  # ensure fresh build with our quantile args
        _, (low_t, high_t) = get_bm25_and_quantiles(
            store, low_quantile=0.1, high_quantile=0.9,
        )
        assert low_t <= high_t
    finally:
        store.close()


# --- RebuilderConfig --------------------------------------------------------


def test_config_default_query_strategy_is_stack_r1_r3() -> None:
    cfg = RebuilderConfig()
    assert cfg.query_strategy == STACK_R1_R3_STRATEGY


def test_config_loads_query_strategy_override(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuilder]\nquery_strategy = "stack-r1-r3"\n'
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.query_strategy == STACK_R1_R3_STRATEGY


def test_config_invalid_query_strategy_falls_back_to_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuilder]\nquery_strategy = "nonsense"\n'
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.query_strategy == STACK_R1_R3_STRATEGY
    err = capsys.readouterr().err
    assert "query_strategy" in err
    assert "nonsense" not in err  # don't echo the bad value verbatim


def test_config_non_string_query_strategy_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuilder]\nquery_strategy = 42\n'
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.query_strategy == STACK_R1_R3_STRATEGY
    assert "query_strategy" in capsys.readouterr().err


def test_config_explicit_legacy_value_loads(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuilder]\nquery_strategy = "legacy-bm25"\n'
    )
    cfg = load_rebuilder_config(tmp_path)
    assert cfg.query_strategy == LEGACY_STRATEGY


# --- rebuild_v14 plumbing ---------------------------------------------------


def test_rebuild_default_query_strategy_is_stack_r1_r3(tmp_path: Path) -> None:
    """Calling rebuild_v14 with no `query_strategy` argument is
    byte-identical to calling it with `query_strategy='stack-r1-r3'`
    after the #291 PR-3 default flip."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "a", "rebuilder uses bayesian posterior",
            lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
        )],
    )
    try:
        turns = [RecentTurn(role="user", text="Tell me about the Rebuilder")]
        without_arg = rebuild_v14(turns, store)
        with_stack = rebuild_v14(turns, store, query_strategy=STACK_R1_R3_STRATEGY)
        assert without_arg == with_stack
    finally:
        store.close()


def test_rebuild_stack_strategy_returns_block(tmp_path: Path) -> None:
    """Smoke-check that the stack strategy runs end-to-end without
    raising and produces a well-formed rebuild block when there is a
    locked belief in the store. The numerical lift is graded by the
    bench harness (#288), not by this unit test."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "a", "rebuilder uses bayesian posterior",
            lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
        )],
    )
    try:
        turns = [RecentTurn(role="user", text="Tell me about the Rebuilder")]
        out = rebuild_v14(turns, store, query_strategy=STACK_R1_R3_STRATEGY)
        # L0 locked always packs; the block must contain the belief id.
        assert "<aelfrice-rebuild>" in out
        assert "id=\"a\"" in out
    finally:
        store.close()


def test_rebuild_stack_strategy_empty_turns_returns_locked_only(
    tmp_path: Path,
) -> None:
    """Empty recent_turns + stack strategy: the query is empty, the
    stack short-circuits, retrieve sees an empty query, and only L0
    locked beliefs pack -- same as legacy."""
    store = _seed(
        tmp_path / "m.db",
        [_mk(
            "a", "rebuilder uses bayesian posterior",
            lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
        )],
    )
    try:
        without_arg = rebuild_v14([], store)
        stacked = rebuild_v14([], store, query_strategy=STACK_R1_R3_STRATEGY)
        # Both paths return the same locked-only block on empty input.
        assert without_arg == stacked
    finally:
        store.close()
