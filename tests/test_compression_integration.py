"""Integration tests for #434 type-aware compression in retrieve_v2.

Covers flag-precedence resolution and the parallel `compressed_beliefs`
field on `RetrievalResult`. Default-OFF path is byte-identical to v1.x:
`compressed_beliefs` is empty when the flag does not resolve True.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from aelfrice.compression import STRATEGY_HEADLINE, STRATEGY_STUB, STRATEGY_VERBATIM
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_TRANSIENT,
    Belief,
)
from aelfrice.retrieval import (
    ENV_TYPE_AWARE_COMPRESSION,
    resolve_use_type_aware_compression,
    retrieve,
    retrieve_v2,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    *,
    retention_class: str = RETENTION_FACT,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class=retention_class,
    )


@pytest.fixture
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_TYPE_AWARE_COMPRESSION, raising=False)


@pytest.fixture
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """cd into an empty tmp dir so .aelfrice.toml lookups don't pick up
    the repo's own config during flag resolution."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- Flag resolution ---------------------------------------------------


def test_default_is_off(_no_env_override: None, _isolated_cwd: Path) -> None:
    assert resolve_use_type_aware_compression() is False


def test_explicit_kwarg_overrides_default(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    assert resolve_use_type_aware_compression(True) is True
    assert resolve_use_type_aware_compression(False) is False


def test_env_overrides_explicit_kwarg(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    monkeypatch.setenv(ENV_TYPE_AWARE_COMPRESSION, "1")
    assert resolve_use_type_aware_compression(False) is True
    monkeypatch.setenv(ENV_TYPE_AWARE_COMPRESSION, "0")
    assert resolve_use_type_aware_compression(True) is False


def test_env_garbage_falls_through(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    monkeypatch.setenv(ENV_TYPE_AWARE_COMPRESSION, "maybe")
    # Garbage env reverts to the next layer; with no kwarg/toml,
    # default OFF wins.
    assert resolve_use_type_aware_compression() is False
    # And a kwarg now decides.
    assert resolve_use_type_aware_compression(True) is True


def test_toml_resolves_when_kwarg_and_env_unset(
    _no_env_override: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_type_aware_compression = true\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_use_type_aware_compression() is True


# --- retrieve_v2 integration ------------------------------------------


def _populate_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "the system uses sqlite for persistence",
                        retention_class=RETENTION_FACT))
    s.insert_belief(_mk(
        "S1",
        "morning thought: the system uses sqlite for persistence. "
        "and other things to drop in the headline strategy.",
        retention_class=RETENTION_SNAPSHOT,
    ))
    s.insert_belief(_mk(
        "T1",
        "PR-window scratch about the sqlite system note " * 3,
        retention_class=RETENTION_TRANSIENT,
    ))
    s.insert_belief(_mk(
        "L1",
        "user-pinned: sqlite is the chosen substrate. immutable.",
        retention_class=RETENTION_SNAPSHOT,
        lock_level=LOCK_USER,
        locked_at="2026-05-08T00:00:00Z",
    ))
    return s


def test_compressed_beliefs_empty_when_flag_off(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_store()
    result = retrieve_v2(s, "sqlite system", use_type_aware_compression=False)
    assert result.compressed_beliefs == []
    # `beliefs` is unchanged.
    assert any(b.id == "F1" for b in result.beliefs)


def test_compressed_beliefs_populated_when_flag_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_store()
    # use_intentional_clustering=False explicit: post-#436 default-flip,
    # the clustering flag is default-on and would mutex with compression.
    result = retrieve_v2(
        s, "sqlite system",
        use_type_aware_compression=True,
        use_intentional_clustering=False,
    )
    assert len(result.compressed_beliefs) == len(result.beliefs)
    # Same order, same belief ids.
    for b, cb in zip(result.beliefs, result.compressed_beliefs, strict=True):
        assert cb.belief.id == b.id


def test_compression_strategy_dispatches_by_retention_class(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_store()
    result = retrieve_v2(
        s, "sqlite system",
        use_type_aware_compression=True,
        use_intentional_clustering=False,
    )
    by_id = {cb.belief.id: cb for cb in result.compressed_beliefs}
    assert by_id["F1"].strategy == STRATEGY_VERBATIM
    assert by_id["S1"].strategy == STRATEGY_HEADLINE
    assert by_id["T1"].strategy == STRATEGY_STUB
    # Locked belief renders verbatim despite snapshot retention class.
    assert by_id["L1"].strategy == STRATEGY_VERBATIM


def test_env_var_alone_enables_compression(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    monkeypatch.setenv(ENV_TYPE_AWARE_COMPRESSION, "1")
    # Post-#436 default-flip, AELFRICE_INTENTIONAL_CLUSTERING must also
    # be disabled in this scope to satisfy the v2.0.0 mutex (the cluster
    # pack accounts in raw tokens; composing it with compressed cost is
    # tracked as a v2.x follow-up). The test still meaningfully exercises
    # "env var alone enables compression" — it just makes the clustering
    # env-disable explicit instead of relying on the (now-flipped) default.
    monkeypatch.setenv("AELFRICE_INTENTIONAL_CLUSTERING", "0")
    s = _populate_store()
    result = retrieve_v2(s, "sqlite system")  # no explicit kwarg
    assert len(result.compressed_beliefs) == len(result.beliefs)


def test_default_call_leaves_compressed_empty(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_store()
    result = retrieve_v2(s, "sqlite system")  # no kwarg, no env, default OFF
    assert result.compressed_beliefs == []


# --- Pack-loop budget rewrite (#434 phase 2) ---------------------------
#
# With the flag ON, pack accounting consumes `cb.rendered_tokens` — so a
# tight budget that fit only the fact-class belief at raw cost can now
# admit the transient-class beliefs whose stub render is ~10 tokens each.


def _populate_pack_widening_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "F1",
        "the system uses sqlite for persistence",
        retention_class=RETENTION_FACT,
    ))
    for i in range(5):
        s.insert_belief(_mk(
            f"T{i}",
            "scratch about sqlite system token " * 30,
            retention_class=RETENTION_TRANSIENT,
        ))
    return s


def test_pack_widens_when_flag_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_pack_widening_store()
    off = retrieve_v2(
        s, "sqlite system",
        budget=80,
        use_entity_index=False,
        use_type_aware_compression=False,
    )
    on = retrieve_v2(
        s, "sqlite system",
        budget=80,
        use_entity_index=False,
        use_type_aware_compression=True,
        use_intentional_clustering=False,
    )
    assert len(on.beliefs) > len(off.beliefs)


def test_pack_byte_identical_when_flag_off(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Flag OFF reproduces pre-#434-phase-2 selection at the same budget.

    Two calls — explicit OFF vs default OFF — must agree byte-for-byte
    on the merged belief id list. This is the byte-identity invariant
    that makes the pack-loop change safe to land default-OFF.
    """
    s = _populate_pack_widening_store()
    explicit_off = retrieve_v2(
        s, "sqlite system",
        budget=80,
        use_entity_index=False,
        use_type_aware_compression=False,
    )
    default_off = retrieve_v2(
        s, "sqlite system",
        budget=80,
        use_entity_index=False,
    )
    assert [b.id for b in explicit_off.beliefs] \
        == [b.id for b in default_off.beliefs]


def test_pack_locked_unchanged_when_flag_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Locks render verbatim, so locked accounting is identical OFF vs ON."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "L1",
        "user-pinned: sqlite is the chosen substrate. immutable.",
        retention_class=RETENTION_SNAPSHOT,
        lock_level=LOCK_USER,
        locked_at="2026-05-08T00:00:00Z",
    ))
    off = retrieve_v2(
        s, "sqlite",
        use_entity_index=False,
        use_type_aware_compression=False,
    )
    on = retrieve_v2(
        s, "sqlite",
        use_entity_index=False,
        use_type_aware_compression=True,
        use_intentional_clustering=False,
    )
    assert [b.id for b in off.beliefs] == [b.id for b in on.beliefs]
    # Locked render is verbatim under compression.
    assert on.compressed_beliefs[0].strategy == STRATEGY_VERBATIM


# --- Bare `retrieve()` flag wiring (#776) ------------------------------
#
# The same toggle that #434 wired into `retrieve_with_tiers` and
# `retrieve_v2` was missing from `retrieve()` — and `rebuild_v14` calls
# `retrieve()`. Without this wiring the A4 bench gate (#775) is a no-op
# because the OFF/ON arms produce byte-identical output. These tests
# lock the wiring: default-OFF byte-identity, env-var / kwarg observability,
# and locks-render-verbatim.


def test_retrieve_pack_widens_when_flag_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    s = _populate_pack_widening_store()
    off = retrieve(
        s, "sqlite system",
        token_budget=80,
        entity_index_enabled=False,
        use_type_aware_compression=False,
    )
    on = retrieve(
        s, "sqlite system",
        token_budget=80,
        entity_index_enabled=False,
        use_type_aware_compression=True,
    )
    assert len(on) > len(off)


def test_retrieve_pack_byte_identical_when_flag_off(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Default-OFF: id list matches an explicit OFF call byte-for-byte."""
    s = _populate_pack_widening_store()
    explicit_off = retrieve(
        s, "sqlite system",
        token_budget=80,
        entity_index_enabled=False,
        use_type_aware_compression=False,
    )
    default_off = retrieve(
        s, "sqlite system",
        token_budget=80,
        entity_index_enabled=False,
    )
    assert [b.id for b in explicit_off] == [b.id for b in default_off]


def test_retrieve_env_var_enables_compression(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    """`AELFRICE_TYPE_AWARE_COMPRESSION=1` flips the pack via the resolver.

    This is the path the A4 bench harness uses: it sets the env var around
    `rebuild_v14`, and `rebuild_v14` calls `retrieve()`. Without the
    wiring this test guards, the env var has no effect.
    """
    s = _populate_pack_widening_store()
    monkeypatch.delenv(ENV_TYPE_AWARE_COMPRESSION, raising=False)
    off = retrieve(
        s, "sqlite system", token_budget=80, entity_index_enabled=False,
    )
    monkeypatch.setenv(ENV_TYPE_AWARE_COMPRESSION, "1")
    on = retrieve(
        s, "sqlite system", token_budget=80, entity_index_enabled=False,
    )
    assert len(on) > len(off)


def test_retrieve_locked_unchanged_when_flag_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Locks render verbatim under bare `retrieve()` too."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(
        "L1",
        "user-pinned: sqlite is the chosen substrate. immutable.",
        retention_class=RETENTION_SNAPSHOT,
        lock_level=LOCK_USER,
        locked_at="2026-05-08T00:00:00Z",
    ))
    off = retrieve(
        s, "sqlite",
        entity_index_enabled=False,
        use_type_aware_compression=False,
    )
    on = retrieve(
        s, "sqlite",
        entity_index_enabled=False,
        use_type_aware_compression=True,
    )
    assert [b.id for b in off] == [b.id for b in on]
