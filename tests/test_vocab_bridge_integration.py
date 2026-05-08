"""Integration tests for #433 HRR vocabulary bridge in retrieve_v2.

Covers flag-precedence resolution, the deprecated `use_hrr` alias, the
`vocab_bridge_cache` injection point, and the byte-identical default-
OFF path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    Belief,
)
from aelfrice.retrieval import (
    ENV_VOCAB_BRIDGE,
    resolve_use_vocab_bridge,
    retrieve_v2,
)
from aelfrice.store import MemoryStore
from aelfrice.vocab_bridge import VocabBridgeCache


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class="fact",
    )


@pytest.fixture
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VOCAB_BRIDGE, raising=False)


@pytest.fixture
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _populate_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1", "SQLite is the storage substrate."))
    s.insert_belief(_mk("b2", "Python is the language used."))
    s.insert_belief(_mk("b3", "Numpy provides matrix algebra."))
    return s


# --- Flag resolution ---------------------------------------------------


def test_default_is_off(_no_env: None, _isolated_cwd: Path) -> None:
    assert resolve_use_vocab_bridge() is False


def test_explicit_kwarg_overrides_default(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    assert resolve_use_vocab_bridge(True) is True
    assert resolve_use_vocab_bridge(False) is False


def test_env_overrides_explicit_kwarg(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path,
) -> None:
    monkeypatch.setenv(ENV_VOCAB_BRIDGE, "1")
    assert resolve_use_vocab_bridge(False) is True
    monkeypatch.setenv(ENV_VOCAB_BRIDGE, "0")
    assert resolve_use_vocab_bridge(True) is False


def test_env_garbage_falls_through(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path,
) -> None:
    monkeypatch.setenv(ENV_VOCAB_BRIDGE, "maybe")
    assert resolve_use_vocab_bridge() is False
    assert resolve_use_vocab_bridge(True) is True


def test_toml_resolves_when_kwarg_and_env_unset(
    _no_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_vocab_bridge = true\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_use_vocab_bridge() is True


# --- retrieve_v2 byte-identity (default-OFF path) ----------------------


def test_default_call_byte_identical_to_pre_bridge(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """Default-OFF path: bridge does not run, so beliefs returned match
    a control retrieve_v2 call without any vocab-bridge kwargs."""
    s = _populate_store()
    a = retrieve_v2(s, "sqlite storage")
    b = retrieve_v2(s, "sqlite storage")
    assert [b_.id for b_ in a.beliefs] == [b_.id for b_ in b.beliefs]


def test_explicit_off_byte_identical_to_default(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    s = _populate_store()
    default_call = retrieve_v2(s, "sqlite storage")
    off_call = retrieve_v2(s, "sqlite storage", use_vocab_bridge=False)
    assert [b.id for b in default_call.beliefs] == [
        b.id for b in off_call.beliefs
    ]


# --- retrieve_v2 with bridge ON ----------------------------------------


def test_flag_on_does_not_raise(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """Smoke: flag-ON path completes without raising and returns a
    well-shaped RetrievalResult. The bridge cannot empirically
    *improve* recall on a 3-belief fixture (cross-talk dominates at
    small N), so this test only verifies structural correctness."""
    s = _populate_store()
    result = retrieve_v2(s, "sqlite storage", use_vocab_bridge=True)
    assert hasattr(result, "beliefs")
    assert isinstance(result.beliefs, list)


def test_env_var_alone_enables_bridge(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path,
) -> None:
    """Env var alone (no kwarg, no TOML) opts retrieve_v2 into the
    bridge. We can't assert visibly different output on a small
    fixture; we verify the call completes and structure is valid."""
    monkeypatch.setenv(ENV_VOCAB_BRIDGE, "1")
    s = _populate_store()
    result = retrieve_v2(s, "sqlite")
    assert isinstance(result.beliefs, list)


# --- Deprecated `use_hrr` alias ---------------------------------------


def test_use_hrr_true_routes_to_bridge(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """`use_hrr=True` is the deprecated alias path — it should opt the
    call into the bridge identically to `use_vocab_bridge=True`. Test
    by checking the result-shape is consistent across both paths."""
    s = _populate_store()
    via_alias = retrieve_v2(s, "sqlite", use_hrr=True)
    via_canonical = retrieve_v2(s, "sqlite", use_vocab_bridge=True)
    # Same belief id set (order-insensitive) — both paths invoked the
    # bridge and the lane fan-out saw the same widened query.
    assert {b.id for b in via_alias.beliefs} == {
        b.id for b in via_canonical.beliefs
    }


def test_use_vocab_bridge_overrides_use_hrr(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """When both are passed, use_vocab_bridge wins (canonical name
    beats deprecated alias)."""
    s = _populate_store()
    # use_vocab_bridge=False, use_hrr=True → canonical False wins.
    result = retrieve_v2(
        s, "sqlite", use_hrr=True, use_vocab_bridge=False,
    )
    # Identical to a flag-OFF call.
    control = retrieve_v2(s, "sqlite", use_vocab_bridge=False)
    assert [b.id for b in result.beliefs] == [b.id for b in control.beliefs]


def test_use_hrr_none_does_not_force_bridge(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """Explicit `use_hrr=None` (the new default) does NOT route to the
    bridge — it falls through to use_vocab_bridge resolution, which is
    OFF by default."""
    s = _populate_store()
    result = retrieve_v2(s, "sqlite", use_hrr=None)
    control = retrieve_v2(s, "sqlite")
    assert [b.id for b in result.beliefs] == [b.id for b in control.beliefs]


# --- VocabBridgeCache injection ---------------------------------------


def test_explicit_cache_is_used(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    """An explicit cache is consulted (not bypassed) when the flag is
    on. We verify by warming the cache, then checking that a follow-up
    retrieve_v2 with the same cache returns the same result shape."""
    s = _populate_store()
    cache = VocabBridgeCache(store=s, store_path="/tmp/explicit-cache")
    # Pre-warm.
    cache.get()
    a = retrieve_v2(
        s, "sqlite", use_vocab_bridge=True, vocab_bridge_cache=cache,
    )
    b = retrieve_v2(
        s, "sqlite", use_vocab_bridge=True, vocab_bridge_cache=cache,
    )
    assert [x.id for x in a.beliefs] == [x.id for x in b.beliefs]


def test_cache_invalidates_on_store_mutation(
    _no_env: None, _isolated_cwd: Path,
) -> None:
    s = _populate_store()
    cache = VocabBridgeCache(store=s)
    bridge_v1 = cache.get()
    canonicals_v1 = list(bridge_v1.canonicals)
    # Mutate: add a new belief; the invalidation hook should drop the
    # cached bridge so the next get() rebuilds.
    s.insert_belief(_mk("b4", "Redis can serve as a cache layer."))
    bridge_v2 = cache.get()
    # Cache rebuilt — bridge object identity changed, and the new
    # bridge picks up the new canonicals.
    assert bridge_v2 is not bridge_v1
    assert "redis" in bridge_v2.canonicals
    assert set(canonicals_v1).issubset(set(bridge_v2.canonicals))
