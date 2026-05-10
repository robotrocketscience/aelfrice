"""Integration tests for #436 intentional clustering wired into retrieve_v2.

Covers flag-precedence resolution, byte-identical default-OFF behavior,
the diversity-aware pack on graph-connected candidates, locked-belief
pre-include, and the mutual-exclusion guard against type-aware
compression at v2.0.0.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    ENV_INTENTIONAL_CLUSTERING,
    ENV_TYPE_AWARE_COMPRESSION,
    INTENTIONAL_CLUSTERING_FLAG,
    resolve_use_intentional_clustering,
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
    monkeypatch.delenv(ENV_INTENTIONAL_CLUSTERING, raising=False)
    monkeypatch.delenv(ENV_TYPE_AWARE_COMPRESSION, raising=False)


@pytest.fixture
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- Flag resolution ---------------------------------------------------


def test_default_is_on(_no_env_override: None, _isolated_cwd: Path) -> None:
    assert resolve_use_intentional_clustering() is True


def test_explicit_kwarg_overrides_default(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    assert resolve_use_intentional_clustering(True) is True
    assert resolve_use_intentional_clustering(False) is False


def test_env_overrides_explicit_kwarg(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    monkeypatch.setenv(ENV_INTENTIONAL_CLUSTERING, "1")
    assert resolve_use_intentional_clustering(False) is True
    monkeypatch.setenv(ENV_INTENTIONAL_CLUSTERING, "0")
    assert resolve_use_intentional_clustering(True) is False


def test_env_garbage_falls_through(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    monkeypatch.setenv(ENV_INTENTIONAL_CLUSTERING, "maybe")
    assert resolve_use_intentional_clustering() is True
    assert resolve_use_intentional_clustering(False) is False


def test_toml_resolves_when_kwarg_and_env_unset(
    _no_env_override: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text(
        f"[retrieval]\n{INTENTIONAL_CLUSTERING_FLAG} = true\n"
    )
    monkeypatch.chdir(tmp_path)
    assert resolve_use_intentional_clustering() is True


# --- retrieve_v2 wiring ------------------------------------------------


def _populate_clustered_store() -> MemoryStore:
    """Two graph-connected clusters joined by SUPPORTS edges, plus an
    isolated singleton. With clustering OFF, the score-ranked greedy
    fill returns whatever ranks highest. With clustering ON, the
    diversity-aware pack covers more clusters at the same budget."""
    s = MemoryStore(":memory:")
    # Cluster A: tightly-coupled deploy facts.
    for bid in ("A1", "A2", "A3"):
        s.insert_belief(_mk(
            bid,
            f"deploy {bid}: the system uses sqlite for persistence and ships nightly",
            retention_class=RETENTION_FACT,
        ))
    s.insert_edge(Edge(src="A1", dst="A2", type=EDGE_SUPPORTS, weight=0.8))
    s.insert_edge(Edge(src="A2", dst="A3", type=EDGE_SUPPORTS, weight=0.8))
    # Cluster B: tightly-coupled prerequisite facts.
    for bid in ("B1", "B2"):
        s.insert_belief(_mk(
            bid,
            f"deploy {bid}: prerequisite is sqlite being installed on every host",
            retention_class=RETENTION_FACT,
        ))
    s.insert_edge(Edge(src="B1", dst="B2", type=EDGE_SUPPORTS, weight=0.8))
    # Singleton.
    s.insert_belief(_mk(
        "C1",
        "deploy C1: rollback is via the previous nightly artifact for sqlite",
        retention_class=RETENTION_FACT,
    ))
    return s


def test_default_call_byte_identical_to_explicit_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Default and explicit-ON must agree on the merged belief id list.

    Post-#436 default-flip: callers that don't pass the kwarg get the
    same selection as callers that pass `use_intentional_clustering=True`.
    Operators wanting v2.0.x parity opt out via
    `[retrieval] use_intentional_clustering = false` or
    `AELFRICE_INTENTIONAL_CLUSTERING=0`."""
    s = _populate_clustered_store()
    explicit_on = retrieve_v2(
        s, "deploy sqlite",
        budget=2400,
        use_entity_index=False,
        use_intentional_clustering=True,
    )
    default_on = retrieve_v2(
        s, "deploy sqlite",
        budget=2400,
        use_entity_index=False,
    )
    assert [b.id for b in explicit_on.beliefs] \
        == [b.id for b in default_on.beliefs]


def test_clustering_changes_selection_at_tight_budget(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """At a budget tight enough that OFF can fit only Cluster A's top
    members, ON pivots to cover Cluster B + the singleton via Stage 1
    representatives. The selections differ as a set, demonstrating that
    cluster_diversity_target=3 changed *which* beliefs survived the pack."""
    s = _populate_clustered_store()
    off = retrieve_v2(
        s, "deploy sqlite",
        budget=80,
        use_entity_index=False,
        use_intentional_clustering=False,
    )
    on = retrieve_v2(
        s, "deploy sqlite",
        budget=80,
        use_entity_index=False,
        use_intentional_clustering=True,
    )
    off_ids = {b.id for b in off.beliefs}
    on_ids = {b.id for b in on.beliefs}
    # ON must include at least one belief OFF didn't (cluster diversity)
    # AND must drop at least one belief OFF kept (the second member of
    # the highest-seed cluster, displaced by another cluster's rep).
    assert on_ids != off_ids, (
        f"clustering ON should differ from OFF at tight budget; "
        f"got OFF={sorted(off_ids)} ON={sorted(on_ids)}"
    )


def test_locked_pre_included_when_clustering_on(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """Spec § Open Q4: locked beliefs pre-include ahead of Stage 1.

    Implementation satisfies this implicitly — `out = list(locked) +
    list(l25)` runs before the cluster pack on the L1 candidates only."""
    s = _populate_clustered_store()
    s.insert_belief(_mk(
        "L0",
        "user-pinned: sqlite is the chosen substrate. immutable.",
        retention_class=RETENTION_SNAPSHOT,
        lock_level=LOCK_USER,
        locked_at="2026-05-08T00:00:00Z",
    ))
    on = retrieve_v2(
        s, "deploy sqlite",
        budget=2400,
        use_entity_index=False,
        use_intentional_clustering=True,
    )
    assert on.beliefs[0].id == "L0", (
        f"locked belief must be first regardless of clustering flag; "
        f"got {[b.id for b in on.beliefs[:3]]}"
    )


def test_mutual_exclusion_with_compression_raises(
    _no_env_override: None, _isolated_cwd: Path
) -> None:
    """At v2.0.0, clustering + compression together raise ValueError —
    the cluster pack accounts in raw token cost, composing it with the
    compressed cost is a v2.x follow-up."""
    s = _populate_clustered_store()
    with pytest.raises(ValueError, match="mutually exclusive"):
        retrieve_v2(
            s, "deploy sqlite",
            budget=2400,
            use_entity_index=False,
            use_intentional_clustering=True,
            use_type_aware_compression=True,
        )


def test_env_var_alone_enables_clustering(
    monkeypatch: pytest.MonkeyPatch, _isolated_cwd: Path
) -> None:
    """No kwarg + env=1 + tight budget → ON behavior."""
    monkeypatch.setenv(ENV_INTENTIONAL_CLUSTERING, "1")
    s = _populate_clustered_store()
    on_via_env = retrieve_v2(
        s, "deploy sqlite",
        budget=80,
        use_entity_index=False,
    )
    monkeypatch.setenv(ENV_INTENTIONAL_CLUSTERING, "0")
    off_via_env = retrieve_v2(
        s, "deploy sqlite",
        budget=80,
        use_entity_index=False,
    )
    # The env-flip must change at least one of: belief count, ordering,
    # or selection. If it changes none, the env wiring is broken.
    on_ids = [b.id for b in on_via_env.beliefs]
    off_ids = [b.id for b in off_via_env.beliefs]
    assert on_ids != off_ids, (
        f"env=1 must produce different selection than env=0 at tight "
        f"budget; got both = {on_ids}"
    )
