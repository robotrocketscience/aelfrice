"""The HRR expand lane's probe-kind override (#1795).

A benchmark arm probes `DERIVED_FROM`, `RELATES_TO`, and `TEMPORAL_NEXT`
through the production `retrieve()` path, which has no parameter for the
kinds. Unset, the override must leave the lane exactly as it was.
"""
from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import hrr_expand as hx
from aelfrice.hrr_index import HRRStructIndex
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import ENV_HRR_EXPAND, retrieve
from aelfrice.store import MemoryStore

_SEED = 7


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
    )


@pytest.fixture
def store() -> Iterator[MemoryStore]:
    """`seed` has one neighbor over each of five edge kinds."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("seed", "the kestrel hovers above the meadow"))
    s.insert_belief(_mk("con", "owls nest in old barns at dusk"))
    s.insert_belief(_mk("sup", "foxes cross the frozen lake"))
    s.insert_belief(_mk("rel", "otters swim in cold rivers"))
    s.insert_belief(_mk("der", "herons stand in shallow reeds"))
    s.insert_belief(_mk("tmp", "badgers dig under hedges at night"))
    s.insert_edge(Edge(src="seed", dst="con", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="sup", dst="seed", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="seed", dst="rel", type=EDGE_RELATES_TO, weight=1.0))
    s.insert_edge(Edge(src="der", dst="seed", type=EDGE_DERIVED_FROM, weight=1.0))
    s.insert_edge(Edge(src="seed", dst="tmp", type=EDGE_TEMPORAL_NEXT, weight=1.0))
    yield s
    s.close()


@pytest.fixture(autouse=True)
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hx, "probe_kinds_override", None)
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)


def _index(store: MemoryStore) -> HRRStructIndex:
    idx = HRRStructIndex(dim=512, seed=_SEED)
    idx.build(store, seed=_SEED)
    return idx


def _ids(store: MemoryStore, kinds: tuple[str, ...] | None = None) -> list[str]:
    return [b.id for b in hx.expand_seeds(store, _index(store), ["seed"], kinds=kinds)]


@pytest.mark.source_scan
def test_the_override_is_unset_by_default() -> None:
    # Read from the source: the autouse fixture resets the attribute, so
    # the live value can't show what the module assigns at import.
    tree = ast.parse(Path(hx.__file__).read_text(encoding="utf-8"))
    values = [
        node.value for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id == "probe_kinds_override"
    ]
    assert len(values) == 1
    assert isinstance(values[0], ast.Constant) and values[0].value is None


def test_unset_the_lane_probes_its_default_kinds_only(store: MemoryStore) -> None:
    default = _ids(store)
    assert default == _ids(store, kinds=hx.hrr_expand_edge_types())
    assert sorted(default) == ["con", "sup"]


def test_the_override_replaces_the_probed_kinds(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hx, "probe_kinds_override", (EDGE_DERIVED_FROM, EDGE_RELATES_TO, EDGE_TEMPORAL_NEXT),
    )
    assert sorted(_ids(store)) == ["der", "rel", "tmp"]


def test_an_explicit_kinds_argument_wins_over_the_override(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hx, "probe_kinds_override", (EDGE_RELATES_TO,))
    assert _ids(store, kinds=(EDGE_CONTRADICTS,)) == ["con"]


def test_the_override_reaches_the_lane_through_retrieve(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_HRR_EXPAND, "1")
    unset = [b.id for b in retrieve(store, "kestrel meadow", user_locks=())]
    assert unset[0] == "seed" and "con" in unset and "rel" not in unset
    monkeypatch.setattr(hx, "probe_kinds_override", (EDGE_RELATES_TO,))
    over = [b.id for b in retrieve(store, "kestrel meadow", user_locks=())]
    assert "rel" in over and "con" not in over


def test_unset_retrieve_returns_what_the_default_kinds_return(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset, the lane behaves as it did before the override existed: the
    # same ranked beliefs as naming the default kinds outright.
    monkeypatch.setenv(ENV_HRR_EXPAND, "1")
    unset = retrieve(store, "kestrel meadow", user_locks=())
    monkeypatch.setattr(hx, "probe_kinds_override", hx.hrr_expand_edge_types())
    named = retrieve(store, "kestrel meadow", user_locks=())
    assert [(b.id, b.content) for b in unset] == [(b.id, b.content) for b in named]
    assert "con" in [b.id for b in unset]
