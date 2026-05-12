"""CLI-level peer seed-fetching tests for `aelf reason` (#713).

Two-scope fixture: a local DB and a peer DB ("peerA") connected via
AELFRICE_KNOWLEDGE_DEPS. All peer beliefs carry scope='global' so they
are visible across the federation boundary.

Covers:
  - --seed-id with a foreign id falls through to find_foreign_owner /
    get_belief_in_scope and seeds the BFS into the peer's edge graph,
    annotating the seed with [scope:peerA] in human output and
    owning_scope='peerA' in --json.
  - Query path: _seeds_with_scopes unions local + peer FTS5 results;
    peer seeds surface in human + --json output with owning_scope.
  - Local-only store: no peer hits, no [scope:...] annotation, no
    owning_scope populated — byte-identical to pre-#690 behaviour.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_belief(bid: str, content: str, *, scope: str = "global") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-04T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
        scope=scope,
    )


def _seed_peer(path: Path) -> tuple[str, str, str]:
    """Create a peer DB with a 2-hop SUPPORTS chain anchor→hop1→hop2.

    All beliefs carry scope='global' so they cross the federation boundary.
    Returns the three belief ids.
    """
    store = MemoryStore(str(path))
    try:
        anchor_id = "peer-anchor-1"
        hop1_id = "peer-hop1-1"
        hop2_id = "peer-hop2-1"
        for b in (
            _mk_belief(anchor_id, "federation anchor knowledge"),
            _mk_belief(hop1_id, "federation hop one knowledge"),
            _mk_belief(hop2_id, "federation hop two knowledge"),
        ):
            store.insert_belief(b)
        store.insert_edge(
            Edge(src=anchor_id, dst=hop1_id, type=EDGE_SUPPORTS, weight=1.0)
        )
        store.insert_edge(
            Edge(src=hop1_id, dst=hop2_id, type=EDGE_SUPPORTS, weight=1.0)
        )
    finally:
        store.close()
    return anchor_id, hop1_id, hop2_id


def _wire_peer(
    tmp_path: Path, peer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps(
            {
                "version": 1,
                "deps": [{"name": "peerA", "path": str(peer_path)}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))


def _run(local_db: Path, *argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str, str]:
    """Two-scope fixture: local DB + peerA DB, no local beliefs.

    Returns (local_db_path, peer_db_path, anchor_id, hop1_id, hop2_id).
    """
    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peerA.db"

    monkeypatch.setenv("AELFRICE_DB", str(local_db))
    _wire_peer(tmp_path, peer_db, monkeypatch)

    anchor_id, hop1_id, hop2_id = _seed_peer(peer_db)
    return local_db, peer_db, anchor_id, hop1_id, hop2_id


@pytest.fixture()
def two_scope_with_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str]:
    """Two-scope fixture with local beliefs for query-path overlap testing.

    Local belief has scope='project' (default); peer beliefs are 'global'.
    Returns (local_db_path, peer_db_path, local_id, peer_anchor_id).
    """
    local_db = tmp_path / "local.db"
    peer_db = tmp_path / "peerA.db"

    monkeypatch.setenv("AELFRICE_DB", str(local_db))
    _wire_peer(tmp_path, peer_db, monkeypatch)

    # Seed local with a belief that has overlapping FTS5 terms.
    local_store = MemoryStore(str(local_db))
    local_id = "local-xyz-1"
    peer_local_id = "peer-qrs-1"
    try:
        local_store.insert_belief(
            _mk_belief(local_id, "knowledge about widgets local", scope="project")
        )
    finally:
        local_store.close()

    # Seed peer with a belief that overlaps on "knowledge".
    peer_store = MemoryStore(str(peer_db))
    try:
        peer_store.insert_belief(
            _mk_belief(peer_local_id, "knowledge about widgets peer", scope="global")
        )
    finally:
        peer_store.close()

    return local_db, peer_db, local_id, peer_local_id


# ---------------------------------------------------------------------------
# --seed-id path: foreign seed fallthrough
# ---------------------------------------------------------------------------


def test_seed_id_foreign_resolves_and_annotates_human(
    two_scope: tuple,
) -> None:
    """--seed-id <foreign-id>: seed surfaces with [scope:peerA] in human output."""
    _local_db, _peer_db, anchor_id, hop1_id, _hop2_id = two_scope
    code, out = _run(_local_db, "reason", "federation", "--seed-id", anchor_id)
    assert code == 0, f"expected exit 0, got {code!r}; output:\n{out}"
    assert f"[scope:peerA] {anchor_id}" in out, (
        f"expected '[scope:peerA] {anchor_id}' in seeds block:\n{out}"
    )
    # BFS should walk into peer edges and surface hop1 in the chain.
    assert hop1_id in out, f"expected peer hop {hop1_id!r} in output:\n{out}"


def test_seed_id_foreign_json_owning_scope(
    two_scope: tuple,
) -> None:
    """--seed-id <foreign-id> --json: seed has owning_scope='peerA'."""
    _local_db, _peer_db, anchor_id, hop1_id, _hop2_id = two_scope
    code, out = _run(
        _local_db, "reason", "federation", "--seed-id", anchor_id, "--json"
    )
    assert code == 0, f"expected exit 0; output:\n{out}"
    payload = json.loads(out)
    seeds = payload["seeds"]
    assert len(seeds) == 1
    assert seeds[0]["id"] == anchor_id
    assert seeds[0]["owning_scope"] == "peerA", (
        f"expected owning_scope='peerA', got {seeds[0]['owning_scope']!r}"
    )
    # Hops should carry owning_scope too.
    hop_ids = [h["id"] for h in payload["hops"]]
    assert hop1_id in hop_ids, f"expected {hop1_id!r} in hops {hop_ids!r}"
    for hop in payload["hops"]:
        assert hop["owning_scope"] == "peerA", (
            f"hop {hop['id']!r} expected owning_scope='peerA', "
            f"got {hop['owning_scope']!r}"
        )


def test_seed_id_missing_everywhere_exits_nonzero(
    two_scope: tuple,
) -> None:
    """--seed-id not found in local or any peer → exit 2."""
    _local_db = two_scope[0]
    code, out = _run(_local_db, "reason", "anything", "--seed-id", "no-such-id-xyz")
    assert code == 2, f"expected exit 2, got {code!r}; output:\n{out}"
    assert "seed-id not found" in out


# ---------------------------------------------------------------------------
# Query path: peer seeds union
# ---------------------------------------------------------------------------


def test_query_path_surfaces_peer_seed_human(
    two_scope_with_local: tuple,
) -> None:
    """Query path: peer FTS5 hit surfaces with [scope:peerA] in human output."""
    _local_db, _peer_db, local_id, peer_id = two_scope_with_local
    code, out = _run(_local_db, "reason", "knowledge widgets")
    assert code == 0, f"expected exit 0; output:\n{out}"
    assert f"[scope:peerA] {peer_id}" in out, (
        f"expected peer seed annotation in seeds block:\n{out}"
    )


def test_query_path_surfaces_peer_seed_json(
    two_scope_with_local: tuple,
) -> None:
    """Query path --json: peer seed has owning_scope='peerA', local has None."""
    _local_db, _peer_db, local_id, peer_id = two_scope_with_local
    code, out = _run(_local_db, "reason", "knowledge widgets", "--json")
    assert code == 0, f"expected exit 0; output:\n{out}"
    payload = json.loads(out)
    seeds_by_id = {s["id"]: s for s in payload["seeds"]}
    assert peer_id in seeds_by_id, (
        f"expected peer seed {peer_id!r} in payload seeds: "
        f"{list(seeds_by_id.keys())!r}"
    )
    assert seeds_by_id[peer_id]["owning_scope"] == "peerA"
    assert local_id in seeds_by_id, (
        f"expected local seed {local_id!r} in payload seeds"
    )
    assert seeds_by_id[local_id]["owning_scope"] is None


# ---------------------------------------------------------------------------
# Local-only: no peer annotation (byte-identical pre-#690 path)
# ---------------------------------------------------------------------------


def test_local_only_no_scope_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local-only store: no [scope:...] annotation, owning_scope=None on seeds."""
    local_db = tmp_path / "local.db"
    monkeypatch.setenv("AELFRICE_DB", str(local_db))
    # No AELFRICE_KNOWLEDGE_DEPS set → no peers wired.

    local_store = MemoryStore(str(local_db))
    a_id = "local-aaa-1"
    b_id = "local-bbb-1"
    try:
        local_store.insert_belief(
            _mk_belief(a_id, "local anchor belief", scope="project")
        )
        local_store.insert_belief(
            _mk_belief(b_id, "local neighbour belief", scope="project")
        )
        local_store.insert_edge(
            Edge(src=a_id, dst=b_id, type=EDGE_SUPPORTS, weight=1.0)
        )
    finally:
        local_store.close()

    # Human output: no [scope:...] tags.
    code, out = _run(local_db, "reason", "local anchor", "--seed-id", a_id)
    assert code == 0, f"expected exit 0; output:\n{out}"
    assert "[scope:" not in out, (
        f"unexpected [scope:...] annotation in local-only output:\n{out}"
    )

    # JSON: seeds[0].owning_scope is null.
    code, out = _run(
        local_db, "reason", "local anchor", "--seed-id", a_id, "--json"
    )
    assert code == 0, f"expected exit 0; output:\n{out}"
    payload = json.loads(out)
    assert payload["seeds"][0]["owning_scope"] is None
    # Hops should also have no owning_scope.
    for hop in payload["hops"]:
        assert hop["owning_scope"] is None, (
            f"hop {hop['id']!r} expected owning_scope=None in local-only store"
        )
