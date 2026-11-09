"""Planted-store tests for the retrieval fire-census probes (#1887).

Each test plants its own store under `tmp_path` with aelfrice's own code
(`tests/fire_census_planted.py` pins HOME, AELFRICE_DB and
AELFRICE_USER_DB there), makes the feature run, and checks the probe.
Hook entry points run in a child that inherits the pinned environment.
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_retrieval as pr
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PROBES = {p.item_id: p for p in pr.PROBES}
TRACEABLE = sorted(pr.TRACEABLE)
# Folded fixes that report untraceable when a store has nothing to read.
EMPTY_UNTRACEABLE = frozenset({"A236", "A239", "A245", "A262"})

ORIGINAL_IDS = frozenset(f"A{i:03d}" for i in range(217, 268))

_LANE_ENV = (
    "AELFRICE_BFS", "AELFRICE_ORDER_POLICY", "AELFRICE_BM25_K3",
    "AELFRICE_BM25F_PER_FIELD", "AELFRICE_HRR_PERSIST", "AELFRICE_BM25F",
    "AELFRICE_META_BELIEF_BFS_DEPTH_BUDGET", "AELFRICE_META_BELIEF_HALF_LIFE",
    "AELFRICE_META_BELIEF_BM25F_ANCHOR_WEIGHT", "AELFRICE_PROJECT_CONTEXT",
    "AELFRICE_RETRIEVAL_TOKEN_BUDGET", "AELFRICE_L1_LIMIT", "AELFRICE_MEMORY_BLOCK",
    "AELFRICE_HRR_EXPAND",
)

TOPIC = ("The kumquat parser normalises citrus manifests before the orchard "
         "scheduler reads them; kumquat batches flush every night.")


@pytest.fixture()
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    for k in _LANE_ENV:
        monkeypatch.delenv(k, raising=False)
    s = plant(tmp_path, monkeypatch)
    yield s
    s.close()


def ups(ps: PlantedStore, prompt: str, session: str = "s1") -> None:
    rc = ps.hook("main", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                          "session_id": session, "cwd": str(ps.repo)})
    assert rc == 0


def run(ps: PlantedStore, item: str) -> cc.Result:
    return ps.probe(PROBES[item])


def seed_topic(ps: PlantedStore, n: int = 3) -> list[str]:
    return [ps.belief(f"{TOPIC} Note {i} on kumquat parser stage {i}.") for i in range(n)]


def audit_rows(ps: PlantedStore) -> int:
    return cc.jsonl_count(cc.jsonl_paths(ps.store_dir / pr.AUDIT_FILENAME))


def append_audit(ps: PlantedStore, row: dict[str, object]) -> None:
    with (ps.store_dir / pr.AUDIT_FILENAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def legacy_ctx(tmp_path: Path) -> cc.Ctx:
    """A pre-aelfrice schema: edges(from_id, to_id, edge_type), no type column."""
    d = tmp_path / "legacy"
    d.mkdir()
    db = d / "memory.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE edges (from_id TEXT, to_id TEXT, edge_type TEXT)")
    conn.execute("INSERT INTO edges VALUES ('a', 'b', 'POTENTIALLY_STALE')")
    conn.execute("CREATE TABLE beliefs (id TEXT, content TEXT)")
    conn.commit()
    conn.close()
    c = cc.Ctx("legacy01", d, None, False, db, "legacy")
    c.open()
    return c


# --- module shape and pinned vocabularies -----------------------------------


def test_ids_match_original_module() -> None:
    assert {p.item_id for p in pr.PROBES} == ORIGINAL_IDS
    assert len(pr.PROBES) == len(ORIGINAL_IDS)


def test_constants_match_code() -> None:
    from aelfrice import bm25, hook, hook_audit, hrr, hrr_expand, hrr_index, models, retrieval
    from aelfrice import entity_extractor as ee
    from aelfrice import sidecar_outcome
    assert pr.AUDIT_FILENAME == hook_audit.AUDIT_FILENAME
    assert pr.HOOK_UPS == hook.AUDIT_HOOK_USER_PROMPT_SUBMIT
    assert pr.HOOK_SESSION_START == hook.AUDIT_HOOK_SESSION_START
    assert set(pr.SIDECAR_OUTCOMES) == set(sidecar_outcome.SIDECAR_OUTCOMES)
    assert set(pr.ORDER_POLICIES) == set(retrieval.ORDER_POLICIES)
    assert pr.EDGE_TYPES == models.EDGE_TYPES
    assert set(pr.META_KEYS) == {retrieval.META_HALF_LIFE_KEY,
                                 retrieval.META_BM25F_ANCHOR_WEIGHT_KEY,
                                 retrieval.META_BFS_DEPTH_BUDGET_KEY}
    assert set(pr.ENTITY_KINDS) == {getattr(ee, n) for n in dir(ee) if n.startswith("KIND_")}
    assert pr.SIDECAR_MAGIC == bm25._SIDECAR_MAGIC  # pyright: ignore[reportPrivateUsage]
    assert pr.SIDECAR_VERSION == bm25._SIDECAR_VERSION  # pyright: ignore[reportPrivateUsage]
    assert pr.SERIALIZE_MAGIC == bm25._SERIALIZE_MAGIC  # pyright: ignore[reportPrivateUsage]
    assert pr.SERIALIZE_VERSION == bm25._SERIALIZE_VERSION  # pyright: ignore[reportPrivateUsage]
    assert "memory.db" + bm25._SIDECAR_SUFFIX == pr.BM25_SIDECAR_NAME  # pyright: ignore[reportPrivateUsage]
    assert pr.HRR_DIRNAME == hrr_index._PERSIST_DIRNAME  # pyright: ignore[reportPrivateUsage]
    assert pr.HRR_DEFAULT_DIM == hrr.DEFAULT_DIM
    assert pr._HASH_N_RE.pattern == retrieval._HASH_N_LITERAL_RE.pattern  # pyright: ignore[reportPrivateUsage]
    assert pr._STRUCT_MARKER_RE.pattern == hrr_index._STRUCT_MARKER_RE.pattern  # pyright: ignore[reportPrivateUsage]
    assert pr.PREFIX_CAP == hook.AUDIT_PROMPT_PREFIX_CAP
    assert pr.QUERY_ENTITY_CAP == retrieval.DEFAULT_QUERY_ENTITY_CAP
    assert pr.LOCK_USER == models.LOCK_USER
    assert tuple(pr.HRR_EXPAND_KINDS) == tuple(hrr_expand.hrr_expand_edge_types())


# --- empty store and untraceable probes ---------------------------------------


@pytest.mark.parametrize("item", sorted(set(TRACEABLE) - EMPTY_UNTRACEABLE))
def test_empty_store_is_silent(ps: PlantedStore, item: str) -> None:
    r = run(ps, item)
    assert r.status == "silent", (item, r.to_json())
    assert (r.count or 0) == 0, item


@pytest.mark.parametrize("item", sorted(EMPTY_UNTRACEABLE))
def test_empty_store_is_untraceable_for_folded_fixes(ps: PlantedStore, item: str) -> None:
    r = run(ps, item)
    assert (r.status, r.count) == ("untraceable", None), (item, r.to_json())


@pytest.mark.parametrize("item", sorted(pr.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(item: str) -> None:
    r = cc.run_one(PROBES[item], None, 5.0)
    assert (r.status, r.count) == ("untraceable", None)
    assert r.note


# --- UPS audit probes -----------------------------------------------------------


def test_a217_hash_n_boost(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    ps.belief("Issue #677 tracks the kumquat parser regression in the orchard.")
    ups(ps, "what is the status of #677 for the kumquat parser")
    r = run(ps, "A217")
    assert (r.status, r.count, r.detail["ups_rows_with_literal"]) == ("fired", 1, 1)


def test_a217_silent_without_literal(ps: PlantedStore) -> None:
    ps.belief("Issue #677 tracks the kumquat parser regression in the orchard.")
    ups(ps, "what is the status of the kumquat parser regression")
    r = run(ps, "A217")
    assert (r.status, r.count) == ("silent", 0)


def test_a217_silent_when_only_a_lock_carries_literal(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    assert ps.lock("Issue #677 is owned by the kumquat parser team.") == 0
    ups(ps, "what is the status of #677 for the kumquat parser")
    r = run(ps, "A217")
    assert (r.status, r.count, r.detail["ups_rows_with_literal"]) == ("silent", 0, 1)
    assert run(ps, "A243").count == 1  # the lock did reach the block


def test_a219_a249_gate_skips_bfs(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_topic(ps)
    monkeypatch.setenv("AELFRICE_BFS", "1")
    ups(ps, "what does the kumquat parser do with citrus manifests")
    r = run(ps, "A219")
    assert (r.status, r.count, r.detail["broad"]) == ("fired", 1, 1)
    r = run(ps, "A249")
    assert (r.status, r.count) == ("fired", 1)


def test_a219_silent_with_bfs_off(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "what does the kumquat parser do with citrus manifests")
    r = run(ps, "A219")
    assert (r.status, r.count, r.detail["evaluated"]) == ("silent", 0, 1)


def test_a241_order_policy(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_topic(ps)
    monkeypatch.setenv("AELFRICE_ORDER_POLICY", "locks_last")
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A241")
    assert (r.status, r.count, r.detail.get("locks_last")) == ("fired", 1, 1)


def test_a241_silent_on_lane(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A241")
    assert (r.status, r.count, r.detail.get("lane")) == ("silent", 0, 1)


def test_a243_locks_injected(ps: PlantedStore) -> None:
    seed_topic(ps)
    assert ps.lock("Always run the kumquat parser tests before merging.") == 0
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A243")
    assert (r.status, r.count, r.detail["locks_injected"]) == ("fired", 1, 1)


def test_a243_silent_without_locks(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser handle manifests")
    assert audit_rows(ps) == 1
    assert run(ps, "A243").status == "silent"


def test_a255_reference_manifest(ps: PlantedStore) -> None:
    seed_topic(ps)
    assert ps.lock("Reference: the kumquat parser spec lives in the orchard handbook.",
                   "--reference") == 0
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A255")
    assert (r.status, r.count, r.detail["ref_lines"], r.detail["reference_locks"]) == \
        ("fired", 1, 1, 1)


def test_a255_silent_on_frozen_lock(ps: PlantedStore) -> None:
    seed_topic(ps)
    assert ps.lock("Always run the kumquat parser tests before merging.") == 0
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A255")
    assert (r.status, r.count, r.detail["reference_locks"]) == ("silent", 0, 0)


def test_a266_budget_trim_coverage_line(ps: PlantedStore) -> None:
    filler = " ".join(["kumquat orchard manifest scheduler"] * 20)
    for i in range(60):
        ps.belief(f"Kumquat note {i}: {filler} stage {i}.")
    ups(ps, "kumquat orchard manifest scheduler notes")
    r = run(ps, "A266")
    assert (r.status, r.count) == ("fired", 1)


def test_a266_silent_when_nothing_trimmed(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    ups(ps, "how does the kumquat parser handle manifests")
    assert audit_rows(ps) == 1
    assert run(ps, "A266").status == "silent"


# --- HRR structural lane --------------------------------------------------------


def _marker_toml(ps: PlantedStore) -> None:
    (ps.repo / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\nprompt_shape_gate_enabled = false\n", encoding="utf-8")


def _supports_chain(ps: PlantedStore) -> str:
    ids = seed_topic(ps, 4)
    for src in ids[:3]:
        assert ps.edge(src, ids[3], "SUPPORTS")
    return ids[3]


def test_a239_structural_lane(ps: PlantedStore) -> None:
    target = _supports_chain(ps)
    _marker_toml(ps)
    ups(ps, f"SUPPORTS:{target}")
    r = run(ps, "A239")
    assert (r.status, r.count, r.detail["ups_marker_rows"]) == ("fired", 1, 1)
    assert (r.detail["signature_rows"], r.detail["edge_rows"]) == (1, 3)


def test_a239_silent_on_textual_fallback(ps: PlantedStore) -> None:
    _supports_chain(ps)
    _marker_toml(ps)
    ups(ps, "SUPPORTS:unknownid kumquat parser manifests")  # unknown target -> textual lane
    r = run(ps, "A239")
    assert (r.status, r.count, r.detail["ups_marker_rows"]) == ("silent", 0, 1)
    assert run(ps, "A249").count == 1  # the textual lane ran and recorded a gate reason


def test_a239_default_gate_skips_marker(ps: PlantedStore) -> None:
    target = _supports_chain(ps)
    ups(ps, f"SUPPORTS:{target}")
    r = run(ps, "A239")
    assert (r.status, r.count, r.detail["ups_marker_rows"]) == ("silent", 0, 1)


def test_a240_silent_without_probed_edges(ps: PlantedStore) -> None:
    a, b = seed_topic(ps, 2)
    assert ps.edge(a, b, "RELATES_TO")
    r = run(ps, "A240")
    assert (r.status, r.count, r.detail["probed_edges"], r.detail["edges_typed"]) == \
        ("silent", 0, 0, 1)


def test_a240_untraceable_with_probed_edges(ps: PlantedStore) -> None:
    _supports_chain(ps)
    r = run(ps, "A240")
    assert (r.status, r.count, r.detail["probed_edges"], r.detail["e_supports"]) == \
        ("untraceable", None, 3, 3)


# --- BM25 sidecar probes --------------------------------------------------------


def test_bm25_sidecar_default(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser handle manifests")
    ups(ps, "kumquat batches flush every night in the orchard")
    r = run(ps, "A244")
    assert (r.status, r.count, r.detail["lane_keyed"], r.detail["sidecar"]) == ("fired", 2, 2, 1)
    assert r.detail["full_rebuild"] >= 1
    r224 = run(ps, "A224")
    reuse = r.detail["fresh"] + r.detail["incremental"]
    assert (r224.status, r224.count, r224.detail["sidecar_file"]) == ("fired", 1 + reuse, 1)
    r = run(ps, "A223")
    assert (r.status, r.count, r.detail["anchor_weight"], r.detail["w3"]) == ("fired", 1, 3, 1)
    r = run(ps, "A252")
    assert (r.status, r.count) == ("fired", 1)
    r = run(ps, "A222")
    assert (r.status, r.detail["sidecar"], r.detail["k3_on"],
            r.detail["inner_version"]) == ("silent", 1, 0, pr.SERIALIZE_VERSION)
    r = run(ps, "A250")
    assert (r.status, r.detail["sidecar"], r.detail["per_field"]) == ("silent", 1, 0)


def test_a224_sidecar_reused_on_second_fire(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser handle manifests")
    first = run(ps, "A224")
    assert (first.status, first.count, first.detail["full_rebuild"]) == ("fired", 1, 1)
    ups(ps, "how does the kumquat parser handle manifests")
    second = run(ps, "A224")
    assert second.count == 2 and second.detail["fresh"] + second.detail["incremental"] == 1


def test_a244_silent_with_bm25f_off(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_topic(ps)
    monkeypatch.setenv("AELFRICE_BM25F", "0")
    ups(ps, "how does the kumquat parser handle manifests")
    assert audit_rows(ps) == 1
    r = run(ps, "A244")
    assert (r.status, r.detail["ran_unkeyed"], r.detail["ran_unkeyed_after_field_ship"]) == \
        ("silent", 1, 1)
    assert run(ps, "A224").status == "silent"


def test_a244_untraceable_for_fires_before_the_field_shipped(ps: PlantedStore) -> None:
    append_audit(ps, {"hook": pr.HOOK_UPS, "ts": "2026-06-01T00:00:00+00:00",
                      "expansion_gate_reason": "narrow"})
    r = run(ps, "A244")
    assert (r.status, r.count, r.detail["ran_unkeyed"]) == ("untraceable", None, 1)


def test_a244_fires_on_the_sidecar_alone(ps: PlantedStore) -> None:
    (ps.store_dir / pr.BM25_SIDECAR_NAME).write_bytes(pr.SIDECAR_MAGIC + bytes(8))
    r = run(ps, "A244")
    assert (r.status, r.count, r.detail["lane_keyed"], r.detail["sidecar"]) == \
        ("fired", 1, 0, 1), r


def test_a244_gate_skipped_warm_is_not_a_lane_fire(ps: PlantedStore) -> None:
    append_audit(ps, {"hook": pr.HOOK_UPS, "ts": "2026-10-01T00:00:00+00:00",
                      "prompt_shape_gate_skip": True, "sidecar_outcome": "fresh"})
    r = run(ps, "A244")
    assert (r.status, r.detail["warm_keyed"], r.detail["lane_keyed"]) == ("silent", 1, 0)


def test_a222_k3(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_topic(ps)
    monkeypatch.setenv("AELFRICE_BM25_K3", "8")
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A222")
    assert (r.status, r.count, r.detail["k3_on"], r.detail["k3_int"]) == ("fired", 1, 1, 8)
    assert r.detail["gen_match"] == 1


def test_a222_old_sidecar_is_not_read_as_k3(ps: PlantedStore) -> None:
    """An inner version 2 blob has no k3; the bytes after anchor_weight are not k3."""
    blob = (pr.SIDECAR_MAGIC + struct.pack("<I", pr.SIDECAR_VERSION) + struct.pack("<Q", 1)
            + struct.pack("<I", 0) + pr.SERIALIZE_MAGIC + struct.pack("<I", 2)
            + struct.pack("<i", 3) + struct.pack("<4d", 1.2, 0.75, 9.0, 8.0))
    (ps.store_dir / pr.BM25_SIDECAR_NAME).write_bytes(blob)
    r = run(ps, "A222")
    assert (r.status, r.count, r.detail["inner_version"], r.detail["k3_on"]) == \
        ("silent", 0, 2, 0)


def test_a250_per_field(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_topic(ps)
    monkeypatch.setenv("AELFRICE_BM25F_PER_FIELD", "1")
    ups(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A250")
    assert (r.status, r.count, r.detail["per_field"]) == ("fired", 1, 1)


# --- HRR persistence probes -------------------------------------------------------


def test_a236_a238_hrr_persisted(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice.retrieval import make_hrr_struct_cache
    _supports_chain(ps)
    monkeypatch.setenv("AELFRICE_HRR_PERSIST", "1")
    make_hrr_struct_cache(ps.open(), store_path=str(ps.db_file)).get()
    r = run(ps, "A238")
    assert (r.status, r.count, r.detail["meta_npz"]) == ("fired", 1, 1)
    r = run(ps, "A236")
    assert (r.status, r.count, r.detail["struct_npy"], r.detail["edges"]) == \
        ("untraceable", None, 1, 3)


def test_a238_silent_on_in_memory_build(ps: PlantedStore) -> None:
    from aelfrice.hrr_index import HRRStructIndexCache
    _supports_chain(ps)
    HRRStructIndexCache(store=ps.open()).get()  # what retrieve_v2 builds
    assert run(ps, "A238").status == "silent"
    r = run(ps, "A236")
    assert (r.status, r.detail["struct_npy"]) == ("untraceable", 0)


# --- store-table probes ------------------------------------------------------------


def test_a220_meta_belief_consumer(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import retrieval
    seed_topic(ps)
    store = ps.open()
    assert retrieval.install_bfs_depth_budget_meta_belief(store, now_ts=int(time.time()))
    monkeypatch.setenv("AELFRICE_META_BELIEF_BFS_DEPTH_BUDGET", "1")
    retrieval.retrieve(store, "kumquat parser manifests", user_locks=())
    r = run(ps, "A220")
    assert (r.status, r.count, r.detail["meta:retrieval.bfs_depth_budget"],
            r.detail["installed"]) == ("fired", 1, 1, 1)


def test_a220_silent_when_installed_but_off(ps: PlantedStore) -> None:
    from aelfrice import retrieval
    seed_topic(ps)
    store = ps.open()
    assert retrieval.install_bfs_depth_budget_meta_belief(store, now_ts=int(time.time()))
    retrieval.retrieve(store, "kumquat parser manifests", user_locks=())
    r = run(ps, "A220")
    assert (r.status, r.count, r.detail["installed"]) == ("silent", 0, 1)


def test_a230_entity_writer(ps: PlantedStore) -> None:
    from aelfrice.entity_extractor import extract_entities
    text = ("Edit src/aelfrice/retrieval.py on branch feat/kumquat-parser to fix "
            "error E1234 before v1.2.3 ships.")
    ents = extract_entities(text)
    expected = len({(e.lower, e.span_start) for e in ents})
    assert expected > 0
    ps.belief(text)
    r = run(ps, "A230")
    assert (r.status, r.count, r.detail["beliefs_with_entities"]) == ("fired", expected, 1)
    assert r.detail.get("file_path", 0) >= 1


# --- L2.5 entity lane (rebuilt from audit + belief_entities) -----------------------


ENTITY_BELIEF = "The kumquat parser entry point is src/kumquat/parser_main.py in the orchard."


def test_a245_entity_lane_fired(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    ps.belief(ENTITY_BELIEF)
    ups(ps, "what does src/kumquat/parser_main.py do")
    r = run(ps, "A245")
    assert (r.status, r.detail["ups_rows"], r.detail["rows_with_entities"],
            r.detail["rows_pool_nonempty"]) == ("fired", 1, 1, 1)
    assert r.count == 1 and r.detail["injected_l1_matched"] >= 1


def test_a245_silent_when_no_entity_matches(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    ps.belief(ENTITY_BELIEF)
    ups(ps, "how do kumquat batches flush every night")
    r = run(ps, "A245")
    assert (r.status, r.count, r.detail["ups_rows"], r.detail["rows_injected_match"]) == \
        ("silent", 0, 1, 0)


def test_a245_ignores_a_locked_entity_belief(ps: PlantedStore) -> None:
    assert ps.lock(ENTITY_BELIEF) == 0
    ups(ps, "what does src/kumquat/parser_main.py do")
    r = run(ps, "A245")
    assert (r.status, r.detail["rows_pool_nonempty"]) == ("silent", 0)


# --- POTENTIALLY_STALE penalty ------------------------------------------------------


def test_a254_untraceable_with_stale_marker(ps: PlantedStore) -> None:
    a, b = seed_topic(ps, 2)
    assert ps.edge(a, b, "POTENTIALLY_STALE")
    r = run(ps, "A254")
    assert (r.status, r.count, r.detail["stale_edges"], r.detail["stale_targets_live"]) == \
        ("untraceable", None, 1, 1)


def test_a254_silent_without_stale_marker(ps: PlantedStore) -> None:
    a, b = seed_topic(ps, 2)
    assert ps.edge(a, b, "SUPPORTS")
    r = run(ps, "A254")
    assert (r.status, r.count, r.detail["stale_edges"]) == ("silent", 0, 0)


def test_legacy_edges_schema(tmp_path: Path) -> None:
    c = legacy_ctx(tmp_path)
    try:
        r = cc.run_one(PROBES["A254"], c)
        assert (r.status, r.detail["legacy_schema"], r.detail["stale_edges"]) == ("silent", 1, 1)
        r = cc.run_one(PROBES["A262"], c)
        assert (r.status, r.count) == ("untraceable", None)
        r = cc.run_one(PROBES["A240"], c)
        assert (r.status, r.detail["edges_typed"]) == ("silent", 0)
    finally:
        c.close()


# --- temporal spine lane ---------------------------------------------------------------


def test_a262_spine_neighbours_in_one_block(ps: PlantedStore) -> None:
    a, b = seed_topic(ps, 2)
    assert ps.edge(a, b, "TEMPORAL_NEXT")
    ups(ps, "how does the kumquat parser handle citrus manifests")
    r = run(ps, "A262")
    assert (r.status, r.detail["tn_edges"], r.detail["ups_post"]) == ("fired", 1, 1)
    assert r.count == r.detail["adj_post"] == 1


def test_a262_silent_when_block_has_no_neighbours(ps: PlantedStore) -> None:
    seed_topic(ps, 2)
    # Locked beliefs reach the block as L0, so their edge is never L1 adjacency.
    x = ps.belief("The greenhouse heater thermostat runs at dawn.", locked=True)
    y = ps.belief("Greenhouse vents open when the heater stops.", locked=True)
    assert ps.edge(x, y, "TEMPORAL_NEXT")
    ups(ps, "how does the kumquat parser handle citrus manifests")
    r = run(ps, "A262")
    assert (r.status, r.count, r.detail["tn_edges"], r.detail["adj_post"]) == \
        ("silent", 0, 1, 0)
    assert r.detail["ups_post"] == 1


def test_a262_untraceable_without_spine_edges(ps: PlantedStore) -> None:
    seed_topic(ps, 3)
    ups(ps, "how does the kumquat parser handle citrus manifests")
    r = run(ps, "A262")
    assert (r.status, r.detail["tn_edges"], r.detail["ups_post"]) == ("untraceable", 0, 1)


# --- type-aware compression ---------------------------------------------------------------


SNAPSHOT = ("The kumquat parser currently runs at version nine in the orchard. "
            "It reads citrus manifests from the scheduler queue and writes batches "
            "that flush every night after the greenhouse closes.")


def test_a263_rebuild_v14_packs_compressed_belief(ps: PlantedStore) -> None:
    from aelfrice.context_rebuilder import rebuild_v14
    from aelfrice.rebuild_log import RecentTurn
    ps.belief(SNAPSHOT, retention_class="snapshot")
    log = ps.store_dir / "rebuild_logs" / "s1.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    turns = [RecentTurn(role="user", text="how does the kumquat parser read citrus manifests",
                        session_id="s1")]
    rebuild_v14(turns, ps.open(), rebuild_log_path=log, rebuild_log_enabled=True,
                session_id_for_log="s1", use_type_aware_compression=True)
    r = run(ps, "A263")
    assert (r.status, r.count, r.detail["compressible"], r.detail["elig_snapshot"]) == \
        ("fired", 1, 1, 1)
    assert r.detail["rb14_packed_compressible"] == 1


def test_a263_untraceable_without_a_rebuild_record(ps: PlantedStore) -> None:
    ps.belief(SNAPSHOT, retention_class="snapshot")
    r = run(ps, "A263")
    assert (r.status, r.count, r.detail["compressible"], r.detail["rb_rows"]) == \
        ("untraceable", None, 1, 0)


def test_a263_silent_when_nothing_is_compressible(ps: PlantedStore) -> None:
    ps.belief(SNAPSHOT, retention_class="fact")
    ps.belief("Short snapshot.", retention_class="snapshot")
    r = run(ps, "A263")
    assert (r.status, r.detail["compressible"], r.detail["elig_snapshot"]) == ("silent", 0, 1)


# --- global log probe ----------------------------------------------------------------


def _ups_stderr_to_log(ps: PlantedStore, prompt: str) -> None:
    """A host wrapper `2>> ~/.aelfrice/logs/hook-failures.log` around the UPS hook."""
    log = ps.global_dir / "logs" / "hook-failures.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    ps.close()
    ps.assert_pinned()
    code = "import sys; from aelfrice.hook import main as f; sys.exit(f() or 0)"
    payload = {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
               "session_id": "s1", "cwd": str(ps.repo)}
    with log.open("ab") as err:
        cp = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload).encode(),
                            env=dict(os.environ), cwd=str(ps.repo), stdout=subprocess.DEVNULL,
                            stderr=err, timeout=120, check=False)
    assert cp.returncode == 0


def test_a251_placeholder_warning(ps: PlantedStore) -> None:
    seed_topic(ps)
    (ps.repo / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_signed_laplacian = true\n", encoding="utf-8")
    _ups_stderr_to_log(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A251")
    assert (r.status, r.count) == ("fired", 1)


def test_a251_silent_when_flag_unset(ps: PlantedStore) -> None:
    seed_topic(ps)
    _ups_stderr_to_log(ps, "how does the kumquat parser handle manifests")
    r = run(ps, "A251")
    assert (r.status, r.count, r.detail["log"]) == ("silent", 0, 1)


def test_a251_global_ctx_without_global_dir_errors() -> None:
    c = cc.Ctx("global", Path("."), None, True, None, "global")
    r = cc.run_one(PROBES["A251"], c)
    assert (r.status, r.note) == ("error", "LookupError")
