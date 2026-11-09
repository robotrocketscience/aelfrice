"""Planted-store tests for the wonder-reason fire-census probes (#1887).

Each test plants its own store under `tmp_path` (see
`tests/fire_census_planted.py`), makes the feature fire through aelfrice
code, and checks the probe. Every traceable probe is also checked silent on
an empty store, and every probe that guards an old schema is checked on a
raw sqlite store with that schema.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aelfrice.models import EDGE_CONTRADICTS, EDGE_SUPPORTS, Phantom
from aelfrice.wonder.lifecycle import wonder_ingest
from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_wonder_reason as pw
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

P = {p.item_id: p for p in pw.PROBES}
DISPATCH = pw.DISPATCH_PREFIX

EXPECTED_IDS = frozenset(
    "A311 A312 A313 A314 A315 A316 A317 A318 A319 A320 A321 A322 A323 A324 A325 A326"
    " A327 A328 A329 A330 A331 A332 A333 A334 A335 A336 A337 A338 A339 A340 A341 A342"
    " A343 A344 A345 A346 A347 A348 A349 A350 A351".split())


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    store = plant(tmp_path, monkeypatch)
    monkeypatch.setenv("AELFRICE_FEED_LOG", "1")
    yield store
    store.close()


def check(ps: PlantedStore, item_id: str, status: str, count: int | None = None) -> cc.Result:
    r = ps.probe(P[item_id])
    assert r.status == status, (item_id, r.to_json())
    if count is not None:
        assert r.count == count, (item_id, r.to_json())
    return r


def phantom(anchors: list[str], generator: str, content: str, score: float = 0.5) -> Phantom:
    return Phantom(constituent_belief_ids=tuple(anchors), generator=generator,
                   content=content, score=score)


def _conn(ps: PlantedStore) -> sqlite3.Connection:
    return ps.open()._conn  # pyright: ignore[reportPrivateUsage]


def test_module_ids_match_the_inventory_range() -> None:
    assert {p.item_id for p in pw.PROBES} == EXPECTED_IDS
    assert pw.TRACEABLE | pw.UNTRACEABLE == EXPECTED_IDS
    assert not pw.TRACEABLE & pw.UNTRACEABLE


@pytest.mark.parametrize("item_id", sorted(pw.TRACEABLE))
def test_empty_store_is_silent(ps: PlantedStore, item_id: str) -> None:
    check(ps, item_id, "silent", 0)


@pytest.mark.parametrize("item_id", sorted(pw.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(item_id: str) -> None:
    r = cc.run_one(P[item_id], None, 5.0)
    assert r.status == "untraceable" and r.count is None and r.note


def test_wonder_persist_graph_walk(ps: PlantedStore) -> None:
    seed = ps.belief("The release train leaves every Tuesday morning.")
    for t in ("Release notes are drafted on Monday.",
              "Hotfixes skip the release train.",
              "Tuesday releases need two approvals."):
        ps.edge(seed, ps.belief(t), EDGE_SUPPORTS)
    assert ps.cli("wonder", "--persist") == 0
    for i in ("A320", "A322", "A343", "A346", "A350"):
        check(ps, i, "fired", 3)
    r = check(ps, "A350", "fired", 3)
    assert r.detail["bfs"] == 3 and r.detail["relates_to_edges"] == 6
    assert r.detail["at_ingest_prior"] == 3 and r.detail["schema_guarded"] == 0
    r = check(ps, "A320", "fired", 3)
    assert r.detail["has_valid_to"] == 1 and r.detail["live"] == 3
    assert r.detail["dispatch"] == 0
    r = check(ps, "A343", "fired", 3)
    assert r.detail["bfs_rows"] == 3 and r.detail["joiner_speculative"] == 3
    r = check(ps, "A346", "fired", 3)
    assert r.detail["orphan_speculative"] == 0 and r.detail["has_corroborations"] == 1
    for i in ("A321", "A323", "A325", "A319", "A344"):
        check(ps, i, "silent", 0)


def _write_docs(ps: PlantedStore, rows: list[dict[str, Any]]) -> Path:
    docs = ps.root / "planted_input" / "docs.jsonl"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return docs


def test_persist_docs_axes(ps: PlantedStore) -> None:
    from aelfrice.wonder.dispatch import build_dispatch_payload

    a = ps.belief("Deploy cadence is weekly for the billing service.")
    b = ps.belief("Deploy cadence is daily for the billing service.")
    ps.edge(a, b, EDGE_CONTRADICTS)
    payload = build_dispatch_payload(ps.open(), "deploy cadence billing zyxqvw")
    names = [ax.name for ax in payload.research_axes]
    assert {"contradiction_resolution", "uncertainty_deep_dive",
            "coverage_extension"} <= set(names), names
    docs = _write_docs(ps, [{"axis_name": n, "content": f"Finding for axis {n}.",
                             "anchor_ids": list(payload.speculative_anchor_ids)}
                            for n in names])
    assert ps.cli("wonder", "--persist-docs", str(docs)) == 0
    r = check(ps, "A321", "fired", len(names))
    assert r.detail["feed_rows"] == 1 and r.detail["has_valid_to"] == 1
    check(ps, "A323", "fired", len(names))
    for i in ("A327", "A328", "A329"):
        check(ps, i, "fired", 1)
    r = check(ps, "A329", "fired", 1)
    assert r.detail["sub_axis"] == 1 and r.detail["unresolved_contradicts"] == 1
    r = check(ps, "A350", "fired", len(names))
    assert r.detail["dispatch"] == len(names)
    for i in ("A320", "A325"):
        check(ps, i, "silent", 0)


def test_persist_docs_without_conditional_axes(ps: PlantedStore) -> None:
    anchor = ps.belief("The cache warms on boot.")
    docs = _write_docs(ps, [{"axis_name": "domain_research", "content": "Doc one.",
                             "anchor_ids": [anchor]}])
    assert ps.cli("wonder", "--persist-docs", str(docs)) == 0
    check(ps, "A321", "fired", 1)
    check(ps, "A323", "fired", 1)
    for i in ("A327", "A328", "A329"):
        check(ps, i, "silent", 0)


def test_auto_wonder_pass(ps: PlantedStore) -> None:
    from aelfrice.wonder.auto_prompt import claim_generator, document_generator

    anchor = ps.belief("The scheduler retries failed jobs three times.")
    res = wonder_ingest(ps.open(), [
        phantom([anchor], claim_generator("internal_gap_analysis", "Claim one."), "Claim one."),
        phantom([anchor], document_generator("internal_gap_analysis", "Doc."), "Doc."),
    ], session_id="s1")
    assert res.inserted == 2
    r = check(ps, "A325", "fired", 2)
    assert r.detail["live"] == 2 and r.detail["retired"] == 0
    r = check(ps, "A350", "fired", 2)
    assert r.detail["auto"] == 2
    for i in ("A320", "A321", "A329"):
        check(ps, i, "silent", 0)


def test_auto_wonder_contradiction_axis(ps: PlantedStore) -> None:
    from aelfrice.wonder.auto_prompt import claim_generator

    anchor = ps.belief("The queue drains in under a minute.")
    gen = claim_generator("contradiction_resolution", "Claim about the queue.")
    assert wonder_ingest(ps.open(), [phantom([anchor], gen, "Claim about the queue.")],
                         session_id="s1").inserted == 1
    r = check(ps, "A329", "fired", 1)
    assert r.detail["auto_axis"] == 1 and r.detail["sub_axis"] == 0


def _three_phantoms(ps: PlantedStore) -> list[str]:
    ids = [ps.belief(f"Anchor statement number {i} about queues.") for i in range(3)]
    res = wonder_ingest(ps.open(), [
        phantom([ids[i]], "bfs+wonder_consolidation", f"Phantom {i} about queues.")
        for i in range(3)])
    assert res.inserted == 3
    rows = _conn(ps).execute(
        "SELECT id FROM beliefs WHERE origin = 'speculative' ORDER BY id").fetchall()
    return [str(r[0]) for r in rows]


def test_manual_gc(ps: PlantedStore) -> None:
    phs = _three_phantoms(ps)
    assert ps.cli("retire", phs[0]) == 0
    assert ps.cli("wonder", "--gc", "--gc-ttl-days", "0") == 0
    r = check(ps, "A319", "fired", 2)
    assert r.detail == {"gc_retired": 2, "autogc_deleted": 0}
    r = check(ps, "A344", "fired", 2)
    assert r.detail["phantoms"] == 3 and r.detail["has_cols"] == 1
    check(ps, "A347", "silent", 0)


def test_gc_dry_run_is_silent(ps: PlantedStore) -> None:
    _three_phantoms(ps)
    assert ps.cli("wonder", "--gc", "--gc-dry-run", "--gc-ttl-days", "0") == 0
    for i in ("A319", "A344"):
        check(ps, i, "silent", 0)


def test_gc_eligibility_counts_old_phantoms(ps: PlantedStore) -> None:
    _three_phantoms(ps)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    conn = _conn(ps)
    conn.execute("UPDATE beliefs SET created_at = ? WHERE origin = 'speculative'", (old,))
    conn.commit()
    r = check(ps, "A344", "silent", 0)
    assert r.detail["eligible_14d"] == 3


def test_sessionstart_autogc(ps: PlantedStore) -> None:
    _three_phantoms(ps)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    conn = _conn(ps)
    conn.execute("UPDATE beliefs SET created_at = ? WHERE origin = 'speculative'", (old,))
    conn.commit()
    rc = ps.hook("session_start", {"hook_event_name": "SessionStart", "session_id": "s1",
                                   "source": "startup", "cwd": str(ps.repo)})
    assert rc == 0
    r = check(ps, "A347", "fired", 3)
    assert r.detail == {"feed_rows": 1}
    check(ps, "A344", "fired", 3)
    check(ps, "A319", "silent", 0)


def test_review_apply(ps: PlantedStore) -> None:
    ids = [ps.belief(f"Review candidate belief {i} about logging.") for i in range(3)]
    out = ps.root / "planted_input" / "review.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert ps.cli("review", "--out", str(out)) == 0
    verdict = {ids[0]: "keep", ids[1]: "remove", ids[2]: "lock"}
    new: list[str] = []
    for ln in out.read_text("utf-8").splitlines():
        for bid, v in verdict.items():
            if f"| {bid} " in ln:
                ln = ln.replace(f"[ ] {v}", f"[x] {v}", 1)
        new.append(ln)
    out.write_text("\n".join(new) + "\n", encoding="utf-8")
    check(ps, "A316", "silent", 0)
    check(ps, "A317", "silent", 0)
    assert ps.cli("review", "--apply", "--out", str(out)) == 0
    r = check(ps, "A316", "fired", 3)
    want = {"lock": 1, "lock_held": 1, "remove": 1, "remove_held": 1, "keep": 1,
            "beliefs": 3, "post_ship": 3}
    assert r.detail == want
    check(ps, "A317", "fired", 1)


def test_reason_writeback_feedback(ps: PlantedStore) -> None:
    bid = ps.belief("The linter runs before every commit.")
    other = ps.belief("The formatter runs on save.")
    ps.feedback(other, 1.0, source="hook")
    assert ps.cli("feedback", other, "used", "--source", "agent") == 0
    check(ps, "A312", "silent", 0)
    assert ps.cli("feedback", bid, "used") == 0
    r = check(ps, "A312", "ambiguous", 1)
    assert r.detail == {"used": 1, "harmful": 0}


def test_auto_promote_on_user_evidence(ps: PlantedStore) -> None:
    from aelfrice.phantom_promotion_opportunity import (
        PhantomPromotionConfig,
        auto_promote_phantoms,
    )

    anchor = ps.belief("Anchor about the staging database.")
    text = "The staging database is rebuilt every night."
    assert wonder_ingest(ps.open(), [
        phantom([anchor], DISPATCH + "domain_research", text)]).inserted == 1
    assert wonder_ingest(ps.open(), [
        phantom([anchor], "bfs+wonder_consolidation", "Manual promote phantom.")]).inserted == 1
    manual = _conn(ps).execute(
        "SELECT id FROM beliefs WHERE content = 'Manual promote phantom.'").fetchone()[0]
    assert ps.cli("promote", manual) == 0
    r = check(ps, "A324", "silent", 0)
    assert r.detail["has_origin"] == 1 and r.detail["live_phantoms"] == 1
    ps.ingest_turns([{"role": "user", "text": text, "session_id": sid}
                     for sid in ("s2", "s3", "s4")])
    promoted = auto_promote_phantoms(store=ps.open(),
                                     config=PhantomPromotionConfig(auto_promote=True))
    assert len(promoted) == 1, promoted
    r = check(ps, "A324", "fired", 1)
    assert r.detail["promoted"] == 1 and r.detail["current"] == 1
    assert r.detail["reverted"] == 0 and r.detail["live_phantoms"] == 0


def test_promotion_opportunity_note(ps: PlantedStore) -> None:
    from aelfrice.phantom_promotion_opportunity import (
        PhantomPromotionConfig,
        evaluate_promotion_opportunities,
    )

    anchor = ps.belief("Anchor about the build cache.")
    store = ps.open()
    assert wonder_ingest(store, [
        phantom([anchor], "bfs+wonder_consolidation", "Build cache phantom.")]).inserted == 1
    pid = _conn(ps).execute(
        "SELECT id FROM beliefs WHERE origin = 'speculative'").fetchone()[0]
    now = datetime.now(timezone.utc)
    for k, sid in enumerate(("s2", "s3"), 1):
        store.record_corroboration(pid, source_type="transcript_ingest", session_id=sid,
                                   ts=(now + timedelta(hours=2 * k)).isoformat())
    found = evaluate_promotion_opportunities(
        store=store, session_id="s9", config=PhantomPromotionConfig(enabled=True))
    assert len(found) == 1
    r = check(ps, "A337", "fired", 1)
    assert r.detail == {"dedup_keys": 1}
    check(ps, "A341", "silent", 0)


def test_phantom_generation_signals(ps: PlantedStore) -> None:
    from aelfrice.phantom_trigger import PhantomGenerationConfig, evaluate_opportunities

    a = ps.belief("The retry budget is five attempts.")
    b = ps.belief("The retry budget is ten attempts.")
    cfg = PhantomGenerationConfig(enabled=True)
    first = evaluate_opportunities(prompt="what does QuuxFrobnicator do", store=ps.open(),
                                   session_id="s1", hit_count=0, config=cfg)
    assert {o.reason for o in first} == {"gap", "new_entity"}, first
    ps.edge(a, b, EDGE_CONTRADICTS)
    second = evaluate_opportunities(prompt="retry budget", store=ps.open(),
                                    session_id="s1", hit_count=2, config=cfg)
    assert [o.reason for o in second] == ["contradiction"], second
    r = check(ps, "A341", "fired", 3)
    assert r.detail == {"gap": 1, "new_entity": 1, "contradiction": 1, "other": 0,
                        "phantom_init": 1}
    for i in ("A338", "A339", "A340"):
        check(ps, i, "fired", 1)
    check(ps, "A337", "silent", 0)


def test_phantom_generation_disabled_writes_no_ring(ps: PlantedStore) -> None:
    from aelfrice.phantom_trigger import PhantomGenerationConfig, evaluate_opportunities

    out = evaluate_opportunities(prompt="what does QuuxFrobnicator do", store=ps.open(),
                                 session_id="s1", hit_count=0,
                                 config=PhantomGenerationConfig(enabled=False))
    assert out == []
    for i in ("A338", "A339", "A340", "A341"):
        check(ps, i, "silent", 0)


# --- old schemas ---------------------------------------------------------------


def _raw_ctx(tmp_path: Path, ddl: list[str], rows: list[tuple[str, tuple[Any, ...]]]) -> cc.Ctx:
    """A store built with raw sqlite3, closed, then opened read-only."""
    d = tmp_path / "raw"
    d.mkdir()
    db = d / "memory.db"
    conn = sqlite3.connect(str(db))
    try:
        for stmt in ddl:
            conn.execute(stmt)
        for sql, params in rows:
            conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
    ctx = cc.Ctx("store01", d, None, False, db, "repo")
    ctx.open()
    return ctx


def _run(ctx: cc.Ctx, item_id: str) -> cc.Result:
    return cc.run_one(P[item_id], ctx)


PRE_TYPE_DDL = [
    "CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, belief_type TEXT,"
    " beta_param REAL, locked INTEGER, created_at TEXT)",
    "CREATE TABLE edges (from_id TEXT, to_id TEXT, edge_type TEXT)",
]


def test_pre_type_schema_reports_silent_with_schema_flags(tmp_path: Path) -> None:
    ctx = _raw_ctx(tmp_path, PRE_TYPE_DDL, [
        ("INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?)",
         ("b1", "An old row.", "factual", 1.0, 0, "2026-01-01T00:00:00Z"))])
    try:
        r = _run(ctx, "A319")
        assert r.status == "silent" and r.detail["legacy_schema"] == 1
        assert r.detail["newest_yyyymm"] == 202601 and r.detail["beliefs_total"] == 1
        r = _run(ctx, "A344")
        assert r.status == "silent" and r.detail["has_cols"] == 0
        r = _run(ctx, "A324")
        assert r.status == "silent" and r.detail["has_origin"] == 0
        for i in sorted(pw.TRACEABLE):
            assert _run(ctx, i).status == "silent", i
    finally:
        ctx.close()


NO_VALID_TO_DDL = [
    "CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, type TEXT, origin TEXT,"
    " alpha REAL, beta REAL, created_at TEXT)",
    "CREATE TABLE edges (src TEXT, dst TEXT, type TEXT, weight REAL)",
    "CREATE TABLE belief_corroborations (id INTEGER PRIMARY KEY, belief_id TEXT,"
    " ingested_at TEXT, source_type TEXT, session_id TEXT, source_path_hash TEXT)",
    "CREATE TABLE feedback_history (id INTEGER PRIMARY KEY, belief_id TEXT, valence REAL,"
    " source TEXT, created_at TEXT)",
]


def test_schema_without_valid_to_still_counts_phantoms(tmp_path: Path) -> None:
    content = "Seed" + pw.JOINER + "neighbour"
    ctx = _raw_ctx(tmp_path, NO_VALID_TO_DDL, [
        ("INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?)",
         ("p1", content, "speculative", "speculative", 0.3, 1.0, "2026-06-01T00:00:00Z")),
        ("INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?)",
         ("p2", "Auto claim.", "speculative", "speculative", 0.3, 1.0, "2026-06-01T00:00:00Z")),
        ("INSERT INTO belief_corroborations (belief_id, source_type, source_path_hash)"
         " VALUES (?, ?, ?)", ("p1", "wonder_ingest", "bfs+wonder_consolidation@0.5000")),
        ("INSERT INTO belief_corroborations (belief_id, source_type, source_path_hash)"
         " VALUES (?, ?, ?)", ("p2", "wonder_ingest", "auto_wonder:v1:x:y@0.5000")),
        ("INSERT INTO edges VALUES (?, ?, ?, ?)", ("p1", "p2", "RELATES_TO", 1.0)),
    ])
    try:
        for i in ("A320", "A322", "A343", "A346"):
            r = _run(ctx, i)
            assert r.status == "fired" and r.count == 1, (i, r.to_json())
        assert _run(ctx, "A320").detail["has_valid_to"] == 0
        assert _run(ctx, "A343").detail["no_valid_to_col"] == 1
        r = _run(ctx, "A325")
        assert r.status == "fired" and r.count == 1 and r.detail["retired"] == 0
        r = _run(ctx, "A350")
        assert r.status == "fired" and r.count == 2
        assert r.detail["schema_guarded"] == 1 and r.detail["relates_to_edges"] == 1
        r = _run(ctx, "A321")
        assert r.status == "silent" and r.detail["has_valid_to"] == 0
        r = _run(ctx, "A316")
        assert r.status == "silent" and r.count == 0
        r = _run(ctx, "A344")
        assert r.status == "silent" and r.detail["has_cols"] == 0
    finally:
        ctx.close()


def test_joiner_fallback_fires_without_a_corroboration_row(tmp_path: Path) -> None:
    ctx = _raw_ctx(tmp_path, NO_VALID_TO_DDL, [
        ("INSERT INTO beliefs VALUES (?, ?, ?, ?, ?, ?, ?)",
         ("p1", "Seed" + pw.JOINER + "neighbour", "speculative", "speculative", 0.3, 1.0,
          "2026-06-01T00:00:00Z"))])
    try:
        r = _run(ctx, "A343")
        assert r.status == "fired" and r.count == 1 and r.detail["bfs_rows"] == 0
        r = _run(ctx, "A346")
        assert r.status == "silent" and r.detail["orphan_speculative"] == 1
    finally:
        ctx.close()
