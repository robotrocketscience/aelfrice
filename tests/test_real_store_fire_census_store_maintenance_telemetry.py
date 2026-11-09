"""Planted-store tests for the store-maintenance-telemetry census probes (#1887).

Each test plants a store under its own `tmp_path` with HOME, AELFRICE_DB and
AELFRICE_USER_DB pinned there (`tests/fire_census_planted.py`), makes the
feature fire through aelfrice's own code, and checks the probe. Named,
legacy and global contexts are built over folders under the same root, laid
out the way the census runner sees them.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_store_maintenance_telemetry as pm
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PROBE = {p.item_id: p for p in pm.PROBES}
EXPECTED_IDS = {"A038", "A039"} | {f"A{i:03d}" for i in range(268, 311)}
GLOBAL_LOG = frozenset(p.item_id for p in pm.PROBES if p.kind == "global-log")

# On a fresh planted store most traceable probes are silent. The fixed A292
# and A298 probes report untraceable when their feature had no input, and
# every store aelfrice opens is in WAL mode, so A309 fires (its silent case
# is a plain sqlite file, tested below).
EMPTY_STATUS: dict[str, str] = {"A280": "untraceable", "A292": "untraceable",
                                "A298": "untraceable", "A309": "fired"}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    store = plant(tmp_path, monkeypatch)
    yield store
    store.close()


def global_ctx(dirpath: Path) -> cc.Ctx:
    return cc.Ctx("global", dirpath, dirpath, True, dirpath / "memory.db", "global")


def store_ctx(ps: PlantedStore, layout: cc.Layout = "repo") -> cc.Ctx:
    ps.close()
    return cc.Ctx("store01", ps.store_dir, ps.global_dir, False, ps.db_file, layout)


def run(item_id: str, ctx: cc.Ctx) -> cc.Result:
    ctx.open()
    try:
        return cc.run_one(PROBE[item_id], ctx)
    finally:
        ctx.close()


def default_ctx(ps: PlantedStore, item_id: str) -> cc.Ctx:
    return global_ctx(ps.global_dir) if item_id in GLOBAL_LOG else store_ctx(ps)


def fired(r: cc.Result, n: int | None = None) -> None:
    assert r.status == "fired", r
    if n is not None:
        assert r.count == n, r


def silent(r: cc.Result) -> None:
    assert r.status == "silent", r


# --- shape -----------------------------------------------------------------


def test_probe_ids_match_the_subsystem() -> None:
    assert {p.item_id for p in pm.PROBES} == EXPECTED_IDS
    assert len(pm.PROBES) == len(EXPECTED_IDS)
    assert pm.TRACEABLE | pm.UNTRACEABLE == EXPECTED_IDS


@pytest.mark.parametrize("item_id", sorted(pm.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(item_id: str) -> None:
    r = cc.run_one(PROBE[item_id], None)
    assert r.status == "untraceable" and r.count is None and r.note


@pytest.mark.parametrize("item_id", sorted(pm.TRACEABLE))
def test_empty_store(ps: PlantedStore, item_id: str) -> None:
    ps.remove_precreated_sentinels()
    r = run(item_id, default_ctx(ps, item_id))
    want = EMPTY_STATUS.get(item_id, "silent")
    assert r.status == want, r
    if want == "silent":
        assert r.count in (0, None), r


# --- A268 marker identity sidecar --------------------------------------------


def named_ctx(ps: PlantedStore, name: str = "side") -> cc.Ctx:
    named = ps.store_dir / "stores" / name / "memory.db"
    return cc.Ctx("named01", named.parent, ps.global_dir, False, named, "named")


def test_a268_git_store_marker_seeds_identity(ps: PlantedStore) -> None:
    from aelfrice import db_paths
    # Repo store: _open_store seeds the repo identity sidecar too, which the
    # probe must not count.
    ps.close()
    db_paths._open_store(db_paths.resolve_store()).close()  # pyright: ignore[reportPrivateUsage]
    assert (ps.store_dir / "identity").is_file()
    silent(run("A268", store_ctx(ps)))
    nctx = named_ctx(ps)
    assert nctx.db_file is not None
    nctx.store_dir.mkdir(parents=True)
    silent(run("A268", nctx))
    marker = ps.repo / ".aelfrice.toml"
    marker.write_text('[store]\npath = "git:side"\n', encoding="utf-8")
    sel = db_paths.StoreSelection(nctx.db_file, db_paths.STORE_RULE_MARKER, marker,
                                  git_store="side")
    db_paths._open_store(sel).close()  # pyright: ignore[reportPrivateUsage]
    fired(run("A268", named_ctx(ps)), 1)


# --- A269 clamp-ghosts -------------------------------------------------------


def test_a269_clamp_ghosts(ps: PlantedStore) -> None:
    from aelfrice.clamp_ghosts import clamp_ghost_alphas
    ps.belief("A ghost alpha row with no audit trail.", alpha=9.0)
    ps.belief("A second ghost alpha row.", alpha=7.0)
    res = clamp_ghost_alphas(ps.open(), dry_run=False)
    assert res.clamped == 2
    fired(run("A269", store_ctx(ps)), 2)


# --- A270 classify-orphans consent ------------------------------------------


def test_a270_consent_scope(ps: PlantedStore) -> None:
    from aelfrice import llm_classifier as lc
    gctx = global_ctx(ps.global_dir)
    p = lc.sentinel_path()
    assert ps.global_dir.resolve() in p.resolve().parents
    lc.write_sentinel(p, model="m", scopes=(lc.CONSENT_SCOPE_ONBOARD_CANDIDATES,))
    silent(run("A270", gctx))  # onboard-only consent does not count
    lc.write_sentinel(p, model="m", scopes=(lc.CONSENT_SCOPE_ONBOARD_CANDIDATES,
                                            lc.CONSENT_SCOPE_STORED_BELIEFS))
    fired(run("A270", global_ctx(ps.global_dir)), 1)


# --- A273 promote-retention --------------------------------------------------


def test_a273_promote_retention(ps: PlantedStore) -> None:
    from aelfrice.doctor import promote_retention
    s = ps.open()
    bid = ps.belief("Snapshot fact corroborated across sessions.", retention_class="snapshot")
    t0 = datetime.now(timezone.utc) - timedelta(hours=10)
    for i, sess in enumerate(("s1", "s2", "s3")):
        s.record_corroboration(bid, source_type="transcript_ingest", session_id=sess,
                               source_path_hash=f"h{i}", ts=_iso(t0 + timedelta(hours=3 * i)))
    rep = promote_retention(s)
    assert rep.promoted == 1
    fired(run("A273", store_ctx(ps)), 1)


# --- A281 legacy auto-migration ---------------------------------------------


def _legacy_pre_v1(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path))
    try:
        c.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, content_hash TEXT, "
                  "alpha REAL, beta REAL, type TEXT, lock_level TEXT, locked_at TEXT, "
                  "created_at TEXT, last_retrieved_at TEXT)")
        c.execute("CREATE TABLE edges (src TEXT, dst TEXT, type TEXT, weight REAL)")
        now = _iso(datetime.now(timezone.utc))
        for i in range(n):
            c.execute("INSERT INTO beliefs VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (f"b{i}", f"legacy row {i}", f"hash{i}", 1.0, 1.0, "factual", "none",
                       None, now, None))
        c.commit()
    finally:
        c.close()


def legacy_ctx(ps: PlantedStore, db: Path) -> cc.Ctx:
    return cc.Ctx("legacy01", db.parent, ps.global_dir, False, db, "legacy")


def test_a281_legacy_auto_migration(ps: PlantedStore) -> None:
    from aelfrice import doctor
    # Silent: a modern legacy-slot store whose rows are all logged.
    ps.ingest("We decided the build uses uv for every environment.", session_id="s1")
    ps.close()
    silent(run("A281", legacy_ctx(ps, ps.db_file)))
    leg = ps.root / "home_projects" / "abc123def456" / "memory.db"
    _legacy_pre_v1(leg, 2)
    # Eligible but not migrated yet: silent, count 0, eligible flag set.
    r = run("A281", legacy_ctx(ps, leg))
    assert r.status == "silent" and r.count == 0 and r.detail["eligible"] == 1, r
    found = doctor._check_legacy_schema_dbs(projects_dir=leg.parent.parent)  # pyright: ignore[reportPrivateUsage]
    assert len(found) == 1
    migrated, failed = doctor._auto_migrate_legacy_dbs(found)  # pyright: ignore[reportPrivateUsage]
    assert len(migrated) == 1 and not failed
    r = run("A281", legacy_ctx(ps, leg))
    fired(r, 2)
    assert r.detail["migrated"] == 1
    # Silent: the same rows probed as a repository store (wrong store kind).
    silent(run("A281", cc.Ctx("store01", leg.parent, ps.global_dir, False, leg, "repo")))


def test_a281_empty_legacy_schema_is_skipped(ps: PlantedStore) -> None:
    leg = ps.root / "home_projects" / "000000000000" / "memory.db"
    _legacy_pre_v1(leg, 0)
    r = run("A281", legacy_ctx(ps, leg))
    assert r.status == "silent" and r.count == 0 and r.detail["empty_legacy"] == 1, r


# --- A280 doctor telemetry stats, A293 / A299 / A303 hook audit ---------------


def test_ups_hook_fires_audit_telemetry_and_rebuild_log(ps: PlantedStore) -> None:
    ps.ingest("We decided to always use uv for Python environments.", session_id="s1")
    ps.ingest("The census reads byte copies of each store read-only.", session_id="s1")
    rc = ps.hook("main", {"hook_event_name": "UserPromptSubmit",
                          "prompt": "which python environment manager do we use with uv",
                          "session_id": "s1", "cwd": str(ps.repo)})
    assert rc == 0
    a = run("A293", store_ctx(ps))
    fired(a, 1)
    assert a.detail.get("user_prompt_submit") == 1
    fired(run("A299", store_ctx(ps)), 1)
    r = run("A303", store_ctx(ps))
    fired(r, 1)
    assert r.detail.get("null_scores") == 1 and r.detail["files"] == 1
    t = run("A280", store_ctx(ps))
    # The rows exist, but the hook wrote them: doctor leaves no trace of a read.
    assert t.status == "untraceable" and t.count is None, t
    assert t.detail["ups_rows"] >= 1 and t.detail["doctor_reads"] == 1
    # Doctor never reads the telemetry of a store outside the repo layout.
    n = run("A280", store_ctx(ps, "named"))
    assert n.status == "untraceable" and n.count is None and n.detail["doctor_reads"] == 0


# --- A283 aelf migrate -------------------------------------------------------


def test_a283_migrate_legacy_global(ps: PlantedStore) -> None:
    from aelfrice import ingest
    from aelfrice.migrate import migrate
    from aelfrice.store import MemoryStore
    ps.ingest("We decided the repository keeps one lockfile.", session_id="s1")
    silent(run("A283", store_ctx(ps)))
    legacy = ps.root / "legacy_global" / "memory.db"
    legacy.parent.mkdir()
    ls = MemoryStore(str(legacy))
    try:
        n = ingest.ingest_turn(ls, "We decided the parser always rejects tabs. "
                                   "The retriever must cap results at ten.", "user",
                               session_id="g1")
    finally:
        ls.close()
    assert n >= 1
    rep = migrate(legacy_path=legacy, target_path=ps.db_file, project_root=ps.repo,
                  apply=True, copy_all=True)
    assert rep.counts.inserted_beliefs == n
    r = run("A283", store_ctx(ps))
    assert r.status == "ambiguous" and r.count == n, r
    silent(run("A283", store_ctx(ps, "legacy")))


# --- A284 aelf store split ---------------------------------------------------


def test_a284_store_split_copy(ps: PlantedStore) -> None:
    from aelfrice.migrate import copy_belief
    from aelfrice.store import MemoryStore
    ps.ingest("We decided the parser module owns its own fixtures.", session_id="s1")
    s = ps.open()
    row = s._conn.execute("SELECT id FROM beliefs LIMIT 1").fetchone()  # pyright: ignore[reportPrivateUsage]
    belief = s.get_belief(str(row[0]))
    assert belief is not None
    ps.close()
    nctx = named_ctx(ps)
    assert nctx.db_file is not None
    nctx.store_dir.mkdir(parents=True)
    MemoryStore(str(nctx.db_file)).close()
    silent(run("A284", named_ctx(ps)))
    target = MemoryStore(str(nctx.db_file))
    try:
        copy_belief(target, belief)
    finally:
        target.close()
    r = run("A284", named_ctx(ps))
    fired(r, 1)
    assert r.detail == {"unlogged": 1, "also_in_repo_store": 1}
    silent(run("A284", store_ctx(ps)))


# --- A285 feed log -----------------------------------------------------------


def test_a285_feed_log(ps: PlantedStore) -> None:
    assert ps.lock("Never push from the home directory repo.") == 0
    r = run("A285", store_ctx(ps))
    fired(r, 1)
    assert r.detail.get("belief.locked") == 1


# --- A287 content-hash dedup -------------------------------------------------


def test_a287_content_hash_hit(ps: PlantedStore) -> None:
    assert ps.lock("Always run the census against byte copies.") == 0
    assert ps.lock("Always run the census against byte copies.") == 0
    r = run("A287", store_ctx(ps))
    fired(r, 1)
    assert r.detail["runtime_hit"] == 1 and r.detail["consolidation_migration"] == 0


# --- A289 version vectors ----------------------------------------------------


def test_a289_version_vectors(ps: PlantedStore) -> None:
    n = ps.ingest("We decided the store uses WAL journaling.", session_id="s1")
    assert n >= 1
    r = run("A289", store_ctx(ps))
    fired(r)
    assert r.detail["belief_versions"] >= n and r.detail["log_versions"] >= 1


# --- A292 guarded migrations -------------------------------------------------


def test_a292_guarded_migration_marker(ps: PlantedStore,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice.store import MemoryStore
    ps.close()
    r = run("A292", store_ctx(ps))
    assert r.status == "untraceable" and r.detail["has_schema_meta"] == 1, r

    def _maybe_rehash_speculative_v2(self: MemoryStore) -> None:
        raise RuntimeError("planted failure")

    monkeypatch.setattr(MemoryStore, "_maybe_rehash_speculative_v2",
                        _maybe_rehash_speculative_v2)
    MemoryStore(str(ps.db_file)).close()
    r = run("A292", store_ctx(ps))
    fired(r, 1)
    assert r.detail["failure_markers"] == 1


# --- A294 classifier stamp ---------------------------------------------------


def test_a294_classifier_stamp(ps: PlantedStore) -> None:
    ps.ingest("We decided the hook never blocks the prompt.", session_id="s1")
    r = run("A294", store_ctx(ps))
    fired(r)
    assert r.detail["unstamped"] == 0 and r.detail["distinct_versions"] == 1


# --- A295 legacy ingest_log synthesis ---------------------------------------


def test_a295_legacy_log_synthesis(ps: PlantedStore) -> None:
    from aelfrice.store import SCHEMA_META_LEGACY_LOG_SYNTH, MemoryStore
    ps.belief("A pre-v2 belief with no ingest_log row.")
    s = ps.open()
    # Simulate a store written before v2.0: the synthesis marker is absent.
    s._conn.execute("DELETE FROM schema_meta WHERE key = ?",  # pyright: ignore[reportPrivateUsage]
                    (SCHEMA_META_LEGACY_LOG_SYNTH,))
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    ps.close()
    MemoryStore(str(ps.db_file)).close()
    fired(run("A295", store_ctx(ps)), 1)


# --- A297 MCP cleanup sentinel ----------------------------------------------


def test_a297_mcp_cleanup_sentinel(ps: PlantedStore) -> None:
    from aelfrice import mcp_cleanup
    g = ps.root / "g297"
    g.mkdir()
    silent(run("A297", global_ctx(g)))
    assert ps.root in mcp_cleanup.MCP_CLEANUP_SENTINEL.resolve().parents
    res = mcp_cleanup.maybe_clean_up_mcp(sentinel_path=g / "mcp-surface-removed",
                                         config_paths=[],
                                         receipt_path=ps.root / "no-receipt.toml")
    assert res.ran
    fired(run("A297", global_ctx(g)), 1)


# --- A298 lock expiry sweep -------------------------------------------------


def test_a298_lock_expiry_sweep(ps: PlantedStore) -> None:
    from aelfrice.store import MemoryStore
    now = datetime.now(timezone.utc)
    ps.belief("A time-boxed lock that has expired.", locked=True,
              lock_expires_at=_iso(now - timedelta(days=1)))
    ps.belief("A time-boxed lock still in force.", locked=True,
              lock_expires_at=_iso(now + timedelta(days=30)))
    # A due lock the open-time sweep has not flipped yet reads as an error.
    r = run("A298", store_ctx(ps))
    assert r.status == "error" and r.count == 1 and r.detail["due_user"] == 1, r
    MemoryStore(str(ps.db_file)).close()
    r = run("A298", store_ctx(ps))
    fired(r, 1)
    assert r.detail["sweep_marker"] == 1


def test_a298_window_not_due_is_silent(ps: PlantedStore) -> None:
    ps.belief("A time-boxed lock still in force.", locked=True,
              lock_expires_at=_iso(datetime.now(timezone.utc) + timedelta(days=30)))
    r = run("A298", store_ctx(ps))
    assert r.status == "silent" and r.detail["windowed"] == 1, r


# --- A301 project_context stamping ------------------------------------------


def test_a301_project_context_stamp(ps: PlantedStore) -> None:
    from aelfrice import db_paths, ingest
    # A direct MemoryStore(path) open stamps nothing.
    n0 = ps.ingest("We decided direct opens carry no identity.", session_id="s1")
    silent(run("A301", store_ctx(ps)))
    s = db_paths._open_store()  # pyright: ignore[reportPrivateUsage]
    try:
        n = ingest.ingest_turn(s, "We decided the repo store stamps its identity.", "user",
                               session_id="s2")
    finally:
        s.close()
    assert n >= 1
    r = run("A301", store_ctx(ps))
    # The identity-carrying open also backfills the n0 rows written before it.
    fired(r, n + n0)
    assert r.detail["backfill_marker"] == 1 and r.detail["empty"] == 0


# --- A306 retired-content revival -------------------------------------------


def test_a306_reassert_revive(ps: PlantedStore) -> None:
    text = "Never read anything under the excluded folders."
    assert ps.lock(text) == 0
    silent(run("A306", store_ctx(ps)))
    s = ps.open()
    bid = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT id FROM beliefs WHERE lock_level='user'").fetchone()[0]
    s.soft_delete_belief(bid)
    assert ps.lock(text) == 0
    r = run("A306", store_ctx(ps))
    fired(r, 1)
    assert r.detail["revive"] == 1 and r.detail["retired"] == 0


def test_a306_no_belief_schema_is_untraceable(ps: PlantedStore) -> None:
    plain = ps.root / "plain306" / "memory.db"
    plain.parent.mkdir()
    sqlite3.connect(str(plain)).close()
    r = run("A306", cc.Ctx("store02", plain.parent, ps.global_dir, False, plain, "repo"))
    assert r.status == "untraceable", r


# --- A307 session-delta telemetry -------------------------------------------


def test_a307_session_delta(ps: PlantedStore) -> None:
    g = ps.root / "g307"
    g.mkdir()
    silent(run("A307", global_ctx(g)))
    ps.ingest("We decided telemetry stays local.", session_id="sd1")
    tpath = g / "telemetry.jsonl"
    assert ps.cli("session-delta", "--id", "sd1", "--telemetry-path", str(tpath)) == 0
    r = run("A307", global_ctx(g))
    fired(r, 1)
    assert r.detail == {"attributed": 0, "unattributed": 1}


# --- A308 sessions table ----------------------------------------------------


def test_a308_commit_ingest_session(ps: PlantedStore) -> None:
    from aelfrice.hook_commit_ingest import _ingest_message  # pyright: ignore[reportPrivateUsage]
    ps.close()
    _ingest_message("feat: the new fetcher supersedes the old fetcher",
                    session_id="c1", project_context="")
    r = run("A308", store_ctx(ps))
    fired(r, 1)
    assert r.detail.get("commit-ingest") == 1


# --- A309 WAL setting --------------------------------------------------------


def test_a309_wal(ps: PlantedStore) -> None:
    plain = ps.root / "plain" / "memory.db"
    plain.parent.mkdir()
    c = sqlite3.connect(str(plain))
    try:
        c.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT)")
        c.commit()
    finally:
        c.close()
    silent(run("A309", cc.Ctx("store02", plain.parent, ps.global_dir, False, plain, "repo")))
    r = run("A309", store_ctx(ps))
    fired(r, 1)
    assert r.detail["store_generation"] == 1
