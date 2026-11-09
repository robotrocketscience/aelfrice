"""Planted-store tests for the cli-setup-hosts-export census probes (#1887).

Each test plants a store under its own `tmp_path` with HOME, AELFRICE_DB and
AELFRICE_USER_DB pinned there (`tests/fire_census_planted.py`), makes the
feature fire through aelfrice's own code, and checks the probe. Global-log
probes read the planted `~/.aelfrice` as the `global` context.

`maybe_migrate_to_uv` runs `uv tool install`; every test that reaches it
puts a stub `uv` that exits 0 first on PATH, so no real install runs.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_cli_setup_hosts_export as pm
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PROBE = {p.item_id: p for p in pm.PROBES}
EXPECTED_IDS = {f"A{i:03d}" for i in range(40, 77)}
GLOBAL_LOG = frozenset(p.item_id for p in pm.PROBES if p.kind == "global-log")


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    store = plant(tmp_path, monkeypatch)
    yield store
    store.close()


def global_ctx(ps: PlantedStore) -> cc.Ctx:
    g = ps.global_dir
    return cc.Ctx("global", g, g, True, g / "memory.db", "global")


def run(ps: PlantedStore, item_id: str, ctx: cc.Ctx | None = None) -> cc.Result:
    """Run one probe: global-log probes on the global context, others on the store."""
    ps.close()
    if ctx is None:
        ctx = global_ctx(ps) if item_id in GLOBAL_LOG else ps.ctx()
    ctx.open()
    try:
        return cc.run_one(PROBE[item_id], ctx)
    finally:
        ctx.close()


def stub_uv(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a `uv` that exits 0 first on PATH."""
    fakebin = root / "fakebin"
    fakebin.mkdir(parents=True, exist_ok=True)
    uv = fakebin / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    import os
    monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ.get("PATH", ""))


def sentinel(ps: PlantedStore, name: str) -> Path:
    return ps.global_dir / name


# --- shape -----------------------------------------------------------------


def test_probe_ids_match_the_subsystem() -> None:
    assert {p.item_id for p in pm.PROBES} == EXPECTED_IDS
    assert len(pm.PROBES) == len(EXPECTED_IDS)
    assert pm.TRACEABLE == {"A043", "A049", "A064", "A065", "A068", "A071", "A075"}
    assert pm.TRACEABLE | pm.UNTRACEABLE == EXPECTED_IDS


@pytest.mark.parametrize("item_id", sorted(pm.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(item_id: str) -> None:
    r = cc.run_one(PROBE[item_id], None)
    assert r.status == "untraceable" and r.count is None and r.note


@pytest.mark.parametrize("item_id", sorted(pm.TRACEABLE))
def test_empty_store_is_silent(ps: PlantedStore, item_id: str) -> None:
    ps.remove_precreated_sentinels()
    r = run(ps, item_id)
    assert r.status == "silent" and r.count == 0, r


# --- A043 curation ---------------------------------------------------------


def test_a043_curation_sources(ps: PlantedStore) -> None:
    b1 = ps.belief("The census reads byte copies only.")
    b2 = ps.belief("Scratch dirs get pruned per result.")
    b3 = ps.belief("Locked rule one for the census.", locked=True)
    b4 = ps.belief("Locked rule two for the census.", locked=True)
    rc = [
        ps.cli("retire", b1), ps.cli("restore", b1),
        ps.cli("delete", b2, "--yes"),
        ps.cli("retire", b3, "--force"),
        ps.cli("delete", b4, "--yes", "--force"),
        ps.cli("retire", b4),  # unknown id now: must write nothing
    ]
    assert rc == [0, 0, 0, 0, 0, 1]
    r = run(ps, "A043")
    assert r.status == "fired" and r.count == 5
    assert r.detail == {
        "user_retired": 1, "user_restored": 1, "user_deleted": 1,
        "user_retired_force": 1, "user_deleted_force": 1,
    }


# --- A049 lock ingest rows -------------------------------------------------


def test_a049_lock_ingest_rows(ps: PlantedStore) -> None:
    rc = [
        ps.lock("Never push from HOME."),
        ps.lock("Use uv for every Python env.", "--reference"),
        ps.lock("Freeze the release branch this week.", "--for", "2w"),
    ]
    assert rc == [0, 0, 0]
    r = run(ps, "A049")
    assert r.status == "fired" and r.count == 3
    assert r.detail == {
        "ingest_rows": 3, "derived_beliefs": 3, "still_locked": 3,
        "tier_frozen": 2, "tier_reference": 1, "with_expiry": 1,
    }


# --- A064 uv migration sentinel --------------------------------------------


def test_a064_uv_migration_sentinel(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import lifecycle
    stub_uv(ps.root, monkeypatch)
    sentinel(ps, "migrated-to-uv").unlink()
    assert lifecycle.MIGRATED_TO_UV_SENTINEL == sentinel(ps, "migrated-to-uv")
    assert run(ps, "A064").status == "silent"
    res = lifecycle.maybe_migrate_to_uv()
    assert (res.attempted, res.succeeded) == (True, True)
    r = run(ps, "A064")
    assert r.status == "fired" and r.count == 1
    assert not lifecycle.maybe_migrate_to_uv().attempted


# --- A065 auto-install lock ------------------------------------------------


def test_a065_auto_install_lock(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import auto_install, cli
    assert run(ps, "A065").status == "silent"
    # The gate admits only a `uv tool` install; the test venv is not one.
    monkeypatch.setattr(auto_install, "is_running_from_uv_tool_install", lambda: True)
    auto_install.auto_install_at_cli_entry(installed_version=cli._AELFRICE_VERSION)  # pyright: ignore[reportPrivateUsage]
    r = run(ps, "A065")
    assert r.status == "fired" and r.count == 1
    assert r.detail == {"lock_file": 1, "stamp_file": 1}


def test_a065_gate_closed_outside_uv_tool(ps: PlantedStore) -> None:
    from aelfrice import auto_install, cli
    auto_install.auto_install_at_cli_entry(installed_version=cli._AELFRICE_VERSION)  # pyright: ignore[reportPrivateUsage]
    r = run(ps, "A065")
    assert r.status == "silent"
    assert r.detail == {"lock_file": 0, "stamp_file": 0}


# --- A068 project-warm sentinel --------------------------------------------


def test_a068_project_warm_sentinel(ps: PlantedStore) -> None:
    from aelfrice import project_warm
    from aelfrice.store import MemoryStore
    work = (ps.root / "workspace").resolve()
    work.mkdir(parents=True)
    # tmp_path sits under /tmp on Linux, which the default deny list refuses.
    ps.global_dir.mkdir(parents=True, exist_ok=True)
    (ps.global_dir / "config.json").write_text(
        json.dumps({"project_warm": {"deny_globs": []}}), encoding="utf-8")
    pid = project_warm._project_id(work)  # pyright: ignore[reportPrivateUsage]
    legacy_db = ps.global_dir / "projects" / pid / "memory.db"
    legacy_db.parent.mkdir(parents=True)
    MemoryStore(str(legacy_db)).close()
    lctx = cc.Ctx("legacy01", legacy_db.parent, ps.global_dir, False, legacy_db, "legacy")
    assert run(ps, "A068", lctx).status == "silent"
    assert ps.cli("project-warm", str(work), "--debounce", "0") == 0
    lctx = cc.Ctx("legacy01", legacy_db.parent, ps.global_dir, False, legacy_db, "legacy")
    r = run(ps, "A068", lctx)
    assert r.status == "fired" and r.count == 1


# --- A071 opt-out sync -----------------------------------------------------


def test_a071_opt_out_sync(ps: PlantedStore) -> None:
    from aelfrice import auto_install, cli
    cli._sync_setup_opt_outs_and_stamp(  # pyright: ignore[reportPrivateUsage]
        argparse.Namespace(commit_ingest=False, search_tool=False))
    auto_install.add_host_opt_out("claude")
    r = run(ps, "A071")
    assert r.status == "fired" and r.count == 2
    assert r.detail == {"hook:commit_ingest": 1, "hook:search_tool": 1, "host:claude": 1}
    # Re-running setup without --no-X rescinds the per-hook opt-outs.
    cli._sync_setup_opt_outs_and_stamp(argparse.Namespace())  # pyright: ignore[reportPrivateUsage]
    r = run(ps, "A071")
    assert r.status == "silent" and r.count == 0
    assert r.detail == {"host:claude": 1}


# --- A075 spine backfill sentinel ------------------------------------------


def test_a075_spine_backfill_sentinel(ps: PlantedStore) -> None:
    from aelfrice import temporal_spine
    sentinel(ps, "spine-backfilled").unlink()
    assert run(ps, "A075").status == "silent"
    for t in ("First step of the session.", "Second step of the session.",
              "Third step of the session."):
        ps.belief(t, session_id="s1")
    res = temporal_spine.maybe_backfill_temporal_spine(ps.open())
    assert (res.ran, res.n_edges) == (True, 2)
    r = run(ps, "A075")
    assert r.status == "fired" and r.count == 1
    c = ps.ctx()
    try:
        assert c.count("SELECT COUNT(*) FROM edges WHERE type='TEMPORAL_NEXT'") == 2
    finally:
        c.close()


# --- aelf setup end to end -------------------------------------------------


def test_setup_fires_the_setup_traces(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_uv(ps.root, monkeypatch)
    sentinel(ps, "migrated-to-uv").unlink()
    sentinel(ps, "spine-backfilled").unlink()
    ps.belief("First step of the session.", session_id="s1")
    ps.belief("Second step of the session.", session_id="s1")
    assert ps.cli("setup", "--scope", "user", "--no-commit-ingest") == 0
    assert run(ps, "A064").status == "fired"
    assert run(ps, "A075").status == "fired"
    a071 = run(ps, "A071")
    assert a071.status == "fired" and a071.detail == {"hook:commit_ingest": 1}
    # setup writes the stamp but never takes the auto-install lock
    a065 = run(ps, "A065")
    assert a065.status == "silent"
    assert a065.detail == {"lock_file": 0, "stamp_file": 1}
    settings = ps.home / ".claude" / "settings.json"
    assert settings.is_file()
