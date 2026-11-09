"""#1887: the fire-census runner, its copy helper and its output rules.

Every store here is planted under `tmp_path` with aelfrice's own code, or
built with raw sqlite3 for the refusal cases. Nothing reads a real store.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from benchmarks.real_store_fire_census import __main__ as runner
from benchmarks.real_store_fire_census import copy_store, core, registry
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PLANTED_TEXT = "We decided to keep every census copy outside the repository."


def _snapshot(root: Path) -> dict[str, tuple[int, str, int]]:
    """relpath -> (size, sha256, mtime_ns) for every file under root."""
    out: dict[str, tuple[int, str, int]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            out[str(p.relative_to(root))] = (st.st_size, digest, st.st_mtime_ns)
    return out


def _wal_db(folder: Path) -> sqlite3.Connection:
    """A WAL-mode db with a committed row still in its -wal (writer left open)."""
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(folder / "memory.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO beliefs VALUES ('b1', 'x')")
    conn.commit()
    return conn


@pytest.fixture
def planted_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A planted store with sidecars, copied with copy_store; returns the copy."""
    ps: PlantedStore = plant(tmp_path / "plant", monkeypatch)
    assert ps.ingest(PLANTED_TEXT, session_id="s1") >= 1
    assert ps.lock("Never run the census on a live store.") == 0
    ps.hook("main", {"hook_event_name": "UserPromptSubmit", "session_id": "s1",
                     "prompt": "where do census copies live", "cwd": str(ps.repo)})
    ps.close()
    dest = tmp_path / "copies" / "kq-planted"
    assert copy_store.main([str(ps.store_dir), str(dest)]) == 0
    return dest


# --- registry and result rules -----------------------------------------------


def test_registry_holds_one_probe_per_inventory_id() -> None:
    ids = [p.item_id for p in registry.all_probes()]
    want = [f"A{i:03d}" for i in range(registry.FIRST_ID, registry.LAST_ID + 1)]
    assert ids == want


def test_every_probe_has_a_kind_and_a_name() -> None:
    for p in registry.all_probes():
        assert p.kind in core.KINDS
        assert p.name.strip()
        assert registry.subsystem_of(p.item_id) in registry.SUBSYSTEM_MODULES


@pytest.mark.parametrize("make", [
    lambda: core.Result("fired", 1, {}, "see a/b"),
    lambda: core.Result("fired", 1, {}, "see a\\b"),
    lambda: core.Result("fired", 1, {"Text With Caps": 1}),
    lambda: core.Result("fired", 1, {"ok": "three"}),  # pyright: ignore[reportArgumentType]
    lambda: core.Result("fired", 1.5),  # pyright: ignore[reportArgumentType]
    lambda: core.Result("maybe", 1),  # pyright: ignore[reportArgumentType]
])
def test_a_result_cannot_carry_text_or_paths(make: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        make()  # pyright: ignore[reportCallIssue]


def test_outcome_folds_store_statuses() -> None:
    assert core.outcome(["silent", "fired"]) == "fired"
    assert core.outcome(["untraceable", "silent"]) == "silent"
    assert core.outcome(["untraceable", "error"]) == "error"
    assert core.outcome(["untraceable"]) == "untraceable"


# --- the runner reads copies and changes nothing -------------------------------


def test_dry_run_opens_no_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "copy"
    folder.mkdir()
    (folder / "memory.db").write_bytes(b"not a sqlite file")
    before = _snapshot(folder)
    assert runner.main([str(folder), "--dry-run", "--only", "A012,A175"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["stores"] == {"store01": "repo"}
    assert [p[0] for p in doc["probes"]] == ["A012", "A175"]
    assert _snapshot(folder) == before


def test_a_run_leaves_every_byte_of_the_copy_unchanged(
        planted_copy: Path, tmp_path: Path) -> None:
    before = _snapshot(planted_copy)
    assert any(name.endswith(".jsonl") for name in before)
    out = tmp_path / "out" / "census.json"
    labels = tmp_path / "private" / "labels.json"
    assert runner.main([str(planted_copy), "--out", str(out), "--labels-out", str(labels)]) == 0
    assert _snapshot(planted_copy) == before
    doc = json.loads(out.read_text("utf-8"))
    assert doc["stores"] == {"store01": "repo"}
    assert doc["summary"]["probes"] == registry.LAST_ID
    assert set(doc["probes"]) == {p.item_id for p in registry.all_probes()}
    assert json.loads(labels.read_text("utf-8")) == {"store01": str(planted_copy / "memory.db")}
    if os.name == "posix":
        assert labels.stat().st_mode & 0o777 == 0o600


def test_the_output_names_no_path_and_no_store_text(
        planted_copy: Path, tmp_path: Path) -> None:
    out = tmp_path / "census.json"
    assert runner.main([str(planted_copy), "--out", str(out)]) == 0
    text = out.read_text("utf-8")
    assert str(tmp_path) not in text
    assert "kq-planted" not in text
    assert "census copy outside" not in text
    assert "/" not in "".join(
        r.get("note", "") for p in json.loads(text)["probes"].values()
        for r in p["stores"].values())


def test_a_planted_store_fires_and_an_empty_one_does_not(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = PlantedStore(tmp_path / "empty")
    monkeypatch.setenv("HOME", str(empty.home))
    monkeypatch.setenv("AELFRICE_DB", str(empty.db_file))
    monkeypatch.setenv("AELFRICE_USER_DB", str(empty.user_db))
    empty.open()
    empty.close()
    empty_copy = tmp_path / "copies" / "zq-empty"
    assert copy_store.main([str(empty.store_dir), str(empty_copy)]) == 0
    out = tmp_path / "census.json"
    labels = tmp_path / "labels.json"
    # Arguments in reverse order: labels follow the sorted paths, not argv.
    assert runner.main([str(empty_copy), str(planted_copy), "--out", str(out),
                        "--labels-out", str(labels)]) == 0
    assert json.loads(labels.read_text("utf-8")) == {
        "store01": str(planted_copy / "memory.db"), "store02": str(empty_copy / "memory.db")}
    doc = json.loads(out.read_text("utf-8"))
    fired_planted = [i for i, p in doc["probes"].items()
                     if p["stores"].get("store01", {}).get("status") == "fired"]
    fired_empty = [i for i, p in doc["probes"].items()
                   if p["stores"].get("store02", {}).get("status") == "fired"]
    assert fired_planted
    # A309 counts WAL journal mode, which every store aelfrice creates has.
    assert fired_empty == ["A309"]
    assert "A309" in fired_planted
    errors = {r.get("note") for p in doc["probes"].values() for r in p["stores"].values()
              if r["status"] == "error"}
    # With no --global-dir, global-log probes have no context to run on.
    assert errors <= {"NoGlobalContext", "LookupError"}


# --- refusals --------------------------------------------------------------------


def test_a_path_inside_a_git_folder_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "repo" / ".git" / "aelfrice"
    folder.mkdir(parents=True)
    sqlite3.connect(str(folder / "memory.db")).close()
    assert runner.main([str(folder), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "live store" in err
    assert str(tmp_path) not in err


def test_a_path_under_the_home_dotdir_is_refused(tmp_path: Path) -> None:
    folder = Path.home() / ".aelfrice" / "census-refusal-test"
    assert runner.is_live_location(folder)
    assert runner.is_live_location(Path.home() / ".aelfrice")
    assert not runner.is_live_location(tmp_path / "copy")


def test_a_store_with_wal_frames_is_refused_and_copy_store_folds_them(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "live-like"
    writer = _wal_db(src)
    try:
        assert core.has_unfolded_wal(src / "memory.db")
        assert runner.main([str(src), "--dry-run"]) == 2
        assert "-wal" in capsys.readouterr().err
        with pytest.raises(core.UnfoldedWalError):
            core.open_ro(src / "memory.db")
        dest = tmp_path / "copy"
        assert copy_store.main([str(src), str(dest)]) == 0
    finally:
        writer.close()
    assert not core.has_unfolded_wal(dest / "memory.db")
    conn = core.open_ro(dest / "memory.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM beliefs").fetchone() == (1,)
    finally:
        conn.close()


def test_output_files_inside_a_store_or_the_repo_are_refused(planted_copy: Path) -> None:
    assert runner.main([str(planted_copy), "--out", str(planted_copy / "x.json")]) == 2
    assert runner.main([str(planted_copy), "--labels-out",
                        str(planted_copy / "labels.json")]) == 2
    assert runner.main([str(planted_copy), "--dry-run", "--labels-out",
                        str(runner.REPO_ROOT / "benchmarks" / "census-labels-refusal-test.json")]) == 2
    assert not (runner.REPO_ROOT / "benchmarks" / "census-labels-refusal-test.json").exists()


def test_only_rejects_an_unknown_id(planted_copy: Path) -> None:
    assert runner.main([str(planted_copy), "--dry-run", "--only", "A999"]) == 2


def test_copy_store_refuses_a_destination_in_the_repo(planted_copy: Path) -> None:
    dest = runner.REPO_ROOT / "benchmarks" / "census-copy-refusal-test"
    assert copy_store.main([str(planted_copy), str(dest)]) == 2
    assert not dest.exists()


# --- layouts ---------------------------------------------------------------------


def test_global_dir_adds_global_and_user_and_global_log_probes_run_there(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gdir = tmp_path / "global-copy"
    (gdir / "user").mkdir(parents=True)
    sqlite3.connect(str(gdir / "memory.db")).close()
    sqlite3.connect(str(gdir / "user" / "memory.db")).close()
    legacy = tmp_path / "legacy-copy"
    legacy.mkdir()
    sqlite3.connect(str(legacy / "memory.db")).close()
    assert runner.main([str(planted_copy), "--legacy", str(legacy),
                        "--global-dir", str(gdir), "--dry-run"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["stores"] == {"store01": "repo", "legacy01": "legacy",
                             "global": "global", "user": "user"}
    global_ids = [p.item_id for p in registry.all_probes() if p.kind == "global-log"]
    assert global_ids
    ctxs, _private = runner.build_contexts([str(planted_copy)], [], [], str(gdir))
    out = runner.run_census([p for p in registry.all_probes() if p.item_id == global_ids[0]],
                            ctxs)
    assert list(out["probes"][global_ids[0]]["stores"]) == ["global"]


def test_the_runner_module_runs_as_a_script(planted_copy: Path) -> None:
    import subprocess
    cp = subprocess.run(
        [sys.executable, "-m", "benchmarks.real_store_fire_census", str(planted_copy),
         "--dry-run", "--only", "A001"],
        cwd=str(runner.REPO_ROOT), capture_output=True, text=True, timeout=120, check=False)
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout)["probes"] == [["A001", "untraceable", "beliefs-graph-feedback"]]
