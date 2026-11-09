"""#1887: the fire-census runner, its copy token, its guard and its copy helper.

Every store here is planted under `tmp_path` with aelfrice's own code, or
built with raw sqlite3 for the refusal cases. Nothing reads a real store.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from benchmarks.real_store_fire_census import __main__ as runner
from benchmarks.real_store_fire_census import copy_store, copy_token, core, guard, registry
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PLANTED_TEXT = "We decided to keep every census copy outside the repository."
DOT_GIT = "." + "git"
FIRMLINK = Path("/System/Volumes/Data")
HAS_FIRMLINK = sys.platform == "darwin" and FIRMLINK.is_dir()
PROBE_TEST_MODULES = sorted(
    str(p.relative_to(runner.REPO_ROOT))
    for p in (runner.REPO_ROOT / "tests").glob("test_real_store_fire_census_*.py")
    if p.name != Path(__file__).name)


def _snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    """relpath -> (kind, size or link, sha256, mtime_ns) for everything under root."""
    out: dict[str, tuple[Any, ...]] = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = ("L", os.readlink(p), st.st_mtime_ns)
        elif p.is_file():
            out[rel] = ("F", st.st_size, hashlib.sha256(p.read_bytes()).hexdigest(),
                        st.st_mtime_ns)
        else:
            out[rel] = ("D", st.st_mtime_ns)
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


def _plain_db(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(folder / "memory.db"))
    conn.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    return folder / "memory.db"


def _gated(copy: Path, *rel: str) -> copy_token.GatedPath:
    """A path in a copy whose reads go through the copy's manifest."""
    gate = copy_token.Gate()
    gate.add(copy_token.load(copy))
    return copy_token.GatedPath(copy.joinpath(*rel), gate=gate)


def _copy(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert copy_store.main([str(src), str(dest)]) == 0
    return dest


@pytest.fixture
def planted_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A planted store with sidecars, copied with copy_store; returns the copy."""
    ps: PlantedStore = plant(tmp_path / "plant", monkeypatch)
    assert ps.ingest(PLANTED_TEXT, session_id="s1") >= 1
    assert ps.lock("Never run the census on a live store.") == 0
    ps.hook("main", {"hook_event_name": "UserPromptSubmit", "session_id": "s1",
                     "prompt": "where do census copies live", "cwd": str(ps.repo)})
    ps.close()
    return _copy(ps.store_dir, tmp_path / "copies" / "kq-planted")


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
    lambda: core.Result("fired", 0),
    lambda: core.Result("ambiguous", None),
    lambda: core.Result("silent", 2),
    lambda: core.Result("untraceable", 1),
])
def test_a_result_cannot_carry_text_or_paths(make: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        make()


def test_outcome_folds_store_statuses() -> None:
    assert core.outcome(["silent", "ambiguous", "fired"]) == "fired"
    assert core.outcome(["silent", "ambiguous"]) == "ambiguous"
    assert core.outcome(["untraceable", "silent"]) == "silent"
    assert core.outcome(["untraceable", "error"]) == "error"
    assert core.outcome(["untraceable"]) == "untraceable"


def test_a_shared_trace_is_ambiguous_and_its_absence_silent() -> None:
    assert core.from_shared_count(2, note="x").status == "ambiguous"
    assert core.from_shared_count(0).status == "silent"


# --- the runner reads only token copies, and changes nothing -----------------------


def test_dry_run_opens_no_db(planted_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _snapshot(planted_copy)
    assert runner.main([str(planted_copy), "--dry-run", "--only", "A012,A175"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["stores"] == {"store01": "repo"}
    assert [p[0] for p in doc["probes"]] == ["A012", "A175"]
    assert _snapshot(planted_copy) == before


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
    assert json.loads(labels.read_text("utf-8")) == {
        "store01": os.path.abspath(planted_copy / "memory.db")}
    if os.name == "posix":
        assert labels.stat().st_mode & 0o777 == 0o600
        assert out.stat().st_mode & 0o777 == 0o600


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
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pin" / "memory.db"))
    empty_copy = _copy(empty.store_dir, tmp_path / "copies" / "zq-empty")
    out = tmp_path / "census.json"
    labels = tmp_path / "labels.json"
    # Arguments in reverse order: labels follow the sorted paths, not argv.
    assert runner.main([str(empty_copy), str(planted_copy), "--out", str(out),
                        "--labels-out", str(labels)]) == 0
    assert json.loads(labels.read_text("utf-8")) == {
        "store01": os.path.abspath(planted_copy / "memory.db"),
        "store02": os.path.abspath(empty_copy / "memory.db")}
    doc = json.loads(out.read_text("utf-8"))

    def positive(label: str) -> list[str]:
        return [i for i, p in doc["probes"].items()
                if p["stores"].get(label, {}).get("status") in ("fired", "ambiguous")]
    assert positive("store01")
    # A309 counts WAL journal mode, which every store aelfrice creates has.
    assert positive("store02") == ["A309"]
    errors = {r.get("note") for p in doc["probes"].values() for r in p["stores"].values()
              if r["status"] == "error"}
    # With no --global-dir, global-log probes have no context to run on.
    assert errors <= {"NoGlobalContext", "LookupError"}


# --- the copy token is the guarantee ----------------------------------------------


def test_a_folder_without_a_token_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _plain_db(tmp_path / "hand-copy").parent
    assert runner.main([str(folder), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "copy token" in err and str(tmp_path) not in err


def test_a_token_moved_beside_another_db_is_refused(planted_copy: Path, tmp_path: Path) -> None:
    other = _plain_db(tmp_path / "other").parent
    shutil.copyfile(planted_copy / copy_token.TOKEN_FILENAME,
                    other / copy_token.TOKEN_FILENAME)
    assert runner.main([str(other), "--dry-run"]) == 2


def test_a_db_changed_after_the_copy_is_refused(planted_copy: Path) -> None:
    conn = sqlite3.connect(str(planted_copy / "memory.db"))
    conn.execute("CREATE TABLE extra (x)")
    conn.commit()
    conn.close()
    assert runner.main([str(planted_copy), "--dry-run"]) == 2


def test_a_hard_link_to_the_copy_db_is_refused(planted_copy: Path, tmp_path: Path) -> None:
    os.link(planted_copy / "memory.db", tmp_path / "second-name.db")
    assert runner.main([str(planted_copy), "--dry-run"]) == 2


def test_a_live_db_hard_linked_into_a_token_copy_is_refused(
        planted_copy: Path, tmp_path: Path) -> None:
    live = _plain_db(tmp_path / "repo" / DOT_GIT / "aelfrice")
    (planted_copy / "memory.db").unlink()
    os.link(live, planted_copy / "memory.db")
    assert runner.main([str(planted_copy), "--dry-run"]) == 2


def test_a_db_swapped_after_the_check_is_never_read(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = _plain_db(tmp_path / "repo" / DOT_GIT / "aelfrice")
    before = _snapshot(live.parent)
    real_run = runner.run_census

    def swap(*a: Any, **k: Any) -> dict[str, Any]:
        (planted_copy / "memory.db").unlink()
        (planted_copy / "memory.db").symlink_to(live)
        return real_run(*a, **k)

    monkeypatch.setattr(runner, "run_census", swap)
    out = tmp_path / "o.json"
    assert runner.main([str(planted_copy), "--only", "A012,A309", "--out", str(out)]) == 2
    assert not out.exists()
    assert _snapshot(live.parent) == before


def test_a_db_replaced_by_another_file_after_the_check_is_refused(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = _plain_db(tmp_path / "other")
    real_run = runner.run_census

    def swap(*a: Any, **k: Any) -> dict[str, Any]:
        os.replace(other, planted_copy / "memory.db")
        return real_run(*a, **k)

    monkeypatch.setattr(runner, "run_census", swap)
    assert runner.main([str(planted_copy), "--only", "A309"]) == 2


def test_a_db_removed_after_the_check_is_refused(
        planted_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = runner.run_census

    def remove(*a: Any, **k: Any) -> dict[str, Any]:
        (planted_copy / "memory.db").unlink()
        return real_run(*a, **k)

    monkeypatch.setattr(runner, "run_census", remove)
    assert runner.main([str(planted_copy), "--only", "A309"]) == 2


def test_open_ro_refuses_an_identity_the_token_did_not_record(planted_copy: Path) -> None:
    st = os.stat(planted_copy / "memory.db")
    with pytest.raises(core.IdentityMismatchError):
        core.open_ro(planted_copy / "memory.db", (st.st_dev, st.st_ino + 1))
    core.open_ro(planted_copy / "memory.db", (st.st_dev, st.st_ino)).close()


def test_the_gate_reads_only_files_its_manifest_lists(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    (src / "hook_audit.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (src / "telemetry").mkdir()
    (src / "telemetry" / "t.jsonl").write_text("{}\n", encoding="utf-8")
    dest = _copy(src, tmp_path / "copy")
    db = _gated(dest, "memory.db")
    root = db.parent
    assert db.gate.identity(db) == guard.ident(dest / "memory.db")
    assert (root / "hook_audit.jsonl").read_text("utf-8") == '{"a": 1}\n'
    assert [p.name for p in (root / "telemetry").glob("*.jsonl")] == ["t.jsonl"]
    assert not (root / "hook_audit.jsonl.1").is_file()
    assert (root / "telemetry").is_dir() and not (root / "nothing").exists()
    with pytest.raises(PermissionError):
        (root / "hook_audit.jsonl").open("a")
    os.link(dest / "hook_audit.jsonl", tmp_path / "second-name")
    with pytest.raises(copy_token.ManifestMismatch):
        (root / "hook_audit.jsonl").is_file()
    os.unlink(tmp_path / "second-name")
    st = os.stat(dest / "hook_audit.jsonl")
    os.utime(dest / "hook_audit.jsonl", ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    with pytest.raises(copy_token.ManifestMismatch):
        (root / "hook_audit.jsonl").read_text("utf-8")
    os.utime(dest / "hook_audit.jsonl", ns=(st.st_atime_ns, st.st_mtime_ns))
    (dest / "hook_audit.jsonl.1").write_text("{}\n", encoding="utf-8")
    with pytest.raises(copy_token.ManifestMismatch):
        (root / "hook_audit.jsonl.1").is_file()
    with pytest.raises(copy_token.ManifestMismatch):
        list(root.iterdir())
    os.unlink(dest / "hook_audit.jsonl.1")
    (dest / "memory.db-shm").symlink_to(tmp_path / "anything")
    with pytest.raises(copy_token.ManifestMismatch):
        db.gate.identity(db)
    os.unlink(dest / "memory.db-shm")
    os.unlink(dest / "telemetry" / "t.jsonl")
    with pytest.raises(copy_token.ManifestMismatch):
        list((root / "telemetry").glob("*.jsonl"))
    _plain_db(tmp_path / "outside")
    with pytest.raises(copy_token.ManifestMismatch):
        (root.parent / "outside" / "memory.db").is_file()
    (dest / "extra-folder").mkdir()
    with pytest.raises(copy_token.ManifestMismatch):
        (root / "extra-folder").is_dir()
    os.rmdir(dest / "extra-folder")
    (dest / "memory.db-shm").symlink_to(tmp_path / "anything")
    with pytest.raises(copy_token.ManifestMismatch):
        core.open_ro(db)
    os.unlink(dest / "memory.db-shm")
    assert not (root.parent / "not-there").is_file()


def test_the_gate_opens_no_symlink_that_appears_after_its_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "log.jsonl").write_text("{}\n", encoding="utf-8")
    dest = _copy(src, tmp_path / "copy")
    p = _gated(dest, "log.jsonl")
    real = p.gate.file_stat

    def swap(q: Any) -> os.stat_result | None:
        st = real(q)
        os.rename(dest / "log.jsonl", tmp_path / "moved")
        os.symlink(tmp_path / "moved", dest / "log.jsonl")
        return st

    monkeypatch.setattr(p.gate, "file_stat", swap)
    with pytest.raises(OSError):
        p.read_text("utf-8")


def test_the_gate_refuses_a_file_replaced_after_its_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "log.jsonl").write_text("{}\n", encoding="utf-8")
    dest = _copy(src, tmp_path / "copy")
    (tmp_path / "other").write_text('{"other": 1}\n', encoding="utf-8")
    p = _gated(dest, "log.jsonl")
    real = p.gate.file_stat

    def swap(q: Any) -> os.stat_result | None:
        st = real(q)
        os.replace(tmp_path / "other", dest / "log.jsonl")
        return st

    monkeypatch.setattr(p.gate, "file_stat", swap)
    with pytest.raises(copy_token.ManifestMismatch):
        p.read_text("utf-8")


def test_the_gate_picks_the_innermost_manifest_and_matches_whole_names(tmp_path: Path) -> None:
    stamp = {"dev": 1, "ino": 2, "nlink": 1, "size": 3, "mtime_ns": 4}
    outer = copy_token.Manifest(tmp_path / "c1", frozenset({"inner"}), {})
    inner = copy_token.Manifest(tmp_path / "c1" / "inner", frozenset(), {"f": stamp})
    gate = copy_token.Gate()
    gate.add(outer)
    gate.add(inner)
    lookup = gate._lookup  # pyright: ignore[reportPrivateUsage]
    assert lookup(tmp_path / "c1" / "inner" / "f") == (inner, "f")
    assert lookup(tmp_path / "c1" / "inner") == (inner, "")
    assert lookup(tmp_path / "c1") == (outer, "")
    gate2 = copy_token.Gate()
    gate2.add(inner)
    gate2.add(outer)
    assert gate2._lookup(tmp_path / "c1" / "inner" / "f") == (inner, "f")  # pyright: ignore[reportPrivateUsage]
    assert lookup(tmp_path / "c10" / "f") == (None, "")
    assert lookup(tmp_path / "c1x") == (None, "")


def test_a_sidecar_link_into_a_live_folder_is_named_as_live(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    live = tmp_path / "repo" / DOT_GIT / "aelfrice"
    live.mkdir(parents=True)
    (live / "memory.db-shm").write_bytes(b"")
    (planted_copy / "memory.db-shm").symlink_to(live / "memory.db-shm")
    assert runner.main([str(planted_copy), "--dry-run"]) == 2
    assert "live store location" in capsys.readouterr().err


def test_a_sidecar_swapped_for_a_symlink_into_a_live_store_exits_2(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    live = tmp_path / "repo" / DOT_GIT / "aelfrice"
    live.mkdir(parents=True)
    (live / "hook_audit.jsonl").write_text('{"hook": "user_prompt_submit"}\n', encoding="utf-8")
    before = _snapshot(live)
    log = planted_copy / "hook_audit.jsonl"
    assert log.is_file()
    log.unlink()
    log.symlink_to(live / "hook_audit.jsonl")
    out = tmp_path / "o.json"
    assert runner.main([str(planted_copy), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "copy token" in err and str(tmp_path) not in err
    assert not out.exists() and _snapshot(live) == before


def test_a_file_planted_in_a_copy_exits_2(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (planted_copy / "hook_audit.jsonl.1").write_text("{}\n", encoding="utf-8")
    assert runner.main([str(planted_copy), "--out", str(tmp_path / "o.json")]) == 2
    assert "doesn't list" in capsys.readouterr().err


def test_a_copy_of_a_copy_is_refused(planted_copy: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    shutil.copytree(planted_copy, again)
    assert runner.main([str(again), "--dry-run"]) == 2
    with pytest.raises(copy_token.TokenError, match="another folder"):
        copy_token.load(again)


def test_gated_and_plain_reads_give_the_same_results(planted_copy: Path) -> None:
    probes = registry.all_probes()
    root = _gated(planted_copy)
    gated = core.Ctx("store01", root, None, False, root / "memory.db")
    plain = core.Ctx("store01", planted_copy, None, False, planted_copy / "memory.db")
    a = runner.run_census(probes, [gated])
    b = runner.run_census(probes, [plain])
    assert a == b


def test_a_malformed_token_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    dest = _copy(src, tmp_path / "copy")
    tok = dest / copy_token.TOKEN_FILENAME
    good: dict[str, Any] = json.loads(tok.read_text("utf-8"))
    entry: dict[str, int] = good["files"]["memory.db"]

    def with_entry(**change: Any) -> dict[str, Any]:
        doc = json.loads(json.dumps(good))
        doc["files"]["memory.db"] = {**entry, **change}
        return doc

    bad: list[dict[str, Any]] = [
        {**good, "token": "not-hex"}, {**good, "version": 1}, {**good, "files": []},
        {**good, "root": {"dev": 1}}, {**good, "dirs": ["../up"]},
        {**good, "files": {"../memory.db": entry}},
        with_entry(ino=str(entry["ino"])), with_entry(size=True), with_entry(nlink=1.0),
        {**good, "files": {"memory.db": {k: v for k, v in entry.items() if k != "size"}}},
    ]
    os.chmod(tok, 0o600)
    for doc in bad:
        tok.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(copy_token.TokenError, match="malformed"):
            copy_token.load(dest)
    tok.write_text(json.dumps(good), encoding="utf-8")
    assert copy_token.load(dest).files["memory.db"] == entry


def test_write_token_refuses_what_the_helper_did_not_copy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "tmp"
    db = _plain_db(root)
    ident = guard.ident(root)
    assert ident is not None
    write = copy_token._write_token  # pyright: ignore[reportPrivateUsage]
    (root / "extra").write_text("x", encoding="utf-8")
    with pytest.raises(copy_token.TokenError, match="didn't copy"):
        write(root, {"memory.db"}, ident)
    os.unlink(root / "extra")
    os.link(db, tmp_path / "other-name.db")
    with pytest.raises(copy_token.TokenError, match="single-link"):
        write(root, {"memory.db"}, ident)
    os.unlink(tmp_path / "other-name.db")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(copy_token.TokenError, match="symlink"):
        write(root, {"memory.db"}, ident)
    os.unlink(root / "link")
    writer = _wal_db(tmp_path / "walled")
    try:
        with pytest.raises(copy_token.TokenError, match="non-empty -wal"):
            write(tmp_path / "walled", {"memory.db", "memory.db-wal", "memory.db-shm"},
                  guard.ident(tmp_path / "walled") or (0, 0))
    finally:
        writer.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    with pytest.raises(copy_token.TokenError, match="live"):
        write(root, {"memory.db"}, ident)
    assert not (root / copy_token.TOKEN_FILENAME).exists()
    assert not hasattr(copy_token, "write_token")


def test_open_ro_refuses_a_symlink_a_folder_and_an_unfolded_wal(tmp_path: Path) -> None:
    real = _plain_db(tmp_path / "real")
    (tmp_path / "link.db").symlink_to(real)
    with pytest.raises(OSError):
        core.open_ro(tmp_path / "link.db")
    with pytest.raises(core.IdentityMismatchError):
        core.open_ro(tmp_path / "real")
    writer = _wal_db(tmp_path / "walled")
    try:
        with pytest.raises(core.UnfoldedWalError):
            core.open_ro(tmp_path / "walled" / "memory.db")
    finally:
        writer.close()


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="verified on macOS only; sqlite on Linux /dev/fd is unverified")
def test_open_ro_reads_the_file_it_checked_even_if_the_path_is_swapped(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked = _plain_db(tmp_path / "a")
    other = tmp_path / "b" / "memory.db"
    other.parent.mkdir()
    c = sqlite3.connect(str(other))
    c.execute("CREATE TABLE swapped_in (x)")
    c.commit()
    c.close()
    real_connect = sqlite3.connect

    def connect(*a: Any, **k: Any) -> sqlite3.Connection:
        os.replace(other, checked)
        return real_connect(*a, **k)

    monkeypatch.setattr(core.sqlite3, "connect", connect)
    conn = core.open_ro(checked, guard.ident(checked))
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()
    assert names == {"beliefs"}


def test_a_global_dir_without_a_token_is_refused(planted_copy: Path, tmp_path: Path) -> None:
    gdir = tmp_path / "global-hand-copy"
    gdir.mkdir()
    assert runner.main([str(planted_copy), "--global-dir", str(gdir), "--dry-run"]) == 2


def test_a_global_dir_below_another_copy_token_is_refused(
        planted_copy: Path, tmp_path: Path) -> None:
    src = tmp_path / "gsrc"
    (src / "a" / "b").mkdir(parents=True)
    _plain_db(src)
    gcopy = _copy(src, tmp_path / "copies" / "g")
    assert runner.main([str(planted_copy), "--global-dir", str(gcopy / "a" / "b"),
                        "--dry-run"]) == 2
    assert runner.main([str(planted_copy), "--global-dir", str(gcopy), "--dry-run"]) == 0


def test_a_global_file_swapped_for_a_symlink_exits_2(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "gsrc"
    src.mkdir()
    (src / "migrated-to-uv").write_text("x", encoding="utf-8")
    gcopy = _copy(src, tmp_path / "copies" / "g")
    live = tmp_path / "home" / ".aelfrice"
    live.mkdir(parents=True)
    (live / "migrated-to-uv").write_text("live", encoding="utf-8")
    (gcopy / "migrated-to-uv").unlink()
    (gcopy / "migrated-to-uv").symlink_to(live / "migrated-to-uv")
    assert runner.main([str(planted_copy), "--global-dir", str(gcopy),
                        "--only", "A064", "--out", str(tmp_path / "o.json")]) == 2
    assert "copy token" in capsys.readouterr().err


def test_global_dir_adds_global_and_user_and_global_log_probes_run_there(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gsrc = tmp_path / "global-src"
    _plain_db(gsrc)
    _plain_db(gsrc / "user")
    _plain_db(gsrc / "projects" / "abc")
    gdir = _copy(gsrc, tmp_path / "copies" / "global")
    capsys.readouterr()
    assert runner.main([str(planted_copy), "--legacy", str(gdir / "projects" / "abc"),
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


def test_a_symlink_to_a_db_with_wal_frames_exits_2(tmp_path: Path) -> None:
    src = tmp_path / "busy"
    writer = _wal_db(src)
    try:
        innocent = tmp_path / "innocent"
        innocent.mkdir()
        (innocent / "memory.db").symlink_to(src / "memory.db")
        assert runner.main([str(innocent), "--only", "A309"]) == 2
    finally:
        writer.close()


# --- the guard: defense in depth, by file identity --------------------------------


def test_guard_refuses_git_folders_by_name_case_and_structure(tmp_path: Path) -> None:
    assert guard.live_reason(_plain_db(tmp_path / "a" / DOT_GIT / "aelfrice"))
    assert guard.live_reason(_plain_db(tmp_path / "b" / ".Git" / "aelfrice"))
    odd = tmp_path / "c" / "not-named-git"
    (odd / "objects").mkdir(parents=True)
    (odd / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert guard.live_reason(_plain_db(odd / "aelfrice"))
    assert guard.live_reason(_plain_db(tmp_path / "d" / "plain")) is None


def test_guard_sees_through_symlinks_and_dotdot(tmp_path: Path) -> None:
    live = _plain_db(tmp_path / "repo" / DOT_GIT / "aelfrice").parent
    link = tmp_path / "linkdir"
    link.symlink_to(live)
    assert guard.live_reason(link / "memory.db")
    assert guard.live_reason(Path(f"{tmp_path}/repo/{DOT_GIT}/../{DOT_GIT}/aelfrice"))


@pytest.mark.skipif(not HAS_FIRMLINK, reason="firmlinks exist on macOS only")
def test_guard_sees_through_a_firmlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dot = tmp_path / "home" / ".aelfrice"
    _plain_db(dot)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    via = Path(str(FIRMLINK) + str(dot.resolve()))
    assert guard.same(via, dot)
    assert guard.live_reason(via / "memory.db")


def test_guard_refuses_the_dotdir_of_home_and_of_the_account(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    acct = tmp_path / "account"
    _plain_db(home / ".aelfrice")
    _plain_db(acct / ".aelfrice")
    monkeypatch.setenv("HOME", str(home))
    assert guard.live_reason(home / ".aelfrice" / "memory.db")
    import pwd
    real = pwd.getpwuid

    class _Pw:
        pw_dir = str(acct)

    def getpwuid(uid: int) -> Any:
        return _Pw() if uid == os.getuid() else real(uid)

    monkeypatch.setattr(pwd, "getpwuid", getpwuid)
    assert guard.live_reason(acct / ".aelfrice" / "memory.db")


def test_guard_refuses_the_env_store_folders(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _plain_db(tmp_path / "pinned")
    udb = _plain_db(tmp_path / "userpin")
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_USER_DB", str(udb))
    assert guard.live_reason(db.parent)
    assert guard.live_reason(db.parent / "census.json")
    assert guard.live_reason(udb.parent / "census.json")
    (tmp_path / "alias.db").symlink_to(db)
    assert guard.live_reason(tmp_path / "alias.db") == "the store AELFRICE_DB names"
    os.link(db, tmp_path / "elsewhere.db")
    assert guard.live_reason(tmp_path / "elsewhere.db")
    plain = _plain_db(tmp_path / "plain")
    os.link(plain, tmp_path / "plain-twin.db")
    assert guard.live_reason(plain) == "a file with another hard link"
    assert guard.live_reason(_plain_db(tmp_path / "unrelated")) is None


def test_guard_refuses_a_sidecar_link_into_a_live_folder(tmp_path: Path) -> None:
    live = tmp_path / "repo" / DOT_GIT / "aelfrice"
    live.mkdir(parents=True)
    (live / "memory.db-shm").write_bytes(b"")
    copy = _plain_db(tmp_path / "copy")
    (copy.parent / "memory.db-shm").symlink_to(live / "memory.db-shm")
    assert guard.live_reason(copy) is None
    assert guard.store_live_reason(copy)


def test_guard_refuses_marker_stores_and_unreadable_markers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = tmp_path / "proj"
    store = _plain_db(proj / "local-store").parent
    (proj / ".aelfrice.toml").write_text('[store]\npath = "local-store"\n', encoding="utf-8")
    assert guard.live_reason(store / "memory.db")
    assert guard.live_reason(_plain_db(proj / "elsewhere")) is None
    broken = tmp_path / "broken"
    _plain_db(broken / "s")
    (broken / ".aelfrice.toml").write_text("[store\npath = ", encoding="utf-8")
    assert guard.live_reason(broken / "s" / "memory.db")
    far = _plain_db(tmp_path / "far-store").parent
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / ".aelfrice.toml").write_text(f'[store]\npath = "{far}"\n', encoding="utf-8")
    assert guard.live_reason(far / "memory.db") is None
    monkeypatch.chdir(cwd)
    assert guard.live_reason(far / "memory.db")


# --- outputs ----------------------------------------------------------------------


def test_outputs_in_a_live_location_or_the_repo_are_refused(
        planted_copy: Path, tmp_path: Path) -> None:
    git_out = tmp_path / "repo" / DOT_GIT / "census.json"
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(git_out)]) == 2
    repo_out = runner.REPO_ROOT / "benchmarks" / "census-out-refusal-test.json"
    repo_labels = runner.REPO_ROOT / "benchmarks" / "census-labels-refusal.json"
    try:
        assert runner.main([str(planted_copy), "--dry-run", "--out", str(repo_out)]) == 2
        assert runner.main([str(planted_copy), "--dry-run", "--labels-out",
                            str(repo_labels)]) == 2
        assert not git_out.exists() and not repo_out.exists()
    finally:
        for p in (repo_out, repo_labels):
            if os.path.lexists(p):
                os.unlink(p)


def test_outputs_beside_a_store_or_a_token_are_refused(planted_copy: Path, tmp_path: Path) -> None:
    assert runner.main([str(planted_copy), "--out", str(planted_copy / "x.json")]) == 2
    beside = _plain_db(tmp_path / "loose").parent
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(beside / "x.json")]) == 2
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / copy_token.TOKEN_FILENAME).write_text("{}", encoding="utf-8")
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(tok / "x.json")]) == 2


def test_an_out_path_that_is_a_symlink_is_refused(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "o.json").symlink_to(target)
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(tmp_path / "o.json")]) == 2
    assert "existing symlink" in capsys.readouterr().err
    assert target.read_text("utf-8") == "keep"


def test_a_planted_temp_symlink_is_never_written_through(
        planted_copy: Path, tmp_path: Path) -> None:
    live = _plain_db(tmp_path / "repo" / DOT_GIT / "aelfrice")
    before = live.read_bytes()
    od = tmp_path / "od"
    od.mkdir()
    (od / "o.json.tmp").symlink_to(live)
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(od / "o.json")]) == 0
    assert live.read_bytes() == before
    assert json.loads((od / "o.json").read_text("utf-8"))["dry_run"] is True


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="needs dir_fd")
def test_an_out_folder_swapped_after_the_check_keeps_the_write_in_the_checked_folder(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = tmp_path / "repo" / DOT_GIT / "aelfrice"
    live.mkdir(parents=True)
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    real_run = runner.run_census

    def swap(*a: Any, **k: Any) -> dict[str, Any]:
        outdir.rename(tmp_path / "outdir-checked")
        outdir.symlink_to(live)
        return real_run(*a, **k)

    monkeypatch.setattr(runner, "run_census", swap)
    assert runner.main([str(planted_copy), "--only", "A001", "--out", str(outdir / "o.json")]) == 0
    assert list(live.iterdir()) == []
    assert (tmp_path / "outdir-checked" / "o.json").is_file()


def test_an_out_path_that_becomes_a_symlink_during_the_run_is_refused(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    out = tmp_path / "od" / "o.json"
    real_run = runner.run_census

    def swap(*a: Any, **k: Any) -> dict[str, Any]:
        out.symlink_to(target)
        return real_run(*a, **k)

    monkeypatch.setattr(runner, "run_census", swap)
    assert runner.main([str(planted_copy), "--only", "A001", "--out", str(out)]) == 2
    assert target.read_text("utf-8") == "keep"
    assert out.is_symlink()
    assert [p.name for p in out.parent.iterdir()] == ["o.json"]


def test_an_out_folder_that_is_a_symlink_exits_2_without_a_path(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    real = tmp_path / "real-out"
    real.mkdir()
    (tmp_path / "link-out").symlink_to(real)
    assert runner.main([str(planted_copy), "--dry-run",
                        "--out", str(tmp_path / "link-out" / "o.json")]) == 2
    err = capsys.readouterr().err
    assert "symlink" in err and str(tmp_path) not in err
    assert list(real.iterdir()) == []


def test_an_out_path_inside_any_copy_is_refused(planted_copy: Path, tmp_path: Path) -> None:
    src = tmp_path / "osrc"
    _plain_db(src)
    other = _copy(src, tmp_path / "copies" / "other")
    deep = other / "a" / "b"
    assert runner.main([str(planted_copy), "--dry-run", "--out", str(deep / "o.json")]) == 2
    assert not deep.exists()


def test_an_out_folder_swapped_between_its_check_and_its_open_is_refused(
        planted_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    import stat as st_mod
    import types
    real_fstat = os.fstat

    def fstat(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        if st_mod.S_ISDIR(st.st_mode):
            vals = list(st[:10])
            vals[1] += 1  # another inode
            return os.stat_result(vals)
        return st

    proxy = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    proxy.fstat = fstat
    monkeypatch.setattr(runner, "os", proxy)
    assert runner.main([str(planted_copy), "--dry-run",
                        "--out", str(tmp_path / "od" / "o.json")]) == 2
    assert "changed while it was checked" in capsys.readouterr().err


def test_dry_run_says_it_writes_no_labels_file(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = tmp_path / "labels.json"
    assert runner.main([str(planted_copy), "--dry-run", "--labels-out", str(labels)]) == 0
    assert "not written" in capsys.readouterr().err
    assert not labels.exists()


def test_out_and_labels_out_must_differ(planted_copy: Path, tmp_path: Path) -> None:
    same = str(tmp_path / "same.json")
    assert runner.main([str(planted_copy), "--out", same, "--labels-out", same]) == 2
    assert not os.path.lexists(same)
    real = tmp_path / "real" / "sub"
    real.mkdir(parents=True)
    (tmp_path / "lnk").symlink_to(tmp_path / "real")
    assert runner.main([str(planted_copy), "--out", str(real / "b.json"),
                        "--labels-out", str(tmp_path / "lnk" / "sub" / "b.json")]) == 2
    assert runner.main([str(planted_copy), "--out", str(real / "C.json"),
                        "--labels-out", str(real / "c.json")]) == 2
    (real / "e.json").write_text("{}", encoding="utf-8")
    assert runner.main([str(planted_copy), "--out", str(real / "e.json"),
                        "--labels-out", str(tmp_path / "lnk" / "sub" / "e.json")]) == 2
    assert (real / "e.json").read_text("utf-8") == "{}"
    (real / "h1.json").write_text("{}", encoding="utf-8")
    os.link(real / "h1.json", real / "h2.json")
    assert runner.main([str(planted_copy), "--out", str(real / "h1.json"),
                        "--labels-out", str(real / "h2.json")]) == 2
    assert sorted(p.name for p in real.iterdir()) == ["e.json", "h1.json", "h2.json"]
    new = tmp_path / "new-a" / "x.json"
    assert runner.main([str(planted_copy), "--out", str(new),
                        "--labels-out", str(tmp_path / "new-a" / "X.json")]) == 2
    assert not (tmp_path / "new-a").exists()
    assert not runner.same_output(real / "b.json", real / "d.json")


def test_a_refused_output_leaves_no_folder_behind(planted_copy: Path, tmp_path: Path) -> None:
    out = tmp_path / "made" / "by" / "out" / "o.json"
    labels = runner.REPO_ROOT / "benchmarks" / "census-labels-refusal.json"
    assert runner.main([str(planted_copy), "--out", str(out), "--labels-out", str(labels)]) == 2
    assert not (tmp_path / "made").exists() and not labels.exists()
    (tmp_path / "real").mkdir()
    (tmp_path / "lnk").symlink_to(tmp_path / "real")
    assert runner.main([str(planted_copy), "--out", str(out),
                        "--labels-out", str(tmp_path / "lnk" / "l.json")]) == 2
    assert not (tmp_path / "made").exists()
    beside = _plain_db(tmp_path / "loose").parent
    assert runner.main([str(planted_copy), "--out", str(out),
                        "--labels-out", str(beside / "l.json")]) == 2
    assert not (tmp_path / "made").exists()


def test_only_rejects_an_unknown_id(planted_copy: Path) -> None:
    assert runner.main([str(planted_copy), "--dry-run", "--only", "A999"]) == 2


# --- copy_store reads bytes only and never writes the source ----------------------------


def test_copy_store_writes_a_manifest_of_every_file_it_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    writer = _wal_db(src)
    try:
        _plain_db(src / "stores" / "named")
        (src / "hook_audit.jsonl").write_text("{}\n", encoding="utf-8")
        dest = _copy(src, tmp_path / "copy")
    finally:
        writer.close()
    doc = json.loads((dest / copy_token.TOKEN_FILENAME).read_text("utf-8"))
    assert sorted(doc["files"]) == ["hook_audit.jsonl", "memory.db", "stores/named/memory.db"]
    assert doc["dirs"] == ["stores", "stores/named"]
    assert (doc["root"]["dev"], doc["root"]["ino"]) == guard.ident(dest)
    for rel, entry in doc["files"].items():
        assert entry == copy_token.stamp_of(os.stat(dest / rel)), rel
    assert copy_token.TOKEN_RE.match(doc["token"])
    assert str(tmp_path) not in (dest / copy_token.TOKEN_FILENAME).read_text("utf-8")


def test_copy_store_refuses_symlinks_and_leaves_their_targets_alone(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    writer = _wal_db(real_dir)
    try:
        before = _snapshot(real_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "memory.db").symlink_to(real_dir / "memory.db")
        assert copy_store.main([str(src), str(tmp_path / "dest")]) == 2
        src2 = tmp_path / "src2"
        _plain_db(src2)
        (src2 / "linked").symlink_to(real_dir)
        assert copy_store.main([str(src2), str(tmp_path / "dest2")]) == 2
        assert _snapshot(real_dir) == before
        assert not (tmp_path / "dest").exists() and not (tmp_path / "dest2").exists()
    finally:
        writer.close()


def test_copy_store_reads_no_shm_and_opens_no_symlink(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    (src / "memory.db-shm").write_bytes(b"\0" * 64)
    real_read = copy_store._read  # pyright: ignore[reportPrivateUsage]
    seen: list[str] = []

    def read(s: Path, rel: Path, out: Path | None) -> Any:
        seen.append(rel.name)
        return real_read(s, rel, out)

    monkeypatch.setattr(copy_store, "_read", read)
    dest = _copy(src, tmp_path / "dest")
    assert "memory.db-shm" not in seen and "memory.db" in seen
    assert not (dest / "memory.db-shm").exists()
    (tmp_path / "s2").mkdir()
    (tmp_path / "s2" / "memory.db").symlink_to(src / "memory.db")
    with pytest.raises(OSError):
        real_read(tmp_path / "s2", Path("memory.db"), None)


def test_copy_store_catches_a_symlink_that_appears_mid_copy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    real_walk = copy_store._walk  # pyright: ignore[reportPrivateUsage]
    calls: list[int] = []

    def walk(p: Path) -> tuple[list[Path], list[Path], bool]:
        calls.append(1)
        if len(calls) == 3:
            (src / "late").symlink_to(tmp_path)
        return real_walk(p)

    monkeypatch.setattr(copy_store, "_walk", walk)
    assert copy_store.main([str(src), str(tmp_path / "dest")]) == 1
    assert not (tmp_path / "dest").exists()


def test_copy_store_reads_a_store_with_an_open_writer_without_changing_it(
        tmp_path: Path) -> None:
    src = tmp_path / "live-like"
    writer = _wal_db(src)
    try:
        before = _snapshot(src)
        dest = _copy(src, tmp_path / "copy")
        assert _snapshot(src) == before
    finally:
        writer.close()
    conn = core.open_ro(_gated(dest, "memory.db"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM beliefs").fetchone() == (1,)
    finally:
        conn.close()


def test_copy_store_gives_up_on_a_writer_that_never_stops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(copy_store, "RETRY_PAUSE_S", 0.0)
    src = tmp_path / "churn"
    src.mkdir()
    c = sqlite3.connect(str(src / "memory.db"))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t(x)")
    c.commit()
    c.close()
    real_read = copy_store._read  # pyright: ignore[reportPrivateUsage]

    def read(s: Path, rel: Path, out: Path | None) -> Any:
        if out is None and rel.name == "memory.db":  # the second pass: a write lands
            w = sqlite3.connect(str(src / "memory.db"))
            w.execute("INSERT INTO t VALUES (1)")
            w.commit()
            w.close()
        return real_read(s, rel, out)

    monkeypatch.setattr(copy_store, "_read", read)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert copy_store.main([str(src), str(tmp_path / "dest")]) == 1
    out, err = capsys.readouterr()
    assert "close the sessions" in err and str(tmp_path) not in err
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert copy_store.ATTEMPTS == 20
    rep = json.loads(out)
    assert rep["attempts"] == rep["busy"] == 20, rep
    readme = (runner.REPO_ROOT / "benchmarks" / "real_store_fire_census" / "README.md"
              ).read_text("utf-8")
    assert "up to 20 times" in readme


def test_copy_store_retries_when_a_file_vanishes_mid_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copy_store, "RETRY_PAUSE_S", 0.0)
    src = tmp_path / "src"
    _plain_db(src)
    (src / "hook_audit.jsonl").write_text("{}\n", encoding="utf-8")
    real_read = copy_store._read  # pyright: ignore[reportPrivateUsage]
    calls: list[int] = []

    def read(s: Path, rel: Path, out: Path | None) -> Any:
        if rel.name == "hook_audit.jsonl" and not calls:
            calls.append(1)
            raise FileNotFoundError("rotated away")
        return real_read(s, rel, out)

    monkeypatch.setattr(copy_store, "_read", read)
    rep = copy_store.copy_store(src, tmp_path / "dest")
    assert rep["quick_check_ok"] == 1 and rep["attempts"] == 2 and rep["busy"] == 1, rep


def test_copy_store_refuses_a_symlink_its_first_walk_sees(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    real_walk = copy_store._walk  # pyright: ignore[reportPrivateUsage]
    calls: list[int] = []

    def walk(p: Path) -> tuple[list[Path], list[Path], bool]:
        calls.append(1)
        if len(calls) != 2:
            return real_walk(p)
        (src / "brief").symlink_to(tmp_path)
        try:
            return real_walk(p)
        finally:
            (src / "brief").unlink()

    monkeypatch.setattr(copy_store, "_walk", walk)
    assert copy_store.main([str(src), str(tmp_path / "dest")]) == 1
    assert not (tmp_path / "dest").exists()


def test_copy_store_sees_a_same_size_write_that_keeps_the_mtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copy_store, "RETRY_PAUSE_S", 0.0)
    monkeypatch.setattr(copy_store, "ATTEMPTS", 2)
    db = _plain_db(tmp_path / "src")
    real_read = copy_store._read  # pyright: ignore[reportPrivateUsage]

    def read(s: Path, rel: Path, out: Path | None) -> Any:
        if out is None and rel.name == "memory.db":
            st = os.stat(db)
            raw = bytearray(db.read_bytes())
            raw[-1] ^= 0xFF
            db.write_bytes(bytes(raw))
            os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns))
        return real_read(s, rel, out)

    monkeypatch.setattr(copy_store, "_read", read)
    rep = copy_store.copy_store(db.parent, tmp_path / "dest")
    assert rep["quick_check_ok"] == 0 and rep["busy"] == 2


def test_copy_store_stops_at_once_on_an_unsound_copy(tmp_path: Path) -> None:
    src = tmp_path / "corrupt"
    db = _plain_db(src)
    conn = sqlite3.connect(str(db))
    conn.executemany("INSERT INTO beliefs VALUES (?)", [(f"id{i:05d}",) for i in range(3000)])
    conn.commit()
    conn.close()
    raw = bytearray(db.read_bytes())
    raw[4096 * 2: 4096 * 3] = b"\x07" * 4096
    db.write_bytes(bytes(raw))
    rep = copy_store.copy_store(src, tmp_path / "dest")
    assert rep["quick_check_ok"] == 0 and rep["attempts"] == 1 and rep["busy"] == 0
    assert not (tmp_path / "dest").exists()


def test_copy_store_refuses_dest_inside_src_by_identity(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    alias = tmp_path / "alias"
    alias.symlink_to(src)
    assert copy_store.main([str(src), str(alias / "inner")]) == 2
    assert not (src / "inner").exists()
    if HAS_FIRMLINK:
        via = str(FIRMLINK) + str(src.resolve() / "inner2")
        assert copy_store.main([str(src), via]) == 2
        assert not (src / "inner2").exists()


def test_copy_store_never_deletes_a_path_it_did_not_create(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    keep = tmp_path / "dest.partial"
    keep.mkdir()
    (keep / "keep.txt").write_text("user data", encoding="utf-8")
    _copy(src, tmp_path / "dest")
    assert (keep / "keep.txt").read_text("utf-8") == "user data"
    existing = tmp_path / "taken"
    existing.mkdir()
    assert copy_store.main([str(src), str(existing)]) == 2
    assert existing.is_dir()


def test_copy_store_refuses_a_destination_in_the_repo(planted_copy: Path) -> None:
    dest = runner.REPO_ROOT / "benchmarks" / "census-copy-refusal-test"
    assert copy_store.main([str(planted_copy), str(dest)]) == 2
    assert not dest.exists()


def test_a_threaded_writer_never_gets_a_torn_copy_through(tmp_path: Path) -> None:
    src = tmp_path / "churn"
    src.mkdir()
    c = sqlite3.connect(str(src / "memory.db"))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t(x)")
    c.commit()
    c.close()
    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            w = sqlite3.connect(str(src / "memory.db"), timeout=5.0)
            w.execute("INSERT INTO t VALUES (1)")
            w.commit()
            w.close()

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        codes = [copy_store.main([str(src), str(tmp_path / f"d{i}")]) for i in range(3)]
    finally:
        stop.set()
        t.join(timeout=30)
    assert set(codes) <= {0, 1}
    for i, code in enumerate(codes):
        d = tmp_path / f"d{i}"
        if code == 0:
            conn = core.open_ro(_gated(d, "memory.db"))
            conn.close()
        else:
            assert not d.exists()


# --- the documented limits match the probe tests ----------------------------------------


_BANNED_OS = frozenset({"open", "stat", "lstat", "listdir", "scandir", "walk", "fwalk",
                        "readlink", "fdopen"})


def _file_access_calls(source: str) -> list[str]:
    """Calls in probe code that would read a file around the copy manifest."""
    import ast
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("open", "Path", "PurePath", "PosixPath"):
            out.append(f.id)
        elif isinstance(f, ast.Attribute):
            base = f.value
            if isinstance(base, ast.Name) and base.id in ("io", "builtins", "sqlite3", "shutil",
                                                           "glob", "pathlib"):
                out.append(f"{base.id}.{f.attr}")
            elif isinstance(base, ast.Name) and base.id == "os" and f.attr in _BANNED_OS:
                out.append(f"os.{f.attr}")
            elif (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                  and base.value.id == "os" and base.attr == "path"):
                out.append(f"os.path.{f.attr}")
            elif f.attr in ("read_text", "read_bytes", "open") and isinstance(base, ast.Call) \
                    and isinstance(base.func, ast.Name) and base.func.id == "Path":
                out.append(f"Path().{f.attr}")
    return out


def test_the_scan_catches_each_way_around_the_gate() -> None:
    planted = "\n".join([
        "open('x')", "io.open('x')", "os.open('x', 0)", "os.path.exists('x')",
        "os.listdir('x')", "Path('x').read_text()", "sqlite3.connect('x')",
        "shutil.copy('a', 'b')", "glob.glob('*')", "pathlib.Path('x')",
    ])
    found = _file_access_calls(planted)
    for want in ("open", "io.open", "os.open", "os.path.exists", "os.listdir", "Path",
                 "Path().read_text", "sqlite3.connect", "shutil.copy", "glob.glob",
                 "pathlib.Path"):
        assert want in found, want
    assert _file_access_calls("ctx.sidecar('x').read_text()\ncc.open_ro(p)\n") == []


def test_probe_modules_read_files_only_through_the_gate() -> None:
    pkg = runner.REPO_ROOT / "benchmarks" / "real_store_fire_census"
    mods = sorted(pkg.glob("probes_*.py"))
    assert len(mods) == len(registry.SUBSYSTEM_MODULES)
    bad = {m.name: c for m in mods if (c := _file_access_calls(m.read_text("utf-8")))}
    assert bad == {}


def test_the_readme_lists_every_probe_limit() -> None:
    readme = (runner.REPO_ROOT / "benchmarks" / "real_store_fire_census" / "README.md"
              ).read_text("utf-8")
    fired_part, _, silent_part = readme.partition("have no test in which they report `silent`")
    for item_id in registry.NO_FIRED_TEST:
        assert f"- `{item_id}`:" in fired_part, item_id
    for item_id in registry.NO_SILENT_RESULT:
        assert f"- `{item_id}`:" in silent_part, item_id


@pytest.mark.timeout(1200)
def test_the_limit_lists_match_what_the_probe_tests_observe(tmp_path: Path) -> None:
    """Run every probe test module with a status log, then compare both lists."""
    log = tmp_path / "status.log"
    env = dict(os.environ, FIRE_CENSUS_STATUS_LOG=str(log))
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests.fire_census_status_log", *PROBE_TEST_MODULES],
        cwd=str(runner.REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=1100, check=False)
    assert cp.returncode == 0, cp.stdout[-2000:]
    seen: dict[str, set[str]] = collections.defaultdict(set)
    for line in log.read_text("utf-8").splitlines():
        item, status = json.loads(line)
        seen[item].add(status)
    traceable = {p.item_id for p in registry.all_probes() if p.kind != "untraceable"}
    assert traceable <= set(seen)
    no_trace = {i for i in traceable if not seen[i] & {"fired", "ambiguous"}}
    no_silent = {i for i in traceable if "silent" not in seen[i]}
    assert no_trace == set(registry.NO_FIRED_TEST)
    assert no_silent == set(registry.NO_SILENT_RESULT)


def test_the_runner_module_runs_as_a_script(planted_copy: Path) -> None:
    cp = subprocess.run(
        [sys.executable, "-m", "benchmarks.real_store_fire_census", str(planted_copy),
         "--dry-run", "--only", "A001"],
        cwd=str(runner.REPO_ROOT), capture_output=True, text=True, timeout=120, check=False)
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout)["probes"] == [["A001", "untraceable", "beliefs-graph-feedback"]]
