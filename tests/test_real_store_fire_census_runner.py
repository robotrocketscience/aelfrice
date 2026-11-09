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


def test_verify_db_names_each_reason_it_refuses(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    dest = _copy(src, tmp_path / "copy")
    db = dest / "memory.db"
    assert copy_token.verify_db(db) == guard.ident(db)
    os.link(db, tmp_path / "second.db")
    with pytest.raises(copy_token.TokenError, match="another hard link"):
        copy_token.verify_db(db)
    os.unlink(tmp_path / "second.db")
    st = os.stat(db)
    os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    with pytest.raises(copy_token.TokenError, match="not the file"):
        copy_token.verify_db(db)
    os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns))
    (dest / "memory.db-shm").symlink_to(tmp_path / "anything")
    with pytest.raises(copy_token.TokenError, match="symlink"):
        copy_token.verify_db(db)
    with pytest.raises(copy_token.TokenError, match="no copy token"):
        copy_token.verify_db(_plain_db(tmp_path / "bare"))


def test_a_malformed_token_is_refused(tmp_path: Path) -> None:
    db = _plain_db(tmp_path / "m")
    st = os.stat(db)
    rec = {"memory.db": copy_token.stamp_of(st)}
    tok = tmp_path / "m" / copy_token.TOKEN_FILENAME
    docs: list[dict[str, Any]] = [{"version": 1, "token": "not-hex", "dbs": rec},
                                  {"version": 2, "token": "a" * 32, "dbs": rec},
                                  {"version": 1, "token": "a" * 32, "dbs": []}]
    for doc in docs:
        tok.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(copy_token.TokenError, match="malformed"):
            copy_token.verify_db(db)
    tok.write_text(json.dumps({"version": 1, "token": "a" * 32, "dbs": rec}), encoding="utf-8")
    assert copy_token.verify_db(db) == (st.st_dev, st.st_ino)


def test_write_token_refuses_a_linked_db_and_an_unfolded_wal(tmp_path: Path) -> None:
    db = _plain_db(tmp_path / "linked")
    os.link(db, tmp_path / "other-name.db")
    with pytest.raises(copy_token.TokenError, match="single-link"):
        copy_token.write_token(db.parent)
    writer = _wal_db(tmp_path / "walled")
    try:
        with pytest.raises(copy_token.TokenError, match="non-empty -wal"):
            copy_token.write_token(tmp_path / "walled")
    finally:
        writer.close()
    assert not (tmp_path / "linked" / copy_token.TOKEN_FILENAME).exists()


def test_open_ro_refuses_a_symlink_and_an_unfolded_wal(tmp_path: Path) -> None:
    real = _plain_db(tmp_path / "real")
    (tmp_path / "link.db").symlink_to(real)
    with pytest.raises(OSError):
        core.open_ro(tmp_path / "link.db")
    writer = _wal_db(tmp_path / "walled")
    try:
        with pytest.raises(core.UnfoldedWalError):
            core.open_ro(tmp_path / "walled" / "memory.db")
    finally:
        writer.close()


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
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


def test_dry_run_says_it_writes_no_labels_file(
        planted_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = tmp_path / "labels.json"
    assert runner.main([str(planted_copy), "--dry-run", "--labels-out", str(labels)]) == 0
    assert "not written" in capsys.readouterr().err
    assert not labels.exists()


def test_only_rejects_an_unknown_id(planted_copy: Path) -> None:
    assert runner.main([str(planted_copy), "--dry-run", "--only", "A999"]) == 2


# --- copy_store reads bytes only and never writes the source ----------------------------


def test_copy_store_writes_a_token_that_records_each_db(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _plain_db(src)
    _plain_db(src / "stores" / "named")
    dest = _copy(src, tmp_path / "copy")
    doc = json.loads((dest / copy_token.TOKEN_FILENAME).read_text("utf-8"))
    assert sorted(doc["dbs"]) == ["memory.db", "stores/named/memory.db"]
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
    conn = core.open_ro(dest / "memory.db", copy_token.verify_db(dest / "memory.db"))
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
    err = capsys.readouterr().err
    assert "close the sessions" in err and str(tmp_path) not in err
    assert sorted(p.name for p in tmp_path.iterdir()) == before


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
            conn = core.open_ro(d / "memory.db", copy_token.verify_db(d / "memory.db"))
            conn.close()
        else:
            assert not d.exists()


# --- the documented limits match the probe tests ----------------------------------------


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
