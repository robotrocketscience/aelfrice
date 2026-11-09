"""Planted-store tests for the locks-core-federation census probes (#1887).

Each test plants its own store under `tmp_path` with HOME, AELFRICE_DB and
AELFRICE_USER_DB pinned there, fires the feature through aelfrice's own
code, and asserts the probe reads 'fired' with the expected count. Every
traceable probe is also asserted 'silent' on an empty planted store, and
every untraceable probe reports 'untraceable'.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aelfrice import core_gate_background as bg
from aelfrice.derivation import _content_hash  # pyright: ignore[reportPrivateUsage]
from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_locks_core_federation as P
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

BY_ID = {p.item_id: p for p in P.PROBES}
TRACEABLE = sorted(P.TRACEABLE)
UNTRACEABLE = sorted(P.UNTRACEABLE)
EXPECTED_IDS = {f"A{i:03d}" for i in range(184, 217)}
BG_WAIT_S = 120.0


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    store = plant(tmp_path, monkeypatch)
    monkeypatch.delenv("AELF_SESSION_ID", raising=False)
    monkeypatch.delenv("AELFRICE_CORE_GATE_SESSION_END", raising=False)
    # tests/conftest.py keeps the background labeler off; the one test
    # that exercises it turns it on with a fake CLI first on PATH.
    monkeypatch.setenv("AELFRICE_CORE_GATE_BACKGROUND", "0")
    yield store
    store.close()


def user_ctx(ps: PlantedStore) -> cc.Ctx:
    return cc.Ctx("user", ps.user_db.parent, ps.global_dir, True, ps.user_db, "user")


def run(ps: PlantedStore, item: str, user: bool = False) -> cc.Result:
    c = user_ctx(ps) if user else ps.ctx()
    ps.close()
    c.open()
    try:
        return cc.run_one(BY_ID[item], c, 60.0)
    finally:
        c.close()


def fired(r: cc.Result, n: int) -> None:
    assert r.status == "fired", r
    assert r.count == n, r


def core_belief(ps: PlantedStore, text: str, session_id: str | None = None) -> str:
    return ps.belief(text, alpha=4.0, beta=1.0, session_id=session_id)


def cli_stdin(ps: PlantedStore, stdin: str, *args: str) -> int:
    old = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        return ps.cli(*args)
    finally:
        sys.stdin = old


def q(ps: PlantedStore, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    ps.close()
    conn = cc.open_ro(ps.db_file)
    try:
        return [tuple(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def batches(ps: PlantedStore, origin: str | None = None) -> list[tuple[str, str]]:
    where = "WHERE origin = ?" if origin else ""
    rows = q(ps, f"SELECT batch_id, items_json FROM core_gate_batches {where} "
                 "ORDER BY created_at, batch_id", (origin,) if origin else ())
    return [(str(r[0]), str(r[1])) for r in rows]


def reply(items_json: str, labels: dict[str, str] | None = None, default: str = "A",
          fence: bool = False) -> str:
    """A classifier reply labeling each batch index; labels keyed by belief id."""
    items: list[dict[str, Any]] = json.loads(items_json)
    body = json.dumps([{"index": d["index"],
                        "label": (labels or {}).get(d["belief_id"], default)} for d in items])
    return f"```json\n{body}\n```" if fence else body


def emit(ps: PlantedStore) -> int:
    return ps.cli("doctor", "core-gate", "--emit")


FAKE_CLI = """#!{py}
import json, sys
prompt = sys.stdin.read()
payload = json.loads(prompt.strip().splitlines()[-1])
labels = [{{"index": d["index"], "label": "A"}} for d in payload]
print(json.dumps({{"result": json.dumps(labels), "total_cost_usd": {cost}}}))
sys.exit({rc})
"""


def fake_cli(dirpath: Path, cost: float = 0.01, rc: int = 0) -> Path:
    """A stand-in for the host CLI the background labeler calls."""
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / bg.CLAUDE_BIN
    p.write_text(FAKE_CLI.format(py=sys.executable, cost=cost, rc=rc))
    p.chmod(0o755)
    return p


def wait_for(path: Path, timeout_s: float = BG_WAIT_S) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if path.is_file() and path.read_text().strip():
            return
        time.sleep(0.25)
    raise AssertionError("background log never written")


# --- registry ----------------------------------------------------------------


def test_probe_ids_match_the_subsystem() -> None:
    assert set(BY_ID) == EXPECTED_IDS
    assert len(P.PROBES) == len(EXPECTED_IDS)
    assert P.TRACEABLE | P.UNTRACEABLE == EXPECTED_IDS


# --- silent on an empty store --------------------------------------------------


@pytest.mark.parametrize("item", TRACEABLE)
def test_silent_on_empty_store(ps: PlantedStore, item: str) -> None:
    r = run(ps, item)
    assert r.status == "silent", r
    assert r.count in (0, None), r


@pytest.mark.parametrize("item", ["A208", "A214", "A216"])
def test_silent_on_empty_user_store(ps: PlantedStore, item: str) -> None:
    from aelfrice.store import MemoryStore

    ps.user_db.parent.mkdir(parents=True, exist_ok=True)
    MemoryStore(str(ps.user_db)).close()
    r = run(ps, item, user=True)
    assert r.status == "silent", r


@pytest.mark.parametrize("item", UNTRACEABLE)
def test_untraceable_probes_say_so(item: str) -> None:
    r = cc.run_one(BY_ID[item], None, 5.0)
    assert r.status == "untraceable" and r.count is None and r.note, r


# --- old schemas ---------------------------------------------------------------


def _raw_store(tmp_path: Path, ddl: list[str]) -> cc.Ctx:
    d = tmp_path / "raw"
    d.mkdir()
    conn = sqlite3.connect(str(d / "memory.db"))
    try:
        for stmt in ddl:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    c = cc.Ctx("store01", d, None, False, d / "memory.db", "repo")
    c.open()
    return c


def test_store_without_gate_tables_is_untraceable_for_a209(tmp_path: Path) -> None:
    c = _raw_store(tmp_path, [
        "CREATE TABLE beliefs (id TEXT, content TEXT, created_at TEXT)",
        "INSERT INTO beliefs VALUES ('b1', 'x', '2026-11-01')"])
    try:
        r = cc.run_one(BY_ID["A209"], c)
    finally:
        c.close()
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["beliefs_since_510"] == 1


def test_store_without_beliefs_table_is_untraceable_for_a205(tmp_path: Path) -> None:
    c = _raw_store(tmp_path, ["CREATE TABLE notes (id TEXT)"])
    try:
        r = cc.run_one(BY_ID["A205"], c)
    finally:
        c.close()
    assert r.status == "untraceable", r


def test_pre_v3_schema_is_silent_for_a205(tmp_path: Path) -> None:
    c = _raw_store(tmp_path, [
        "CREATE TABLE beliefs (id TEXT, content TEXT)",
        "INSERT INTO beliefs VALUES ('b1', 'x')"])
    try:
        r = cc.run_one(BY_ID["A205"], c)
    finally:
        c.close()
    assert r.status == "silent" and r.detail["beliefs"] == 1, r


# --- core gate -----------------------------------------------------------------


def test_emit_open_then_cli_accept(ps: PlantedStore) -> None:
    core_belief(ps, "The census copies every store before it reads one.")
    core_belief(ps, "Planted stores live under the test root directory.")
    assert emit(ps) == 0
    fired(run(ps, "A185"), 1)
    r = run(ps, "A204")
    fired(r, 1)
    assert r.detail["open_doctor"] == 1 and r.detail["claimed_items"] == 2
    assert run(ps, "A184").status == "silent"
    [(bid, items)] = batches(ps, "doctor")
    # Fenced reply: the strict parser strips one fence.
    assert cli_stdin(ps, reply(items, fence=True), "core-gate", "accept", bid) == 0
    r = run(ps, "A184")
    fired(r, 1)
    assert r.detail["doctor_accepted"] == 1
    assert run(ps, "A204").status == "silent"
    assert run(ps, "A186").status == "silent"
    assert run(ps, "A209").status == "silent"
    # Labels and the batch are under the version the shipped code computes.
    r = run(ps, "A197")
    assert r.status == "silent" and r.detail["cur_labels"] == 2, r


def test_cli_accept_refuses_partial_reply(ps: PlantedStore) -> None:
    core_belief(ps, "A refused reply writes nothing to the label cache.")
    core_belief(ps, "Partial classifier replies are rejected whole.")
    assert emit(ps) == 0
    [(bid, _items)] = batches(ps, "doctor")
    one = json.dumps([{"index": 0, "label": "A"}])
    assert cli_stdin(ps, one, "core-gate", "accept", bid) == 1
    assert run(ps, "A184").status == "silent"


def test_rerun_subset_leaves_accepted_batch_without_labels(ps: PlantedStore) -> None:
    texts = ["Rerun keeps beliefs that are still in core.",
             "Rerun drops beliefs that were locked since the accept.",
             "Rerun derives a new batch id from the smaller hash set."]
    for t in texts:
        core_belief(ps, t)
    assert emit(ps) == 0
    [(bid, items)] = batches(ps, "doctor")
    assert cli_stdin(ps, reply(items), "core-gate", "accept", bid) == 0
    assert ps.lock(texts[1]) == 0
    assert ps.cli("doctor", "core-gate", "--rerun", bid) == 0
    r = run(ps, "A186")
    fired(r, 1)
    assert r.detail["subset_rerun_batches"] == 1


def test_label_limit_and_gate_exclusion(ps: PlantedStore) -> None:
    now = datetime.now(timezone.utc)
    b_id = ps.belief("It failed on the second run.")
    store = ps.open()
    for k, sid in ((1, "e1"), (2, "e2")):
        store.record_corroboration(
            b_id, source_type="transcript_ingest", session_id=sid,
            ts=(now + timedelta(hours=k)).isoformat())
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]
    c_id = core_belief(ps, "Run the tests before every push.")
    a_id = core_belief(ps, "Python's json module is part of the standard library.")
    assert emit(ps) == 0
    [(bid, items)] = batches(ps, "doctor")
    assert len(json.loads(items)) == 3
    labels = {b_id: "B", c_id: "C", a_id: "A"}
    assert cli_stdin(ps, reply(items, labels), "core-gate", "accept", bid) == 0
    fired(run(ps, "A193"), 1)
    r = run(ps, "A196")
    fired(r, 2)
    assert r.detail["c_excluded"] == 1 and r.detail["b_excluded"] == 1


def test_b_label_under_limit_not_counted(ps: PlantedStore) -> None:
    b_id = core_belief(ps, "That value is stale.")
    assert emit(ps) == 0
    [(bid, items)] = batches(ps, "doctor")
    assert cli_stdin(ps, reply(items, {b_id: "B"}), "core-gate", "accept", bid) == 0
    assert run(ps, "A193").status == "silent"
    assert run(ps, "A196").status == "silent"


def test_classifier_version_change_orphans_labels(ps: PlantedStore) -> None:
    old = "The old classifier labeled this belief."
    ps.belief(old)
    ps.open().put_core_gate_labels({_content_hash(old): "A"},
                                   classifier_version="core-gate-0000000000000000",
                                   batch_id=None, labeled_at="2020-01-01T00:00:00+00:00")
    # One stored version, but not the one the shipped code computes: the
    # corrected probe sees the orphan before any newer label is written.
    r = run(ps, "A197")
    fired(r, 1)
    assert r.detail["versions"] == 1 and r.detail["cur_labels"] == 0
    core_belief(ps, "The new classifier batches this belief.")
    assert emit(ps) == 0
    r = run(ps, "A197")
    fired(r, 1)
    assert r.detail["versions"] == 2 and r.detail["stale_labels"] == 1
    assert r.detail["cur_batches"] == 1 and r.detail["bad_format"] == 0


def test_self_check_runs_on_fourth_accept(ps: PlantedStore) -> None:
    for i in range(151):
        core_belief(ps, f"Planted core belief number {i} for the self check run.")
    assert emit(ps) == 0
    bs = batches(ps, "doctor")
    assert len(bs) == 4
    for bid, items in bs[:3]:
        assert cli_stdin(ps, reply(items), "core-gate", "accept", bid) == 0
    assert run(ps, "A198").status == "silent"
    bid, items = bs[3]
    assert cli_stdin(ps, reply(items), "core-gate", "accept", bid) == 0
    r = run(ps, "A198")
    fired(r, 1)
    assert r.detail["runs_with_4plus"] == 1


def test_session_end_continuation(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_CORE_GATE_SESSION_END", "1")
    core_belief(ps, "Session end batches hold this session's beliefs.", session_id="sess-c")
    assert ps.hook("stop", {"hook_event_name": "Stop", "session_id": "sess-c",
                            "cwd": str(ps.repo)}) == 0
    r = run(ps, "A209")
    fired(r, 1)
    assert r.detail["core_session_beliefs_since_510"] == 1
    r = run(ps, "A199")
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["session_end_no_bg_row"] == 1
    r = run(ps, "A204")
    fired(r, 1)
    assert r.detail["open_session_end"] == 1
    assert run(ps, "A194").status == "silent"
    [(bid, items)] = batches(ps, "session_end")
    assert cli_stdin(ps, reply(items), "core-gate", "accept", bid) == 0
    r = run(ps, "A184")
    fired(r, 1)
    assert r.detail["session_end_accepted_no_bg_row"] == 1


def test_session_end_skips_other_sessions(ps: PlantedStore,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_CORE_GATE_SESSION_END", "1")
    core_belief(ps, "A belief from another session is not a candidate.", session_id="sess-x")
    assert ps.hook("stop", {"hook_event_name": "Stop", "session_id": "sess-y",
                            "cwd": str(ps.repo)}) == 0
    assert run(ps, "A209").status == "silent"


def test_background_labeler_via_stop(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    binp = ps.root / "bin"
    fake_cli(binp)
    # PATH holds the fake and the system dirs only, so the real host CLI
    # can never be the one found.
    path = os.pathsep.join([str(binp), str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", path)
    monkeypatch.delenv("AELFRICE_CORE_GATE_BACKGROUND", raising=False)
    core_belief(ps, "The background child labels the session batch.", session_id="sess-b")
    assert ps.hook("stop", {"hook_event_name": "Stop", "session_id": "sess-b",
                            "cwd": str(ps.repo)}) == 0
    wait_for(ps.store_dir / P.BG_LOG)
    r = run(ps, "A194")
    fired(r, 1)
    assert r.detail.get("accepted") == 1 and r.detail["session_end_accepted_by_bg"] == 1
    fired(run(ps, "A209"), 1)
    assert run(ps, "A199").status == "silent"
    assert run(ps, "A184").status == "silent"
    # The corrected A195 counts every run the fixed cap governed.
    r = run(ps, "A195")
    fired(r, 1)
    assert r.detail["cost_ge_0_09"] == 0 and r.detail["timeout"] == 0


def test_background_budget_and_timeout(ps: PlantedStore) -> None:
    core_belief(ps, "A run that spends its budget leaves the batch open.")
    core_belief(ps, "A run that times out leaves the batch open too.")
    assert emit(ps) == 0
    [(bid, _items)] = batches(ps, "doctor")
    cli_path = fake_cli(ps.root / "bin", cost=0.1, rc=1)
    cp = subprocess.run([sys.executable, "-P", "-m", "aelfrice.core_gate_background", bid,
                         str(cli_path)], env=dict(os.environ), capture_output=True,
                        timeout=120, check=False)
    assert cp.returncode == 1
    r = run(ps, "A195")
    fired(r, 1)
    assert r.detail["cost_ge_0_09"] == 1

    def timeout_runner(argv: list[str], **kw: object) -> object:
        raise subprocess.TimeoutExpired(argv, 300)

    store = ps.open()
    rec = bg.label_batch(store, bid, claude=str(cli_path), run=timeout_runner)  # pyright: ignore[reportArgumentType]
    bg._append_log(store.db_path, rec)  # pyright: ignore[reportPrivateUsage]
    r = run(ps, "A195")
    fired(r, 2)
    assert r.detail["timeout"] == 1 and r.detail["invoked_runs"] == 2
    fired(run(ps, "A194"), 2)
    fired(run(ps, "A204"), 1)


# --- peer search precondition ---------------------------------------------------


def test_peer_visible_rows(ps: PlantedStore) -> None:
    ps.belief("A project scoped belief stays out of peer search.")
    assert run(ps, "A205").status == "silent"
    ps.belief("Every repo here uses uv for Python environments.", scope="global")
    ps.belief("The shared team rule is to run tests before a push.", scope="shared:team")
    r = run(ps, "A205")
    fired(r, 2)
    assert r.detail["active_global"] == 1 and r.detail["active_shared"] == 1
    assert r.detail["active_project"] == 1


# --- locks -----------------------------------------------------------------------


def test_lock_session_inference(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ps.lock("Never read a real store from the census.") == 0
    r = run(ps, "A192")
    # A lock with no session id to infer is not an inference.
    assert r.status == "silent" and r.count == 0, r
    assert r.detail["null_session"] == 1 and r.detail["lock_ingests"] == 1
    monkeypatch.setenv("AELF_SESSION_ID", "sess-env")
    assert ps.lock("Count rows and never print text.") == 0
    r = run(ps, "A192")
    fired(r, 1)
    assert r.detail["with_session"] == 1 and r.detail["lock_ingests"] == 2


def test_lock_suffix_eligibility(ps: PlantedStore) -> None:
    ps.belief("Always remember this for two weeks: use the staging bucket.", session_id="s1")
    r = run(ps, "A211")
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["window_spec_post"] == 1 and r.detail["directives"] == 1


def test_lock_suffix_not_eligible(ps: PlantedStore) -> None:
    ps.belief("Always run the tests before every push.", session_id="s1")
    ps.belief("Always remember this for two weeks: no session, never evaluated.")
    r = run(ps, "A211")
    assert r.status == "silent" and r.count == 0, r
    assert r.detail["rows"] == 1 and r.detail["directives"] == 1
    assert r.detail["window_spec"] == 0


def test_time_boxed_lock_and_sweep(ps: PlantedStore) -> None:
    assert ps.lock("Use the staging bucket this week.", "--for", "7d") == 0
    r = run(ps, "A212")
    fired(r, 1)
    assert r.detail["windowed_active_locked"] == 1
    assert ps.open().sweep_expired_locks(now="2999-01-01T00:00:00+00:00") == 1
    r = run(ps, "A212")
    fired(r, 1)
    assert r.detail["expire_rows"] == 1 and r.detail["sweep_marker"] == 1
    assert r.detail["windowed_active_locked"] == 0


def test_permanent_lock_is_not_time_boxed(ps: PlantedStore) -> None:
    assert ps.lock("A lock without a window never expires.") == 0
    assert run(ps, "A212").status == "silent"


def test_typed_lock_outcome_ledger(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "sess-l",
               "cwd": str(ps.repo), "prompt": "/aelf:lock Census probes return integers only."}
    assert ps.hook("main", payload) == 0
    r = run(ps, "A213")
    fired(r, 1)
    assert r.detail["failed"] == 0
    monkeypatch.setenv("AELFRICE_USER_DB", os.environ["AELFRICE_DB"])
    payload["prompt"] = "/aelf:lock --user This lock is refused by the same file guard."
    assert ps.hook("main", payload) == 0
    r = run(ps, "A213")
    fired(r, 2)
    assert r.detail["failed"] == 1 and r.detail["user_scope"] == 1


def test_user_lock_store_and_merge(ps: PlantedStore) -> None:
    assert ps.lock("Prefer uv for every Python environment.", "--user") == 0
    r = run(ps, "A216", user=True)
    fired(r, 2)
    assert r.detail["user_store_active_locks"] == 1
    assert r.detail["user_store_lock_ingest_rows"] == 1
    fired(run(ps, "A216"), 1)  # feed row beside the repo store
    assert ps.hook("main", {"hook_event_name": "UserPromptSubmit", "session_id": "sess-u",
                            "cwd": str(ps.repo),
                            "prompt": "which python environment tool do we use"}) == 0
    r = run(ps, "A214")
    fired(r, 1)
    assert r.detail["user_entries"] == 1
    assert run(ps, "A208").status == "silent"


def test_user_store_fires_after_its_lock_is_unlocked(ps: PlantedStore) -> None:
    assert ps.lock("A user lock that is later unlocked.", "--user") == 0
    [(bid,)] = [(r[0],) for r in _user_rows(ps, "SELECT id FROM beliefs WHERE lock_level = 'user'")]
    assert ps.cli("unlock", "--user", str(bid)) == 0
    r = run(ps, "A216", user=True)
    fired(r, 1)
    assert r.detail["user_store_active_locks"] == 0
    assert r.detail["user_store_lock_ingest_rows"] == 1
    assert r.detail["user_store_unlock_audits"] == 1


def _user_rows(ps: PlantedStore, sql: str) -> list[tuple[Any, ...]]:
    conn = cc.open_ro(ps.user_db)
    try:
        return [tuple(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def test_repo_lock_wins_over_same_user_lock(ps: PlantedStore) -> None:
    text = "Never push from the home directory repository."
    assert ps.lock(text) == 0
    assert ps.lock(text, "--user") == 0
    assert ps.hook("main", {"hook_event_name": "UserPromptSubmit", "session_id": "sess-w",
                            "cwd": str(ps.repo),
                            "prompt": "can I push from the home directory"}) == 0
    r = run(ps, "A208")
    fired(r, 1)
    assert r.detail["dropped_user_locks"] == 1
    assert run(ps, "A214").status == "silent"


def test_promote_user(ps: PlantedStore) -> None:
    text = "Batch and cache every repeated verification."
    assert ps.lock(text) == 0
    [(bid,)] = q(ps, "SELECT id FROM beliefs WHERE lock_level = 'user'")
    assert ps.cli("lock", "--promote-user", str(bid)) == 0
    fired(run(ps, "A190"), 1)
    r = run(ps, "A216", user=True)
    assert r.status == "fired" and r.detail["user_store_active_locks"] == 1, r
    assert run(ps, "A216").status == "silent"  # promote rows are not --user rows
    assert ps.cli("lock", "--promote-user", str(bid)) == 0  # duplicate: nothing added
    fired(run(ps, "A190"), 1)


def test_results_carry_no_paths(ps: PlantedStore) -> None:
    for item in TRACEABLE:
        blob = json.dumps(run(ps, item).to_json())
        assert "/" not in blob and "\\" not in blob, item
