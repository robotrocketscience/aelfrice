"""#1887: ingest-extraction census probes against planted stores.

Each test plants a store under `tmp_path` with aelfrice's own code (ingest,
the transcript logger, the scanner, the onboard handshake, commit ingest,
the claude-memory mirror, CLI handlers), then reads it read-only through the
probe. Every traceable probe is also checked on an empty store, and every
probe is run on a pre-aelfrice schema to show it reports instead of raising.

An `aelf` shim that only records its first argument sits first on `PATH`,
so a detached spawn from the logger never reaches a real `aelf`. No test
runs git or calls a model API.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_ingest_extraction as pie
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PROBES: dict[str, cc.Probe] = {p.item_id: p for p in pie.PROBES}
EMPTY_SILENT = sorted(pie.TRACEABLE - pie.UNTRACEABLE_WHEN_EMPTY)

T1 = "We deploy the billing service from the main branch every Tuesday."
T2 = "The staging database runs on port 5433 for every environment."

needs_posix_shim = pytest.mark.skipif(
    sys.platform == "win32", reason="the aelf shim is a POSIX shell script")


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "aelf").write_text(
        '#!/bin/sh\necho "$1" >> "$(dirname "$0")/aelf_calls.log"\nexit 0\n')
    (shim / "aelf").chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "0")
    for var in ("AELFRICE_LOG_REWORDED_SUPPORT", "AELFRICE_AUTO_RELATIONSHIPS",
                "AELFRICE_MIRROR_CLAUDE_MEMORY", "CLAUDE_CODE_ENTRYPOINT",
                "AELFRICE_CAPTURE_PRINT_MODE", "AELF_SESSION_ID"):
        monkeypatch.delenv(var, raising=False)
    s = plant(tmp_path, monkeypatch)
    yield s
    s.close()


def probe(ps: PlantedStore, item: str) -> cc.Result:
    return ps.probe(PROBES[item])


def assert_fired(r: cc.Result, count: int) -> None:
    assert r.status == "fired", r
    assert r.count == count, r


def logger(payload: dict[str, Any]) -> None:
    from aelfrice import transcript_logger as tl
    assert tl.main(stdin=io.StringIO(json.dumps(payload)), stderr=io.StringIO()) == 0


def ups(ps: PlantedStore, prompt: str, sid: str = "s1") -> None:
    logger({"hook_event_name": "UserPromptSubmit", "prompt": prompt,
            "session_id": sid, "cwd": str(ps.repo)})


def stop(ps: PlantedStore, text: str | None, sid: str = "s1") -> None:
    p: dict[str, Any] = {"hook_event_name": "Stop", "session_id": sid, "cwd": str(ps.repo)}
    if text is not None:
        p["last_assistant_message"] = text
    logger(p)


def ingest_logged_turns(ps: PlantedStore) -> None:
    """Ingest the logger's own turns.jsonl, as the Stop flush does."""
    from aelfrice.ingest import ingest_jsonl
    ingest_jsonl(ps.open(), ps.store_dir / "transcripts" / "turns.jsonl")


def shim_calls(ps: PlantedStore, timeout_s: float = 10.0) -> list[str]:
    """First args the aelf shim recorded; the spawn is detached, so poll."""
    log = ps.root / "bin" / "aelf_calls.log"
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if log.is_file() and log.read_text().strip():
            return log.read_text().split()
        time.sleep(0.05)
    return []


def make_project(root: Path) -> Path:
    """A plain folder (no .git, so the scanner runs no git) with 2 doc
    paragraphs and 2 docstrings."""
    proj = root / "proj"
    (proj / "docs").mkdir(parents=True)
    (proj / "docs" / "guide.md").write_text(
        "The billing service reads its settings from the shared config file.\n\n"
        "Every deploy writes a release note into the operations channel.\n")
    (proj / "pkg").mkdir()
    (proj / "pkg" / "mod.py").write_text(
        '"""Parse invoice records from the upstream ledger export."""\n\n'
        "def total(rows):\n"
        '    """Sum the amount column across every parsed invoice row."""\n'
        "    return sum(r for r in rows)\n")
    return proj


def legacy_ctx(tmp_path: Path) -> cc.Ctx:
    """A pre-aelfrice store: beliefs and edges under the predecessor column names."""
    d = tmp_path / "legacy"
    d.mkdir()
    db = d / "memory.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, "
                     "belief_type TEXT, alpha REAL, beta_param REAL, locked INTEGER, "
                     "source_type TEXT)")
        conn.execute("CREATE TABLE edges (from_id TEXT, to_id TEXT, edge_type TEXT)")
        conn.execute("INSERT INTO edges VALUES ('a', 'b', 'supports')")
        conn.commit()
    finally:
        conn.close()
    c = cc.Ctx("legacy01", d, None, False, db, "legacy")
    c.open()
    return c


# --- registry shape --------------------------------------------------------


def test_ids_match_the_subsystem() -> None:
    assert {p.item_id for p in pie.PROBES} == {f"A{i:03d}" for i in range(132, 184)}
    assert len(pie.PROBES) == 52
    assert pie.UNTRACEABLE_WHEN_EMPTY <= pie.TRACEABLE


# --- empty store -----------------------------------------------------------


@pytest.mark.parametrize("item", EMPTY_SILENT)
def test_empty_store_silent(ps: PlantedStore, item: str) -> None:
    r = probe(ps, item)
    assert r.status == "silent", (item, r)
    assert r.count in (0, None), (item, r)


@pytest.mark.parametrize("item", sorted(pie.UNTRACEABLE_WHEN_EMPTY))
def test_empty_store_untraceable_for_join_probes(ps: PlantedStore, item: str) -> None:
    r = probe(ps, item)
    assert r.status == "untraceable", (item, r)


@pytest.mark.parametrize("item", sorted(pie.UNTRACEABLE))
def test_untraceable_probes_report_untraceable(item: str) -> None:
    r = cc.run_one(PROBES[item], None)
    assert r.status == "untraceable" and r.count is None and r.note, (item, r)


# --- pre-aelfrice schema -----------------------------------------------------


@pytest.mark.parametrize("item", sorted(pie.TRACEABLE))
def test_legacy_schema_never_errors(tmp_path: Path, item: str) -> None:
    c = legacy_ctx(tmp_path)
    try:
        r = cc.run_one(PROBES[item], c)
    finally:
        c.close()
    assert r.status != "error", (item, r)
    assert r.status != "fired", (item, r)


def test_legacy_schema_statuses(tmp_path: Path) -> None:
    c = legacy_ctx(tmp_path)
    try:
        got = {i: cc.run_one(PROBES[i], c) for i in
               ("A133", "A140", "A169", "A175", "A179", "A182")}
    finally:
        c.close()
    assert got["A133"].status == "silent" and got["A133"].detail["missing_cols"] == 3
    assert got["A140"].status == "silent" and got["A140"].detail["origin_col"] == 0
    assert got["A169"].status == "silent" and got["A169"].detail == {"legacy_edge_rows": 1}
    assert got["A175"].status == "untraceable"
    assert got["A179"].status == "silent" and "beliefs_total" in got["A179"].detail
    assert got["A182"].status == "silent" and got["A182"].note == "no origin column"


# --- transcript ingest ------------------------------------------------------


def test_transcript_ingest_family(ps: PlantedStore) -> None:
    ps.ingest_turns([
        {"role": "user", "text": T1, "session_id": "s1"},
        {"role": "assistant", "text": "Noted, Tuesday deploys it is.", "session_id": "s1"},
        {"role": "user", "text": T2, "session_id": "s1"},
    ])
    assert_fired(probe(ps, "A170"), 2)
    r = probe(ps, "A182")
    assert_fired(r, 2)
    assert r.detail["ut_factual"] == 2 and r.detail["ut_undeflated"] == 2, r
    assert r.detail["eligible_rows"] == 2 and r.detail["eligible_derived"] == 2, r
    r = probe(ps, "A179")
    assert_fired(r, 2)
    assert r.detail["deflated"] == 0 and r.detail["undeflated"] == 2, r
    r = probe(ps, "A147")
    assert_fired(r, 2)
    assert r.detail == {"factual": 2}
    r = probe(ps, "A178")
    assert_fired(r, 2)
    assert r.detail["transcript_label"] == 2
    r = probe(ps, "A149")
    assert_fired(r, 2)
    assert r.detail == {"ingest": 2}
    r = probe(ps, "A157")
    assert_fired(r, 1)
    assert r.detail["intra_turn_logged"] == 0
    r = probe(ps, "A175")
    # aelf spine backfill writes the same edge.
    assert r.status == "ambiguous" and r.count == 1, r
    assert r.detail["spine_non_transcript"] == 0 and r.detail["eligible"] == 1, r
    assert r.detail["eligible_covered"] == 1 and r.detail["backfill_sentinel"] == 1, r
    assert probe(ps, "A146").status == "silent"
    assert probe(ps, "A166").status == "silent"


def test_a175_silent_when_no_belief_is_eligible(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user", "text": T1, "session_id": "s1"}])
    r = probe(ps, "A175")
    assert r.status == "silent" and r.detail["eligible"] == 0, r
    assert r.detail["transcript_beliefs"] == 1, r


def test_a134_ingest_transcript_command(ps: PlantedStore) -> None:
    turns = ps.root / "in" / "turns.jsonl"
    turns.parent.mkdir()
    with turns.open("w") as f:
        for i, t in enumerate((T1, T2)):
            f.write(json.dumps({"role": "user", "text": t, "session_id": "s1",
                                "ts": f"2026-11-08T00:00:0{i}+00:00"}) + "\n")
    assert ps.cli("ingest-transcript", str(turns)) == 0
    r = probe(ps, "A134")
    assert r.status == "ambiguous" and r.count == 2, r
    assert r.detail == {"transcript_rows": 2, "default_label": 2, "sessions": 1}


def test_a175_is_ambiguous_without_a_backfill_sentinel(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user", "text": T1, "session_id": "s1"},
                     {"role": "user", "text": T2, "session_id": "s1"}])
    ps.close()
    ps.remove_precreated_sentinels()
    r = probe(ps, "A175")
    assert r.status == "ambiguous" and r.count == 1, r
    assert r.detail["backfill_sentinel"] == 0


def test_a146_corroboration(ps: PlantedStore) -> None:
    ps.ingest_turns([
        {"role": "user", "text": T1, "session_id": "s1"},
        {"role": "user", "text": T1, "session_id": "s2"},
    ])
    r = probe(ps, "A146")
    assert_fired(r, 1)
    assert r.detail == {"user": 1}


def test_a166_question_drop(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user", "text": "What port does the staging database use?",
                      "session_id": "s1"}])
    r = probe(ps, "A166")
    assert_fired(r, 1)
    assert r.detail == {"transcript": 1}
    assert probe(ps, "A172").status == "silent"
    r = probe(ps, "A182")
    assert r.status == "silent" and r.note == "", r


def test_a172_speculative_float(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user",
                      "text": "Maybe we could move the billing service to a separate cluster.",
                      "session_id": "s1"}])
    r = probe(ps, "A172")
    assert_fired(r, 1)
    assert r.detail == {"transcript": 1}
    assert probe(ps, "A166").status == "silent"


def test_a156_inline_code_kept(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user",
                      "text": "Run `uv sync --frozen` before you start the integration tests.",
                      "session_id": "s1"}])
    r = probe(ps, "A156")
    assert_fired(r, 1)
    assert r.detail["user_post"] == 1 and r.detail["holed_post"] == 1, r
    assert r.detail["span_in_stored"] == 1, r


def test_a156_silent_when_post_merge_rows_hold_no_span(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user", "text": T1, "session_id": "s1"}])
    r = probe(ps, "A156")
    assert r.status == "silent" and r.detail["user_post"] == 1, r
    assert r.detail["span_in_stored"] == 0 and r.detail["holed"] == 0, r


def test_a174_subfloor_demotion(ps: PlantedStore) -> None:
    ps.ingest_turns([{"role": "user", "session_id": "s1",
                      "text": T1 + "\n\nAcceptance criteria:\n\n" + T2}])
    assert_fired(probe(ps, "A174"), 1)
    r = probe(ps, "A157")
    assert r.status == "silent" and r.detail["intra_turn_logged"] == 1, r


def test_a167_reworded_support(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_LOG_REWORDED_SUPPORT", "1")
    # The matcher needs the user's words in order inside one target clause.
    ps.belief("The billing service reads its settings from the shared config file on startup.",
              source="doc:ops.md:p0", session_id="old")
    # The target must be created strictly before the supporter, and sqlite's
    # julianday() keeps milliseconds only, so the turn carries a later ts.
    ps.ingest_turns([{"role": "user", "session_id": "new", "ts": "2099-01-01T00:00:00+00:00",
                      "text": "The billing service reads settings from the shared config file."}])
    r = probe(ps, "A167")
    assert_fired(r, 1)
    assert r.detail == {"applied": 0}


def test_a169_kept_contradicts_writer_is_not_counted(
        ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_AUTO_RELATIONSHIPS", "1")
    ps.ingest_turns([
        {"role": "user", "session_id": "s1",
         "text": "The staging database always runs on port 5433."},
        {"role": "user", "session_id": "s1",
         "text": "The staging database never runs on port 5433."},
    ])
    r = probe(ps, "A169")
    assert r.status == "silent" and r.count == 0, r
    assert r.detail["contradicts_unanchored_w1"] == 1 and r.detail["unexplained"] == 0, r


def test_a169_reverted_writer_shape_is_ambiguous(ps: PlantedStore) -> None:
    a = ps.belief(T1)
    b = ps.belief(T2)
    ps.edge(a, b, "SUPPORTS")
    r = probe(ps, "A169")
    assert r.status == "ambiguous" and r.count == 1, r
    assert r.detail["supports_null_anchor"] == 1, r


def test_a169_resolver_supersedes_is_explained(ps: PlantedStore) -> None:
    a = ps.belief(T1)
    b = ps.belief(T2)
    ps.edge(a, b, "SUPERSEDES", weight=1.0)
    ps.edge(a, b, "RESOLVES", weight=0.0)
    r = probe(ps, "A169")
    assert r.status == "silent", r
    assert r.detail["supersedes_resolver"] == 1 and r.detail["unexplained"] == 0, r


# --- speaker gate and noise filter, joined to the logger's own turns ---------


def test_a171_assistant_lines_read_and_dropped(ps: PlantedStore) -> None:
    ups(ps, T1)
    stop(ps, "Done, the deploy schedule is recorded for every Tuesday morning.")
    ingest_logged_turns(ps)
    r = probe(ps, "A171")
    assert_fired(r, 1)
    assert r.detail["violation"] == 0 and r.detail["role_user"] == 1, r
    assert r.detail["assistant_lines_in_ingested_sessions"] == 1, r


def test_a171_untraceable_without_an_ingested_session(ps: PlantedStore) -> None:
    ups(ps, T1)
    stop(ps, "Done, the deploy schedule is recorded for every Tuesday morning.")
    r = probe(ps, "A171")
    assert r.status == "untraceable" and r.detail["log_assistant_lines"] == 1, r


def test_a177_noise_sentence_dropped_at_ingest(ps: PlantedStore) -> None:
    ups(ps, T1 + " Standing by.")
    ingest_logged_turns(ps)
    r = probe(ps, "A177")
    assert_fired(r, 1)
    assert r.detail["joined_turns"] == 1 and r.detail["noise_logged"] == 0, r


def test_a177_silent_when_joined_turn_has_no_noise(ps: PlantedStore) -> None:
    ups(ps, T1)
    ingest_logged_turns(ps)
    r = probe(ps, "A177")
    assert r.status == "silent" and r.detail["joined_turns"] == 1, r


# --- onboard scanner and handshake -------------------------------------------


def test_scanner_extractors(ps: PlantedStore) -> None:
    from aelfrice.scanner import scan_repo
    res = scan_repo(ps.open(), make_project(ps.root))
    assert res.inserted == 4, res
    assert_fired(probe(ps, "A150"), 2)
    assert_fired(probe(ps, "A139"), 2)
    r = probe(ps, "A179")
    assert_fired(r, 4)
    assert r.detail["deflated"] == 4 and r.detail["undeflated"] == 0, r
    r = probe(ps, "A149")
    assert_fired(r, 4)
    assert probe(ps, "A151").status == "silent"
    assert probe(ps, "A148").status == "silent"
    assert probe(ps, "A152").status == "silent"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=census", "-c",
                    "user.email=census@example.invalid", "-c", "commit.gpgsign=false",
                    *args], check=True, capture_output=True, timeout=60)


def test_a151_git_log_subjects_are_scanned(ps: PlantedStore) -> None:
    from aelfrice.scanner import scan_repo
    proj = make_project(ps.root)
    _git(proj, "init", "-q")
    _git(proj, "add", "-A")
    _git(proj, "commit", "-q", "-m",
         "The billing service now reads its settings from the shared config file")
    res = scan_repo(ps.open(), proj)
    assert res.inserted >= 1, res
    assert_fired(probe(ps, "A151"), 1)


def test_a171_an_assistant_row_in_ingest_log_is_a_contract_error(ps: PlantedStore) -> None:
    ps.close()
    conn = sqlite3.connect(str(ps.db_file))
    try:
        conn.execute("INSERT INTO ingest_log (id, ts, source_kind, source_path, raw_text, "
                     "raw_meta, session_id) VALUES ('x1', '2026-11-08T00:00:00+00:00', "
                     "'transcript', 'transcript', 'x', ?, 's1')",
                     (json.dumps({"role": "assistant"}),))
        conn.commit()
    finally:
        conn.close()
    r = probe(ps, "A171")
    assert r.status == "error" and r.count == 1, r


def test_a161_scan_rows_make_the_filter_untraceable(ps: PlantedStore) -> None:
    from aelfrice.scanner import scan_repo
    scan_repo(ps.open(), make_project(ps.root))
    r = probe(ps, "A161")
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["scan_rows"] == 4 and r.detail["scan_rows_noise"] == 0, r


def test_a148_llm_router_fallback(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import llm_classifier as llm
    from aelfrice.llm_classifier import ScannerRouter
    from aelfrice.scanner import scan_repo

    def offline(*_: object, **__: object) -> object:
        # _call_with_retries builds the SDK client; a failed build raises this.
        raise llm.LLMTransientError("offline")

    monkeypatch.setattr(llm, "_call_with_retries", offline)
    router = ScannerRouter(api_key="census-no-key", sdk_module=types.ModuleType("fake_sdk"))
    scan_repo(ps.open(), make_project(ps.root), llm_router=router)
    r = probe(ps, "A148")
    assert_fired(r, 4)
    assert r.detail == {"fallback_rows": 4}


def test_a154_a162_handshake(ps: PlantedStore) -> None:
    from aelfrice.classification import (
        HostClassification,
        accept_classifications,
        start_onboard_session,
    )
    store = ps.open()
    start = start_onboard_session(store, make_project(ps.root))
    assert len(start.sentences) == 4
    accept_classifications(store, start.session_id, [
        HostClassification(0, "requirement", True),
        HostClassification(1, "factual", False),
    ])
    r = probe(ps, "A154")
    assert_fired(r, 1)
    assert r.detail == {"completed": 1, "override_rows": 1}
    assert_fired(probe(ps, "A162"), 1)
    assert probe(ps, "A147").status == "silent"  # override rows are not classifier rows
    r = probe(ps, "A161")
    assert r.status == "silent" and r.detail["onboard_cands"] == 4, r


def test_a152_commit_date_created_at(ps: PlantedStore) -> None:
    """The commit date normally comes from git; here it is set on the pending
    session row so no git runs. accept_classifications applies it."""
    from aelfrice.classification import HostClassification, accept_classifications
    from aelfrice.models import OnboardSession
    store = ps.open()
    cands = [
        {"index": 0, "text": "The billing service reads settings from the shared file.",
         "source": "doc:docs/a.md:p0", "commit_date": "2025-01-02T03:04:05Z"},
        {"index": 1, "text": "Every deploy writes a release note into the channel.",
         "source": "doc:docs/a.md:p1", "commit_date": None},
    ]
    store.insert_onboard_session(OnboardSession(
        session_id="census0session0001", repo_path=str(ps.root / "proj"), state="pending",
        candidates_json=json.dumps(cands), created_at="2026-11-08T00:00:00Z",
        completed_at=None))
    accept_classifications(store, "census0session0001", [
        HostClassification(0, "factual", True), HostClassification(1, "factual", True),
    ], now="2026-11-08T01:00:00Z")
    r = probe(ps, "A152")
    assert_fired(r, 1)
    assert r.detail == {"handshake": 1, "scan": 0}


# --- commit ingest -----------------------------------------------------------

COMMIT_MSG = "The new parser supersedes the legacy tokenizer."


def test_commit_ingest_message(ps: PlantedStore) -> None:
    """Shared body of the git hook and the legacy path, without git."""
    from aelfrice.hook_commit_ingest import _ingest_message  # pyright: ignore[reportPrivateUsage]
    ps.close()
    _ingest_message(COMMIT_MSG, session_id="0123456789abcdef", project_context=str(ps.repo))
    r = probe(ps, "A141")
    assert_fired(r, 1)
    assert r.detail["sessions_all"] == 1 and r.detail["sessions_pre_hook"] == 0, r
    assert r.detail["log_commit_all"] == 2 and r.detail["log_git_other"] == 0, r
    r = probe(ps, "A144")
    assert_fired(r, 2)
    assert r.detail == {"beliefs": 2, "edge_supersedes": 1}
    r = probe(ps, "A158")
    assert r.status == "silent" and r.detail["post_510"] == 1, r
    assert probe(ps, "A182").status == "silent"
    r = probe(ps, "A143")
    assert r.status == "untraceable" and r.detail["commit_sessions"] == 1, r
    assert r.detail["named_store"] == 0, r


def test_a158_legacy_era_commit_session(ps: PlantedStore,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A commit ingested while only the PostToolUse path existed (clock pinned)."""
    from aelfrice import hook_commit_ingest as hci
    monkeypatch.setattr(hci, "_iso_now", lambda: "2026-09-01T00:00:00+00:00")
    ps.close()
    hci._ingest_message(  # pyright: ignore[reportPrivateUsage]
        COMMIT_MSG, session_id="fedcba9876543210", project_context=str(ps.repo))
    r = probe(ps, "A158")
    assert_fired(r, 1)
    assert r.detail["pre_cutoff"] == 1 and r.detail["post_510"] == 0, r
    r = probe(ps, "A141")
    assert r.status == "silent" and r.detail["sessions_pre_hook"] == 1, r


# --- claude-memory mirror and reconcile ---------------------------------------

FACT = ("---\nname: deploy day\ndescription: when we deploy\nmetadata:\n  type: user\n---\n"
        "We deploy the billing service every Tuesday morning.\n")
FACT2 = ("---\nname: port\ndescription: staging port\nmetadata:\n  type: project\n---\n"
         "The staging database listens on port 5433.\n")


def test_a140_mirror_hook(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_MIRROR_CLAUDE_MEMORY", "1")
    mdir = ps.home / ".claude" / "projects" / "-census-proj" / "memory"
    mdir.mkdir(parents=True)
    fact = mdir / "deploy_day.md"
    fact.write_text(FACT)
    rc = ps.hook("claude_memory_mirror", {
        "tool_name": "Write", "tool_input": {"file_path": str(fact)},
        "tool_response": {}, "cwd": str(ps.repo)})
    assert rc == 0
    r = probe(ps, "A140")
    assert_fired(r, 1)
    assert r.detail["user_validated"] == 1 and r.detail["sentinel"] == 0, r
    assert r.detail["untimed"] == 1, r
    assert probe(ps, "A164").status == "silent"


def test_a164_reconcile(ps: PlantedStore) -> None:
    from aelfrice.claude_memory import derive_memory_dir, reconcile_sentinel_path
    from aelfrice.claude_memory_reconcile import maybe_reconcile_claude_memory
    proj = ps.root / "proj"
    proj.mkdir()
    mdir = derive_memory_dir(proj)
    assert ps.home in mdir.parents
    mdir.mkdir(parents=True)
    (mdir / "deploy_day.md").write_text(FACT)
    (mdir / "port.md").write_text(FACT2)
    res = maybe_reconcile_claude_memory(
        ps.open(), project_path=proj, sentinel_path=reconcile_sentinel_path(ps.db_file),
        opted_out=False)
    assert res.ran and res.n_ingested == 2, res
    r = probe(ps, "A164")
    assert_fired(r, 1)
    assert r.detail == {"facts": 2}
    r = probe(ps, "A140")
    assert_fired(r, 2)
    assert r.detail["sentinel"] == 1 and r.detail["after_sentinel"] == 0, r


# --- transcript logger hooks --------------------------------------------------


def test_logger_user_and_location(ps: PlantedStore) -> None:
    ups(ps, T1)
    assert_fired(probe(ps, "A159"), 1)
    r = probe(ps, "A176")
    assert_fired(r, 1)
    assert r.detail == {"archive_files": 0}
    assert probe(ps, "A138").status == "silent"


def test_a145_duplicate_guard(ps: PlantedStore) -> None:
    ups(ps, T1)
    ups(ps, T1)
    assert_fired(probe(ps, "A145"), 1)
    assert_fired(probe(ps, "A159"), 1)


def test_a138_assistant_reply(ps: PlantedStore) -> None:
    ups(ps, T1)
    stop(ps, "Done, the deploy schedule is recorded.")
    stop(ps, None, sid="s2")
    r = probe(ps, "A138")
    assert_fired(r, 1)
    assert r.detail == {"empty_stub": 1}


@needs_posix_shim
def test_a165_precompact_rotation(ps: PlantedStore) -> None:
    ups(ps, T1)
    logger({"hook_event_name": "PreCompact", "session_id": "s1", "cwd": str(ps.repo)})
    r = probe(ps, "A165")
    assert_fired(r, 1)
    assert r.detail == {"archive_files": 1}
    assert shim_calls(ps) == ["ingest-transcript"]
    assert_fired(probe(ps, "A159"), 1)  # the user line moved to the archive


@needs_posix_shim
def test_a173_stop_flush(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "2")
    ups(ps, T1)
    stop(ps, "Done, the deploy schedule is recorded.")
    r = probe(ps, "A173")
    assert_fired(r, 1)
    assert r.detail == {"turns_at_last_flush": 2}
    assert shim_calls(ps) == ["ingest-transcript"]


# --- read path and maintenance ---------------------------------------------------


def test_a180_rebuild_log_triples(ps: PlantedStore) -> None:
    from aelfrice.rebuild_log import record_user_prompt_submit_log
    store = ps.open()
    bid = ps.belief(T2, source="doc:ops.md:p0")
    b = store.get_belief(bid)
    assert b is not None
    record_user_prompt_submit_log(
        prompt="The module src/parser.py supersedes the legacy tokenizer design.",
        session_id="s1", hits_pre_dedup=[b], hits_post_dedup=[b],
        log_path=ps.store_dir / "rebuild_logs" / "s1.jsonl", enabled=True)
    r = probe(ps, "A180")
    assert_fired(r, 1)
    assert r.detail == {"rows_with_query": 1}


def test_a133_prune_noise(ps: PlantedStore) -> None:
    ps.belief("What does the parser return on empty input?", source="doc:a.md:p0")
    keep = ps.belief("The parser returns an empty list on empty input.", source="doc:a.md:p1")
    gone = ps.belief("How is the cache invalidated after a deploy?", source="doc:a.md:p2")
    other = ps.belief("The scheduler flushes kumquat batches every night.", source="doc:a.md:p3")
    assert ps.cli("retire", gone) == 0  # user retire writes user_retired; excluded
    ps.open().soft_delete_belief(other)  # another unaudited retire, of text that is not noise
    assert ps.cli("doctor", "--prune-noise", "--apply") == 0
    r = probe(ps, "A133")
    assert r.status == "ambiguous" and r.count == 1, r
    assert r.detail == {"retired_unaudited": 2, "retired_noise_now": 1, "beliefs_rows": 4}
    c = ps.ctx()
    try:
        assert c.count("SELECT COUNT(*) FROM beliefs WHERE id=? AND valid_to IS NULL",
                       (keep,)) == 1
    finally:
        c.close()
