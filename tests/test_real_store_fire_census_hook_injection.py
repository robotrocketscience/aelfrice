"""Planted-store tests for the hook-injection census probes (#1887).

Each test plants a store under `tmp_path` with `tests.fire_census_planted`,
makes the feature fire through aelfrice's own code (hook entry points in a
child process, CLI handlers, `MemoryStore` writes) and checks the probe.
Every traceable probe is also run on an empty store.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
import zlib
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from aelfrice.models import Belief
from benchmarks.real_store_fire_census import core as cc
from benchmarks.real_store_fire_census import probes_hook_injection as ph
from tests.fire_census_planted import PlantedStore, plant

pytestmark = pytest.mark.timeout(300)

PROBES = {p.item_id: p for p in ph.PROBES}
TRACEABLE = sorted(ph.TRACEABLE)
UNTRACEABLE = sorted(ph.UNTRACEABLE)

# Probes that report untraceable, not silent, on an empty store: they need
# an audit row, a ring file or a forward-only log field before they can
# tell fired from silent.
EMPTY_UNTRACEABLE = frozenset({"A077", "A095", "A101", "A107", "A120"})

ORIGINAL_IDS = frozenset(
    ["A077", "A078", "A079", "A080", "A081", "A082", "A083", "A084", "A085", "A086",
     "A087", "A088", "A089", "A090", "A091", "A092", "A093", "A094", "A095", "A096",
     "A097", "A098", "A099", "A100", "A101", "A102", "A103", "A104", "A105", "A106",
     "A107", "A108", "A109", "A110", "A111", "A112", "A113", "A114", "A115", "A116",
     "A117", "A118", "A119", "A120", "A121", "A122", "A123", "A124", "A125", "A126",
     "A127", "A128", "A129", "A130", "A131"])

_CLEAR_ENV = (
    "AELF_AUTOLOCK_CORRECTIONS", "AELFRICE_BELIEF_CATEGORIES", "AELFRICE_CADENCE_ENABLED",
    "AELFRICE_CADENCE_POLICY", "AELFRICE_CADENCE_K", "AELFRICE_CADENCE_CTX_THRESHOLD",
    "AELFRICE_CADENCE_CTX_BYTE_WINDOW", "AELFRICE_CADENCE_SHADOW_MODE_ENABLED",
    "AELFRICE_CADENCE_P3_VELOCITY_THRESHOLD", "AELFRICE_CADENCE_P3_SUBSTANTIVE_WINDOW",
    "AELFRICE_CADENCE_P3_SUBSTANTIVE_THRESHOLD", "AELFRICE_EXPLORATION",
    "AELFRICE_EXPLORATION_CADENCE", "AELFRICE_EXPLORATION_SLOTS", "AELFRICE_PROVENANCE_RENDER",
    "AELFRICE_TURN_DIFFERENTIAL", "AELFRICE_MEMORY_BLOCK", "AELFRICE_HOOK_AUDIT",
    "AELFRICE_REBUILD_LOG", "AELFRICE_PROJECT_CONTEXT", "AELFRICE_RETRIEVAL_TOKEN_BUDGET",
    "AELFRICE_L1_LIMIT", "AELFRICE_SESSION_START_CORE_BUDGET", "AELFRICE_HOOK_BLOCK_CEILING",
    "AELFRICE_AGENT_CONTEXT", "AELF_SESSION_ID", "AELFRICE_SESSIONSTART_RECAP",
    "AELFRICE_CORE_GATE_SESSION_END",
)

TOPIC = ("The kumquat parser normalises citrus manifests before the orchard "
         "scheduler reads them; kumquat batches flush every night.")
LOCK = "Census probes report integer counts and fixed labels only."


@pytest.fixture()
def ps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PlantedStore]:
    s = plant(tmp_path, monkeypatch)
    monkeypatch.setenv("AELF_NO_SIDECAR_WARM", "1")
    monkeypatch.setenv("AELFRICE_WONDER_AUTOGC", "0")
    for k in _CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    yield s
    s.close()


# --- drivers ---------------------------------------------------------------


def ups(ps: PlantedStore, prompt: str, session: str = "s1", cwd: Path | None = None) -> None:
    rc = ps.hook("main", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                          "session_id": session, "cwd": str(cwd or ps.repo)})
    assert rc == 0


def session_start(ps: PlantedStore, session: str = "s1", source: str = "startup") -> None:
    rc = ps.hook("session_start", {"hook_event_name": "SessionStart", "session_id": session,
                                   "source": source, "cwd": str(ps.repo)})
    assert rc == 0


def stop(ps: PlantedStore, session: str = "s1", transcript: Path | None = None) -> None:
    payload: dict[str, Any] = {"hook_event_name": "Stop", "session_id": session,
                               "cwd": str(ps.repo), "stop_hook_active": False}
    if transcript is not None:
        payload["transcript_path"] = str(transcript)
    assert ps.hook("stop", payload) == 0


def write_turns(session: str, texts: list[str]) -> None:
    p = Path(os.environ["AELFRICE_TRANSCRIPTS_DIR"]) / "turns.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"role": "user" if i % 2 == 0 else "assistant", "text": t,
                                "session_id": session,
                                "ts": datetime.now(timezone.utc).isoformat()}) + "\n")


def write_host_transcript(ps: PlantedStore, last_user: str, pad: int) -> Path:
    p = ps.root / "host_transcript.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "assistant", "message": {
            "role": "assistant", "content": "x" * pad}}) + "\n")
        f.write(json.dumps({"type": "user", "message": {
            "role": "user", "content": last_user}}) + "\n")
    return p


def audit(ps: PlantedStore, hook: str = "user_prompt_submit") -> list[dict[str, Any]]:
    return [r for p in cc.jsonl_paths(ps.store_dir / ph.AUDIT_FILENAME)
            for r in cc.iter_jsonl(p) if r.get("hook") == hook]


def run(ps: PlantedStore, item: str) -> cc.Result:
    return ps.probe(PROBES[item])


def seed_topic(ps: PlantedStore, n: int = 3, **kw: Any) -> list[str]:
    return [ps.belief(f"{TOPIC} Note {i} on kumquat parser stage {i}.", **kw) for i in range(n)]


def make_git_repo(root: Path) -> None:
    """A one-commit repository written as loose objects (no git command)."""
    g = root / ".git"
    (g / "objects").mkdir(parents=True)
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "refs" / "tags").mkdir(parents=True)

    def obj(kind: str, body: bytes) -> str:
        data = f"{kind} {len(body)}\0".encode() + body
        sha = hashlib.sha1(data).hexdigest()
        p = g / "objects" / sha[:2] / sha[2:]
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(zlib.compress(data))
        return sha

    tree = obj("tree", b"")
    ident = "Census <census@example.invalid> 1700000000 +0000"
    commit = obj("commit", (f"tree {tree}\nauthor {ident}\ncommitter {ident}\n\n"
                            "fix: planted census subject (#123)\n").encode())
    (g / "refs" / "heads" / "feat-census").write_text(commit + "\n")
    (g / "HEAD").write_text("ref: refs/heads/feat-census\n")
    (g / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = false\n")


def wait_for(pred: Callable[[], bool], timeout_s: float = 60.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.25)
    return bool(pred())


def assert_fired(r: cc.Result, count: int) -> None:
    assert r.status == "fired", r
    assert r.count == count, r


def raw_ctx(db: Path) -> cc.Ctx:
    c = cc.Ctx("store01", db.parent, None, False, db, "legacy")
    c.open()
    return c


# --- module shape and copied vocabularies ----------------------------------


def test_module_ids_match_the_original_set() -> None:
    assert set(PROBES) == ORIGINAL_IDS
    assert len(ph.PROBES) == len(ORIGINAL_IDS)


def test_constants_match_aelfrice() -> None:
    from aelfrice import (
        cadence,
        feed_log,
        hook,
        hook_audit,
        hook_payload,
        hook_search,
        hook_search_tool,
        injection_ledger,
        models,
        provenance_render,
        rebuild_log,
        session_exclusions,
        session_ring,
        sidecar_warm,
        transcript_logger,
    )
    assert ph.AUDIT_FILENAME == hook_audit.AUDIT_FILENAME
    assert ph.COMMAND_OUTCOMES_FILENAME == hook_audit.COMMAND_OUTCOMES_FILENAME
    assert ph.COMMAND_OUTCOME_HOOK == hook_audit.COMMAND_OUTCOME_HOOK
    assert ph.HOOK_UPS == hook.AUDIT_HOOK_USER_PROMPT_SUBMIT
    assert ph.HOOK_SS == hook.AUDIT_HOOK_SESSION_START
    assert ph.HOOK_WARM == sidecar_warm.WARM_AUDIT_HOOK
    assert ph.REBUILD_LOG_DIRNAME == rebuild_log.REBUILD_LOG_DIRNAME
    assert ph.CADENCE_RESUME_CACHE == hook._CADENCE_RESUME_CACHE_FILENAME  # pyright: ignore[reportPrivateUsage]
    assert ph.CADENCE_SHADOW_DIRNAME == cadence.CADENCE_SHADOW_DIRNAME
    assert ph.SESSION_STATE_FILENAME == hook.SESSION_STATE_FILENAME
    assert ph.EXCLUSIONS_FILENAME == session_exclusions.EXCLUSIONS_FILENAME
    assert ph.LEDGER_FILENAME == injection_ledger.LEDGER_FILENAME
    assert ph.RING_FILENAME == session_ring.SESSION_RING_FILENAME
    assert ph.RECAP_LAST_TS_FILENAME == hook._RECAP_LAST_TS_FILENAME  # pyright: ignore[reportPrivateUsage]
    assert ph.FEED_FILENAME == feed_log.FEED_FILENAME
    assert ph.TRANSCRIPTS_DIRNAME == transcript_logger.TRANSCRIPTS_SUBDIR
    assert ph.TURNS_FILENAME == transcript_logger.TURNS_FILENAME
    assert hook.TELEMETRY_SUBPATH.endswith("/".join(ph.UPS_TELEMETRY))
    assert hook_search_tool.TELEMETRY_SUBPATH.endswith("/".join(ph.SEARCH_TELEMETRY))
    assert ph.OPEN_TAG == hook.OPEN_TAG
    assert ph.CLOSE_TAG == hook.CLOSE_TAG
    assert ph.BASELINE_OPEN_TAG == hook.SESSION_START_OPEN_TAG
    assert ph.SESSION_START_OPEN == hook.SESSION_START_SUBBLOCK_OPEN
    assert ph.SESSION_START_CLOSE == hook.SESSION_START_SUBBLOCK_CLOSE
    assert (ph.CORE_OPEN, ph.CORE_CLOSE) == (hook.CORE_OPEN_TAG, hook.CORE_CLOSE_TAG)
    assert ph.RECENT_WORK_OPEN == hook.RECENT_WORK_OPEN_TAG
    assert ph.RESUME_OPEN == hook.RESUME_OPEN_TAG
    assert hook._FRAMING_HEADER.startswith(ph.FRAMING_HEAD)  # pyright: ignore[reportPrivateUsage]
    assert hook._SPECULATIVE_FRAMING_SENTENCE.startswith(  # pyright: ignore[reportPrivateUsage]
        ph.SPECULATIVE_SENTENCE_HEAD)
    assert hook._cap_belief_content("x" * 2000).endswith(  # pyright: ignore[reportPrivateUsage]
        ph.TRUNCATION_MARKER)
    assert ph.PROVENANCE_SECTIONS == (provenance_render.SECTION_LOCKED,
                                      provenance_render.SECTION_OBSERVED,
                                      provenance_render.SECTION_INFERRED)
    assert ph.HOOK_PAYLOAD_CHAR_LIMIT == hook_payload.HOOK_PAYLOAD_CHAR_LIMIT
    assert ph.LOCK_OVERFLOW_RE.search(hook_payload.lock_overflow_line(["b1"]))
    assert ph.SESSION_START_CORE_BUDGET == hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET
    assert ph.HOOK_TOKEN_BUDGET == hook.DEFAULT_HOOK_TOKEN_BUDGET
    assert ph.HOOK_BLOCK_TOKEN_CEILING == hook.HOOK_BLOCK_TOKEN_CEILING
    assert ph.MAX_CONV_AWARE_WEIGHT == hook.MAX_CONV_AWARE_WEIGHT
    assert ph.CADENCE_RESUME_TTL_S == hook._CADENCE_RESUME_TTL_SECONDS  # pyright: ignore[reportPrivateUsage]
    assert ph.RECAP_THRESHOLD == hook._DEFAULT_RECAP_THRESHOLD  # pyright: ignore[reportPrivateUsage]
    assert ph.FEEDBACK_SOURCE_HOOK == hook_search.HOOK_FEEDBACK_SOURCE
    assert ph.LOCK_INGEST_KINDS == (models.INGEST_SOURCE_CLI_REMEMBER,
                                    models.INGEST_SOURCE_MCP_REMEMBER)
    assert ph.POLICIES == (cadence.POLICY_P1_EVERY_K_TURNS, cadence.POLICY_P2_CTX_THRESHOLD,
                           cadence.POLICY_P3_VELOCITY, cadence.POLICY_P3_SUBSTANTIVE)
    assert ph.DEFAULT_P2_WATERMARK == int(cadence.DEFAULT_CTX_THRESHOLD
                                          * cadence.DEFAULT_CTX_BYTE_WINDOW)
    assert ph.DEFAULT_P3_VELOCITY_THRESHOLD == cadence.DEFAULT_P3_VELOCITY_THRESHOLD
    assert ph.DEFAULT_P3_SUBSTANTIVE_WINDOW == cadence.DEFAULT_P3_SUBSTANTIVE_WINDOW
    assert ph.DEFAULT_P3_SUBSTANTIVE_THRESHOLD == f"{cadence.DEFAULT_P3_SUBSTANTIVE_THRESHOLD:.3f}"
    assert ph.REBUILDER_TURN_WINDOW == rebuild_log.RebuilderConfig().turn_window_n
    assert set(ph.COMMAND_REASONS) == {r.value for r in hook.CommandReason}
    assert set(ph._RECAP_EVENTS) == set(hook._RECAP_BELIEF_WRITE_EVENTS)  # pyright: ignore[reportPrivateUsage]


def test_gate_and_query_shapes_match_aelfrice() -> None:
    from aelfrice import hook
    from aelfrice.retrieval import seen_manifest_line
    for prompt, lab in (("ok", "trivial:short"), ("<system-reminder> hello there", "system-tag"),
                        ("keep going", "trivial:short"), ("please do that thing", None)):
        skip, reason = hook._should_skip_bm25(prompt)  # pyright: ignore[reportPrivateUsage]
        got = ph._gate_label({"prompt_shape_gate_skip": reason}) if skip else None  # pyright: ignore[reportPrivateUsage]
        assert got == lab
    turn = type("T", (), {"text": "earlier turn"})()
    composed = hook._build_conversation_aware_query("fix the parser", [turn])  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert ph._composed_weight(composed) is not None  # pyright: ignore[reportPrivateUsage]
    assert ph._composed_weight("fix the parser") is None  # pyright: ignore[reportPrivateUsage]
    # A prompt that opens with a space is detected (the original scan stopped at offset 0).
    assert ph.composed_end(" a b  a b  a b", 3) is not None
    b = Belief(id="b_x", content="topic text", content_hash="h", alpha=1.0, beta=1.0,
               type="factual", lock_level="none", locked_at=None, created_at="t",
               last_retrieved_at=None, session_id=None, origin="agent_inferred")
    assert ph.SEEN_LINE_RE.search("\n  " + seen_manifest_line(b))
    assert ph.CORE_LINE_RE.match(hook._core_belief_line(b))  # pyright: ignore[reportPrivateUsage]


# --- empty store -----------------------------------------------------------


@pytest.mark.parametrize("item", TRACEABLE)
def test_empty_store(ps: PlantedStore, item: str) -> None:
    r = run(ps, item)
    if item in EMPTY_UNTRACEABLE:
        assert r.status == "untraceable" and r.count is None, r
    else:
        assert r.status == "silent", r
        assert r.count in (0, None), r


@pytest.mark.parametrize("item", UNTRACEABLE)
def test_untraceable_probes_report_untraceable(ps: PlantedStore, item: str) -> None:
    r = run(ps, item)
    assert r.status == "untraceable" and r.count is None and r.note, r


# --- planted fires ---------------------------------------------------------


def test_a077_core_section(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    for i in range(3):
        ps.belief(f"Core rule {i}: the orchard scheduler flushes kumquat batches nightly.",
                  alpha=4.0, beta=1.0)
    ups(ps, "what does the orchard scheduler do with kumquat batches")
    r = run(ps, "A077")
    assert_fired(r, 1)
    assert r.detail["core_beliefs"] == 3
    assert r.detail["post_cap_core_rows"] == 1


def test_a077_first_prompt_without_core_is_silent(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    ups(ps, "census probes report integer counts")
    r = run(ps, "A077")
    assert r.status == "silent", r
    assert r.detail["session_start_rows"] == 1


def test_a078_core_budget_near_cap(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    pad = "lorem ipsum dolor sit amet " * 14
    for i in range(40):
        ps.belief(f"Qualifying core belief {i} about granary ledgers. {pad}", alpha=4.0, beta=1.0)
    ups(ps, "census probes report integer counts")
    r = run(ps, "A078")
    assert_fired(r, 1)
    assert r.detail["over_cap_rows"] == 0


def test_a079_recent_work(ps: PlantedStore) -> None:
    gitrepo = ps.root / "gitrepo"
    make_git_repo(gitrepo)
    ps.lock(LOCK)
    ups(ps, "census probes report integer counts", cwd=gitrepo)
    r = run(ps, "A079")
    assert_fired(r, 1)
    assert r.detail["rows_with_commits"] == 1


def test_a081_autolock(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELF_AUTOLOCK_CORRECTIONS", "1")
    ps.lock(LOCK)  # a CLI lock: must not be counted
    ps.belief("Use uv run for every python command instead of pip.", type="correction",
              session_id="s1")
    ps.belief("Planted agent memory about the granary schedule.", session_id="s1")
    stop(ps, "s1")
    r = run(ps, "A081")
    assert_fired(r, 2)
    assert r.detail["type_correction"] == 1


def test_a082_joins_telemetry_to_audit_rows(ps: PlantedStore) -> None:
    for i in range(12):
        assert ps.lock(f"Granary rule {i}: " + "keep the granary ledger exact. " * 32) == 0
    pad = "the granary ledger is checked twice before the flush window closes " * 4
    for i in range(40):
        ps.belief(f"Granary ledger note {i}: {pad}")
    ups(ps, "what are the granary ledger rules")
    r = run(ps, "A082")
    assert r.detail["ups_rows"] == 1 and r.detail["joined"] == 1, r
    assert r.detail["trim_fires"] == 1, r
    # Under the payload bound the 6000 constant cannot be the binding limit.
    assert r.status == "silent", r


def test_a083_category_focus(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice import category as catmod
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    bid = ps.belief("Run the smoke suite before every deploy to staging.")
    st = ps.open()
    st.upsert_category(name="deploy", always_on=False,
                       trigger_json=catmod.CategoryTrigger(keywords=("deploy",)).to_json(),
                       default_lock=catmod.normalize_default_lock(None))
    st.assign_belief_to_category(bid, "deploy")
    ups(ps, "how should we deploy the service to staging today")
    r = run(ps, "A083")
    assert_fired(r, 1)
    assert r.detail["categories"] == 1


def test_a084_sidecar_warm(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AELF_NO_SIDECAR_WARM", raising=False)
    seed_topic(ps)
    ps.lock(LOCK)
    session_start(ps, "s1")
    assert wait_for(lambda: len(audit(ps, ph.HOOK_WARM)) >= 1)
    assert_fired(run(ps, "A084"), 1)


def _cadence_env(monkeypatch: pytest.MonkeyPatch, policy: str, **extra: str) -> None:
    monkeypatch.setenv("AELFRICE_CADENCE_ENABLED", "1")
    monkeypatch.setenv("AELFRICE_CADENCE_POLICY", policy)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_a085_cadence_p1(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    _cadence_env(monkeypatch, "p1_every_k_turns", AELFRICE_CADENCE_K="1")
    seed_topic(ps)
    ps.lock(LOCK)
    write_turns("s1", ["Tell me how the kumquat parser flushes batches",
                       "The kumquat parser flushes nightly."])
    ups(ps, "how does the kumquat parser flush batches")
    stop(ps, "s1")
    r = run(ps, "A085")
    assert_fired(r, 1)
    assert r.detail == {"resume_cache": 1, "shadow_fired_rows": 0}
    assert run(ps, "A087").status == "silent"
    # The Stop-side cache write is an A091 fire on its own.
    r = run(ps, "A091")
    assert_fired(r, 1)
    assert r.detail["cache_policy_p1_every_k_turns"] == 1 and r.detail["recap_rows"] == 0


def test_a086_a087_a092_cadence_p2_shadow(ps: PlantedStore,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    _cadence_env(monkeypatch, "p2_ctx_threshold", AELFRICE_CADENCE_CTX_BYTE_WINDOW="1000",
                 AELFRICE_CADENCE_CTX_THRESHOLD="0.5",
                 AELFRICE_CADENCE_SHADOW_MODE_ENABLED="1")
    seed_topic(ps)
    write_turns("s1", ["Tell me how the kumquat parser flushes batches",
                       "The kumquat parser flushes nightly."])
    tp = write_host_transcript(ps, "thanks", 800)
    stop(ps, "s1", tp)
    assert_fired(run(ps, "A086"), 1)
    r = run(ps, "A087")
    assert_fired(r, 2)
    assert r.detail == {"resume_cache": 1, "shadow_fired_rows": 1}
    r = run(ps, "A092")
    assert_fired(r, 1)
    assert r.detail["fired_rows"] == 1


def test_a088_a090_cadence_p3_substantive(ps: PlantedStore,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    _cadence_env(monkeypatch, "p3_substantive", AELFRICE_CADENCE_P3_SUBSTANTIVE_WINDOW="1",
                 AELFRICE_CADENCE_P3_SUBSTANTIVE_THRESHOLD="0.5",
                 AELFRICE_CADENCE_SHADOW_MODE_ENABLED="1")
    seed_topic(ps)
    write_turns("s1", ["Tell me how the kumquat parser flushes batches",
                       "The kumquat parser flushes nightly."])
    tp = write_host_transcript(
        ps, "Please refactor the kumquat parser so nested manifests flush in order.", 200)
    stop(ps, "s1", tp)
    r = run(ps, "A088")
    assert r.status == "fired", r
    assert r.detail["resume_cache"] == 1
    r = run(ps, "A090")
    assert_fired(r, 1)
    assert r.detail == {"velocity_rows": 0, "substantive_rows": 1}


def test_a089_a090_cadence_p3_velocity(ps: PlantedStore,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _cadence_env(monkeypatch, "p3_velocity", AELFRICE_CADENCE_P3_VELOCITY_THRESHOLD="100",
                 AELFRICE_CADENCE_SHADOW_MODE_ENABLED="1")
    seed_topic(ps)
    ps.lock(LOCK)
    write_turns("s1", ["Tell me how the kumquat parser flushes batches",
                       "The kumquat parser flushes nightly."])
    ups(ps, "how does the kumquat parser flush batches")
    tp = write_host_transcript(ps, "and the orchard scheduler too please", 600)
    stop(ps, "s1", tp)
    r = run(ps, "A089")
    assert_fired(r, 2)
    assert r.detail == {"resume_cache": 1, "shadow_fired_rows": 1}
    r = run(ps, "A090")
    assert_fired(r, 1)
    assert r.detail == {"velocity_rows": 1, "substantive_rows": 0}


def test_a091_cadence_resume(ps: PlantedStore) -> None:
    from aelfrice import hook
    ps.lock(LOCK)
    hook._write_cadence_resume_cache(  # pyright: ignore[reportPrivateUsage]
        "planted rebuild synthesis body", "s0", "p1_every_k_turns", sys.stderr)
    ups(ps, "census probes report integer counts", session="s2")
    r = run(ps, "A091")
    assert_fired(r, 2)
    assert r.detail["cache_file"] == 1 and r.detail["recap_rows"] == 1


def test_a093_collapse_duplicate_hashes(ps: PlantedStore) -> None:
    (ps.repo / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\ncollapse_duplicate_hashes = true\n")
    text = "Granary ledgers reconcile on the first of each month."
    assert ps.cli("lock", "--user", text) == 0
    ps.belief(text, source="repo-copy")
    ups(ps, "when do granary ledgers reconcile each month")
    r = run(ps, "A093")
    assert_fired(r, 1)
    assert r.detail["dropped_duplicates"] == 1


def test_a095_conversation_aware_query(ps: PlantedStore) -> None:
    seed_topic(ps)
    write_turns("s1", ["We are refactoring the kumquat parser module today",
                       "I updated the kumquat parser flush order."])
    ups(ps, "what about that parser change now")
    r = run(ps, "A095")
    assert_fired(r, 1)
    assert r.detail["composed_w3"] == 1 and r.detail["composed_with_turns"] == 1


def test_a095_prompt_only_is_silent(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "what about that parser change now")
    r = run(ps, "A095")
    assert r.status == "silent", r
    assert r.detail["prompt_only"] == 1


def test_a096_a101_coverage_line(ps: PlantedStore) -> None:
    pad = "the manifest is checked twice before the flush window closes " * 4
    for i in range(80):
        ps.belief(f"Kumquat parser note {i}: {pad}")
    ups(ps, "kumquat parser notes on the flush window")
    assert_fired(run(ps, "A096"), 1)
    r = run(ps, "A101")
    assert_fired(r, 1)
    assert r.detail["ups_blocks"] == 1 and r.detail["budget_rows"] == 1
    assert r.detail["cov_after_close"] == 1 and r.detail["cov_inside"] == 0


def test_a101_budget_applies_without_coverage_line(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    assert run(ps, "A096").status == "silent"
    r = run(ps, "A101")
    assert_fired(r, 1)
    assert r.detail["cov_after_close"] == 0


def test_a101_silent_when_a_ups_block_packs_only_locks(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    ups(ps, "what do census probes report")
    r = run(ps, "A101")
    assert r.status == "silent" and r.count == 0, r
    assert r.detail["ups_rows"] >= 1 and r.detail["budget_rows"] == 0


def test_a097_exploration(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    monkeypatch.setenv("AELFRICE_EXPLORATION_CADENCE", "1")
    pad = "the manifest is checked twice before the flush window closes " * 4
    for i in range(80):
        ps.belief(f"Kumquat parser note {i}: {pad}")
    ups(ps, "kumquat parser notes on the flush window")
    assert_fired(run(ps, "A097"), 1)


def test_ups_writers_a098_a100_a102_a127_a129_a130(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    rows = audit(ps)
    assert len(rows) == 1
    n = int(rows[0]["n_beliefs"])
    assert n >= 2
    r = run(ps, "A098")
    assert_fired(r, n)
    assert r.detail["distinct_beliefs"] == n
    assert_fired(run(ps, "A100"), n)
    r = run(ps, "A102")
    assert_fired(r, n)
    assert r.detail["with_consumers"] == 0 and r.detail["ref_pos"] == 0
    assert r.detail["ref_unscored"] == n and r.detail["rows_post_510"] == n
    assert r.detail["ups_block_fires"] == 1
    r = run(ps, "A127")
    assert_fired(r, 1)
    assert r.detail == {"ups": 1, "session_start": 0}
    r = run(ps, "A129")
    assert_fired(r, 1)
    assert r.detail["suppressed_with_hits"] == 0
    assert_fired(run(ps, "A130"), 1)


def test_a102_block_without_writer_rows_is_an_error(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    st = ps.open()
    st._conn.execute("DELETE FROM injection_events")  # pyright: ignore[reportPrivateUsage]
    st._conn.commit()  # pyright: ignore[reportPrivateUsage]
    r = run(ps, "A102")
    assert r.status == "error" and r.note == "writer missed", r


def test_a129_off_switch_is_not_counted(ps: PlantedStore,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    ps.lock(LOCK)
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A129")
    assert r.status == "silent", r
    assert r.detail["suppressed_with_hits"] == 1
    assert run(ps, "A098").status == "silent"
    assert run(ps, "A102").status == "silent"


def test_a099_session_start_locked(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A099")
    assert_fired(r, 1)
    assert r.detail["locked_elements"] == 1
    ups(ps, "and what about the orchard scheduler")
    assert_fired(run(ps, "A099"), 1)


def test_a103_render_cap(ps: PlantedStore) -> None:
    ps.belief("Kumquat parser flush detail. " + "manifest " * 250)
    ups(ps, "kumquat parser flush detail")
    r = run(ps, "A103")
    assert_fired(r, 1)
    assert r.detail["truncated_elements"] == 1


def _compact_scene(ps: PlantedStore, lock: bool = True) -> None:
    if lock:
        ps.lock(LOCK)
    write_turns("s1", ["Tell me how the KumquatParser flushes Orchard batches",
                       "The KumquatParser flushes Orchard batches nightly."])
    session_start(ps, "s1", source="compact")


def test_a105_compact_rebuild(ps: PlantedStore) -> None:
    seed_topic(ps)
    _compact_scene(ps)
    r = run(ps, "A105")
    assert_fired(r, 1)
    assert r.detail["rebuilder_rows"] == 1 and r.detail["audit_only_fires"] == 1


def test_a105_lockless_compact_seen_through_marker(ps: PlantedStore) -> None:
    seed_topic(ps)
    _compact_scene(ps, lock=False)
    assert ps.hook("transcript_logger", {"hook_event_name": "PostCompact", "session_id": "s1",
                                         "cwd": str(ps.repo)}) == 0
    r = run(ps, "A105")
    assert_fired(r, 1)
    assert r.detail["compact_ss_audit_rows"] == 0 and r.detail["marker_matched_fires"] == 1


def test_a105_startup_is_not_counted(ps: PlantedStore) -> None:
    seed_topic(ps)
    ps.lock(LOCK)
    write_turns("s1", ["Tell me how the kumquat parser flushes batches"])
    session_start(ps, "s1", source="startup")
    assert run(ps, "A105").status == "silent"


def test_a111_rebuild_floor(ps: PlantedStore) -> None:
    for i in range(12):
        ps.belief(f"KumquatParser Orchard entry {i} in the cellar ledger.")
    _compact_scene(ps)
    r = run(ps, "A111")
    assert_fired(r, 1)
    assert r.detail["l1_floored"] >= 1


def test_a112_rebuilder_budget(ps: PlantedStore) -> None:
    pad = "a session belief that the rebuilder packs by session membership " * 6
    for i in range(60):
        ps.belief(f"Session note {i}: {pad}", session_id="s1")
    _compact_scene(ps)
    r = run(ps, "A112")
    assert_fired(r, 1)
    assert r.detail["dropped_by_budget"] >= 1


def test_a113_query_rewrite(ps: PlantedStore) -> None:
    (ps.repo / ".aelfrice.toml").write_text('[rebuilder]\nquery_strategy = "stack-r1-r3"\n')
    seed_topic(ps)
    _compact_scene(ps)
    r = run(ps, "A113")
    assert_fired(r, 1)
    assert r.detail["diff_post_500"] == 1 and r.detail["diff_stack_shaped"] == 1


def test_a113_legacy_is_silent(ps: PlantedStore) -> None:
    seed_topic(ps)
    _compact_scene(ps)
    r = run(ps, "A113")
    assert r.status == "silent", r
    assert r.detail["rows"] == 1 and r.detail["diff"] == 0


def test_a106_bash_search(ps: PlantedStore) -> None:
    seed_topic(ps)
    assert ps.hook("search_tool", {"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                   "tool_input": {"command": "grep -rn kumquat src"},
                                   "session_id": "s1", "cwd": str(ps.repo)}) == 0
    r = run(ps, "A106")
    assert_fired(r, 1)
    assert r.detail["grep"] == 1


def test_a107_grep_lane_orphan_ring_ids(ps: PlantedStore) -> None:
    ps.belief("Granary ledgers reconcile on the first of each month.")
    seed_topic(ps)
    ups(ps, "when do granary ledgers reconcile each month")
    assert ps.hook("search_tool", {"hook_event_name": "PreToolUse", "tool_name": "Grep",
                                   "tool_input": {"pattern": "kumquat parser"},
                                   "session_id": "s1", "cwd": str(ps.repo)}) == 0
    r = run(ps, "A107")
    assert r.status == "fired", r
    assert r.detail["orphan_ids"] >= 1 and r.detail["bash_tel_inj"] == 0
    assert r.count == r.detail["orphan_ids"], r


def test_a107_ups_only_ring_is_untraceable(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A107")
    assert r.status == "untraceable" and r.note == "no orphan ring ids", r


def test_a109_prompt_shape_gate(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "ok")
    r = run(ps, "A109")
    assert_fired(r, 1)
    assert r.detail == {"trivial:short": 1}


def test_a110_provenance_render(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_PROVENANCE_RENDER", "1")
    ps.lock(LOCK)
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    assert_fired(run(ps, "A110"), 1)


def test_a117_scope_out(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    assert ps.cli("scope-out", "kumquat") == 0
    r = run(ps, "A117")
    assert_fired(r, 1)
    assert r.detail["bound_to_current_session"] == 1


def test_a118_session_window(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches", session="s1")
    assert run(ps, "A118").status == "silent"
    ups(ps, "how does the kumquat parser flush batches", session="s2")
    assert_fired(run(ps, "A118"), 2)


def test_a119_baseline(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    session_start(ps, "s1")
    r = run(ps, "A119")
    assert_fired(r, 1)
    assert r.detail == {"startup": 1}


def test_a120_single_session_start_is_untraceable(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    session_start(ps, "s1")
    r = run(ps, "A120")
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["last_ts_file"] == 1 and r.detail["inner_gaps"] == 0
    assert r.detail["inner_gaps_ge3"] == 0


def _recap_gap(ps: PlantedStore, locks: int) -> cc.Result:
    ps.lock(LOCK)  # an empty store's SessionStart emits no block, so no audit row
    session_start(ps, "s1")
    time.sleep(1.1)
    for i in range(locks):
        assert ps.lock(f"Recap rule {i}: granary ledgers are reconciled monthly.") == 0
    time.sleep(1.1)
    session_start(ps, "s2")
    return run(ps, "A120")


def test_a120_gap_at_the_threshold_is_replayed(ps: PlantedStore) -> None:
    r = _recap_gap(ps, ph.RECAP_THRESHOLD)
    assert r.status == "untraceable" and r.count is None, r
    assert r.detail["inner_gaps_ge3"] == 1
    assert r.detail["max_inner_gap"] == ph.RECAP_THRESHOLD


def test_a120_gap_below_the_threshold_is_not_counted(ps: PlantedStore) -> None:
    r = _recap_gap(ps, ph.RECAP_THRESHOLD - 1)
    assert r.status == "untraceable", r
    assert r.detail["inner_gaps_ge3"] == 0
    assert r.detail["max_inner_gap"] == ph.RECAP_THRESHOLD - 1


def test_a120_recap_off_still_replays_a_gap(
        ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the recap off no line prints, yet the feed rows are the same:
    why the probe reports untraceable rather than fired."""
    monkeypatch.setenv("AELFRICE_SESSIONSTART_RECAP", "0")
    r = _recap_gap(ps, ph.RECAP_THRESHOLD)
    assert r.status == "untraceable", r
    assert r.detail["inner_gaps_ge3"] == 1
    assert r.detail["last_ts_file"] == 0


def test_a121_speculative(ps: PlantedStore) -> None:
    ps.belief(TOPIC + " Speculative join of two parser beliefs.", origin="speculative")
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A121")
    assert_fired(r, 1)
    assert r.detail["live_phantoms"] == 1
    assert r.detail["phantom_rows"] == 1 and r.detail["phantom_rows_no_sentence"] == 0


def _legacy_db(tmp_path: Path, cols: str, row: tuple[Any, ...]) -> Path:
    db = tmp_path / "legacy" / "memory.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(f"CREATE TABLE beliefs ({cols})")
        conn.execute(f"INSERT INTO beliefs VALUES ({','.join('?' * len(row))})", row)
        conn.commit()
    finally:
        conn.close()
    return db


def test_a121_pre_v1_schema_is_silent(tmp_path: Path) -> None:
    db = _legacy_db(tmp_path, "id TEXT, content TEXT, belief_type TEXT", ("b1", "x", "factual"))
    c = raw_ctx(db)
    try:
        r = cc.run_one(PROBES["A121"], c)
    finally:
        c.close()
    assert r.status == "silent" and r.detail["beliefs_rows"] == 1, r


def test_a123_pre_582_schema_is_untraceable(tmp_path: Path) -> None:
    db = _legacy_db(tmp_path, "id TEXT, content TEXT, type TEXT", ("b1", "x", "factual"))
    c = raw_ctx(db)
    try:
        r = cc.run_one(PROBES["A123"], c)
    finally:
        c.close()
    assert r.status == "untraceable" and r.note == "pre-582 schema", r


def test_a123_stop_without_batch_is_untraceable(ps: PlantedStore) -> None:
    ps.belief("Planted agent memory about the granary schedule.", session_id="s1")
    stop(ps, "s1")
    r = run(ps, "A123")
    assert r.status == "untraceable", r
    assert r.detail["would_fire_sessions"] == 1 and r.detail["stop_runs_proven"] == 0


def test_a123_session_end_batch_proves_listing(ps: PlantedStore,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_CORE_GATE_SESSION_END", "1")
    ps.belief("Planted agent memory about the granary schedule.", session_id="s1",
              alpha=4.0, beta=1.0)
    ps.belief("Use uv run for every python command instead of pip.", type="correction",
              session_id="s1", alpha=4.0, beta=1.0)
    time.sleep(1.1)
    stop(ps, "s1")
    r = run(ps, "A123")
    assert_fired(r, 1)
    assert r.detail["stop_runs_proven"] == 1 and r.detail["candidates_at_runs_sum"] >= 1


def test_a124_agent_context(ps: PlantedStore) -> None:
    other = ps.belief("Granary ledgers reconcile on the first of each month.")
    ps.injection(other, session_id="s0")
    seed_topic(ps)
    assert ps.hook("agent_context", {
        "hook_event_name": "PreToolUse", "tool_name": "Agent", "session_id": "s1",
        "tool_input": {"prompt": "Investigate how the kumquat parser flushes batches"},
        "cwd": str(ps.repo)}) == 0
    r = run(ps, "A124")
    assert_fired(r, 1)
    assert r.detail["exposure_rows"] >= 1


def test_a124_ups_exposures_are_not_counted(ps: PlantedStore) -> None:
    seed_topic(ps)
    ups(ps, "how does the kumquat parser flush batches")
    time.sleep(1.2)
    ups(ps, "and how does the kumquat parser order batches")
    assert run(ps, "A098").status == "fired"
    assert run(ps, "A124").status == "silent"


def test_a125_turn_differential(ps: PlantedStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_TURN_DIFFERENTIAL", "1")
    seed_topic(ps)
    session_start(ps, "s1")
    ups(ps, "how does the kumquat parser flush batches")
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A125")
    assert_fired(r, 1)
    assert r.detail["ledger_file"] == 1


def test_a125_first_prompt_seen_pointers_are_not_counted(ps: PlantedStore) -> None:
    ps.lock(LOCK)
    seed_topic(ps, alpha=4.0, beta=1.0)
    ups(ps, "how does the kumquat parser flush batches")
    rows = audit(ps)
    assert len(rows) == 1 and ph.SEEN_LINE_RE.search(rows[0]["rendered_block"])
    ups(ps, "how does the kumquat parser flush batches")
    r = run(ps, "A125")
    assert r.status == "silent", r
    assert r.detail["ledger_file"] == 0


def test_a128_typed_lock(ps: PlantedStore) -> None:
    ups(ps, "/aelf:lock Never deploy on Fridays without a rollback plan")
    r = run(ps, "A128")
    assert_fired(r, 1)
    assert r.detail == {"ok": 1}


def test_a131_lock_overflow(ps: PlantedStore) -> None:
    for i in range(12):
        assert ps.lock(f"Standing rule {i}: " + "keep the granary ledger exact. " * 32) == 0
    ups(ps, "what are the granary ledger rules")
    r = run(ps, "A131")
    assert_fired(r, 1)
    assert r.detail["rows_over_limit"] == 0
