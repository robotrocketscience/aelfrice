"""The #1795 real-store A/B harness: sample, run, packet, score, and proxy.

Every store and transcript is built from synthetic text in a temp directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shlex
import shutil
import sqlite3
import sys
from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

import pytest

import aelfrice.hook as hook_mod
from aelfrice import hrr_expand
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.retrieval import ENV_HRR_EXPAND, last_lane_telemetry
from aelfrice.store import MemoryStore, read_only_uri
from benchmarks import graded_gate, relevance_scale
from benchmarks import real_store_ab_1795 as bench

_QUERY = "kestrel meadow"
_SNAP = "s0"
"""A stand-in store snapshot for key rows and packets built in memory."""


class _approx:
    """Float equality within 1e-9, typed for the strict checker."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            other, self.expected, abs_tol=1e-9)

    def __hash__(self) -> int:
        return hash(self.expected)


@pytest.fixture(autouse=True)
def pinned_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "proc-cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv(ENV_HRR_EXPAND, raising=False)


def _add(
    s: MemoryStore, bid: str, content: str, *,
    lock: str = LOCK_NONE, created: str = "2026-10-01T00:00:00Z",
    expires: str | None = None,
) -> None:
    s.insert_belief(Belief(
        id=bid, content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=lock,
        locked_at=None if lock == LOCK_NONE else created,
        created_at=created, last_retrieved_at=None, lock_expires_at=expires,
    ))


def _filler(i: int) -> str:
    return " ".join(f"zq{i}{c}" for c in "abcde")


def _store(tmp_path: Path) -> Path:
    """A store under `<tmp>/repo/aelfrice/` with writer pairs, HRR neighbors,
    a lock, and 40 filler beliefs no query reaches."""
    path = tmp_path / "repo" / "aelfrice" / "memory.db"
    path.parent.mkdir(parents=True)
    s = MemoryStore(str(path))
    _add(s, "w1", "the deployment script always runs the database migration step")
    _add(s, "w2", "the deployment script never runs the database migration step")
    _add(s, "w3", "python deps are fast")
    _add(s, "w4", "python deps are slow")
    _add(s, "w5", "the release checklist covers the changelog entry for each version",
         created="2026-09-01T00:00:00Z")
    _add(s, "w6", "the release checklist covers the changelog entry for every version")
    _add(s, "seed", "the kestrel hovers above the meadow before it dives")
    _add(s, "con", "owls nest in old barns at dusk")
    _add(s, "rel", "otters swim in cold rivers at dawn")
    _add(s, "lock1", "always run the test suite before pushing", lock=LOCK_USER)
    for i in range(40):
        _add(s, f"f{i:02d}", _filler(i))
    s.insert_edge(Edge(src="seed", dst="con", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="seed", dst="rel", type=EDGE_RELATES_TO, weight=1.0))
    s.close()
    return path


def _turn(role: str, text: str, sid: str, n: int) -> dict[str, Any]:
    return {"schema_version": 1, "ts": f"2026-10-0{1 + n // 10}T00:00:{n % 10:02d}Z",
            "role": role, "text": text, "session_id": sid, "turn_id": f"t{n:03d}",
            "context": {}}


def _write_transcripts(store: Path, archive: list[dict[str, Any]],
                       live: list[dict[str, Any]]) -> None:
    tdir = store.parent / "transcripts"
    (tdir / "archive").mkdir(parents=True)
    marker = {"schema_version": 1, "ts": "2026-10-01T00:00:00Z", "event": "compaction_start"}
    (tdir / "archive" / "turns-20261001T000000Z.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [marker, *archive]), encoding="utf-8")
    (tdir / "turns.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in live) + "not json\n", encoding="utf-8")


def _session_turns() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive = [
        _turn("user", "how does the deployment script handle the migration step", "s1", 0),
        _turn("assistant", "It runs the database migration step first.", "s1", 1),
        _turn("user", "ok", "s1", 2),
        _turn("user", "<task-notification>done</task-notification>", "s1", 3),
        _turn("user", "what does the kestrel do above the meadow", "s1", 4),
        _turn("assistant", "Per seed, the kestrel hovers then dives.", "s1", 5),
    ]
    live = [
        _turn("user", "how does the deployment script handle the migration step", "s2", 6),
        _turn("user", "tell me about the release checklist and the changelog", "s2", 7),
        _turn("assistant", "The release checklist covers the changelog entry.", "s2", 8),
    ] + [_turn("user", f"question number {i} about python deps speed", "s3", 10 + i)
         for i in range(8)]
    return archive, live


# --- Sample ------------------------------------------------------------------


def test_transcripts_are_read_archive_first_in_line_order(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    # A stray binary file in the archive is read past, not fatal.
    (path.parent / "transcripts" / "archive" / "zz.bin").write_bytes(b"\xff\xfe\x00\n")
    turns = bench.read_turns(bench.transcript_files(path))
    assert [t.turn_id for t in turns] == [r["turn_id"] for r in archive + live]
    assert [t.seq for t in turns] == list(range(len(turns)))


def test_the_sample_drops_what_the_production_gate_drops_and_dedupes(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    turns = bench.read_turns(bench.transcript_files(path))
    rows = bench.sample_prompts(turns, seed=1, max_n=40)
    texts = [r["text"] for r in rows]
    assert "ok" not in texts
    assert not any(t.startswith("<task-notification>") for t in texts)
    assert texts.count("how does the deployment script handle the migration step") == 1
    assert rows[0]["session_id"] == "s1" and rows[0]["turn_id"] == "t000"
    users = [t for t in turns if t.role == "user"]
    assert len(rows) == len({t.text for t in users if not hook_mod._should_skip_bm25(t.text)[0]})  # pyright: ignore[reportPrivateUsage]


def test_the_sample_reuses_the_hook_gate_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bench._should_skip_bm25 is hook_mod._should_skip_bm25  # pyright: ignore[reportPrivateUsage]
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    turns = bench.read_turns(bench.transcript_files(path))
    def skip_all(_text: str) -> tuple[bool, str | None]:
        return True, "x"

    monkeypatch.setattr(bench, "_should_skip_bm25", skip_all)
    with pytest.raises(bench.InputError):
        bench.sample_prompts(turns, seed=1, max_n=40)


def test_the_sample_is_seeded_and_in_turn_order(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    turns = bench.read_turns(bench.transcript_files(path))
    a = bench.sample_prompts(turns, seed=5, max_n=4)
    assert a == bench.sample_prompts(turns, seed=5, max_n=4)
    assert len(a) == 4
    assert [r["seq"] for r in a] == sorted(r["seq"] for r in a)
    picks = {tuple(r["prompt_id"] for r in bench.sample_prompts(turns, seed=s, max_n=4))
             for s in range(6)}
    assert len(picks) > 1


def test_the_context_is_the_sessions_earlier_turns(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    rows = bench.sample_prompts(bench.read_turns(bench.transcript_files(path)), seed=1, max_n=40)
    kestrel = next(r for r in rows if r["text"].startswith("what does the kestrel"))
    assert hook_mod.DEFAULT_CONV_AWARE_WINDOW == 4
    assert kestrel["context"] == [
        {"role": "user", "text": "how does the deployment script handle the migration step"},
        {"role": "assistant", "text": "It runs the database migration step first."},
        {"role": "user", "text": "ok"},
        {"role": "user", "text": "<task-notification>done</task-notification>"},
    ]
    last = next(r for r in rows if r["text"] == "question number 7 about python deps speed")
    assert len(last["context"]) == 4
    assert last["context"][0]["text"] == "question number 3 about python deps speed"
    checklist = next(r for r in rows if r["text"].startswith("tell me about"))
    assert [c["text"] for c in checklist["context"]] == [
        "how does the deployment script handle the migration step"]


def test_sample_cli_writes_the_rows(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    out = tmp_path / "prompts.jsonl"
    assert bench.main(["sample", "--store", str(path), "--seed", "3", "--out", str(out)]) == 0
    rows = bench.read_prompts(out)
    assert {"prompt_id", "text", "session_id", "turn_id", "ts", "seq", "context"} <= set(rows[0])


def test_sample_takes_at_most_40_prompts_per_store_by_default(tmp_path: Path) -> None:
    # #1795 registers a seeded sample of at most 40 prompts per store.
    path = _store(tmp_path)
    live = [_turn("user", f"question number {i} about python deps speed", "s1", i)
            for i in range(45)]
    _write_transcripts(path, [], live)
    out = tmp_path / "prompts.jsonl"
    assert bench.main(["sample", "--store", str(path), "--seed", "3", "--out", str(out)]) == 0
    assert len(bench.read_prompts(out)) == 40


# --- Run: arms and isolation -------------------------------------------------


def _prompts() -> list[dict[str, Any]]:
    return [{"prompt_id": "p1", "text": _QUERY, "context": []}]


def _run(path: Path, tmp_path: Path, arm: str) -> list[str]:
    work = tmp_path / f"work-{arm}"
    work.mkdir(exist_ok=True)
    return bench.run_arm(path, _prompts(), arm, work)["prompts"][0]["ranked"]


def test_the_env_override_reaches_the_hrr_expand_lane_through_retrieve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `retrieve()` passes use_hrr_expand=False; the resolver is env-first.
    path = _store(tmp_path)
    store = MemoryStore(str(path), read_only=True)
    try:
        off = bench.retrieve_packed(store, _QUERY)
        off_lane = last_lane_telemetry().hrr_expand
        monkeypatch.setenv(ENV_HRR_EXPAND, "1")
        on = bench.retrieve_packed(store, _QUERY)
        on_lane = last_lane_telemetry().hrr_expand
    finally:
        store.close()
    assert off[0] == "seed" and "con" not in off and off_lane == 0
    assert "con" in on and on_lane == 1


def _direct(path: Path, query: str) -> list[str]:
    store = MemoryStore(str(path), read_only=True)
    try:
        return bench.retrieve_packed(store, query)
    finally:
        store.close()


def test_a0_ignores_ambient_env_and_project_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _store(tmp_path)
    query = "the deployment script migration release checklist"
    prompt: list[dict[str, Any]] = [{"prompt_id": "p", "text": query, "context": []}]

    def a0() -> list[str]:
        return bench.run_arm(path, prompt, bench.ARM_A0, tmp_path)["prompts"][0]["ranked"]

    clean = a0()
    # Each ambient setting below changes what the same call returns.
    (Path.cwd() / ".aelfrice.toml").write_text("[retrieval]\nuse_bm25f_anchors = false\n")
    assert _direct(path, query) != clean
    (Path.cwd() / ".aelfrice.toml").unlink()
    monkeypatch.setenv(ENV_HRR_EXPAND, "1")
    assert _direct(path, query) != clean
    (Path.cwd() / ".aelfrice.toml").write_text("[retrieval]\nuse_bm25f_anchors = false\n")
    cwd = Path.cwd()
    assert a0() == clean
    assert os.environ[ENV_HRR_EXPAND] == "1"
    assert Path.cwd() == cwd


def test_a0_ignores_a_project_config_above_the_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The empty config in the work dir stops discovery before an ancestor's.
    path = _store(tmp_path)
    query = "the deployment script migration release checklist"
    prompt: list[dict[str, Any]] = [{"prompt_id": "p", "text": query, "context": []}]
    work = tmp_path / "project" / "work"
    work.mkdir(parents=True)

    def a0() -> list[str]:
        return bench.run_arm(path, prompt, bench.ARM_A0, work)["prompts"][0]["ranked"]

    clean = a0()
    (work / ".aelfrice.toml").unlink(missing_ok=True)
    (tmp_path / "project" / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_bm25f_anchors = false\n")
    cwd = Path.cwd()
    monkeypatch.chdir(work)
    assert _direct(path, query) != clean
    monkeypatch.chdir(cwd)
    assert a0() == clean


def test_the_arm_config_file_is_named_by_the_discovery_constant(tmp_path: Path) -> None:
    # The repo rule keeps the config file's literal name out of a module
    # with an upward directory loop, and write_packet walks the key path's
    # parents.
    work = tmp_path / "w"
    work.mkdir()
    with bench.arm_environment(bench.ARM_A0, work):
        assert (work / CONFIG_FILENAME).read_text(encoding="utf-8") == ""
    assert CONFIG_FILENAME not in Path(bench.__file__).read_text(encoding="utf-8")


def test_a1_writes_edges_on_the_copy_and_expands(tmp_path: Path) -> None:
    path = _store(tmp_path)
    before = _edges(path)
    work = tmp_path / "work-A1"
    work.mkdir()
    out = bench.run_arm(path, _prompts(), bench.ARM_A1, work)
    assert out["writers"]["contradicts"]["n_edges_written"] >= 1
    ranked = out["prompts"][0]["ranked"]
    assert "con" in ranked and "rel" not in ranked
    assert _edges(path) == before


def test_the_writers_edges_reach_a1s_retrieval(tmp_path: Path) -> None:
    # Only the edges the writers add join w1 to w2 (CONTRADICTS) and w3 to
    # w4 (SUPPORTS), so the lane reaches w2 and w4 through them alone.
    path = _store(tmp_path)
    assert not {e[:2] for e in _edges(path)} & {("w1", "w2"), ("w3", "w4")}
    prompts: list[dict[str, Any]] = [{"prompt_id": "c", "text": "always", "context": []},
                                     {"prompt_id": "s", "text": "fast", "context": []}]
    ranked: dict[str, list[list[str]]] = {}
    for arm in (bench.ARM_A1W, bench.ARM_A1):
        work = tmp_path / f"reach-{arm}"
        work.mkdir()
        ranked[arm] = [p["ranked"] for p in bench.run_arm(path, prompts, arm, work)["prompts"]]
    assert ranked[bench.ARM_A1W] == [["w1"], ["w3"]]
    assert ranked[bench.ARM_A1] == [["w1", "w2"], ["w3", "w4"]]


_MIXED = "deployment script database migration release checklist changelog version kestrel meadow"


def test_a1w_runs_the_writers_with_the_lane_off(tmp_path: Path) -> None:
    # The writers' edges reorder default retrieval through the clustering
    # pack, so A1W, not A0, is the control that isolates the lane.
    path = _store(tmp_path)
    prompt: list[dict[str, Any]] = [{"prompt_id": "p", "text": _MIXED, "context": []}]
    ranked: dict[str, list[str]] = {}
    for arm in (bench.ARM_A0, bench.ARM_A1W, bench.ARM_A1):
        work = tmp_path / f"mix-{arm}"
        work.mkdir()
        out = bench.run_arm(path, prompt, arm, work)
        ranked[arm] = out["prompts"][0]["ranked"]
        assert ("writers" in out) is (arm != bench.ARM_A0)
        if arm == bench.ARM_A1W:
            assert last_lane_telemetry().hrr_expand == 0
    assert ranked[bench.ARM_A1W] != ranked[bench.ARM_A0]
    assert "con" not in ranked[bench.ARM_A1W] and "con" in ranked[bench.ARM_A1]


def test_a2_probes_the_existing_edge_kinds_and_restores_the_override(tmp_path: Path) -> None:
    path = _store(tmp_path)
    ranked = _run(path, tmp_path, bench.ARM_A2)
    assert "rel" in ranked and "con" not in ranked
    assert hrr_expand.probe_kinds_override is None
    assert bench.PROBE_KINDS_A2 == ("DERIVED_FROM", "RELATES_TO", "TEMPORAL_NEXT")


def test_every_arm_retrieves_through_a_read_only_handle_on_an_unchanged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #1795 opens every copy `mode=ro` for retrieval, so no arm's ranking
    # can write the copy or read posteriors rewritten before it ran.
    path = _store(tmp_path)
    conn = sqlite3.connect(read_only_uri(str(path)), uri=True)
    try:
        source = conn.execute("SELECT id, alpha, beta FROM beliefs ORDER BY id").fetchall()
    finally:
        conn.close()
    real = bench.retrieve_packed
    seen: list[str] = []

    def spy(store: MemoryStore, query: str) -> list[str]:
        db = store._conn  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("UPDATE beliefs SET alpha = alpha")
        rows = db.execute("SELECT id, alpha, beta FROM beliefs ORDER BY id").fetchall()
        assert [tuple(r) for r in rows] == source
        seen.append(query)
        return real(store, query)

    monkeypatch.setattr(bench, "retrieve_packed", spy)
    for arm in bench.ARMS:
        work = tmp_path / f"work-{arm}"
        work.mkdir()
        bench.run_arm(path, _prompts(), arm, work)
    assert len(seen) == len(bench.ARMS)


def _crowded_store(tmp_path: Path, n: int) -> Path:
    """`n` beliefs that match `_QUERY`, each with a RELATES_TO edge to `rel`."""
    path = tmp_path / "crowd" / "memory.db"
    path.parent.mkdir(parents=True)
    s = MemoryStore(str(path))
    _add(s, "rel", "otters swim in cold rivers at dawn")
    for i in range(n):
        _add(s, f"k{i:02d}", f"the kestrel {i} hovers above the meadow note{i}")
        s.insert_edge(Edge(src=f"k{i:02d}", dst="rel", type=EDGE_RELATES_TO, weight=1.0))
    for i in range(20):
        _add(s, f"f{i:02d}", _filler(i))
    s.close()
    return path


def test_a_lane_addition_past_the_top_ten_is_judged_and_counted(tmp_path: Path) -> None:
    # L1 fills the top 10, so A2's lane appends `rel` past the cut.
    path = _crowded_store(tmp_path, 12)
    runs: dict[str, Any] = {}
    for arm in (bench.ARM_A0, bench.ARM_A2):
        work = tmp_path / f"crowd-{arm}"
        work.mkdir()
        runs[arm] = bench.run_arm(path, _prompts(), arm, work)
    a0, a2 = (runs[a]["prompts"][0]["ranked"] for a in (bench.ARM_A0, bench.ARM_A2))
    assert "rel" not in a0 and a2.index("rel") >= bench.TOP_K
    assert a2[:bench.TOP_K] == a0[:bench.TOP_K]
    work = tmp_path / "crowd-pk"
    work.mkdir()
    _packet_rows, key = bench.build_packet(runs, bench.prepare_copy(path, work), 1,
                                           snapshot=_SNAP)
    by = {r["belief_id"]: r for r in _items(key) if r["arms"]}
    assert by["rel"]["arms"] == {"A2": a2.index("rel") + 1}
    # Both arms packed the beliefs past the cut, so they cancel unjudged.
    assert set(by) == set(a0[:bench.TOP_K]) | {"rel"}
    # The null arm never draws a belief an arm packed, judged or not.
    for seed in range(10):
        _rows, k = bench.build_packet(runs, work / "memory.db", seed, snapshot=_SNAP)
        assert not {r["belief_id"] for r in _items(k) if r["arms"] is None} & set(a2)
    # Graded relevant: A2 gains `rel` on every prompt, all of it past the cut.
    # The 2026-10-23 amendment to #1795 decides on the packed gain, so H3
    # passes, and the top-10 gain of 0 is only reported.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A2": k + 1}, True, k < 2, k < 2)
                 for k in range(10)]
        rows.append((f"r{p}", f"p{p}", {"A2": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    rep = _score(rows)
    g = rep["arms"]["A2"]["graders"][0]
    assert g["mean_gain"] == _approx(1.0)
    assert g["ci95"] == [_approx(1.0), _approx(1.0)]
    assert g["share_gaining"] == _approx(1.0)
    assert g["top10_mean_gain"] == _approx(0.0)
    assert g["top10_ci95"] == [_approx(0.0), _approx(0.0)]
    assert g["top10_share_gaining"] == _approx(0.0)
    assert g["gain_verdict"] == bench.V_PASS
    assert rep["hypotheses"]["H3"] == bench.V_PASS
    assert g["precision_drop"] == _approx(0.0)
    assert not any(k.startswith("packed_") for k in g)


def test_the_top_ten_gain_decides_nothing() -> None:
    # A2 moves A0's relevant 11th belief into its top 10 and leaves out
    # A0's relevant 12th: a top-10 gain of 1 that clears every bar, and a
    # packed gain of -1, which fails.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A2": k + 1}, True, k < 2, k < 2)
                 for k in range(9)]
        rows.append((f"o{p}", f"p{p}", {"A0": 10, "A2": 11}, True, False, False))
        rows.append((f"m{p}", f"p{p}", {"A0": 11, "A2": 10}, True, True, True))
        rows.append((f"g{p}", f"p{p}", {"A0": 12}, True, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    rep = _score(rows)
    g = rep["arms"]["A2"]["graders"][0]
    assert g["top10_mean_gain"] == _approx(1.0)
    assert g["top10_ci95"] == [_approx(1.0), _approx(1.0)]
    assert g["mean_gain"] == _approx(-1.0)
    assert rep["hypotheses"]["H3"] == bench.V_FAIL


def test_h4_counts_additions_to_the_packed_result() -> None:
    # Per prompt, A3 moves A0's rank-11 belief into its top 10 and packs a
    # bridging belief at rank 11. Only the second is new to the packed result.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A3": k + 1}, True, k < 2, k < 2)
                 for k in range(9)]
        rows.append((f"o{p}", f"p{p}", {"A0": 10}, True, False, False))
        rows.append((f"m{p}", f"p{p}", {"A0": 11, "A3": 10}, True, True, True))
        rows.append((f"b{p}", f"p{p}", {"A3": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    g = _score(rows)["arms"]["A3"]["graders"][0]
    assert g["mean_gain"] == _approx(1.0)
    assert (g["relevant_additions"], g["bridging"]) == (4, 4)


def test_a_belief_a0_also_packed_is_no_a3_addition() -> None:
    # A3 moves A0's 10th belief and its 11th to the top. Both are in A0's
    # packed result, so neither is an addition.
    rows: list[Row] = []
    for p in range(4):
        rows.append((f"t{p}", f"p{p}", {"A0": 10, "A3": 1}, False, True, True))
        rows.append((f"e{p}", f"p{p}", {"A0": 11, "A3": 2}, False, True, True))
    g = _score(rows)["arms"]["A3"]["graders"][0]
    assert (g["relevant_additions"], g["bridging"]) == (0, 0)
    assert g["bridging_share"] is None


def test_the_proxy_sign_counts_the_whole_packed_result() -> None:
    # A1's only used belief sits at rank 11, which the graders' gain counts.
    unused = [{"id": f"u{k}", "used": False} for k in range(bench.TOP_K)]
    proxy = {"prompts": [{"arms": {
        "A0": unused, "A1": [*unused, {"id": "x", "used": True}]}}]}
    assert bench.proxy_gain_signs(proxy, ["A1"]) == {"A1": 1}
    # Both arms use their 10th belief, and A0 also its 11th, so A0 uses more.
    nine = unused[:bench.TOP_K - 1]
    used = [{"id": f"y{k}", "used": True} for k in range(2)]
    proxy = {"prompts": [{"arms": {"A0": [*nine, *used], "A1": [*nine, used[0]]}}]}
    assert bench.proxy_gain_signs(proxy, ["A1"]) == {"A1": -1}


def test_a0_takes_the_top_ten_non_locked_beliefs(tmp_path: Path) -> None:
    path = _store(tmp_path)
    ranked = bench.run_arm(
        path, [{"prompt_id": "p", "text": "always run the test suite before pushing",
                "context": []}], bench.ARM_A0, tmp_path)["prompts"][0]["ranked"]
    assert "lock1" not in ranked


def test_the_judged_top_of_each_run_is_ten(tmp_path: Path) -> None:
    # #1795: the judged set is the top 10 non-locked beliefs.
    assert bench.TOP_K == 10
    path = _crowded_store(tmp_path, 12)
    work = tmp_path / "top"
    work.mkdir()
    run = bench.run_arm(path, _prompts(), bench.ARM_A0, work)
    ranked = run["prompts"][0]["ranked"]
    assert len(ranked) == 12
    _rows, key = bench.build_packet({bench.ARM_A0: run}, work / "memory.db", 1,
                                    snapshot=_SNAP)
    assert {r["belief_id"] for r in _items(key) if r["arms"]} == set(ranked[:10])


def test_the_run_mirrors_the_hook_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _store(tmp_path)
    seen: list[dict[str, Any]] = []
    real = bench.retrieve

    def spy(store: MemoryStore, query: str, **kw: Any) -> Any:
        seen.append({"query": query, **kw})
        return real(store, query, **kw)

    monkeypatch.setattr(bench, "retrieve", spy)
    prompts: list[dict[str, Any]] = [
        {"prompt_id": "p", "text": _QUERY,
         "context": [{"role": "assistant", "text": "otters"}]},
        # No earlier turn: the hook retrieves on the bare prompt.
        {"prompt_id": "q", "text": _QUERY, "context": []},
    ]
    bench.run_arm(path, prompts, bench.ARM_A0, tmp_path)
    kwargs = {
        "token_budget": hook_mod.DEFAULT_HOOK_TOKEN_BUDGET,
        "manifest_reference_locks": True,
        "belief_cost_fn": hook_mod._ups_belief_line_cost,  # pyright: ignore[reportPrivateUsage]
        "user_locks": (),
    }
    assert seen == [
        {"query": f"{_QUERY} {_QUERY} {_QUERY} otters", **kwargs},
        {"query": _QUERY, **kwargs},
    ]


# --- A1 writers -------------------------------------------------------------


def _edges(db: Path) -> set[tuple[str, str, str]]:
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        return {(str(a), str(b), str(t)) for a, b, t in
                conn.execute("SELECT src, dst, type FROM edges").fetchall()}
    finally:
        conn.close()


def test_the_a1_writers_produce_each_edge_type(tmp_path: Path) -> None:
    path = _store(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(path, work)
    report = bench.apply_writers(db)
    edges = _edges(db)
    assert ("w1", "w2", EDGE_CONTRADICTS) in edges
    assert ("w3", "w4", EDGE_SUPPORTS) in edges
    assert ("w6", "w5", EDGE_SUPERSEDES) in edges  # newer member -> oldest
    assert report["supports"]["n_edges_written"] >= 1
    assert report["supersedes"]["n_edges_written"] >= 1
    again = bench.apply_writers(db)
    assert again["supports"]["n_edges_written"] == 0
    assert again["supersedes"]["n_edges_written"] == 0


def test_the_restored_writers_stop_at_eight_edges_per_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ten REFINES pairs share f00, on alternate sides, and an 11-member
    # cluster has f00 as its oldest member. The cap lets 8 edges reach f00
    # and gates the other 2: one with f00 on each side.
    others = [f"f{i:02d}" for i in range(1, 11)]

    def refines(_store: MemoryStore, **_kw: Any) -> Any:
        return SimpleNamespace(n_candidate_pairs=10, truncated=False, pairs=[
            SimpleNamespace(label=bench.LABEL_REFINES,
                            belief_a_id="f00" if k % 2 == 0 else b,
                            belief_b_id=b if k % 2 == 0 else "f00")
            for k, b in enumerate(others)])

    def cluster(_store: MemoryStore, **_kw: Any) -> Any:
        return SimpleNamespace(n_candidate_pairs=55, truncated=False,
                               clusters=[SimpleNamespace(member_ids=["f00", *others])])

    monkeypatch.setattr(bench, "relationships_audit", refines)
    monkeypatch.setattr(bench, "dedup_audit", cluster)
    assert bench.DEFAULT_MAX_EDGES_PER_BELIEF == 8
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    store = MemoryStore(str(db))
    try:
        supports = bench.write_supports_edges(store)
        supersedes = bench.write_supersedes_edges(store)
    finally:
        store.close()
    for report in (supports, supersedes):
        assert (report["n_edges_written"], report["n_edges_skipped_gated"]) == (8, 2)
    edges = _edges(db)
    assert {b for a, b, t in edges if a == "f00" and t == EDGE_SUPPORTS} == set(others[:8])
    assert {a for a, b, t in edges if b == "f00" and t == EDGE_SUPERSEDES} == set(others[:8])


def test_the_writer_reports_carry_the_audits_truncated_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `pair_cap` keeps a real pass from truncating, so stubs set the flag.
    def refines(_store: MemoryStore, **_kw: Any) -> Any:
        return SimpleNamespace(n_candidate_pairs=0, truncated=True, pairs=[])

    def cluster(_store: MemoryStore, **_kw: Any) -> Any:
        return SimpleNamespace(n_candidate_pairs=0, truncated=True, clusters=[])

    monkeypatch.setattr(bench, "relationships_audit", refines)
    monkeypatch.setattr(bench, "dedup_audit", cluster)
    work = tmp_path / "w"
    work.mkdir()
    store = MemoryStore(str(bench.prepare_copy(_store(tmp_path), work)))
    try:
        assert bench.write_supports_edges(store)["truncated"] is True
        assert bench.write_supersedes_edges(store)["truncated"] is True
    finally:
        store.close()


def test_the_writers_reach_pairs_past_the_audit_cap(tmp_path: Path) -> None:
    # 110 near-duplicates hold 5995 candidate pairs, over the audits' default
    # cap of 5000. Pairs sort by id before the cut, so a capped pass gives
    # the later `z` beliefs no edge at all.
    path = tmp_path / "cap" / "memory.db"
    path.parent.mkdir(parents=True)
    s = MemoryStore(str(path))
    for i in range(110):
        _add(s, f"a{i:03d}",
             f"the deployment script runs the database migration step on host number {i}")
    for i in range(20):
        _add(s, f"z{i:03d}",
             f"python deps install fast on the shared build runner in the east region {i}")
    _add(s, "zz0", "the release script always tags the build")
    _add(s, "zz1", "the release script never tags the build")
    s.close()
    report = bench.apply_writers(path)
    assert report["supports"]["truncated"] is False
    assert report["supersedes"]["truncated"] is False
    late = {t for a, b, t in _edges(path) if a.startswith("z") or b.startswith("z")}
    assert late == {EDGE_SUPPORTS, EDGE_SUPERSEDES, EDGE_CONTRADICTS}


def test_the_writers_name_the_commits_that_removed_them() -> None:
    # 01d4dcfb and 6c630223 deleted the writer functions; c3a500fa only
    # dropped their ingest wiring.
    assert "01d4dcfb" in (bench.write_supports_edges.__doc__ or "")
    assert "6c630223" in (bench.write_supersedes_edges.__doc__ or "")
    assert "removed in c3a500fa" not in (bench.__doc__ or "")


def test_census_counts_probed_edges_per_100_beliefs(tmp_path: Path) -> None:
    path = _store(tmp_path)
    before = bench.census([path], tmp_path, writers=False)
    row = before["stores"][0]
    assert row["beliefs"] == 50
    assert row["probed_edges"] == 1  # the CONTRADICTS edge; RELATES_TO isn't probed
    assert row["probed_per_100_beliefs"] == _approx(2.0)
    assert before["H1"] == bench.V_NOT_MEASURED
    after = bench.census([path, path], tmp_path, writers=True)
    assert after["stores"][0]["probed_edges"] > 1
    assert after["H1"] == bench.V_PASS


# --- A3: PMI ------------------------------------------------------------------


def _pmi_texts() -> list[str]:
    """40 beliefs. `kestrel` is in 4, with `falconry` or `falconries` in all
    4, `aglet` in 3, and `common` in all 4; `common` is in 8 in all and
    `the` in every one."""
    texts = [
        "the kestrel falconry aglet note0 common",
        "the kestrel falconry aglet note1 common",
        "the kestrel falconries aglet note2 common",
        "the kestrel falconries hood common",
    ]
    texts += [f"the common {_filler(i)}" for i in range(4)]
    texts += [f"the {_filler(i)}" for i in range(4, 36)]
    return texts


def test_the_pmi_constants_are_pinned() -> None:
    assert (bench.PMI_MIN_COUNT, bench.PMI_MAX_DF_SHARE, bench.PMI_TOP_N_PER_TERM,
            bench.PMI_MAX_ADDED) == (3, 0.10, 2, 6)


def test_pmi_neighbors_rank_by_pmi_then_co_count_under_the_df_cap() -> None:
    exp = bench.PMIExpander(_pmi_texts())
    # PMI(kestrel, falconri) = PMI(kestrel, aglet) = log 10, and the larger
    # co-count breaks the tie. PMI(kestrel, common) = log 5 > 0, but
    # `common` is in 8 of 40 beliefs, over the 10% cap.
    rows = exp.neighbors("kestrel")
    assert [r[0] for r in rows] == ["falconri", "aglet"]
    assert _approx(math.log(10)) == rows[0][1] and rows[0][2] == 4
    assert exp.neighbors("the") == [] and exp.neighbors("common") == []


def _pmi_pair_texts(both: int) -> list[str]:
    """400 beliefs: `alpha` and `beta` in 40 each, together in `both`."""
    texts = [f"alpha beta {_filler(i)}" for i in range(both)]
    texts += [f"alpha {_filler(i)}" for i in range(both, 40)]
    texts += [f"beta {_filler(i)}" for i in range(40, 80 - both)]
    return texts + [_filler(i) for i in range(80 - both, 400)]


def test_pmi_neighbors_need_a_positive_pmi() -> None:
    # PMI = log(400 * 4 / (40 * 40)) = log 1 = 0 exactly: no neighbor.
    assert bench.PMIExpander(_pmi_pair_texts(4)).neighbors("alpha") == []
    rows = bench.PMIExpander(_pmi_pair_texts(5)).neighbors("alpha")
    assert [(r[0], r[2]) for r in rows] == [("beta", 5)]
    assert _approx(math.log(1.25)) == rows[0][1]


def test_pmi_expansion_writes_each_stem_as_its_commonest_surface_word() -> None:
    exp = bench.PMIExpander(_pmi_texts())
    # `falconry` and `falconries` tie at 2 beliefs each; the smaller word wins.
    assert exp.expand("where is the kestrel") == ["falconries", "aglet"]
    assert exp.expand("the") == []


def test_pmi_expansion_is_byte_reproducible_and_order_free() -> None:
    texts = _pmi_texts()
    prompts = ["where is the kestrel", "falconry aglet", "nothing matches here"]
    want = json.dumps([bench.PMIExpander(texts).expand(p) for p in prompts])
    for seed in range(5):
        shuffled = list(texts)
        random.Random(seed).shuffle(shuffled)
        assert json.dumps([bench.PMIExpander(shuffled).expand(p) for p in prompts]) == want


def test_a3_appends_the_pmi_words_to_the_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _store(tmp_path)
    def fixed(_self: bench.PMIExpander, _prompt: str) -> list[str]:
        return ["otters"]

    monkeypatch.setattr(bench.PMIExpander, "expand", fixed)
    seen: list[str] = []
    real = bench.retrieve

    def spy(store: MemoryStore, query: str, **kw: Any) -> Any:
        seen.append(query)
        return real(store, query, **kw)

    monkeypatch.setattr(bench, "retrieve", spy)
    out = bench.run_arm(path, _prompts(), bench.ARM_A3, tmp_path)
    assert seen == [f"{_QUERY} otters"]
    assert out["prompts"][0]["added_terms"] == ["otters"]
    assert "rel" in out["prompts"][0]["ranked"]


def test_a3_expands_the_conversation_aware_query(tmp_path: Path) -> None:
    # #1795 registers A3 as a PMI expansion of the query terms, and the
    # query holds the earlier turns. Only the context names `kestrel`.
    path = tmp_path / "pmi.db"
    s = MemoryStore(str(path))
    for i, text in enumerate(_pmi_texts()):
        _add(s, f"b{i:02d}", text)
    s.close()
    prompt: dict[str, Any] = {"prompt_id": "p", "text": "where does it nest",
                              "context": [{"role": "user", "text": "the kestrel"}]}
    assert bench.PMIExpander(_pmi_texts()).expand(prompt["text"]) == []
    row = bench.run_arm(path, [prompt], bench.ARM_A3, tmp_path)["prompts"][0]
    assert row["added_terms"] == ["falconries", "aglet"]


def test_a3_without_pmi_words_ranks_like_a0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A3 changes one thing from A0, the PMI words. With none to add, A3 must
    # rank like A0. The expand lane would add `con` on this store.
    def none(_self: bench.PMIExpander, _prompt: str) -> list[str]:
        return []

    monkeypatch.setattr(bench.PMIExpander, "expand", none)
    path = _store(tmp_path)
    a0 = _run(path, tmp_path, bench.ARM_A0)
    assert a0 == ["seed"]
    assert _run(path, tmp_path, bench.ARM_A3) == a0


def test_only_a3_adds_pmi_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed(_self: bench.PMIExpander, _prompt: str) -> list[str]:
        return ["otters"]

    monkeypatch.setattr(bench.PMIExpander, "expand", fixed)
    seen: list[str] = []
    real = bench.retrieve

    def spy(store: MemoryStore, query: str, **kw: Any) -> Any:
        seen.append(query)
        return real(store, query, **kw)

    monkeypatch.setattr(bench, "retrieve", spy)
    path = _store(tmp_path)
    others = [a for a in bench.ARMS if a != bench.ARM_A3]
    for arm in others:
        work = tmp_path / f"only-{arm}"
        work.mkdir()
        row = bench.run_arm(path, _prompts(), arm, work)["prompts"][0]
        assert "added_terms" not in row
    assert seen == [_QUERY] * len(others)


# --- Packet -------------------------------------------------------------------


def _runs(a0: list[str], a1: list[str], a0b: list[str], a1b: list[str]) -> dict[str, Any]:
    def one_arm(arm: str, r1: list[str], r2: list[str]) -> dict[str, Any]:
        return {"arm": arm, "prompts": [
            {"prompt_id": "p1", "prompt": "kestrel meadow", "ranked": r1},
            {"prompt_id": "p2", "prompt": "deployment migration", "ranked": r2},
        ]}
    return {bench.ARM_A0: one_arm(bench.ARM_A0, a0, a0b),
            bench.ARM_A1: one_arm(bench.ARM_A1, a1, a1b)}


def _packet(tmp_path: Path, runs: Mapping[str, Any], seed: int = 9) -> tuple[str, str]:
    path = _store(tmp_path) if not (tmp_path / "repo").exists() else (
        tmp_path / "repo" / "aelfrice" / "memory.db")
    work = tmp_path / f"pk{len(list(tmp_path.iterdir()))}"
    work.mkdir()
    db, snapshot = bench.prepare_snapshot(path, work)
    packet, key = bench.build_packet(runs, db, seed, snapshot=snapshot)
    out = work / "out"
    bench.write_packet(packet, key, out, work / "key.jsonl")
    return (out / "packet.jsonl").read_text(), (work / "key.jsonl").read_text()


def _items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The item rows of a key's rows."""
    return [r for r in rows if r["kind"] == bench.KEY_ITEM]


def _key_items(text: str) -> list[dict[str, Any]]:
    """The item rows of a key file's text."""
    return _items([json.loads(line) for line in text.splitlines()])


def test_the_key_is_written_outside_the_packet_directory(tmp_path: Path) -> None:
    # #1795: blind packets and a hidden key. A grader handed the packet
    # directory must not find the key in it.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out = tmp_path / "packet"
    bench.write_packet(packet, key, out, tmp_path / "key.jsonl")
    assert sorted(p.name for p in out.iterdir()) == ["instructions.txt", "packet.jsonl"]
    assert json.loads((tmp_path / "key.jsonl").read_text())["belief_id"] == "B1"
    for inside in (out / "key.jsonl", out / "sub" / "key.jsonl"):
        with pytest.raises(bench.InputError, match="outside"):
            bench.write_packet(packet, key, out, inside)
        assert not inside.exists()


def test_a_relative_packet_directory_still_keeps_the_key_out(tmp_path: Path) -> None:
    # `--out pkt` beside `--key <cwd>/pkt/key.jsonl` names one directory twice.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    inside = Path.cwd() / "pkt" / "key.jsonl"
    with pytest.raises(bench.InputError, match="outside"):
        bench.write_packet(packet, key, Path("pkt"), inside)
    assert not inside.exists()


def test_a_key_path_that_links_into_the_packet_directory_is_refused(tmp_path: Path) -> None:
    # The write follows the link, so the guard must follow it too.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out = tmp_path / "pk"
    link = tmp_path / "key.jsonl"
    link.symlink_to(out / "key.jsonl")
    with pytest.raises(bench.InputError, match="outside"):
        bench.write_packet(packet, key, out, link)
    assert not (out / "key.jsonl").exists()


def test_a_key_path_inside_the_packet_directory_that_links_out_is_refused(
    tmp_path: Path,
) -> None:
    # The link stays in the directory the graders get and reads back the key.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out, elsewhere = tmp_path / "pk", tmp_path / "elsewhere"
    out.mkdir()
    elsewhere.mkdir()
    (out / "key.jsonl").symlink_to(elsewhere / "key.jsonl")
    (out / "sub").symlink_to(elsewhere, target_is_directory=True)
    for inside in (out / "key.jsonl", out / "sub" / "key.jsonl"):
        with pytest.raises(bench.InputError, match="outside"):
            bench.write_packet(packet, key, out, inside)
    assert not (elsewhere / "key.jsonl").exists()


@pytest.mark.parametrize("linked", ["packet.jsonl", "old-key.jsonl"])
def test_a_key_file_with_a_hard_link_in_the_packet_directory_is_refused(
    tmp_path: Path, linked: str,
) -> None:
    # The write goes through the key's inode, which a hard link also names:
    # the key would land in the graders' packet.jsonl, or in an old key's
    # second name inside the packet directory.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out, key_path = tmp_path / "pk", tmp_path / "key.jsonl"
    out.mkdir()
    key_path.write_text("old\n", encoding="utf-8")
    os.link(key_path, out / linked)
    with pytest.raises(bench.InputError, match="hard link"):
        bench.write_packet(packet, key, out, key_path)
    assert (out / linked).read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize("name", ["packet.jsonl", "instructions.txt"])
@pytest.mark.parametrize("link", ["symlink", "hard link"])
def test_a_link_at_a_packet_file_name_is_refused_before_any_write(
    tmp_path: Path, name: str, link: str,
) -> None:
    # #1802: a symlink at out/packet.jsonl to the key path sends the packet
    # there, and the key write then leaves the graders reading the key.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out, key_path = tmp_path / "pk", tmp_path / "key.jsonl"
    out.mkdir()
    if link == "symlink":
        (out / name).symlink_to(key_path)
    else:
        other = tmp_path / "other.txt"
        other.write_text("old\n", encoding="utf-8")
        os.link(other, out / name)
    with pytest.raises(bench.InputError, match="is a link"):
        bench.write_packet(packet, key, out, key_path)
    assert not key_path.exists()
    assert sorted(p.name for p in out.iterdir()) == [name]
    if link == "hard link":
        assert (out / name).read_text(encoding="utf-8") == "old\n"


def test_a_plain_packet_file_is_rewritten(tmp_path: Path) -> None:
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    out = tmp_path / "pk"
    out.mkdir()
    (out / "packet.jsonl").write_text("stale\n", encoding="utf-8")
    bench.write_packet(packet, key, out, tmp_path / "key.jsonl")
    assert json.loads((out / "packet.jsonl").read_text(encoding="utf-8"))["id"] == "x"


def test_the_key_guard_follows_the_filesystem_case_rule(tmp_path: Path) -> None:
    # macOS folds case by default, so there `PK/key.jsonl` lands in `pk/`.
    packet = [{"id": "x", "prompt": "p", "belief": "b"}]
    key: list[dict[str, Any]] = [{"id": "x", "belief_id": "B1", "arms": None}]
    (tmp_path / "fold").mkdir()
    folds = (tmp_path / "FOLD").exists()
    out, other = tmp_path / "pk", tmp_path / "PK" / "key.jsonl"
    if folds:
        with pytest.raises(bench.InputError, match="outside"):
            bench.write_packet(packet, key, out, other)
    else:
        other.parent.mkdir()
        bench.write_packet(packet, key, out, other)
    assert not (out / "key.jsonl").exists()


def test_the_packet_hides_arm_and_rank(tmp_path: Path) -> None:
    first = _runs(["seed", "con"], ["con", "rel", "seed"], ["w1", "w2"], ["w2"])
    # Same union per prompt, other arms and ranks.
    second = _runs(["rel", "seed", "con"], ["seed"], ["w2"], ["w2", "w1"])
    p1, k1 = _packet(tmp_path, first)
    p2, k2 = _packet(tmp_path, second)
    assert p1 == p2
    assert k1 != k2
    rows = [json.loads(line) for line in p1.splitlines()]
    assert all(set(r) == {"id", "prompt", "belief"} for r in rows)
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    key = {r["id"]: r for r in _key_items(k1)}
    for r in rows:
        k = key[r["id"]]
        assert r["id"] == bench.item_id(9, k["snapshot"], k["prompt_id"], k["belief_id"])


def test_two_stores_give_one_prompt_and_belief_distinct_item_ids(tmp_path: Path) -> None:
    # Belief ids hash the content and prompt ids hash the turn, so two stores
    # can share both. In the concatenated labels, one store's label for a
    # shared item id would overwrite the other's.
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    runs = _one_prompt_runs(["seed"], ["con"])
    ids = [{r["id"] for r in _items(bench.build_packet(runs, db, 1, snapshot=snap)[1])}
           for snap in ("s1", "s2")]
    assert ids[0] and not ids[0] & ids[1]


def test_the_key_records_arms_ranks_and_a_median_sized_null_arm(tmp_path: Path) -> None:
    runs = _runs(["seed", "con"], ["con", "rel", "seed"], ["w1", "w2"], ["w2"])
    _p, k = _packet(tmp_path, runs)
    key = _key_items(k)
    by = {(r["prompt_id"], r["belief_id"]): r for r in key}
    assert by[("p1", "con")]["arms"] == {"A0": 2, "A1": 1}
    assert by[("p1", "rel")]["arms"] == {"A1": 2}
    nulls = [r for r in _items(key) if r["arms"] is None]
    # Union sizes 3 and 2; the lower median is 2 per prompt.
    assert sorted(r["prompt_id"] for r in nulls) == ["p1", "p1", "p2", "p2"]
    retrieved = {(r["prompt_id"], r["belief_id"]) for r in _items(key) if r["arms"]}
    assert all((r["prompt_id"], r["belief_id"]) not in retrieved for r in nulls)
    assert all(r["belief_id"] != "lock1" for r in nulls)
    assert by[("p1", "seed")]["shares_prompt_stem"] is True
    assert by[("p1", "rel")]["shares_prompt_stem"] is False


def test_a_belief_sharing_only_a_stemmed_word_is_not_bridging() -> None:
    # #1795 H4: a bridging addition shares no stemmed word with the prompt.
    # No word matches exactly here, but both stem to `deploi` and `migrat`.
    assert bench.shares_prompt_stem("deploying migrations", "the deployed migration")
    assert not bench.shares_prompt_stem("deploying migrations", "otters swim at dawn")


def _one_prompt_runs(a0: list[str], a1: list[str]) -> dict[str, Any]:
    return {arm: {"arm": arm, "prompts": [
        {"prompt_id": "p1", "prompt": "kestrel meadow", "ranked": ranked}]}
        for arm, ranked in ((bench.ARM_A0, a0), (bench.ARM_A1, a1))}


def test_the_key_counts_a_prompt_that_drew_no_row(tmp_path: Path) -> None:
    # p2 packs nothing in either arm, and the lower median of the judged
    # counts (2 and 0) is 0, so p2 draws no null belief and gets no row.
    _p, k = _packet(tmp_path, _runs(["seed"], ["con"], [], []))
    assert {r["prompt_id"] for r in _key_items(k)} == {"p1"}
    store = json.loads(k.splitlines()[0])
    assert (store["kind"], store["prompts_sampled"]) == (bench.KEY_STORE, 2)


def test_a_belief_in_any_arms_top_ten_is_judged(tmp_path: Path) -> None:
    # Both arms pack `seed`: A0 at rank 1, A1 at rank 12. A0's top 10 holds
    # it, so it is judged although no compared pair differs on it.
    fill = [f"f{i:02d}" for i in range(11)]
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    _rows, key = bench.build_packet(_one_prompt_runs(["seed", *fill], [*fill, "seed"]), db, 1,
                                    snapshot=_SNAP)
    by = {r["belief_id"]: r["arms"] for r in _items(key) if r["arms"]}
    assert by["seed"] == {"A0": 1, "A1": 12}


def test_a_belief_only_the_control_packed_past_the_top_ten_is_judged(tmp_path: Path) -> None:
    # A0 packs `seed` at rank 12 and A1 never packs it. The gain subtracts
    # it, so the packet must judge it although no top 10 holds it.
    fill = [f"f{i:02d}" for i in range(11)]
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    _rows, key = bench.build_packet(_one_prompt_runs([*fill, "seed"], fill), db, 1,
                                    snapshot=_SNAP)
    by = {r["belief_id"]: r["arms"] for r in _items(key) if r["arms"]}
    assert by["seed"] == {"A0": 12}


def test_the_null_arm_is_sized_by_the_judged_count_not_the_packed_count(
    tmp_path: Path,
) -> None:
    # Both arms pack 12 beliefs. `f10` sits past rank 10 in both and is
    # unjudged, so 11 are judged and the null arm draws 11.
    fill = [f"f{i:02d}" for i in range(11)]
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    _rows, key = bench.build_packet(_one_prompt_runs(["seed", *fill], [*fill, "seed"]), db, 1,
                                    snapshot=_SNAP)
    assert sum(1 for r in _items(key) if r["arms"]) == 11
    assert sum(1 for r in _items(key) if r["arms"] is None) == 11


def test_the_null_arm_never_draws_a_retrieved_or_locked_belief(tmp_path: Path) -> None:
    path = _store(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(path, work)
    many = ["seed", "con", "rel", "w1", "w2", "w3", "w4", "w5", "w6"]
    runs = _runs(many[:5], many[5:], many[4:], many[:4])
    for seed in range(40):
        _packet_rows, key = bench.build_packet(runs, db, seed, snapshot=_SNAP)
        retrieved = {(r["prompt_id"], r["belief_id"]) for r in _items(key) if r["arms"]}
        nulls = [(r["prompt_id"], r["belief_id"]) for r in _items(key) if r["arms"] is None]
        assert len(nulls) == 18
        assert not retrieved & set(nulls)
        assert all(b != "lock1" for _p, b in nulls)


def test_the_packet_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    runs = _runs(["seed"], ["con"], ["w1"], ["w2"])
    assert _packet(tmp_path, runs, 3) == _packet(tmp_path, runs, 3)
    assert _packet(tmp_path, runs, 3)[0] != _packet(tmp_path, runs, 4)[0]


def test_the_packet_hands_the_graders_the_1804_instructions(tmp_path: Path) -> None:
    # #1809: the graders score 0 to 3 with the scale and wording #1804
    # calibrated, not the binary rubric of session 1.
    out = tmp_path / "packet"
    bench.write_packet([{"id": "x", "prompt": "p", "belief": "b"}], [], out,
                       tmp_path / "key.jsonl")
    text = (out / "instructions.txt").read_text(encoding="utf-8")
    assert text == relevance_scale.INSTRUCTIONS
    for score, meaning in relevance_scale.SCALE:
        assert f"  {score:<5} {meaning}" in text.splitlines()
    assert '  {"id": "<row id>", "score": 2}' in text.splitlines()
    assert '"correct"' not in text


# --- Score --------------------------------------------------------------------


Row = tuple[str, str, dict[str, int] | None, bool, bool | None, bool | None]


def _graded(labels: Mapping[str, bool | None]) -> dict[str, int | None]:
    """Scores that read as `labels` at the cut of 2: relevant 2, not relevant 1.

    Each sits next to the cut, so a cut of 1 or 3 changes the figures the
    score tests check.
    """
    return {i: None if v is None else 2 if v else 1 for i, v in labels.items()}


Scores = dict[str, int | None]


def _cut(scores: Scores) -> dict[str, bool | None]:
    """A grader's scores read at the harness's cut."""
    return bench.binarize(scores, bench.RELEVANT_CUT)


def _both(labels: Mapping[str, bool | None]) -> tuple[Scores, Scores]:
    """Grader a's and c's scores, both `_graded(labels)`."""
    return _graded(labels), _graded(labels)


def _key(rows: list[Row], sampled: int | None = None) -> tuple[
        bench.Key, dict[str, int | None], dict[str, int | None]]:
    """A one-store key and grader a's and c's scores, from each row's labels through
    `_graded`; `sampled` defaults to the rows' prompts."""
    n = len({r[1] for r in rows}) if sampled is None else sampled
    ran = tuple(sorted({bench.CONTROL_ARM, *(a for r in rows if r[2] for a in r[2])}))
    items = [bench.KeyItem(id=i, prompt_id=p, arms=arms, shares_prompt_stem=s,
                           snapshot=_SNAP)
             for i, p, arms, s, _a, _b in rows]
    key = bench.make_key(items, {_SNAP: bench.KeyStore(prompts_sampled=n, arms_run=ran)})
    return key, _graded({r[0]: r[4] for r in rows}), _graded({r[0]: r[5] for r in rows})


def _hand(
    a0_rel: list[int], arm_rel: list[int], *, arm: str = bench.ARM_A1,
    null_rel: int = 0, null_n: int = 10, flip_b: int = 0, bridging: int = 0,
    unclear: int = 0, unclear_arm: int = 0,
) -> list[Row]:
    """Per prompt: 10 A0 items and 10 arm items, disjoint.

    `a0_rel[p]` and `arm_rel[p]` of them are relevant to both graders.
    `flip_b` of A0's not-relevant items per prompt are relevant to grader B
    only. `bridging` of each prompt's relevant arm items share no stem.
    `unclear` of A0's not-relevant items per prompt are unclear to both,
    and `unclear_arm` of the arm's.
    """
    rows: list[Row] = []
    for p, (r0, r1) in enumerate(zip(a0_rel, arm_rel, strict=True)):
        for k in range(10):
            rel = k < r0
            b = True if (not rel and 10 - flip_b <= k) else rel
            lab: bool | None = None if (not rel and r0 <= k < r0 + unclear) else rel
            rows.append((f"a{p}-{k}", f"p{p}", {bench.ARM_A0: k + 1}, True,
                         lab, b if lab is not None else None))
        for k in range(10):
            rel = k < r1
            lab1: bool | None = None if (not rel and r1 <= k < r1 + unclear_arm) else rel
            rows.append((f"x{p}-{k}", f"p{p}", {arm: k + 1}, not (rel and k < bridging),
                         lab1, lab1))
        for k in range(null_n):
            rel = k < null_rel
            rows.append((f"n{p}-{k}", f"p{p}", None, False, rel, rel))
    return rows


def _census(*rates: float, snapshots: tuple[str, ...] = (_SNAP,),
            writers: bool = True) -> dict[str, Any]:
    """A census report with one row per snapshot and its probed-type rate."""
    return {"writers": writers, "stores": [
        {"snapshot": s, "probed_per_100_beliefs": r}
        for s, r in zip(snapshots, rates, strict=True)]}


_H1_PASS = _census(2.0)
_H1_FAIL = _census(0.5)


def _score(rows: list[Row], sampled: int | None = None, **kw: Any) -> dict[str, Any]:
    key, a, b = _key(rows, sampled)
    kw.setdefault("census_report", _H1_PASS)
    return bench.score(key, a, b, seed=1, **kw)


def test_paired_gain_share_and_precision_on_a_hand_example() -> None:
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4]))
    g = rep["arms"]["A1"]["graders"][0]
    assert g["mean_gain"] == _approx(1.0)  # gains 1, 1, 0, 2
    assert g["share_gaining"] == _approx(0.75)
    assert g["precision"] == _approx(12 / 40)
    assert g["precision_a0"] == _approx(8 / 40)
    assert g["precision_drop"] == _approx(-0.1)
    assert 0.0 < g["ci95"][0] <= 1.0 <= g["ci95"][1]
    assert rep["kappa"] == _approx(1.0)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H2"] == bench.V_PASS
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_PASS}
    assert rep["hypotheses"]["H3"] == bench.V_NOT_MEASURED


def test_a_prompt_the_key_holds_no_row_for_still_counts_in_the_gain() -> None:
    # 4 prompts sampled. Only p0 has rows: A0 packs `a`, A3 packs `a` and
    # `x`, all relevant. p1 to p3 packed nothing and drew no null belief.
    rows: list[Row] = [
        ("a", "p0", {bench.ARM_A0: 1, bench.ARM_A3: 1}, True, True, True),
        ("x", "p0", {bench.ARM_A3: 2}, True, True, True),
    ]
    rep = _score(rows, sampled=4)
    g = rep["arms"]["A3"]["graders"][0]
    assert g["mean_gain"] == _approx(0.25)  # gains 1, 0, 0, 0
    assert g["share_gaining"] == _approx(0.25)
    assert g["top10_mean_gain"] == _approx(0.25)
    assert (rep["prompts"], rep["keyed_prompts"]) == (4, 1)


def test_a_prompt_the_key_holds_no_row_for_still_counts_in_the_lane_gain() -> None:
    # H2_split's lane half is A1 against A1W over the same sampled prompts.
    rows: list[Row] = [
        ("a", "p0", {bench.ARM_A0: 1, bench.ARM_A1W: 1, bench.ARM_A1: 1}, True, True, True),
        ("x", "p0", {bench.ARM_A1: 2}, True, True, True),
    ]
    lane = _score(rows, sampled=4)["arms"]["A1"]["vs_A1W"]["graders"][0]
    assert lane["mean_gain"] == _approx(0.25)  # gains 1, 0, 0, 0
    assert lane["share_gaining"] == _approx(0.25)


def test_score_rejects_a_key_with_more_prompts_than_its_store_sampled() -> None:
    rows: list[Row] = [
        ("a", "p0", {bench.ARM_A0: 1}, True, True, True),
        ("b", "p1", {bench.ARM_A0: 1}, True, True, True),
    ]
    with pytest.raises(bench.InputError, match="holds 2 prompts"):
        _score(rows, sampled=1)


def _store_key_rows(snapshot: str, sampled: int,
                    arms_run: tuple[str, ...] = ("A0", "A3")) -> list[dict[str, Any]]:
    """One store's key: its store row, then A0 packs `a` and A3 packs `a` and
    `x`, all on prompt p0."""
    store: dict[str, Any] = {"kind": "store", "snapshot": snapshot,
                             "prompts_sampled": sampled, "arms_run": list(arms_run)}
    return [store, *({"kind": "item", "id": f"{snapshot}-{b}", "prompt_id": "p0",
                      "arms": arms, "shares_prompt_stem": True, "snapshot": snapshot}
                     for b, arms in (("a", {"A0": 1, "A3": 1}), ("x", {"A3": 2})))]


def test_a_store_whose_arms_packed_nothing_still_counts_when_pooled(tmp_path: Path) -> None:
    # Store sb's arms packed nothing for any of its 3 prompts, so it has no
    # item row and draws no null belief. Its prompts still count in the
    # pooled paired gain, and its proxy report still has a store to match.
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    empty: dict[str, Any] = {arm: {"arm": arm, "prompts": [
        {"prompt_id": f"p{n}", "prompt": "kestrel meadow", "ranked": []} for n in range(3)]}
        for arm in (bench.ARM_A0, bench.ARM_A1)}
    sb = bench.build_packet(empty, db, 1, snapshot="sb")
    assert sb == ([], [{"kind": "store", "snapshot": "sb", "prompts_sampled": 3,
                        "arms_run": ["A0", "A1"]}])
    rows = bench.build_packet(_one_prompt_runs(["seed"], ["seed", "con"]), db, 1,
                              snapshot="sa")[1] + sb[1]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels: dict[str, bool | None] = {r["id"]: r["arms"] is not None for r in _items(rows)}
    proxies: list[dict[str, Any]] = [{"snapshot": snap, "prompts": [
        {"prompt_id": pid, "arms": {"A0": [], "A1": []}} for pid in pids]}
        for snap, pids in (("sa", ["p1"]), ("sb", ["p0", "p1", "p2"]))]
    rep = bench.score(bench.read_key(path), *_both(labels), seed=1, proxy_reports=proxies)
    assert (rep["prompts"], rep["keyed_prompts"], rep["stores"]) == (4, 1, 2)
    # A1 gains `con` on sa's one prompt and nothing on sb's three.
    assert rep["arms"]["A1"]["graders"][0]["mean_gain"] == _approx(0.25)
    assert rep["arms"]["A1"]["graders"][0]["share_gaining"] == _approx(0.25)


def test_a_key_concatenated_from_several_stores_scores_them_pooled(tmp_path: Path) -> None:
    # #1795 decides H2 to H6 over every store's prompts, and each store's
    # key records its own sample. Both stores name their prompt p0.
    for counts in ((4, 4), (4, 6)):
        rows = _store_key_rows("s1", counts[0]) + _store_key_rows("s2", counts[1])
        path = tmp_path / "key.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        labels: dict[str, bool | None] = {str(r["id"]): True for r in _items(rows)}
        rep = bench.score(bench.read_key(path), *_both(labels), seed=1)
        assert (rep["prompts"], rep["keyed_prompts"]) == (sum(counts), 2)
        assert rep["stores"] == 2
        g = rep["arms"]["A3"]["graders"][0]
        assert g["mean_gain"] == _approx(2 / sum(counts))  # p0 gains 1 in each store
        assert g["share_gaining"] == _approx(2 / sum(counts))


def test_score_refuses_a_pooled_key_whose_stores_ran_different_arms(tmp_path: Path) -> None:
    # s2's runs had no A3, so its A0-only items would count as A3's losses.
    s2 = _store_key_rows("s2", 1, ("A0",))
    s2 = [s2[0], {**s2[1], "arms": {"A0": 1}}]
    rows = _store_key_rows("s1", 1) + s2
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels: dict[str, bool | None] = {str(r["id"]): True for r in _items(rows)}
    with pytest.raises(bench.InputError, match="s1 ran A0, A3; s2 ran A0"):
        bench.score(bench.read_key(path), *_both(labels), seed=1)
    # An arm that ran on every store is scored although no row records it.
    ran = _store_key_rows("s1", 1, ("A0", "A2", "A3"))
    path.write_text("".join(json.dumps(r) + "\n" for r in ran), encoding="utf-8")
    labels = {str(r["id"]): True for r in _items(ran)}
    assert sorted(bench.score(bench.read_key(path), *_both(labels), seed=1)["arms"]) == [
        "A2", "A3"]
    for bad in (["A3"], ["A0", "A0", "A3"], ["A0", "A9"], "A0"):
        path.write_text(json.dumps({**ran[0], "arms_run": bad}) + "\n", encoding="utf-8")
        with pytest.raises(bench.InputError, match="arms_run with A0"):
            bench.read_key(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in [{**ran[0], "arms_run": ["A0"]},
                                                             ran[1]]), encoding="utf-8")
    with pytest.raises(bench.InputError, match="an arm outside arms_run"):
        bench.read_key(path)


def test_the_key_records_the_arms_the_runs_covered(tmp_path: Path) -> None:
    _p, k = _packet(tmp_path, _runs(["seed"], ["con"], [], []))
    stores = [r for r in map(json.loads, k.splitlines()) if r["kind"] == bench.KEY_STORE]
    assert [r["arms_run"] for r in stores] == [["A0", "A1"]]


def _proxy_report(snapshot: str, gains: list[int]) -> dict[str, Any]:
    """One store's proxy report: per prompt, A3 uses `gain` more beliefs than A0."""
    def used(n: int) -> list[dict[str, Any]]:
        return [{"id": f"u{k}", "used": True} for k in range(n)]

    return {"snapshot": snapshot, "prompts": [
        {"prompt_id": f"p{n}", "arms": {"A0": used(max(-g, 0)), "A3": used(max(g, 0))}}
        for n, g in enumerate(gains)]}


def _pooled_key(tmp_path: Path) -> tuple[bench.Key, dict[str, bool | None]]:
    """Store s1 with 1 prompt and s2 with 3; A3 gains 1 relevant belief on p0 of each.

    Each store also has a null item both graders call not relevant, so H0 holds.
    """
    rows = _store_key_rows("s1", 1) + _store_key_rows("s2", 3)
    nulls = [{**r, "id": f"{r['snapshot']}-n", "arms": None}
             for r in _items(rows) if r["id"].endswith("-a")]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows + nulls), encoding="utf-8")
    labels: dict[str, bool | None] = {str(r["id"]): True for r in _items(rows)}
    return bench.read_key(path), {**labels, **{str(r["id"]): False for r in nulls}}


def test_h6_pools_the_proxy_reports_of_every_store_in_the_key(tmp_path: Path) -> None:
    # s1's proxy gains -3 on its 1 prompt and s2's +1 on each of its 3.
    # Pooled over the 4 prompts the proxy gains 0. s1 alone gives -1, s2
    # alone +1, and the mean of the two stores' means -1.
    items, labels = _pooled_key(tmp_path)
    s1, s2 = _proxy_report("s1", [-3]), _proxy_report("s2", [1, 1, 1])
    assert bench.proxy_gain_signs(s1, ["A3"]) == {"A3": -1}
    assert bench.proxy_gain_signs(s2, ["A3"]) == {"A3": 1}
    for reports in ([s1, s2], [s2, s1]):
        rep = bench.score(items, *_both(labels), seed=1, proxy_reports=reports)
        assert rep["hypotheses"]["H6"]["A3"] == {
            "proxy_sign": 0, "grader_signs": [1, 1], "verdict": bench.V_DISAGREE}


def test_score_refuses_a_proxy_report_that_matches_no_store_one_to_one(
    tmp_path: Path,
) -> None:
    items, labels = _pooled_key(tmp_path)
    s1, s2 = _proxy_report("s1", [1]), _proxy_report("s2", [1, 1, 1])
    bad: list[tuple[list[dict[str, Any]], str]] = [
        ([s1], "no proxy report for the key's snapshot s2"),
        ([s1, s2, _proxy_report("s3", [1])], "no key rows for the proxy report's snapshot s3"),
        ([s1, {k: v for k, v in s2.items() if k != "snapshot"}], "proxy report 2 records no"),
        ([s1, {**s2, "snapshot": ""}], "proxy report 2 records no"),
        ([s1, s2, s1], "two proxy reports record snapshot s1"),
    ]
    for reports, match in bad:
        with pytest.raises(bench.InputError, match=match):
            bench.score(items, *_both(labels), seed=1, proxy_reports=reports)
    # Without a proxy, H6 is not scored and nothing is matched.
    rep = bench.score(items, *_both(labels), seed=1)
    assert rep["hypotheses"]["H6"] == bench.V_NOT_MEASURED


def test_score_refuses_a_proxy_report_that_misses_an_arm_or_a_prompt(tmp_path: Path) -> None:
    # An arm or a prompt the proxy never measured would count as packing
    # nothing, and H6 would report a sign the proxy never saw.
    items, labels = _pooled_key(tmp_path)
    s1 = _proxy_report("s1", [1])
    full = _proxy_report("s2", [1, 1, 1])["prompts"]
    bad: list[tuple[list[dict[str, Any]], str]] = [
        ([{**r, "arms": {"A0": r["arms"]["A0"]}} for r in full],
         "measured A0 on prompt p0, not the key's A0, A3"),
        ([{**r, "arms": {**r["arms"], "A2": []}} for r in full],
         "measured A0, A2, A3 on prompt p0, not the key's A0, A3"),
        ([], "covers 0 prompts, but the key's store sampled 3"),
        (full[:2], "covers 2 prompts, but the key's store sampled 3"),
        # A prompt the graders never saw would join H6's proxy sign.
        (_proxy_report("s2", [1, 1, 1, 9])["prompts"],
         "covers 4 prompts, but the key's store sampled 3"),
        ([{**r, "prompt_id": f"q{n}"} for n, r in enumerate(full)], "has no row for prompt p0"),
        ([full[0], full[0], full[1]], "repeats prompt p0"),
        ([*full[:2], {"arms": full[2]["arms"]}], "has a malformed prompt row"),
        ([*full[:2], {"prompt_id": "p2", "arms": []}], "has a malformed prompt row"),
        ([*full[:2], {"prompt_id": "p2", "arms": {"A0": [{"id": "a"}], "A3": []}}],
         "has a malformed prompt row"),
        ([*full[:2], {"prompt_id": "p2", "arms": {"A0": [], "A3": {}}}],
         "has a malformed prompt row"),
    ]
    for prompts, match in bad:
        with pytest.raises(bench.InputError, match=f"proxy report for snapshot s2 {match}"):
            bench.score(items, *_both(labels), seed=1,
                        proxy_reports=[s1, {"snapshot": "s2", "prompts": prompts}])
    rep = bench.score(items, *_both(labels), seed=1,
                      proxy_reports=[s1, {"snapshot": "s2", "prompts": full}])
    assert rep["hypotheses"]["H6"]["A3"]["proxy_sign"] == 1


def test_score_cli_takes_one_proxy_report_per_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _store_key_rows("s1", 1) + _store_key_rows("s2", 3)
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels = [tmp_path / "a.jsonl", tmp_path / "c.jsonl"]
    for path in labels:
        path.write_text("".join(json.dumps({"id": r["id"], "score": 3}) + "\n"
                                for r in _items(rows)), encoding="utf-8")
    proxies: list[str] = []
    for report in (_proxy_report("s1", [-2]), _proxy_report("s2", [1, 1, 1])):
        proxies.append(str(tmp_path / f"proxy-{report['snapshot']}.json"))
        Path(proxies[-1]).write_text(json.dumps(report), encoding="utf-8")
    usage = ["score", "--key", str(key), "--labels", *map(str, labels), "--proxy"]
    assert bench.main([*usage, *proxies]) == 0
    assert json.loads(capsys.readouterr().out)["hypotheses"]["H6"]["A3"]["proxy_sign"] == 1
    assert bench.main([*usage, proxies[0]]) == 1
    assert "no proxy report for the key's snapshot s2" in capsys.readouterr().err


def test_a_zero_gain_fails_and_a_precision_drop_blocks() -> None:
    rep = _score(_hand([4, 4, 4, 4], [3, 3, 3, 3]))
    assert rep["arms"]["A1"]["graders"][0]["precision_drop"] == _approx(0.1)
    assert rep["hypotheses"]["H2"] == bench.V_FAIL
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_FAIL}


def test_a_precision_drop_of_exactly_the_bar_passes_h5() -> None:
    # #1795 H5 kills only a drop above 0.05. 0.8 - 0.75 is 0.05 plus float error.
    on_bar = _score(_hand([8, 8, 8, 8], [7, 8, 8, 7]))
    assert on_bar["arms"]["A1"]["graders"][0]["precision_drop"] == _approx(0.05)
    assert on_bar["hypotheses"]["H5"] == {"A1": bench.V_PASS}
    over = _score(_hand([8, 8, 8, 8], [7, 7, 8, 7]))
    assert over["arms"]["A1"]["graders"][0]["precision_drop"] == _approx(0.075)
    assert over["hypotheses"]["H5"] == {"A1": bench.V_FAIL}


def test_a_precision_drop_just_over_the_bar_blocks() -> None:
    # A0 is 320/400. One unclear arm item makes the arm 299/399, a drop of
    # 0.0506, so a bar moved anywhere from 0.0507 up lets it pass.
    rows = [(r[0], r[1], r[2], r[3], None, None) if r[0] == "x0-9" else r
            for r in _hand([8] * 40, [7] * 21 + [8] * 19)]
    rep = _score(rows)
    assert rep["arms"]["A1"]["graders"][0]["precision_drop"] == _approx(0.8 - 299 / 399)
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_FAIL}


def test_a_precision_drop_a_billionth_over_the_bar_fails_h5() -> None:
    # A0's top 10s hold 5085 relevant of 9199 graded and the arm's 4616 of
    # 9181, a drop of 0.05 plus 5.9e-10. Pooled stores reach counts this size.
    rows: list[Row] = [(f"a{k}", "p0", {bench.ARM_A0: 1}, True, k < 5085, k < 5085)
                       for k in range(9199)]
    rows += [(f"x{k}", "p0", {bench.ARM_A1: 1}, True, k < 4616, k < 4616)
             for k in range(9181)]
    rows += [(f"n{k}", "p0", None, False, False, False) for k in range(10)]
    rep = _score(rows)
    g = rep["arms"]["A1"]["graders"][0]
    assert (g["graded_a0"], g["graded"]) == (9199, 9181)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_FAIL}


def test_an_interval_holding_zero_kills_the_gain() -> None:
    rep = _score(_hand([2, 2, 2, 2], [6, 2, 2, 0]))
    g = rep["arms"]["A1"]["graders"][0]
    assert g["mean_gain"] == _approx(0.5)
    assert g["ci95"][0] <= 0.0 <= g["ci95"][1]
    assert rep["hypotheses"]["H2"] == bench.V_FAIL


def test_a_small_positive_gain_is_inconclusive() -> None:
    # Gains 0, 0, 1, 0 x5 prompts: mean 0.2, interval above 0, under the 0.3 bar.
    rep = _score(_hand([2] * 20, [2, 2, 3, 2, 2] * 4))
    g = rep["arms"]["A1"]["graders"][0]
    assert g["mean_gain"] == _approx(0.2)
    assert g["ci95"][0] > 0.0
    assert rep["hypotheses"]["H2"] == bench.V_INCONCLUSIVE


def test_combine_passes_only_when_both_graders_pass() -> None:
    p, f, i, n = bench.V_PASS, bench.V_FAIL, bench.V_INCONCLUSIVE, bench.V_INSUFFICIENT
    assert bench.combine([p, p]) == p
    assert bench.combine([p, f]) == f and bench.combine([f, p]) == f
    assert bench.combine([p, i]) == i and bench.combine([i, p]) == i
    assert bench.combine([n, f]) == n and bench.combine([p, n]) == n


def test_an_arm_one_grader_fails_fails() -> None:
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    # Grader B calls only A1's first 2 items per prompt relevant: no gain.
    b_view = [(r[0], r[1], r[2], r[3], r[4], r[5] if not r[0].startswith("x")
               else int(r[0].split("-")[1]) < 2) for r in rows]
    swapped = [(r[0], r[1], r[2], r[3], r[5], r[4]) for r in b_view]
    for case in (b_view, swapped):
        rep = _score(case)
        assert rep["hypotheses"]["H0"] == bench.V_PASS
        verdicts = [g["gain_verdict"] for g in rep["arms"]["A1"]["graders"]]
        assert sorted(verdicts) == [bench.V_FAIL, bench.V_PASS]
        assert rep["hypotheses"]["H2"] == bench.V_FAIL


def test_a_pass_needs_a_tenth_of_the_prompts_to_gain() -> None:
    # Mean and interval clear the bars; only the gaining share differs.
    assert bench.gain_verdict(0.3, (0.1, 0.5), 0.10) == bench.V_PASS
    assert bench.gain_verdict(0.3, (0.1, 0.5), 0.05) == bench.V_INCONCLUSIVE
    assert bench.gain_verdict(0.29, (0.1, 0.5), 0.5) == bench.V_INCONCLUSIVE


def test_low_kappa_makes_every_verdict_unreliable() -> None:
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], flip_b=6))
    assert rep["kappa"] < bench.KAPPA_BAR
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    assert rep["hypotheses"]["H2"] == bench.V_UNRELIABLE
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_UNRELIABLE}


def _null_table(tt: int, tf: int, ft: int, ff: int) -> list[Row]:
    """Null-arm items with grader A's and B's labels in the given counts."""
    pairs = [(True, True)] * tt + [(True, False)] * tf + [(False, True)] * ft
    pairs += [(False, False)] * ff
    return [(f"n{k}", "p0", None, False, a, b) for k, (a, b) in enumerate(pairs)]


def test_a_kappa_of_exactly_the_bar_holds_h0() -> None:
    # The exact kappa of this table is 3/5; the float one is 0.5999999999999996.
    on_bar = _score(_null_table(2, 1, 1, 14))
    assert on_bar["kappa"] == _approx(0.6)
    assert on_bar["hypotheses"]["H0"] == bench.V_PASS
    under = _score(_null_table(2, 1, 2, 13))
    assert under["kappa"] == _approx(48 / 102)
    assert under["hypotheses"]["H0"] == bench.V_FAIL


def test_a_kappa_a_billionth_under_the_bar_fails_h0() -> None:
    # The exact kappa of this table is 0.6 minus 9.8e-10, and both null
    # rates are under the ceiling.
    rep = _score(_null_table(3031, 1503, 1520, 20963))
    assert rep["kappa_n"] == 27017
    assert all(n["rate"] <= bench.NULL_RATE_CEILING for n in rep["null_arm"])
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    # The bars compared are the decimals #1795 states, not their nearest floats.
    assert bench.exact_bar(bench.KAPPA_BAR) == Fraction(3, 5)
    assert bench.exact_bar(bench.PRECISION_DROP_BAR) == Fraction(1, 20)


def test_kappa_takes_each_graders_own_marginals() -> None:
    # One grader calls 1 of 10 relevant and the other 2: chance agreement
    # is 0.74 and kappa 8/13, over the bar, whichever grader is which.
    for table in (_null_table(1, 0, 1, 8), _null_table(1, 1, 0, 8)):
        key, a, b = _key(table)
        assert bench.exact_kappa([i.id for i in key.items], _cut(a), _cut(b)) == Fraction(8, 13)
        rep = _score(table)
        assert rep["kappa"] == _approx(8 / 13)
        assert rep["hypotheses"]["H0"] == bench.V_PASS


def test_graders_who_share_no_graded_item_fail_h0() -> None:
    # Each grader grades one null item, under the ceiling, but the other
    # grader left that item unclear, so no item gives a kappa.
    rows: list[Row] = [("n0", "p0", None, False, False, None),
                       ("n1", "p0", None, False, None, False)]
    key, a, b = _key(rows)
    assert bench.exact_kappa([i.id for i in key.items], _cut(a), _cut(b)) is None
    rep = _score(rows)
    assert (rep["kappa"], rep["kappa_n"]) == (None, 0)
    assert [n["graded"] for n in rep["null_arm"]] == [1, 1]
    assert rep["hypotheses"]["H0"] == bench.V_FAIL


@pytest.mark.parametrize(("a_score", "c_score"), [(1, 0), (0, 0), (3, 3)])
def test_graders_who_put_every_item_on_one_side_of_the_cut_fail_h0(
    a_score: int, c_score: int,
) -> None:
    # As in the #1804 calibration, kappa is then undefined and never meets
    # the bar, though the graders never disagree at the cut.
    key, a, c = _key(_hand([2, 2, 2, 2], [3, 3, 2, 4]))
    one_side: tuple[Scores, Scores] = (
        dict.fromkeys(a, a_score), dict.fromkeys(c, c_score))
    assert bench.exact_kappa([i.id for i in key.items], *map(_cut, one_side)) is None
    rep = bench.score(key, *one_side, seed=1, census_report=_H1_PASS)
    assert (rep["kappa"], rep["kappa_n"], rep["agreed"]) == (None, len(a), len(a))
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_UNRELIABLE}


def test_a_kappa_of_1_needs_both_sides_of_the_cut() -> None:
    # One item both graders call relevant and the rest neither: kappa is 1.
    rows = _null_table(1, 0, 0, 17)
    key, a, b = _key(rows)
    assert bench.exact_kappa([i.id for i in key.items], _cut(a), _cut(b)) == 1
    rep = _score(rows)
    assert rep["kappa"] == _approx(1.0)
    assert rep["hypotheses"]["H0"] == bench.V_PASS


def test_an_item_one_grader_left_unclear_stays_out_of_kappa() -> None:
    # The null table's kappa is exactly the bar. Each A0 item is unclear to
    # one grader and relevant to the other: counted as not relevant, either
    # item adds a disagreement and drops kappa under the bar.
    rows = [*_null_table(2, 1, 1, 14), ("a0", "p0", {bench.ARM_A0: 1}, True, None, True),
            ("a1", "p0", {bench.ARM_A0: 2}, True, True, None)]
    key, a, b = _key(rows)
    assert bench.exact_kappa([i.id for i in key.items], _cut(a), _cut(b)) == Fraction(3, 5)
    rep = _score(rows)
    assert rep["kappa_n"] == 18
    assert rep["hypotheses"]["H0"] == bench.V_PASS


def test_a_high_null_arm_rate_makes_every_verdict_unreliable() -> None:
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3))
    assert rep["null_arm"][0]["rate"] == _approx(0.3)
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    assert rep["hypotheses"]["H2"] == bench.V_UNRELIABLE


def test_either_graders_null_rate_over_the_ceiling_fails_h0() -> None:
    # #1795 H0 holds each grader to the ceiling: 7 of 40 is 0.175, 9 of 40 is 0.225.
    for table, rates in ((_null_table(7, 0, 2, 31), [0.175, 0.225]),
                         (_null_table(7, 2, 0, 31), [0.225, 0.175])):
        rep = _score(table)
        assert [n["rate"] for n in rep["null_arm"]] == [_approx(r) for r in rates]
        assert rep["kappa"] >= bench.KAPPA_BAR
        assert rep["hypotheses"]["H0"] == bench.V_FAIL


def test_a_grader_with_no_graded_null_item_fails_h0() -> None:
    # Grader A leaves every null item unclear: a rate of 0 out of 0.
    rows = [(r[0], r[1], r[2], r[3], None if r[2] is None else r[4], r[5])
            for r in _hand([2, 2, 2, 2], [3, 3, 2, 4])]
    rep = _score(rows)
    assert rep["null_arm"][0]["graded"] == 0
    assert rep["kappa"] == _approx(1.0)
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    assert rep["hypotheses"]["H2"] == bench.V_UNRELIABLE


def test_too_few_graded_items_is_insufficient() -> None:
    rep = _score(_hand([2, 2], [3, 3]))
    assert rep["arms"]["A1"]["graders"][0]["graded"] == 20 < graded_gate.MIN_GRADED
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_INSUFFICIENT}


def test_either_side_under_min_graded_is_insufficient() -> None:
    # 7 graded per prompt on one side: 28, under 30; the other side has 40.
    for rows in (_hand([2, 2, 2, 2], [3, 3, 2, 4], unclear=3),
                 _hand([2, 2, 2, 2], [3, 3, 2, 4], unclear_arm=3)):
        rep = _score(rows)
        g = rep["arms"]["A1"]["graders"][0]
        assert sorted((g["graded"], g["graded_a0"])) == [28, 40]
        assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
        assert rep["hypotheses"]["H5"] == {"A1": bench.V_INSUFFICIENT}


def test_exactly_min_graded_on_each_side_is_sufficient() -> None:
    # The 2026-10-08 ruling: a verdict needs at least MIN_GRADED graded items.
    rows = _hand([2, 2, 2], [3, 3, 3])
    rep = _score(rows)
    g = rep["arms"]["A1"]["graders"][0]
    assert (g["graded"], g["graded_a0"]) == (graded_gate.MIN_GRADED, graded_gate.MIN_GRADED)
    assert rep["hypotheses"]["H2"] == bench.V_PASS
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_PASS}
    # One not-relevant item turned unclear leaves its side one short.
    for unclear in ("a0-9", "x0-9"):
        short: list[Row] = [(r[0], r[1], r[2], r[3], None, None) if r[0] == unclear else r
                            for r in rows]
        rep = _score(short)
        assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
        assert rep["hypotheses"]["H5"] == {"A1": bench.V_INSUFFICIENT}


def test_min_graded_counts_the_top_tens_not_the_packed_result() -> None:
    # The 2026-10-24 ruling keeps the count on the top 10s. Each side packs
    # 20 graded items per prompt, 10 of them past rank 10: 40 graded in the
    # packed results, 20 in the top 10s.
    rows: list[Row] = []
    for p in range(2):
        for k in range(20):
            rows.append((f"a{p}-{k}", f"p{p}", {bench.ARM_A0: k + 1}, True, k < 2, k < 2))
            rows.append((f"x{p}-{k}", f"p{p}", {bench.ARM_A1: k + 1}, True, k < 4, k < 4))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    rep = _score(rows)
    g = rep["arms"]["A1"]["graders"][0]
    assert (g["graded"], g["graded_a0"]) == (20, 20)
    assert g["mean_gain"] == _approx(2.0)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_INSUFFICIENT}


def test_a_failed_h0_overrides_insufficient() -> None:
    # #1795 H0: either bar failing makes every arm result `unreliable`.
    rep = _score(_hand([2, 2], [3, 3], null_rel=5))
    assert rep["arms"]["A1"]["graders"][0]["gain_verdict"] == bench.V_INSUFFICIENT
    assert rep["hypotheses"]["H2"] == bench.V_UNRELIABLE
    assert rep["hypotheses"]["H5"] == {"A1": bench.V_UNRELIABLE}


def test_unclear_labels_are_left_out_of_precision() -> None:
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], unclear=5))
    g = rep["arms"]["A1"]["graders"][0]
    assert g["graded_a0"] == 20
    assert g["precision_a0"] == _approx(8 / 20)


def test_unclear_labels_are_left_out_of_the_gain_and_the_additions() -> None:
    # One not-relevant arm item per prompt is unclear to both graders.
    # Counted as relevant, it would add 1 to every prompt's gain.
    g = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], unclear_arm=1))["arms"]["A1"]["graders"][0]
    assert g["mean_gain"] == _approx(1.0)
    assert g["top10_mean_gain"] == _approx(1.0)
    a3 = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], arm=bench.ARM_A3, unclear_arm=1))
    assert a3["arms"]["A3"]["graders"][0]["relevant_additions"] == 12


def test_h4_needs_half_of_a3s_relevant_additions_to_bridge() -> None:
    passing = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], arm=bench.ARM_A3, bridging=2))
    g = passing["arms"]["A3"]["graders"][0]
    assert (g["relevant_additions"], g["bridging"]) == (12, 8)
    assert g["bridging_share"] == _approx(8 / 12)
    assert passing["hypotheses"]["H4"] == bench.V_PASS
    failing = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], arm=bench.ARM_A3, bridging=1))
    assert failing["arms"]["A3"]["graders"][0]["bridging_share"] == _approx(4 / 12)
    assert failing["hypotheses"]["H4"] == bench.V_FAIL


def test_h4_fails_low_bridging_on_an_inconclusive_gain() -> None:
    # #1795 H4 kills on bridging whatever the gain. A mean gain of 0.2 is
    # inconclusive when every addition bridges, and fails when none does.
    a0, a3 = [2] * 40, [3] * 8 + [2] * 32
    bridged = _score(_hand(a0, a3, arm=bench.ARM_A3, bridging=3))
    g = bridged["arms"]["A3"]["graders"][0]
    assert g["mean_gain"] == _approx(0.2) and g["ci95"][0] > 0.0
    assert g["bridging_share"] == _approx(1.0)
    assert bridged["hypotheses"]["H4"] == bench.V_INCONCLUSIVE
    unbridged = _score(_hand(a0, a3, arm=bench.ARM_A3))
    assert unbridged["arms"]["A3"]["graders"][0]["bridging_share"] == _approx(0.0)
    assert unbridged["hypotheses"]["H4"] == bench.V_FAIL


def test_h2_is_a_lane_verdict_only_when_the_census_passed_h1() -> None:
    # #1795 H1: under 1 per 100, A1 is substrate-starved, not a lane verdict.
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    starved = _score(rows, census_report=_H1_FAIL)
    assert starved["hypotheses"]["H1"] == bench.V_FAIL
    assert starved["hypotheses"]["H2"] == bench.V_STARVED
    assert starved["arms"]["A1"]["gain_verdict"] == bench.V_PASS
    no_census = _score(rows, census_report=None)
    assert no_census["hypotheses"]["H1"] == bench.V_NOT_MEASURED
    assert no_census["hypotheses"]["H2"] == bench.V_NEEDS_CENSUS
    assert _score(rows)["hypotheses"]["H2"] == bench.V_PASS
    # A failed H0 outranks the substrate: the instrument comes first.
    unreliable = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3),
                        census_report=_H1_FAIL)
    assert unreliable["hypotheses"]["H2"] == bench.V_UNRELIABLE
    # H3 and H4 don't depend on the writers' substrate.
    a2 = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], arm=bench.ARM_A2),
                census_report=_H1_FAIL)
    assert a2["hypotheses"]["H3"] == bench.V_PASS


def _split_rows(a0_rel: list[int], a1w_rel: list[int], a1_rel: list[int],
                null_rel: int = 0) -> list[Row]:
    """`_hand` for A0 and A1W, plus 10 disjoint A1 items per prompt."""
    rows = _hand(a0_rel, a1w_rel, arm=bench.ARM_A1W, null_rel=null_rel)
    for p, r in enumerate(a1_rel):
        rows += [(f"y{p}-{k}", f"p{p}", {bench.ARM_A1: k + 1}, True, k < r, k < r)
                 for k in range(10)]
    return rows


def test_h2_split_separates_the_writers_from_the_lane() -> None:
    # A1 gains over A0 only what A1W already gains: the writers, not the lane.
    writers_only = _score(_split_rows([2, 2, 2, 2], [3, 3, 2, 4], [3, 3, 2, 4]))
    assert writers_only["hypotheses"]["H2"] == bench.V_PASS
    assert writers_only["hypotheses"]["H2_split"] == {
        "writers": bench.V_PASS, "lane": bench.V_FAIL}
    lane = writers_only["arms"]["A1"]["vs_A1W"]["graders"][0]
    assert lane["mean_gain"] == _approx(0.0)
    # A1 gains over A1W, and A1W nothing over A0: the lane.
    lane_only = _score(_split_rows([2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 2, 4]))
    assert lane_only["hypotheses"]["H2_split"] == {
        "writers": bench.V_FAIL, "lane": bench.V_PASS}
    starved = _score(_split_rows([2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 2, 4]),
                     census_report=_H1_FAIL)
    assert starved["hypotheses"]["H2_split"] == {
        "writers": bench.V_STARVED, "lane": bench.V_STARVED}
    assert _score(_hand([2, 2, 2, 2], [3, 3, 2, 4]))["hypotheses"]["H2_split"] == (
        bench.V_NOT_MEASURED)


def test_the_lane_reads_grader_a_then_grader_c() -> None:
    # A1 gains 1, 1, 0, 2 over A1W for the grader that keeps A1's labels.
    # The other calls only A1's first 2 items per prompt relevant, A1W's
    # count: no lane gain, and the lane fails.
    rows = _split_rows([2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 2, 4])
    one_fails = [r if not r[0].startswith("y") else (*r[:5], int(r[0].split("-")[1]) < 2)
                 for r in rows]
    swapped = [(*r[:4], r[5], r[4]) for r in one_fails]
    for case, gains, verdicts in (
        (one_fails, [1.0, 0.0], [bench.V_PASS, bench.V_FAIL]),
        (swapped, [0.0, 1.0], [bench.V_FAIL, bench.V_PASS]),
    ):
        rep = _score(case)
        assert rep["hypotheses"]["H0"] == bench.V_PASS
        lane = rep["arms"]["A1"]["vs_A1W"]["graders"]
        assert [g["mean_gain"] for g in lane] == [_approx(x) for x in gains]
        assert [g["gain_verdict"] for g in lane] == verdicts
        assert rep["hypotheses"]["H2_split"]["lane"] == bench.V_FAIL


def test_h1_withholds_the_writer_arms_h5_and_h6() -> None:
    # #1795 H1: under 1 per 100, A1 is substrate-starved, not a lane
    # verdict, on any hypothesis. A2 doesn't depend on the writers.
    def rows(null_rel: int = 0) -> list[Row]:
        out = _split_rows([2, 2, 2, 2], [3, 3, 2, 4], [3, 3, 2, 4], null_rel=null_rel)
        for p, r in enumerate([3, 3, 2, 4]):
            out += [(f"z{p}-{k}", f"p{p}", {bench.ARM_A2: k + 1}, True, k < r, k < r)
                    for k in range(10)]
        return out

    used = [{"id": "b", "used": True}, {"id": "c", "used": True}]
    proxy = _proxy_rows(4 * [{
        "A0": [{"id": "a", "used": True}], "A1": used, "A1W": used, "A2": used}])
    held = _score(rows(), proxy_reports=[proxy])["hypotheses"]
    assert held["H5"] == {"A1": bench.V_PASS, "A1W": bench.V_PASS, "A2": bench.V_PASS}
    assert {a: v["verdict"] for a, v in held["H6"].items()} == {
        "A1": bench.V_AGREE, "A1W": bench.V_AGREE, "A2": bench.V_AGREE}
    for census, want in ((_H1_FAIL, bench.V_STARVED), (None, bench.V_NEEDS_CENSUS)):
        rep = _score(rows(), census_report=census, proxy_reports=[proxy])
        hyp = rep["hypotheses"]
        assert hyp["H5"] == {"A1": want, "A1W": want, "A2": bench.V_PASS}
        assert {a: v["verdict"] for a, v in hyp["H6"].items()} == {
            "A1": want, "A1W": want, "A2": bench.V_AGREE}
        assert hyp["H6"]["A1"]["grader_signs"] == [1, 1]
        assert rep["arms"]["A1"]["h5_verdict"] == bench.V_PASS
    # A failed H0 outranks the substrate.
    bad = _score(rows(null_rel=3), census_report=_H1_FAIL, proxy_reports=[proxy])
    assert set(bad["hypotheses"]["H5"].values()) == {bench.V_UNRELIABLE}
    assert {v["verdict"] for v in bad["hypotheses"]["H6"].values()} == {bench.V_UNRELIABLE}


def test_a_failed_h0_makes_both_halves_of_the_h2_split_unreliable() -> None:
    # The lane half passes on these labels when H0 holds.
    rows = _split_rows([2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 2, 4], null_rel=3)
    for census in (_H1_PASS, _H1_FAIL):
        rep = _score(rows, census_report=census)
        assert rep["hypotheses"]["H0"] == bench.V_FAIL
        assert rep["hypotheses"]["H2_split"] == {
            "writers": bench.V_UNRELIABLE, "lane": bench.V_UNRELIABLE}
        assert rep["arms"]["A1"]["vs_A1W"]["gain_verdict"] == bench.V_UNRELIABLE


def test_an_unclear_null_item_is_left_out_of_the_null_rate() -> None:
    # 5 null items: relevant, unclear, and 3 not relevant. The rate is 1 of
    # the 4 graded, 0.25, over the 0.20 ceiling. Counting the unclear one
    # would give 1 of 5, on the ceiling.
    labels: list[bool | None] = [True, None, False, False, False]
    rows: list[Row] = [(f"n{k}", "p0", None, False, v, v) for k, v in enumerate(labels)]
    rep = _score(rows)
    assert rep["null_arm"][0]["graded"] == 4
    assert rep["null_arm"][0]["rate"] == _approx(0.25)
    assert rep["hypotheses"]["H0"] == bench.V_FAIL


def test_a_null_rate_on_the_ceiling_holds_h0() -> None:
    on = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=2))
    assert on["null_arm"][0]["rate"] == _approx(0.2)
    assert on["hypotheses"]["H0"] == bench.V_PASS


def test_a_mean_gain_on_the_kill_bar_is_not_killed() -> None:
    assert bench.gain_verdict(0.1, (0.05, 0.2), 0.5) == bench.V_INCONCLUSIVE
    assert bench.gain_verdict(0.09, (0.05, 0.2), 0.5) == bench.V_FAIL
    assert bench.gain_verdict(0.3, (0.0, 0.5), 0.5) == bench.V_FAIL


def test_a_bridging_share_on_the_bar_passes_h4() -> None:
    # 16 relevant additions, 2 per prompt bridging: 8 of 16.
    rep = _score(_hand([2, 2, 2, 2], [4, 4, 4, 4], arm=bench.ARM_A3, bridging=2))
    assert rep["arms"]["A3"]["graders"][0]["bridging_share"] == _approx(0.5)
    assert rep["hypotheses"]["H4"] == bench.V_PASS


def _substrate_store(tmp_path: Path, beliefs: int, edges: int = 1) -> Path:
    """`beliefs` active beliefs and `edges` CONTRADICTS edges, a probed kind."""
    path = tmp_path / f"sub{beliefs}-{edges}" / "memory.db"
    path.parent.mkdir(parents=True)
    s = MemoryStore(str(path))
    for i in range(beliefs):
        _add(s, f"b{i:03d}", _filler(i))
    for k in range(edges):
        s.insert_edge(Edge(src=f"b{2 * k:03d}", dst=f"b{2 * k + 1:03d}",
                           type=EDGE_CONTRADICTS, weight=1.0))
    s.close()
    return path


def test_h1_passes_at_exactly_one_probed_edge_per_100_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_writers(_db: Path) -> dict[str, dict[str, int]]:
        return {}

    monkeypatch.setattr(bench, "apply_writers", no_writers)
    on = bench.census([_substrate_store(tmp_path, 100)], tmp_path, writers=True)
    assert on["median_probed_per_100_beliefs"] == _approx(1.0)
    assert on["H1"] == bench.V_PASS
    under = bench.census([_substrate_store(tmp_path, 101)], tmp_path, writers=True)
    assert under["H1"] == bench.V_FAIL


def test_h1_reads_the_median_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_writers(_db: Path) -> dict[str, dict[str, int]]:
        return {}

    monkeypatch.setattr(bench, "apply_writers", no_writers)
    # 0, 3, and 1 probed edges per 100 beliefs: the median is 1, a pass.
    stores = [_substrate_store(tmp_path, 100, n) for n in (0, 3, 1)]
    rep = bench.census(stores, tmp_path, writers=True)
    assert [r["probed_per_100_beliefs"] for r in rep["stores"]] == [0.0, 3.0, 1.0]
    assert rep["median_probed_per_100_beliefs"] == _approx(1.0)
    assert rep["H1"] == bench.V_PASS
    # Two stores under the bar outvote one far above it.
    starved = bench.census(
        [_substrate_store(tmp_path, 200, n) for n in (0, 1, 50)], tmp_path, writers=True)
    assert starved["median_probed_per_100_beliefs"] == _approx(0.5)
    assert starved["H1"] == bench.V_FAIL
    # With 4 stores, the median is the mean of the middle two: 0.5 and 1.5.
    even = bench.census(
        [_substrate_store(tmp_path, 400, n) for n in (0, 2, 6, 100)], tmp_path, writers=True)
    assert even["median_probed_per_100_beliefs"] == _approx(1.0)
    assert even["H1"] == bench.V_PASS


def test_the_census_records_the_snapshot_run_records(tmp_path: Path) -> None:
    path = _store(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    snapshot = bench.prepare_snapshot(path, work)[1]
    rep = bench.census([path], tmp_path, writers=True)
    assert rep["stores"][0]["snapshot"] == snapshot


def test_h1_is_scored_only_on_a_census_of_the_keys_stores() -> None:
    # #1795 H1 is the median of the stores under test: a census of other
    # stores, or a hand-written verdict, can't release the writer arms.
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    for census, match in (
        ({"H1": "pass"}, "lists no store"),
        (_census(2.0, snapshots=("other",)), "no census row for the key's snapshot s0"),
        (_census(2.0, 2.0, snapshots=(_SNAP, "other")), "no key rows for the census row's"),
        (_census(2.0, 2.0, snapshots=(_SNAP, _SNAP)), "two census rows record snapshot s0"),
        ({"writers": True, "stores": [{"probed_per_100_beliefs": 2.0}]}, "records no snapshot"),
        ({"writers": True, "stores": [{"snapshot": _SNAP}]}, "no probed_per_100_beliefs"),
        ({"writers": True, "stores": [{"snapshot": _SNAP, "probed_per_100_beliefs": True}]},
         "no probed_per_100_beliefs"),
    ):
        with pytest.raises(bench.InputError, match=match):
            _score(rows, census_report=census)


def test_score_takes_h1_from_the_census_rows() -> None:
    # The rows decide, not the census's own H1 field, and a census run
    # without the writers measures no H1.
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    for census, h1, h2 in (
        ({**_census(0.5), "H1": "pass"}, bench.V_FAIL, bench.V_STARVED),
        ({**_census(1.0), "H1": "fail"}, bench.V_PASS, bench.V_PASS),
        ({**_census(2.0, writers=False), "H1": "pass"}, bench.V_NOT_MEASURED,
         bench.V_NEEDS_CENSUS),
    ):
        hyp = _score(rows, census_report=census)["hypotheses"]
        assert (hyp["H1"], hyp["H2"]) == (h1, h2)


def test_a_pooled_key_takes_h1_from_the_median_of_its_stores() -> None:
    def key_of(rates: tuple[float, ...]) -> tuple[bench.Key, dict[str, bool | None]]:
        items = [bench.KeyItem(id=f"i{n}", prompt_id="p", arms={bench.ARM_A0: 1},
                               shares_prompt_stem=True, snapshot=f"s{n}")
                 for n in range(len(rates))]
        stores = {f"s{n}": bench.KeyStore(prompts_sampled=1, arms_run=("A0", "A1"))
                  for n in range(len(rates))}
        return bench.make_key(items, stores), {i.id: True for i in items}

    # The median of 0, 3, and 1 per 100 beliefs is 1, a pass, and of 0,
    # 0.5, and 50 it is 0.5, a fail.
    for rates, want in (((0.0, 3.0, 1.0), bench.V_PASS), ((0.0, 0.5, 50.0), bench.V_FAIL)):
        key, labels = key_of(rates)
        census = _census(*rates, snapshots=tuple(sorted(key.stores)))
        rep = bench.score(key, *_both(labels), seed=1, census_report=census)
        assert rep["hypotheses"]["H1"] == want


def test_the_interval_is_the_2_5_and_97_5_percentile_of_the_resamples() -> None:
    # #1802: integer gains tie the sorted means around both percentile
    # indices, so an index off by 1 or 2 passed. Random gains don't tie.
    source = random.Random(1802)
    gains = [source.random() for _ in range(40)]
    rng = random.Random(7)
    means = sorted(sum(gains[rng.randrange(40)] for _ in range(40)) / 40
                   for _ in range(10_000))
    assert len(set(means[247:252])) == 5 and len(set(means[9748:9753])) == 5
    # Index floor(0.025 * 9999) and ceil(0.975 * 9999).
    assert bench.paired_bootstrap_ci(gains, 7) == (means[249], means[9750])
    # A 90% interval would end at indices 499 and 9500.
    assert means[249] < means[499] and means[9500] < means[9750]


def test_the_bars_are_the_pre_registered_ones() -> None:
    assert (bench.KAPPA_BAR, bench.NULL_RATE_CEILING, bench.SUBSTRATE_BAR, bench.GAIN_BAR,
            bench.GAIN_KILL, bench.GAINING_SHARE_BAR, bench.BRIDGING_SHARE_BAR,
            bench.PRECISION_DROP_BAR) == (0.60, 0.20, 1.0, 0.3, 0.1, 0.10, 0.5, 0.05)
    assert bench.BOOTSTRAP_RESAMPLES == 10_000


def test_the_docstrings_cite_the_amendment_and_its_judged_sets() -> None:
    for doc in (bench.__doc__, bench.score.__doc__):
        text = " ".join((doc or "").split())
        assert "2026-10-23 amendment" in text
        assert "H2, H3, H4, and `H2_split`" in text and "whole packed" in text
        assert "H5" in text and "top 10" in text


def _proxy_rows(arms: list[dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """A proxy report on the `_SNAP` store whose prompt p<n> measured `arms[n]`."""
    return {"snapshot": _SNAP, "prompts": [
        {"prompt_id": f"p{n}", "later_turns": 1, "arms": dict(a)} for n, a in enumerate(arms)]}


def test_h6_compares_the_proxy_sign_with_each_grader() -> None:
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    proxy = _proxy_rows([
        {"A0": [{"id": "a", "used": True}], "A1": [{"id": "b", "used": u}]}
        for u in (False, False, True, False)])
    rep = _score(rows, proxy_reports=[proxy])
    assert rep["hypotheses"]["H6"]["A1"]["verdict"] == bench.V_DISAGREE
    proxy["prompts"][0]["arms"]["A1"] = [{"id": "b", "used": True}, {"id": "c", "used": True}]
    proxy["prompts"][1]["arms"]["A1"] = [{"id": "b", "used": True}, {"id": "c", "used": True}]
    rep = _score(rows, proxy_reports=[proxy])
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [1, 1], "verdict": bench.V_AGREE}
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3), proxy_reports=[proxy])
    assert rep["hypotheses"]["H6"]["A1"]["verdict"] == bench.V_UNRELIABLE


def test_h6_needs_the_proxy_to_agree_with_both_graders() -> None:
    # Grader B calls only A1's first 2 items per prompt relevant: no gain.
    rows = [(r[0], r[1], r[2], r[3], r[4], r[5] if not r[0].startswith("x")
             else int(r[0].split("-")[1]) < 2) for r in _hand([2, 2, 2, 2], [3, 3, 2, 4])]
    proxy = _proxy_rows(4 * [
        {"A0": [{"id": "a", "used": False}], "A1": [{"id": "a", "used": True}]}])
    rep = _score(rows, proxy_reports=[proxy])
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [1, 0], "verdict": bench.V_DISAGREE}


def test_h6_takes_each_graders_sign_from_the_packed_gain() -> None:
    # A1 adds a relevant belief past rank 10 on every prompt: a top-10 gain
    # of 0 but a packed gain of 1, the gain H2 is decided on.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A1": k + 1}, True, k < 2, k < 2)
                 for k in range(10)]
        rows.append((f"r{p}", f"p{p}", {"A1": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    proxy = _proxy_rows(4 * [
        {"A0": [{"id": "a", "used": False}], "A1": [{"id": "a", "used": True}]}])
    rep = _score(rows, proxy_reports=[proxy])
    assert rep["arms"]["A1"]["graders"][0]["top10_mean_gain"] == _approx(0.0)
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [1, 1], "verdict": bench.V_AGREE}


# --- Proxy -------------------------------------------------------------------


def test_the_proxy_finds_a_cited_or_restated_belief_in_a_later_turn(tmp_path: Path) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    turns = bench.read_turns(bench.transcript_files(path))
    prompt: dict[str, Any] = {"prompt_id": "p1", "text": archive[4]["text"], "session_id": "s1",
              "turn_id": "t004", "ts": archive[4]["ts"], "context": []}
    early: dict[str, Any] = {"prompt_id": "p2", "text": live[1]["text"], "session_id": "s2",
             "turn_id": "t007", "ts": live[1]["ts"], "context": []}
    runs = {bench.ARM_A0: {"arm": "A0", "prompts": [
        {"prompt_id": "p1", "prompt": prompt["text"], "ranked": ["seed", "w1"]},
        {"prompt_id": "p2", "prompt": early["text"], "ranked": ["w5", "w1"]}]}}
    work = tmp_path / "w"
    work.mkdir()
    rep = bench.proxy(runs, [prompt, early], turns, bench.prepare_copy(path, work),
                      snapshot=_SNAP)
    used = {r["prompt_id"]: {b["id"]: b["used"] for b in r["arms"]["A0"]}
            for r in rep["prompts"]}
    # t005 cites `seed`; w1 is only restated in an earlier turn of s1.
    assert used["p1"] == {"seed": True, "w1": False}
    # t008 restates most of w5's stems; w1 isn't in s2 after t007.
    assert used["p2"] == {"w5": True, "w1": False}


def test_the_proxy_covers_the_whole_packed_result(tmp_path: Path) -> None:
    # The packed gain counts beliefs past the top 10, so the proxy reports them too.
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    turns = bench.read_turns(bench.transcript_files(path))
    prompt: dict[str, Any] = {"prompt_id": "p1", "text": archive[4]["text"], "session_id": "s1",
                              "turn_id": "t004", "ts": archive[4]["ts"], "context": []}
    ranked = [f"f{i:02d}" for i in range(bench.TOP_K)] + ["seed"]
    runs = {bench.ARM_A0: {"arm": "A0", "prompts": [
        {"prompt_id": "p1", "prompt": prompt["text"], "ranked": ranked}]}}
    work = tmp_path / "w"
    work.mkdir()
    rep = bench.proxy(runs, [prompt], turns, bench.prepare_copy(path, work), snapshot=_SNAP)
    used = {b["id"]: b["used"] for b in rep["prompts"][0]["arms"]["A0"]}
    assert len(used) == bench.TOP_K + 1 and used["seed"] is True


def test_the_proxy_does_not_count_the_prompt_turn_as_a_later_turn(tmp_path: Path) -> None:
    # The prompt restates `seed` word for word; only the reply comes after it.
    text = "the kestrel hovers above the meadow before it dives"
    turns = [
        bench.Turn(seq=0, role="user", text=text, session_id="s1", ts="t0", turn_id="u1"),
        bench.Turn(seq=1, role="assistant", text="done", session_id="s1", ts="t1",
                   turn_id="u2"),
    ]
    prompt: dict[str, Any] = {"prompt_id": "p", "text": text, "session_id": "s1",
                              "turn_id": "u1", "ts": "t0", "context": []}
    runs = {bench.ARM_A0: {"arm": "A0", "prompts": [
        {"prompt_id": "p", "prompt": text, "ranked": ["seed"]}]}}
    work = tmp_path / "w"
    work.mkdir()
    rep = bench.proxy(runs, [prompt], turns, bench.prepare_copy(_store(tmp_path), work),
                      snapshot=_SNAP)
    row = rep["prompts"][0]
    assert row["later_turns"] == 1
    assert row["arms"]["A0"] == [{"id": "seed", "used": False}]


def test_the_proxy_refuses_a_prompt_missing_from_the_transcripts_beside_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # A copy of the database alone has no transcripts beside it, so the
    # proxy would find no later turn and call every belief unused.
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    d = tmp_path / "out"
    d.mkdir()
    prompts = d / "prompts.jsonl"
    assert bench.main(["sample", "--store", str(path), "--seed", "1",
                       "--out", str(prompts)]) == 0
    frozen = tmp_path / "frozen" / "memory.db"
    frozen.parent.mkdir()
    bench.copy_store(path, frozen)
    run = str(d / "A0.json")
    assert bench.main(["run", "--store", str(frozen), "--prompts", str(prompts),
                       "--arm", "A0", "--out", run]) == 0
    usage = ["proxy", "--store", str(frozen), "--runs", run, "--prompts", str(prompts),
             "--out", str(d / "proxy.json")]
    capsys.readouterr()
    assert bench.main(usage) == 1
    assert "not in the transcripts beside the store" in capsys.readouterr().err
    assert not (d / "proxy.json").exists()
    shutil.copytree(path.parent / "transcripts", frozen.parent / "transcripts")
    assert bench.main(usage) == 0
    rows = json.loads((d / "proxy.json").read_text(encoding="utf-8"))["prompts"]
    assert any(b["used"] for r in rows for b in r["arms"]["A0"])


def test_the_proxy_finds_the_prompt_by_session_turn_id_or_timestamp_and_text() -> None:
    find = bench._find_turn  # pyright: ignore[reportPrivateUsage]
    turns = [
        bench.Turn(seq=0, role="user", text="hi", session_id="s1", ts="t0", turn_id="u1"),
        bench.Turn(seq=1, role="user", text="hi", session_id="s2", ts="t0", turn_id="u1"),
        bench.Turn(seq=2, role="user", text="go", session_id="s2", ts="t1", turn_id=None),
    ]
    # Turn ids repeat across sessions; the session picks the turn.
    assert find(turns, {"turn_id": "u1", "session_id": "s2"}) == 1
    # Without a turn id, session, timestamp, and text must all match.
    assert find(turns, {"session_id": "s2", "ts": "t1", "text": "go"}) == 2
    assert find(turns, {"session_id": "s2", "ts": "t0", "text": "go"}) is None


def test_proxy_overlap_bar() -> None:
    stems = frozenset({"a", "b", "c", "d", "e"})
    assert bench.proxy_used("x1", stems, [("no", frozenset({"a", "b", "c"}))]) is True
    assert bench.proxy_used("x1", stems, [("no", frozenset({"a", "b"}))]) is False
    assert bench.proxy_used("x1", stems, [("see x1", frozenset())]) is True


# --- Source store safety and CLI ---------------------------------------------


def _target(database: str) -> Path | None:
    """The file a `sqlite3.connect` argument opens, URI or plain path."""
    if database == ":memory:":
        return None
    if database.startswith("file:"):
        parsed = urlparse(database)
        return Path(unquote(parsed.path)).resolve()
    return Path(database).resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def connect_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, bool]]]:
    calls: list[tuple[str, bool]] = []
    real: Callable[..., sqlite3.Connection] = sqlite3.connect

    def spy(database: Any, *args: Any, **kw: Any) -> sqlite3.Connection:
        calls.append((str(database), bool(kw.get("uri", False))))
        return real(database, *args, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy)
    yield calls


def test_no_subcommand_writes_or_opens_the_source_for_writing(
    tmp_path: Path, connect_spy: list[tuple[str, bool]],
) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    before = _sha(path)
    files = sorted(p.name for p in path.parent.iterdir())
    connect_spy.clear()  # building the store opened it read-write
    d = tmp_path / "out"
    d.mkdir()
    prompts = d / "prompts.jsonl"
    runs: list[str] = []
    assert bench.main(["census", "--store", str(path), "--writers"]) == 0
    assert bench.main(["sample", "--store", str(path), "--seed", "1",
                       "--out", str(prompts)]) == 0
    for arm in bench.ARMS:
        runs.append(str(d / f"{arm}.json"))
        assert bench.main(["run", "--store", str(path), "--prompts", str(prompts),
                           "--arm", arm, "--out", runs[-1]]) == 0
    assert bench.main(["packet", "--runs", *runs, "--store", str(path), "--seed", "2",
                       "--out", str(d / "pk"), "--key", str(d / "key.jsonl")]) == 0
    assert sorted(p.name for p in (d / "pk").iterdir()) == ["instructions.txt", "packet.jsonl"]
    assert bench.main(["packet", "--runs", *runs, "--store", str(path), "--seed", "2",
                       "--out", str(d / "pk"), "--key", str(d / "pk" / "key.jsonl")]) == 1
    assert bench.main(["proxy", "--store", str(path), "--runs", *runs, "--prompts",
                       str(prompts), "--out", str(d / "proxy.json")]) == 0
    assert _sha(path) == before
    # Not writing the source covers the files beside it too.
    assert sorted(p.name for p in path.parent.iterdir()) == files
    ro = read_only_uri(str(path))
    on_source = [c for c in connect_spy if _target(c[0]) == path.resolve()]
    assert on_source and all(uri and db.startswith(ro) for db, uri in on_source)


def _wal_header(path: Path) -> bytes:
    """The header's read and write format bytes, 2 and 2 in WAL mode."""
    return path.read_bytes()[18:20]


def test_a_copy_of_an_idle_wal_store_adds_no_file_beside_it(tmp_path: Path) -> None:
    # A `mode=ro` reader of a WAL store with no log beside it creates the
    # shared-memory file and an empty log, and can't remove them.
    src = tmp_path / "src" / "memory.db"
    src.parent.mkdir()
    MemoryStore(str(src)).close()
    assert _wal_header(src) == b"\x02\x02"
    assert [p.name for p in src.parent.iterdir()] == ["memory.db"]
    bench.copy_store(src, tmp_path / "copy.db")
    assert [p.name for p in src.parent.iterdir()] == ["memory.db"]
    with sqlite3.connect(tmp_path / "copy.db") as c:
        assert c.execute("SELECT count(*) FROM beliefs").fetchone() == (0,)


def test_a_copy_of_a_store_in_use_reads_its_log(tmp_path: Path) -> None:
    src = tmp_path / "src" / "memory.db"
    src.parent.mkdir()
    MemoryStore(str(src)).close()
    writer = sqlite3.connect(src)
    try:
        writer.execute("CREATE TABLE late (x)")
        writer.execute("INSERT INTO late VALUES (1)")
        writer.commit()
        assert (src.parent / "memory.db-wal").stat().st_size > 0
        bench.copy_store(src, tmp_path / "copy.db")
    finally:
        writer.close()
    with sqlite3.connect(tmp_path / "copy.db") as c:
        assert c.execute("SELECT x FROM late").fetchall() == [(1,)]


@pytest.mark.parametrize("link_name", ["elsewhere/memory.db", "src/other.db"])
def test_a_copy_through_a_link_to_a_store_in_use_reads_its_log(
    tmp_path: Path, link_name: str,
) -> None:
    # SQLite keeps the log beside the file a link points to, so the copy
    # must look for it there, not beside the link.
    src = tmp_path / "src" / "memory.db"
    src.parent.mkdir()
    MemoryStore(str(src)).close()
    link = tmp_path / link_name
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(src)
    writer = sqlite3.connect(src)
    try:
        writer.execute("CREATE TABLE late (x)")
        writer.execute("INSERT INTO late VALUES (1)")
        writer.commit()
        assert (src.parent / "memory.db-wal").stat().st_size > 0
        bench.copy_store(link, tmp_path / "copy.db")
    finally:
        writer.close()
    with sqlite3.connect(tmp_path / "copy.db") as c:
        assert c.execute("SELECT x FROM late").fetchall() == [(1,)]


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_a_store_with_a_sidecar_file_is_not_read_as_immutable(
    tmp_path: Path, connect_spy: list[tuple[str, bool]], suffix: str,
) -> None:
    src = tmp_path / "src" / "memory.db"
    src.parent.mkdir()
    MemoryStore(str(src)).close()
    src.with_name(src.name + suffix).touch()
    connect_spy.clear()
    bench.copy_store(src, tmp_path / "copy.db")
    assert (read_only_uri(str(src)), True) in connect_spy
    assert not any("immutable" in db for db, _uri in connect_spy)


@pytest.mark.parametrize("change", ["mtime", "log"])
def test_a_store_that_changes_during_an_idle_copy_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    # The idle copy takes no lock, so a writer that starts during it must
    # fail the copy rather than tear it.
    src = tmp_path / "src" / "memory.db"
    src.parent.mkdir()
    MemoryStore(str(src)).close()
    real: Callable[..., sqlite3.Connection] = sqlite3.connect

    def touching(database: Any, *args: Any, **kw: Any) -> sqlite3.Connection:
        conn = real(database, *args, **kw)
        if "immutable=1" in str(database):
            if change == "log":
                src.with_name(src.name + "-wal").touch()
            else:
                st = src.stat()
                os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return conn

    monkeypatch.setattr(sqlite3, "connect", touching)
    with pytest.raises(bench.InputError, match="changed while it was copied"):
        bench.copy_store(src, tmp_path / "copy.db")
    assert not (tmp_path / "copy.db").exists()


def test_every_subcommand_has_help() -> None:
    for cmd in ("census", "sample", "run", "packet", "score", "proxy"):
        with pytest.raises(SystemExit) as exc:
            bench.main([cmd, "--help"])
        assert exc.value.code == 0


def test_the_usage_examples_parse_and_give_score_a_census() -> None:
    # #1802: a score example without --census leaves H1 not measured and
    # the lane verdicts needing a census, so every one passes it, and the
    # census example saves the file it reads.
    marker = "-m benchmarks.real_store_ab_1795 "
    lines = [ln.split(marker, 1)[1] for ln in (bench.__doc__ or "").splitlines()
             if marker in ln]
    parser = bench.build_parser()
    seen: set[str] = set()
    for line in lines:
        args = parser.parse_args(shlex.split(line))
        seen.add(args.cmd)
        if args.cmd == "score":
            assert args.census == Path("census.json"), line
        if args.cmd == "census":
            assert args.writers and args.out == Path("census.json"), line
    assert seen == {"census", "sample", "run", "packet", "score", "proxy"}


def test_census_out_writes_the_json_it_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = _store(tmp_path)
    out = tmp_path / "census.json"
    assert bench.main(["census", "--store", str(path), "--writers", "--out", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert bench.main(["census", "--store", str(path), "--writers"]) == 0
    printed = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == printed
    assert json.loads(printed)["stores"]


def test_bad_input_exits_1(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.db")
    assert bench.main(["census", "--store", missing]) == 1
    assert bench.main(["sample", "--store", missing, "--seed", "1",
                       "--out", str(tmp_path / "p.jsonl")]) == 1
    # A store with no transcripts beside it has nothing to sample.
    assert bench.main(["sample", "--store", str(_store(tmp_path)), "--seed", "1",
                       "--out", str(tmp_path / "p.jsonl")]) == 1
    assert bench.main(["sample", "--store", str(_store(tmp_path / "b")), "--seed", "1",
                       "--max", "0", "--out", str(tmp_path / "p.jsonl")]) == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1}\n')
    assert bench.main(["score", "--key", str(bad), "--labels", str(bad), str(bad)]) == 1
    with pytest.raises(SystemExit) as exc:
        bench.main(["run", "--store", missing, "--prompts", str(bad), "--arm", "A9",
                    "--out", str(tmp_path / "o.json")])
    assert exc.value.code == 1


def _run_file(tmp_path: Path, name: str, arm: str, pids: list[str],
              snapshot: str = "s0") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"arm": arm, "snapshot": snapshot, "prompts": [
        {"prompt_id": p, "prompt": "q", "ranked": ["seed"]} for p in pids]}), encoding="utf-8")
    return path


def _add_late_belief(path: Path) -> None:
    s = MemoryStore(str(path))
    _add(s, "late", "the kestrel meadow gained a late note")
    s.close()


def test_a_run_records_the_snapshot_it_read(tmp_path: Path) -> None:
    path = _store(tmp_path)

    def run_on(arm: str, work: str) -> dict[str, Any]:
        (tmp_path / work).mkdir()
        return bench.run_arm(path, _prompts(), arm, tmp_path / work)

    first = run_on(bench.ARM_A0, "w1")
    again = run_on(bench.ARM_A2, "w2")
    assert first["snapshot"] == again["snapshot"]
    # `packet` and `proxy` fingerprint their own copy the same way.
    (tmp_path / "pk").mkdir()
    assert bench.prepare_snapshot(path, tmp_path / "pk")[1] == first["snapshot"]
    _add_late_belief(path)
    later = run_on(bench.ARM_A2, "w3")
    assert later["snapshot"] != first["snapshot"]
    assert "late" in later["prompts"][0]["ranked"]


def test_the_snapshot_is_taken_before_the_copy_migrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A migration may stamp the copy with the time it ran, so a fingerprint
    # taken after it would differ between two copies of one source.
    path = _store(tmp_path)
    real = bench.MemoryStore
    stamps = iter(range(1, 10))

    def migrating(db: str, **kw: Any) -> MemoryStore:
        if not kw.get("read_only"):
            conn = sqlite3.connect(db)
            conn.execute(f"PRAGMA user_version = {next(stamps)}")
            conn.close()
        return real(db, **kw)

    monkeypatch.setattr(bench, "MemoryStore", migrating)
    snaps: list[str] = []
    for work in (tmp_path / "m1", tmp_path / "m2"):
        work.mkdir()
        snaps.append(bench.prepare_snapshot(path, work)[1])
    assert snaps[0] == snaps[1]


def test_a_lock_that_expires_between_two_copies_changes_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The read-write open of each copy expires the time-boxed locks due by
    # the clock. A lock that expires between two subcommands leaves the
    # copies' bytes equal but hands the later arms a non-locked belief.
    path = _store(tmp_path)
    s = MemoryStore(str(path))
    _add(s, "boxed", "the kestrel meadow lock ends at the new year", lock=LOCK_USER,
         expires="2030-01-01T00:00:00+00:00")
    s.close()
    real = MemoryStore.sweep_expired_locks
    clock = ["2029-12-31T00:00:00+00:00"]

    def sweep(self: MemoryStore, *, now: str | None = None) -> int:
        return real(self, now=clock[0])

    monkeypatch.setattr(MemoryStore, "sweep_expired_locks", sweep)
    snaps: list[str] = []
    for n, now in enumerate(("2029-12-31T00:00:00+00:00", "2029-12-31T12:00:00+00:00",
                             "2030-01-02T00:00:00+00:00")):
        clock[0] = now
        work = tmp_path / f"c{n}"
        work.mkdir()
        db, snap = bench.prepare_snapshot(path, work)
        snaps.append(snap)
        locked = {bid for bid, _c, lock in bench.active_beliefs(db) if lock != LOCK_NONE}
        assert ("boxed" in locked) == (n < 2)
    assert snaps[0] == snaps[1] != snaps[2]


def test_read_runs_rejects_runs_of_different_snapshots(tmp_path: Path) -> None:
    # A belief the store gains between two runs would count as one arm's gain.
    a0 = _run_file(tmp_path, "a0.json", bench.ARM_A0, ["p1"], snapshot="s0")
    a2 = _run_file(tmp_path, "a2.json", bench.ARM_A2, ["p1"], snapshot="s1")
    with pytest.raises(bench.InputError, match="different snapshots"):
        bench.read_runs([a0, a2])
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"arm": "A0", "prompts": []}), encoding="utf-8")
    with pytest.raises(bench.InputError, match="a snapshot"):
        bench.read_run(bare)


def test_packet_and_proxy_refuse_a_store_that_changed_since_the_runs(
    tmp_path: Path,
) -> None:
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    d = tmp_path / "out"
    d.mkdir()
    prompts = d / "prompts.jsonl"
    assert bench.main(["sample", "--store", str(path), "--seed", "1",
                       "--out", str(prompts)]) == 0
    runs: list[str] = []
    for arm in (bench.ARM_A0, bench.ARM_A2):
        runs.append(str(d / f"{arm}.json"))
        assert bench.main(["run", "--store", str(path), "--prompts", str(prompts),
                           "--arm", arm, "--out", runs[-1]]) == 0
    packet = ["packet", "--runs", *runs, "--store", str(path), "--seed", "2",
              "--out", str(d / "pk"), "--key", str(d / "key.jsonl")]
    usage = ["proxy", "--store", str(path), "--runs", *runs, "--prompts", str(prompts),
             "--out", str(d / "proxy.json")]
    assert bench.main(packet) == 0 and bench.main(usage) == 0
    _add_late_belief(path)
    (d / "key.jsonl").unlink()
    (d / "proxy.json").unlink()
    assert bench.main(packet) == 1 and bench.main(usage) == 1
    assert not (d / "key.jsonl").exists() and not (d / "proxy.json").exists()


def test_the_proxy_report_records_the_snapshot_the_key_does(tmp_path: Path) -> None:
    # score matches a proxy report to the key rows of its store by snapshot.
    path = _store(tmp_path)
    archive, live = _session_turns()
    _write_transcripts(path, archive, live)
    d = tmp_path / "out"
    d.mkdir()
    prompts = d / "prompts.jsonl"
    assert bench.main(["sample", "--store", str(path), "--seed", "1",
                       "--out", str(prompts)]) == 0
    runs: list[str] = []
    for arm in (bench.ARM_A0, bench.ARM_A2):
        runs.append(str(d / f"{arm}.json"))
        assert bench.main(["run", "--store", str(path), "--prompts", str(prompts),
                           "--arm", arm, "--out", runs[-1]]) == 0
    assert bench.main(["packet", "--runs", *runs, "--store", str(path), "--seed", "2",
                       "--out", str(d / "pk"), "--key", str(d / "key.jsonl")]) == 0
    assert bench.main(["proxy", "--store", str(path), "--runs", *runs, "--prompts",
                       str(prompts), "--out", str(d / "proxy.json")]) == 0
    report = json.loads((d / "proxy.json").read_text(encoding="utf-8"))
    run = json.loads(Path(runs[0]).read_text(encoding="utf-8"))
    assert report["snapshot"] == run["snapshot"]
    key = bench.read_key(d / "key.jsonl")
    assert set(key.stores) == {i.snapshot for i in key.items} == {report["snapshot"]}


def test_read_runs_rejects_a_repeated_arm_a_missing_a0_and_other_prompts(
    tmp_path: Path,
) -> None:
    a0 = _run_file(tmp_path, "a0.json", bench.ARM_A0, ["p1", "p2"])
    a1 = _run_file(tmp_path, "a1.json", bench.ARM_A1, ["p2", "p1"])
    assert set(bench.read_runs([a0, a1])) == {"A0", "A1"}
    again = _run_file(tmp_path, "a1b.json", bench.ARM_A1, ["p1", "p2"])
    with pytest.raises(bench.InputError, match="two runs for arm A1"):
        bench.read_runs([a0, a1, again])
    with pytest.raises(bench.InputError, match="no A0 run"):
        bench.read_runs([a1])
    other = _run_file(tmp_path, "a2.json", bench.ARM_A2, ["p1", "p3"])
    with pytest.raises(bench.InputError, match="different prompts"):
        bench.read_runs([a0, other])


def _store_row(arms_run: list[str]) -> dict[str, Any]:
    return {"kind": "store", "snapshot": _SNAP, "prompts_sampled": 1, "arms_run": arms_run}


def test_read_key_rejects_a_key_without_an_a0_item(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        _store_row(["A0", "A1"]),
        {"kind": "item", "id": "i1", "prompt_id": "p", "arms": {"A1": 1},
         "shares_prompt_stem": True, "snapshot": _SNAP},
        {"kind": "item", "id": "i2", "prompt_id": "p", "arms": None,
         "shares_prompt_stem": False, "snapshot": _SNAP}]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(bench.InputError, match="no A0 item"):
        bench.read_key(path)
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(bench.InputError, match="no A0 item"):
        bench.read_key(path)
    rows[1]["arms"] = {"A0": 3, "A1": 1}
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [i.id for i in bench.read_key(path).items] == ["i1", "i2"]


def test_read_key_rejects_a_row_without_a_sampled_count_or_snapshot(tmp_path: Path) -> None:
    store = _store_row(["A0"])
    item: dict[str, Any] = {"kind": "item", "id": "i1", "prompt_id": "p", "arms": {"A0": 1},
                            "shares_prompt_stem": True, "snapshot": _SNAP}
    path = tmp_path / "key.jsonl"

    def write() -> None:
        path.write_text(json.dumps(store) + "\n" + json.dumps(item) + "\n", encoding="utf-8")

    write()
    key = bench.read_key(path)
    assert key.stores == {_SNAP: bench.KeyStore(prompts_sampled=1, arms_run=("A0",))}
    assert [i.snapshot for i in key.items] == [_SNAP]
    cases: list[tuple[dict[str, Any], str, object, str]] = [
        (store, "prompts_sampled", b, "prompts_sampled and arms_run") for b in (None, 0, True, "1")]
    cases += [(row, "snapshot", b, "need a snapshot") for row in (store, item) for b in (None, "", 1)]
    cases += [(row, "kind", b, "need kind store or item") for row in (store, item)
              for b in (None, "items")]
    cases += [(item, "snapshot", "s9", "item i1 is of snapshot s9, which has no store row")]
    for row, field, bad, match in cases:
        good = row[field]
        row[field] = bad
        write()
        with pytest.raises(bench.InputError, match=match):
            bench.read_key(path)
        row[field] = good


def test_read_key_rejects_a_repeated_item_id(tmp_path: Path) -> None:
    # A store's item rows concatenated twice would double every paired gain
    # and the kappa sample.
    rows = _store_key_rows("s1", 1)
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows + rows[1:]), encoding="utf-8")
    with pytest.raises(bench.InputError, match="repeats item s1-a"):
        bench.read_key(path)


def test_read_key_refuses_one_stores_keys_from_two_packets(tmp_path: Path) -> None:
    # Two packets of one store's runs, with different seeds, give every
    # judged belief a new item id. Concatenated, their keys would double
    # every paired gain and the graded counts MIN_GRADED checks.
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    runs = _one_prompt_runs(["seed"], ["seed", "con"])
    keys = [bench.build_packet(runs, db, seed, snapshot=_SNAP)[1] for seed in (1795, 1796)]
    assert not {r["id"] for r in _items(keys[0])} & {r["id"] for r in _items(keys[1])}
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in keys[0] + keys[1]), encoding="utf-8")
    with pytest.raises(bench.InputError, match=f"a second store row for snapshot {_SNAP}"):
        bench.read_key(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in keys[0]), encoding="utf-8")
    assert len(bench.read_key(path).items) == len(_items(keys[0]))


def test_read_key_rejects_an_unknown_arm(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        _store_row(["A0"]),
        {"kind": "item", "id": "i1", "prompt_id": "p", "arms": {"A0": 1},
         "shares_prompt_stem": True, "snapshot": _SNAP},
        {"kind": "item", "id": "i2", "prompt_id": "p", "arms": {"A9": 1},
         "shares_prompt_stem": True, "snapshot": _SNAP}]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(bench.InputError, match=r"key\.jsonl:3:"):
        bench.read_key(path)


def test_read_prompts_rejects_a_repeated_prompt_id(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    row: dict[str, Any] = {"prompt_id": "p", "text": "q", "context": []}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(bench.read_prompts(path)) == 1
    path.write_text(2 * (json.dumps(row) + "\n"), encoding="utf-8")
    with pytest.raises(bench.InputError, match="duplicate prompt ids"):
        bench.read_prompts(path)


def test_read_turns_keeps_only_user_and_assistant_lines(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    path.write_text("".join(json.dumps({"role": r, "text": "hello"}) + "\n"
                            for r in ("system", "user", "tool", "assistant")), encoding="utf-8")
    assert [t.role for t in bench.read_turns([path])] == ["user", "assistant"]


def test_belief_texts_rejects_an_id_the_store_lacks(tmp_path: Path) -> None:
    db = _store(tmp_path)
    assert bench.belief_texts(db, ["seed"]) == {
        "seed": "the kestrel hovers above the meadow before it dives"}
    with pytest.raises(bench.InputError, match="belief nope is not in the store"):
        bench.belief_texts(db, ["seed", "nope"])


def test_score_cli_reads_a_packet_key_and_two_label_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    runs = _runs(["seed", "con"], ["con", "rel", "seed"], ["w1", "w2"], ["w2"])
    _p, k = _packet(tmp_path, runs)
    key_path = tmp_path / "key.jsonl"
    key_path.write_text(k, encoding="utf-8")
    labels = [tmp_path / "a.jsonl", tmp_path / "c.jsonl"]
    # Relevant: everything A0 retrieved. The null arm is all not relevant.
    for path in labels:
        path.write_text("".join(
            json.dumps({"id": r["id"], "score": 2 if r["arms"] and "A0" in r["arms"] else 1})
            + "\n" for r in _key_items(k)), encoding="utf-8")
    census = tmp_path / "census.json"
    snapshot = json.loads(k.splitlines()[0])["snapshot"]
    census.write_text(json.dumps(_census(2.0, snapshots=(snapshot,))), encoding="utf-8")
    assert bench.main(["score", "--key", str(key_path), "--labels", *map(str, labels),
                       "--census", str(census)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
    # A1 holds both of A0's beliefs on p1 (gain 0) and 1 of 2 on p2 (gain -1).
    assert rep["arms"]["A1"]["graders"][0]["mean_gain"] == _approx(-0.5)


# --- Calibrated grading (#1809) -----------------------------------------------


@pytest.mark.parametrize(("score", "relevant"), [(0, False), (1, False), (2, True), (3, True)])
def test_an_item_is_relevant_at_a_score_of_2_or_more(score: int, relevant: bool) -> None:
    # Both graders give `score` to each arm item they called relevant. The
    # arm's 3, 3, 2, and 4 a prompt against A0's 2 gain 1 a prompt when
    # read as relevant, and lose A0's 2 when not.
    key, a, c = _key(_hand([2, 2, 2, 2], [3, 3, 2, 4]))
    moved = dict.fromkeys((i for i, s in a.items() if i.startswith("x") and s == 2), score)
    rep = bench.score(key, {**a, **moved}, {**c, **moved}, seed=1, census_report=_H1_PASS)
    want = 1.0 if relevant else -2.0
    assert [g["mean_gain"] for g in rep["arms"]["A1"]["graders"]] == [_approx(want)] * 2
    assert rep["bars"]["relevant_cut"] == 2


@pytest.mark.parametrize(("score", "h0"), [(1, bench.V_PASS), (2, bench.V_FAIL)])
def test_the_null_arm_rate_reads_each_score_at_the_cut(score: int, h0: str) -> None:
    # 3 of each prompt's 10 null items get `score` from both graders.
    key, a, c = _key(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3))
    moved = dict.fromkeys((i for i, s in a.items() if i.startswith("n") and s == 2), score)
    rep = bench.score(key, {**a, **moved}, {**c, **moved}, seed=1, census_report=_H1_PASS)
    rate = 0.3 if score >= 2 else 0.0
    assert [n["rate"] for n in rep["null_arm"]] == [_approx(rate)] * 2
    assert rep["hypotheses"]["H0"] == h0


def test_kappa_compares_the_graders_at_the_cut_not_their_raw_scores() -> None:
    # The null table's kappa is exactly the bar, with 16 of 18 items agreed.
    # Grader a moving every 2 to 3 and every 1 to 0 keeps each item on its
    # side of the cut. Moving one 2 to 1 crosses it.
    key, a, c = _key(_null_table(2, 1, 1, 14))
    far = {i: 3 if s == 2 else 0 for i, s in a.items()}
    rep = bench.score(key, far, c, seed=1, census_report=_H1_PASS)
    assert (rep["kappa"], rep["agreed"], rep["hypotheses"]["H0"]) == (
        _approx(0.6), 16, bench.V_PASS)
    rep = bench.score(key, {**a, "n0": 1}, c, seed=1, census_report=_H1_PASS)
    assert rep["agreed"] == 15
    assert rep["hypotheses"]["H0"] == bench.V_FAIL


def test_a_null_score_is_left_out_as_in_the_calibration() -> None:
    # Grader a scores null on n0, which both graders otherwise call
    # relevant: kappa counts the 17 items both scored, and a's null-arm
    # rate the 17 items a scored, 2 of them relevant.
    key, a, c = _key(_null_table(2, 1, 1, 14))
    rep = bench.score(key, {**a, "n0": None}, c, seed=1, census_report=_H1_PASS)
    assert (rep["kappa_n"], rep["agreed"]) == (17, 15)
    assert [(n["graded"], n["called_relevant"]) for n in rep["null_arm"]] == [(17, 2), (18, 3)]
    assert rep["score_counts"]["a"]["null"] == 1


def test_the_report_counts_each_graders_scores_and_nulls() -> None:
    # Grader a's 18 null-arm items start as three 2s and fifteen 1s, and
    # c's as three 2s and fifteen 1s.
    key, a, c = _key(_null_table(2, 1, 1, 14))
    a = {**a, "n0": 3, "n4": 0, "n5": None, "n6": None}
    c = {**c, "n7": None}
    rep = bench.score(key, a, c, seed=1, census_report=_H1_PASS)
    assert rep["graders"] == ["a", "c"]
    assert rep["score_counts"] == {
        "a": {"0": 1, "1": 12, "2": 2, "3": 1, "null": 2},
        "c": {"0": 0, "1": 14, "2": 3, "3": 0, "null": 1},
    }


def test_score_cli_refuses_a_binary_label_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _store_key_rows("s1", 1)
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    scores, binary = tmp_path / "scores.jsonl", tmp_path / "binary.jsonl"
    scores.write_text("".join(json.dumps({"id": r["id"], "score": 2}) + "\n"
                              for r in _items(rows)), encoding="utf-8")
    binary.write_text("".join(json.dumps({"id": r["id"], "correct": True}) + "\n"
                              for r in _items(rows)), encoding="utf-8")
    for pair in ((binary, scores), (scores, binary)):
        assert bench.main(["score", "--key", str(key), "--labels", *map(str, pair)]) == 1
        assert f'{binary}:1: a binary "correct" label' in capsys.readouterr().err


def test_score_cli_reads_grader_a_then_grader_c(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Grader a's file scores the first item null and the rest 3, and grader
    # c's scores every item 0, so each count names the grader of its file.
    rows = _store_key_rows("s1", 1)
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ids = [r["id"] for r in _items(rows)]
    a, c = tmp_path / "a.jsonl", tmp_path / "c.jsonl"
    a.write_text("".join(json.dumps({"id": i, "score": None if n == 0 else 3}) + "\n"
                         for n, i in enumerate(ids)), encoding="utf-8")
    c.write_text("".join(json.dumps({"id": i, "score": 0}) + "\n" for i in ids),
                 encoding="utf-8")
    assert bench.main(["score", "--key", str(key), "--labels", str(a), str(c)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["score_counts"] == {
        "a": {"0": 0, "1": 0, "2": 0, "3": len(ids) - 1, "null": 1},
        "c": {"0": len(ids), "1": 0, "2": 0, "3": 0, "null": 0},
    }


@pytest.mark.parametrize("spelling", [
    "same path", "dotdot path",
    pytest.param("symlink", marks=pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX symlinks")),
])
def test_score_cli_refuses_one_file_for_both_graders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spelling: str,
) -> None:
    rows = _store_key_rows("s1", 1)
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    scores, copy = tmp_path / "scores.jsonl", tmp_path / "copy.jsonl"
    for path in (scores, copy):
        path.write_text("".join(json.dumps({"id": r["id"], "score": 2}) + "\n"
                                for r in _items(rows)), encoding="utf-8")
    if spelling == "same path":
        other = scores
    elif spelling == "dotdot path":
        other = tmp_path / ".." / tmp_path.name / scores.name
    else:
        other = tmp_path / "link.jsonl"
        other.symlink_to(scores)
    assert bench.main(["score", "--key", str(key), "--labels", str(scores), str(other)]) == 1
    assert (f"--labels {scores} and {other} are one file; each grader needs its own"
            in capsys.readouterr().err)
    assert bench.main(["score", "--key", str(key), "--labels", str(scores), str(copy)]) == 0


@pytest.mark.parametrize(("grader", "extra", "drop", "error"), [
    ("a", {"elsewhere": 3}, None, "grader a scored 1 rows the key lacks, elsewhere among them"),
    ("c", {"elsewhere": None}, None,
     "grader c scored 1 rows the key lacks, elsewhere among them"),
    ("c", {}, "n3", "grader c has no score for 1 key rows, n3 among them"),
])
def test_score_refuses_a_score_file_that_is_not_the_keys_rows(
    grader: str, extra: dict[str, int | None], drop: str | None, error: str,
) -> None:
    key, a, c = _key(_null_table(2, 1, 1, 14))
    scores = {"a": a, "c": c}
    scores[grader].update(extra)
    if drop is not None:
        del scores[grader][drop]
    with pytest.raises(bench.InputError, match=f"^{error}$"):
        bench.score(key, scores["a"], scores["c"], seed=1, census_report=_H1_PASS)


def test_the_docstring_and_help_name_graders_a_and_c(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "grader a and grader c" in " ".join((bench.__doc__ or "").split())
    with pytest.raises(SystemExit):
        bench.main(["score", "--help"])
    assert "grader a's and grader c's score files" in " ".join(capsys.readouterr().out.split())
