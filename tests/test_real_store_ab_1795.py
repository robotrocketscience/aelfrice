"""The #1795 real-store A/B harness: sample, run, packet, score, and proxy.

Every store and transcript is built from synthetic text in a temp directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

import pytest

import aelfrice.hook as hook_mod
from aelfrice import hrr_expand
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
from benchmarks import graded_gate
from benchmarks import real_store_ab_1795 as bench

_QUERY = "kestrel meadow"


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
) -> None:
    s.insert_belief(Belief(
        id=bid, content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=lock,
        locked_at=None if lock == LOCK_NONE else created,
        created_at=created, last_retrieved_at=None,
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
    _packet_rows, key = bench.build_packet(runs, bench.prepare_copy(path, work), 1)
    by = {r["belief_id"]: r for r in key if r["arms"]}
    assert by["rel"]["arms"] == {"A2": a2.index("rel") + 1}
    # Both arms packed the beliefs past the cut, so they cancel unjudged.
    assert set(by) == set(a0[:bench.TOP_K]) | {"rel"}
    # The null arm never draws a belief an arm packed, judged or not.
    for seed in range(10):
        _rows, k = bench.build_packet(runs, work / "memory.db", seed)
        assert not {r["belief_id"] for r in k if r["arms"] is None} & set(a2)
    # Graded relevant: A2 gains `rel` on every prompt, all of it past the cut.
    # #1795 registers the gain over the top 10, so the packed gain is only
    # reported and H3 fails.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A2": k + 1}, True, k < 2, k < 2)
                 for k in range(10)]
        rows.append((f"r{p}", f"p{p}", {"A2": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    rep = _score(rows)
    g = rep["arms"]["A2"]["graders"][0]
    assert g["mean_gain"] == _approx(0.0)
    assert g["packed_mean_gain"] == _approx(1.0)
    assert g["packed_ci95"] == [_approx(1.0), _approx(1.0)]
    assert g["packed_share_gaining"] == _approx(1.0)
    assert g["gain_verdict"] == bench.V_FAIL
    assert rep["hypotheses"]["H3"] == bench.V_FAIL
    assert g["precision_drop"] == _approx(0.0)


def test_h4_counts_additions_to_the_top_ten() -> None:
    # Per prompt, A3 moves A0's rank-11 belief into its top 10 and packs a
    # bridging belief at rank 11. Only the first is a top-10 addition.
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
    assert (g["relevant_additions"], g["bridging"]) == (4, 0)


def test_the_proxy_sign_counts_the_top_ten_only() -> None:
    # A1's only used belief sits at rank 11, where the graders' gain doesn't look.
    unused = [{"id": f"u{k}", "used": False} for k in range(bench.TOP_K)]
    proxy = {"prompts": [{"arms": {
        "A0": unused, "A1": [*unused, {"id": "x", "used": True}]}}]}
    assert bench.proxy_gain_signs(proxy, ["A1"]) == {"A1": 0}


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
    _rows, key = bench.build_packet({bench.ARM_A0: run}, work / "memory.db", 1)
    assert {r["belief_id"] for r in key if r["arms"]} == set(ranked[:10])


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
    # Ten REFINES pairs share f00, and an 11-member cluster has f00 as its
    # oldest member. The cap lets 8 edges reach f00 and gates the other 2.
    others = [f"f{i:02d}" for i in range(1, 11)]

    def refines(_store: MemoryStore, **_kw: Any) -> Any:
        return SimpleNamespace(n_candidate_pairs=10, truncated=False, pairs=[
            SimpleNamespace(label=bench.LABEL_REFINES, belief_a_id="f00", belief_b_id=b)
            for b in others])

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
    packet, key = bench.build_packet(runs, bench.prepare_copy(path, work), seed)
    out = work / "out"
    bench.write_packet(packet, key, out, work / "key.jsonl")
    return (out / "packet.jsonl").read_text(), (work / "key.jsonl").read_text()


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
    key = {r["id"]: r for r in map(json.loads, k1.splitlines())}
    for r in rows:
        k = key[r["id"]]
        assert r["id"] == bench.item_id(9, k["prompt_id"], k["belief_id"])


def test_the_key_records_arms_ranks_and_a_median_sized_null_arm(tmp_path: Path) -> None:
    runs = _runs(["seed", "con"], ["con", "rel", "seed"], ["w1", "w2"], ["w2"])
    _p, k = _packet(tmp_path, runs)
    key = [json.loads(line) for line in k.splitlines()]
    by = {(r["prompt_id"], r["belief_id"]): r for r in key}
    assert by[("p1", "con")]["arms"] == {"A0": 2, "A1": 1}
    assert by[("p1", "rel")]["arms"] == {"A1": 2}
    nulls = [r for r in key if r["arms"] is None]
    # Union sizes 3 and 2; the lower median is 2 per prompt.
    assert sorted(r["prompt_id"] for r in nulls) == ["p1", "p1", "p2", "p2"]
    retrieved = {(r["prompt_id"], r["belief_id"]) for r in key if r["arms"]}
    assert all((r["prompt_id"], r["belief_id"]) not in retrieved for r in nulls)
    assert all(r["belief_id"] != "lock1" for r in nulls)
    assert by[("p1", "seed")]["shares_prompt_stem"] is True
    assert by[("p1", "rel")]["shares_prompt_stem"] is False


def _one_prompt_runs(a0: list[str], a1: list[str]) -> dict[str, Any]:
    return {arm: {"arm": arm, "prompts": [
        {"prompt_id": "p1", "prompt": "kestrel meadow", "ranked": ranked}]}
        for arm, ranked in ((bench.ARM_A0, a0), (bench.ARM_A1, a1))}


def test_a_belief_in_any_arms_top_ten_is_judged(tmp_path: Path) -> None:
    # Both arms pack `seed`: A0 at rank 1, A1 at rank 12. A0's top 10 holds
    # it, so it is judged although no compared pair differs on it.
    fill = [f"f{i:02d}" for i in range(11)]
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    _rows, key = bench.build_packet(_one_prompt_runs(["seed", *fill], [*fill, "seed"]), db, 1)
    by = {r["belief_id"]: r["arms"] for r in key if r["arms"]}
    assert by["seed"] == {"A0": 1, "A1": 12}


def test_the_null_arm_is_sized_by_the_judged_count_not_the_packed_count(
    tmp_path: Path,
) -> None:
    # Both arms pack 12 beliefs. `f10` sits past rank 10 in both and is
    # unjudged, so 11 are judged and the null arm draws 11.
    fill = [f"f{i:02d}" for i in range(11)]
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(_store(tmp_path), work)
    _rows, key = bench.build_packet(_one_prompt_runs(["seed", *fill], [*fill, "seed"]), db, 1)
    assert sum(1 for r in key if r["arms"]) == 11
    assert sum(1 for r in key if r["arms"] is None) == 11


def test_the_null_arm_never_draws_a_retrieved_or_locked_belief(tmp_path: Path) -> None:
    path = _store(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    db = bench.prepare_copy(path, work)
    many = ["seed", "con", "rel", "w1", "w2", "w3", "w4", "w5", "w6"]
    runs = _runs(many[:5], many[5:], many[4:], many[:4])
    for seed in range(40):
        _packet_rows, key = bench.build_packet(runs, db, seed)
        retrieved = {(r["prompt_id"], r["belief_id"]) for r in key if r["arms"]}
        nulls = [(r["prompt_id"], r["belief_id"]) for r in key if r["arms"] is None]
        assert len(nulls) == 18
        assert not retrieved & set(nulls)
        assert all(b != "lock1" for _p, b in nulls)


def test_the_packet_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    runs = _runs(["seed"], ["con"], ["w1"], ["w2"])
    assert _packet(tmp_path, runs, 3) == _packet(tmp_path, runs, 3)
    assert _packet(tmp_path, runs, 3)[0] != _packet(tmp_path, runs, 4)[0]


def test_the_instructions_define_the_three_labels(tmp_path: Path) -> None:
    for word in ("relevant", "not relevant", "unclear", '"correct"'):
        assert word in bench.INSTRUCTIONS


# --- Score --------------------------------------------------------------------


Row = tuple[str, str, dict[str, int] | None, bool, bool | None, bool | None]


def _key(rows: list[Row]) -> tuple[list[bench.KeyItem], dict[str, bool | None],
                                   dict[str, bool | None]]:
    items = [bench.KeyItem(id=i, prompt_id=p, arms=arms, shares_prompt_stem=s)
             for i, p, arms, s, _a, _b in rows]
    return items, {r[0]: r[4] for r in rows}, {r[0]: r[5] for r in rows}


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


_H1_PASS: dict[str, Any] = {"H1": "pass"}


def _score(rows: list[Row], **kw: Any) -> dict[str, Any]:
    items, a, b = _key(rows)
    kw.setdefault("census_report", _H1_PASS)
    return bench.score(items, a, b, seed=1, **kw)


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


def test_a_high_null_arm_rate_makes_every_verdict_unreliable() -> None:
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3))
    assert rep["null_arm"][0]["rate"] == _approx(0.3)
    assert rep["hypotheses"]["H0"] == bench.V_FAIL
    assert rep["hypotheses"]["H2"] == bench.V_UNRELIABLE


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
    assert g["packed_mean_gain"] == _approx(1.0)
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


def test_h2_is_a_lane_verdict_only_when_the_census_passed_h1() -> None:
    # #1795 H1: under 1 per 100, A1 is substrate-starved, not a lane verdict.
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    starved = _score(rows, census_report={"H1": "fail"})
    assert starved["hypotheses"]["H1"] == bench.V_FAIL
    assert starved["hypotheses"]["H2"] == bench.V_STARVED
    assert starved["arms"]["A1"]["gain_verdict"] == bench.V_PASS
    no_census = _score(rows, census_report=None)
    assert no_census["hypotheses"]["H1"] == bench.V_NOT_MEASURED
    assert no_census["hypotheses"]["H2"] == bench.V_NEEDS_CENSUS
    assert _score(rows)["hypotheses"]["H2"] == bench.V_PASS
    # A failed H0 outranks the substrate: the instrument comes first.
    unreliable = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3),
                        census_report={"H1": "fail"})
    assert unreliable["hypotheses"]["H2"] == bench.V_UNRELIABLE
    # H3 and H4 don't depend on the writers' substrate.
    a2 = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], arm=bench.ARM_A2),
                census_report={"H1": "fail"})
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
                     census_report={"H1": "fail"})
    assert starved["hypotheses"]["H2_split"] == {
        "writers": bench.V_STARVED, "lane": bench.V_STARVED}
    assert _score(_hand([2, 2, 2, 2], [3, 3, 2, 4]))["hypotheses"]["H2_split"] == (
        bench.V_NOT_MEASURED)


def test_a_failed_h0_makes_both_halves_of_the_h2_split_unreliable() -> None:
    # The lane half passes on these labels when H0 holds.
    rows = _split_rows([2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 2, 4], null_rel=3)
    for census in (_H1_PASS, {"H1": "fail"}):
        rep = _score(rows, census_report=census)
        assert rep["hypotheses"]["H0"] == bench.V_FAIL
        assert rep["hypotheses"]["H2_split"] == {
            "writers": bench.V_UNRELIABLE, "lane": bench.V_UNRELIABLE}
        assert rep["arms"]["A1"]["vs_A1W"]["gain_verdict"] == bench.V_UNRELIABLE


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


def test_the_interval_is_the_2_5_and_97_5_percentile_of_the_resamples() -> None:
    gains = [0.0] * 39 + [40.0]
    rng = random.Random(7)
    means = sorted(sum(gains[rng.randrange(40)] for _ in range(40)) / 40
                   for _ in range(10_000))
    # Index floor(0.025 * 9999) and ceil(0.975 * 9999).
    assert bench.paired_bootstrap_ci(gains, 7) == (means[249], means[9750])
    assert means[9750] < means[-1]


def test_the_bars_are_the_pre_registered_ones() -> None:
    assert (bench.KAPPA_BAR, bench.NULL_RATE_CEILING, bench.SUBSTRATE_BAR, bench.GAIN_BAR,
            bench.GAIN_KILL, bench.GAINING_SHARE_BAR, bench.BRIDGING_SHARE_BAR,
            bench.PRECISION_DROP_BAR) == (0.60, 0.20, 1.0, 0.3, 0.1, 0.10, 0.5, 0.05)
    assert bench.BOOTSTRAP_RESAMPLES == 10_000


def test_h6_compares_the_proxy_sign_with_each_grader() -> None:
    rows = _hand([2, 2, 2, 2], [3, 3, 2, 4])
    proxy = {"prompts": [{"arms": {
        "A0": [{"id": "a", "used": True}], "A1": [{"id": "b", "used": u}]}}
        for u in (False, False, True, False)]}
    rep = _score(rows, proxy_report=proxy)
    assert rep["hypotheses"]["H6"]["A1"]["verdict"] == bench.V_DISAGREE
    proxy["prompts"][0]["arms"]["A1"] = [{"id": "b", "used": True}, {"id": "c", "used": True}]
    proxy["prompts"][1]["arms"]["A1"] = [{"id": "b", "used": True}, {"id": "c", "used": True}]
    rep = _score(rows, proxy_report=proxy)
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [1, 1], "verdict": bench.V_AGREE}
    rep = _score(_hand([2, 2, 2, 2], [3, 3, 2, 4], null_rel=3), proxy_report=proxy)
    assert rep["hypotheses"]["H6"]["A1"]["verdict"] == bench.V_UNRELIABLE


def test_h6_needs_the_proxy_to_agree_with_both_graders() -> None:
    # Grader B calls only A1's first 2 items per prompt relevant: no gain.
    rows = [(r[0], r[1], r[2], r[3], r[4], r[5] if not r[0].startswith("x")
             else int(r[0].split("-")[1]) < 2) for r in _hand([2, 2, 2, 2], [3, 3, 2, 4])]
    proxy = {"prompts": [{"arms": {"A0": [{"id": "a", "used": False}],
                                   "A1": [{"id": "a", "used": True}]}}]}
    rep = _score(rows, proxy_report=proxy)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [1, 0], "verdict": bench.V_DISAGREE}


def test_h6_takes_each_graders_sign_from_the_top_ten_gain() -> None:
    # A1 adds a relevant belief past rank 10 on every prompt: a packed gain
    # of 1 but a top-10 gain of 0, the gain H2 is decided on.
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A1": k + 1}, True, k < 2, k < 2)
                 for k in range(10)]
        rows.append((f"r{p}", f"p{p}", {"A1": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    proxy = {"prompts": [{"arms": {"A0": [{"id": "a", "used": False}],
                                   "A1": [{"id": "a", "used": True}]}}]}
    rep = _score(rows, proxy_report=proxy)
    assert rep["arms"]["A1"]["graders"][0]["packed_mean_gain"] == _approx(1.0)
    assert rep["hypotheses"]["H6"]["A1"] == {
        "proxy_sign": 1, "grader_signs": [0, 0], "verdict": bench.V_DISAGREE}


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
    rep = bench.proxy(runs, [prompt, early], turns, bench.prepare_copy(path, work))
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
    rep = bench.proxy(runs, [prompt], turns, bench.prepare_copy(path, work))
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
    rep = bench.proxy(runs, [prompt], turns, bench.prepare_copy(_store(tmp_path), work))
    row = rep["prompts"][0]
    assert row["later_turns"] == 1
    assert row["arms"]["A0"] == [{"id": "seed", "used": False}]


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
    # A `mode=ro` reader of a WAL store may leave the shared-memory file and
    # an empty log beside it; neither holds data.
    added = sorted({p.name for p in path.parent.iterdir()} - set(files))
    assert set(added) <= {"memory.db-wal", "memory.db-shm"}
    wal = path.parent / "memory.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    ro = read_only_uri(str(path))
    on_source = [c for c in connect_spy if _target(c[0]) == path.resolve()]
    assert on_source and set(on_source) == {(ro, True)}


def test_every_subcommand_has_help() -> None:
    for cmd in ("census", "sample", "run", "packet", "score", "proxy"):
        with pytest.raises(SystemExit) as exc:
            bench.main([cmd, "--help"])
        assert exc.value.code == 0


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


def _run_file(tmp_path: Path, name: str, arm: str, pids: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"arm": arm, "prompts": [
        {"prompt_id": p, "prompt": "q", "ranked": ["seed"]} for p in pids]}), encoding="utf-8")
    return path


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


def test_read_key_rejects_a_key_without_an_a0_item(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"id": "i1", "prompt_id": "p", "arms": {"A1": 1}, "shares_prompt_stem": True},
        {"id": "i2", "prompt_id": "p", "arms": None, "shares_prompt_stem": False}]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(bench.InputError, match="no A0 item"):
        bench.read_key(path)
    rows[0]["arms"] = {"A0": 3, "A1": 1}
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [i.id for i in bench.read_key(path)] == ["i1", "i2"]


def test_read_key_rejects_an_unknown_arm(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"id": "i1", "prompt_id": "p", "arms": {"A0": 1}, "shares_prompt_stem": True},
        {"id": "i2", "prompt_id": "p", "arms": {"A9": 1}, "shares_prompt_stem": True}]
    path = tmp_path / "key.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(bench.InputError, match=r"key\.jsonl:2:"):
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
    labels = tmp_path / "labels.jsonl"
    # Relevant: everything A0 retrieved. The null arm is all not relevant.
    labels.write_text("".join(
        json.dumps({"id": r["id"], "correct": bool(r["arms"]) and "A0" in r["arms"]}) + "\n"
        for r in map(json.loads, k.splitlines())), encoding="utf-8")
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"H1": "pass"}), encoding="utf-8")
    assert bench.main(["score", "--key", str(key_path), "--labels", str(labels), str(labels),
                       "--census", str(census)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["hypotheses"]["H0"] == bench.V_PASS
    assert rep["hypotheses"]["H2"] == bench.V_INSUFFICIENT
    # A1 holds both of A0's beliefs on p1 (gain 0) and 1 of 2 on p2 (gain -1).
    assert rep["arms"]["A1"]["graders"][0]["mean_gain"] == _approx(-0.5)
