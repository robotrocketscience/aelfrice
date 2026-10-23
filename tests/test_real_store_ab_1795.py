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
    rows: list[Row] = []
    for p in range(4):
        rows += [(f"a{p}-{k}", f"p{p}", {"A0": k + 1, "A2": k + 1}, True, k < 2, k < 2)
                 for k in range(10)]
        rows.append((f"r{p}", f"p{p}", {"A2": 11}, False, True, True))
        rows += [(f"n{p}-{k}", f"p{p}", None, False, False, False) for k in range(10)]
    g = _score(rows)["arms"]["A2"]["graders"][0]
    assert g["top10_mean_gain"] == _approx(0.0)
    assert g["mean_gain"] == _approx(1.0)
    assert g["gain_verdict"] == bench.V_PASS
    assert g["precision_drop"] == _approx(0.0)


def test_a0_takes_the_top_ten_non_locked_beliefs(tmp_path: Path) -> None:
    path = _store(tmp_path)
    ranked = bench.run_arm(
        path, [{"prompt_id": "p", "text": "always run the test suite before pushing",
                "context": []}], bench.ARM_A0, tmp_path)["prompts"][0]["ranked"]
    assert "lock1" not in ranked
    assert len(ranked) <= bench.TOP_K


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
    bench.write_packet(packet, key, out)
    return (out / "packet.jsonl").read_text(), (out / "key.jsonl").read_text()


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
    unclear: int = 0,
) -> list[Row]:
    """Per prompt: 10 A0 items and 10 arm items, disjoint.

    `a0_rel[p]` and `arm_rel[p]` of them are relevant to both graders.
    `flip_b` of A0's not-relevant items per prompt are relevant to grader B
    only. `bridging` of each prompt's relevant arm items share no stem.
    `unclear` of A0's not-relevant items per prompt are unclear to both.
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
            rows.append((f"x{p}-{k}", f"p{p}", {arm: k + 1}, not (rel and k < bridging),
                         rel, rel))
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


def test_too_few_graded_items_is_insufficient() -> None:
    rep = _score(_hand([2, 2], [3, 3]))
    assert rep["arms"]["A1"]["graders"][0]["graded"] == 20 < graded_gate.MIN_GRADED
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


def _split_rows(a0_rel: list[int], a1w_rel: list[int], a1_rel: list[int]) -> list[Row]:
    """`_hand` for A0 and A1W, plus 10 disjoint A1 items per prompt."""
    rows = _hand(a0_rel, a1w_rel, arm=bench.ARM_A1W)
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
    # The gain counts beliefs past the top 10, so the proxy does too.
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
                       "--out", str(d / "pk")]) == 0
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
