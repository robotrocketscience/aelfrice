"""Tests for `scripts/audit_rebuild_log.py` (#288 phase-1c).

Covers the summary maths, drop-reason bucketing, JSONL iteration with
malformed-line tolerance, and CLI exit codes.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_AUDIT_PATH = _REPO_ROOT / "scripts" / "audit_rebuild_log.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_rebuild_log", _AUDIT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_module()


# ---- helpers -----------------------------------------------------------


def _record(
    *,
    candidates: list[dict],
    n_candidates: int,
    n_packed: int,
    n_dropped_floor: int = 0,
    n_dropped_dedup: int = 0,
    n_dropped_budget: int = 0,
) -> dict:
    return {
        "ts": "2026-05-02T00:00:00Z",
        "session_id": "s1",
        "input": {
            "recent_turns_hash": "abc",
            "n_recent_turns": 1,
            "extracted_query": "q",
            "extracted_entities": [],
            "extracted_intent": None,
        },
        "candidates": candidates,
        "pack_summary": {
            "n_candidates": n_candidates,
            "n_packed": n_packed,
            "n_dropped_by_floor": n_dropped_floor,
            "n_dropped_by_dedup": n_dropped_dedup,
            "n_dropped_by_budget": n_dropped_budget,
            "total_chars_packed": 100,
        },
    }


def _packed(belief_id: str, rank: int) -> dict:
    return {
        "belief_id": belief_id,
        "rank": rank,
        "scores": {"bm25": None, "posterior_mean": None,
                   "reranker": None, "final": None},
        "lock_level": "none",
        "decision": "packed",
        "reason": None,
    }


def _dropped(belief_id: str, rank: int, reason: str) -> dict:
    return {
        "belief_id": belief_id,
        "rank": rank,
        "scores": {"bm25": None, "posterior_mean": None,
                   "reranker": None, "final": None},
        "lock_level": "none",
        "decision": "dropped",
        "reason": reason,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


# ---- summarise ---------------------------------------------------------


def test_summarise_counts_records_and_pack_rate(tmp_path: Path) -> None:
    rows = [
        _record(
            candidates=[_packed("a", 1), _packed("b", 2),
                        _dropped("c", 3, "below_floor:0.40")],
            n_candidates=3, n_packed=2, n_dropped_floor=1,
        ),
        _record(
            candidates=[_packed("d", 1),
                        _dropped("e", 2, "below_floor:0.50"),
                        _dropped("f", 3, "below_floor:0.60")],
            n_candidates=3, n_packed=1, n_dropped_floor=2,
        ),
    ]
    s = audit._summarise(rows)
    assert s["n_records"] == 2
    assert s["n_truncated"] == 0
    assert s["pack_rate_mean"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert s["packed_ranks"] == Counter({1: 2, 2: 1})
    assert s["drop_reasons"] == Counter({
        "below_floor:0.40": 1,
        "below_floor:0.50": 1,
        "below_floor:0.60": 1,
    })


def test_summarise_counts_truncated_marker_separately() -> None:
    rows = [
        _record(candidates=[_packed("a", 1)], n_candidates=1, n_packed=1),
        {"truncated": True, "ts": "2026-05-02T00:00:00Z",
         "reason": "size_cap", "cap_bytes": 5_242_880},
    ]
    s = audit._summarise(rows)
    assert s["n_records"] == 1
    assert s["n_truncated"] == 1


def test_summarise_handles_missing_pack_summary() -> None:
    rows = [
        {"ts": "x", "session_id": None, "input": {},
         "candidates": [_packed("a", 1)]},  # no pack_summary
    ]
    s = audit._summarise(rows)
    assert s["n_records"] == 1
    assert s["n_no_pack_summary"] == 1
    # packed rank still picked up from the candidate list
    assert s["packed_ranks"] == Counter({1: 1})


# ---- drop-reason bucketing --------------------------------------------


def test_bucketise_drop_reasons_groups_on_prefix() -> None:
    raw = Counter({
        "below_floor:0.40": 3,
        "below_floor:0.41": 2,
        "content_hash_collision_with:abc": 4,
        "content_hash_collision_with:def": 1,
        "budget": 7,
        "below_floor:0.42": 1,
    })
    bucketed = audit._bucketise_drop_reasons(raw)
    assert bucketed == Counter({
        "below_floor": 6,
        "content_hash_collision_with": 5,
        "budget": 7,
    })


# ---- iteration / file handling ----------------------------------------


def test_iter_records_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps({"ts": "x", "candidates": []}) + "\n"
        "this is not json\n"
        "\n"
        + json.dumps({"ts": "y", "candidates": []}) + "\n",
        encoding="utf-8",
    )
    rows = list(audit._iter_records(p))
    assert len(rows) == 2
    assert rows[0]["ts"] == "x"
    assert rows[1]["ts"] == "y"


def test_collect_paths_expands_directory(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")
    out = audit._collect_paths([tmp_path])
    assert sorted(p.name for p in out) == ["a.jsonl", "b.jsonl"]


# ---- CLI --------------------------------------------------------------


def test_cli_returns_1_on_no_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = audit.main([str(tmp_path / "missing")])
    assert rc == 1


def test_cli_returns_0_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    p = tmp_path / "s1.jsonl"
    _write_jsonl(p, [
        _record(candidates=[_packed("a", 1),
                            _dropped("b", 2, "below_floor:0.40")],
                n_candidates=2, n_packed=1, n_dropped_floor=1),
    ])
    rc = audit.main([str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "records:       1" in out
    assert "below_floor" in out
    assert "rank  1: 1" in out


def test_cli_percentile_nearest_rank() -> None:
    assert audit._percentile([], 50) == 0.0
    assert audit._percentile([0.5], 50) == 0.5
    # 5 values, p50 -> middle, p90 -> last
    assert audit._percentile([0.1, 0.2, 0.3, 0.4, 0.5], 50) == 0.3
    assert audit._percentile([0.1, 0.2, 0.3, 0.4, 0.5], 90) == 0.5


# ---- calibration mode (#365 R1) ---------------------------------------


def _write_calibration_corpus(path: Path, fixtures: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fx in fixtures:
            f.write(json.dumps(fx, separators=(",", ":")) + "\n")


def _toy_corpus() -> list[dict]:
    """Three-query toy corpus where the BM25 lane will surface the
    relevant belief (the only one whose tokens overlap the query) for
    each fixture. Used to drive the calibration smoke tests below.
    """
    return [
        {
            "id": "q1",
            "query": "python asyncio event loop scheduler coroutine",
            "known_belief_content": (
                "asyncio event loop schedules coroutines and "
                "callbacks via a single-threaded scheduler"
            ),
            "noise_belief_contents": [
                "the GIL prevents true thread parallelism in CPython",
                "subprocess module launches child processes",
                "multiprocessing spawns separate processes",
                "threading provides OS-level threads",
            ],
        },
        {
            "id": "q2",
            "query": "sqlite WAL journal mode concurrent reads",
            "known_belief_content": (
                "SQLite WAL journal mode allows concurrent readers "
                "and one writer without blocking reads"
            ),
            "noise_belief_contents": [
                "PostgreSQL MVCC uses row-level versioning",
                "Redis persistence uses RDB snapshots and AOF logs",
                "MySQL InnoDB uses a clustered primary index",
                "database indexes trade write overhead for reads",
            ],
        },
        {
            "id": "q3",
            "query": "rust borrow checker lifetimes references",
            "known_belief_content": (
                "Rust borrow checker enforces lifetime rules on "
                "references at compile time"
            ),
            "noise_belief_contents": [
                "Go has a garbage collector for memory management",
                "C++ uses RAII for deterministic destruction",
                "Java has a generational garbage collector",
                "Swift uses ARC for reference counting",
            ],
        },
    ]


def test_calibrate_corpus_emits_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "toy.jsonl"
    _write_calibration_corpus(corpus, _toy_corpus())

    rc = audit.main(["--calibrate-corpus", str(corpus)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "calibration harness" in out
    assert "n_queries:    3" in out
    assert "P@10:" in out
    assert "ROC-AUC:" in out
    assert "Spearman" in out


def test_calibrate_corpus_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Ship gate per #365: same (corpus, seed) -> bytes-identical
    metric lines across reruns."""
    corpus = tmp_path / "toy.jsonl"
    _write_calibration_corpus(corpus, _toy_corpus())

    rc1 = audit.main(["--calibrate-corpus", str(corpus), "--seed", "42"])
    out1 = capsys.readouterr().out
    rc2 = audit.main(["--calibrate-corpus", str(corpus), "--seed", "42"])
    out2 = capsys.readouterr().out

    assert rc1 == 0 and rc2 == 0
    metric_lines1 = [
        line for line in out1.splitlines()
        if line.startswith(("P@", "ROC-AUC:", "Spearman"))
    ]
    metric_lines2 = [
        line for line in out2.splitlines()
        if line.startswith(("P@", "ROC-AUC:", "Spearman"))
    ]
    assert metric_lines1 == metric_lines2
    assert metric_lines1, "expected metric lines in output"


def test_calibrate_corpus_missing_file_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = audit.main([
        "--calibrate-corpus", str(tmp_path / "no-such-corpus.jsonl"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "calibration corpus not found" in err


def test_calibrate_corpus_empty_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")
    rc = audit.main(["--calibrate-corpus", str(corpus)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "corpus is empty" in err


def test_calibrate_corpus_skips_malformed_rows(tmp_path: Path) -> None:
    corpus = tmp_path / "mixed.jsonl"
    corpus.write_text(
        "this is not json\n"
        + json.dumps({"id": "q1", "query": "q", "known_belief_content": "k"})
        + "\n"
        + json.dumps({
            "id": "q2", "query": "q", "known_belief_content": "k",
            "noise_belief_contents": "not-a-list",
        })
        + "\n",
        encoding="utf-8",
    )
    fixtures = audit._load_calibration_fixtures(corpus)
    assert fixtures == []


def test_calibrate_rejects_non_positive_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "toy.jsonl"
    _write_calibration_corpus(corpus, _toy_corpus())
    rc = audit.main([
        "--calibrate-corpus", str(corpus), "--k", "0",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--k must be positive" in err


def test_audit_mode_requires_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No --calibrate-corpus and no positional paths -> usage error."""
    with pytest.raises(SystemExit) as excinfo:
        audit.main([])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "paths required" in err
