"""Tests for ``aelfrice.eval_harness`` (#365 R4 lift)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import eval_harness as eh
from aelfrice.calibration_metrics import CalibrationReport


def _toy_corpus() -> list[dict]:
    return [
        {
            "id": "q1",
            "query": "python asyncio event loop",
            "known_belief_content": (
                "asyncio event loop runs coroutines on a single-threaded "
                "scheduler"
            ),
            "noise_belief_contents": [
                "threading module spawns OS-level threads",
                "multiprocessing forks separate processes",
                "the GIL serializes CPython bytecode execution",
                "subprocess launches child processes",
            ],
        },
        {
            "id": "q2",
            "query": "SQLite WAL journal mode",
            "known_belief_content": (
                "SQLite WAL allows concurrent readers and one writer "
                "without blocking reads"
            ),
            "noise_belief_contents": [
                "PostgreSQL MVCC uses row-level versioning",
                "Redis persistence uses RDB snapshots and AOF logs",
                "MySQL InnoDB uses a clustered primary index",
                "database indexes trade write overhead for reads",
            ],
        },
    ]


def _write_corpus(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )


def test_default_corpus_path_is_under_repo_root() -> None:
    expected = "benchmarks/posterior_ranking/fixtures/default.jsonl"
    assert str(eh.DEFAULT_CALIBRATION_CORPUS).endswith(expected)


def test_load_calibration_fixtures_skips_malformed_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "this is not json\n"
        + json.dumps({"id": "q1"})  # missing required fields
        + "\n"
        + json.dumps({
            "id": "q2", "query": "q", "known_belief_content": "k",
            "noise_belief_contents": "not-a-list",
        })
        + "\n"
        + json.dumps({
            "id": "q3", "query": "q", "known_belief_content": "k",
            "noise_belief_contents": ["n1"],
        })
        + "\n",
        encoding="utf-8",
    )
    fixtures = eh.load_calibration_fixtures(path)
    assert [f["id"] for f in fixtures] == ["q3"]


def test_load_calibration_fixtures_empty_file_returns_empty_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert eh.load_calibration_fixtures(path) == []


def test_run_calibration_returns_report(tmp_path: Path) -> None:
    fixtures = _toy_corpus()
    report = eh.run_calibration_on_fixtures(fixtures, k=10, seed=0)
    assert isinstance(report, CalibrationReport)
    assert report.n_queries == len(fixtures)
    assert report.k == 10
    assert 0.0 <= report.p_at_k <= 1.0
    assert report.n_observations >= report.n_queries
    if report.roc_auc is not None:
        assert 0.0 <= report.roc_auc <= 1.0
    if report.spearman_rho is not None:
        assert -1.0 <= report.spearman_rho <= 1.0


def test_run_calibration_is_deterministic_at_fixed_seed() -> None:
    fixtures = _toy_corpus()
    r1 = eh.run_calibration_on_fixtures(fixtures, k=10, seed=42)
    r2 = eh.run_calibration_on_fixtures(fixtures, k=10, seed=42)
    assert r1 == r2


def test_run_calibration_rejects_non_positive_k() -> None:
    fixtures = _toy_corpus()
    with pytest.raises(ValueError, match="k must be positive"):
        eh.run_calibration_on_fixtures(fixtures, k=0, seed=0)
    with pytest.raises(ValueError, match="k must be positive"):
        eh.run_calibration_on_fixtures(fixtures, k=-1, seed=0)


def test_run_calibration_rejects_empty_fixtures() -> None:
    with pytest.raises(ValueError, match="fixtures must be non-empty"):
        eh.run_calibration_on_fixtures([], k=10, seed=0)


def test_format_calibration_report_is_deterministic(tmp_path: Path) -> None:
    fixtures = _toy_corpus()
    report = eh.run_calibration_on_fixtures(fixtures, k=10, seed=0)
    text1 = eh.format_calibration_report(
        report, corpus_path=tmp_path / "toy.jsonl", seed=0,
    )
    text2 = eh.format_calibration_report(
        report, corpus_path=tmp_path / "toy.jsonl", seed=0,
    )
    assert text1 == text2
    assert "calibration harness" in text1
    assert "P@10:" in text1
    assert "ROC-AUC:" in text1
    assert "Spearman" in text1
    assert text1.endswith("\n")


def test_format_calibration_report_renders_undefined_metrics(
    tmp_path: Path,
) -> None:
    """ROC-AUC / ρ render as 'n/a (undefined)' when the metric is None."""
    report = CalibrationReport(
        p_at_k=0.42,
        k=10,
        n_queries=3,
        n_truncated_queries=1,
        roc_auc=None,
        spearman_rho=None,
        n_observations=15,
    )
    text = eh.format_calibration_report(
        report, corpus_path=tmp_path / "x.jsonl", seed=7,
    )
    assert "P@10:        0.4200" in text
    assert "ROC-AUC:      n/a (undefined)" in text
    assert "Spearman ρ:   n/a (undefined)" in text
    assert "truncated:    1" in text


def test_format_calibration_report_omits_truncated_when_zero(
    tmp_path: Path,
) -> None:
    report = CalibrationReport(
        p_at_k=0.5, k=10, n_queries=2, n_truncated_queries=0,
        roc_auc=0.75, spearman_rho=0.5, n_observations=10,
    )
    text = eh.format_calibration_report(
        report, corpus_path=tmp_path / "x.jsonl", seed=0,
    )
    assert "truncated:" not in text


def _store_id_to_content(store) -> dict[str, str]:
    out: dict[str, str] = {}
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is not None:
            out[bid] = b.content
    return out


def test_build_calibration_store_inserts_one_known_plus_n_noise() -> None:
    fixture = {
        "id": "q1",
        "query": "x",
        "known_belief_content": "the known one",
        "noise_belief_contents": ["n1", "n2", "n3"],
    }
    store = eh.build_calibration_store(fixture, seed=0)
    try:
        contents = sorted(_store_id_to_content(store).values())
        assert contents == sorted(["the known one", "n1", "n2", "n3"])
    finally:
        store.close()


def test_build_calibration_store_seed_controls_noise_order() -> None:
    """Same seed reproduces noise content<->ID mapping; different seed
    produces a different mapping."""
    fixture = {
        "id": "q1",
        "query": "x",
        "known_belief_content": "the known one",
        "noise_belief_contents": [f"n{i}" for i in range(20)],
    }
    s1 = eh.build_calibration_store(fixture, seed=0)
    s2 = eh.build_calibration_store(fixture, seed=0)
    s3 = eh.build_calibration_store(fixture, seed=99)
    try:
        m1 = _store_id_to_content(s1)
        m2 = _store_id_to_content(s2)
        m3 = _store_id_to_content(s3)
        assert m1 == m2  # determinism at fixed seed
        assert m1 != m3  # 20-row shuffle differs across seeds
        assert sorted(m1.keys()) == sorted(m3.keys())  # same belief IDs
    finally:
        s1.close()
        s2.close()
        s3.close()
