"""Tests for ``aelf eval`` (#365 R4 Phase B)."""
from __future__ import annotations

import io
import json
from pathlib import Path

from aelfrice import cli


def _toy_corpus(path: Path) -> Path:
    rows = [
        {
            "id": "q1",
            "query": "python asyncio event loop",
            "known_belief_content": (
                "asyncio event loop runs coroutines on a single-threaded "
                "scheduler"
            ),
            "noise_belief_contents": [
                "threading module spawns OS-level threads",
                "the GIL serializes CPython bytecode execution",
                "subprocess launches child processes",
            ],
        },
        {
            "id": "q2",
            "query": "SQLite WAL journal mode",
            "known_belief_content": (
                "SQLite WAL allows concurrent readers and one writer"
            ),
            "noise_belief_contents": [
                "PostgreSQL MVCC uses row-level versioning",
                "Redis persistence uses RDB snapshots and AOF logs",
            ],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    return path


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    rc = cli.main(argv, out=out)
    return rc, out.getvalue()


def test_eval_default_corpus_emits_text_block(tmp_path: Path) -> None:
    rc, output = _run(["eval"])
    assert rc == 0
    assert "calibration harness" in output
    assert "P@10:" in output
    assert "ROC-AUC:" in output
    assert "Spearman" in output


def test_eval_custom_corpus(tmp_path: Path) -> None:
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc, output = _run(["eval", "--corpus", str(corpus)])
    assert rc == 0
    assert "calibration harness — corpus toy.jsonl" in output
    assert "n_queries:    2" in output


def test_eval_json_mode_emits_one_line_object(tmp_path: Path) -> None:
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc, output = _run(["eval", "--corpus", str(corpus), "--json"])
    assert rc == 0
    payload = json.loads(output)
    assert payload["k"] == 10
    assert payload["n_queries"] == 2
    assert payload["corpus"] == str(corpus)
    assert payload["seed"] == 0
    assert "p_at_k" in payload
    assert "roc_auc" in payload
    assert "spearman_rho" in payload


def test_eval_is_deterministic_at_fixed_seed(tmp_path: Path) -> None:
    """Ship gate per #365: same (corpus, seed) -> bytes-identical output."""
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc1, out1 = _run(["eval", "--corpus", str(corpus), "--seed", "42"])
    rc2, out2 = _run(["eval", "--corpus", str(corpus), "--seed", "42"])
    assert rc1 == 0 and rc2 == 0
    assert out1 == out2


def test_eval_json_is_deterministic_at_fixed_seed(tmp_path: Path) -> None:
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc1, out1 = _run([
        "eval", "--corpus", str(corpus), "--seed", "7", "--json",
    ])
    rc2, out2 = _run([
        "eval", "--corpus", str(corpus), "--seed", "7", "--json",
    ])
    assert rc1 == 0 and rc2 == 0
    assert out1 == out2


def test_eval_k_flag_changes_p_at_k_label(tmp_path: Path) -> None:
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc, output = _run([
        "eval", "--corpus", str(corpus), "--k", "5",
    ])
    assert rc == 0
    assert "P@5:" in output
    assert "P@10:" not in output


def test_eval_rejects_non_positive_k(tmp_path: Path) -> None:
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc, output = _run([
        "eval", "--corpus", str(corpus), "--k", "0",
    ])
    assert rc == 2
    assert "--k must be positive" in output


def test_eval_missing_corpus_returns_1(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.jsonl"
    rc, output = _run(["eval", "--corpus", str(missing)])
    assert rc == 1
    assert "calibration corpus not found" in output


def test_eval_empty_corpus_returns_1(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    rc, output = _run(["eval", "--corpus", str(empty)])
    assert rc == 1
    assert "corpus is empty" in output


def test_eval_json_keys_sorted_for_diff_stability(tmp_path: Path) -> None:
    """JSON output keys are sorted so the line is diff-stable across
    Python versions / dict-ordering changes — important for the future
    R5 CI status-check surface that diffs runs."""
    corpus = _toy_corpus(tmp_path / "toy.jsonl")
    rc, output = _run(["eval", "--corpus", str(corpus), "--json"])
    assert rc == 0
    payload_keys = [
        m for m in json.loads(output).keys()
    ]
    assert payload_keys == sorted(payload_keys)
