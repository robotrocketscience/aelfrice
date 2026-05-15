"""Unit tests for ``scripts/validate_a4_corpus.py`` — #844 diagnostic.

Two layers:

1. **Coverage primitive** (``_normalize_tokens`` / ``_coverage``) —
   pure-function unit tests, no rebuilder dependency.

2. **Per-row diagnose** (``diagnose_row``) — synthetic two-row corpus
   exercising each flag value. Hits ``rebuild_v14`` so it's slower
   than a unit test but still under a second.

The R18 trap row (``transcript_pre_clear`` pre-answers the question)
must surface ``beliefs_load_bearing_flag=False``. The healthy row
(answer tokens live only in beliefs) must surface
``beliefs_load_bearing_flag=True``. The rest of the assertion set
pins the JSONL emission contract and the ``--strict`` exit code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
validate_a4_corpus = import_module("validate_a4_corpus")


# ---------------------------------------------------------------------
# Coverage primitive
# ---------------------------------------------------------------------

def test_normalize_tokens_casefolds_and_strips_punctuation() -> None:
    assert validate_a4_corpus._normalize_tokens("Hello, World!") == ["hello", "world"]


def test_normalize_tokens_preserves_intratoken_apostrophe() -> None:
    assert validate_a4_corpus._normalize_tokens("don't stop") == ["don't", "stop"]


def test_coverage_empty_expected_is_vacuous_one() -> None:
    assert validate_a4_corpus._coverage("", "any block content") == 1.0


def test_coverage_no_overlap_is_zero() -> None:
    assert validate_a4_corpus._coverage("apple banana", "completely unrelated") == 0.0


def test_coverage_partial_overlap_is_fraction() -> None:
    # "apple banana cherry" — 3 tokens, "apple" appears, others don't.
    assert validate_a4_corpus._coverage("apple banana cherry", "apple pie") == pytest.approx(1 / 3)


def test_coverage_unicode_nfc_normalized() -> None:
    decomposed = "café"  # cafe + combining acute
    composed = "café"
    assert validate_a4_corpus._coverage(decomposed, composed) == 1.0


def test_strip_retrieved_beliefs_removes_section_and_preserves_rest() -> None:
    block = (
        "<aelfrice-rebuild>"
        "<recent-turns><turn role=\"user\">keep me</turn></recent-turns>"
        "<retrieved-beliefs budget_used=\"100/4000\">"
        "<belief id=\"b1\">drop me</belief>"
        "</retrieved-beliefs>"
        "<continue/></aelfrice-rebuild>"
    )
    stripped = validate_a4_corpus._strip_retrieved_beliefs(block)
    assert "keep me" in stripped
    assert "drop me" not in stripped
    assert "retrieved-beliefs" not in stripped
    assert "<continue/>" in stripped


def test_strip_retrieved_beliefs_is_noop_when_section_absent() -> None:
    block = "<aelfrice-rebuild><recent-turns>x</recent-turns></aelfrice-rebuild>"
    assert validate_a4_corpus._strip_retrieved_beliefs(block) == block


# ---------------------------------------------------------------------
# Per-row diagnose (requires the rebuilder)
# ---------------------------------------------------------------------

def _healthy_row() -> dict:
    """A row whose expected answer depends on belief content.

    The expected answer token ``mauve`` appears only in the belief
    content, not in ``transcript_pre_clear``. Stripping beliefs must
    drop the coverage to ~0.
    """
    return {
        "id": "healthy-belief-load-bearing",
        "transcript_pre_clear": [
            {"role": "user", "text": "What is the launch color?"},
            {"role": "assistant", "text": "Let me check."},
        ],
        "beliefs": [
            {
                "id": "b-mauve",
                "content": "The launch color is mauve.",
                "retention_class": "fact",
                "lock_level": "user",
            }
        ],
        "expected_post_clear_answers": ["mauve"],
        "rebuilder_token_budget": 4000,
    }


def _r18_trap_row() -> dict:
    """A row that pre-answers the continuation question in
    ``transcript_pre_clear``.

    Expected answer tokens all appear in the prior assistant turn.
    Stripping beliefs must NOT drop the coverage — the answer is
    derivable from ``<recent-turns>`` alone. Surface:
    ``beliefs_load_bearing_flag=False``.
    """
    return {
        "id": "r18-trap-pre-answered",
        "transcript_pre_clear": [
            {"role": "user", "text": "What is the launch color?"},
            {"role": "assistant", "text": "The launch color is mauve."},
        ],
        "beliefs": [
            {
                "id": "b-unrelated",
                "content": "The mascot is a stoat.",
                "retention_class": "fact",
                "lock_level": "user",
            }
        ],
        "expected_post_clear_answers": ["mauve"],
        "rebuilder_token_budget": 4000,
    }


def test_diagnose_row_healthy_marks_beliefs_load_bearing() -> None:
    row = _healthy_row()
    with tempfile.TemporaryDirectory() as tmp:
        diag = validate_a4_corpus.diagnose_row(
            row,
            tmp_root=Path(tmp),
            arm_delta=0.001,
            beliefs_delta=0.05,
            db_counter=[0],
        )
    assert diag.id == "healthy-belief-load-bearing"
    assert diag.beliefs_load_bearing_flag is True
    assert diag.max_beliefs_delta > 0.05
    assert diag.n_expected_answers == 1


def test_diagnose_row_r18_trap_marks_beliefs_not_load_bearing() -> None:
    row = _r18_trap_row()
    with tempfile.TemporaryDirectory() as tmp:
        diag = validate_a4_corpus.diagnose_row(
            row,
            tmp_root=Path(tmp),
            arm_delta=0.001,
            beliefs_delta=0.05,
            db_counter=[0],
        )
    assert diag.id == "r18-trap-pre-answered"
    assert diag.beliefs_load_bearing_flag is False
    assert diag.max_beliefs_delta <= 0.05


# ---------------------------------------------------------------------
# CLI / emission contract
# ---------------------------------------------------------------------

def _write_corpus(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows to ``<tmp>/v2_0/compression_a4_fidelity/rows.jsonl``
    and return the v2_0 root path."""
    corpus_root = tmp_path / "v2_0"
    mod = corpus_root / "compression_a4_fidelity"
    mod.mkdir(parents=True)
    with (mod / "rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return corpus_root


def test_main_emits_jsonl_per_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus_root = _write_corpus(tmp_path, [_healthy_row(), _r18_trap_row()])
    rc = validate_a4_corpus.main(["--corpus-root", str(corpus_root)])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    records = [json.loads(line) for line in out]
    assert {r["id"] for r in records} == {
        "healthy-belief-load-bearing",
        "r18-trap-pre-answered",
    }
    for r in records:
        assert set(r.keys()) >= {
            "id",
            "arm_divergence_flag",
            "beliefs_load_bearing_flag",
            "n_expected_answers",
            "max_arm_delta",
            "max_beliefs_delta",
        }


def test_main_strict_exits_two_on_r18_trap(tmp_path: Path) -> None:
    corpus_root = _write_corpus(tmp_path, [_r18_trap_row()])
    out_path = tmp_path / "diagnostic.jsonl"
    rc = validate_a4_corpus.main([
        "--corpus-root", str(corpus_root),
        "--out", str(out_path),
        "--strict",
    ])
    assert rc == 2
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records[0]["beliefs_load_bearing_flag"] is False


def test_main_strict_passes_on_healthy(tmp_path: Path) -> None:
    corpus_root = _write_corpus(tmp_path, [_healthy_row()])
    out_path = tmp_path / "diagnostic.jsonl"
    rc = validate_a4_corpus.main([
        "--corpus-root", str(corpus_root),
        "--out", str(out_path),
        "--strict",
    ])
    assert rc == 0


def test_main_missing_corpus_returns_one(tmp_path: Path) -> None:
    # No compression_a4_fidelity subdir under the root.
    rc = validate_a4_corpus.main(["--corpus-root", str(tmp_path)])
    assert rc == 1
